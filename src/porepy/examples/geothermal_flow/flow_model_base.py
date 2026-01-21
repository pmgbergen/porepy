from __future__ import annotations
import logging
import time
import csv
import numpy as np
import scipy.sparse as sps
from scipy.sparse.csgraph import reverse_cuthill_mckee
from typing import Callable, Optional, cast, Any

import porepy as pp
from porepy.models.compositional_flow import (
    CompositionalFractionalFlowTemplate as FlowTemplate,
)

# PETSc imports (only if available)
try:
    import petsc4py
    petsc4py.init()
    from petsc4py import PETSc
    PETSC_AVAILABLE = True
except ImportError:
    PETSC_AVAILABLE = False
    logging.warning("*** ITERATIVE SOLVER NOT AVAILABLE ***")
    logging.warning("PETSc not available. All linear systems will use direct solver (MUMPS/UMFPACK).")
    logging.warning("For large systems, consider installing PETSc for iterative solver options.")

# Configure logging to show info messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Ensure specific loggers are enabled for linear solver information
logging.getLogger('porepy.models.solution_strategy').setLevel(logging.INFO)
logging.getLogger('porepy').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

to_Mega = 1.0e-6

class FlowModelBase(FlowTemplate):
    def __init__(self, params):
        super().__init__(params)
        self.newton_iterations_per_timestep = []
        self.total_newton_iterations = 0
        # Flag to use PETSc with MUMPS solver
        self.use_petsc = params.get("use_petsc", False)

        # Preconditioner selection for PETSc solver
        self.petsc_preconditioner = params.get("petsc_preconditioner", "bjacobi")
        valid_preconditioners = {"bjacobi", "asm", "jacobi", "lump_colsum", "amg_hypre","ilu0","lu", "cpr"}
        if self.petsc_preconditioner not in valid_preconditioners:
            logger.warning(f"Invalid preconditioner '{self.petsc_preconditioner}'. Using 'bjacobi' as default.")
            self.petsc_preconditioner = "bjacobi"

        # Flag to enable Cuthill-McKee permutation for bandwidth reduction
        self.use_cuthill_mckee = params.get("use_cuthill_mckee", True)

        # Check if PETSc is available when requested
        if self.use_petsc and not PETSC_AVAILABLE:
            logger.warning("*** SOLVER CONFIGURATION MISMATCH ***")
            logger.warning("PETSc iterative solver was requested (use_petsc=True) but PETSc is not available.")
            logger.warning("All linear systems will use the default direct solver instead.")
            logger.warning("To use iterative solvers, install PETSc with: pip install petsc petsc4py")
            self.use_petsc = False

    def solve_linear_system_petsc(self, A: sps.spmatrix, b: np.ndarray, preconditioner: str = "asm") -> np.ndarray:
        """
        Solve linear system using PETSc with selectable preconditioners and detailed logging.
        """
        if not PETSC_AVAILABLE:
            raise RuntimeError("PETSc is not available")

        # Validate preconditioner
        valid_preconditioners = {"bjacobi", "asm", "jacobi", "lump_colsum", "amg_hypre", "ilu0", "lu", "cpr"}
        if preconditioner not in valid_preconditioners:
            logger.warning(f"Invalid preconditioner '{preconditioner}'. Using 'lu' as default.")
            preconditioner = "lu"

        logger.info(f"Solving linear system with PETSc {preconditioner.upper()}")

        # 1. Convert to CSR and prepare working vector
        A_csr = A.tocsr()
        b_working = b.copy()

        # Initialize permutation variables to None for safety
        perm = None
        eq_perm = None
        var_perm = None
        field_split = None

        # 1.5. Apply equation permutation
        try:
            A_csr, b_working, eq_perm, var_perm, field_split = self.apply_equation_permutation(A_csr, b_working)
        except Exception as e:
            logger.warning(f"Equation permutation failed: {e}. Continuing with original ordering.")
            if preconditioner == "cpr":
                logger.warning("CPR requires successful equation permutation. Falling back to 'asm'.")
                preconditioner = "asm"

        # 2. Apply Cuthill-McKee permutation
        if self.use_cuthill_mckee and preconditioner not in ["lu", "cpr"]:
            try:
                perm = reverse_cuthill_mckee(A_csr, symmetric_mode=False)
                A_csr = A_csr[perm, :][:, perm]
                b_working = b_working[perm]
            except Exception as e:
                logger.warning(f"Cuthill-McKee permutation failed: {e}. Continuing with original ordering.")
                perm = None

        # 3. Regularize Diagonal
        if preconditioner not in ["lump_colsum", "lu"]:
            diagonal = A_csr.diagonal()
            zero_diag_indices = np.where(np.abs(diagonal) < 1e-14)[0]
            if len(zero_diag_indices) > 0:
                logger.info(f"Regularizing {len(zero_diag_indices)} zero diagonal entries")
                A_lil = A_csr.tolil()
                matrix_norm = np.mean(np.abs(A_csr.data))
                regularization_value = max(1e-12, matrix_norm * 1e-8)
                for idx in zero_diag_indices:
                    A_lil[idx, idx] = regularization_value
                A_csr = A_lil.tocsr()

        # 4. Apply Matrix Scaling
        row_scaling, col_scaling, A_csr, b_scaled = self._apply_matrix_scaling(A_csr, b_working)

        # 5. Create PETSc Matrices/Vectors
        petsc_A = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
        petsc_A.assemblyBegin()
        petsc_A.assemblyEnd()

        petsc_b = PETSc.Vec().createWithArray(b_scaled)
        petsc_x = PETSc.Vec().createWithArray(np.zeros_like(b_scaled))

        # 6. Setup KSP
        ksp = PETSc.KSP().create()
        ksp_prefix = "fluid_buoyancy_"
        ksp.setOptionsPrefix(ksp_prefix)

        # Initialize explicit references for cleanup
        petsc_M = None
        is_p = None
        is_t = None

        # 7. Configure Solver
        if preconditioner == "lu":
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setFactorSolverType("mumps")
            ksp.setOperators(A=petsc_A, P=petsc_A)
        else:
            ksp.setType(PETSc.KSP.Type.FGMRES)
            ksp.setGMRESRestart(50)

            # Setup Operators
            if preconditioner == "lump_colsum":
                col_sums = np.array(np.abs(A_csr).sum(axis=0)).flatten()
                zero_cols = np.where(col_sums < 1e-14)[0]
                if len(zero_cols) > 0:
                    col_sums[zero_cols] = 1e-12
                diag_vals = 1.0 / col_sums

                petsc_M = PETSc.Mat().createAIJ(size=A_csr.shape)
                petsc_M.setUp()
                for i in range(len(diag_vals)):
                    petsc_M.setValue(i, i, diag_vals[i])
                petsc_M.assemblyBegin()
                petsc_M.assemblyEnd()
                ksp.setOperators(A=petsc_A, P=petsc_M)
            else:
                ksp.setOperators(A=petsc_A, P=petsc_A)

            # Setup PC
            pc = ksp.getPC()
            opts = PETSc.Options()

            if preconditioner == "cpr":
                if not field_split:
                    raise RuntimeError("CPR preconditioner requires 'field_split' data.")

                try:
                    n_pressure = field_split.get('pressure',
                                                 field_split.get('pressure_size', list(field_split.values())[0]))
                except (AttributeError, IndexError):
                    raise RuntimeError("Could not parse 'field_split' dictionary.")

                n_total = A_csr.shape[0]
                is_p = PETSc.IS().createStride(n_pressure, first=0, step=1)
                is_t = PETSc.IS().createStride(n_total - n_pressure, first=n_pressure, step=1)

                pc.setType(PETSc.PC.Type.FIELDSPLIT)
                pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
                pc.setFieldSplitIS(('pressure', is_p), ('transport', is_t))

                # --- Block 0: Pressure ---
                # Protection: Use LU for small matrices (<10k rows) to avoid AMG setup failures
                if n_pressure < 10000:
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_ksp_type", "preonly")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_type", "lu")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_factor_shift_type", "nonzero")
                else:
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_ksp_type", "preonly")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_type", "lu")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_factor_mat_solver_type", "mumps")

                # --- Block 1: Transport ---
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_ksp_type", "richardson")
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_pc_type", "ilu")
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_pc_factor_levels", "0")
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_pc_factor_shift_type", "nonzero")
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_pc_factor_shift_amount", "1e-10")

            elif preconditioner == "ilu0":
                pc.setType(PETSc.PC.Type.ILU)
                pc.setFactorLevels(0)
                opts.setValue(f"-{ksp_prefix}pc_factor_shift_type", "nonzero")
                opts.setValue(f"-{ksp_prefix}pc_factor_shift_amount", "1e-12")

            elif preconditioner == "amg_hypre":
                pc.setType(PETSc.PC.Type.HYPRE)
                pc.setHYPREType("boomeramg")
                opts.setValue(f"-{ksp_prefix}pc_hypre_boomeramg_strong_threshold", "0.25")
                opts.setValue(f"-{ksp_prefix}pc_hypre_boomeramg_coarsen_type", "HMIS")
                opts.setValue(f"-{ksp_prefix}pc_hypre_boomeramg_interp_type", "ext+i")

            elif preconditioner == "bjacobi":
                pc.setType(PETSc.PC.Type.BJACOBI)
            elif preconditioner == "asm":
                pc.setType(PETSc.PC.Type.ASM)
                pc.setASMOverlap(1)
            elif preconditioner == "jacobi":
                pc.setType(PETSc.PC.Type.JACOBI)
            elif preconditioner == "lump_colsum":
                pc.setType(PETSc.PC.Type.MAT)

        # 8. Finalize Options
        ksp.setFromOptions()

        # Apply tolerances AFTER setFromOptions to strictly enforce them.
        # This overrides any command-line defaults or database presets.
        ksp.setTolerances(rtol=1.0e-5, atol=1.0e-8, max_it=500)

        # Optional: Log the actual tolerances PETSc is using to be 100% sure
        r_tol, a_tol, div_tol, max_its = ksp.getTolerances()
        logger.info(f"KSP Tolerances Enforced | rtol: {r_tol}, atol: {a_tol}, max_it: {max_its}")

        # 9. Solve and Log
        solution = None
        try:
            # Step A: Explicitly time the Preconditioner Setup
            t_setup_start = time.time()
            ksp.setUp()
            t_setup_end = time.time()
            setup_dur = t_setup_end - t_setup_start

            # Step B: Time the Solve
            t_solve_start = time.time()
            ksp.solve(petsc_b, petsc_x)
            t_solve_end = time.time()
            solve_dur = t_solve_end - t_solve_start

            # Step C: Retrieve Metrics
            iters = ksp.getIterationNumber()
            resid = ksp.getResidualNorm()

            # Step D: Log Report
            logger.info(
                f"PETSc {preconditioner.upper()} Report | Setup: {setup_dur:.4f}s | Solve: {solve_dur:.4f}s | Iters: {iters} | Residual: {resid:.4e}")

            if ksp.getConvergedReason() < 0:
                logger.warning(f"Solver failed. Reason: {ksp.getConvergedReason()}")
            else:
                # 10. Unscale and Reverse Permutations
                scaled_sol = petsc_x.getArray().copy()
                unscaled_sol = col_scaling * scaled_sol

                if perm is not None:
                    cuthill_reversed_sol = np.zeros_like(unscaled_sol)
                    cuthill_reversed_sol[perm] = unscaled_sol
                    unscaled_sol = cuthill_reversed_sol

                if var_perm is not None:
                    solution = np.zeros_like(unscaled_sol)
                    solution[var_perm] = unscaled_sol
                else:
                    solution = unscaled_sol

        except Exception as e:
            # Fallback for LU
            if preconditioner == "lu" and "mumps" in str(e).lower():
                logger.warning("MUMPS failed. Retrying with PETSc native LU...")
                try:
                    pc.setFactorSolverType("petsc")
                    ksp.setFromOptions()
                    ksp.solve(petsc_b, petsc_x)
                    if ksp.getConvergedReason() >= 0:
                        scaled_sol = petsc_x.getArray().copy()
                        unscaled_sol = col_scaling * scaled_sol

                        if perm is not None:
                            cuthill_reversed_sol = np.zeros_like(unscaled_sol)
                            cuthill_reversed_sol[perm] = unscaled_sol
                            unscaled_sol = cuthill_reversed_sol

                        if var_perm is not None:
                            solution = np.zeros_like(unscaled_sol)
                            solution[var_perm] = unscaled_sol
                        else:
                            solution = unscaled_sol
                except Exception as e2:
                    logger.error(f"Fallback solver failed: {e2}")
            else:
                logger.error(f"Solver execution error: {e}")
        # Cleanup
        petsc_A.destroy()
        petsc_b.destroy()
        petsc_x.destroy()
        ksp.destroy()
        if petsc_M: petsc_M.destroy()
        if is_p: is_p.destroy()
        if is_t: is_t.destroy()

        return solution

    def _apply_matrix_scaling(self, A_csr, b):
        """
        Apply row and column scaling to improve matrix conditioning.

        Parameters:
        -----------
        A_csr : scipy sparse matrix
            Input matrix in CSR format
        b : numpy array
            Right-hand side vector

        Returns:
        --------
        tuple
            (row_scaling, col_scaling, scaled_A_csr, scaled_b) where scaling factors and scaled matrix/vector
        """

        # Compute row and column norms for scaling
        row_norms = np.array(np.sqrt((A_csr.multiply(A_csr)).sum(axis=1))).flatten()
        col_norms = np.array(np.sqrt((A_csr.multiply(A_csr)).sum(axis=0))).flatten()

        # Avoid division by zero
        row_norms = np.where(row_norms < 1e-16, 1.0, row_norms)
        col_norms = np.where(col_norms < 1e-16, 1.0, col_norms)

        # Create scaling factors (inverse of norms for better conditioning)
        row_scaling = 1.0 / np.sqrt(row_norms)
        col_scaling = 1.0 / np.sqrt(col_norms)

        # Apply scaling: D_r * A * D_c where D_r, D_c are diagonal scaling matrices
        A_scaled = sps.diags(row_scaling) @ A_csr @ sps.diags(col_scaling)

        # Scale right-hand side: D_r * b
        b_scaled = row_scaling * b

        logger.debug(f"Matrix scaling applied. Row norm range: [{np.min(row_norms):.2e}, {np.max(row_norms):.2e}], "
                    f"Col norm range: [{np.min(col_norms):.2e}, {np.max(col_norms):.2e}]")

        return row_scaling, col_scaling, A_scaled.tocsr(), b_scaled

    def solve_linear_system(self) -> np.ndarray:
        """
        Solve the linear system using either PETSc GMRES with selectable preconditioner or default solver.

        Preconditioner options (set via petsc_preconditioner parameter):
        - 'bjacobi': Block Jacobi preconditioner (default)
        - 'asm': Additive Schwarz Method
        - 'jacobi': Point Jacobi preconditioner
        - 'lump_colsum': Lumped column sum diagonal preconditioner
        - 'amg_hypre': Algebraic Multigrid with Hypre BoomerAMG

        Returns:
            np.ndarray: Solution vector (the nonlinear increment).
        """
        if self.use_petsc and PETSC_AVAILABLE:
            # Use PETSc solver with selected preconditioner
            A, b = self.linear_system
            solution = self.solve_linear_system_petsc(A, b, preconditioner=self.petsc_preconditioner)
            if solution is None:
                logger.warning(f"PETSc iterative solver with {self.petsc_preconditioner.upper()} preconditioner failed to converge.")
                return super().solve_linear_system()
            return solution
        else:
            # Check if PETSc was requested but not available
            if self.use_petsc and not PETSC_AVAILABLE:
                logger.info("*** SOLVER FALLBACK ***")
                logger.info("PETSc was requested but not available. Using default direct solver.")

            # Use default solver
            solution = super().solve_linear_system()
            if solution is None:
                raise RuntimeError("Linear solver returned None - this should not happen")
            return solution

    def set_equations(self):
        super().set_equations()
        self.set_buoyancy_discretization_parameters()

    def set_nonlinear_discretizations(self) -> None:
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

    def before_nonlinear_iteration(self) -> None:
        self.update_buoyancy_driven_fluxes()
        self.rediscretize()

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2") * to_Mega
        size = np.sum([g.num_cells for g in subdomains]).astype(int)
        gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
        gravity_field.set_name("gravity_field")
        return gravity_field

    def _get_non_reference_component(self) -> str | None:
        """
        Return the non-reference component name for binary mixtures.

        Convention: self.fluid.components is expected to be an indexable sequence
        where the reference component is at index 0 and the other (active)
        component is at index 1. If there are at least two components, return
        the second one. Otherwise return None.
        """
        return self.get_components()[1].name

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: Optional[np.ndarray],
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        if self._is_nonlinear_problem():

            total_volume = 0.0
            for sd in self.mdg.subdomains():
                total_volume += np.sum(
                    self.equation_system.evaluate(self.volume_integral(pp.ad.Scalar(1), [sd], dim=1)))

            # nonlinear_increment based norm
            nonlinear_increment_norm = self.compute_nonlinear_increment_norm(
                nonlinear_increment
            )

            # Residual per subsystem
            residual_norm = np.linalg.norm(residual) * total_volume
            # Check convergence requiring both the increment and residual to be small.
            converged_inc = (
                nl_params["nl_convergence_tol"] is np.inf
                or nonlinear_increment_norm < nl_params["nl_convergence_tol"]
            )
            converged_res = (
                nl_params["nl_convergence_tol_res"] is np.inf
                or residual_norm < nl_params["nl_convergence_tol_res"]
            )
            converged = converged_inc and converged_res
            diverged = False
        else:
            raise ValueError(
                "Gravitational segregation is nonlinear in its simpler form."
            )
        print("residual norm: ", residual_norm)
        return converged, diverged

    def permute_equations_and_variables(self):
        """
        Permute equations and variables in the following order:
        1. Elliptic equations: mass_balance_equation, interface_darcy_flux_equation, well_flux_equation
        2. Transport equations: component mass balance for the non-reference component, energy_balance_equation,
           interface_fourier_flux_equation, interface_enthalpy_flux_equation, well_enthalpy_flux_equation
        3. Algebraic equations: elimination_of_s_gas_on_grids_[0], elimination_of_x_<comp>_liq_on_grids_[0],
           elimination_of_x_<comp>_gas_on_grids_[0], elimination_of_temperature_on_grids_[0]

        Returns:
            tuple: (equation_permutation, variable_permutation) where each is an array of indices
        """

        # Inputs provided
        equation_keys = list(self.equation_system.assembled_equation_indices.keys())
        variables_keys = list(set([v.name for v in self.equation_system.variables]))

        # Initialize the dictionary
        variable_equation_map = {}

        # Helper function to find equation in list
        def find_eq(keyword, eq_list):
            for eq in eq_list:
                if keyword in eq:
                    return eq
            return None

        # 1. Map Global Conservation Laws & Fluxes (Standard Physics Mappings)
        # Pressure <-> Mass Balance
        variable_equation_map['pressure'] = (
            find_eq('mass_balance_equation', equation_keys),
            'pressure'
        )

        # Determine active (non-reference) component for binary mixtures
        active_comp = self._get_non_reference_component()
        # Fallback: use 'CO2' if component cannot be determined to preserve prior behaviour
        if not active_comp:
            active_comp = 'CO2'

        # z_<comp> <-> Component Mass Balance
        z_key = f"z_{active_comp}"
        variable_equation_map[z_key] = (
            find_eq(f'component_mass_balance_equation_{active_comp}', equation_keys),
            z_key
        )

        # Enthalpy <-> Energy Balance
        variable_equation_map['enthalpy'] = (
            find_eq('energy_balance_equation', equation_keys),
            'enthalpy'
        )

        # Fluxes (Direct name matching)
        flux_vars = ['interface_darcy_flux', 'interface_fourier_flux', 'interface_enthalpy_flux']
        for var in flux_vars:
            # Matches e.g. "interface_darcy_flux" to "interface_darcy_flux_equation"
            variable_equation_map[var] = (find_eq(var, equation_keys), var)

        # 2. Map Local Elimination/Constraint Equations
        # These look for the variable name inside the elimination string
        # e.g., "s_gas" is found inside "elimination_of_s_gas_..."
        elimination_vars = ['s_gas', f'x_{active_comp}_liq', f'x_{active_comp}_gas', 'temperature']

        for var in elimination_vars:
            # Search for the equation string that contains "elimination_of_{var}"
            target_str = f"elimination_of_{var}"
            found_eq = find_eq(target_str, equation_keys)

            if found_eq:
                variable_equation_map[var] = (found_eq, var)

        def find_variable_idxs(name):
            if 'interface' in name:
                md_var = self.equation_system.md_variable(name, self.mdg.interfaces())
            else:
                md_var = self.equation_system.md_variable(name, self.mdg.subdomains())
            var_dof = self.equation_system.dofs_of(md_var.sub_vars)
            return var_dof

        equation_e_indices = []
        variable_e_indices = []

        # order for field split
        elliptic_keys = ['pressure', 'interface_darcy_flux']
        elliptic_keys.extend(elimination_vars)
        for key in elliptic_keys:
            eq_name, var_name = variable_equation_map.get(key, (None, None))
            if eq_name and var_name:
                # Get equation indices
                eq_idxs = self.equation_system.assembled_equation_indices[eq_name]
                equation_e_indices.extend(eq_idxs)

                # Get variable indices
                var_dofs = find_variable_idxs(var_name)
                variable_e_indices.extend(var_dofs)
                assert len(eq_idxs) == len(var_dofs), f"Mismatch in lengths for {key}: {len(eq_idxs)} equations vs {len(var_dofs)} variables"

        equation_t_indices = []
        variable_t_indices = []
        transport_keys = ['enthalpy', 'interface_enthalpy_flux','interface_fourier_flux', z_key]
        for key in transport_keys:
            eq_name, var_name = variable_equation_map.get(key, (None, None))
            if eq_name and var_name:
                # Get equation indices
                eq_idxs = self.equation_system.assembled_equation_indices[eq_name]
                equation_t_indices.extend(eq_idxs)

                # Get variable indices
                var_dofs = find_variable_idxs(var_name)
                variable_t_indices.extend(var_dofs)
                assert len(eq_idxs) == len(var_dofs), f"Mismatch in lengths for {key}: {len(eq_idxs)} equations vs {len(var_dofs)} variables"

        equation_indices = equation_e_indices + equation_t_indices
        variable_indices = variable_e_indices + variable_t_indices
        return np.array(equation_indices), np.array(variable_indices), {'elliptic': len(equation_e_indices), 'transport': len(equation_t_indices)}

    def apply_equation_permutation(self, A: sps.spmatrix, b: np.ndarray) -> tuple[sps.spmatrix, np.ndarray, np.ndarray | None, np.ndarray | None, dict | None]:
        """
        Apply equation and variable permutation to the linear system.

        Args:
            A: Jacobian matrix
            b: Right-hand side vector

        Returns:
            tuple: (permuted_A, permuted_b, equation_permutation, variable_permutation)
        """
        try:
            eq_perm, var_perm, field_split = self.permute_equations_and_variables()

            # Permute rows (equations) and columns (variables) of the matrix
            A_permuted = A[eq_perm, :][:, var_perm]

            # Permute the right-hand side vector
            b_permuted = b[eq_perm]

            logger.info(f"Applied equation permutation: {len(eq_perm)} equations, {len(var_perm)} variables")

            return A_permuted, b_permuted, eq_perm, var_perm, field_split

        except Exception as e:
            logger.warning(f"Failed to apply equation permutation: {e}. Using original ordering.")
            return A, b, None, None, None

    def assemble_linear_system(self) -> None:
        """Custom assemble linear system that updates Jacobian every 0, 3, 6, 9... Newton iterations.

        This method implements a dedicated solution strategy that:
        - Assembles the full linear system (Jacobian + residual) at iterations 0, 3, 6, 9, etc.
        - Updates only the residual part for other iterations (1, 2, 4, 5, 7, 8, etc.)
        """
        t_0 = time.time()

        # Get current Newton iteration number
        iteration_num = self.nonlinear_solver_statistics.num_iteration

        if iteration_num % 2 == 0 or iteration_num < 10:
            # Update both Jacobian and residual at iterations 0, 3, 6, 9, ...
            logger.info(f"Newton iteration {iteration_num}: Updating Jacobian and residual")
            self.linear_system = self.equation_system.assemble(evaluate_jacobian=True)
        else:
            # Update only residual at iterations 1, 2, 4, 5, 7, 8, ...
            logger.info(f"Newton iteration {iteration_num}: Updating residual only")
            if hasattr(self, 'linear_system') and self.linear_system is not None:
                # Keep the existing Jacobian, update only the residual
                new_residual = self.equation_system.assemble(evaluate_jacobian=False)
                # Update the residual part of the linear system (tuple format: (matrix, rhs))
                self.linear_system = (
                    self.linear_system[0],  # Keep existing Jacobian
                    -new_residual  # Update residual with new evaluation
                )
            else:
                # Fallback: if no previous linear system exists, assemble full system
                logger.warning("No previous linear system found, assembling full system")
                if self._apply_schur_complement_reduction():
                    assert self.schur_complement_primary_variables, (
                        "Primary column block for Schur technique not found."
                    )
                    assert self.schur_complement_primary_equations, (
                        "Primary row block for Schur technique not defined."
                    )
                    self.linear_system = self.equation_system.assemble_schur_complement_system(
                        self.schur_complement_primary_equations,
                        self.schur_complement_primary_variables,
                        inverter=cast(
                            Callable[[sps.spmatrix], sps.spmatrix],
                            self.params.get("schur_complement_inverter", None),
                        ),
                    )
                else:
                    self.linear_system = self.equation_system.assemble()

        t_1 = time.time()
        logger.debug(f"Assembled linear system in {t_1 - t_0:.2e} seconds.")

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        # Track Newton iterations for this timestep
        current_iterations = self.nonlinear_solver_statistics.num_iteration
        self.newton_iterations_per_timestep.append(current_iterations)
        self.total_newton_iterations += current_iterations

        # Print Newton iteration info for current timestep
        current_time = self.time_manager.time
        timestep_number = self.time_manager.time_index
        logger.info(f"Timestep {timestep_number} (t={current_time:.2e}): {current_iterations} Newton iterations")

    def write_newton_iterations_to_csv(self, filename="newton_iterations.csv"):
        """Write Newton iteration data to CSV file."""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Timestep', 'Time', 'Newton_Iterations'])

            # Write data for each timestep
            for i, iterations in enumerate(self.newton_iterations_per_timestep):
                timestep_number = i + 1
                # Calculate time value based on time manager
                time_value = self.time_manager.schedule[0] + (timestep_number * self.time_manager.dt_init)
                writer.writerow([timestep_number, f"{time_value:.6e}", iterations])

            # Write summary row
            writer.writerow(['', '', ''])
            writer.writerow(['SUMMARY', '', ''])
            writer.writerow(['Total_Timesteps', len(self.newton_iterations_per_timestep), ''])
            writer.writerow(['Total_Newton_Iterations', self.total_newton_iterations, ''])
            if self.newton_iterations_per_timestep:
                avg_iterations = self.total_newton_iterations / len(self.newton_iterations_per_timestep)
                writer.writerow(['Average_Iterations_Per_Timestep', f"{avg_iterations:.2f}", ''])
                writer.writerow(['Max_Iterations', max(self.newton_iterations_per_timestep), ''])
                writer.writerow(['Min_Iterations', min(self.newton_iterations_per_timestep), ''])

        print(f"Newton iteration data written to {filename}")
        print(f"Total Newton iterations: {self.total_newton_iterations}")
        print(f"Total timesteps: {len(self.newton_iterations_per_timestep)}")
        avg_iterations = 0.0
        if self.newton_iterations_per_timestep:
            avg_iterations = self.total_newton_iterations / len(self.newton_iterations_per_timestep)
        print(f"Average iterations per timestep: {avg_iterations:.2f}")
