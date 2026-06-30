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
    CompositionalFlowTemplate as FlowTemplate,
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
    # Trust-region state, persistent across nonlinear iterations (reset each time step).
    _trust_radius: float = None

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

    def _solve_linear_system_core(self) -> np.ndarray:
        """
        Core linear solve (no step control): PETSc GMRES with a selectable preconditioner,
        or the default direct solver. This is wrapped by :meth:`solve_linear_system`,
        which adds the optional line-search / trust-region step control.

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

    def solve_linear_system(self) -> np.ndarray:
        """Solve the linear system and apply the configured nonlinear step control.

        Selected by ``params["step_control_method"]`` (default ``"None"``):

        - ``"None"``  : plain Newton, no step control.
        - ``"LS"``    : backtracking line search (Armijo), applied only when the full
          Newton step would increase the residual.
        - ``"TR"``    : CFL-based trust region.
        - ``"TR-LS"`` : trust region followed by a line-search refinement.

        Residual reporting and solution post-processing are intentionally left to the
        model (see the overridable hooks :meth:`compute_residuals_by_category`,
        :meth:`postprocessing_overshoots`, :meth:`postprocessing_thermal_overshoots`);
        this method only chooses and applies the step.
        """
        _, residual_vector = self.linear_system
        residual_norm_current = np.linalg.norm(residual_vector)

        step_control_method = self.params.get("step_control_method", "None")
        step_control_alpha_min = self.params.get("step_control_alpha_min", 0.01)
        activate_after_iteration = self.params.get("activate_step_control_after_iter", 1)
        activate_step_control_Q = (
            self.nonlinear_solver_statistics.num_iterations > activate_after_iteration
        )

        # Reset the trust radius at the start of each time step (iteration 0).
        if self.nonlinear_solver_statistics.num_iterations == 0:
            self._trust_radius = 1.0

        if step_control_method == "None":
            solution = self._solve_linear_system_core()

        elif step_control_method == "LS":
            solution = self._solve_linear_system_core()
            residual_future = self.compute_residual_from_increment(solution, restore_state=True)
            increasing_residual_Q = np.linalg.norm(residual_future) > residual_norm_current
            if increasing_residual_Q and activate_step_control_Q:
                print("Step control: Line Search (LS)")
                alpha = self.backtracking_line_search(
                    solution, residual_vector, alpha_min=step_control_alpha_min
                )
                solution *= alpha
                print(f"Line search: accepted alpha = {alpha:.4f}")

        elif step_control_method == "TR":
            if activate_step_control_Q:
                print("Step control: Trust Region (TR)")
                solution, self._trust_radius = self.trust_region_solve(
                    trust_radius=self._trust_radius
                )
            else:
                solution = self._solve_linear_system_core()

        elif step_control_method == "TR-LS":
            if activate_step_control_Q:
                print("Step control: Trust Region + Line Search (TR-LS)")
                solution, self._trust_radius = self.trust_region_solve(
                    trust_radius=self._trust_radius
                )
                residual_after_tr = self.compute_residual_from_increment(
                    solution, restore_state=True
                )
                if np.linalg.norm(residual_after_tr) > residual_norm_current * 0.9:
                    alpha = self.backtracking_line_search(
                        solution, residual_vector, alpha_min=step_control_alpha_min
                    )
                    solution *= alpha
                    print(f"  TR-LS: line search alpha = {alpha:.4f}")
            else:
                solution = self._solve_linear_system_core()

        else:
            raise ValueError(
                f"Unknown step_control_method: {step_control_method}. "
                f"Valid options are: 'None', 'LS', 'TR', 'TR-LS'"
            )

        if self.params.get("reduce_linear_system_q", False):
            raise NotImplementedError(
                "The 'reduce_linear_system_q' case is not yet implemented."
            )

        return solution

    def compute_residual_from_increment(
        self, nonlinear_increment: np.ndarray, restore_state: bool = True
    ) -> np.ndarray:
        """
        Compute the residual after applying a nonlinear increment.

        This method follows the logic for residual evaluation:
        1. Save current state (if restore_state=True)
        2. Apply the nonlinear increment
        3. Update derived quantities
        4. Update buoyancy-driven fluxes
        5. Rediscretize
        6. Assemble the residual
        7. Restore original state (if restore_state=True)
        8. Return the residual vector

        Parameters:
            nonlinear_increment: The increment to apply to current variable values
            restore_state: If True, restore the original state after computing residual.
                          Set to False when this is the final accepted increment.

        Returns:
            The residual vector
        """
        # Save current state if we need to restore it later
        if restore_state:
            x_current = self.equation_system.get_variable_values(iterate_index=0).copy()

        # Apply the nonlinear increment additively to the current iterate
        self.equation_system.set_variable_values(
            values=nonlinear_increment, additive=True, iterate_index=0
        )

        # Update derived quantities
        self.update_derived_quantities()

        # Update buoyancy-driven fluxes (skipped when the direction is lagged/frozen)
        self.refresh_buoyancy_direction()

        # Rediscretize
        self.rediscretize()

        # Assemble the current nonlinear residual
        current_nonlinear_residual = self.equation_system.assemble(evaluate_jacobian=False)

        # Restore original state if requested
        if restore_state:
            try:
                self.equation_system.set_variable_values(x_current, iterate_index=0)
            except TypeError:
                self.equation_system.set_variable_values(x_current)

            # CRITICAL: Must also restore derived quantities and discretization
            # Otherwise the state is corrupted for the next iteration
            self.update_derived_quantities()
            self.refresh_buoyancy_direction()
            self.rediscretize()

        return current_nonlinear_residual

    def estimate_mixed_dimensional_cfl_number(self) -> tuple[float, float, float]:
        """
        Estimate the MD-CFL number

        Uses the same divergence operator as mass balance equations:
            ∂(φρ)/∂t + ∇·(ρ q) = 0

        where div = pp.ad.Divergence(subdomains, dim=1)

        Returns:
            tuple: (cfl_max, div_max, dx_min)
                - cfl_max: Maximum CFL number over all cells
                - div_max: Maximum divergence magnitude [1/s]
                - dx_min: Minimum cell size [m]
        """

        # Get the subdomains
        subdomains = self.mdg.subdomains(dim=self.nd)
        # Get current time step
        dt = self.time_manager.dt

        # Get characteristic cell size
        cell_diameters = self.volume_integral(1,subdomains,dim=1).value(self.equation_system)
        dx_min = np.min(cell_diameters)

        # === Use PorePy's AD operators (same as mass balance equations) ===

        # 1. Get Darcy flux using AD operator
        darcy_flux_ad = self.darcy_flux(subdomains)

        # 2. Get density on cells
        density_ad = self.fluid.density(subdomains)
        density_values = density_ad.value(self.equation_system)

        # 3. Use PorePy's Divergence operator (consistent with mass balance)
        div_operator = pp.ad.Divergence(subdomains, dim=1)

        # Compute divergence of Darcy flux [m³/s/m³ = 1/s]
        div_darcy_ad = div_operator @ darcy_flux_ad
        div_mass_flux = div_darcy_ad.value(self.equation_system)

        # Absolute divergence
        abs_div = np.abs(div_mass_flux)

        # 4. Get accumulation density: φρ [kg/m³]
        porosity_op = self.porosity(subdomains)
        porosity = porosity_op.value(self.equation_system)

        accumulation_density = porosity * density_values

        # 5. CFL number: CFL = dt * |∇·(ρq)| / (φρ)
        cfl_per_cell = np.nan_to_num(dt * abs_div / (accumulation_density) , nan=0.0, posinf=0.0)
        cfl_max = np.max(cfl_per_cell)
        div_max = np.max(np.nan_to_num(abs_div / (accumulation_density) , nan=0.0, posinf=0.0))

        return cfl_per_cell, cfl_max, div_max, dx_min

    def trust_region_solve(
            self,
            trust_radius: float = 1.0,
            eta: float = 0.1,
    ) -> tuple[np.ndarray, float]:
        """
        Simplified CFL-based Trust Region solver.

        Strategy:
        - Calculate effective trust radius as: cfl_target / cfl_current
        - Trust pressure Newton step completely (parabolic, well-behaved)
        - Apply CFL-based trust region to hyperbolic variables (enthalpy, composition)
        """
        # Get target CFL parameter
        cfl_target = self.params.get("trust_region_cfl_max_target", 1.0)

        # Get Jacobian and residual
        jacobian_matrix, residual_vector = self.linear_system
        residual_norm_current = np.linalg.norm(residual_vector)

        # Estimate current CFL number
        cfl_per_cell, cfl_current, div_max, dx_min = self.estimate_mixed_dimensional_cfl_number()
        print(f"  TR-CFL: Current CFL={cfl_current:.4f}, div_max={div_max:.2e} 1/s, dx_min={dx_min:.2e} m")

        h_op = self.enthalpy(self.mdg.subdomains())
        h_values = h_op.value(self.equation_system)
        CFL_energy = np.max(cfl_per_cell * np.abs(h_values))

        # Calculate effective trust radius: cfl_target / cfl_current
        if CFL_energy > 1e-3:
            trust_radius = cfl_target / CFL_energy
            print(f"  TR-CFL: Effective trust_radius = {cfl_target:.2f} / {cfl_current:.4f} = {trust_radius:.4e}")
        else:
            trust_radius = 1.0
            print(f"  TR-CFL: Low CFL, using trust_radius = 1.0")

        # Compute pure Newton step
        pk_newton = self._solve_linear_system_core()
        self.postprocessing_overshoots(pk_newton)

        # Get DOF indices for each variable
        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])

        # Compute norms for each block
        p_step_norm = np.linalg.norm(pk_newton[p_dof_idx])
        h_step_norm = np.linalg.norm(pk_newton[h_dof_idx])
        z_step_norm = np.linalg.norm(pk_newton[z_dof_idx])
        hyperbolic_step_norm = np.sqrt(h_step_norm**2 + z_step_norm**2)

        print(f"  TR: ||Δp||={p_step_norm:.2e}, ||Δh||={h_step_norm:.2e}, ||Δz||={z_step_norm:.2e}")
        print(f"  TR: ||Δ_hyperbolic||={hyperbolic_step_norm:.2e}, trust_radius={trust_radius:.2e}")

        # # Trust parabolic (pressure), limit hyperbolic (enthalpy, composition)
        pk_solution = pk_newton.copy()
        residual_full_vec = self.compute_residual_from_increment(pk_newton, restore_state=True)
        residual_norm_full_vec = np.linalg.norm(residual_full_vec)

        #
        # if hyperbolic_step_norm > trust_radius:
        #     # Scale back ONLY the hyperbolic components
        #     scaling_factor = trust_radius / hyperbolic_step_norm
        #     pk_solution[h_dof_idx] *= scaling_factor
        #     pk_solution[z_dof_idx] *= scaling_factor
        #     print(f"  TR: Scaling hyperbolic by {scaling_factor:.4f} (CFL limit)")
        #     print(f"  TR: Pressure step UNTOUCHED (parabolic)")
        # else:
        #     print(f"  TR: Full Newton step (hyperbolic within CFL-based radius)")

        pk_solution *= trust_radius
        # Evaluate step quality
        residual_new_vec = self.compute_residual_from_increment(pk_solution, restore_state=True)
        residual_norm_new = np.linalg.norm(residual_new_vec)

        print(f"  TR: ||R_full_step||={residual_norm_full_vec:.4e}, ||R_new||={residual_norm_new:.4e}")

        # Accept step if residual decreased or near convergence
        accept_step = residual_norm_new <  residual_norm_full_vec

        if accept_step:
            print(f"  TR: ✓ ACCEPTING")
        else:
            print(f"  TR: ✗ REJECTING")
            pk_solution = np.zeros_like(pk_solution)

        # Return solution and new trust radius (recalculate next iteration)
        return pk_solution, trust_radius

    def backtracking_line_search(
        self,
        delta_x: np.ndarray,
        current_residual: np.ndarray,
        alpha_init: float = 1.0,
        rho: float = 0.5,
        max_iterations: int = 25,
        alpha_min: float = 0.01,  # Minimum acceptable step length
    ) -> float:
        """
        Backtracking line search with Armijo condition (robust, Jacobian-predicted).

        Uses the Jacobian to cheaply predict residuals for candidate alphas and
        only performs the expensive full residual assembly + postprocessing for
        promising alphas. If the Jacobian-based prediction rejects many alphas
        in a row, a full evaluation is forced after a configurable threshold to
        avoid never confirming a valid step.
        """

        # Basic metrics
        residual_norm_current = np.linalg.norm(current_residual)
        phi_current = 0.5 * (residual_norm_current ** 2)

        alpha = alpha_init
        best_alpha: Optional[float] = None
        best_residual = np.inf

        # Parameters
        c_armijo = self.params.get("line_search_armijo", 1e-3)
        use_jacobian_prediction = self.params.get("line_search_use_jacobian_prediction", True)
        force_full_after = int(self.params.get("line_search_force_full_after_predicted_rejects", 3))

        # Try to obtain Jacobian and a J @ delta_x product for cheap prediction
        jacobian_matrix = None
        Jp_full = None
        if use_jacobian_prediction:
            try:
                jacobian_matrix, _ = self.linear_system
                # Compute once (may fail for some sparse/delayed matrix types)
                try:
                    Jp_full = jacobian_matrix @ delta_x
                except Exception:
                    Jp_full = None
            except Exception:
                jacobian_matrix = None
                Jp_full = None

        # Counters and bookkeeping
        full_evals = 0
        predicted_rejects = 0
        tried_alphas: list[float] = []

        # enforce strictly positive alpha_min
        alpha_min = max(alpha_min, 1e-12)

        # Generate alpha sequence deterministically to avoid floating underflow to zero
        for i in range(max_iterations):
            # deterministic alpha sequence: alpha_i = alpha_init * rho**i
            alpha_i = alpha_init * (rho ** i)
            # clamp to alpha_min if below
            if alpha_i < alpha_min:
                alpha = float(alpha_min)
                last_alpha = True
            else:
                alpha = float(alpha_i)
                last_alpha = False

            tried_alphas.append(alpha)
            scaled_increment = alpha * delta_x

            # Jacobian-based cheap prediction
            predicted_ok = False
            if (Jp_full is not None) and use_jacobian_prediction:
                r_pred = current_residual + alpha * Jp_full
                dphi_pred = float(np.dot(current_residual, Jp_full))
                if dphi_pred < 0.0:
                    phi_pred = 0.5 * (np.linalg.norm(r_pred) ** 2)
                    predicted_ok = phi_pred <= phi_current + c_armijo * alpha * dphi_pred
                else:
                    predicted_ok = np.linalg.norm(r_pred) < residual_norm_current

            # Decide whether to run a full expensive residual assembly
            force_full = (predicted_rejects >= force_full_after) and use_jacobian_prediction
            do_full_eval = (Jp_full is None) or predicted_ok or (not use_jacobian_prediction) or force_full

            if not do_full_eval:
                predicted_rejects += 1
                print(f"  Line search iter {i+1}: alpha={alpha:.4f} rejected by Jacobian prediction")
                # If this was the last allowable alpha, break to finalization
                if last_alpha:
                    break
                continue

            # Full evaluation (may be expensive)
            try:
                # Raw residual after applying the scaled increment
                residual_trial = self.compute_residual_from_increment(scaled_increment, restore_state=True)

                # If residual contains non-finite values, treat as failed eval
                if not np.all(np.isfinite(residual_trial)):
                    print(f"  Line search iter {i+1}: non-finite residual from raw eval (alpha={alpha:.4f}), skipping")
                    predicted_rejects = 0
                    if last_alpha:
                        break
                    continue

                # Category breakdown and per-cell exceedences
                _, _, diff_norm_trial, alg_norm_trial, alg_exceeds_trial = self.compute_residuals_by_category(
                    residual_trial
                )

                # Apply same postprocessing that will be applied to accepted steps
                trial_increment_post = scaled_increment.copy()
                try:
                    self.postprocessing_overshoots(trial_increment_post)
                except Exception:
                    pass
                if diff_norm_trial < alg_norm_trial:
                    try:
                        self.postprocessing_thermal_overshoots(trial_increment_post, alg_exceeds_trial)
                    except Exception:
                        pass

                # Residual after postprocessing
                residual_after_post = self.compute_residual_from_increment(trial_increment_post, restore_state=True)

                # If residual after postprocessing contains non-finite values, treat as failed eval
                if not np.all(np.isfinite(residual_after_post)):
                    print(f"  Line search iter {i+1}: non-finite residual after postprocessing (alpha={alpha:.4f}), skipping")
                    predicted_rejects = 0
                    if last_alpha:
                        break
                    continue

                # Norm of the residual after postprocessing
                residual_norm_new = float(np.linalg.norm(residual_after_post))

                # Update counters and best-known (only accept alphas >= alpha_min)
                full_evals += 1
                if (residual_norm_new < best_residual) and (alpha >= alpha_min):
                    best_residual = residual_norm_new
                    best_alpha = float(alpha)

                # Compute directional derivative dphi using Jacobian (if available)
                if Jp_full is not None:
                    Jp_for_dphi = Jp_full
                elif jacobian_matrix is not None:
                    try:
                        Jp_for_dphi = jacobian_matrix @ delta_x
                    except Exception:
                        Jp_for_dphi = None
                else:
                    Jp_for_dphi = None

                dphi = float(np.dot(current_residual, Jp_for_dphi)) if (Jp_for_dphi is not None) else 0.0
                phi_new = 0.5 * (residual_norm_new ** 2)

                # Armijo acceptance
                accepted = False
                if (Jp_for_dphi is not None) and (dphi < 0.0):
                    if phi_new <= phi_current + c_armijo * alpha * dphi:
                        accepted = True
                else:
                    # fallback: require strict residual norm decrease
                    if residual_norm_new < residual_norm_current:
                        accepted = True

                if accepted:
                    reduction_factor = residual_norm_new / residual_norm_current if residual_norm_current > 0 else 0.0
                    print(
                        f"  Line search iter {i+1}: alpha={alpha:.4f}, ||r||={residual_norm_new:.4e} (accepted, factor: {reduction_factor:.4f})"
                    )
                    return alpha

                print(
                    f"  Line search iter {i+1}: alpha={alpha:.4f}, ||r||={residual_norm_new:.4e} (rejected, factor: {residual_norm_new/residual_norm_current:.4f})"
                )

                # reset predicted_rejects if we performed a full eval
                predicted_rejects = 0

            except Exception as e:
                print(f"  Line search iter {i+1}: full evaluation failed at alpha={alpha:.4f}: {e}")

        # End loop: if we never performed any full evals, try a final full eval at the initial alpha
        if full_evals == 0:
            final_alpha = tried_alphas[0] if len(tried_alphas) > 0 else alpha_init
            final_alpha = max(alpha_min, final_alpha)
            print(f"  Line search: no full evals performed; doing final full eval at alpha={final_alpha:.4e}")
            try:
                scaled_increment = final_alpha * delta_x
                residual_trial = self.compute_residual_from_increment(scaled_increment, restore_state=True)
                _, _, diff_norm_trial, alg_norm_trial, alg_exceeds_trial = self.compute_residuals_by_category(
                    residual_trial
                )
                trial_increment_post = scaled_increment.copy()
                try:
                    self.postprocessing_overshoots(trial_increment_post)
                except Exception:
                    pass
                if diff_norm_trial < alg_norm_trial:
                    try:
                        self.postprocessing_thermal_overshoots(trial_increment_post, alg_exceeds_trial)
                    except Exception:
                        pass
                residual_after_post = self.compute_residual_from_increment(trial_increment_post, restore_state=True)
                residual_norm_new = float(np.linalg.norm(residual_after_post))
                if residual_norm_new < residual_norm_current:
                    print(f"  Line search: final eval accepted alpha={final_alpha:.4e}, ||r||={residual_norm_new:.4e}")
                    return final_alpha
                else:
                    print(f"  Line search: final eval rejected; falling back to alpha={alpha_min:.4e}")
                    return alpha_min
            except Exception as e:
                print(f"  Line search: final full eval failed: {e}; falling back to alpha_min={alpha_min:.4e}")
                return alpha_min

        # We have at least one confirmed full evaluation; return best confirmed alpha or fallback
        print(f"  Line search summary: full_evals={full_evals}, predicted_rejects={predicted_rejects}, tried_alphas={len(tried_alphas)}")
        if (best_alpha is not None) and np.isfinite(best_residual) and (best_alpha >= alpha_min):
            print(f"  Line search: using best alpha={best_alpha:.4f} with ||r||={best_residual:.4e}")
            return best_alpha
        # No valid best found above alpha_min -> fallback to conservative minimum
        print(f"  Line search: no good confirmed step found, using fallback alpha={alpha_min:.4f}")
        return float(alpha_min)

    # ----------------------------------------------------------------------------------
    #  Overridable hooks used by the step control above. Base implementations are generic
    #  / no-ops; models with a differential-vs-algebraic structure or variable clamping
    #  (e.g. the Driesner brine model) override them.
    # ----------------------------------------------------------------------------------
    def compute_residuals_by_category(self, residual):
        """Split the residual into differential vs algebraic categories.

        Base default: a single ``"overall"`` differential category (algebraic norm = 0,
        no per-cell exceedances), which is enough for the generic line search. Override
        to expose a physics-specific breakdown.

        Returns:
            ``(diff_residuals, alg_residuals, diff_norm, alg_norm, alg_exceeds)``.
        """
        norm = float(np.linalg.norm(residual))
        return {"overall": norm}, {}, norm, 0.0, {}

    def postprocessing_overshoots(self, delta_x):
        """Clamp/limit overshoots in the increment. No-op by default; override per model."""

    def postprocessing_thermal_overshoots(self, delta_x, alg_exceeds):
        """Clamp thermal/algebraic overshoots. No-op by default; override per model."""

    def set_equations(self):
        super().set_equations()
        self.set_buoyancy_discretization_parameters()

    def set_nonlinear_discretizations(self) -> None:
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

    def lag_buoyancy_direction(self) -> bool:
        """Whether to freeze the buoyancy upwind direction over each time step.

        When ``params["lag_buoyancy_direction"]`` is True (default False) the buoyancy
        upwind direction -- the hybrid inter-phase gravity flux (HU) or the per-phase
        phase potentials (PPU) -- is evaluated once per time step from the previous
        converged state (in :meth:`before_time_step`) and held fixed through the step's
        Newton iterations, instead of being refreshed every iteration. This follows Weis
        et al. (2014, Geofluids 14:347-371, p.353), who use the old velocity field to
        define the upwind nodes for the whole step (cheaper, no visible effect on the
        results, and it removes the upwind-direction flip-flop at flow reversal). The
        option applies to BOTH the hybrid and phase-potential schemes.
        """
        return bool(self.params.get("lag_buoyancy_direction", False))

    def refresh_buoyancy_direction(self) -> None:
        """Per-iteration refresh of the buoyancy upwind direction, unless it is lagged."""
        if not self.lag_buoyancy_direction():
            self.update_buoyancy_driven_fluxes()

    def before_time_step(self) -> None:
        super().before_time_step()
        # Lagged scheme: freeze the buoyancy upwind direction at the previous converged
        # state (now the current iterate) for the whole time step, then rediscretize so
        # the frozen direction is in place before the first nonlinear assembly.
        if self.lag_buoyancy_direction():
            self.update_buoyancy_driven_fluxes()
            self.rediscretize()

    def  after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        super().after_nonlinear_iteration(nonlinear_increment)
        self.refresh_buoyancy_direction()
        self.rediscretize()

    # def before_nonlinear_iteration(self) -> None:
    #
    #     # Try to read iterate-0 vector
    #     try:
    #         x0 = self.equation_system.get_variable_values(iterate_index=0).copy()
    #     except Exception as e:
    #         logger.debug(f"Could not read iterate-0 variable values: {e}")
    #         return
    #
    #     # Determine DOF indices for fields we want to update
    #     try:
    #         p_dof_idx = self.equation_system.dofs_of(['pressure'])
    #         comp_name = None
    #         try:
    #             comp_name = self._get_non_reference_component()
    #         except Exception:
    #             comp_name = None
    #         if comp_name:
    #             z_var = f"z_{comp_name}"
    #         else:
    #             # fallback to any z-like variable if component unknown
    #             z_var = next((v.name for v in self.equation_system.variables if v.name.startswith("z_")), None)
    #             if z_var is None:
    #                 logger.debug("No composition variable found to update.")
    #         if z_var:
    #             z_dof_idx = self.equation_system.dofs_of([z_var])
    #         else:
    #             z_dof_idx = np.array([], dtype=int)
    #
    #         h_dof_idx = self.equation_system.dofs_of(['enthalpy'])
    #         t_dof_idx = self.equation_system.dofs_of(['temperature'])
    #         s_dof_idx = self.equation_system.dofs_of(['s_gas'])
    #     except Exception as e:
    #         logger.warning(f"Failed to determine DOF indices for thermo update: {e}")
    #         return
    #
    #     # Extract current control parameters (flatten)
    #     try:
    #         p0 = np.asarray(x0[p_dof_idx]).flatten() if p_dof_idx.size else np.array([])
    #         z0 = np.asarray(x0[z_dof_idx]).flatten() if z_dof_idx.size else np.array([])
    #         h0 = np.asarray(x0[h_dof_idx]).flatten() if h_dof_idx.size else np.array([])
    #     except Exception as e:
    #         logger.warning(f"Failed to extract current DOF values: {e}")
    #         return
    #
    #     # If any of p,z,h are missing, skip sampling
    #     if p0.size == 0 or h0.size == 0 or z0.size == 0:
    #         logger.debug("Insufficient data for thermo sampling (p/z/h missing). Skipping iterate-0 update.")
    #         return
    #
    #     # Build sampling points and sample thermo state
    #     try:
    #         par_points = np.vstack((z0, h0, p0)).T
    #         self.vtk_sampler.sample_at(par_points)
    #
    #         sampled = self.vtk_sampler.sampled_could.point_data
    #
    #         # Update temperature if available
    #         if "Temperature" in sampled and t_dof_idx.size:
    #             star_t = np.asarray(sampled["Temperature"]).flatten()
    #             if star_t.size == t_dof_idx.size:
    #                 x0[t_dof_idx] = star_t
    #             else:
    #                 # try broadcast-safe assignment for per-cell values
    #                 x0[t_dof_idx] = star_t[: t_dof_idx.size]
    #
    #         # Update saturation if available
    #         if "S_v" in sampled and s_dof_idx.size:
    #             star_s = np.clip(np.asarray(sampled["S_v"]).flatten(), 0.0, 1.0)
    #             if star_s.size == s_dof_idx.size:
    #                 x0[s_dof_idx] = star_s
    #             else:
    #                 x0[s_dof_idx] = star_s[: s_dof_idx.size]
    #
    #         # Optionally update composition if sampler provides a matching key
    #         if z_dof_idx.size:
    #             comp_keys = [f"X_{comp_name}_liq" if comp_name else None,
    #                          f"x_{comp_name}_liq" if comp_name else None,
    #                          z_var]
    #             for key in comp_keys:
    #                 if key and key in sampled:
    #                     star_comp = np.asarray(sampled[key]).flatten()
    #                     if star_comp.size == z_dof_idx.size:
    #                         x0[z_dof_idx] = star_comp
    #                     else:
    #                         x0[z_dof_idx] = star_comp[: z_dof_idx.size]
    #                     break
    #
    #     except Exception as e:
    #         logger.warning(f"Thermo sampling/update failed: {e}")
    #         return
    #
    #     # Write modified iterate-0 back into the equation system (with fallbacks)
    #     try:
    #         if hasattr(self.equation_system, "set_variable_values"):
    #             try:
    #                 self.equation_system.set_variable_values(x0, iterate_index=0)
    #             except TypeError:
    #                 self.equation_system.set_variable_values(x0)
    #         elif hasattr(self.equation_system, "iterates"):
    #             try:
    #                 self.equation_system.iterates[0] = x0
    #             except Exception as e:
    #                 logger.warning(f"Could not assign to equation_system.iterates[0]: {e}")
    #         else:
    #             logger.warning("No supported method found to write iterate-0 values back to equation_system.")
    #     except Exception as e:
    #         logger.warning(f"Failed to write updated iterate-0 back to equation system: {e}")
    #
    #     # keep existing bookkeeping and discretization updates
    #     # self.before_nonlinear_loop()
    #     self.update_buoyancy_driven_fluxes()
    #     self.rediscretize()

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

    def _local_elimination_pairs(self, equation_keys) -> list[tuple[str, str]]:
        """``(variable, equation)`` for every local-elimination (algebraic) equation.

        Discovered from the ``elimination_of_<variable>_on_grids_...`` naming, so this
        captures the eliminated saturations, partial fractions and temperature for ANY
        number of phases/components -- no variable names are hardcoded. For example
        ``elimination_of_x_CH4_gas_on_grids_[0]`` yields the variable ``x_CH4_gas``.
        """
        prefix = "elimination_of_"
        pairs: list[tuple[str, str]] = []
        for equation in equation_keys:
            if equation.startswith(prefix):
                variable = equation[len(prefix):].rsplit("_on_grids", 1)[0]
                pairs.append((variable, equation))
        return pairs

    def permute_equations_and_variables(self):
        """Reorder the (equation, variable) dofs into a CPR-friendly two-field split.

        * **elliptic / pressure field** -- pressure, the interface Darcy flux, and all
          local algebraic eliminations (eliminated saturations, partial fractions and
          temperature); solved (near-)exactly inside the CPR preconditioner.
        * **transport field** -- enthalpy, the thermal interface fluxes, and one overall
          fraction ``z_<comp>`` per NON-reference component.

        The split is derived from the fluid mixture and the equation system, so it works
        unchanged for two-phase/two-component, three-phase/three-component, and beyond --
        no variable or equation names are hardcoded.

        Returns:
            ``(equation_permutation, variable_permutation, field_sizes)`` where the
            permutations are index arrays and ``field_sizes`` is
            ``{'elliptic': n_e, 'transport': n_t}``.
        """
        assembled = self.equation_system.assembled_equation_indices
        equation_keys = list(assembled.keys())

        def equation_named(keyword: str, exclude: str | None = None) -> str | None:
            """First assembled equation whose key contains ``keyword`` (and not
            ``exclude``). ``exclude`` disambiguates the global 'mass_balance_equation'
            from the per-component 'component_mass_balance_equation_*'."""
            return next(
                (eq for eq in equation_keys
                 if keyword in eq and (exclude is None or exclude not in eq)),
                None,
            )

        def variable_dofs(name: str):
            domains = self.mdg.interfaces() if "interface" in name else self.mdg.subdomains()
            md_var = self.equation_system.md_variable(name, domains)
            return self.equation_system.dofs_of(md_var.sub_vars)

        # One overall-fraction variable + component balance per NON-reference component
        # (the reference component, index 0, is fixed by the unity closure).
        component_pairs = [
            (f"z_{c.name}", equation_named(f"component_mass_balance_equation_{c.name}"))
            for c in self.get_components()[1:]
        ]

        # (variable, equation) pairs, in permutation order, for each field.
        elliptic_pairs = [
            ("pressure", equation_named("mass_balance_equation", exclude="component")),
            ("interface_darcy_flux", equation_named("interface_darcy_flux")),
            *self._local_elimination_pairs(equation_keys),
        ]
        transport_pairs = [
            ("enthalpy", equation_named("energy_balance_equation")),
            ("interface_enthalpy_flux", equation_named("interface_enthalpy_flux")),
            ("interface_fourier_flux", equation_named("interface_fourier_flux")),
            *component_pairs,
        ]

        def collect(pairs):
            equation_idx: list[int] = []
            variable_idx: list[int] = []
            for variable, equation in pairs:
                if equation is None or equation not in assembled:
                    continue
                rows = assembled[equation]
                dofs = variable_dofs(variable)
                assert len(rows) == len(dofs), (
                    f"{variable!r}: {len(rows)} equation rows vs {len(dofs)} variable dofs"
                )
                equation_idx.extend(rows)
                variable_idx.extend(dofs)
            return equation_idx, variable_idx

        elliptic_eq, elliptic_var = collect(elliptic_pairs)
        transport_eq, transport_var = collect(transport_pairs)
        return (
            np.array(elliptic_eq + transport_eq),
            np.array(elliptic_var + transport_var),
            {"elliptic": len(elliptic_eq), "transport": len(transport_eq)},
        )

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
        iteration_num = self.nonlinear_solver_statistics.num_iterations

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
        mode = (
            "Jacobian + residual"
            if (iteration_num % 2 == 0 or iteration_num < 10)
            else "residual only"
        )
        logger.info(f"Assembled {mode} in {t_1 - t_0:.2e} seconds.")

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        # Track Newton iterations for this timestep
        current_iterations = self.nonlinear_solver_statistics.num_iterations
        self.newton_iterations_per_timestep.append(current_iterations)
        self.total_newton_iterations += current_iterations

        # Print Newton iteration info for current timestep
        current_time = self.time_manager.time
        timestep_number = self.time_manager.time_index
        logger.info(f"Timestep {timestep_number} (t={current_time:.2e}): {current_iterations} Newton iterations")

        print("*" * 60)
        day_to_second = 86400
        second_to_year = 1.0 / (365 * day_to_second)
        super().after_nonlinear_convergence()  # type:ignore[safe-super]
        print("Number of iterations: ", self.nonlinear_solver_statistics.num_iterations)
        print("Time value (year): ", self.time_manager.time * second_to_year)
        print("Time index: ", self.time_manager.time_index)
        print("*" * 60)
        print("")

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
