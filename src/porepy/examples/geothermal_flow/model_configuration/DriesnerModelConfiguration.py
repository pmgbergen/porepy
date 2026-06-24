from typing import Callable, Literal, Union, cast, Optional, Any

import numpy as np
import time
import porepy as pp
import porepy.compositional as ppc

from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceCriteria,
    ConvergenceInfoCollection,
    ConvergenceStatusCollection,
    DivergenceCriteria,
    SimulationStatus,
)
from porepy.examples.geothermal_flow.flow_model_base import FlowModelBase

from ..vtk_sampler import VTKSampler
from .constitutive_description.BrineConstitutiveDescription import (
    FluidMixture,
    SecondaryEquations,
)
from .geometry_description.geometry_market import SimpleGeometry as ModelGeometry


class BoundaryConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    vtk_sampler_ptz: VTKSampler
    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_inlet = 50.0
        p_outlet = 25.0
        xc = boundary_grid.cell_centers.T
        p_linear = lambda x: (x[0] * p_outlet + (2000.0 - x[0]) * p_inlet) / 2000.0
        p = np.array(list(map(p_linear, xc)))
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 623.15
        t_outlet = 423.15
        T = t_outlet * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-3
        return h

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        z_init = 0.0
        z_inlet = 0.0
        if component.name == "H2O":
            z_H2O = (1 - z_init) * np.ones(boundary_grid.num_cells)
            z_H2O[inlet_idx] = 1 - z_inlet
            return z_H2O
        else:
            z_NaCl = z_init * np.ones(boundary_grid.num_cells)
            z_NaCl[inlet_idx] = z_inlet
            return z_NaCl


class InitialConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 50.0
        p_outlet = 25.0
        xc = sd.cell_centers.T
        p_linear = lambda x: (x[0] * p_outlet + (2000.0 - x[0]) * p_inlet) / 2000.0
        p_init = np.array(list(map(p_linear, xc)))
        return p_init

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15
        return np.ones(sd.num_cells) * t_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-3
        return h_init

    def ic_values_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        return z * np.ones(sd.num_cells)


class DriesnerBrineFlowModel(  # type:ignore[misc]
    ModelGeometry,
    FluidMixture,
    InitialConditions,
    BoundaryConditions,
    SecondaryEquations,
    FlowModelBase,
):
    # Trust region state (persistent across iterations)
    _trust_radius: float = None

    def relative_permeability(
        self, phase: pp.ad.Operator, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        if phase.name == "liq":
            sr = pp.ad.Scalar(0.3)
            s_red = (phase.saturation(domains) - sr) / (pp.ad.Scalar(1.0) - sr)
            kr = pp.ad.Scalar(0.5) * ((s_red**2) ** 0.5 + s_red)
        else:
            kr = phase.saturation(domains)
        return kr

    @property
    def vtk_sampler(self):
        return self._vtk_sampler

    @vtk_sampler.setter
    def vtk_sampler(self, vtk_sampler):
        self._vtk_sampler = vtk_sampler

    @property
    def vtk_sampler_ptz(self):
        return self._vtk_sampler_ptz

    @vtk_sampler_ptz.setter
    def vtk_sampler_ptz(self, vtk_sampler):
        self._vtk_sampler_ptz = vtk_sampler

    # def after_simulation(self):
    #     self.exporter.write_pvd()

    def initial_condition(self):
        super().initial_condition()
        self.set_buoyancy_discretization_parameters()

    def update_flux_values(self):
        super().update_flux_values()
        self.refresh_buoyancy_direction()

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

    def check_convergence(
        self,
        model: pp.PorePyModel,
        nonlinear_increment: np.ndarray,
    ) -> tuple[
        ConvergenceStatusCollection,
        ConvergenceStatusCollection,
        ConvergenceInfoCollection,
    ]:
        """Check convergence and divergence based on passed criteria.

        Parameters:
            model: The model instance specifying the problem to be solved, knowing
                of its metrics for measuring states and residuals.
            nonlinear_increment: Newly obtained solution increment vector.

        Returns:
            tuple[ConvergenceStatusCollection, ConvergenceStatusCollection,
            ConvergenceInfoCollection]: Status and
                info about convergence and divergence.

        """
        # Fetch the residual and current iterate.
        # Update derived quantities
        model.update_derived_quantities()
        # Update buoyancy-driven fluxes (skipped when the direction is lagged/frozen)
        model.refresh_buoyancy_direction()
        # Rediscretize
        model.rediscretize()

        residual = model.equation_system.assemble(evaluate_jacobian=False)
        iterate = model.equation_system.get_variable_values(iterate_index=0)

        # Check convergence status based on current iteration.
        convergence_status, convergence_info = self.convergence_criteria.check(
            increment=nonlinear_increment,
            reference_increment=iterate,
            residual=residual,
            reference_residual=residual,
        )

        # Check divergence status based on current iteration.
        divergence_status = self.divergence_criteria.check(
            increment=nonlinear_increment,
            reference_increment=iterate,
            residual=residual,
            reference_residual=residual,
            num_iterations=self.iteration_index,
        )

        return convergence_status, divergence_status, convergence_info

    def get_equation_indices_by_category(self):
        """
        Get equation indices organized by category: differential and algebraic.

        Returns:
            tuple: (diff_eq_indices, alg_eq_indices)
                - diff_eq_indices: dict of differential equation indices
                - alg_eq_indices: dict of algebraic equation indices
        """
        eq_indices = self.equation_system.assembled_equation_indices

        # Find equation names dynamically
        try:
            temp_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_temperature"))
            s_gas_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_s_gas"))
            x_nacl_gas_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_x_NaCl_gas"))
            x_nacl_liq_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_x_NaCl_liq"))
        except StopIteration as e:
            raise KeyError(f"A required elimination equation was not found in the equation system. {e}")

        # Differential equations (conservation laws)
        diff_eq_indices = {
            'pressure': eq_indices['mass_balance_equation'],
            'composition_NaCl': eq_indices['component_mass_balance_equation_NaCl'],
            'enthalpy': eq_indices['energy_balance_equation'],
        }

        # Algebraic equations (elimination/closure relations)
        alg_eq_indices = {
            'temperature': eq_indices[temp_elim_name],
            'saturation': eq_indices[s_gas_elim_name],
            'mass_fraction_NaCl_gas': eq_indices[x_nacl_gas_elim_name],
            'mass_fraction_NaCl_liquid': eq_indices[x_nacl_liq_elim_name],
        }

        return diff_eq_indices, alg_eq_indices

    def compute_residuals_by_category(self, residual: np.ndarray) -> tuple[dict, dict, float, float, dict]:
        """
        Compute residual norms organized by category.

        Parameters:
            residual: Full residual vector

        Returns:
            tuple: (diff_residuals, alg_residuals, diff_norm, alg_norm, alg_exceeds)
                - diff_residuals: dict of individual differential equation norms
                - alg_residuals: dict of individual algebraic equation norms
                - diff_norm: combined differential equations norm
                - alg_norm: combined algebraic equations norm
                - alg_exceeds: dict mapping algebraic variable name to an array of
                  cell indices where the algebraic residual magnitude exceeds the
                  combined differential residual magnitude for the same cell.
        """
        diff_eq_indices, alg_eq_indices = self.get_equation_indices_by_category()

        # Compute individual differential equation norms
        diff_residuals = {}
        diff_components_arrays = {}
        for name, indices in diff_eq_indices.items():
            arr = residual[indices]
            diff_components_arrays[name] = np.asarray(arr).ravel()
            diff_residuals[name] = np.linalg.norm(diff_components_arrays[name])

        # Compute individual algebraic equation norms
        alg_residuals = {}
        alg_components_arrays = {}
        for name, indices in alg_eq_indices.items():
            arr = residual[indices]
            alg_components_arrays[name] = np.asarray(arr).ravel()
            alg_residuals[name] = np.linalg.norm(alg_components_arrays[name])

        # Compute combined norms by category
        differential_components = [diff_components_arrays[name] for name in diff_eq_indices.keys()]
        algebraic_components = [alg_components_arrays[name] for name in alg_eq_indices.keys()]

        # Combined global norms
        try:
            differential_norm = np.linalg.norm(np.concatenate(differential_components))
        except Exception:
            differential_norm = float(np.linalg.norm(np.hstack(list(diff_components_arrays.values()))))
        try:
            algebraic_norm = np.linalg.norm(np.concatenate(algebraic_components))
        except Exception:
            algebraic_norm = float(np.linalg.norm(np.hstack(list(alg_components_arrays.values()))))

        # Now compute per-cell exceedance: for each algebraic variable, find cell
        # indices where |alg_residual(cell)| > combined_diff_cell_norm(cell).
        # This requires that the per-variable residual arrays have the same length
        # (number of cells). If lengths don't match, we fall back to empty masks
        # and print a diagnostic.
        alg_exceeds: dict = {}
        # Determine if we can compute per-cell combined differential norm
        diff_lengths = [arr.size for arr in differential_components]
        alg_lengths = {name: arr.size for name, arr in alg_components_arrays.items()}

        if len(set(diff_lengths)) == 1:
            # All differential components have same length -> can compute per-cell vector
            n_cells = diff_lengths[0]
            # build per-cell combined differential norm (Euclidean over components)
            diff_stack = np.vstack([diff_components_arrays[name] for name in diff_eq_indices.keys()])
            # shape (n_components, n_cells) -> compute norm over axis=0
            combined_diff_per_cell = np.linalg.norm(diff_stack, axis=0)

            # For each algebraic variable, compare if lengths match and compute mask
            for name, arr in alg_components_arrays.items():
                if arr.size == n_cells:
                    mask = np.abs(arr) > combined_diff_per_cell
                    alg_exceeds[name] = np.nonzero(mask)[0]
                else:
                    # Length mismatch; cannot compute per-cell comparison reliably
                    alg_exceeds[name] = np.array([], dtype=int)
                    print(
                        f"Warning: cannot compare per-cell residuals for '{name}' (size={arr.size}) to differential components (n_cells={n_cells})."
                    )
        else:
            # Differential components have different sizes -- cannot build per-cell comparison
            for name in alg_components_arrays.keys():
                alg_exceeds[name] = np.array([], dtype=int)
            print(
                "Warning: differential residual components have inconsistent lengths; skipping per-cell exceedance check."
            )

        return diff_residuals, alg_residuals, differential_norm, algebraic_norm, alg_exceeds

    def compute_residual_norm(
        self, residual: Optional[np.ndarray], reference_residual: np.ndarray
    ) -> float:
        """
        Compute residual norm for convergence check.

        NOTE: Only differential equations are checked for convergence since
        algebraic equations (temperature, saturation, phase fractions) can
        always be reconstructed from the differential variables.

        Parameters:
            residual: Current residual vector
            reference_residual: Reference residual (not used currently)

        Returns:
            Residual norm based on differential equations only
        """
        if residual is None:
            return np.nan

        # Use unified method to compute residuals by category
        diff_residuals, alg_residuals, differential_norm, algebraic_norm, _ = \
            self.compute_residuals_by_category(residual)

        # Return only differential equations norm for convergence check
        # Algebraic equations can be reconstructed and don't need to be converged
        return differential_norm

    def solve_linear_system(self) -> np.ndarray:
        """
        Solves the linear system of equations, analyzes residuals, and applies
        post-processing steps to the solution.

        Returns:
            np.ndarray: The solution vector of the linear system.
        """
        # Solve the Linear System
        start_time = time.time()

        _, residual_vector = self.linear_system

        # Use unified method to compute residuals by category
        diff_residuals, alg_residuals, differential_residual_norm, algebraic_residual_norm, alg_exceeds = \
            self.compute_residuals_by_category(residual_vector)

        # Report Residuals
        print("\n Report Residuals ")
        print(f"Overall residual norm: {np.linalg.norm(residual_vector):.4e}")

        print("Residual norms for differential equations:")
        for name, norm in diff_residuals.items():
            print(f"  - {name.capitalize()}: {norm:.4e}")

        print("Residual norms for algebraic equations:")
        for name, norm in alg_residuals.items():
            print(f"  - {name.capitalize()}: {norm:.4e}")

        # Print brief summary of algebraic residual exceedances (per-cell)
        print("\nAlgebraic residuals exceeding combined differential residual per cell:")
        for name, idxs in alg_exceeds.items():
            try:
                count = int(idxs.size)
            except Exception:
                count = len(idxs)
            sample = idxs[:10].tolist() if count > 0 else []
            print(f"  - {name}: {count} cells (examples: {sample})")

        # selector for the step control
        # - line search (LS)
        # - trust region (TR)
        # - trust region with line search (TR-LS)
        # - plain newton (no step control) (none)

        step_control_method = self.params.get("step_control_method", "LS")

        # Reset lambda at the start of each time step (iteration 0)
        if self.nonlinear_solver_statistics.num_iterations == 0:
            self._trust_radius = 1.0
            print("Trust region: Reset trust_radius = 1.0 at start of time step")

        residual_norm_current = np.linalg.norm(residual_vector)

        # Get configuration parameters
        step_control_alpha_min = self.params.get("step_control_alpha_min", 0.01)
        activate_after_iteration = self.params.get("activate_step_control_after_iter", 1)
        activate_step_control_Q = self.nonlinear_solver_statistics.num_iterations > activate_after_iteration

        # === CASE 1: Plain Newton (no step control) ===
        if step_control_method == "None":
            print("Step control: Plain Newton (no step control)")
            solution = super().solve_linear_system()

        # === CASE 2: Line Search (LS) ===
        elif step_control_method == "LS":
            solution = super().solve_linear_system()

            # Check if we should activate line search
            residual_future = self.compute_residual_from_increment(solution, restore_state=True)
            residual_norm_future = np.linalg.norm(residual_future)
            increasing_residual_Q = residual_norm_future > residual_norm_current

            if increasing_residual_Q and activate_step_control_Q:
                print("Step control: Line Search (LS)")
                alpha = self.backtracking_line_search(
                    solution, residual_vector, alpha_min=step_control_alpha_min
                )
                solution *= alpha
                print(f"Line search: accepted alpha = {alpha:.4f}")
            else:
                print("Step control: LS not needed (residual decreasing or early iteration)")

        # === CASE 3: Trust Region (TR) ===
        elif step_control_method == "TR":
            if activate_step_control_Q:
                print("Step control: Trust Region (TR) with Levenberg-Marquardt")
                solution, self._trust_radius = self.trust_region_solve(trust_radius=self._trust_radius)
            else:
                print("Step control: TR not active yet (early iteration), using plain Newton")
                solution = super().solve_linear_system()

        # === CASE 4: Trust Region + Line Search (TR-LS) ===
        elif step_control_method == "TR-LS":
            if activate_step_control_Q:
                print("Step control: Trust Region + Line Search (TR-LS)")
                # Step 1: Trust Region solve
                solution, self._trust_radius = self.trust_region_solve(trust_radius=self._trust_radius)

                # Step 2: Line Search on TR solution
                residual_after_tr = self.compute_residual_from_increment(solution, restore_state=True)
                residual_norm_after_tr = np.linalg.norm(residual_after_tr)

                if residual_norm_after_tr > residual_norm_current * 0.9:
                    print("  TR-LS: Applying line search refinement")
                    alpha = self.backtracking_line_search(
                        solution, residual_vector, alpha_min=step_control_alpha_min
                    )
                    solution *= alpha
                    print(f"  TR-LS: Line search alpha = {alpha:.4f}")
                else:
                    print("  TR-LS: Line search not needed (TR solution good)")
            else:
                print("Step control: TR-LS not active yet (early iteration), using plain Newton")
                solution = super().solve_linear_system()

        else:
            raise ValueError(f"Unknown step_control_method: {step_control_method}. "
                           f"Valid options are: 'None', 'LS', 'TR', 'TR-LS'")

        if self.params.get("reduce_linear_system_q", False):
            raise NotImplementedError("The 'reduce_linear_system_q' case is not yet implemented.")

        end_time = time.time()
        print(f"Elapsed time for linear solve: {end_time - start_time:.4f} seconds\n")

        # Post-processing solution overshoots
        self.postprocessing_overshoots(solution)

        # Conditional thermal overshoot post-processing
        # Apply if differential residual is smaller than algebraic residual
        if differential_residual_norm < algebraic_residual_norm:
            print(f"\nThermal overshoot condition triggered:")
            print(f"  Differential norm ({differential_residual_norm:.4e}) < "
                  f"Algebraic norm ({algebraic_residual_norm:.4e})")
            print("  Applying thermal overshoot post-processing...")
            self.postprocessing_thermal_overshoots(solution, alg_exceeds)
        else:
            print(f"\nThermal overshoot condition NOT triggered:")
            print(f"  Differential norm ({differential_residual_norm:.4e}) >= "
                  f"Algebraic norm ({algebraic_residual_norm:.4e})")

        return solution

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
        pk_newton = super().solve_linear_system()
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

    def postprocessing_overshoots(self, delta_x):

        # Define the lambda expression
        inside_ratio = lambda arr, min_v, max_v: 1.0 - np.mean((arr < min_v) | (arr > max_v))


        _, _, tmin, tmax, _, _ = self.vtk_sampler_ptz.search_space.bounds
        tmin -= self.vtk_sampler_ptz.translation_factors[1]
        tmax -= self.vtk_sampler_ptz.translation_factors[1]

        zmin, zmax, hmin, hmax, pmin, pmax = self.vtk_sampler.search_space.bounds
        z_scale, h_scale, p_scale = self.vtk_sampler.conversion_factors
        zmin /= z_scale
        zmax /= z_scale
        hmin /= h_scale
        hmax /= h_scale
        pmin /= p_scale
        pmax /= p_scale

        tb = time.time()
        x0 = self.equation_system.get_variable_values(iterate_index=0)
        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])
        t_dof_idx = self.equation_system.dofs_of(['temperature'])
        s_dof_idx = self.equation_system.dofs_of(['s_gas'])
        xs_v_dof_idx = self.equation_system.dofs_of(['x_NaCl_gas'])
        xs_l_dof_idx = self.equation_system.dofs_of(['x_NaCl_liq'])

        p_0 = x0[p_dof_idx]
        z_0 = x0[z_dof_idx]
        h_0 = x0[h_dof_idx]
        t_0 = x0[t_dof_idx]

        # control overshoots in:
        # pressure
        new_p = delta_x[p_dof_idx] + p_0
        p_scale = inside_ratio(new_p, pmin, pmax)
        new_p = np.clip(new_p, pmin, pmax)
        delta_x[p_dof_idx] = new_p - p_0

        # composition
        new_z = delta_x[z_dof_idx] + z_0
        z_scale = inside_ratio(new_z, 0.0, zmax)
        new_z = np.clip(new_z, 0.0, zmax)
        delta_x[z_dof_idx] = new_z - z_0

        # enthalpy
        new_h = delta_x[h_dof_idx] + h_0
        h_scale = inside_ratio(new_h, hmin, hmax)
        new_h = np.clip(new_h, hmin, hmax)
        delta_x[h_dof_idx] = new_h - h_0

        # temperature
        new_t = delta_x[t_dof_idx] + t_0
        t_scale = inside_ratio(new_t, tmin, tmax)
        new_t = np.clip(new_t, tmin, tmax)
        delta_x[t_dof_idx] = new_t - t_0

        # secondary fractions
        for dof_idx in [s_dof_idx, xs_v_dof_idx, xs_l_dof_idx]:
            new_q = delta_x[dof_idx] + x0[dof_idx]
            new_q = np.clip(new_q, 0.0, 1.0)
            delta_x[dof_idx] = new_q - x0[dof_idx]

        te = time.time()
        print("Elapsed time for postprocessing overshoots: ", te - tb)
        return np.min([p_scale,z_scale,h_scale,t_scale])

    def postprocessing_thermal_overshoots(self, delta_x, alg_exceeds: dict):

        tb = time.time()
        x0 = self.equation_system.get_variable_values(iterate_index=0)
        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])
        t_dof_idx = self.equation_system.dofs_of(['temperature'])
        s_dof_idx = self.equation_system.dofs_of(['s_gas'])

        p_0 = x0[p_dof_idx]
        z_0 = x0[z_dof_idx]
        h_0 = x0[h_dof_idx]
        t_0 = x0[t_dof_idx]
        s_0 = x0[s_dof_idx]

        # control overshoots in:
        # pressure
        new_p = delta_x[p_dof_idx] + p_0
        delta_x[p_dof_idx] = new_p - p_0

        # composition
        new_z = delta_x[z_dof_idx] + z_0
        delta_x[z_dof_idx] = new_z - z_0

        # enthalpy
        new_h = delta_x[h_dof_idx] + h_0
        delta_x[h_dof_idx] = new_h - h_0

        # temperature
        new_t = delta_x[t_dof_idx] + t_0
        delta_x[t_dof_idx] = new_t - t_0

        # saturation
        new_s = delta_x[s_dof_idx] + s_0
        idx_mp = np.where(np.abs(new_s * (1 - new_s)) > 0.0)[0]
        idx_sp = np.where(np.isclose(np.abs(new_s * (1 - new_s)), 0.0))[0]

        # if idx_mp.size != 0:
        #     # correct saturation and temperature from enthalpy
        #     par_points = np.array((z_0[idx_mp], h_0[idx_mp], p_0[idx_mp])).T
        #     self.vtk_sampler.sample_at(par_points)
        #
        #     star_t = self.vtk_sampler.sampled_could.point_data["Temperature"]
        #     delta_x[t_dof_idx[idx_mp]] = star_t - t_0[idx_mp]
        #     star_s = self.vtk_sampler.sampled_could.point_data["S_v"]
        #     delta_x[s_dof_idx[idx_mp]] = star_s - s_0[idx_mp]

        if len(alg_exceeds) != 0:
            # correct from enthalpy using  temperature idxs
            idx_temp = alg_exceeds.get('temperature', np.array([], dtype=int))
            if len(idx_temp) != 0:
                par_points = np.array((new_z[idx_temp], new_h[idx_temp], new_p[idx_temp])).T
                self.vtk_sampler.sample_at(par_points)
                star_t = self.vtk_sampler.sampled_could.point_data["Temperature"]
                delta_x[t_dof_idx[idx_temp]] = star_t - t_0[idx_temp]
                star_s = self.vtk_sampler.sampled_could.point_data["S_v"]
                delta_x[s_dof_idx[idx_temp]] = star_s - s_0[idx_temp]

            idx_sat = alg_exceeds.get('saturation', np.array([], dtype=int))
            if len(idx_sat) !=0:
                par_points = np.array((new_z[idx_sat], new_h[idx_sat], new_p[idx_sat])).T
                self.vtk_sampler.sample_at(par_points)
                star_s = self.vtk_sampler.sampled_could.point_data["S_v"]
                delta_x[s_dof_idx[idx_sat]] = star_s - s_0[idx_sat]

        # if idx_sp.size != 0:
        #     # correct enthalpy from temperature
        #     par_points = np.array((new_z[idx_sp], new_t[idx_sp], new_p[idx_sp])).T
        #     self.vtk_sampler_ptz.sample_at(par_points)
        #     star_h = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-3
        #     delta_x[h_dof_idx[idx_sp]] = star_h - h_0[idx_sp]
        #     star_s = self.vtk_sampler_ptz.sampled_could.point_data["S_v"]
        #     delta_x[s_dof_idx[idx_sp]] = star_s - s_0[idx_sp]


        te = time.time()
        print("Elapsed time for postprocessing thermal overshoots: ", te - tb)
        return

