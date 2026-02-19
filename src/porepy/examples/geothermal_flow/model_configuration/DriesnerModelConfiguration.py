from typing import Callable, Literal, Union, cast, Optional

import numpy as np
import time
import porepy as pp
import porepy.compositional as ppc

# from porepy.models.compositional_flow import CompositionalFlowTemplate as FlowTemplate
# from porepy.models.compositional_flow import (
#     CompositionalFractionalFlowTemplate as FlowModelBase,
# )
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
        h = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-6
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
        h_init = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-6
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

    def after_simulation(self):
        self.exporter.write_pvd()

    def initial_condition(self):
        super().initial_condition()
        self.set_buoyancy_discretization_parameters()

    def update_flux_values(self):
        super().update_flux_values()
        self.update_buoyancy_driven_fluxes()

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

        # Update buoyancy-driven fluxes
        self.update_buoyancy_driven_fluxes()

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
            self.update_buoyancy_driven_fluxes()
            self.rediscretize()

        return current_nonlinear_residual

    def compute_residual_norm(
        self, residual: Optional[np.ndarray], reference_residual: np.ndarray
    ) -> float:
        if residual is None:
            return np.nan

        # Retrieve Equation Indices
        eq_indices = self.equation_system.assembled_equation_indices

        # Find equation names dynamically
        try:
            temp_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_temperature"))
            s_gas_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_s_gas"))
            x_nacl_gas_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_x_NaCl_gas"))
            x_nacl_liq_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_x_NaCl_liq"))
        except StopIteration as e:
            raise KeyError(f"A required elimination equation was not found in the equation system. {e}")

        # Map equation types to their corresponding indices
        diff_eq_indices = {
            'pressure': eq_indices['mass_balance_equation'],
            'composition_NaCl': eq_indices['component_mass_balance_equation_NaCl'],
            'enthalpy': eq_indices['energy_balance_equation'],
            'temperature': eq_indices[temp_elim_name],
        }
        alg_eq_indices = {
            'saturation': eq_indices[s_gas_elim_name],
            'mass_fraction_NaCl_gas': eq_indices[x_nacl_gas_elim_name],
            'mass_fraction_NaCl_liquid': eq_indices[x_nacl_liq_elim_name],
        }

        res_p_norm = np.linalg.norm(residual[diff_eq_indices['pressure']])
        res_z_norm = np.linalg.norm(residual[diff_eq_indices['composition_NaCl']])
        res_h_norm = np.linalg.norm(residual[diff_eq_indices['enthalpy']])
        res_t_norm = np.linalg.norm(residual[diff_eq_indices['temperature']]) / np.sqrt(len(diff_eq_indices['temperature']))
        res_s_norm = np.linalg.norm(residual[alg_eq_indices['saturation']])
        res_xs_v_norm = np.linalg.norm(residual[alg_eq_indices['mass_fraction_NaCl_gas']])
        res_xs_l_norm = np.linalg.norm(residual[alg_eq_indices['mass_fraction_NaCl_liquid']])

        res_t_norm /= 100.0
        # sub_residuals = [res_p_norm, res_z_norm, res_h_norm, res_t_norm, res_s_norm, res_xs_v_norm,res_xs_l_norm]
        sub_residuals = [res_p_norm, res_z_norm, res_h_norm]
        residual_norm = np.max(sub_residuals)
        return residual_norm

    def solve_linear_system(self) -> np.ndarray:
        """
        Solves the linear system of equations, analyzes residuals, and applies
        post-processing steps to the solution.

        Returns:
            np.ndarray: The solution vector of the linear system.
        """
        # Retrieve Equation Indices
        eq_indices = self.equation_system.assembled_equation_indices

        # Find equation names dynamically
        try:
            temp_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_temperature"))
            s_gas_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_s_gas"))
            x_nacl_gas_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_x_NaCl_gas"))
            x_nacl_liq_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_x_NaCl_liq"))
        except StopIteration as e:
            raise KeyError(f"A required elimination equation was not found in the equation system. {e}")

        # Map equation types to their corresponding indices
        diff_eq_indices = {
            'pressure': eq_indices['mass_balance_equation'],
            'composition_NaCl': eq_indices['component_mass_balance_equation_NaCl'],
            'enthalpy': eq_indices['energy_balance_equation'],
            'temperature': eq_indices[temp_elim_name],
        }
        alg_eq_indices = {
            'saturation': eq_indices[s_gas_elim_name],
            'mass_fraction_NaCl_gas': eq_indices[x_nacl_gas_elim_name],
            'mass_fraction_NaCl_liquid': eq_indices[x_nacl_liq_elim_name],
        }

        # Solve the Linear System
        start_time = time.time()

        _, residual_vector = self.linear_system
        residual_norm_current = np.linalg.norm(residual_vector)
        solution = super().solve_linear_system()

        if self.params.get("reduce_linear_system_q", False):
            raise NotImplementedError("The 'reduce_linear_system_q' case is not yet implemented.")

        end_time = time.time()
        print(f"Elapsed time for linear solve: {end_time - start_time:.4f} seconds\n")

        # Report Residuals
        print("\n Report Residuals ")
        print(f"Overall residual norm: {np.linalg.norm(residual_vector):.4e}")
        print("Residual norms for differential equations:")
        for name, indices in diff_eq_indices.items():
            print(f"  - {name.capitalize()}: {np.linalg.norm(residual_vector[indices]):.4e}")

        print("Residual norms for algebraic equations:")
        for name, indices in alg_eq_indices.items():
            print(f"  - {name.capitalize()}: {np.linalg.norm(residual_vector[indices]):.4e}")



        # Post-processing solution overshoots
        self.postprocessing_overshoots(solution)

        # Identify indices where the residual for primary variables exceeds a tolerance
        residual_tolerance = 1.0e-2
        pressure_high_res_idx = np.where(np.abs(residual_vector[diff_eq_indices['pressure']]) > residual_tolerance)[0]
        composition_high_res_idx = \
        np.where(np.abs(residual_vector[diff_eq_indices['composition_NaCl']]) > residual_tolerance)[0]
        enthalpy_high_res_idx = np.where(np.abs(residual_vector[diff_eq_indices['enthalpy']]) > residual_tolerance)[0]
        temperature_high_res_idx = np.where(np.abs(residual_vector[diff_eq_indices['temperature']]) > residual_tolerance)[0]

        # Combine unique indices for thermal overshoot post-processing
        thermal_indices = np.unique(np.concatenate([
            pressure_high_res_idx,
            composition_high_res_idx,
            enthalpy_high_res_idx,
            temperature_high_res_idx
        ]))
        balance_indices = np.unique(np.concatenate([
            pressure_high_res_idx,
            composition_high_res_idx,
            enthalpy_high_res_idx,
        ]))
        # n_iter = 5
        # if temperature_high_res_idx.size != 0 and balance_indices.size ==0:
        # if self.nonlinear_solver_statistics.num_iteration > 10:
        #     self.postprocessing_thermal_overshoots(solution)

        # residual_vector_current = self.compute_residual_from_increment(np.zeros_like(solution), restore_state=True)
        # residual_norm_current_loc = np.linalg.norm(residual_vector_current)
        # assert np.isclose(residual_norm_current, residual_norm_current_loc)

        # Apply step control method: line search, trust region, or none
        step_control_method = self.params.get("step_control_method", "line_search")


        residual_future = self.compute_residual_from_increment(solution, restore_state=True)
        residual_norm_future = np.linalg.norm(residual_future)

        # local_dx = solution.copy()
        # self.postprocessing_thermal_overshoots(local_dx)
        # residual_future_post = self.compute_residual_from_increment(local_dx, restore_state=True)
        # residual_norm_future_post = np.linalg.norm(residual_future_post)

        # print(f"Residual comparison: current={residual_norm_current:.4e}, "
        #       f"future={residual_norm_future:.4e}, future_post={residual_norm_future_post:.4e}")
        #
        # if residual_norm_future_post < residual_norm_future:
        #     print(
        #         f"Post-processing check: residual norm after thermal overshoot correction: {residual_norm_future_post:.4e}")
        #     print("Thermal overshoot post-processing reduced the residual. Applying correction.")
        #     solution = local_dx
        # else:
        #     print(f"Thermal overshoot post-processing INCREASED residual from {residual_norm_future:.4e} "
        #           f"to {residual_norm_future_post:.4e}. NOT applying correction.")

        step_control_threshold = self.params.get("step_control_threshold", 1e-6)
        step_control_alpha_min = self.params.get("step_control_alpha_min", 0.01)

        increasing_residual_Q = residual_norm_future > residual_norm_current
        activate_step_control_method_Q = self.nonlinear_solver_statistics.num_iteration > 3

        if step_control_method == "line_search" and increasing_residual_Q and activate_step_control_method_Q:
            # Use backtracking line search
            alpha = self.backtracking_line_search(
                solution, residual_vector, alpha_min=step_control_alpha_min
            )
            solution *= alpha
            print(f"Line search: accepted alpha = {alpha:.4f}")

        elif step_control_method == "trust_region" and activate_step_control_method_Q:
            # Use trust region method
            # Get or initialize trust radius
            if self._trust_radius is None:
                self._trust_radius = self.params.get("trust_region_initial_radius", 1.0)

            trust_region_min_radius = self.params.get("trust_region_min_radius", 0.1)

            # Get Jacobian for quadratic model (optional, method will fallback if not available)
            try:
                jacobian_matrix, _ = self.linear_system
            except:
                jacobian_matrix = None

            alpha, self._trust_radius = self.trust_region_step(
                solution,
                residual_vector,
                trust_radius=self._trust_radius,
                alpha_min=step_control_alpha_min,
                min_radius=trust_region_min_radius,
                jacobian=jacobian_matrix  # Pass Jacobian for quadratic model
            )
            solution *= alpha
            print(f"Trust region: accepted alpha = {alpha:.4f}, new radius = {self._trust_radius:.4f}")


        elif step_control_method in ["line_search", "trust_region"]:
            # Skip step control if already well-converged or in early iterations
            print(f"Step control skipped: residual norm {residual_norm_current:.4e} "
                  f"(method: {step_control_method})")
        # else:
        #     local_dx = solution.copy()
        #     self.postprocessing_thermal_overshoots(local_dx)
        #     residual_norm_future_post = self.compute_residual_from_increment(local_dx, restore_state=True)
        #     if residual_norm_future_post < residual_norm_future:
        #         print(
        #             f"Post-processing check: residual norm after thermal overshoot correction: {residual_norm_future_post:.4e}")
        #         print("Thermal overshoot post-processing reduced the residual. Applying correction.")
        #         solution = local_dx

        return solution

    def backtracking_line_search(
        self,
        delta_x: np.ndarray,
        current_residual: np.ndarray,
        alpha_init: float = 1.0,
        rho: float = 0.5,
        c: float = 1e-4,
        max_iterations: int = 15,
        alpha_min: float = 0.01,  # Minimum acceptable step length
    ) -> float:
        """
        Backtracking line search with Armijo condition.

        Parameters:
            delta_x: Newton step (correction)
            current_residual: Residual at current iterate
            alpha_init: Initial step length (default: 1.0)
            rho: Step reduction factor (default: 0.5)
            c: Armijo parameter (default: 1e-4)
            max_iterations: Maximum backtracking steps (default: 10)
            alpha_min: Minimum acceptable step length (default: 0.01)

        Returns:
            Accepted step length alpha
        """

        residual_norm_current = np.linalg.norm(current_residual)
        alpha = alpha_init
        best_alpha = alpha_init
        best_residual = np.inf

        # Tolerance for accepting step even if residual doesn't decrease
        # (useful when already near convergence)
        relative_tolerance = 1.1  # Accept if residual increases by less than 10%

        for i in range(max_iterations):
            # Don't try alphas below the minimum threshold
            if alpha < alpha_min:
                print(f"  Line search: alpha={alpha:.4f} below minimum {alpha_min:.4f}, "
                      f"using best found alpha={best_alpha:.4f}")
                break

            # Compute the increment scaled by alpha
            scaled_increment = alpha * delta_x

            # Evaluate residual at new point using compute_residual_from_increment
            # restore_state=True so we can try different alphas
            try:
                residual_new = self.compute_residual_from_increment(
                    scaled_increment, restore_state=True
                )
                residual_norm_new = np.linalg.norm(residual_new)

                # Track the best alpha found so far
                if residual_norm_new < best_residual:
                    best_residual = residual_norm_new
                    best_alpha = alpha

                # Accept step if:
                # 1. Residual decreases, OR
                # 2. Residual increase is negligible (within tolerance)
                if residual_norm_new < residual_norm_current * relative_tolerance:
                    # Accept this step length
                    reduction_factor = residual_norm_new / residual_norm_current
                    print(f"  Line search iter {i+1}: alpha={alpha:.4f}, "
                          f"||r||={residual_norm_new:.4e} (accepted, factor: {reduction_factor:.4f})")
                    return alpha
                else:
                    print(f"  Line search iter {i+1}: alpha={alpha:.4f}, "
                          f"||r||={residual_norm_new:.4e} (rejected, factor: {residual_norm_new/residual_norm_current:.4f})")

            except Exception as e:
                print(f"  Line search iter {i+1}: failed at alpha={alpha:.4f}: {e}")

            # Reduce step length
            alpha *= rho

        # If no sufficient decrease found, return the best alpha found (if above minimum)
        # or a reasonable fallback
        if best_alpha >= alpha_min:
            print(f"  Line search: using best alpha={best_alpha:.4f} with ||r||={best_residual:.4e}")
            return best_alpha
        else:
            print(f"  Line search: no good step found, using fallback alpha={alpha_min:.4f}")
            return alpha_min

    def trust_region_step(
        self,
        delta_x: np.ndarray,
        current_residual: np.ndarray,
        trust_radius: float = None,
        eta_1: float = 0.1,
        eta_2: float = 0.75,
        gamma_1: float = 0.5,
        gamma_2: float = 2.0,
        alpha_min: float = 0.01,
        alpha_max: float = 1.0,
        max_radius: float = 10.0,
        min_radius: float = 0.1,
        jacobian = None,  # NEW: Jacobian matrix for quadratic model
    ) -> tuple[float, float]:
        """
        Trust region method for step length control based on residual norms.

        Uses a quadratic model based on the Jacobian to predict residual reduction:
        m(p) = 0.5*||r||^2 + r^T*J*p + 0.5*p^T*(J^T*J)*p

        This method computes the actual reduction vs. predicted reduction in residual
        to determine whether to accept/reject the step and how to adjust the trust region.

        Parameters:
            delta_x: Newton step (correction)
            current_residual: Residual at current iterate
            trust_radius: Current trust region radius (if None, uses ||delta_x||)
            eta_1: Threshold for poor agreement (default: 0.1)
            eta_2: Threshold for good agreement (default: 0.75)
            gamma_1: Trust radius reduction factor (default: 0.5)
            gamma_2: Trust radius expansion factor (default: 2.0)
            alpha_min: Minimum acceptable step length (default: 0.01)
            alpha_max: Maximum step length (default: 1.0)
            max_radius: Maximum trust region radius (default: 10.0)
            min_radius: Minimum trust region radius (default: 0.1)
            jacobian: Jacobian matrix (if available) for quadratic model

        Returns:
            tuple: (alpha, new_trust_radius)
                - alpha: Accepted step length
                - new_trust_radius: Updated trust region radius for next iteration
        """
        # Compute current residual norm
        residual_norm_current = np.linalg.norm(current_residual)

        # Initialize trust radius if not provided
        if trust_radius is None or trust_radius < min_radius:
            trust_radius = max(min_radius, np.linalg.norm(delta_x) * 0.5)
            print(f"  Trust region: initializing/resetting radius to {trust_radius:.4f}")

        # Compute step norm
        step_norm = np.linalg.norm(delta_x)

        # Determine initial alpha based on trust radius
        if step_norm > trust_radius:
            alpha_trial = trust_radius / step_norm
        else:
            alpha_trial = min(alpha_max, 1.0)

        # Ensure alpha is within bounds
        alpha_trial = max(alpha_min, min(alpha_max, alpha_trial))

        print(f"  Trust region: radius={trust_radius:.4f}, step_norm={step_norm:.4f}, "
              f"alpha_trial={alpha_trial:.4f}")

        # Compute scaled increment
        scaled_increment = alpha_trial * delta_x

        # Evaluate residual at new point
        residual_new = self.compute_residual_from_increment(
            scaled_increment, restore_state=True
        )
        residual_norm_new = np.linalg.norm(residual_new)


        # Compute actual reduction in residual
        actual_reduction = residual_norm_current - residual_norm_new

        # Predict reduction using quadratic model
        # For a proper trust region, we use: m(p) = f + g^T*p + 0.5*p^T*H*p
        # where f = 0.5*||r||^2, g = J^T*r, H ≈ J^T*J (Gauss-Newton)
        # The predicted reduction is: -g^T*p - 0.5*p^T*H*p
        # For scaling by alpha: p = alpha * delta_x

        # Try to use Jacobian for quadratic model
        if jacobian is not None:
            try:
                # Compute J * delta_x (reuse if Newton step: delta_x solves J*delta_x = -r)
                J_delta_x = jacobian @ delta_x

                # Quadratic model prediction for step p = alpha * delta_x:
                # pred_red = -alpha * g^T * delta_x - 0.5 * alpha^2 * delta_x^T * (J^T*J) * delta_x
                #          = -alpha * (J^T*r)^T * delta_x - 0.5 * alpha^2 * (J*delta_x)^T * (J*delta_x)
                #          = -alpha * r^T * (J*delta_x) - 0.5 * alpha^2 * ||J*delta_x||^2

                linear_term = -np.dot(current_residual, J_delta_x)
                quadratic_term = -0.5 * np.dot(J_delta_x, J_delta_x)

                predicted_reduction = alpha_trial * linear_term + (alpha_trial ** 2) * quadratic_term

                print(f"  Trust region: using quadratic model (linear={linear_term:.4e}, "
                      f"quad={quadratic_term:.4e})")

            except Exception as e:
                # Fallback to simple linear model if computation fails
                print(f"  Trust region: quadratic model failed, using simple model ({e})")
                predicted_reduction = alpha_trial * residual_norm_current
        else:
            # Fallback to simple linear model if Jacobian not available
            print(f"  Trust region: Jacobian not available, using simple linear model")
            predicted_reduction = alpha_trial * residual_norm_current

        # Compute reduction ratio (rho)
        # IMPORTANT: rho should only be meaningful when predicted_reduction > 0
        # If predicted reduction is negative or zero, the model is predicting an increase

        reduction_factor = residual_norm_new / residual_norm_current

        # First check: Did the residual actually decrease?
        if actual_reduction <= 0:
            # Residual increased or stayed the same - this is always bad
            rho = -1.0  # Force poor agreement
            print(f"  Trust region: ||r_old||={residual_norm_current:.4e}, "
                  f"||r_new||={residual_norm_new:.4e}, rho={rho:.4f} (REJECTED: residual increased)")
        elif predicted_reduction <= 0:
            # Model predicts increase, but we got decrease - model is very wrong
            rho = 0.0  # Poor agreement, but not as bad as actual increase
            print(f"  Trust region: ||r_old||={residual_norm_current:.4e}, "
                  f"||r_new||={residual_norm_new:.4e}, rho={rho:.4f} (WARNING: model predicted increase)")
        elif abs(predicted_reduction) < 1e-14:
            # Predicted reduction is too small to be meaningful
            rho = 0.0
            print(f"  Trust region: ||r_old||={residual_norm_current:.4e}, "
                  f"||r_new||={residual_norm_new:.4e}, rho={rho:.4f} (WARNING: predicted reduction too small)")
        else:
            # Normal case: both actual and predicted reductions are positive
            rho = actual_reduction / predicted_reduction
            print(f"  Trust region: ||r_old||={residual_norm_current:.4e}, "
                  f"||r_new||={residual_norm_new:.4e}, rho={rho:.4f}")

        # Decide whether to accept the step and how to adjust trust radius

        # SAFETY CHECK: Never accept a step that significantly increases the residual
        # This is an absolute safeguard regardless of rho
        max_allowed_increase = 1.5  # Allow up to 50% increase
        if reduction_factor > max_allowed_increase:
            print(f"  Trust region: SAFETY REJECT - residual increased by {reduction_factor:.2f}x "
                  f"(max allowed: {max_allowed_increase:.2f}x)")
            new_trust_radius = max(min_radius, trust_radius * 0.25)  # Aggressive shrink

            # Try much smaller step
            for alpha_try in [alpha_trial * 0.1, alpha_trial * 0.01, alpha_min]:
                if alpha_try < alpha_min * 0.1:
                    break
                try:
                    scaled_inc_try = alpha_try * delta_x
                    res_norm_try = self.compute_residual_from_increment(
                        scaled_inc_try, restore_state=True
                    )
                    if res_norm_try < residual_norm_current * max_allowed_increase:
                        print(f"  Trust region: found safer alpha={alpha_try:.4e}, "
                              f"||r||={res_norm_try:.4e}")
                        return alpha_try, new_trust_radius
                except:
                    continue

            # If nothing works, use minimum step
            print(f"  Trust region: no safe step found, using alpha={alpha_min:.4e}")
            return alpha_min, new_trust_radius

        if rho < eta_1:
            # Poor agreement: reject step or use smaller step, shrink trust region
            print(f"  Trust region: poor agreement (rho={rho:.4f} < {eta_1:.4f}), "
                  f"shrinking radius by {gamma_1:.2f}x")
            new_trust_radius = max(min_radius, trust_radius * gamma_1)

            # Check if we're stuck at minimum radius
            if trust_radius <= min_radius * 1.01:
                # Already at minimum, can't shrink further
                # This suggests the Newton step is very poor
                print(f"  Trust region: at minimum radius, trying line search approach")
                # Try different alphas via simple backtracking
                for alpha_try in [alpha_trial * 0.5, alpha_trial * 0.25, alpha_min]:
                    if alpha_try < alpha_min:
                        break
                    try:
                        scaled_inc_try = alpha_try * delta_x
                        res_norm_try = self.compute_residual_from_increment(
                            scaled_inc_try, restore_state=True
                        )
                        if res_norm_try < residual_norm_current * 1.5:
                            print(f"  Trust region: found acceptable alpha={alpha_try:.4f}, "
                                  f"||r||={res_norm_try:.4e}")
                            return alpha_try, min_radius
                    except:
                        continue

                # If nothing works, reset radius to a larger value for next iteration
                print(f"  Trust region: resetting radius to {min_radius * 5:.4f} for next iteration")
                return alpha_min, min_radius * 5

            # Normal shrinking case
            if reduction_factor < 1.1:
                # Still some reduction, accept with smaller alpha
                alpha_accepted = max(alpha_min, alpha_trial * gamma_1)
                print(f"  Trust region: accepting reduced step alpha={alpha_accepted:.4f}")
                return alpha_accepted, new_trust_radius
            else:
                # No improvement, use minimum step
                print(f"  Trust region: no improvement, using alpha={alpha_min:.4f}")
                return alpha_min, new_trust_radius

        elif rho < eta_2:
            # Moderate agreement: accept step, keep trust radius
            print(f"  Trust region: moderate agreement (rho={rho:.4f}), "
                  f"keeping radius={trust_radius:.4f}")
            new_trust_radius = trust_radius
            return alpha_trial, new_trust_radius

        else:
            # Good agreement: accept step, expand trust radius
            print(f"  Trust region: good agreement (rho={rho:.4f} >= {eta_2:.4f}), "
                  f"expanding radius by {gamma_2:.2f}x")
            new_trust_radius = min(trust_radius * gamma_2, max_radius)

            # If we're not at the boundary, try full step
            if alpha_trial >= 0.99:
                return 1.0, new_trust_radius
            else:
                return alpha_trial, new_trust_radius

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

    def postprocessing_thermal_overshoots(self, delta_x, idx_temp=[]):

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

        if idx_mp.size != 0:
            # correct saturation and temperature from enthalpy
            par_points = np.array((z_0[idx_mp], h_0[idx_mp], p_0[idx_mp])).T
            self.vtk_sampler.sample_at(par_points)

            star_t = self.vtk_sampler.sampled_could.point_data["Temperature"]
            delta_x[t_dof_idx[idx_mp]] = star_t - t_0[idx_mp]
            star_s = self.vtk_sampler.sampled_could.point_data["S_v"]
            star_s = np.clip(star_s, 0.0, 1.0)
            delta_x[s_dof_idx[idx_mp]] = star_s - s_0[idx_mp]

        # if idx_temp.size != 0:
        #     # correct temperature from enthalpy
        #     par_points = np.array((new_z[idx_temp], new_h[idx_temp], new_p[idx_temp])).T
        #     self.vtk_sampler.sample_at(par_points)
        #     star_t = self.vtk_sampler.sampled_could.point_data["Temperature"]
        #     delta_x[t_dof_idx[idx_temp]] = star_t - t_0[idx_temp]
        #     star_s = self.vtk_sampler.sampled_could.point_data["S_v"]
        #     star_s = np.clip(star_s, 0.0, 1.0)
        #     delta_x[s_dof_idx[idx_temp]] = star_s - s_0[idx_temp]

        if idx_sp.size != 0:
            # correct enthalpy from temperature
            par_points = np.array((new_z[idx_sp], new_t[idx_sp], new_p[idx_sp])).T
            self.vtk_sampler_ptz.sample_at(par_points)
            star_h = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-6
            delta_x[h_dof_idx[idx_sp]] = star_h - h_0[idx_sp]
            star_s = self.vtk_sampler_ptz.sampled_could.point_data["S_v"]
            star_s = np.clip(star_s, 0.0, 1.0)
            delta_x[s_dof_idx[idx_sp]] = star_s - s_0[idx_sp]


        te = time.time()
        print("Elapsed time for postprocessing thermal overshoots: ", te - tb)
        return

