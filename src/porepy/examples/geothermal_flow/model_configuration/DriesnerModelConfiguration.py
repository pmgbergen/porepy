from typing import Callable, Literal, Union, cast, Optional

import numpy as np
import time
import porepy as pp
import porepy.compositional as ppc

from porepy.models.compositional_flow import CompositionalFlowTemplate as FlowTemplate
# from porepy.models.compositional_flow import (
#     CompositionalFractionalFlowTemplate as FlowTemplate,
# )

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
    FlowTemplate,
):
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

    def set_nonlinear_discretizations(self):
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

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

        sub_residuals = [res_p_norm, res_z_norm, res_h_norm, res_t_norm, res_s_norm, res_xs_v_norm,res_xs_l_norm]
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
        # solution *= scaling_factor
        # print(f"Newton correction scale factor: {scaling_factor:.4f}")

        # Identify indices where the residual for primary variables exceeds a tolerance
        residual_tolerance = 1.0
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
        # if thermal_indices.size != 0:
        #     self.postprocessing_thermal_overshoots(solution)

        # Scale down the Newton correction if the non-linear solver is struggling
        # if self.nonlinear_solver_statistics.num_iteration > 5:
        #     scaling_factor = max(0.01, 0.95 ** (self.nonlinear_solver_statistics.num_iteration))
        #     solution *= scaling_factor
        #     print(f"Newton correction scale factor: {scaling_factor:.4f}")
        #
            # res_norm_km1 = self.nonlinear_solver_statistics.residual_norms[-1]
            # accepted_solution = self.equation_system.get_variable_values(iterate_index=0)
            # print("Scaling Newton correction with current residual norm: ", res_norm_km1)
            # for k_search in range(5):
            #     scaling_factor = max(0.01, 0.95 ** (k_search))
            #     self.equation_system.set_variable_values(
            #         values=accepted_solution + scaling_factor * solution, additive=False, iterate_index=0
            #     )
            #     self.update_derived_quantities()
            #     search_res_norm = np.linalg.norm(self.equation_system.assemble(evaluate_jacobian=False))
            #     if res_norm_km1 > search_res_norm:
            #         self.equation_system.set_variable_values(
            #             values=accepted_solution, additive=False, iterate_index=0
            #         )
            #         break
            #     print(f"Newton correction scale factor: {scaling_factor:.4f}")
            #     print(f"Search residual norm: {search_res_norm:.4f}")
            # solution *= scaling_factor

        return solution

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

    def postprocessing_thermal_overshoots(self, delta_x):

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
            # correct temperature from enthalpy
            par_points = np.array((new_z[idx_mp], new_h[idx_mp], new_p[idx_mp])).T
            self.vtk_sampler.sample_at(par_points)
            star_t = self.vtk_sampler.sampled_could.point_data["Temperature"]
            delta_x[t_dof_idx[idx_mp]] = star_t - t_0[idx_mp]
            star_s = self.vtk_sampler.sampled_could.point_data["S_v"]
            star_s = np.clip(star_s, 0.0, 1.0)
            delta_x[s_dof_idx[idx_mp]] = star_s - s_0[idx_mp]

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