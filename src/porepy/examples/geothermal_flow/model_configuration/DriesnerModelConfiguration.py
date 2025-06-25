from typing import Callable, Literal, Union, cast

import numpy as np
import time
import porepy as pp
import porepy.compositional as ppc

# from porepy.models.compositional_flow import CompositionalFlowTemplate as FlowTemplate
from porepy.models.compositional_flow import (
    CompositionalFractionalFlowTemplate as FlowTemplate,
)

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

    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""

        eq_idx = self.equation_system.assembled_equation_indices
        equation_names = list(eq_idx.keys())
        # retrieve equations names
        t_name = [name for name in equation_names if name.startswith("elimination_of_temperature")][0]
        s_name = [name for name in equation_names if name.startswith("elimination_of_s_gas")][0]
        xs_v_name = [name for name in equation_names if name.startswith("elimination_of_x_NaCl_gas")][0]
        xs_l_name = [name for name in equation_names if name.startswith("elimination_of_x_NaCl_liq")][0]

        # differential
        p_eq_idx = eq_idx['mass_balance_equation']
        z_eq_idx = eq_idx['component_mass_balance_equation_NaCl']
        h_eq_idx = eq_idx['energy_balance_equation']
        t_eq_idx = eq_idx[t_name]
        # algebraic
        s_eq_idx = eq_idx[s_name]
        xs_v_eq_idx = eq_idx[xs_v_name]
        xs_l_eq_idx = eq_idx[xs_l_name]

        tb = time.time()
        _, res_g = self.linear_system
        sol = super().solve_linear_system()
        reduce_linear_system_q = self.params.get("reduce_linear_system_q", False)
        if reduce_linear_system_q:
            raise ValueError("Case not implemented yet.")
        te = time.time()
        print("Overall residual norm: ", np.linalg.norm(res_g))
        print("Variables involving differential operators:")
        print("Pressure residual norm: ", np.linalg.norm(res_g[p_eq_idx]))
        print("Composition residual norm: ", np.linalg.norm(res_g[z_eq_idx]))
        print("Enthalpy residual norm: ", np.linalg.norm(res_g[h_eq_idx]))
        print("Temperature residual norm: ", np.linalg.norm(res_g[t_eq_idx]))
        print("Variables involving algebraic operators:")
        print("Saturation residual norm: ", np.linalg.norm(res_g[s_eq_idx]))
        print("Xs_v residual norm: ", np.linalg.norm(res_g[xs_v_eq_idx]))
        print("Xs_l residual norm: ", np.linalg.norm(res_g[xs_l_eq_idx]))
        print("Elapsed time linear solve: ", te - tb)
        print("")

        res_norm = np.linalg.norm(res_g)
        p_res_norm = np.linalg.norm(res_g[p_eq_idx])
        z_res_norm = np.linalg.norm(res_g[z_eq_idx])
        h_res_norm = np.linalg.norm(res_g[h_eq_idx])
        primary_res_norm = np.max([p_res_norm,z_res_norm,h_res_norm])
        self.postprocessing_overshoots(sol)
        eps_tol = 1.0e-4
        p_idx = np.where(res_g[p_eq_idx] > eps_tol)[0]
        z_idx = np.where(res_g[z_eq_idx] > eps_tol)[0]
        h_idx = np.where(res_g[h_eq_idx] > eps_tol)[0]
        alpha_idx = np.unique(np.concatenate([p_idx,z_idx,h_idx]))
        self.postprocessing_thermal_overshoots(sol, alpha_idx)

        # if self.nonlinear_solver_statistics.num_iteration > 10:
        dx_scale = np.max([0.05, 0.98 ** self.nonlinear_solver_statistics.num_iteration])
        print("Searching ...")
        print("Newton correction factor: ", dx_scale)
        print("")
        sol *= dx_scale

        return sol

    def postprocessing_overshoots(self, delta_x):

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
        xw_v_dof_idx = self.equation_system.dofs_of(['x_H2O_gas'])
        xw_l_dof_idx = self.equation_system.dofs_of(['x_H2O_liq'])
        xs_v_dof_idx = self.equation_system.dofs_of(['x_NaCl_gas'])
        xs_l_dof_idx = self.equation_system.dofs_of(['x_NaCl_liq'])

        p_0 = x0[p_dof_idx]
        z_0 = x0[z_dof_idx]
        h_0 = x0[h_dof_idx]
        t_0 = x0[t_dof_idx]

        # control overshoots in:
        # pressure
        new_p = delta_x[p_dof_idx] + p_0
        new_p = np.clip(new_p, 1.0e-3, 70.0)
        delta_x[p_dof_idx] = new_p - p_0

        # composition
        new_z = delta_x[z_dof_idx] + z_0
        new_z = np.clip(new_z, 0.0, 0.35)
        delta_x[z_dof_idx] = new_z - z_0

        # enthalpy
        new_h = delta_x[h_dof_idx] + h_0
        new_h = np.clip(new_h, 0.0, 4.0)
        delta_x[h_dof_idx] = new_h - h_0

        # temperature
        new_t = delta_x[t_dof_idx] + t_0
        new_t = np.clip(new_t, 0.0, 1273.15)
        delta_x[t_dof_idx] = new_t - t_0

        # secondary fractions
        for dof_idx in [s_dof_idx, xw_v_dof_idx, xw_l_dof_idx, xs_v_dof_idx, xs_l_dof_idx]:
            new_q = delta_x[dof_idx] + x0[dof_idx]
            new_q = np.clip(new_q, 0.0, 1.0)
            delta_x[dof_idx] = new_q - x0[dof_idx]

        te = time.time()
        print("Elapsed time for postprocessing overshoots: ", te - tb)
        return

    def postprocessing_thermal_overshoots(self, delta_x, alpha_idx):

        if alpha_idx.size == 0:
            return

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
        xw_v_dof_idx = self.equation_system.dofs_of(['x_H2O_gas'])
        xw_l_dof_idx = self.equation_system.dofs_of(['x_H2O_liq'])
        xs_v_dof_idx = self.equation_system.dofs_of(['x_NaCl_gas'])
        xs_l_dof_idx = self.equation_system.dofs_of(['x_NaCl_liq'])

        p_0 = x0[p_dof_idx]
        z_0 = x0[z_dof_idx]
        h_0 = x0[h_dof_idx]
        t_0 = x0[t_dof_idx]
        s_0 = x0[s_dof_idx]

        # control overshoots in:
        # pressure
        new_p = delta_x[p_dof_idx] + p_0
        new_p = np.clip(new_p, 1.0e-3, 70.0)
        delta_x[p_dof_idx] = new_p - p_0

        # composition
        new_z = delta_x[z_dof_idx] + z_0
        new_z = np.clip(new_z, 0.0, 0.35)
        delta_x[z_dof_idx] = new_z - z_0

        # enthalpy
        new_h = delta_x[h_dof_idx] + h_0
        new_h = np.clip(new_h, 0.0, 4.0)
        delta_x[h_dof_idx] = new_h - h_0

        # temperature
        new_t = delta_x[t_dof_idx] + t_0
        new_t = np.clip(new_t, 0.0, 1273.15)
        delta_x[t_dof_idx] = new_t - t_0

        # temperature
        new_t = delta_x[t_dof_idx] + t_0
        new_t = np.clip(new_t, 0.0, 1273.15)
        delta_x[t_dof_idx] = new_t - t_0

        # saturation
        new_s = delta_x[s_dof_idx] + s_0
        new_s = np.clip(new_s, 0.0, 1.0)
        idx_tp = np.where((new_s[alpha_idx] > 0.0) & (new_s[alpha_idx] < 1.0))[0]
        # Get indexes of values that are close to 0.0 or close to 1.0
        idx_sp = np.where(np.isclose(new_s[alpha_idx], 0.0) | np.isclose(new_s[alpha_idx], 1.0))[0]

        if idx_tp.size != 0:
            # correct temperature from enthalpy
            par_points = np.array((new_z[idx_tp], new_h[idx_tp], new_p[idx_tp])).T
            self.vtk_sampler.sample_at(par_points)
            star_t = self.vtk_sampler.sampled_could.point_data["Temperature"]
            delta_x[t_dof_idx[idx_tp]] = star_t - t_0[idx_tp]
            star_s = self.vtk_sampler.sampled_could.point_data["S_v"]
            star_s = np.clip(star_s, 0.0, 1.0)
            delta_x[s_dof_idx[idx_tp]] = star_s - s_0[idx_tp]
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
        print("Elapsed time for postprocessing overshoots: ", te - tb)
        return