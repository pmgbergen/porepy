from typing import Callable

import numpy as np

import porepy as pp

from ...vtk_sampler import VTKSampler

class BCBase(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

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

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-6
        return h

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        z_init = 0.0
        z_inlet = 0.0
        z_NaCl = z_init * np.ones(boundary_grid.num_cells)
        z_NaCl[inlet_idx] = z_inlet
        return z_NaCl

    def bc_values_fractional_flow_component(
            self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        return np.zeros(bg.num_cells)

    def bc_values_fractional_flow_energy(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return self.bc_values_enthalpy(bg)


class BC_single_phase_high_pressure(BCBase):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_inlet = 50.0
        p_outlet = 25.0
        xc = boundary_grid.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
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

class BC_single_phase_moderate_pressure(BCBase):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_inlet = 40.0
        p_outlet = 20.0
        xc = boundary_grid.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p = np.array(list(map(p_linear, xc)))
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 723.15
        t_outlet = 573.15
        T = t_outlet * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T


class BC_single_phase_low_pressure(BCBase):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_inlet = 15.0
        p_outlet = 1.0
        xc = boundary_grid.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p = np.array(list(map(p_linear, xc)))
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 773.15
        t_outlet = 623.15
        T = t_outlet * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T


class BC_two_phase_moderate_pressure(BCBase):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_inlet = 20.0
        p_outlet = 1.0
        xc = boundary_grid.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p = np.array(list(map(p_linear, xc)))
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 673.15
        t_outlet = 423.15
        T = t_outlet * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T

class BC_two_phase_low_pressure(BCBase):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_inlet = 5.0
        p_outlet = 1.0
        xc = boundary_grid.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p = np.array(list(map(p_linear, xc)))
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 573.15
        t_outlet = 353.15
        T = t_outlet * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T