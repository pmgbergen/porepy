from typing import Callable

import numpy as np

import porepy as pp

from ...obl_sampler import VTKSampler

class BCBase(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    obl_sampler_ptz: VTKSampler

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

    def bc_salinity(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Overall NaCl fraction of the boundary fluid; zero unless a solver
        overrides it."""
        return np.zeros(boundary_grid.num_cells)

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = self.bc_salinity(boundary_grid)
        par_points = np.array((z_NaCl, t, p)).T
        self.obl_sampler_ptz.sample_at(par_points)
        h = self.obl_sampler_ptz.sampled_could.point_data["H"] * 1.0e-3
        return h

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        return self.bc_salinity(boundary_grid)

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

class BC_two_phase_Figure_8_left_panel(BCBase):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # inlet_idx, outlet_idx = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, np.concatenate(self.get_inlet_outlet_sides(sd)), "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, outlet_idx, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_outlet = 1.0
        return np.ones(boundary_grid.num_cells) * p_outlet

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_outlet = 283.15
        T =  t_outlet * np.ones(boundary_grid.num_cells)
        t_inlet = 673.15
        T[inlet_idx] = t_inlet
        return T

class BC_two_phase_steady_state(BCBase):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        _, outlet_facets = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, outlet_facets, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_inlet = 15.0
        p_outlet = 5.0
        xc = boundary_grid.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p = np.array(list(map(p_linear, xc)))
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        t_inlet = 723.15
        t_outlet = 473.15
        xc = boundary_grid.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        T_linear = (
            lambda x: (x[dir_idx] * t_outlet + (2000.0 - x[dir_idx]) * t_inlet) / 2000.0
        )
        T = np.array(list(map(T_linear, xc)))
        return T


class BC_three_phase_closed(BCBase):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # get_inlet_outlet_sides -> (inlet, outlet); the TOP is the outlet slot. Unpack it
        # (the whole tuple cannot be passed straight to pp.BoundaryCondition as `faces`).
        _, top_facets = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, top_facets, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_top = 10.0
        p_bottom = 15.0
        xc = boundary_grid.cell_centers.T
        dir_idx = 1
        p_linear = (
            lambda x: (x[dir_idx] * p_top + (100.0 - x[dir_idx]) * p_bottom) / 100.0
        )
        p = np.array(list(map(p_linear, xc)))
        return p

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # np.zeros (not zeros_like): zeros_like of the integer num_cells is a 0-d scalar,
        # which breaks the boundary-operator concatenation on mixed-dimensional grids.
        return np.zeros(boundary_grid.num_cells)