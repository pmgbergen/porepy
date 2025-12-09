from typing import Callable
import numpy as np

import porepy as pp
from ...vtk_sampler import VTKSampler


class BCSinglePhaseHighPressure(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""
    """BC Set up for Liquid phase flow"""

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
    
    def bc_type_advective_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        p_inlet = 50.0e6
        p_outlet = 25.0e6
        p = p_outlet * np.ones(boundary_grid.num_cells)
        p[inlet_idx] = p_inlet
        p[outlet_idx] = p_outlet
        return p
    
    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 623.15  # [K]
        t_outlet = 423.15  # [K]
        T = t_outlet * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T
    
    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = np.zeros_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h = self.vtk_sampler_ptz.sampled_cloud.point_data['H']
        return h

    def bc_values_enthalpy_const(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Constant enthalpy on all boundaries."""
        h_constant = 639.0e3  # J/kg (approx for water at 423 K, ~25-50 MPa)
        
        return h_constant * np.ones(boundary_grid.num_cells)
