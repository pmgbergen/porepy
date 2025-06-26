from typing import Callable
import numpy as np

import porepy as pp
from ...vtk_sampler import VTKSampler


class BCBrineSystem2D(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for temperature
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for pressure
        if sd.dim == 0 and "production_well" in sd.tags:
            return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "dir")
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")
   
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")
    
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sd = boundary_grid.parent
        if sd.dim == 0 and "production_well" in sd.tags:
            return np.array([self._p_PRODUCTION[0]])
        return np.zeros(boundary_grid.num_cells)
