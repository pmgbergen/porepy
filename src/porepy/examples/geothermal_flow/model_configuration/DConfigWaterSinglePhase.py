import numpy as np

import porepy as pp
import porepy.compositional as ppc
from porepy.models.compositional_flow import (
    BoundaryConditionsCF,
    CFModelMixin,
    InitialConditionsCF,
    PrimaryEquationsCF,
)
from .constitutive_description.PureWaterConstitutiveDescription import (
    FluidMixture,
    SecondaryEquations,
)
from .geometry_description.geometry_market import SimpleGeometry1D as ModelGeometry

## Bc to simulate pure-water with single liquid phase flow.
class BoundaryConditions(BoundaryConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""
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
    
    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        h_inlet = 1.5e6
        h_outlet = 6.45e5
        h = h_outlet * np.ones(boundary_grid.num_cells)
        h[inlet_idx] = h_inlet
        return h
    
    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 623.0   #[K]
        t_outlet = 423.0  #[K]
        T = t_outlet * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T
    
    def bc_values_overall_fraction(
        self, 
        component: ppc.Component, 
        boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        z_init = 1.0e-100
        z_inlet = 1.0e-100
        if component.name == "H2O":
            z_H2O = (1 - z_init) * np.ones(boundary_grid.num_cells)
            z_H2O[inlet_idx] = 1 - z_inlet
            return z_H2O
        else:
            z_NaCl = z_init * np.ones(boundary_grid.num_cells)
            z_NaCl[inlet_idx] = z_inlet
            return z_NaCl
   
class InitialConditions(InitialConditionsCF):
    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial pressure distribution that varies linearly from
           the inlet to the outlet of the domain.
        """
        p_inlet = 50.0e6
        p_outlet = 25.0e6
        domain_length = 2000.0 #in m
        cell_centers_x = sd.cell_centers[0]
        pressure_gradient = (p_outlet - p_inlet) / domain_length
        p_init = p_inlet + pressure_gradient * cell_centers_x
        return p_init
    
    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h_init = 6.45e5
        return np.ones(sd.num_cells) * h_init
    
    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423 #[K]
        return np.ones(sd.num_cells) * t_init
    
    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)
    
    
class ModelEquations(
    PrimaryEquationsCF,
    SecondaryEquations,
):
    """Collecting primary flow and transport equations, and secondary equations
    which provide substitutions for independent saturations and partial fractions.
    """

    def set_equations(self):
        """Call to the equation. Parent classes don't use super(). User must provide
        proper order resultion.

        I don't know why, but the other models are doing it this way was well.
        Maybe it has something to do with the sparsity pattern.

        """
        # Flow and transport in MD setting
        PrimaryEquationsCF.set_equations(self)
        # local elimination of dangling secondary variables
        SecondaryEquations.set_equations(self)

class DriesnerWaterFlowModel(
    ModelGeometry,
    FluidMixture,
    InitialConditions,
    BoundaryConditions,
    ModelEquations,
    CFModelMixin,
):
    def relative_permeability(
        self, 
        saturation: pp.ad.Operator
    ) -> pp.ad.Operator:
        return saturation

    @property
    def vtk_sampler(self):
        return self._vtk_sampler

    @vtk_sampler.setter
    def vtk_sampler(self, vtk_sampler):
        self._vtk_sampler = vtk_sampler