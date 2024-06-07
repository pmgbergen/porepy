import LinearTracerConstitutiveDescription
import numpy as np
from Geometries import SimpleGeometry as ModelGeometry

import porepy as pp
import porepy.composite as ppc
from porepy.models.compositional_flow import (
    BoundaryConditionsCF,
    CFModelMixin,
    InitialConditionsCF,
    PrimaryEquationsCF,
)


class BoundaryConditions(BoundaryConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx, _ = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, facet_idx, "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")

    def bc_type_advective_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        p_inlet = 15.0e6
        p_outlet = 10.0e6
        p = p_inlet * np.ones(boundary_grid.num_cells)
        p[inlet_idx] = p_inlet
        p[outlet_idx] = p_outlet
        return p

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        h_init = 2.0e6
        h_inlet = 2.0e6
        h = h_init * np.ones(boundary_grid.num_cells)
        h[inlet_idx] = h_inlet
        return h

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        z_init = 0.15
        z_inlet = 1.0
        if component.name == "H2O":
            z_H2O = (1 - z_init) * np.ones(boundary_grid.num_cells)
            z_H2O[inlet_idx] = 1 - z_inlet
            return z_H2O
        else:
            z_NaCl = z_init * np.ones(boundary_grid.num_cells)
            z_NaCl[inlet_idx] = z_inlet
            return z_NaCl

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        h = self.bc_values_enthalpy(boundary_grid)
        factor = 630.0 / 2.0e6
        T = factor * h
        return T


class InitialConditions(InitialConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_init = 15.0e6
        p_outlet = 15.0e6
        xc = sd.cell_centers.T

        def p_D(xv):
            p_val = (1 - xv[0] / 2) * p_init + (xv[0] / 2) * p_outlet
            return p_val

        p_vals = np.fromiter(map(p_D, xc), dtype=float)
        return p_vals

    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h = 2.0e6
        return np.ones(sd.num_cells) * h

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.15
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        h = self.initial_enthalpy(sd)
        factor = 630.0 / 2.0e6
        T = factor * h
        return T


class SecondaryEquations(LinearTracerConstitutiveDescription.SecondaryEquations):
    pass


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


class LinearTracerFlowModel(
    ModelGeometry,
    LinearTracerConstitutiveDescription.FluidMixture,
    InitialConditions,
    BoundaryConditions,
    ModelEquations,
    CFModelMixin,
):

    def relative_permeability(self, saturation: pp.ad.Operator) -> pp.ad.Operator:
        return saturation
