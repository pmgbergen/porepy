import numpy as np
from .geometry_description.geometry_market import SimpleGeometry as ModelGeometry
from .constitutive_description.BrineConstitutiveDescription import SecondaryEquations
from .constitutive_description.BrineConstitutiveDescription import FluidMixture

import porepy as pp
import porepy.compositional as ppc
from porepy.models.compositional_flow import (
    BoundaryConditionsCF,
    CFModelMixin,
    InitialConditionsCF,
    PrimaryEquationsCF,
)


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
        p_inlet = 20.0e6
        p_outlet = 15.0e6
        xc = boundary_grid.cell_centers.T
        l = 2.0

        def p_linear(xv):
            p_v = p_inlet * (1 - xv[0] / l) + p_outlet * (xv[0] / l)
            return p_v

        p = np.fromiter(map(p_linear, xc), dtype=float)
        return p
        p[inlet_idx] = p_inlet
        p[outlet_idx] = p_outlet
        return p

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        h_inlet = 1.5e6
        h_outlet = 2.2e6
        xc = boundary_grid.cell_centers.T
        l = 2.0

        def h_linear(xv):
            h_v = h_inlet * (1 - xv[0] / l) + h_outlet * (xv[0] / l)
            return h_v

        h = np.fromiter(map(h_linear, xc), dtype=float)
        h[inlet_idx] = h_inlet
        return h

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        z_init = 0.2
        z_inlet = 0.02
        if component.name == "H2O":
            z_H2O = (1 - z_init) * np.ones(boundary_grid.num_cells)
            z_H2O[inlet_idx] = 1 - z_inlet
            return z_H2O
        else:
            z_NaCl = z_init * np.ones(boundary_grid.num_cells)
            z_NaCl[inlet_idx] = z_inlet
            return z_NaCl

    # def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
    #     h = self.bc_values_enthalpy(boundary_grid)
    #     factor = 630.0 / 2.0e6
    #     T = factor * h
    #     return T


class InitialConditions(InitialConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 15.0e6
        p_outlet = 15.0e6
        xc = sd.cell_centers.T
        l = 2.0

        def p_linear(xv):
            p_v = p_inlet * (1 - xv[0] / l) + p_outlet * (xv[0] / l)
            return p_v

        p = np.fromiter(map(p_linear, xc), dtype=float)
        return p

    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h_inlet = 2.2e6
        h_outlet = 2.2e6
        xc = sd.cell_centers.T
        l = 2.0

        def h_linear(xv):
            h_v = h_inlet * (1 - xv[0] / l) + h_outlet * (xv[0] / l)
            return h_v

        h = np.fromiter(map(h_linear, xc), dtype=float)
        return h

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.2
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)

    # def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
    #     h = self.initial_enthalpy(sd)
    #     factor = 630.0 / 2.0e6
    #     T = factor * h
    #     return T

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


class DriesnerBrineFlowModel(
    ModelGeometry,
    FluidMixture,
    InitialConditions,
    BoundaryConditions,
    ModelEquations,
    CFModelMixin,
):
    def relative_permeability(self, saturation: pp.ad.Operator) -> pp.ad.Operator:
        return saturation

    @property
    def obl(self):
        return self._obl

    @obl.setter
    def obl(self, obl):
        self._obl = obl
