import porepy as pp
import numpy as np
import porepy.composite as ppc
import TracerConstitutiveDescription
from Geometries import Benchmark3DC3 as ModelGeometry
from porepy.models.compositional_flow import (
    BoundaryConditionsCF,
    CFModelMixin,
    InitialConditionsCF,
    PrimaryEquationsCF,
)

class BoundaryConditions(BoundaryConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.west | sides.east, "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.west | sides.east, "dir")

    def bc_type_advective_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.all_bf, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        p_inlet = 15.0e6
        p_outlet = 10.0e6
        xcs = boundary_grid.cell_centers.T
        l = 10.0
        def p_D(xc):
            x, y, z = xc
            return p_inlet * (1 - x / l) + p_outlet * (x / l)

        p_D_iter = map(p_D, xcs)
        p_D_vals = np.fromiter(p_D_iter, dtype=float)
        return p_D_vals

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        h_init = 2.5e6
        h_inlet = 2.5e6
        h = h_init * np.ones(boundary_grid.num_cells)
        h[sides.west] = h_inlet
        return h

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        z_init = 0.1
        z_inlet = 1.0
        if component.name == "H2O":
            z_H2O = (1 - z_init) * np.ones(boundary_grid.num_cells)
            z_H2O[sides.west] = 1 - z_inlet
            return z_H2O
        else:
            z_NaCl = z_init * np.ones(boundary_grid.num_cells)
            z_NaCl[sides.west] = z_inlet
            return z_NaCl

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        h = self.bc_values_enthalpy(boundary_grid)
        factor = 630.0 / 2.5e6
        T = factor * h
        return T


class InitialConditions(InitialConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def intial_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 15.0e6
        p_outlet = 10.0e6
        xcs = sd.cell_centers.T
        l = 10.0
        def p_D(xc):
            x, y, z = xc
            return p_inlet * (1 - x / l) + p_outlet * (x / l)

        p_D_iter = map(p_D, xcs)
        p_D_vals = np.fromiter(p_D_iter, dtype=float)
        return p_D_vals

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        h = self.initial_enthalpy(sd)
        factor = 630.0 / 2.5e6
        T = factor * h
        return T

    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h = 2.5e6
        return np.ones(sd.num_cells) * h

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.1
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class SecondaryEquations(TracerConstitutiveDescription.SecondaryEquations):
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


class TracerFlowModel(
    ModelGeometry,
    TracerConstitutiveDescription.FluidMixture,
    InitialConditions,
    BoundaryConditions,
    ModelEquations,
    CFModelMixin,
):
    pass

