"""
This module contains the implementation of Case 4 from the 2D flow benchmark [1].

The setup is composed of 64 fractures grouped in 13 different connected networks,
ranging from isolated fractures up to tens of fractures each.

Note:
    At the current stage, the setup is meant only for performance profiling and does not
    fully match the paper.

References:
    - [1] Flemisch, B., Berre, I., Boon, W., Fumagalli, A., Schwenck, N., Scotti, A.,
      ... & Tatomir, A. (2018). Benchmarks for single-phase flow in fractured porous
      media. Advances in Water Resources, 111, 239-258.

"""

import numpy as np

import porepy as pp
from porepy.models.protocol import PorePyModel

solid_constants = pp.SolidConstants(residual_aperture=1e-4)


class Geometry(PorePyModel):
    """Geometry specification."""

    def set_fractures(self) -> None:
        """Setting a fracture list from the fracture set library."""
        self._fractures = pp.applications.md_grids.fracture_sets.benchmark_2d_case_4()

    @property
    def domain(self) -> pp.Domain:
        """Domain of the problem."""
        # return pp.Domain({'xmax': 700, 'ymax': 600})
        return pp.Domain({"xmax": 50, "ymax": 50})


class BoundaryConditions(PorePyModel):
    """Boundary conditions for Case 4 of the 2D flow benchmark.

    Inflow on west (left) and prescribed pressure on east (right).

    """

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure value of one atmosphere (101325 Pa) on west side."""
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        values[bounds.west] = self.units.convert_units(101325, "Pa")
        return values

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign Dirichlet to the east and west boundary. The rest are Neumann by
        default."""
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.east + bounds.west, "dir")
        return bc

    def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Inflow on the west boundary.

        Per PorePy convention, the sign is negative for inflow and the value is
        integrated over the boundary cell volumes. Since the inflow boundary contains a
        fracture, the latter includes the fracture specific volume.

        Parameters:
            boundary_grid: Boundary grid.

        Returns:
            Boundary values.

        """
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        return values


# Ignore type errors inherent to the ``SinglePhaseFlow`` class.
class FlowBenchmark2dCase4Model(  # type: ignore[misc]
    Geometry,
    BoundaryConditions,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """Mixer class for case 4 from the 2d flow benchmark."""
