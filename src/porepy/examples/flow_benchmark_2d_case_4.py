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

from typing import Literal

import numpy as np

import porepy as pp
from porepy.examples.flow_benchmark_2d_case_1 import FractureSolidConstants
from porepy.models.constitutive_laws import DimensionDependentPermeability

solid_constants = FractureSolidConstants(
    residual_aperture=1e-2,  # m
    permeability=1e-14,  # m^2
    normal_permeability=1e-8,  # m^2
    fracture_permeability=1e-8,  # m^2
)


class Geometry(pp.PorePyModel):
    """Geometry specification."""

    def set_fractures(self) -> None:
        """Setting a fracture list from the fracture set library."""
        self._fractures = pp.applications.md_grids.fracture_sets.benchmark_2d_case_4()

    def set_domain(self) -> None:
        """Domain of the problem."""
        x_extent = self.units.convert_units(700, "m")
        y_extent = self.units.convert_units(600, "m")
        self._domain = pp.Domain(
            {"xmin": 0, "xmax": x_extent, "ymin": 0, "ymax": y_extent}
        )

    def grid_type(self) -> Literal["simplex"]:
        """Set a simplex grid, which is the only grid type that can represent this
        fracture geometry.

        Returns:
            str: Grid type.

        """
        return "simplex"

    def meshing_arguments(self) -> dict[str, float]:
        """Meshing arguments for mixed-dimensional grid generation.

        Set a default cell size of 10 m. This will give a grid of about 23K cells in the
        2d domain.

        Note that due to complexities of this fracture network, it is not possible to
        get Gmsh to create a grid with less than about 12K cells in the 2d domain. This
        can be achieved by setting the cell size to 30 m (possibly also lower). Coarser
        grids may be possible, but this will require interaction with Gmsh on a more
        detailed lever than what is available through PorePy.

        Returns:
            Meshing arguments compatible with
            :meth:`~porepy.grids.mdg_generation.create_mdg`.

        """
        # Default value of 10, scaled by the length unit.
        cell_size = self.units.convert_units(10, "m")
        default_meshing_args: dict[str, float] = {"cell_size": cell_size}
        # If meshing arguments are provided in the params, they should already be scaled
        # by the length unit.
        return self.params.get("meshing_arguments", default_meshing_args)


class BoundaryConditions(pp.PorePyModel):
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
        values = np.zeros(boundary_grid.num_cells)
        return values


class Permeability(DimensionDependentPermeability):
    """Tangential permeability specification for Case 1 of the 2D flow benchmark.

    The normal permeability is handled by the `SolidConstants`' `normal_permeability`
    parameter.

    """

    solid: FractureSolidConstants

    def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of fractures.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        size = sum([sd.num_cells for sd in subdomains])
        permeability = pp.wrap_as_dense_ad_array(
            self.solid.fracture_permeability, size, name="fracture permeability"
        )
        return self.isotropic_second_order_tensor(subdomains, permeability)


# Ignore type errors inherent to the ``SinglePhaseFlow`` class.
class FlowBenchmark2dCase4Model(  # type: ignore[misc]
    Geometry,
    BoundaryConditions,
    Permeability,
    pp.SinglePhaseFlow,
):
    """Mixer class for case 4 from the 2d flow benchmark."""
