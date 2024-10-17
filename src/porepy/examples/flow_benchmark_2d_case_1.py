"""This module contains the implementation of Case 1 from the 2D flow benchmark [1].

We provide the two variants of this benchmark, i.e., Case 1a (conductiv fractures) and
Case 1b (blocking fractures). They are specified by passing the material constants
`solid_constants_conductive_fractures` and `solid_constants_blocking_fractures`,
respectively.

References:
    - [1] Flemisch, B., Berre, I., Boon, W., Fumagalli, A., Schwenck, N., Scotti, A.,
      ... & Tatomir, A. (2018). Benchmarks for single-phase flow in fractured porous
      media. Advances in Water Resources, 111, 239-258.

"""

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

import porepy as pp
from porepy.applications.discretizations.flux_discretization import FluxDiscretization
from porepy.models.constitutive_laws import DimensionDependentPermeability


@dataclass(frozen=True, kw_only=True)
class FractureSolidConstants(pp.SolidConstants):
    """Solid constants tailored to the current model."""

    fracture_permeability: pp.number = 1.0
    """[m^2]"""


solid_constants_conductive_fractures = FractureSolidConstants(
    residual_aperture=1e-4,
    fracture_permeability=1e4,
    normal_permeability=1e4,
)
solid_constants_blocking_fractures = FractureSolidConstants(
    residual_aperture=1e-4,
    fracture_permeability=1e-4,
    normal_permeability=1e-4,
)


class Geometry:
    """Geometry specification for Case 1 of the 2D flow benchmark."""

    def set_fractures(self) -> None:
        """Setting a fracture list from the fracture set library."""
        self._fractures = pp.applications.md_grids.fracture_sets.benchmark_2d_case_1()


class BoundaryConditions:
    """Boundary conditions for Case 1 of the 2D flow benchmark.

    Inflow on west (left) and prescribed pressure on east (right).

    """

    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]
    """Boundary sides of the domain. Defined by a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """
    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    specific_volume: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]]], pp.ad.Operator
    ]
    """Function that returns the specific volume of a subdomain or interface.

    Normally provided by a mixin of instance
    :class:`~porepy.models.constitutive_laws.DimensionReduction`.

    """
    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure value of 1 Pa on east side."""
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        values[bounds.east] = self.fluid.convert_units(1, "Pa")
        return values

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign Dirichlet to the east boundary. The rest are Neumann by default."""
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.east, "dir")
        return bc

    def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Inflow on the west boundary.

        Per PorePy convention, the sign is negative for inflow and the value is
        integrated over the boundary cell volumes. Since the inflow boundary contains
        a fracture, the latter includes the fracture specific volume.

        Parameters:
            boundary_grid: Boundary grid.

        Returns:
            Boundary values.

        """
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        # Inflow on the west boundary. Sign as per PorePy convention.
        val = self.fluid.convert_units(-1, "m * s^-1")
        # Integrate over the boundary cell volumes.
        values[bounds.west] = val * boundary_grid.cell_volumes[bounds.west]
        # Scale with specific volume.
        sd = boundary_grid.parent
        trace = np.abs(sd.cell_faces)
        specific_volumes = self.specific_volume([sd]).value(self.equation_system)
        values *= boundary_grid.projection() @ trace @ specific_volumes
        return values


class Permeability(DimensionDependentPermeability):
    """Tangential permeability specification for Case 1 of the 2D flow benchmark.

    The normal permeability is handled by the `SolidConstants`' `normal_permeability`
    parameter.

    """

    params: dict
    """Dictionary of parameters."""
    solid: FractureSolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.

    Tailored to the current model with the additional parameter
    `fracture_permeability`.

    """

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


class FlowBenchmark2dCase1Model(  # type:ignore[misc]
    FluxDiscretization,
    Geometry,
    Permeability,
    BoundaryConditions,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """Complete model class for case 1 from the 2d flow benchmark."""
