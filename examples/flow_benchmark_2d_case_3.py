"""
This module contains the implementation of Case 3 from the 2D flow benchmark [1].

Note:
    We provide the two variants of this benchmark, i.e., Case 3a (top-to-bottom flow)
    and Case 3b (left-to-right flow). The former is provided by the class
    `FlowBenchmark2dCase3aModel` whereas the latter by the class
    `FlowBenchmark2dCase3bModel`.

References:
    - [1] Flemisch, B., Berre, I., Boon, W., Fumagalli, A., Schwenck, N., Scotti, A.,
      ... & Tatomir, A. (2018). Benchmarks for single-phase flow in fractured porous
      media. Advances in Water Resources, 111, 239-258.

"""
from typing import Callable

import numpy as np

import porepy as pp
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMassDirNorthSouth,
    BoundaryConditionsMassDirWestEast,
)
from porepy.applications.discretizations.flux_discretization import FluxDiscretization
from porepy.applications.md_grids.fracture_sets import benchmark_2d_case_3
from porepy.models.constitutive_laws import DimensionDependentPermeability


solid_constants = pp.SolidConstants({"residual_aperture": 1e-4})


class FlowBenchmark2dCase3Geometry:
    """Geometry specification."""

    def set_fractures(self) -> None:
        """Setting a fracture list from the fracture set library."""
        self._fractures = benchmark_2d_case_3()


class FlowBenchmark2dCase3aBoundaryConditions(BoundaryConditionsMassDirNorthSouth):
    """Boundary conditions for Case 3a, flow from top to bottom."""

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure value of 4 on top and 1 on bottom side."""
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        values[bounds.north] = self.fluid.convert_units(4, "Pa")
        values[bounds.south] = self.fluid.convert_units(1, "Pa")
        return values


class FlowBenchmark2dCase3bBoundaryConditions(BoundaryConditionsMassDirWestEast):
    """Boundary conditions for Case 3b, flow from left to right."""

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure value of 4 on left/west and 1 on right/east side."""
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        values[bounds.west] = self.fluid.convert_units(4, "Pa")
        values[bounds.east] = self.fluid.convert_units(1, "Pa")
        return values


class FlowBenchmark2dCase3Permeability(DimensionDependentPermeability):
    """Tangential and normal permeability specification."""

    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """

    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """Map from subdomains to the adjacent interfaces. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """

    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """

    @property
    def fracture_permeabilities(self) -> np.ndarray:
        """Permeability of the fractures.

        Ordering corresponds to definition of fractures in the geometry.

        """
        return np.array([1, 1, 1, 1e-8, 1e-8, 1, 1, 1, 1, 1]) * 1e4

    def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of fractures.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        if len(subdomains) == 0:
            return pp.wrap_as_dense_ad_array(1, size=0)
        permeabilities = []
        for sd in subdomains:
            permeabilities.append(
                self.solid.convert_units(
                    self.fracture_permeabilities[sd.frac_num] * np.ones(sd.num_cells),
                    "m^2",
                )
            )
        permeability = pp.wrap_as_dense_ad_array(np.concatenate(permeabilities))
        return self.isotropic_second_order_tensor(subdomains, permeability)

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Permeability of intersections.

        Parameters:
            interfaces: List of interfaces.

        Returns:
            Cell-wise permeability operator.

        """
        if len(interfaces) == 0:
            return pp.wrap_as_dense_ad_array(1, size=0)
        permeabilities = []
        for intf in interfaces:
            # Get hold of the fracture subdomain.
            sd_high, sd_low = self.mdg.interface_to_subdomain_pair(intf)
            if intf.dim == 1:
                # The normal permeability equals the fracture permeability.
                val = self.fracture_permeabilities[sd_low.frac_num]
            else:
                # Get the fractures intersecting the interface.
                interfaces_lower = self.subdomains_to_interfaces([sd_low], [1])
                # Get the higher-dimensional neighbors.
                parent_subdomains = self.interfaces_to_subdomains(interfaces_lower)
                # Only consider the higher-dimensional neighbors, i.e. the fractures.
                fracture_permeabilities = [
                    self.fracture_permeabilities[sd.frac_num]
                    for sd in parent_subdomains
                    if sd.dim == intf.dim + 1
                ]
                unique_fracture_permeabilities = np.unique(fracture_permeabilities)
                val = unique_fracture_permeabilities.size / np.sum(
                    1 / unique_fracture_permeabilities
                )
            permeabilities.append(
                self.solid.convert_units(val * np.ones(intf.num_cells), "m^2")
            )
        return pp.wrap_as_dense_ad_array(
            np.hstack(permeabilities), name="normal_permeability"
        )


class FlowBenchmark2dCase3aModel(  # type:ignore[misc]
    FluxDiscretization,
    FlowBenchmark2dCase3Geometry,
    FlowBenchmark2dCase3Permeability,
    FlowBenchmark2dCase3aBoundaryConditions,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """Mixer class for case 3a (top-to-bottom flow) from the 2d flow benchmark."""


class FlowBenchmark2dCase3bModel(  # type:ignore[misc]
    FluxDiscretization,
    FlowBenchmark2dCase3Geometry,
    FlowBenchmark2dCase3Permeability,
    FlowBenchmark2dCase3bBoundaryConditions,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """Mixer class for case 3b (left-to-right flow) from the 2d flow benchmark."""
