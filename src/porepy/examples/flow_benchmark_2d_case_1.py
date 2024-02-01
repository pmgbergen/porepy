"""
This module contains the implementation of Case 1 from the 2D flow benchmark [1].

Note:
    We provide the two variants of this benchmark, i.e., Case 1a (conductiv fractures)
    and Case 1b (blocking fractures). The former is provided by the class
    `FlowBenchmark2dCase1aModel` whereas the latter by the class
    `FlowBenchmark2dCase1bModel`.

References:
    - [1] Flemisch, B., Berre, I., Boon, W., Fumagalli, A., Schwenck, N., Scotti, A.,
      ... & Tatomir, A. (2018). Benchmarks for single-phase flow in fractured porous
      media. Advances in Water Resources, 111, 239-258.

"""
import numpy as np

import porepy as pp
from porepy.applications.discretizations.flux_discretization import FluxDiscretization
from porepy.models.constitutive_laws import DimensionDependentPermeability

solid_constants = pp.SolidConstants({"residual_aperture": 1e-4})


class Geometry:
    """Geometry specification for Case 1 of the 2D flow benchmark."""

    def set_fractures(self) -> None:
        """Setting a fracture list from the fracture set library."""
        self._fractures = pp.applications.md_grids.fracture_sets.benchmark_2d_case_1()


class BoundaryConditions:
    """Boundary conditions for Case 1 of the 2D flow benchmark.

    Inflow on west (left) and prescribed pressure on east (right).

    """

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure value of 1 Pa on east side."""
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        values[bounds.west] = self.fluid.convert_units(1, "Pa")
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

    def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of fractures.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        size = sum([sd.num_cells for sd in subdomains])
        val = self.params.get("fracture_permeability", 1e-4)
        permeability = pp.wrap_as_dense_ad_array(
            self.solid.convert_units(val, "m^2"), size
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
