"""
This module contains the implementation of Case 3 from the 3D flow benchmark [1].

Note:
    The class `FlowBenchmark3dCase3Model` admits the parameter keyword
    `refinement_level`, which can take values 0, 1, 2, 3, to control the mesh refinement
    level. Level `0` contains approximately 30K three-dimensional cells, level `1`
    contains 140K three-dimensional cells, level `2` contains 350K three-dimensional
    cells, and level `3` contains 500K three-dimensional cells.

References:
    [1] Berre, I., Boon, W. M., Flemisch, B., Fumagalli, A., Gläser, D., Keilegavlen,
        E., ... & Zulian, P. (2021). Verification benchmarks for single-phase flow in
        three-dimensional fractured porous media. Advances in Water Resources, 147,
        103759.

"""

from typing import cast

import numpy as np

import porepy as pp
from porepy.applications.discretizations.flux_discretization import FluxDiscretization
from porepy.applications.md_grids.mdg_library import benchmark_3d_case_3
from porepy.examples.flow_benchmark_2d_case_1 import (
    FractureSolidConstants,
    Permeability,
)
from porepy.fracs.fracture_network_3d import FractureNetwork3d

solid_constants = FractureSolidConstants(
    residual_aperture=1e-2,
    normal_permeability=1e4,
    fracture_permeability=1e4,
)


class Geometry(pp.ModelGeometry):
    """Define Geometry as specified in Section 5.3 of the benchmark study [1]."""

    params: dict
    """User-defined model parameters."""

    def set_geometry(self) -> None:
        """Create mixed-dimensional grid and fracture network."""

        # Create mixed-dimensional grid and fracture network.
        self.mdg, self.fracture_network = benchmark_3d_case_3(
            refinement_level=self.params.get("refinement_level", 0)
        )
        self.nd: int = self.mdg.dim_max()

        # Obtain domain and fracture list directly from the fracture network.
        self._domain = cast(pp.Domain, self.fracture_network.domain)
        self._fractures = self.fracture_network.fractures

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections.
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fracture + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures.
            self.well_network.mesh(self.mdg)


class IntersectionPermeability(Permeability):
    def intersection_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Constant intersection permeability.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the permeability.

        """
        size = sum(sd.num_cells for sd in subdomains)
        # Use `fracture_permeability` as intersection permeability under the assumption
        # that they are equal. This is valid in the current benchmark case.
        permeability = pp.wrap_as_dense_ad_array(
            self.solid.fracture_permeability, size, name="intersection permeability"
        )
        return self.isotropic_second_order_tensor(subdomains, permeability)


class BoundaryConditions(pp.BoundaryConditionMixin):
    """Define inlet and outlet boundary conditions as specified by the benchmark."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign Dirichlet to the top and bottom  part of the north (y=y_max)
        boundary."""
        # Retrieve boundary sides.
        bounds = self.domain_boundary_sides(sd)
        # Get Dirichlet faces.
        dir_faces = np.zeros(sd.num_faces, dtype=bool)
        north_top_dir_cells = sd.face_centers[2][bounds.north] > (2 / 3)
        north_bottom_dir_faces = sd.face_centers[2][bounds.north] < (1 / 3)
        dir_faces[bounds.north] = north_top_dir_cells + north_bottom_dir_faces
        # Assign boundary conditions, the rest are Neumann by default.
        bc = pp.BoundaryCondition(sd, dir_faces, "dir")
        return bc

    def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Assign non-zero Darcy flux to the middle south (y=y_min) boundary."""
        # Retrieve boundary sides and cell centers.
        bounds = self.domain_boundary_sides(boundary_grid)
        cc = boundary_grid.cell_centers
        # Get inlet faces.
        inlet_faces = np.zeros(boundary_grid.num_cells, dtype=bool)
        inlet_faces[bounds.south] = (cc[2][bounds.south] < (2 / 3)) & (
            cc[2][bounds.south] > (1 / 3)
        )
        # Assign unitary flow. Negative since fluid is entering into the domain.
        val = self.units.convert_units(-1, "m * s^-1")
        values = np.zeros(boundary_grid.num_cells)
        values[inlet_faces] = val * boundary_grid.cell_volumes[inlet_faces]
        return values


class FlowBenchmark3dCase3Model(  # type:ignore[misc]
    FluxDiscretization,
    Geometry,
    IntersectionPermeability,
    BoundaryConditions,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """Mixer class for case 3 from the 3d flow benchmark."""
