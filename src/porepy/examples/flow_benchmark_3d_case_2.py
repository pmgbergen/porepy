"""
This module contains the implementation of Case 2 from the 3D flow benchmark [1].

Note:
    The class `FlowBenchmark3dCase2Model` admits the parameter keyword
    `refinement_level`, which can take values 0, 1, 2, to control the mesh refinement
    level. Level `0` contains approximately 500 three-dimensional cells, level `1`
    contains 4K three-dimensional cells, and level `2` contains 32K three-dimensional
    cells.

    To set up case (a) with conductive fractures, use `solid_constants_conductive`.
    To set up case (b) with blocking fractures, use `solid_constants_blocking`.

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
from porepy.applications.md_grids.mdg_library import benchmark_3d_case_2
from porepy.examples.flow_benchmark_2d_case_1 import (
    FractureSolidConstants,
    Permeability,
)
from porepy.fracs.fracture_network_3d import FractureNetwork3d

solid_constants_conductive = FractureSolidConstants(
    residual_aperture=1e-4,
    normal_permeability=1e4,
    fracture_permeability=1e4,
)

solid_constants_blocking = FractureSolidConstants(
    residual_aperture=1e-4,
    normal_permeability=1e-4,
    fracture_permeability=1e-4,
)

class Geometry(pp.PorePyModel):
    """Define the Geometry as specified in Section 5.3 of [1]. """

    def set_geometry(self) -> None:
        """Create mixed-dimensional grid adn fracture network."""

        # Create mixed-dimensional grid and fracture network.
        self.mdg, self.fracture_network = benchmark_3d_case_2(
            refinement_level=self.params.get("refinement_level", 0)
        )
        self.nd: int = self.mdg.dim_max()

        # Obtain domain and fracture list directly from the fracture network.
        self._domain = cast(pp.Domain, self.fracture_network.domain)
        self._fractures = self.fracture_network.fractures

        # Create projection between local and global coordinates fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        # Create well network.
        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections.
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fractures + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures
            self.well_network.mesh(self.mdg)


class PermeabilitySpecification(Permeability):
    """Set up permeability values in the mixed-dimensional grid."""

    def _low_perm_zones(self, sd: pp.Grid) -> np.ndarray:
        """Helper mask corresponding to low permeability zones in the matrix."""

        # Safeguard against wrong dimensionality.
        if sd.dim < 3:
            raise ValueError('_low_perm_zones is only meaningful for 3d.')

        cc = sd.cell_centers

        zone_0 = np.logical_and(cc[0, :] > 0.5,  cc[1, :] < 0.5)
        zone_1 = np.logical_and.reduce(
            tuple(
                [
                    cc[0, :] < 0.75,
                    cc[1, :] > 0.5,
                    cc[1, :] < 0.75,
                    cc[2, :] > 0.5,
                ]
            )
        )
        zone_2 = np.logical_and.reduce(
            tuple(
                [
                    cc[0, :] > 0.625,
                    cc[0, :] < 0.75,
                    cc[1, :] > 0.5,
                    cc[1, :] < 0.625,
                    cc[2, :] > 0.5,
                    cc[2, :] < 0.75,
                ]
            )
        )

        return np.logical_or.reduce(tuple([zone_0, zone_1, zone_2]))

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Heterogeneous matrix permeability. See [1] for the details."""

        # Assign permeability values
        vals = []
        for sd in subdomains:
            kxx = np.ones(sd.num_cells)
            if sd.dim == 3:
                # Set unit permeability except in the low permeability zones.
                kxx[self._low_perm_zones(sd)] = 1e-1
                vals.append(kxx)
            else:
                vals.append(kxx)  # This is just a placeholder, it will not be used.

        # This seems to be needed for setting up the model
        if len(subdomains) > 0:
            permeability = pp.wrap_as_dense_ad_array(np.hstack(vals))
        else:
            permeability = self.solid.permeability

        return self.isotropic_second_order_tensor(subdomains, permeability)

    def intersection_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Constant intersection permeability.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the permeability

        """
        size = sum(sd.num_cells for sd in subdomains)
        # Use `fracture_permeability` as intersection permeability under the assumption
        # that they are equal. This is valid in the current benchmark case.
        permeability = pp.wrap_as_dense_ad_array(
            self.solid.fracture_permeability, size, name="intersection_permeability"
        )
        return self.isotropic_second_order_tensor(subdomains, permeability)


class BoundaryConditions(pp.PorePyModel):
    """Define inlet and oulet boundary conditions as specified by the benchmark."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign Dirichlet boundary condition at outlet boundary."""
        b_faces = sd.tags['domain_boundary_faces'].nonzero()

        if b_faces != 0:
            b_faces_centers = sd.face_centers[:, b_faces]
            b_outflow = np.logical_and.reduce(
                tuple(b_faces_centers[i, :] > 0.875 + 1e-8 for i in range(3))
            )
            # Outlet faces correspond to Dirichlet boundary conditions. The rest are set
            # as Neumann by default.
            dir_faces = b_faces[0][b_outflow[0]]
            bc = pp.BoundaryCondition(sd, dir_faces, "dir")

        else:
            bc = pp.BoundaryCondition(sd)

        return bc

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assign unitary Darcy flux at the inlet boundary.

            \partial\Omega_inlet = \{x \in \partial\Omega : x_1, x_2, x_3 < 0.25 \}

        """
        cc = bg.cell_centers
        nc = bg.num_cells
        volumes = bg.cell_volumes

        # Retrieve inlet faces.
        inlet_faces = np.logical_and.reduce(
            tuple(cc[i, :] < 0.25 + 1e-8 for i in range(3))
        )
        val = self.units.convert_units(-1, "m * s^-1")
        values = np.zeros(nc)
        values[inlet_faces] = val * volumes[inlet_faces]

        return values

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assign unitary pressure at the outlet boundary:

            \partial\Omega_outlet = \{x \in \partial\Omega : x_1, x_2, x_3 > 0.875\}

        """
        cc = bg.cell_centers
        nc = bg.num_cells

        # Retrieve outlet faces
        outlet_faces = np.logical_and.reduce(
            tuple(cc[i, :] > 0.875 + 1e-8 for i in range(3))
        )
        val = self.units.convert_units(1, "Pa")
        values = np.zeros(nc)
        values[outlet_faces] = val

        return values


class FlowBenchmark3dCase2Model(  # type:ignore[misc]
    FluxDiscretization,
    Geometry,
    PermeabilitySpecification,
    BoundaryConditions,
    pp.SinglePhaseFlow
):
    """Mixer class for Case 2: Regular Network from the 3D flow benchmark."""
