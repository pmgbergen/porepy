"""Contains the geometry for the cold injection examples.

Most notably, contains a 2D geometry with 0D points as injection and production wells.

Fractures are added on top of it.

Individual configurations are possible in ``config.py``.

"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.fracs.wells_3d import _add_interface

from .config import ModelConfig


def _set_random_seed(*args):
    s = 2025
    for a in args:
        s += int(a)
    np.random.seed(s)


class PointWells(ModelConfig):
    """Geometry adding point grids as wells after super-call in set_geometry."""

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmin": 0.0,
                "xmax": self.units.convert_units(self._DOMAIN_DIMENSIONS[0], "m"),
                "ymin": 0.0,
                "ymax": self.units.convert_units(self._DOMAIN_DIMENSIONS[1], "m"),
            }
        )

    def set_geometry(self):
        super().set_geometry()

        for i, injection_point in enumerate(self._INJECTION_POINTS):
            self._add_well(injection_point, i, "injection")

        for i, production_point in enumerate(self._PRODUCTION_POINTS):
            self._add_well(production_point, i, "production")

    def _add_well(
        self,
        point: np.ndarray,
        well_index: int,
        well_type: Literal["injection", "production"],
    ) -> None:
        """Helper method to construct a well in 2D as a PointGrid and add respective
        interface.

        Parameters:
            point: Point in space representing well.
            well_index: Assigned number for well of type ``well_type``.
            well_type: Label to add a tag to the point grid labelng as injector or
            producer.

        """
        matrix = self.mdg.subdomains(dim=self.nd)[0]
        assert isinstance(point, np.ndarray)
        p: np.ndarray
        if point.shape == (2,):
            p = np.zeros(3)
            p[:2] = point
        elif point.shape == (3,):
            p = point
        else:
            raise ValueError(
                f"Point for well {(well_type, well_index)} must be 1D array of length "
                + "2 or 3."
            )

        sd_0d = pp.PointGrid(self.units.convert_units(p, "m"))
        # Tag for processing of equations.
        sd_0d.tags[f"{well_type}_well"] = well_index
        sd_0d.compute_geometry()

        self.mdg.add_subdomains(sd_0d)

        # Motivated by wells_3d.py#L828
        cell_matrix = matrix.closest_cell(sd_0d.cell_centers)
        cell_well = np.array([0], dtype=int)
        cell_cell_map = sps.coo_matrix(
            (np.ones(1, dtype=bool), (cell_well, cell_matrix)),
            shape=(sd_0d.num_cells, matrix.num_cells),
        )

        _add_interface(0, matrix, sd_0d, self.mdg, cell_cell_map)


class RandomFracturesAndPointWells2D(PointWells):
    """2D matrix with point grids as injection and production points, and a random
    distribution of fractures.

    Alternative for the ``WellNetwork3d`` in 2d.

    """

    def set_fractures(self) -> None:
        x_min, y_min = 0.0, 0.0
        x_max = self._DOMAIN_DIMENSIONS[0]
        y_max = self._DOMAIN_DIMENSIONS[1]
        domain_width = x_max - x_min
        domain_height = y_max - y_min
        min_length = domain_height
        max_length = domain_width * 0.8

        fractures = []
        _set_random_seed(self._NUM_FRACTURES)
        for _ in range(self._NUM_FRACTURES):
            # Random center within bounds
            x_center = np.random.uniform(
                x_min + 0.1 * domain_width, x_max - 0.1 * domain_width
            )
            y_center = np.random.uniform(
                y_min + 0.1 * domain_height, y_max - 0.1 * domain_height
            )

            # Random angle and length
            theta = np.random.uniform(-np.pi / 3, np.pi / 3)
            length = np.random.uniform(min_length, max_length)

            dx = 0.5 * length * np.cos(theta)
            dy = 0.5 * length * np.sin(theta)

            x0, x1 = x_center - dx, x_center + dx
            y0, y1 = y_center - dy, y_center + dy

            coords = np.array([[x0, x1], [y0, y1]])
            fractures.append(pp.LineFracture(coords))

        self._fractures = fractures


class HorizontalFractureAndPointWells2D(PointWells):
    """Introduces a single horizontal fracture in the middel of the domain,
    with length equal to half the domain length in x-direction."""

    def set_fractures(self) -> None:
        x_min, y_min = 0.0, 0.0
        x_max = self._DOMAIN_DIMENSIONS[0]
        y_max = self._DOMAIN_DIMENSIONS[1]
        domain_width = x_max - x_min
        domain_height = y_max - y_min

        x_center = x_min + domain_width * 0.5
        y_center = y_min + domain_height * 0.5

        frac_width = domain_width * 0.6

        self._fractures = [
            pp.LineFracture(
                np.array(
                    [
                        [x_center - 0.5 * frac_width, x_center + 0.5 * frac_width],
                        [y_center, y_center],
                    ]
                )
            )
        ]


class GeometryBenchmark3d_case4(pp.PorePyModel):
    """Define Geometry as specified in Section 5.3 of the benchmark study [1]."""

    def set_geometry(self) -> None:
        """Create mixed-dimensional grid and fracture network."""

        # Create mixed-dimensional grid and fracture network.
        self.mdg, self.fracture_network = benchmark_3d_case_4()
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


def benchmark_3d_case_4() -> tuple[pp.MixedDimensionalGrid, FractureNetwork3d]:
    """
    Create a mixed-dimensional grid for the geometry of case 4 from [1].

    Note:
        The mixed-dimensional grid is created by reading a `geo` file, so there is no
        direct way of prescribing meshing arguments.

    Reference:
        [1] Berre, I., Boon, W. M., Flemisch, B., Fumagalli, A., Gläser, D.,
        Keilegavlen, E., ... & Zulian, P. (2021). Verification benchmarks for
        single-phase flow in three-dimensional fractured porous media. Advances in Water
        Resources, 147, 103759.

    Returns:
        Tuple containing a:

            :obj:`~pp.MixedDimensionalGrid`:
                The grid for the specified refinement level.

            :obj:`~pp.FractureNetwork3d`:
                The fracture network.

    """
    # Get directory pointing to the `geo` file
    abs_path = Path(__file__)
    benchmark_path = abs_path.parent / "gmsh_file_library/benchmark_3d_case_4"
    full_path = benchmark_path / "gmsh_frac_file.geo"
    # Set file permissions. This turned out to be important for GH actions.
    full_path.chmod(777)

    # Create mixed-dimensional grid
    mdg = pp.fracture_importer.dfm_from_gmsh(full_path, dim=3)

    # Also import fracture network
    fracture_network_path = benchmark_path / "fracture_network.csv"
    # Set file permissions. This turned out to be important for GH actions.
    fracture_network_path.chmod(777)

    network = pp.fracture_importer.network_from_csv(
        fracture_network_path, check_convexity=False
    )

    return mdg, cast(FractureNetwork3d, network)
