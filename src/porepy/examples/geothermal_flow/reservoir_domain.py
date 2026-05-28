from typing import Literal, Sequence
import numpy as np

import porepy as pp
from porepy.fracs.wells_3d import _add_interface
import scipy.sparse as sps


class DisconnectedFracturedDomain2D(pp.PorePyModel):
    """A class to represent a simple 2D geometry for a simulation domain."""

    # Domain dimensions
    _domain_x_length: float = 100.0
    _domain_y_length: float = 30.0

    # Producer well coordinates.
    prod_x_point: float = _domain_x_length - 15.0
    prod_y_point: float = _domain_y_length * 0.5

    # Injector well location
    _injection_points: list[np.ndarray] = [
        np.array([_domain_x_length - prod_x_point, prod_y_point])
    ]
    # Production well location
    _production_points: list[np.ndarray] = [np.array([prod_x_point, prod_y_point])]

    def set_domain(self) -> None:
        x_length_in_m = self.units.convert_units(self._domain_x_length, "m")
        y_length_in_m = self.units.convert_units(self._domain_y_length, "m")
        box: dict[str, pp.number] = {"xmax": x_length_in_m, "ymax": y_length_in_m}
        self._domain = pp.Domain(box)

    def set_fractures(self) -> None:
        frac1 = pp.LineFracture(np.array([[0.0, 12.0], [15.0, 15.0]]))
        frac2 = pp.LineFracture(
            np.array([[self._production_points[0][0] - 40, 90 - 30], [14.0, 14.0]])
        )
        frac3 = pp.LineFracture(
            np.array([[self._production_points[0][0] - 35, 90 - 30], [20.5, 8.0]])
        )
        frac4 = pp.LineFracture(
            np.array(
                [
                    [
                        self._production_points[0][0],
                        self._production_points[0][0] + 7.0,
                    ],
                    [
                        self._production_points[0][1],
                        self._production_points[0][1] - 5.0,
                    ],
                ]
            )
        )
        self._fractures = [frac1, frac2, frac3, frac4]

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.7, "m")  # was 0.7
        frac_cell_size = self.units.convert_units(1.0, "m")  # was 1.0
        mesh_args: dict[str, float] = {
            "cell_size": cell_size,
            "cell_size_fracture": frac_cell_size,
        }
        return mesh_args

    def set_geometry(self):
        """Create the injection and production wells."""

        super().set_geometry()
        for i, injection_point in enumerate(self._injection_points):
            self._add_well(injection_point, i, "injection")

        for i, production_point in enumerate(self._production_points):
            self._add_well(production_point, i, "production")

    def closest_face(self, grid: pp.Grid, point: np.ndarray) -> int:
        """Return index of closest face center to given point."""
        dists = np.linalg.norm(grid.face_centers - point.reshape(-1, 1), axis=0)
        return int(np.argmin(dists))

    def _add_well(
        self,
        point: np.ndarray,
        well_index: int,
        well_type: Literal["injection", "production"],
    ) -> None:
        """Insert single well as point grid and connect to matrix."""

        # Convert to 3D coordinates (for porepy PointGrid)
        p = np.zeros(3)
        p[:2] = point

        # Create 0D grid
        sd_0d = pp.PointGrid(p)
        sd_0d.tags[f"{well_type}_well"] = well_index
        sd_0d.compute_geometry()

        # This object must have been passed or prepared by a mixin
        self.mdg.add_subdomains(sd_0d)

        # Couple well to the matrix (0D <--> 2D coupling is allowed with wells in PorePy)
        matrix = self.mdg.subdomains(dim=self.domain.dim)[0]
        cell_matrix = matrix.closest_cell(sd_0d.cell_centers)
        cell_well = np.array([0], dtype=int)
        cell_cell_map = sps.coo_matrix(
            (np.ones(1, dtype=bool), (cell_well, cell_matrix)),
            shape=(sd_0d.num_cells, matrix.num_cells),
        )
        _add_interface(0, matrix, sd_0d, self.mdg, cell_cell_map)

    def point_line_distance(
        self, point: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> float:
        """Shortest distance between a point and a line segment in 2D or 3D."""
        line_vec = end - start
        p_vec = point - start
        line_len2 = np.dot(line_vec, line_vec)
        if line_len2 == 0:
            return np.linalg.norm(p_vec)  # start == end (degenerate segment)
        t = np.clip(np.dot(p_vec, line_vec) / line_len2, 0, 1)
        projection = start + t * line_vec
        return np.linalg.norm(point - projection)

    def _filter_wells(
        self,
        subdomains: Sequence[pp.Grid],
        well_type: Literal["production", "injection"],
    ) -> tuple[list[pp.Grid], list[pp.Grid]]:
        """Helper method to return the partitioning of subdomains into wells of defined
        ``well_type`` and other grids.

        Parameters:
            subdomains: A list of subdomains.
            well_type: Well type to filter out (injector or producer).

        Returns:
            A  2-tuple containing
            1. All 0D grids tagged as wells of type ``well_type``.
            2. All other grids found in ``subdomains``.

        """
        tag = f"{well_type}_well"
        wells = [sd for sd in subdomains if sd.dim == 0 and tag in sd.tags]
        other_sds = [sd for sd in subdomains if sd not in wells]
        return wells, other_sds


class ConnectedFracturedDomain2D(pp.PorePyModel):
    """A class to represent a simple 2D geometry for a simulation domain."""

    # Domain dimensions
    _domain_x_length: float = 100.0
    _domain_y_length: float = 30.0

    # Producer well coordinates.
    prod_x_point: float = _domain_x_length - 15.0
    prod_y_point: float = _domain_y_length * 0.5

    # Injector well location
    _injection_points: list[np.ndarray] = [
        np.array([_domain_x_length - prod_x_point, prod_y_point + 1])
    ]

    # Production well location
    _production_points: list[np.ndarray] = [np.array([prod_x_point, prod_y_point])]

    def set_domain(self) -> None:
        x_length_in_m = self.units.convert_units(self._domain_x_length, "m")
        y_length_in_m = self.units.convert_units(self._domain_y_length, "m")
        box: dict[str, pp.number] = {"xmax": x_length_in_m, "ymax": y_length_in_m}
        self._domain = pp.Domain(box)

    def set_fractures(self) -> None:
        frac1 = pp.LineFracture(np.array([[15.0, 25.0], [15.0 + 1, 10]]))
        frac2 = pp.LineFracture(np.array([[25.0, 35.0], [10.0, 15]]))
        frac3 = pp.LineFracture(np.array([[35.0, 45.0], [15.0, 10]]))
        frac4 = pp.LineFracture(np.array([[45.0, 55.0], [10.0, 15]]))
        frac5 = pp.LineFracture(np.array([[55.0, 65.0], [15.0, 10.0]]))
        frac6 = pp.LineFracture(np.array([[65.0, 75.0], [10.0, 15.0]]))
        frac7 = pp.LineFracture(np.array([[75.0, 85.0], [15.0, 10.0]]))
        frac7 = pp.LineFracture(np.array([[75.0, 80.0], [15.0, 10.0]]))
        frac8 = pp.LineFracture(np.array([[80.0, 85.0], [10.0, 15.0]]))
        frac9 = pp.LineFracture(np.array([[85.0, 85.0], [15.0, 18.0]]))
        self._fractures = [frac1, frac2, frac3, frac4, frac5, frac6, frac7, frac8]

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(1.0, "m")
        frac_cell_size = self.units.convert_units(1.0, "m")
        mesh_args: dict[str, float] = {
            "cell_size": cell_size,
            "cell_size_fracture": frac_cell_size,
        }
        return mesh_args

    def set_geometry(self):
        """Create the injection and production wells."""

        super().set_geometry()
        for i, injection_point in enumerate(self._injection_points):
            self._add_well(injection_point, i, "injection")

        for i, production_point in enumerate(self._production_points):
            self._add_well(production_point, i, "production")

    def closest_face(self, grid: pp.Grid, point: np.ndarray) -> int:
        """Return index of closest face center to given point."""
        dists = np.linalg.norm(grid.face_centers - point.reshape(-1, 1), axis=0)
        return int(np.argmin(dists))

    def _add_well(
        self,
        point: np.ndarray,
        well_index: int,
        well_type: Literal["injection", "production"],
    ) -> None:
        """Insert single well as point grid and connect to matrix."""

        # Convert to 3D coordinates (for porepy PointGrid)
        p = np.zeros(3)
        p[:2] = point

        # Create 0D grid
        sd_0d = pp.PointGrid(p)
        sd_0d.tags[f"{well_type}_well"] = well_index
        sd_0d.compute_geometry()

        # This object must have been passed or prepared by a mixin
        self.mdg.add_subdomains(sd_0d)

        # Couple well to the matrix (0D <--> 2D coupling is allowed with wells in PorePy)
        matrix = self.mdg.subdomains(dim=self.domain.dim)[0]
        cell_matrix = matrix.closest_cell(sd_0d.cell_centers)
        cell_well = np.array([0], dtype=int)
        cell_cell_map = sps.coo_matrix(
            (np.ones(1, dtype=bool), (cell_well, cell_matrix)),
            shape=(sd_0d.num_cells, matrix.num_cells),
        )
        _add_interface(0, matrix, sd_0d, self.mdg, cell_cell_map)

    def _filter_wells(
        self,
        subdomains: Sequence[pp.Grid],
        well_type: Literal["production", "injection"],
    ) -> tuple[list[pp.Grid], list[pp.Grid]]:
        """Helper method to return the partitioning of subdomains into wells of defined
        ``well_type`` and other grids.

        Parameters:
            subdomains: A list of subdomains.
            well_type: Well type to filter out (injector or producer).

        Returns:
            A  2-tuple containing
            1. All 0D grids tagged as wells of type ``well_type``.
            2. All other grids found in ``subdomains``.

        """
        tag = f"{well_type}_well"
        wells = [sd for sd in subdomains if sd.dim == 0 and tag in sd.tags]
        other_sds = [sd for sd in subdomains if sd not in wells]
        return wells, other_sds
