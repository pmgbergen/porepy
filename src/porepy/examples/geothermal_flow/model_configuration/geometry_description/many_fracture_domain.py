import numpy as np
import porepy as pp
from typing import Sequence, Literal
from abc import abstractmethod
from porepy.fracs.wells_3d import _add_interface
import scipy.sparse as sps


class Geometry(pp.PorePyModel):

    @abstractmethod
    def get_inlet_outlet_sides(
        self, sd: pp.Grid | pp.BoundaryGrid
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def harvest_sphere_members(xc, rc, x):
        dx = x - xc
        r = np.linalg.norm(dx, axis=1)
        return np.where(r < rc, True, False)


class RandomFracturesPointWellsDomain(Geometry):
    """
    Defines a 2D rectangular simulation domain with randomly distributed fractures
    and point-based injection and production wells.

    This geometry is designed for testing flow and transport simulations involving
    fractured domains. The fractures are represented as random 2D line segments,
    while wells are modeled as 0D point grids coupled to the 2D matrix domain.

    Notes
    -----
    - Fractures are generated with random orientation, length, and position within
      10% margins from domain boundaries.
    - Each well is represented by a 0D grid connected to the 2D matrix grid through
      an interface mapping.
    - Random seed is fixed (42) for reproducibility of fracture configurations.
    """
   
    _number_of_fractures: int = 100
    _domain_x_length: float = 100.0
    _domain_y_length: float = 30.0
    _dist_from_ref_point: float = 5.0

    # Corners of the rectangle (in CCW order)
    _bottom_left = np.array([0.0, 0.0, 0.0])
    _bottom_right = np.array([_domain_x_length, 0.0, 0.0])
    _top_right = np.array([_domain_x_length, _domain_y_length, 0.0])
    _top_left = np.array([0.0, _domain_y_length, 0.0])

    # Outlet coverage (example: right boundary)
    _outlet_coverage: list[tuple[np.ndarray, np.ndarray]] = [
        (_top_left, _bottom_left)
    ]

    # Inlet coverage (all 4 sides)
    _inlet_coverage: list[tuple[np.ndarray, np.ndarray]] = [
        # (_bottom_left,  _bottom_right),  # bottom
        # (_bottom_right, _top_right),     # right
        # (_top_right,    _top_left),      # top
        # (_top_left,     _bottom_left),   # left
    ]
    prod_x_point: float = _domain_x_length - 15.0
    prod_y_point: float = _domain_y_length*0.5

    _injection_points: list[np.ndarray] = [
        np.array([_domain_x_length-prod_x_point, prod_y_point])
    ]
    _production_points: list[np.ndarray] = [
        np.array([prod_x_point, prod_y_point])
    ]

    def set_domain(self) -> None:
        x_length_in_m = self.units.convert_units(self._domain_x_length, "m")
        y_length_in_m = self.units.convert_units(self._domain_y_length, "m")
        box: dict[str, pp.number] = {
            "xmax": x_length_in_m,
            "ymax": y_length_in_m
        }
        self._domain = pp.Domain(box)
    
    def set_fractures(self) -> None:
        
        # Optional for reproducibility
        np.random.seed(42)
        x_min, y_min = 0.0, 0.0
        x_max = self._domain_x_length
        y_max = self._domain_y_length
        domain_width = x_max - x_min
        domain_height = y_max - y_min
        min_length = 3.0
        max_length = 15.0

        fractures = []
        for _ in range(self._number_of_fractures):
            # Random center within bounds
            x_center = np.random.uniform(
                x_min + 0.1 * domain_width, 
                x_max - 0.1 * domain_width
            )
            y_center = np.random.uniform(
                y_min + 0.1 * domain_height, 
                y_max - 0.1 * domain_height
            )

            # Random angle and length
            theta = np.random.uniform(0, np.pi)
            length = np.random.uniform(min_length, max_length)

            dx = 0.5 * length * np.cos(theta)
            dy = 0.5 * length * np.sin(theta)

            x0, x1 = x_center - dx, x_center + dx
            y0, y1 = y_center - dy, y_center + dy

            coords = np.array([[x0, x1], [y0, y1]])
            fractures.append(pp.LineFracture(coords))
        self._fractures = fractures
      
    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")
 
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(2.0, "m")
        frac_cell_size = self.units.convert_units(2.0, "m")
        mesh_args: dict[str, float] = {
            "cell_size": cell_size,
            "cell_size_fracture": frac_cell_size
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
        self.mdg.add_subdomains(sd_0d)

        # Couple well to the matrix (0D <--> 2D coupling is allowed with wells)
        matrix = self.mdg.subdomains(dim=self.domain.dim)[0]
        cell_matrix = matrix.closest_cell(sd_0d.cell_centers)
        cell_well = np.array([0], dtype=int)
        cell_cell_map = sps.coo_matrix(
            (np.ones(1, dtype=bool), (cell_well, cell_matrix)),
            shape=(sd_0d.num_cells, matrix.num_cells),
        )
        _add_interface(0, matrix, sd_0d, self.mdg, cell_cell_map)