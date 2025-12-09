from abc import abstractmethod
from typing import Literal

import numpy as np
import porepy as pp


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


class SimpleGeometryHorizontal(Geometry):
    """A class to represent a simple 1D geometry for a simulation domain.
    The start of domain serve as inlet and end of domain serves as the outlet
    """

    _dist_from_ref_point: float = 5.0
    _inlet_centre: np.ndarray = np.array([0.0, 5.0, 0.0])
    _outlet_centre: np.ndarray = np.array([2000.0, 5.0, 0.0])

    def set_domain(self) -> None:
        x_length = self.units.convert_units(2000.0, "m")
        y_length = self.units.convert_units(10.0, "m")
        box: dict[str, pp.number] = {"xmax": x_length, "ymax": y_length}
        self._domain = pp.Domain(box)

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(10.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def get_inlet_outlet_sides(
        self, sd: pp.Grid | pp.BoundaryGrid
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(sd, pp.Grid):
            face_centers = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            face_centers = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf

        def find_facets(center: np.ndarray) -> np.ndarray:
            logical = Geometry.harvest_sphere_members(
                center, self._dist_from_ref_point, face_centers[bf_indices]
            )
            return bf_indices[logical]

        inlet_facets = find_facets(self._inlet_centre)
        outlet_facets = find_facets(self._outlet_centre)
        return inlet_facets, outlet_facets
    