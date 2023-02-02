import porepy as pp
import numpy as np
from typing import Optional, Union
from typing import NamedTuple, Union
import numpy.typing as npt

__all__ = ["Domain"]


# -----> Utility functions
def bounding_box_of_point_cloud(
        point_cloud: np.ndarray, overlap: pp.number = 0
) -> dict[str, float]:
    """Obtain a bounding box of a point cloud.

    Parameters:
        point_cloud: ``shape=(nd, num_points)``

            Point cloud. nd should be 2 or 3.
        overlap: ``default=0``

            Extension of the bounding box outside the point cloud. Scaled with extent of
            the point cloud in the respective dimension.

    Returns:
        The domain represented as a dictionary with keywords ``xmin``, ``xmax``,
        ``ymin``, ``ymax``, and (if ``nd == 3``) ``zmin`` and ``zmax``.

    """
    max_coord = point_cloud.max(axis=1)
    min_coord = point_cloud.min(axis=1)
    dx = max_coord - min_coord
    bounding_box = {
        "xmin": min_coord[0] - dx[0] * overlap,
        "xmax": max_coord[0] + dx[0] * overlap,
    }
    if max_coord.size > 1:
        bounding_box["ymin"] = min_coord[1] - dx[1] * overlap
        bounding_box["ymax"] = max_coord[1] + dx[1] * overlap

    if max_coord.size == 3:
        bounding_box["zmin"] = min_coord[2] - dx[2] * overlap
        bounding_box["zmax"] = max_coord[2] + dx[2] * overlap

    return bounding_box


class Domain:
    """Class for the geometrical representation of the domain.

    Attributes:
        bounding_box (``dict[str, pp.number]``): Dictionary containing the bounding
            box of the domain. See __init__ documentation.
        polytope (``list[np.ndarray]``): Polytope (polygon for 2d and polyhedron for
            3d) defining the domain. See __init__ documentation.
        dim (int): Dimension on the domain.
        is_boxed (bool): Whether the domain is a box. We assume that if ``polytope``
            is used for instantiation, ``is_boxed = False``.

    """

    def __init__(
            self,
            bounding_box: Optional[dict[str, pp.number]] = None,
            polytope: Optional[list[np.ndarray]] = None,
    ):
        """

        Parameters:
            bounding_box: Dictionary containing the minimum and maximum coordinates
                of the vertices of the bounding box of the domain.

                Expected keywords are ``xmin``, ``xmax``, ``ymin``, ``ymax``, and if 3d,
                ``zmin`` and ``zmax``.
            polytope: Either a 2d or a 3d-polytope representing a non-boxed domain.

                A 2d-polytope (e.g., a polygon) is defined as a list of arrays of
                ``shape = (2, 2)``. Every array corresponds to one side of the
                polygon. The rows of each array contain the coordinates of the
                vertices defining the line, i.e., first row the x-coordinates and
                second row the y-coordinates.

                A 3d-polytope (e.g., a polyhedron) is a list of arrays of ``shape = (3,
                num_vertex)``. Every array represents a polygon with ``num_vertex``
                vertices. The rows of each array contain the coordinates of the
                vertices defining the polygon, i.e., first row the x-coordinates,
                second row the y-coordinates, and third row the z-coordinates.

        """

        # Make sure only one argument is passed
        if bounding_box is not None and polytope is not None:
            raise ValueError("Too many arguments. Expected box OR polytope.")

        if bounding_box is not None:
            self.bounding_box: dict[str, pp.number] = bounding_box
            self.polytope: list[np.ndarray] = self.polytope_from_bounding_box()
            self.dim: int = self.dimension_from_bounding_box()
            self.is_boxed: bool = True
        elif polytope is not None:
            self.polytope: list[np.ndarray] = polytope
            self.dim: int = polytope[0].shape[0]
            self.bounding_box: dict[str, pp.number] = self.bounding_box_from_polytope()
            self.is_boxed: bool = False  # A non-boxed domain is assumed in this case
        else:
            raise ValueError("Not enough arguments. Expected box OR polytope.")

    def bounding_box_from_polytope(self) -> dict[str, pp.number]:

        if self.dim == 2:

            # For a polygon, is easier to convert the polytope into an array
            polygon = np.empty(shape=(len(self.polytope), 4))
            for idx, side in enumerate(self.polytope):
                polygon[idx, :] = np.concatenate(side)

            xmin = np.min(polygon[:, :1])
            xmax = np.max(polygon[:, :1])
            ymin = np.min(polygon[:, 2:])
            ymax = np.max(polygon[:, 2:])

            box = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

        else:
            # For a polyhedron, we have to consider that the domain can be composed
            # of polygons with different number of vertices. This means that we
            # cannot use the trick from above

            x_coo = []
            y_coo = []
            z_coo = []

            for polygon in self.polytope:
                x_coo.append(polygon[0])
                y_coo.append(polygon[1])
                z_coo.append(polygon[2])

            x = np.concatenate(x_coo, axis=0)
            y = np.concatenate(y_coo, axis=0)
            z = np.concatenate(z_coo, axis=0)

            box = {
                "xmin": np.min(x),
                "xmax": np.max(x),
                "ymin": np.min(y),
                "ymax": np.max(y),
                "zmin": np.min(z),
                "zmax": np.max(z),
            }

        return box

    def polytope_from_bounding_box(self) -> list[np.ndarray]:

        if "zmin" in self.bounding_box.keys() and "zmax" in self.bounding_box.keys():
            polytope = self._polyhedron_from_bounding_box()
        else:
            polytope = self._polygon_from_bounding_box()

        return polytope

    def _polygon_from_bounding_box(self) -> list[np.ndarray]:

        x0 = self.bounding_box["xmin"]
        x1 = self.bounding_box["xmax"]
        y0 = self.bounding_box["ymin"]
        y1 = self.bounding_box["ymax"]

        west = np.array([[x0, x0], [y0, y1]])
        east = np.array([[x1, x1], [y1, y0]])
        south = np.array([[x1, x0], [y0, y0]])
        north = np.array([[x0, x1], [y1, y1]])

        bound_lines = [west, east, south, north]

        return bound_lines

    def _polyhedron_from_bounding_box(self) -> list[np.ndarray]:

        x0 = self.bounding_box["xmin"]
        x1 = self.bounding_box["xmax"]
        y0 = self.bounding_box["ymin"]
        y1 = self.bounding_box["ymax"]
        z0 = self.bounding_box["zmin"]
        z1 = self.bounding_box["zmax"]

        west = np.array([[x0, x0, x0, x0], [y0, y1, y1, y0], [z0, z0, z1, z1]])
        east = np.array([[x1, x1, x1, x1], [y0, y1, y1, y0], [z0, z0, z1, z1]])
        south = np.array([[x0, x1, x1, x0], [y0, y0, y0, y0], [z0, z0, z1, z1]])
        north = np.array([[x0, x1, x1, x0], [y1, y1, y1, y1], [z0, z0, z1, z1]])
        bottom = np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [z0, z0, z0, z0]])
        top = np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [z1, z1, z1, z1]])

        bound_planes = [west, east, south, north, bottom, top]

        return bound_planes

    def dimension_from_bounding_box(self) -> int:

        # Required keywords
        kw_1d = "xmin" and "xmax"
        kw_2d = kw_1d and "ymin" and "ymax"
        kw_3d = kw_2d and "zmin" and "zmax"

        if kw_3d in self.bounding_box.keys():
            return 3
        elif kw_2d in self.bounding_box.keys():
            return 2
        elif kw_1d in self.bounding_box.keys():
            return 1
        else:
            raise ValueError

class DomainSides(NamedTuple):
    """Type for domain sides."""

    all_bf: npt.NDArray[np.int_]
    """All boundary faces."""
    east: npt.NDArray[np.bool_]
    """East boundary faces."""
    west: npt.NDArray[np.bool_]
    """West boundary faces."""
    north: npt.NDArray[np.bool_]
    """North boundary faces."""
    south: npt.NDArray[np.bool_]
    """South boundary faces."""
    top: npt.NDArray[np.bool_]
    """Top boundary faces."""
    bottom: npt.NDArray[np.bool_]
    """Bottom boundary faces."""