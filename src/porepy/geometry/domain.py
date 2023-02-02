import porepy as pp
import numpy as np
from typing import Optional, Union

__all__ = ["Domain"]


class Domain:
    """Class for the geometrical representation of the domain."""

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
            self.box: dict[str, pp.number] = bounding_box
            self.polytope: list[np.ndarray] = polytope_from_box(bounding_box)
            self.dim: int = dim_from_box(bounding_box)
            self.is_boxed: bool = True
        elif polytope is not None:
            self.polytope: list[np.ndarray] = polytope
            self.dim: int = polytope[0].shape[0]
            self.box: dict[str, pp.number] = box_from_polytope(polytope)
            self.is_boxed: bool = False
        else:
            raise ValueError("Not enough arguments. Expected box OR polytope.")


def box_from_polytope(polytope: list[np.ndarray]) -> dict[str, pp.number]:

    # polygon (list of lines): list of np.ndarrays of shape (2, 2).
    # polyhedron (list of polygons): list of np.ndarrays of shape (3, num_vertices).

    if polytope[0].shape[1] == 2:

        num_sides = len(polytope)

        # For a polygon, is easier to convert the polytope into an array
        polygon = np.empty(shape=(num_sides, 4))
        for idx, side in enumerate(polytope):
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

        for polygon in polytope:
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


def polytope_from_box(box: dict[str, pp.number]) -> list[np.ndarray]:

    if "zmin" in box.keys() and "zmax" in box.keys():
        polytope = _polyhedron_from_box(box)
    else:
        polytope = _polygon_from_box(box)

    return polytope


def _polygon_from_box(box: dict[str, pp.number]) -> list[np.ndarray]:

    x0 = box["xmin"]
    x1 = box["xmax"]
    y0 = box["ymin"]
    y1 = box["ymax"]

    west = np.array([[x0, x0], [y0, y1]])
    east = np.array([[x1, x1], [y1, y0]])
    south = np.array([[x1, x0], [y0, y0]])
    north = np.array([[x0, x1], [y1, y1]])

    bound_lines = [west, east, south, north]

    return bound_lines


def _polyhedron_from_box(box: dict[str, pp.number]) -> list[np.ndarray]:

    x0 = box["xmin"]
    x1 = box["xmax"]
    y0 = box["ymin"]
    y1 = box["ymax"]
    z0 = box["zmin"]
    z1 = box["zmax"]

    west = np.array([[x0, x0, x0, x0], [y0, y1, y1, y0], [z0, z0, z1, z1]])
    east = np.array([[x1, x1, x1, x1], [y0, y1, y1, y0], [z0, z0, z1, z1]])
    south = np.array([[x0, x1, x1, x0], [y0, y0, y0, y0], [z0, z0, z1, z1]])
    north = np.array([[x0, x1, x1, x0], [y1, y1, y1, y1], [z0, z0, z1, z1]])
    bottom = np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [z0, z0, z0, z0]])
    top = np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [z1, z1, z1, z1]])

    bound_planes = [west, east, south, north, bottom, top]

    return bound_planes


def dim_from_box(box: dict[str, pp.number]) -> int:

    # Required keywords
    kw_1d = "xmin" and "xmax"
    kw_2d = kw_1d and "ymin" and "ymax"
    kw_3d = kw_2d and "zmin" and "zmax"

    if kw_3d in box.keys():
        return 3
    elif kw_2d in box.keys():
        return 2
    elif kw_1d in box.keys():
        return 1
    else:
        raise ValueError
