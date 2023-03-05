"""Module containing classes and functions for defining and manipulating a domain."""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import numpy.typing as npt

import porepy as pp


class Domain:
    """Class for the geometrical representation of the domain.

    There are two ways of constructing a domain in PorePy:

    - (1) By passing the bounding box as a dictionary (we assume therefore that the
      domain is a box) or,

    - (2) By passing a polytope (polygon in 2d and polyhedron in 3d), which are lists of
      numpy arrays defining a general domain (this also includes non-convex domains).

    Examples:

        .. code:: python3

            # Create a domain from a bounding box
            domain_from_box = pp.Domain(
                bounding_box={"xmin": 0, "xmax": 1, "ymin": 0, "ymin": 1}
            )

            # Create a domain from a 2d-polytope
            line_1 = np.array([[0, 0], [0, 1]])
            line_2 = np.array([[0, 0.5], [1, 1.5]])
            line_3 = np.array([[0.5, 1], [1.5, 1]])
            line_4 = np.array([[1, 1], [1, 0]])
            line_5 = np.array([[1, 0], [0, 0]])
            irregular_pentagon = [line_1, line_2, line_3, line_4, line_5]
            domain_from_polytope = pp.Domain(polytope=irregular_pentagon)

    Raises:
        - ``ValueError`` if both ``bounding_box`` AND ``polytope`` are given.
        - ``ValueError`` if no arguments are given.

    Parameters:
        bounding_box: Dictionary containing the minimum and maximum coordinates
            defining the bounding box of the domain.

            Expected keywords are ``xmin``, ``xmax``, ``ymin``, ``ymax``, and if 3d,
            ``zmin`` and ``zmax``.
        polytope: Either a 2d or a 3d-polytope representing a non-boxed domain.

            A 2d-polytope (e.g., a polygon) is defined as a list of arrays of
            ``shape = (2, 2)``. Every array corresponds to one side of the polygon.
            The rows of each array contain the coordinates of the vertices defining
            the line, i.e., first row the x-coordinates and second row the
            y-coordinates.

            A 3d-polytope (e.g., a polyhedron) is a list of arrays of ``shape = (3,
            num_vertex)``. Every array represents a polygon with ``num_vertex``
            vertices. The rows of each array contain the coordinates of the vertices
            defining the polygon, i.e., first row the x-coordinates, second row the
            y-coordinates, and third row the z-coordinates.

    """

    def __init__(
        self,
        bounding_box: Optional[dict[str, pp.number]] = None,
        polytope: Optional[list[np.ndarray]] = None,
    ) -> None:

        # Sanity check
        if bounding_box is not None and polytope is not None:
            raise ValueError("Too many arguments. Expected box OR polytope.")

        # Attributes declaration
        self.bounding_box: dict[str, pp.number]
        """Bounding box of the domain. See class documentation for details."""

        self.polytope: list[np.ndarray]
        """Polytope defining the domain. See class documentation for details."""

        self.dim: int
        """Dimension of the domain. Inferred from the bounding box or the polytope."""

        self.is_boxed: bool
        """Whether the domain is a box or not.

        If ``polytope`` is used to instantiate the class, we assume that
        ``is_boxed=False`` (irrespective of whether the polytope is a box or not).

        """

        if bounding_box is not None:
            self.bounding_box = bounding_box
            self.polytope = self.polytope_from_bounding_box()
            self.dim = bounding_box_dimension(bounding_box)
            self.is_boxed = True
        elif polytope is not None:
            self.polytope = polytope
            self.dim = polytope[0].shape[0]
            self.bounding_box = self.bounding_box_from_polytope()
            self.is_boxed = False
        else:
            raise ValueError("Not enough arguments. Expected box OR polytope.")

    def __repr__(self) -> str:
        if self.is_boxed:
            s = f"pp.Domain(bounding_box={self.bounding_box})"
        else:
            # TODO: Create a function that prints a polytope prettily
            s = f"pp.Domain(polytope=\n {self.polytope} \n )"
        return s

    def __str__(self) -> str:

        if self.is_boxed:
            if self.dim == 2:
                s = f"Rectangle with bounding box: {self.bounding_box}."
            else:
                s = f"Cuboid with bounding box: {self.bounding_box}."
        else:
            if self.dim == 2:
                s = (
                    f"Polygon with {len(self.polytope)} sides and bounding box: "
                    f"{self.bounding_box}."
                )
            else:
                s = (
                    f"Polyhedron with {len(self.polytope)} bounding planes and "
                    f"bounding box: {self.bounding_box}."
                )

        return s

    def __eq__(self, other: object) -> bool:

        # Two domains are equal if they have the same polytope. Note that this assumes
        # that the arrays of the polytope list are stored in the exact same order.
        # TODO: The condition for equality is too strict, we might want to consider
        #  the possibility that polytopes are defined in different orders

        if not isinstance(other, pp.Domain):
            return NotImplemented
        else:
            check = []
            for item_self, item_other in zip(self.polytope, other.polytope):
                check.append(np.all(item_self == item_other))
            return all(check)

    def bounding_box_from_polytope(self) -> dict[str, pp.number]:
        """Obtain the bounding box of a polytope.

        Returns:
            Dictionary containing the bounding box information.

        """
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
        """Obtain the polytope representation of a bounding box.

        Returns:
            List of arrays representing the polytope. Shape of array will depend
            on the dimension of the bounding box. See __init__ documentation for a
            detailed explanation.

        """
        if "zmin" in self.bounding_box.keys() and "zmax" in self.bounding_box.keys():
            polytope = self._polyhedron_from_bounding_box()
        else:
            polytope = self._polygon_from_bounding_box()

        return polytope

    def _polygon_from_bounding_box(self) -> list[np.ndarray]:
        """Compute the polygon associated with a two-dimensional bounding box.

        Returns:
            List of four arrays of ``shape = (2, 2)`` representing the sides of
            the polygon (e.g., the rectangle).

        """
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
        """Compute the polyhedron associated with a three-dimensional bounding box.

        Returns:
            List of six arrays of ``shape = (3, 4)`` representing the bounding
            planes of the polyhedron (e.g., the box).

        """
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


# -----> Functions related to manipulation/coversion of bounding boxes
def bounding_box_dimension(bounding_box: dict[str, pp.number]) -> int:
    """Obtain the dimension of a bounding box.

    Returns:
        Dimension of the bounding box.

    """
    # Required keywords
    kw_1d = "xmin" and "xmax"
    kw_2d = kw_1d and "ymin" and "ymax"
    kw_3d = kw_2d and "zmin" and "zmax"

    if kw_3d in bounding_box.keys():
        return 3
    elif kw_2d in bounding_box.keys():
        return 2
    elif kw_1d in bounding_box.keys():
        return 1
    else:
        raise ValueError


def bounding_box_of_point_cloud(
    point_cloud: np.ndarray, overlap: pp.number = 0
) -> dict[str, float]:
    """Obtain the bounding box of a point cloud.

    Parameters:
        point_cloud: ``shape=(nd, num_points)``.

            Point cloud. nd should be 2 or 3.
        overlap: ``default=0``

            Extension of the bounding box outside the point cloud. Scaled with extent of
            the point cloud in the respective dimension.

    Returns:
        The domain represented as a dictionary with keywords ``xmin``, ``xmax``,
        ``ymin``, ``ymax``, and (if ``nd == 3``) ``zmin`` and ``zmax``.

    """
    # Sanity check
    if point_cloud.shape[0] not in [2, 3]:
        raise ValueError("Point cloud must be either 2d or 3d.")

    # Get min/max coordinates and range of the point cloud
    max_coord = point_cloud.max(axis=1)
    min_coord = point_cloud.min(axis=1)
    point_cloud_range = max_coord - min_coord

    # First and second dimension always preseent
    bounding_box = {
        "xmin": min_coord[0] - point_cloud_range[0] * overlap,
        "xmax": max_coord[0] + point_cloud_range[0] * overlap,
        "ymin": min_coord[1] - point_cloud_range[1] * overlap,
        "ymax": max_coord[1] + point_cloud_range[1] * overlap,
    }

    # Add third dimension if ``nd==3``.
    if max_coord.size == 3:
        bounding_box["zmin"] = min_coord[2] - point_cloud_range[2] * overlap
        bounding_box["zmax"] = max_coord[2] + point_cloud_range[2] * overlap

    return bounding_box


def mdg_minmax_coordinates(
    mdg: pp.MixedDimensionalGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the minimum and maximum coordinates of a mixed-dimensional grid.

    Parameters:
        mdg: The mixed-dimensional grid for which the min/max coordinates will be
            computed.

    Returns:
        A 2-tuple containing

        :obj:`~numpy.ndarray`: ``shape=(3, )``

            Minimum node coordinates in each direction.

        :obj:`~numpy.ndarray`: ``shape=(3, )``

            Maximum node coordinates in each direction.

    """

    c_0s = np.empty((3, mdg.num_subdomains()))
    c_1s = np.empty((3, mdg.num_subdomains()))

    for i, grid in enumerate(mdg.subdomains()):
        c_0s[:, i], c_1s[:, i] = grid_minmax_coordinates(grid)

    min_vals = np.amin(c_0s, axis=1)
    max_vals = np.amax(c_1s, axis=1)

    return min_vals, max_vals


def grid_minmax_coordinates(sd: pp.Grid) -> tuple[np.ndarray, np.ndarray]:
    """Return the minimum and maximum coordinates of a grid.

    Parameters:
        sd: The grid for which the bounding box will be computed.

    Returns:
        A 2-tuple containing

        :obj:`~numpy.ndarray`: ``shape=(3, )``

            Minimum node coordinates in each direction.

        :obj:`~numpy.ndarray`: ``shape=(3, )``

            Maximum node coordinates in each direction.

    """

    if sd.dim == 0:
        coords = sd.cell_centers
    else:
        coords = sd.nodes

    return np.amin(coords, axis=1), np.amax(coords, axis=1)
