"""Compute bounding boxes of geometric objects.
"""
from __future__ import annotations

from typing import Union

import numpy as np

import porepy as pp


def from_points(pts, overlap: float = 0) -> dict[str, float]:
    """Obtain a bounding box for a point cloud.

    Parameters:
        pts: np.ndarray (nd x np). Point cloud. nd should be 2 or 3
        overlap: Extension of the bounding box outside the point cloud. Scaled with
            extent of the point cloud in the respective dimension. Defaults to 0.

    Returns:
        The domain represented as a dictionary with keywords ``xmin``, ``xmax``,
        ``ymin``, ``ymax``, and (if ``nd == 3``) ``zmin`` and ``zmax``.

    """
    max_coord = pts.max(axis=1)
    min_coord = pts.min(axis=1)
    dx = max_coord - min_coord
    domain = {
        "xmin": min_coord[0] - dx[0] * overlap,
        "xmax": max_coord[0] + dx[0] * overlap,
    }
    if max_coord.size > 1:
        domain["ymin"] = min_coord[1] - dx[1] * overlap
        domain["ymax"] = max_coord[1] + dx[1] * overlap

    if max_coord.size == 3:
        domain["zmin"] = min_coord[2] - dx[2] * overlap
        domain["zmax"] = max_coord[2] + dx[2] * overlap
    return domain


def from_grid(g: pp.Grid) -> tuple[np.ndarray, np.ndarray]:
    """Return the bounding box of the grid.

    Parameters:
        g: The grid for which the bounding box is to be computed.

    Returns:
        ndarray ``(shape=(3,))``:
            Minimum node coordinates in each direction.

        ndarray ``(shape=(3,))``:
            Maximum node coordinates in each direction.

    """
    if g.dim == 0:
        coords = g.cell_centers
    else:
        coords = g.nodes
    return np.amin(coords, axis=1), np.amax(coords, axis=1)


def from_md_grid(
    mdg: pp.MixedDimensionalGrid, as_dict: bool = False
) -> Union[dict[str, float], tuple[np.ndarray, np.ndarray]]:
    """Return the bounding box of a mixed-dimensional grid.

    Parameters:
        mdg: Mixed-dimensional grid for which the bounding box is to be computed.
        as_dict: If ``True``, the bounding box is returned as a dictionary, if
            ``False``, it is represented by arrays with max and min values. Defaults to
            ``False``.

    Returns:
        dictionary *or* tuple of np.ndarray: If ``as_dict`` is ``True``, the bounding box is
            represented as a dictionary with keys ``xmin``, ``xmax``, ``ymin``,
            ``ymax``, ``zmin``, and ``zmax``. Else, two ``ndarrays`` are returned,
            containing the min and max values of the coordinates, respectively.

    """
    c_0s = np.empty((3, mdg.num_subdomains()))
    c_1s = np.empty((3, mdg.num_subdomains()))

    for i, grid in enumerate(mdg.subdomains()):
        c_0s[:, i], c_1s[:, i] = from_grid(grid)

    min_vals = np.amin(c_0s, axis=1)
    max_vals = np.amax(c_1s, axis=1)

    if as_dict:
        return {
            "xmin": min_vals[0],
            "xmax": max_vals[0],
            "ymin": min_vals[1],
            "ymax": max_vals[1],
            "zmin": min_vals[2],
            "zmax": max_vals[2],
        }
    else:
        return min_vals, max_vals


def make_bounding_planes_from_box(box: dict[str, float]) -> list[np.ndarray]:
    """Translate the bounding box into fractures. Tag them as boundaries.

    For now the domain specification is limited to a box consisting of six planes.

    Parameters:
        box: Dictionary containing max and min coordinates in three dimensions. The
            dictionary should contain the keys ``xmin``, ``xmax``, ``ymin``, ``ymax``,
            ``zmin``, and ``zmax``.

    Returns:
        list of the six bounding planes defined by 3 x 4 coordinate values.

    """
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
