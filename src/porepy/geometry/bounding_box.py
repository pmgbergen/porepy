"""Module containing methods for computing bounding boxes and domains.

In PorePy, a _bounding box_ is a 2-tuple of np.ndarray containing the minimum and
maximum node coordinates of a grid (or mixed-dimensional grid), whereas a _domain_ is a
dictionary with fields "xmin", "xmax", "ymin", "ymax", (and if 3d) "zmin" and "zmax".

The two objects contain the same geometrical information and their distinction is
mainly needed for legacy reasons.

"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

import porepy as pp

# Custom typings
number = pp.number
Box = tuple[np.ndarray, np.ndarray]
Domain = dict[str, number]


def domain_from_points(pts: np.ndarray, overlap: number = 0) -> Domain:
    """Generate a domain from a point cloud.

    Parameters:
        pts: ``shape=(nd, np)``

            Point cloud. ``nd`` should be ``2`` or ``3``.
        overlap: ``default=0``

            Extension of the domain outside the point cloud. Scaled with extent of
            the point cloud in the respective dimension.

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


def bounding_box_from_grid(sd: pp.Grid) -> Box:
    """Return the bounding box of a subdomain grid.

    Parameters:
        sd: The grid for which the bounding box is to be computed.

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


def bounding_box_from_mdg(mdg: pp.MixedDimensionalGrid) -> Box:
    """Return the bounding box of a mixed-dimensional grid.

    Parameters:
        mdg: Mixed-dimensional grid for which the bounding box is to be computed.

    A 2-tuple containing

        :obj:`~numpy.ndarray`: ``shape=(3, )``

            Minimum node coordinates in each direction.

        :obj:`~numpy.ndarray`: ``shape=(3, )``

            Maximum node coordinates in each direction.

    """
    c_0s = np.empty((3, mdg.num_subdomains()))
    c_1s = np.empty((3, mdg.num_subdomains()))

    for idx, sd in enumerate(mdg.subdomains()):
        c_0s[:, idx], c_1s[:, idx] = bounding_box_from_grid(sd)

    return np.amin(c_0s, axis=1), np.amax(c_1s, axis=1)


def bounding_box_as_domain(bounding_box: Box) -> Domain:
    """Convert a bounding box into a domain.

    Parameters:
        bounding_box: 2-tuple of arrays containing the minimum and maximum node
            coordinates.

    Returns:
        The domain represented as a dictionary with keywords ``xmin``, ``xmax``,
        ``ymin``, ``ymax``, and (if ``nd == 3``) ``zmin`` and ``zmax``.

    """
    b_min, b_max = bounding_box
    nd = b_min.shape[0]
    assert 2 <= nd <= 3

    if nd == 2:
        domain = {
            "xmin": b_min[0],
            "xmax": b_max[0],
            "ymin": b_min[1],
            "ymax": b_max[1],
        }
    else:
        domain = {
            "xmin": b_min[0],
            "xmax": b_max[0],
            "ymin": b_min[1],
            "ymax": b_max[1],
            "zmin": b_min[2],
            "zmax": b_max[2],
        }

    return domain


def bounding_planes_from_domain(domain: Domain) -> list[np.ndarray]:
    """Translate a domain into fractures. Tag them as boundaries.

    For now the domain specification is limited to a box consisting of six planes.

    Parameters:
        domain: Dictionary containing max and min coordinates in three dimensions. The
            dictionary should contain the keys ``xmin``, ``xmax``, ``ymin``, ``ymax``,
            ``zmin``, and ``zmax``.

    Returns:
        List of the six bounding planes defined by 3 x 4 coordinate values.

    """
    # Sanity check
    required_keys = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
    for key in required_keys:
        assert key in domain.keys()

    x0 = domain["xmin"]
    x1 = domain["xmax"]
    y0 = domain["ymin"]
    y1 = domain["ymax"]
    z0 = domain["zmin"]
    z1 = domain["zmax"]

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
