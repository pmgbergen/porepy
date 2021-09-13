"""
Compute bounding boxes of geometric objects.
"""
from typing import Dict, List

import numpy as np

import porepy as pp

module_sections = ["geometry", "gridding"]


@pp.time_logger(sections=module_sections)
def from_points(pts, overlap=0):
    """Obtain a bounding box for a point cloud.

    Parameters:
        pts: np.ndarray (nd x npt). Point cloud. nd should be 2 or 3
        overlap (double, defaults to 0): Extension of the bounding box outside
            the point cloud. Scaled with extent of the point cloud in the
            respective dimension.

    Returns:
        domain (dictionary): Containing keywords xmin, xmax, ymin, ymax, and
            possibly zmin and zmax (if nd == 3)

    """
    max_coord = pts.max(axis=1)
    min_coord = pts.min(axis=1)
    dx = max_coord - min_coord
    domain = {
        "xmin": min_coord[0] - dx[0] * overlap,
        "xmax": max_coord[0] + dx[0] * overlap,
        "ymin": min_coord[1] - dx[1] * overlap,
        "ymax": max_coord[1] + dx[1] * overlap,
    }

    if max_coord.size == 3:
        domain["zmin"] = min_coord[2] - dx[2] * overlap
        domain["zmax"] = max_coord[2] + dx[2] * overlap
    return domain


def make_bounding_planes_from_box(box: Dict[str, float]) -> List[np.ndarray]:
    """
    Translate the bounding box into fractures. Tag them as boundaries.
    For now limited to a box consisting of six planes.

    Parameters:
        box: dictionary containing max and min coordinates in three dimensions.

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
