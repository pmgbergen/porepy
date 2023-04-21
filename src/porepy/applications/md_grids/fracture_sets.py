"""Generator functions for fracture sets, i.e. lists of
:class:`~porepy.fracs.fractures.Fracture` objects.

"""

from __future__ import annotations

import numpy as np

import porepy as pp


def orthogonal_fractures_2d(size: pp.number) -> list[pp.LineFracture]:
    """Return a list of two orthogonal fractures of specified side length.

    Parameters:
        size: The side length of the line fractures.

    Returns:
        List of two orthogonal fractures. Fracture i has constant i coordinate
            equal to size / 2.

    """
    coords_a = [0.5, 0.5]
    coords_b = [0, 1]
    pts = []
    pts.append(np.array([coords_a, coords_b]) * size)
    pts.append(np.array([coords_b, coords_a]) * size)
    return [pp.LineFracture(pts[i]) for i in range(2)]


def orthogonal_fractures_3d(size: pp.number) -> list[pp.PlaneFracture]:
    """Return a list of three orthogonal fractures of specified side length.

    Parameters:
        size: The side length of the plane fractures.

    Returns:
        List of three orthogonal fractures. Fracture i has constant i coordinate equal
            to size / 2.

    """
    coords_a = [0.5, 0.5, 0.5, 0.5]
    coords_b = [0, 0, 1, 1]
    coords_c = [0, 1, 1, 0]
    pts = []
    pts.append(np.array([coords_a, coords_b, coords_c]) * size)
    pts.append(np.array([coords_b, coords_a, coords_c]) * size)
    pts.append(np.array([coords_b, coords_c, coords_a]) * size)
    return [pp.PlaneFracture(pts[i]) for i in range(3)]
