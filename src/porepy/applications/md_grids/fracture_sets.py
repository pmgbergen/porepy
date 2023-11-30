"""Generator functions for fracture sets, i.e. lists of
:class:`~porepy.fracs.fractures.Fracture` objects.

"""

from __future__ import annotations

from typing import Optional

import numpy as np

import porepy as pp


def orthogonal_fractures_2d(
    size: pp.number,
    fracture_endpoints: Optional[list[np.ndarray]] = None,
) -> list[pp.LineFracture]:
    """Return a list of two orthogonal fractures of specified side length.

    Parameters:
        size: The side length of the line fractures.
        fracture_endpoints: List of arrays containing the endpoints of the fractures.
            Array i contains the non-constant endpoints of fracture i, i.e. the first
            entry is the y coordinates of fracture one and the second entry is the x
            coordinate of fracture two.

    Returns:
        List of two orthogonal fractures. Fracture i has constant i coordinate
            equal to size / 2.

    """
    if fracture_endpoints is None:
        fracture_endpoints = [np.array([0, size]), np.array([0, size])]
    # Prepare for stacking.
    fracture_endpoints = [pts.reshape((1, 2)) for pts in fracture_endpoints]

    constant_coords = np.array([0.5, 0.5]).reshape((1, 2)) * size
    pts = []
    # First fracture has constant x coordinate equal to size / 2, y coordinate varies.
    pts.append(np.vstack((constant_coords, fracture_endpoints[0])))
    # Second fracture has constant y coordinate equal to size / 2, x coordinate varies.
    pts.append(np.vstack((fracture_endpoints[1], constant_coords)))
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


def benchmark_regular_2d_fractures(
    size: Optional[pp.number] = 1,
) -> list[pp.LineFracture]:
    """Return a list of regular fractures as used in the 2d benchmark study.

    Parameters:
        size: The side length of the line fractures.

    Returns:
        List of fractures.

    """
    points = (
        np.array(
            [
                [0.0, 0.5],
                [1.0, 0.5],
                [0.5, 0.0],
                [0.5, 1.0],
                [0.5, 0.75],
                [1.0, 0.75],
                [0.75, 0.5],
                [0.75, 1.0],
                [0.5, 0.625],
                [0.75, 0.625],
                [0.625, 0.5],
                [0.625, 0.75],
            ]
        ).T
        * size
    )
    fracs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]).T
    return pp.frac_utils.pts_edges_to_linefractures(points, fracs)


def seven_fractures_one_L_intersection(
    size: Optional[pp.number] = 1,
) -> list[pp.LineFracture]:
    """Return a list of seven fractures with one L intersection.

    First used in example one of the paper `Finite volume discretization for poroelastic
    media with fractures modeled by contact mechanics.` by Berge et al. 2019.

    Parameters:
        size: The domain size in y direction. The domain size in x direction is
            2 * size.

    Returns:
        List of fractures.

    """
    points = np.array(
        [
            [0.2, 0.7],
            [0.5, 0.7],
            [0.8, 0.65],
            [1, 0.3],
            [1.8, 0.4],
            [0.2, 0.3],
            [0.6, 0.25],
            [1.0, 0.4],
            [1.7, 0.85],
            [1.5, 0.65],
            [2.0, 0.55],
            [1.5, 0.05],
            [1.4, 0.25],
        ]
    ).T
    # The fracture endpoints are given as indices in the points array
    fracs = np.array([[0, 1], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]).T
    return pp.frac_utils.pts_edges_to_linefractures(points, fracs)
