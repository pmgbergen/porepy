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


def benchmark_2d_case_1(size: pp.number = 1) -> list[pp.LineFracture]:
    """Return a list of regular fractures as used in case 1 of the 2d benchmark study by
    Flemisch et al. 2018.

    Parameters:
        size: The side length of the domain.

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


def benchmark_2d_case_3(size: pp.number = 1) -> list[pp.LineFracture]:
    """Return a list of fractures as used in case 3 of the 2d benchmark study by
    Flemisch et al. 2018.

    Parameters:
        size: The side length of the domain.

    Returns:
        List of fractures.

    """
    points = [
        np.array([[0.0500, 0.2200], [0.4160, 0.0624]]),
        np.array([[0.0500, 0.2500], [0.2750, 0.1350]]),
        np.array([[0.1500, 0.4500], [0.6300, 0.0900]]),
        np.array([[0.1500, 0.4000], [0.9167, 0.5000]]),
        np.array([[0.6500, 0.849723], [0.8333, 0.167625]]),
        np.array([[0.7000, 0.849723], [0.2350, 0.167625]]),
        np.array([[0.6000, 0.8500], [0.3800, 0.2675]]),
        np.array([[0.3500, 0.8000], [0.9714, 0.7143]]),
        np.array([[0.7500, 0.9500], [0.9574, 0.8155]]),
        np.array([[0.1500, 0.4000], [0.8363, 0.9727]]),
    ]
    fractures = [pp.LineFracture(pts * size) for pts in points]
    return fractures


def seven_fractures_one_L_intersection(size: pp.number = 1) -> list[pp.LineFracture]:
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
