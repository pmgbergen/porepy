"""
This file contains utility functions for setting up grid buckets for 2d networks.
The default is simplex grids, but Cartesian grids are also possible for the simplest
geometries.
The provided geometries are:
        Simple unit square geometries
    single_horizontal: Single horizontal line at y=0.5
    two_intersecting: Two lines intersecting at (0.5, 0.5)
        More complex geometries
    seven_fractures_one_L: Seven fractures with one L intersection
    benchmark_regular: Six fractures intersecting in 3 X and 6 Y intersections
"""
from __future__ import annotations

from typing import Optional

import numpy as np

import porepy as pp
import porepy.grids.standard_grids.utils as utils


def _n_cells(mesh_args: np.ndarray | dict | None) -> np.ndarray:
    """Convert mesh_args to n_cells argument of cartesian grid construction."""
    if mesh_args is None:
        return np.array([2, 2])
    else:
        if isinstance(mesh_args, list):
            mesh_args = np.array(mesh_args)
        assert isinstance(mesh_args, np.ndarray)
        return mesh_args


def single_horizontal(
    mesh_args: Optional[dict | np.ndarray] = None,
    x_endpoints: Optional[np.ndarray] = None,
    simplex: bool = True,
):
    """
    Create a grid bucket for a domain containing a single horizontal fracture at y=0.5.

    Args:
        mesh_args:  For triangular grids: Dictionary containing at least "mesh_size_frac". If
                        the optional values of "mesh_size_bound" and "mesh_size_min" are
                        not provided, these are set by utils.set_mesh_sizes.
                    For cartesian grids: np.array containing number of cells in x and y
                        direction.
        x_endpoints (list): Contains the x coordinates of the two endpoints. If not
            provided, the endpoints will be set to [0, 1]

    Returns:
        Mixed-dimensional grid and domain.

    """
    if x_endpoints is None:
        x_endpoints = np.array([0, 1])

    domain = utils.unit_domain(2)
    if simplex:
        if mesh_args is None:
            mesh_args = {"mesh_size_frac": 0.2}

        if not isinstance(mesh_args, dict):
            # The numpy-array format for mesh_args should not be used for simplex grids
            raise ValueError("Mesh arguments should be a dictionary for simplex grids")

        points = np.array([x_endpoints, [0.5, 0.5]])
        edges = np.array([[0], [1]])
        mdg = utils.make_mdg_2d_simplex(mesh_args, points, edges, domain)

    else:
        fracture = np.array([x_endpoints, [0.5, 0.5]])
        mdg = pp.meshing.cart_grid([fracture], _n_cells(mesh_args), physdims=np.ones(2))
    return mdg, domain


def single_vertical(
    mesh_args: Optional[np.ndarray | dict] = None,
    y_endpoints: Optional[np.ndarray] = None,
    simplex: Optional[bool] = True,
):
    """
    Create a grid bucket for a domain containing a single vertical fracture at x=0.5.

    Args:
        mesh_args:
            For triangular grids: Dictionary containing at least "mesh_size_frac". If
                the optional values of "mesh_size_bound" and "mesh_size_min" are
                not provided, these are set by utils.set_mesh_sizes.
            For cartesian grids: Array containing number of cells in x and y
                    direction.
        y_endpoints:
            Contains the y coordinates of the two endpoints. If not
            provided, the endpoints will be set to [0, 1]
        simplex:
            Grid type. If False, a Cartesian grid is returned.


    Returns:
        Mixed-dimensional grid and domain.

    """
    if y_endpoints is None:
        y_endpoints = np.array([0, 1])
    domain = utils.unit_domain(2)
    if simplex:
        if mesh_args is None:
            mesh_args = {"mesh_size_frac": 0.2}

        if not isinstance(mesh_args, dict):
            # The numpy-array format for mesh_args should not be used for simplex grids
            raise ValueError("Mesh arguments should be a dictionary for simplex grids")

        points = np.array([[0.5, 0.5], y_endpoints])
        edges = np.array([[0], [1]])
        mdg = utils.make_mdg_2d_simplex(mesh_args, points, edges, domain)

    else:
        fracture = np.array([[0.5, 0.5], y_endpoints])
        mdg = pp.meshing.cart_grid([fracture], _n_cells(mesh_args), physdims=np.ones(2))
    return mdg, domain


def two_intersecting(
    mesh_args: Optional[dict | np.ndarray] = None,
    x_endpoints: Optional[list] = None,
    y_endpoints: Optional[list] = None,
    simplex: bool = True,
):
    """
    Create a grid bucket for a domain containing fractures, one horizontal and one vertical
    at y=0.5 and x=0.5 respectively.

    Args:
        mesh_args:  For triangular grids: Dictionary containing at least "mesh_size_frac". If
                        the optional values of "mesh_size_bound" and "mesh_size_min" are
                        not provided, these are set by utils.set_mesh_sizes.
                    For cartesian grids: List containing number of cells in x and y
                        direction.
        x_endpoints (list): containing the x coordinates of the two endpoints of the
            horizontal fracture. If not provided, the endpoints will be set to [0, 1].
        y_endpoints (list): Contains the y coordinates of the two endpoints of the
            vertical fracture. If not provided, the endpoints will be set to [0, 1].
        simplex (bool): Whether to use triangular or Cartesian 2d grid.

    Returns:
        Mixed-dimensional grid and domain.

    """

    if x_endpoints is None:
        x_endpoints = [0, 1]
    if y_endpoints is None:
        y_endpoints = [0, 1]
    domain = utils.unit_domain(2)
    if simplex:
        if mesh_args is None:
            mesh_args = {"mesh_size_frac": 0.2}

        if not isinstance(mesh_args, dict):
            # The numpy-array format for mesh_args should not be used for simplex grids
            raise ValueError("Mesh arguments should be a dictionary for simplex grids")

        points = np.array(
            [
                [x_endpoints[0], x_endpoints[1], 0.5, 0.5],
                [0.5, 0.5, y_endpoints[0], y_endpoints[1]],
            ]
        )
        edges = np.array([[0, 2], [1, 3]])
        mdg = utils.make_mdg_2d_simplex(mesh_args, points, edges, domain)

    else:
        fracture0 = np.array([x_endpoints, [0.5, 0.5]])
        fracture1 = np.array([[0.5, 0.5], y_endpoints])
        mdg = pp.meshing.cart_grid(
            [fracture0, fracture1],
            _n_cells(mesh_args),
            physdims=[domain["xmax"], domain["ymax"]],
        )
        mdg.compute_geometry()
    return mdg, domain


def seven_fractures_one_L_intersection(mesh_args: dict):
    """
    Create a grid bucket for a domain containing the network introduced as example 1 of
    Berge et al. 2019: Finite volume discretization for poroelastic media with fractures
    modeled by contact mechanics.

    Args:
        mesh_args: Dictionary containing at least "mesh_size_frac". If the optional
            values of "mesh_size_bound" and "mesh_size_min" are not provided, these are
            set by utils.set_mesh_sizes.

    Returns:
        Mixed-dimensional grid and domain.

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
    edges = np.array([[0, 1], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]).T
    domain = {"xmin": 0, "ymin": 0, "xmax": 2, "ymax": 1}
    mdg = utils.make_mdg_2d_simplex(mesh_args, points, edges, domain)
    return mdg, domain


def benchmark_regular(mesh_args: dict, is_coarse: bool = False):
    """
    Create a grid bucket for a domain containing the network introduced as example 2 of
    Berre et al. 2018: Benchmarks for single-phase flow in fractured porous media.

    Args:
        mesh_args: Dictionary containing at least "mesh_size_frac". If the optional
            values of "mesh_size_bound" and "mesh_size_min" are not provided, these are
            set by utils.set_mesh_sizes.
    Returns:
        Mixed-dimensional grid and domain.
    """
    points = np.array(
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
    domain = utils.unit_domain(2)
    edges = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]).T
    mdg = utils.make_mdg_2d_simplex(mesh_args, points, edges, domain=domain)
    if is_coarse:
        pp.coarsening.coarsen(mdg, "by_volume")
    return mdg, domain
