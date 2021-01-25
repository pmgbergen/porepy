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

import numpy as np

import porepy as pp
import porepy.grids.standard_grids.utils as utils

module_sections = ["grids", "gridding"]
unit_domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}


@pp.time_logger(sections=module_sections)
def single_horizontal(mesh_args=None, x_endpoints=None, simplex=True):
    """
    Create a grid bucket for a domain containing a single horizontal fracture at y=0.5.

    Args:
        mesh_args:  For triangular grids: Dictionary containing at least "mesh_size_frac". If
                        the optional values of "mesh_size_bound" and "mesh_size_min" are
                        not provided, these are set by utils.set_mesh_sizes.
                    For cartesian grids: List containing number of cells in x and y
                        direction.
        x_endpoints (list): Contains the x coordinates of the two endpoints. If not
            provided, the endpoints will be set to [0, 1]

    Returns:
        Grid bucket for the domain.

    """
    if x_endpoints is None:
        x_endpoints = [0, 1]

    if simplex:
        if mesh_args is None:
            mesh_args = {"mesh_size_frac": 0.2}
        points = np.array([x_endpoints, [0.5, 0.5]])
        edges = np.array([[0], [1]])
        gb = utils.make_gb_2d_simplex(mesh_args, points, edges, unit_domain)

    else:
        fracture = np.array([x_endpoints, [0.5, 0.5]])
        gb = pp.meshing.cart_grid(
            [fracture], mesh_args, physdims=[unit_domain["xmax"], unit_domain["ymax"]]
        )
    return gb, unit_domain


@pp.time_logger(sections=module_sections)
def single_vertical(mesh_args=None, y_endpoints=None, simplex=True):
    """
    Create a grid bucket for a domain containing a single vertical fracture at x=0.5.

    Args:
        mesh_args:  For triangular grids: Dictionary containing at least "mesh_size_frac". If
                        the optional values of "mesh_size_bound" and "mesh_size_min" are
                        not provided, these are set by utils.set_mesh_sizes.
                    For cartesian grids: List containing number of cells in x and y
                        direction.
        y_endpoints (list): Contains the y coordinates of the two endpoints. If not
            provided, the endpoints will be set to [0, 1]

    Returns:
        Grid bucket for the domain.

    """
    if y_endpoints is None:
        y_endpoints = [0, 1]

    if simplex:
        if mesh_args is None:
            mesh_args = {"mesh_size_frac": 0.2}
        points = np.array([[0.5, 0.5], y_endpoints])
        edges = np.array([[0], [1]])
        gb = utils.make_gb_2d_simplex(mesh_args, points, edges, unit_domain)

    else:
        fracture = np.array([[0.5, 0.5], y_endpoints])
        gb = pp.meshing.cart_grid(
            [fracture], mesh_args, physdims=[unit_domain["xmax"], unit_domain["ymax"]]
        )
    return gb


@pp.time_logger(sections=module_sections)
def two_intersecting(mesh_args=None, x_endpoints=None, y_endpoints=None, simplex=True):
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
        Grid bucket for the domain.

    """

    if x_endpoints is None:
        x_endpoints = [0, 1]
    if y_endpoints is None:
        y_endpoints = [0, 1]

    if simplex:
        if mesh_args is None:
            mesh_args = {"mesh_size_frac": 0.2}
        points = np.array(
            [
                [x_endpoints[0], x_endpoints[1], 0.5, 0.5],
                [0.5, 0.5, y_endpoints[0], y_endpoints[1]],
            ]
        )
        edges = np.array([[0, 2], [1, 3]])
        gb = utils.make_gb_2d_simplex(mesh_args, points, edges, unit_domain)

    else:
        fracture0 = np.array([x_endpoints, [0.5, 0.5]])
        fracture1 = np.array([[0.5, 0.5], y_endpoints])
        gb = pp.meshing.cart_grid(
            [fracture0, fracture1],
            mesh_args,
            physdims=[unit_domain["xmax"], unit_domain["ymax"]],
        )
        gb.compute_geometry()
    return gb, unit_domain


@pp.time_logger(sections=module_sections)
def seven_fractures_one_L_intersection(mesh_args=None):
    """
    Create a grid bucket for a domain containing the network introduced as example 1 of
    Berge et al. 2019: Finite volume discretization for poroelastic media with fractures
    modeled by contact mechanics.

    Args:
        mesh_args: Dictionary containing at least "mesh_size_frac". If the optional
            values of "mesh_size_bound" and "mesh_size_min" are not provided, these are
            set by utils.set_mesh_sizes.

    Returns:
        Grid bucket for the domain.

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
    gb = utils.make_gb_2d_simplex(mesh_args, points, edges, domain)
    return gb, domain


@pp.time_logger(sections=module_sections)
def benchmark_regular(mesh_args, is_coarse=False):
    """
    Create a grid bucket for a domain containing the network introduced as example 2 of
    Berre et al. 2018: Benchmarks for single-phase flow in fractured porous media.

    Args:
        mesh_args: Dictionary containing at least "mesh_size_frac". If the optional
            values of "mesh_size_bound" and "mesh_size_min" are not provided, these are
            set by utils.set_mesh_sizes.
    Returns:
        Grid bucket for the domain.
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
    edges = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]).T
    gb = utils.make_gb_2d_simplex(mesh_args, points, edges, domain=unit_domain)
    if is_coarse:
        pp.coarsening.coarsen(gb, "by_volume")
    return gb, unit_domain
