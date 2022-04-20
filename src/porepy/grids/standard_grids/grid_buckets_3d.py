"""
This file contains utility functions for setting up grid buckets for 3d networks.
The default is simplex grids, but Cartesian grids are also possible for the simplest
geometries.
The provided geometries are:
        Simple unit square geometries
    single_horizontal: Single horizontal plane at z=0.5
"""

import numpy as np

import porepy as pp
import porepy.grids.standard_grids.utils as utils


def single_horizontal(mesh_args=None, x_coords=None, y_coords=None, simplex=True):
    """
    Create a grid bucket for a domain containing a horizontal rectangular fracture at z=0.5.

    Args:
        mesh_args:  For triangular grids: Dictionary containing at least "mesh_size_frac". If
                        the optional values of "mesh_size_bound" and "mesh_size_min" are
                        not provided, these are set by utils.set_mesh_sizes.
                    For cartesian grids: ndarray containing number of cells in x, y and z direction.                        direction.
        x_coords (list): Contains the two x coordinates of the four fracture corner points.
            If not, the coordinates will be set to [0, 1].
        y_coords (list): Contains the two y coordinates of the four fracture corner points.
            If not, the coordinates will be set to [0, 1].

    Returns:
        Grid bucket for the domain.

    """
    if x_coords is None:
        x_coords = [0, 1]
    if y_coords is None:
        y_coords = [0, 1]

    domain = utils.unit_domain(3)
    x_pts = [x_coords[0], x_coords[0], x_coords[1], x_coords[1]]
    y_pts = [y_coords[0], y_coords[1], y_coords[1], y_coords[0]]
    fracture = np.array([x_pts, y_pts, [0.5, 0.5, 0.5, 0.5]])
    if simplex:
        if mesh_args is None:
            mesh_args = {"mesh_size_frac": 0.2, "mesh_size_min": 0.2}
        network = pp.FractureNetwork3d([pp.Fracture(fracture)], domain=domain)
        gb = network.mesh(mesh_args)

    else:
        gb = pp.meshing.cart_grid([fracture], mesh_args, physdims=np.ones(3))
    return gb, domain
