"""
Module containing a worker function to generate mixed-dimensional grids in an unified
way.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional, Union

import numpy as np

import porepy as pp
import porepy.grids.standard_grids.utils as utils
from porepy.fracs.utils import linefractures_to_pts_edges, pts_edges_to_linefractures


def __simplex_2d_grid(
    fracture_network: pp.FractureNetwork2d, mesh_arguments: dict[str], **kwargs
) -> pp.MixedDimensionalGrid:
    utils.set_mesh_sizes(mesh_arguments)
    mdg = fracture_network.mesh(mesh_arguments)
    mdg.compute_geometry()

    return mdg


def __simplex_3d_grid(
    fracture_network: pp.FractureNetwork3d, mesh_arguments: dict[str], **kwargs
) -> pp.MixedDimensionalGrid:
    utils.set_mesh_sizes(mesh_arguments)
    mdg = fracture_network.mesh(mesh_arguments)
    mdg.compute_geometry()
    return mdg


def __coord_cart_2d(self, phys_dims, dev, pos):
    xmax = phys_dims[0]
    ymax = phys_dims[1]

    x = np.array(pos)
    y = np.array([dev, ymax - dev])
    return np.array([x, y])


def __cartersian_2d(
    fracture_network: pp.FractureNetwork2d, mesh_arguments: dict[str], **kwargs
):

    phys_dims = mesh_arguments["phys_dims"]
    n_cells = mesh_arguments["n_cells"]
    fractures = list(fracture_network.pts.T[fracture_network.edges.T])
    mdg = pp.meshing.cart_grid(
        fracs=fractures, physdims=phys_dims, nx=np.array(n_cells)
    )
    return mdg


def coord_cart_3d(self, phys_dims, dev, pos):
    xmax = phys_dims[0]
    ymax = phys_dims[1]
    zmax = phys_dims[2]

    z = np.array([dev, dev, zmax - dev, zmax - dev])

    if pos[1] == "x":
        x = np.ones(4) * pos[0]
        y = np.array([dev, ymax - dev, ymax - dev, dev])
    elif pos[1] == "y":
        x = np.array([dev, xmax - dev, xmax - dev, dev])
        y = np.ones(4) * pos[0]
    return np.array([x, y, z])


def cartersian_3d():

    # Generate mixed-dimensional mesh
    phys_dims = [50, 50, 10]
    n_cells = [20, 20, 10]
    bounding_box_points = np.array(
        [[0, phys_dims[0]], [0, phys_dims[1]], [0, phys_dims[2]]]
    )
    box = pp.geometry.bounding_box.from_points(bounding_box_points)

    frac1 = coord_cart_3d(phys_dims, 2, (25, "x"))
    frac2 = coord_cart_3d(phys_dims, 2, (25, "y"))
    frac3 = coord_cart_3d(phys_dims, 2, (23, "y"))
    frac4 = coord_cart_3d(phys_dims, 2, (27, "y"))
    mdg = pp.meshing.cart_grid(
        fracs=[frac1, frac2, frac3, frac4], physdims=phys_dims, nx=np.array(n_cells)
    )
    return mdg


def _validate_arguments(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    mesh_arguments: dict[str],
    **kwargs,
):
    validity_q = True

    return validity_q


def create_mdg(
    fracture_network: Union[pp.FractureNetwork2d, pp.FractureNetwork3d],
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    mesh_arguments: dict[str],
    **kwargs,
) -> pp.MixedDimensionalGrid:

    if not _validate_arguments(grid_type, mesh_arguments, **kwargs):
        raise (ValueError)

    # Assertion for FN type
    assert isinstance(fracture_network, pp.FractureNetwork2d) or isinstance(
        fracture_network, pp.FractureNetwork3d
    )

    # TODO: Collect examples for Tensor grids
    # 2d cases
    if isinstance(fracture_network, pp.FractureNetwork2d):
        if grid_type == "simplex":
            mdg = __simplex_2d_grid(fracture_network, mesh_arguments, **kwargs)
        elif grid_type == "cartesian":
            mdg = __cartersian_2d(fracture_network, mesh_arguments, **kwargs)

    # 3d cases
    if isinstance(fracture_network, pp.FractureNetwork3d):
        if grid_type == "simplex":
            mdg = __simplex_3d_grid(fracture_network, mesh_arguments, **kwargs)
        elif grid_type == "cartesian":
            mdg = cartersian_3d()

    return mdg
