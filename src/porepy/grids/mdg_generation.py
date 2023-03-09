"""
Module containing a function to generate mixed-dimensional grids. It encapsulates
different lower-level md-grid generation.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional, Union

import numpy as np

import porepy as pp
import porepy.grids.standard_grids.utils as utils


def _validate_arguments(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    mesh_arguments: dict[str],
    **kwargs,
):
    validity_q = True

    return validity_q


def create_mdg(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    mesh_arguments: dict[str],
    fracture_network: Optional[
        Union[pp.FractureNetwork2d, pp.FractureNetwork3d]
    ] = None,
    **kwargs,
) -> pp.MixedDimensionalGrid:

    if not _validate_arguments(grid_type, mesh_arguments, **kwargs):
        raise (ValueError)

    # Assertion for FN type
    assert (
        isinstance(fracture_network, pp.FractureNetwork2d)
        or isinstance(fracture_network, pp.FractureNetwork3d)
        or isinstance(fracture_network, None)
    )

    # lower level mdg generation
    # TODO: Collect examples for Tensor grids
    # 2d cases
    if isinstance(fracture_network, pp.FractureNetwork2d):
        if grid_type == "simplex":
            assert type(mesh_arguments) == type(kwargs)
            # Equivalence: expected mesh_size
            h_size = mesh_arguments["mesh_size"]
            lower_level_arguments = {}
            lower_level_arguments["mesh_size_frac"] = h_size
            lower_level_arguments.update(kwargs)
            utils.set_mesh_sizes(lower_level_arguments)
            mdg = fracture_network.mesh(lower_level_arguments)
            mdg.compute_geometry()
        elif grid_type == "cartesian":
            # mesh_size
            xmin = fracture_network.domain.bounding_box["xmin"]
            xmax = fracture_network.domain.bounding_box["xmax"]
            ymin = fracture_network.domain.bounding_box["ymin"]
            ymax = fracture_network.domain.bounding_box["ymax"]

            h_size = mesh_arguments["mesh_size"]
            n_x = round((xmax - xmin) / h_size)
            n_y = round((ymax - ymin) / h_size)

            fractures = [f.pts for f in fracture_network.fractures]
            mdg = pp.meshing.cart_grid(
                fracs=fractures, physdims=[ymax, xmax], nx=np.array([n_x, n_y])
            )

        elif grid_type == "tensor_grid":

            # mesh_size
            xmin = fracture_network.domain.bounding_box["xmin"]
            xmax = fracture_network.domain.bounding_box["xmax"]
            ymin = fracture_network.domain.bounding_box["ymin"]
            ymax = fracture_network.domain.bounding_box["ymax"]

            h_size = mesh_arguments["mesh_size"]
            n_x = round((xmax - xmin) / h_size) + 1
            n_y = round((ymax - ymin) / h_size) + 1
            x_space = np.linspace(xmin, xmax, num=n_x)
            y_space = np.linspace(ymin, ymax, num=n_y)

            fractures = [f.pts for f in fracture_network.fractures]
            # fracs: list[np.ndarray], x: np.ndarray, y = None, z = None, ** kwargs
            mdg = pp.meshing.tensor_grid(fracs=fractures, x=x_space, y=y_space)

    # 3d cases
    if isinstance(fracture_network, pp.FractureNetwork3d):
        if grid_type == "simplex":
            # Equivalence: expected mesh_size
            h_size = mesh_arguments["mesh_size"]
            lower_level_arguments = {}
            lower_level_arguments["mesh_size_frac"] = h_size
            lower_level_arguments.update(kwargs)
            utils.set_mesh_sizes(lower_level_arguments)
            mdg = fracture_network.mesh(lower_level_arguments)
            mdg.compute_geometry()
        elif grid_type == "cartesian":

            # mesh_size
            xmin = fracture_network.domain.bounding_box["xmin"]
            xmax = fracture_network.domain.bounding_box["xmax"]
            ymin = fracture_network.domain.bounding_box["ymin"]
            ymax = fracture_network.domain.bounding_box["ymax"]
            zmin = fracture_network.domain.bounding_box["zmin"]
            zmax = fracture_network.domain.bounding_box["zmax"]

            h_size = mesh_arguments["mesh_size"]
            n_x = round((xmax - xmin) / h_size)
            n_y = round((ymax - ymin) / h_size)
            n_z = round((zmax - zmin) / h_size)

            fractures = [f.pts for f in fracture_network.fractures]
            mdg = pp.meshing.cart_grid(
                fracs=fractures,
                physdims=[ymax, xmax, zmax],
                nx=np.array([n_x, n_y, n_z]),
            )

        elif grid_type == "tensor_grid":

            # mesh_size
            xmin = fracture_network.domain.bounding_box["xmin"]
            xmax = fracture_network.domain.bounding_box["xmax"]
            ymin = fracture_network.domain.bounding_box["ymin"]
            ymax = fracture_network.domain.bounding_box["ymax"]
            zmin = fracture_network.domain.bounding_box["zmin"]
            zmax = fracture_network.domain.bounding_box["zmax"]

            h_size = mesh_arguments["mesh_size"]
            n_x = round((xmax - xmin) / h_size) + 1
            n_y = round((ymax - ymin) / h_size) + 1
            n_z = round((zmax - zmin) / h_size) + 1

            x_space = np.linspace(xmin, xmax, num=n_x)
            y_space = np.linspace(ymin, ymax, num=n_y)
            z_space = np.linspace(zmin, zmax, num=n_y)

            fractures = [f.pts for f in fracture_network.fractures]
            mdg = pp.meshing.tensor_grid(
                fracs=fractures, x=x_space, y=y_space, z=z_space
            )

    return mdg
