"""This package is primarily envisioned as a library of mixed-dimensional grids
for use in tests. Other usage should be covered by the model_geometries.
TODO: Decide whether to keep or make mdgs on demand in the tests using domains,
fracture_sets and create_mdg.

"""

from __future__ import annotations

from typing import Literal, Tuple

import porepy as pp

from . import domains, fracture_sets


def square_with_orthogonal_fractures(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    meshing_args: dict,
    fracture_indices: list[int],
    size: pp.number,
    **meshing_kwargs,
) -> Tuple[pp.MixedDimensionalGrid, pp.Domain, pp.FractureNetwork2d]:
    """Create a mixed-dimensional grid for a square domain with up to two
    orthogonal fractures.

    Parameters:
        meshing_args: Keyword arguments for meshing as used by pp.create_mdg.
        fracture_indices: Which fractures to include in the grid. Fracture i has
            constant i coordinates = size / 2.
        size: Side length of square.
        **kwargs: Keyword arguments for meshing as used by pp.create_mdg.

    Returns:
        Tuple containing a:

            :obj:`~pp.MixedDimensionalGrid`:
                Mixed-dimensional grid.

            :obj:`~pp.Domain`:
                Domain object.

            :obj:`~pp.FractureNetwork2d`:
                Fracture network. The fracture set is empty if fracture_indices == 0.

    """
    all_fractures = fracture_sets.orthogonal_fractures_2d(size)
    fractures = [all_fractures[i] for i in fracture_indices]
    domain = domains.nd_cube_domain(2, size)
    fracture_network = pp.create_fracture_network(fractures, domain)
    mdg = pp.create_mdg(grid_type, meshing_args, fracture_network, **meshing_kwargs)
    return mdg, domain, fracture_network


def cube_with_orthogonal_fractures(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    meshing_args: dict,
    fracture_indices: list[int],
    size: pp.number,
    **meshing_kwargs,
) -> Tuple[pp.MixedDimensionalGrid, pp.Domain, pp.FractureNetwork3d]:
    """Create a mixed-dimensional grid for a cube domain with up to three
    orthogonal fractures.

    Parameters:
        meshing_args: Keyword arguments for meshing as used by pp.create_mdg.
        fracture_indices: Which fractures to include in the grid. Fracture i has
            constant i coordinates = size / 2.
        size: Side length of cube.
        **kwargs: Keyword arguments for meshing as used by pp.create_mdg.

    Returns:
        Tuple containing a:

            :obj:`~pp.MixedDimensionalGrid`:
                Mixed-dimensional grid.

            :obj:`~pp.Domain`:
                Domain object.

            :obj:`~pp.FractureNetwork3d`:
                Fracture network. The fracture set is empty if fracture_indices == 0.

    """
    all_fractures = fracture_sets.orthogonal_fractures_3d(size)
    fractures = [all_fractures[i] for i in fracture_indices]
    domain = domains.nd_cube_domain(3, size)
    fracture_network = pp.FractureNetwork3d(fractures, domain)
    mdg = pp.create_mdg(grid_type, meshing_args, fracture_network, **meshing_kwargs)
    return mdg, domain, fracture_network
