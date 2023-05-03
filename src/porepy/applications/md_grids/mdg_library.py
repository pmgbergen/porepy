"""Library of mixed-dimensional grids.

Mainly for use in tests. Other usage should be covered by the model_geometries.

"""

from __future__ import annotations

from typing import Literal

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d

from . import domains, fracture_sets


def square_with_orthogonal_fractures(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    meshing_args: dict[str, float],
    fracture_indices: list[int],
    size: pp.number,
    **meshing_kwargs,
) -> tuple[pp.MixedDimensionalGrid, FractureNetwork2d]:
    """Create a mixed-dimensional grid for a square domain with up to two
    orthogonal fractures.

        meshing_args: Keyword arguments for meshing as used by
            :meth:`~porepy.grids.mdg_generation.create_mdg`.
        fracture_indices: Which fractures to include in the grid. Fracture i has
            constant i coordinates = size / 2.
        **meshing_kwargs: Keyword arguments for meshing as used by
            :meth:`~porepy.grids.mdg_generation.create_mdg`.

    Returns:
        Tuple containing a:

            :obj:`~pp.MixedDimensionalGrid`:
                Mixed-dimensional grid.

            :obj:`~pp.FractureNetwork2d`:
                Fracture network. The fracture set is empty if fracture_indices == 0.

    """
    all_fractures = fracture_sets.orthogonal_fractures_2d(size)
    fractures = [all_fractures[i] for i in fracture_indices]
    domain = domains.nd_cube_domain(2, size)
    fracture_network = pp.create_fracture_network(fractures, domain)
    assert isinstance(fracture_network, FractureNetwork2d)  # for mypy
    mdg = pp.create_mdg(grid_type, meshing_args, fracture_network, **meshing_kwargs)
    return mdg, fracture_network


def cube_with_orthogonal_fractures(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    meshing_args: dict,
    fracture_indices: list[int],
    size: pp.number,
    **meshing_kwargs,
) -> tuple[pp.MixedDimensionalGrid, FractureNetwork3d]:
    """Create a mixed-dimensional grid for a cube domain with up to three
    orthogonal fractures.

    Parameters:
        meshing_args: Keyword arguments for meshing as used by pp.create_mdg.
        fracture_indices: Which fractures to include in the grid. Fracture i has
            constant i coordinates = size / 2.
        size: Side length of cube.
        **kwargs: Keyword arguments for meshing as used by
            :meth:`~porepy.grids.mdg_generation.create_mdg`.

    Returns:
        Tuple containing a:

            :obj:`~pp.MixedDimensionalGrid`:
                Mixed-dimensional grid.

            :obj:`~pp.FractureNetwork3d`:
                Fracture network. The fracture set is empty if fracture_indices == 0.

    """
    all_fractures = fracture_sets.orthogonal_fractures_3d(size)
    fractures = [all_fractures[i] for i in fracture_indices]
    domain = domains.nd_cube_domain(3, size)
    fracture_network = pp.create_fracture_network(fractures, domain)
    assert isinstance(fracture_network, FractureNetwork3d)  # for mypy
    mdg = pp.create_mdg(grid_type, meshing_args, fracture_network, **meshing_kwargs)
    return mdg, fracture_network
