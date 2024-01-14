"""Library of mixed-dimensional grids.

Mainly for use in tests. Other usage should be covered by the model_geometries.

"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, cast

import numpy as np

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d

from . import domains, fracture_sets


def square_with_orthogonal_fractures(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    meshing_args: dict[str, float],
    fracture_indices: list[int],
    fracture_endpoints: Optional[list[np.ndarray]] = None,
    size: pp.number = 1,
    **meshing_kwargs,
) -> tuple[pp.MixedDimensionalGrid, FractureNetwork2d]:
    """Create a mixed-dimensional grid for a square domain with up to two orthogonal
    fractures.

        meshing_args: Keyword arguments for meshing as used by
            :meth:`~porepy.grids.mdg_generation.create_mdg`.
        fracture_indices: Which fractures to include in the grid. Fracture i has
            constant i coordinates = size / 2.
        fracture_endpoints: List np.arrays containing the endpoints of the fractures.
            List item i contains the non-constant endpoints of fracture i, i.e. the
            first entry is the y coordinates of fracture one and the second entry is the
            x coordinate of fracture two. Should have the same length as
            fracture_indices. If not provided, the endpoints will be set to [0, 1].
        size: Side length of square.
        **meshing_kwargs: Keyword arguments for meshing as used by
            :meth:`~porepy.grids.mdg_generation.create_mdg`.

    Returns:
        Tuple containing a:

            :obj:`~pp.MixedDimensionalGrid`:
                Mixed-dimensional grid.

            :obj:`~pp.fracs.fracture_network_2d.FractureNetwork2d`:
                Fracture network. The fracture set is empty if fracture_indices == 0.

    """
    if fracture_endpoints is None:
        fracture_endpoints = []
    if len(fracture_endpoints) != 2:
        # Set default endpoints (0, size) for fractures if not provided
        all_endpoints = [np.array([0, size]), np.array([0, size])]

        for ind, endpoint in zip(fracture_indices, fracture_endpoints):
            all_endpoints[ind] = endpoint
        fracture_endpoints = all_endpoints

    all_fractures = fracture_sets.orthogonal_fractures_2d(size, fracture_endpoints)
    fractures = [all_fractures[i] for i in fracture_indices]
    domain = domains.nd_cube_domain(2, size)
    # Cast to FractureNetwork2d to avoid ambiguity leading to mypy errors
    fracture_network = cast(
        FractureNetwork2d, pp.create_fracture_network(fractures, domain)
    )
    mdg = pp.create_mdg(grid_type, meshing_args, fracture_network, **meshing_kwargs)
    return mdg, fracture_network


def cube_with_orthogonal_fractures(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    meshing_args: dict,
    fracture_indices: list[int],
    size: pp.number = 1,
    **meshing_kwargs,
) -> tuple[pp.MixedDimensionalGrid, FractureNetwork3d]:
    """Create a mixed-dimensional grid for a cube domain with up to three orthogonal
    fractures.

    Parameters:
        meshing_args: Keyword arguments for meshing as used by pp.create_mdg.
        fracture_indices: Which fractures to include in the grid. Fracture i has
            constant i coordinates = size / 2.
        size: Side length of cube.
        **meshing_kwargs: Keyword arguments for meshing as used by
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

    # Cast to FractureNetwork3d to avoid ambiguity leading to mypy errors
    fracture_network = cast(
        FractureNetwork3d, pp.create_fracture_network(fractures, domain)
    )
    mdg = pp.create_mdg(grid_type, meshing_args, fracture_network, **meshing_kwargs)
    return mdg, fracture_network


def seven_fractures_one_L_intersection(
    meshing_args: dict, **meshing_kwargs
) -> tuple[pp.MixedDimensionalGrid, FractureNetwork2d]:
    """
    Create a md-grid for a domain containing the network introduced as example 1 of
    Berge et al. 2019: Finite volume discretization for poroelastic media with fractures
    modeled by contact mechanics.

    Parameters:
        meshing_args: Dictionary containing at least "cell_size". If the optional
            values of "cell_size_boundary" and "mesh_size_min" are not provided, these
            are set by utils.set_mesh_sizes.
        **meshing_kwargs: Keyword arguments for meshing as used by
            :meth:`~porepy.grids.mdg_generation.create_mdg`.

    Returns:
        Mixed-dimensional grid and fracture network.

    """
    domain = pp.Domain(bounding_box={"xmin": 0, "ymin": 0, "xmax": 2, "ymax": 1})
    fractures = fracture_sets.seven_fractures_one_L_intersection()
    # Cast to FractureNetwork2d to avoid ambiguity leading to mypy errors
    fracture_network = cast(
        FractureNetwork2d, pp.create_fracture_network(fractures, domain)
    )
    mdg = pp.create_mdg("simplex", meshing_args, fracture_network, **meshing_kwargs)

    return mdg, fracture_network


def benchmark_regular_2d(
    meshing_args: dict, is_coarse: bool = False, **meshing_kwargs
) -> tuple[pp.MixedDimensionalGrid, FractureNetwork2d]:
    """
    Create a grid bucket for a domain containing the network introduced as example 2 of
    Berre et al. 2018: Benchmarks for single-phase flow in fractured porous media.

    Parameters:
        meshing_args: Dictionary containing at least "mesh_size_frac". If the optional
            values of "mesh_size_bound" and "mesh_size_min" are not provided, these are
            set by utils.set_mesh_sizes.
        is_coarse: If True, coarsen the grid by volume.
        **meshing_kwargs: Keyword arguments for meshing as used by
            :meth:`~porepy.grids.mdg_generation.create_mdg`.

    Returns:
        Mixed-dimensional grid and fracture network.

    """
    domain = domains.nd_cube_domain(2, 1)
    fractures = fracture_sets.benchmark_2d_case_1()
    # Cast to FractureNetwork2d to avoid ambiguity leading to mypy errors
    fracture_network = cast(
        FractureNetwork2d, pp.create_fracture_network(fractures, domain)
    )
    mdg = pp.create_mdg("simplex", meshing_args, fracture_network, **meshing_kwargs)

    if is_coarse:
        pp.coarsening.coarsen(mdg, "by_volume")
    return mdg, fracture_network


def benchmark_3d_case_3(
    refinement_level: Literal[0, 1, 2, 3] = 0,
) -> pp.MixedDimensionalGrid:
    """
    Create a mixed-dimensional grid for the geometry of case 3 from [1].

    Note:
        The mixed-dimensional grid is created by reading a `geo` file, so there is no
        direct way of prescribing meshing arguments.

    Reference:
        Berre, I., Boon, W. M., Flemisch, B., Fumagalli, A., Gläser, D., Keilegavlen,
        E., ... & Zulian, P. (2021). Verification benchmarks for single-phase flow in
        three-dimensional fractured porous media. Advances in Water Resources, 147,
        103759.

    Parameters:
        refinement_level: An integer denoting the level of refinement. Use `0` to
            generate a mixed-dimensional grid with approximately 30K 2D cells,
            `1` for 140K 2D cells, `2` for 350K cells, and `3` for 500K cells.
            Default is `0`.

    Returns:
        Mixed-dimensional grid.

    """
    # Get directory pointing to the `geo` file
    abs_path = Path(__file__)
    benchmark_path = abs_path.parent / Path("geo_library") / Path("benchmark_3d_case_3")
    if refinement_level == 0:
        full_path = benchmark_path / Path("mesh30k.geo")
    elif refinement_level == 1:
        full_path = benchmark_path / Path("mesh140k.geo")
    elif refinement_level == 2:
        full_path = benchmark_path / Path("mesh350k.geo")
    elif refinement_level == 3:
        full_path = benchmark_path / Path("mesh500k.geo")
    else:
        raise NotImplementedError("Refinement level not available.")

    # Create mixed-dimensional grid
    mdg = pp.fracture_importer.dfm_from_gmsh(str(full_path), dim=3)

    # In-situ weak test
    assert (mdg.dim_max() == 3) and (mdg.dim_min() == 1)
    assert (len(mdg.subdomains()) == 16) and (len(mdg.interfaces()) == 22)

    return mdg
