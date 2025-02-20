"""Library of mixed-dimensional grids.

Mainly for use in tests. Other usage should be covered by the model_geometries.

"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, cast, Union

import numpy as np

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.grids import mortar_grid


from . import domains, fracture_sets


def square_with_orthogonal_fractures(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    meshing_args: dict[str, float],
    fracture_indices: list[int],
    fracture_endpoints: Optional[list[np.ndarray]] = None,
    size: pp.number = 1,
    non_matching: bool = False,
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

    if non_matching:
        old_fracture_grids = mdg.subdomains(dim=1)
        fracture_refinement_ratio = meshing_kwargs.get(
            "fracture_refinement_ratio", 2 * np.ones(len(old_fracture_grids))
        )
        if isinstance(fracture_refinement_ratio, (float, int)):
            fracture_refinement_ratio = (
                np.ones(len(old_fracture_grids)) * fracture_refinement_ratio
            )
        interface_refinement_ratio = meshing_kwargs.get(
            "interface_refinement_ratio", 2 * np.ones(len(mdg.interfaces(dim=1)))
        )
        if isinstance(interface_refinement_ratio, (float, int)):
            interface_refinement_ratio = (
                np.ones(len(mdg.interfaces(dim=1))) * interface_refinement_ratio
            )

        if grid_type == "simplex":
            meshing_args["mesh_size_frac"] = (
                meshing_args["mesh_size_frac"] / fracture_refinement_ratio
                # fracture_refinement_ratio ought to be a scalar here.
            )
            # The interface mesh size is set to the same as the fracture mesh size with
            # a simplex grid.
        elif grid_type == "cartesian":
            for key in ["cell_size", "cell_size_x", "cell_size_y"]:
                if key in meshing_args:
                    meshing_args[key] = meshing_args[key] / interface_refinement_ratio
        elif grid_type == "tensor_grid":
            raise NotImplementedError("Tensor grid not implemented yet.")

        # Refine the fracture grids
        new_fracture_grids = [
            pp.refinement.refine_grid_1d(g=old_grid, ratio=ratio)
            for old_grid, ratio in zip(old_fracture_grids, fracture_refinement_ratio)
        ]
        # Create a mapping between the old and new fracture grids.
        grid_map = dict(zip(old_fracture_grids, new_fracture_grids))

        # Refine the interface grids
        mdg_new = square_with_orthogonal_fractures(
            grid_type,
            meshing_args,
            fracture_indices,
            fracture_endpoints,
            size,
            non_matching=False,
        )
        intf_map = {}
        for intf in mdg_new.interfaces(dim=1):
            # Then, for each interface, we fetch the secondary grid which belongs to it.
            _, g_sec = mdg_new.interface_to_subdomain_pair(intf)

            # We then loop through all the interfaces in the original grid (recipient).
            for intf_coarse in mdg.interfaces(dim=1):
                # Fetch the secondary grid of the interface in the original grid.
                _, g_sec_coarse = mdg.interface_to_subdomain_pair(intf_coarse)

                # Checking the fracture number of the secondary grid in the recipient
                # mdg. If they are the same, i.e., if the fractures are the same ones,
                # we update the interface map.
                if g_sec_coarse.frac_num == g_sec.frac_num:
                    intf_map.update({intf_coarse: intf})

        # Finally replace the subdomains and interfaces in the original
        # mixed-dimensional grid.
        mdg.replace_subdomains_and_interfaces(
            sd_map=grid_map,
            intf_map=cast(
                dict[
                    pp.MortarGrid,
                    Union[pp.MortarGrid, dict[mortar_grid.MortarSides, pp.Grid]],
                ],
                intf_map,
            ),
        )

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
        Tuple containing a:

            :obj:`~pp.MixedDimensionalGrid`:
                The grid, constructed from the meshing arguments.

            :obj:`~pp.FractureNetwork2d`:
                The fracture network.

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
        Tuple containing a:

            :obj:`~pp.MixedDimensionalGrid`:
                The grid, constructed from the meshing arguments.

            :obj:`~pp.FractureNetwork2d`:
                The fracture network.

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
) -> tuple[pp.MixedDimensionalGrid, FractureNetwork3d]:
    """
    Create a mixed-dimensional grid for the geometry of case 3 from [1].

    Note:
        The mixed-dimensional grid is created by reading a `geo` file, so there is no
        direct way of prescribing meshing arguments.

    Reference:
        [1] Berre, I., Boon, W. M., Flemisch, B., Fumagalli, A., Gläser, D., Keilegavlen,
        E., ... & Zulian, P. (2021). Verification benchmarks for single-phase flow in
        three-dimensional fractured porous media. Advances in Water Resources, 147,
        103759.

    Parameters:
        refinement_level: An integer denoting the level of refinement. Use `0` to
            generate a mixed-dimensional grid with approximately 30K 3D cells,
            `1` for 140K 3D cells, `2` for 350K 3D cells, and `3` for 500K 3D cells.
            Default is `0`.

    Returns:
        Tuple containing a:

            :obj:`~pp.MixedDimensionalGrid`:
                The grid for the specified refinement level.

            :obj:`~pp.FractureNetwork3d`:
                The fracture network.

    """
    # Sanity check on input argument
    if refinement_level not in [0, 1, 2, 3]:
        raise NotImplementedError("Refinement level not available.")

    # Get directory pointing to the `geo` file
    abs_path = Path(__file__)
    benchmark_path = (
        abs_path.parent / Path("gmsh_file_library") / Path("benchmark_3d_case_3")
    )
    num_cells = [30, 140, 350, 500][refinement_level]
    full_path = benchmark_path / Path(f"mesh{num_cells}k.geo")

    # Set file permissions. This turned out to be important for GH actions.
    full_path.chmod(777)

    # Create mixed-dimensional grid
    mdg = pp.fracture_importer.dfm_from_gmsh(str(full_path), dim=3)

    # Also import fracture network
    fracture_network_path = benchmark_path / Path("fracture_network.csv")
    # Set file permissions. This turned out to be important for GH actions.
    fracture_network_path.chmod(777)

    network = pp.fracture_importer.network_3d_from_csv(str(fracture_network_path))

    return mdg, network
