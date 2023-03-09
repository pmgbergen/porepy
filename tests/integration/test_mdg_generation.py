"""

Collection of tests for validating the mixed-dimensional grid generation

Functionalities being tested:
* Generation with/without fractures
* Generation with/without wells (lines in 3d)
* Generation of meshes with type {"simplex", "cartesian", "tensor_grid"}
* Generation of meshes with dimension {2,3}

"""
from typing import List, Literal, Optional, Union

import numpy as np
import unittest
import pytest


import porepy as pp


# The collection of fractures
fracture_2d_data: List[np.array] = [
    np.array([[0.0, 2.0], [0.0, 0.0]]),
    np.array([[1.0, 1.0], [0.0, 1.0]]),
    np.array([[2.0, 2.0], [0.0, 2.0]]),
]
fracture_3d_data: List[np.array] = [
    np.array([[2.0, 3.0, 3.0, 2.0], [2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]]),
    np.array([[2.0, 3.0, 3.0, 2.0], [1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.0]]),
    np.array([[1.0, 4.0, 4.0, 1.0], [3.0, 3.0, 3.0, 3.0], [1.0, 1.0, 2.0, 2.0]]),
]

# The collection of wells
well_data: List[np.array] = [
    np.array([[0, 2, 2, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
    np.array([[0, 1, 1, 0], [0, 0, 1, 1], [-1, -1, 1, 1]]),
]

# The collection of types
mdg_types: List[str] = ["simplex", "cartesian", "tensor_grid"]

# The box domains being used
domain_2d = pp.Domain({"xmin": 0, "xmax": 5, "ymin": 0, "ymax": 5})
domain_3d = pp.Domain(
    {"xmin": 0, "xmax": 5, "ymin": 0, "ymax": 5, "zmin": 0, "zmax": 5}
)
domains: List[pp.Domain] = [domain_2d, domain_3d]

# Extra mesh arguments
simplex_extra_args: dict[str] = {"mesh_size_bound": 1.0, "mesh_size_min": 0.1}
cartesian_extra_args: dict[str] = {"n_cells": [5, 5], "phys_dims": [5, 5]}
cartesian_extra_args: dict[str] = {}
tensor_grid_extra_args: dict[str] = {"n_cells": [5, 5], "phys_dims": [5, 5]}
extra_args_data: List[dict[str]] = [
    simplex_extra_args,
    cartesian_extra_args,
    tensor_grid_extra_args,
]


def _generate_wells(well_indices: List[int]):
    """Construct well network.

    Parameters:
        well_indices (list): combination of wells

    Returns:
        pp.WellNetwork3d: collection of Wells with geometrical information
    """


def _generate_network(domain: pp.Domain, fracture_indices: List[int]):
    """Construct well network.

    Parameters:
        domain (pp.Domain): geometry representation of the higher dimensional geometrical entity
        fracture_indices (list): combination of fractures

    Returns:
        Union[pp.FractureNetwork2d, pp.FractureNetwork3d]: Collection of Fractures with
        geometrical information
    """

    disjoint_fractures = None
    if domain.dim == 2:
        geometry = [fracture_2d_data[id] for id in fracture_indices]
        disjoint_fractures = list(map(pp.LineFracture, geometry))
    elif domain.dim == 3:
        geometry = [fracture_3d_data[id] for id in fracture_indices]
        disjoint_fractures = list(map(pp.PlaneFracture, geometry))

    network = pp.create_fracture_network(disjoint_fractures, domain)
    return network


def _high_level_mdg_generation(grid_type, fracture_network, well_network):
    """Generates a mixed-dimensional grid.

    Parameters:
        fracture_network Union[pp.FractureNetwork2d, pp.FractureNetwork3d]: selected
        pp.FractureNetwork<n>d with n in {2,3}
        well_network pp.WellNetwork3d: selected pp.WellNetwork3d.

    Returns:
        pp.MixedDimensionalGrid: Container of grids and its topological relationship
         along with a surrounding matrix defined by domain.
    """
    # common mesh argument
    mesh_arguments: dict[str] = {"mesh_size": 0.5}

    extra_arg_index = mdg_types.index(grid_type)
    extra_arguments = extra_args_data[extra_arg_index]
    mdg = pp.create_mdg(grid_type, mesh_arguments, fracture_network, **extra_arguments)
    return mdg


test_parameters = [
    ("simplex", domains[0], []),
    ("simplex", domains[0], [0, 1, 2]),
    ("simplex", domains[1], [0, 1, 2]),
    ("cartesian", domains[0], []),
    ("cartesian", domains[0], [0, 1, 2]),
    ("cartesian", domains[1], [0, 1, 2]),
    ("tensor_grid", domains[0], []),
    ("tensor_grid", domains[0], [0, 1, 2]),
    ("tensor_grid", domains[1], [0, 1, 2]),
]


@pytest.mark.parametrize("grid_type, domain, fracture_indices", test_parameters)
def test_generation(grid_type, domain, fracture_indices) -> None:
    """Test the geneated mdg object."""
    fracture_network = _generate_network(domain, fracture_indices)
    mdg = _high_level_mdg_generation(grid_type, fracture_network, None)
    pp.Exporter(mdg, "generated_mdg").write_vtu()

    # For now testing the code complete the execution
    tautology = True
    assert tautology


if __name__ == "__main__":
    unittest.main()
