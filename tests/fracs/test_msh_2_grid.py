"""Test functions in :module:`porepy.fracs.msh_2_grid`."""

import numpy as np
import pytest

import porepy as pp
from porepy.fracs.msh_2_grid import tag_grid

geo_elements: list[str] = [
    "point",
    "line",
    "triangle",
    "tetrahedron",
]
"""Different geometric elements in gmsh.

Additionally, gmsh supports quadrangles, hexahedra, and prisms. However, these cannot
occur in a simplex grid.

"""

rng: np.random.Generator = np.random.default_rng(seed=42)
"""Random number generator."""


@pytest.fixture
def simplex_grid(request: pytest.FixtureRequest) -> pp.Grid:
    # TODO Add 1D and 0D grids to test for grids created by
    # ``msh_2_grid.create_1d_grids`` and ``msh_2_grid.create_0d_grids``.
    dims: tuple = request.param
    if len(dims) == 2:
        return pp.TriangleGrid(dims=dims)
    elif len(dims) == 3:
        return pp.TetrahedralGrid(dims=dims)
    else:
        raise ValueError(f"Invalid dimensions: {dims}")


@pytest.fixture
def num_phys_names(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def phys_names(num_phys_names: int) -> dict[int, str]:
    return {i: f"phys_name_{i}" for i in range(num_phys_names)}


@pytest.fixture
def cell_info(num_phys_names: int, simplex_grid: pp.Grid) -> dict[str, np.ndarray]:
    dic: dict[str, np.ndarray] = {}
    for geo_element in geo_elements:
        # Find the number of elements of the given type present in the grid.
        if geo_element == "point":
            num_elements: int = simplex_grid.num_nodes
        elif geo_element == "line":
            if simplex_grid.dim == 2:
                num_elements = simplex_grid.num_faces
            else:
                continue
        elif geo_element == "triangle":
            if simplex_grid.dim == 2:
                num_elements = simplex_grid.num_cells
            else:
                num_elements = simplex_grid.num_faces
        elif geo_element == "tetrahedron" and simplex_grid.dim == 3:
            num_elements = simplex_grid.num_cells

        # Add random tags for the elements.
        dic[geo_element] = rng.integers(0, num_phys_names, num_elements)
    return dic


@pytest.mark.parametrize("simplex_grid", [(2, 2), (2, 2, 2)], indirect=True)
@pytest.mark.parametrize("num_phys_names", [0, 1, 10], indirect=True)
def test_tag_grids(
    simplex_grid: pp.Grid,
    phys_names: dict[int, str],
    cell_info: dict[str, np.ndarray],
) -> None:
    """Assert that the tags are correctly assigned to the grid."""
    tagged_grid: pp.Grid = tag_grid(simplex_grid, phys_names, cell_info)
    for geo_element in cell_info:
        for phys_ind, phys_name in phys_names.items():
            assert np.all(
                tagged_grid.tags[f"{phys_name}_{geo_element}"]
                == (cell_info[geo_element] == phys_ind)
            )
