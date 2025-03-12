"""Test functions in :module:`porepy.fracs.msh_2_grid`."""

import copy
import pathlib

import numpy as np
import pytest

import porepy as pp
from porepy.fracs.msh_2_grid import (
    gmsh_element_types,
    tag_grid,
    create_0d_grids,
    create_1d_grids,
)
from porepy.fracs.simplex import _read_gmsh_file

rng: np.random.Generator = np.random.default_rng(seed=42)
dirname: pathlib.Path = pathlib.Path(__file__).parent

filename: str = str(
    dirname
    / ".."
    / ".."
    / "src"
    / "porepy"
    / "applications"
    / "md_grids"
    / "gmsh_file_library"
    / "benchmark_3d_case_3"
    / "mesh30k.msh"
)


@pytest.fixture
def simplex_grid(request: pytest.FixtureRequest) -> pp.Grid:
    # TODO Add 1D and 0D grids to test for grids created by
    # ``msh_2_grid.create_1d_grids`` and ``msh_2_grid.create_0d_grids``.
    dims: tuple = request.param
    bounding_box = {
        "xmin": 0,
        "xmax": request.param[0],
        "ymin": 0,
        "ymax": request.param[1],
    }
    if len(dims) == 3:
        bounding_box.update({"zmin": 0, "zmax": dims[2]})
    domain = pp.Domain(bounding_box)
    fn = pp.create_fracture_network([], domain)
    return pp.create_mdg("simplex", {"cell_size": 0.5}, fn).subdomains()[0]


@pytest.fixture
def cell_info_from_gmsh() -> dict[str, np.ndarray]:
    """Import a gmsh file and return the cell information, which has all possible gmsh
    element types as keys.

    """
    # TODO This is a temporary solution. The file should be in the test directory and
    # doesn't need to include 30k cells.
    pts, cells, cell_info, phys_names = _read_gmsh_file(filename)

    return cell_info


@pytest.fixture
def num_phys_names(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def phys_names(num_phys_names: int) -> dict[int, str]:
    return {i: f"phys_name_{i}" for i in range(num_phys_names)}


@pytest.fixture
def cell_info(num_phys_names: int, simplex_grid: pp.Grid) -> dict[str, np.ndarray]:
    dic: dict[str, np.ndarray] = {}
    for gmsh_element_type in gmsh_element_types:
        # Find the number of elements of the given type present in the grid.
        if gmsh_element_type == "point":
            num_elements: int = simplex_grid.num_nodes
        elif gmsh_element_type == "line":
            if simplex_grid.dim == 2:
                num_elements = simplex_grid.num_faces
            else:
                continue
        elif gmsh_element_type == "triangle":
            if simplex_grid.dim == 2:
                num_elements = simplex_grid.num_cells
            else:
                num_elements = simplex_grid.num_faces
        elif gmsh_element_type == "tetra":
            if simplex_grid.dim == 3:
                num_elements = simplex_grid.num_cells
            else:
                continue

        # Add random tags for the elements.
        dic[gmsh_element_type] = rng.integers(0, num_phys_names, num_elements)
    return dic


def test_gmsh_elements(cell_info_from_gmsh: dict[str, np.ndarray]) -> None:
    """Assert that the list of gmsh elements is exhaustive.

    TODO Might be removed after developing. This requires an gmsh file in the test
    directory which isn't super pretty. On the other hand, this acts as a reminder to
    extend ``gmsh_elements`` and update some functions in case the functionality in
    ``msh_2_grid`` is ever expanded to non-simplex grids.

    """
    gmsh_element_types_copy: list[str] = copy.copy(gmsh_element_types)
    for grid_element_type in cell_info_from_gmsh:
        gmsh_element_types_copy.remove(grid_element_type)
    assert len(gmsh_element_types_copy) == 0


@pytest.mark.parametrize("simplex_grid", [(2, 2), (2, 2, 2)], indirect=True)
@pytest.mark.parametrize("num_phys_names", [1, 5, 10], indirect=True)
def test_tag_grids(
    simplex_grid: pp.Grid,
    phys_names: dict[int, str],
    cell_info: dict[str, np.ndarray],
) -> None:
    """Assert that the tags are correctly assigned to the grid."""
    tagged_grid: pp.Grid = tag_grid(simplex_grid, phys_names, cell_info)
    for gmsh_element_type in cell_info:
        for phys_ind, phys_name in phys_names.items():
            assert np.all(
                tagged_grid.tags[f"{phys_name}_{gmsh_element_type}s"]
                == (cell_info[gmsh_element_type] == phys_ind)
            )


@pytest.fixture
def test_create_0d_grids() -> list[pp.PointGrid]:
    pts, cells, cell_info, phys_names = _read_gmsh_file(filename)

    g_0d = create_0d_grids(pts, cells, phys_names, cell_info)

    test_tag_grids(g_0d, phys_names, cell_info)


@pytest.fixture
def test_create_1d_grids() -> list[pp.Grid]:
    pts, cells, cell_info, phys_names = _read_gmsh_file(filename)

    g_1d = create_1d_grids(pts, cells, phys_names, cell_info)
    test_tag_grids(g_1d, phys_names, cell_info)
