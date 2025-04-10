"""Test functions in :module:`porepy.fracs.msh_2_grid`."""

import copy
import pathlib

import numpy as np
import pytest

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.fracs.msh_2_grid import (
    create_0d_grids,
    create_1d_grids,
    create_2d_grids,
    create_3d_grids,
    gmsh_element_types,
    tag_grid,
)
from porepy.fracs.simplex import _read_gmsh_file
from porepy.grids.mdg_generation import _preprocess_simplex_args

rng: np.random.Generator = np.random.default_rng(seed=42)
dirname: pathlib.Path = pathlib.Path(__file__).parent


@pytest.fixture
def dims(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture
def num_phys_names(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def phys_names(num_phys_names: int) -> dict[int, str]:
    return {i: f"phys_name_{i}" for i in range(num_phys_names)}


@pytest.fixture
def fracture_network(dims: tuple) -> pp.fracture_network:
    if len(dims) == 2:
        bounding_box = {
            "xmin": 0,
            "xmax": dims[0],
            "ymin": 0,
            "ymax": dims[1],
        }
        # Two intersecting line fractures
        fractures = [
            pp.LineFracture(np.array([[0, dims[0]], [dims[1] / 2, dims[1] / 2]])),
            pp.LineFracture(np.array([[dims[0] / 2, dims[0] / 2], [0, dims[1]]])),
        ]

    if len(dims) == 3:
        bounding_box = {
            "xmin": 0,
            "xmax": dims[0],
            "ymin": 0,
            "ymax": dims[1],
            "zmin": 0,
            "zmax": dims[2],
        }
        # Three intersecting plane fractures creating both a line fracture intersection
        # and a point fracture intersection.
        fractures = [
            pp.PlaneFracture(
                np.array(
                    [
                        [0, dims[0], dims[0], 0],
                        [0, 0, dims[1], dims[1]],
                        [dims[2] / 2, dims[2] / 2, dims[2] / 2, dims[2] / 2],
                    ]
                ),
            ),
            pp.PlaneFracture(
                np.array(
                    [
                        [0, dims[0], dims[0], 0],
                        [dims[1] / 2, dims[1] / 2, dims[1] / 2, dims[1] / 2],
                        [0, 0, dims[2], dims[2]],
                    ]
                ),
            ),
            pp.PlaneFracture(
                np.array(
                    [
                        [dims[0] / 2, dims[0] / 2, dims[0] / 2, dims[0] / 2],
                        [0, dims[1], dims[1], 0],
                        [0, 0, dims[2], dims[2]],
                    ]
                ),
            ),
        ]
    domain = pp.Domain(bounding_box)
    return pp.create_fracture_network(fractures, domain)


@pytest.fixture
def simplex_grids(fracture_network: pp.fracture_network) -> list[pp.Grid]:
    mdg = pp.create_mdg("simplex", {"cell_size": 0.5}, fracture_network)
    return mdg.subdomains()


@pytest.fixture
def create_gmsh_file(
    fracture_network: pp.fracture_network, num_phys_names: int, tmp_path: pathlib.Path
) -> str:
    """Create a gmsh file with the given number of physical names."""
    msh_file: pathlib.Path = tmp_path / "test.msh"

    # Use functionality from pp.create_mdg and fracture_network.mesh to create a gmsh
    # file from the fracture_network.

    # The following are going to get shifted to extra_args by _preprocess_simplex_args.
    kwargs: dict = {"file_name": str(msh_file), "write_geo": False}

    # Without the if-else construction, _prepare_simplex_args fails.
    if isinstance(fracture_network, FractureNetwork2d):
        lower_level_args, extra_args, kwargs = _preprocess_simplex_args(
            {"cell_size": 0.5}, kwargs, FractureNetwork2d.mesh
        )
    elif isinstance(fracture_network, FractureNetwork3d):
        lower_level_args, extra_args, kwargs = _preprocess_simplex_args(
            {"cell_size": 0.5}, kwargs, FractureNetwork3d.mesh
        )

    fracture_network.mesh(
        lower_level_args,
        *extra_args,
        **kwargs,
    )

    # Add physical names.
    # TODO Didn't manage this yet, it's tricky. Basically, we want to loop through all
    # lines in the .msh file and assign different physical names to some of the cells,
    # faces, lines, and vertices.
    # TODO Also this takesa lot of time.
    # with pathlib.Path(msh_file).open("r+") as f:
    # data: list[str] = f.readlines()
    # data_cp = []
    # for line in data:
    #     data_cp.append(line)
    #     while line != "$PhysicalNames\n":
    #         continue

    # prev_num_phys_names = int(data[4][:-1])

    # # Add physical names.
    # for i in range(num_phys_names):
    #     data.insert(5, f'0 1 "phys_name_{prev_num_phys_names + i}"\n')
    # # Update the number of physical names.
    # data[4] = f"{prev_num_phys_names + num_phys_names}\n"

    # # Assign physical names to entities.
    # for i in range(5, len(data)):
    #     pass

    # f.writelines(data)

    return str(msh_file)


@pytest.fixture
def cell_info_from_gmsh(create_gmsh_file: pathlib.Path) -> dict[str, np.ndarray]:
    """Import a gmsh file and return the cell information, which has all possible gmsh
    element types as keys.

    """
    # TODO This is a temporary solution. The file should be in the test directory and
    # doesn't need to include 30k cells.
    _, __, cell_info, ___ = _read_gmsh_file(str(create_gmsh_file))
    return cell_info


@pytest.fixture
def cell_info_from_mdg(
    num_phys_names: int, simplex_grids: list[pp.Grid]
) -> dict[str, np.ndarray]:
    """Create a random cell_info dictionary for the given grid that assigns physical
    names to each grid element.

    """
    simplex_grid = simplex_grids[0]
    dic: dict[str, np.ndarray] = {}
    for gmsh_element_type in gmsh_element_types:
        # Find the number of elements of the given type present in the grid.
        if gmsh_element_type == "vertex":
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
        if num_phys_names > 0:
            dic[gmsh_element_type] = rng.integers(0, num_phys_names, num_elements)
        else:
            continue
    return dic


@pytest.mark.parametrize(
    "create_function, expected_grid_type",
    [
        (create_0d_grids, pp.PointGrid),
        (create_1d_grids, pp.Grid),
        (create_2d_grids, pp.Grid),
        (create_3d_grids, pp.Grid),
    ],
)
@pytest.mark.parametrize("dims", [(2, 2), (2, 2, 2)], indirect=True)
@pytest.mark.parametrize("num_phys_names", [0, 1, 5], indirect=True)
def test_create_grids(
    create_function,
    expected_grid_type,
    create_gmsh_file: str,
    dims,
    num_phys_names,
) -> None:
    """Test that create_nd_grids functions produce grids with correct tags."""
    pts, cells, cell_info, phys_names = _read_gmsh_file(create_gmsh_file)

    grids = create_function(pts, cells, phys_names, cell_info)

    if isinstance(grids, tuple):
        grids, _ = grids

    for g in grids:
        assert isinstance(g, expected_grid_type)
        for gmsh_element_type in cell_info:
            for tag in np.unique(cell_info[gmsh_element_type]):
                phys_name = phys_names[tag].lower()
                assert np.all(
                    g.tags[f"{phys_name}_{gmsh_element_type}s"]
                    == (cell_info[gmsh_element_type] == tag)
                )


@pytest.mark.parametrize(
    "num_phys_names", [1], indirect=True
)  # This parameter is needed to create the gmsh file.
@pytest.mark.parametrize("dims", [(2, 2, 2)], indirect=True)
def test_gmsh_elements(
    cell_info_from_gmsh: dict[str, np.ndarray],
    dims,
    num_phys_names,
) -> None:
    """Test whether ``msh_2_grid.gmsh_element_types`` is exhaustive, i.e., whether it
    contains all element types that can possibly be present in a gmsh file relevant for
    PorePy.

    TODO Might be removed after developing. This requires an gmsh file in the test
    directory which isn't super pretty. On the other hand, this acts as a reminder to
    extend ``gmsh_elements`` and update some functions in case the functionality in
    ``msh_2_grid`` is ever expanded to non-simplex grids.

    """
    gmsh_element_types_copy: list[str] = copy.copy(gmsh_element_types)

    for grid_element_type in cell_info_from_gmsh:
        gmsh_element_types_copy.remove(grid_element_type)
    assert len(gmsh_element_types_copy) == 0


@pytest.mark.parametrize("dims", [(2, 2), (2, 2, 2)], indirect=True)
@pytest.mark.parametrize("num_phys_names", [0, 1, 5, 10], indirect=True)
def test_tag_grids(
    simplex_grids: list[pp.Grid],
    phys_names: dict[int, str],
    cell_info_from_mdg: dict[str, np.ndarray],
    dims,
    num_phys_names,
) -> None:
    """Test that ``tag_grids`` correctly assigns tags to the grid.

    TODO: Check whether the unused parameters are needed. If not, remove them."""
    for grid in simplex_grids:
        tagged_grid: pp.Grid = tag_grid(grid, phys_names, cell_info_from_mdg)
        for gmsh_element_type in cell_info_from_mdg:
            for tag, phys_name in phys_names.items():
                assert np.all(
                    tagged_grid.tags[f"{phys_name}_{gmsh_element_type}s"]
                    == (cell_info_from_mdg[gmsh_element_type] == tag)
                )
