"""Test of meshing of 2d and 3d domains with fractures, using structured grids."""
import numpy as np
import pytest

import porepy as pp
from porepy.fracs import structured


@pytest.mark.parametrize(
    "data",
    [
        {  # 2d domain, no fractures
            "f_set": [],
            "expected_num_grids": [0, 0, 1],
            "expected_num_cells": [[0], [0], [25]],
            "dim": 2,
        },
        {  # 2d domain, X-intersection
            "f_set": [np.array([[2, 4], [3, 3]]), np.array([[3, 3], [2, 4]])],
            "expected_num_grids": [1, 2, 1],
            "expected_num_cells": [[1], [2, 2], [25]],
            "dim": 2,
        },
        {  # 2d domain, single fracture
            "f_set": [np.array([[2, 4], [3, 3]])],
            "expected_num_grids": [2, 1, 1],
            "expected_num_cells": [[0], [2], [25]],
            "dim": 2,
        },
        {  # 2d domain, T-intersection
            "f_set": [np.array([[2, 4], [3, 3]]), np.array([[3, 3], [5, 4]])],
            "expected_num_grids": [1, 2, 1],
            "expected_num_cells": [[1], [2, 1], [25]],
            "dim": 2,
        },
        {  # 2d domain, L-intersection
            "f_set": [np.array([[2, 4], [3, 3]]), np.array([[4, 4], [3, 4]])],
            "expected_num_grids": [1, 2, 1],
            "expected_num_cells": [[1], [2, 1], [25]],
            "dim": 2,
        },
        {  # 3d domain, no fractures
            "f_set": [],
            "expected_num_grids": [0, 0, 0, 1],
            "expected_num_cells": [[], [], [], [125]],
            "dim": 3,
        },
        {  # 3d domain, single fracture
            "f_set": [np.array([[2, 4, 4, 2], [2, 2, 4, 4], [3, 3, 3, 3]])],
            "expected_num_grids": [0, 0, 1, 1],
            "expected_num_cells": [[], [], [4], [125]],
            "dim": 3,
        },
        {  # 3d domain, two intersecting fractures
            "f_set": [
                np.array([[2, 4, 4, 2], [2, 2, 4, 4], [3, 3, 3, 3]]),
                np.array([[2, 4, 4, 2], [3, 3, 3, 3], [2, 2, 4, 4]]),
            ],
            "expected_num_grids": [0, 1, 2, 1],
            "expected_num_cells": [[], [2], [4, 4], [125]],
            "dim": 3,
        },
        {  # 3d domain, three intersecting fractures
            "f_set": [
                np.array([[2, 4, 4, 2], [2, 2, 4, 4], [3, 3, 3, 3]]),
                np.array([[2, 4, 4, 2], [3, 3, 3, 3], [2, 2, 4, 4]]),
                np.array([[3, 3, 3, 3], [2, 4, 4, 2], [2, 2, 4, 4]]),
            ],
            "expected_num_grids": [1, 6, 3, 1],
            # There are three intersection lines, but each of them will be split in two
            "expected_num_cells": [[1], [1, 1, 1, 1, 1, 1], [4, 4, 4], [125]],
            "dim": 3,
        },
        {  # 3d domain, two intersecting fractures, T-intersection
            "f_set": [
                np.array([[2, 4, 4, 2], [2, 2, 4, 4], [3, 3, 3, 3]]),
                np.array([[2, 4, 4, 2], [3, 3, 3, 3], [3, 3, 4, 4]]),
            ],
            "expected_num_grids": [0, 1, 2, 1],
            "expected_num_cells": [[], [2], [4, 2], [125]],
            "dim": 3,
        },
        {  # 3d domain, two intersecting fractures, L-intersection
            "f_set": [
                np.array([[2, 4, 4, 2], [2, 2, 4, 4], [3, 3, 3, 3]]),
                np.array([[2, 4, 4, 2], [4, 4, 4, 4], [3, 3, 4, 4]]),
            ],
            "expected_num_grids": [0, 1, 2, 1],
            "expected_num_cells": [[], [2], [4, 2], [125]],
            "dim": 3,
        },
    ],
)
@pytest.mark.parametrize("cart_grid", [True, False])
@pytest.mark.parametrize("perturb", [True, False])
def test_structured_meshing(data: dict, cart_grid: bool, perturb: bool):
    """
    Test that the number of grids and cells are correct for a 2d domain with one or more
    fractures.

    In effect, this function tests the construction of a set of structured grids to
    describe a domain with fractures. The collection of the grids into a
    mixed-dimensional grid, including splitting of faces and nodes, is not tested here.

    Parameters:
        data: Dictionary containing fields:
            f_set: List of fractures expected_num_grids: Number of grids in each
            dimension.
            expected_num_cells: Number of cells in each grid.
            dim: Spatial dimension.
        cart_grid: Whether to use Cartesian or tensor grids.
        perturb: Whether to perturb the fracture before meshing.

    """
    f_set = data["f_set"]
    expected_num_grids = data["expected_num_grids"]
    expected_num_cells = data["expected_num_cells"]
    dim = data["dim"]

    if perturb:
        if len(f_set) == 0:
            # No need to perturb if no fractures
            return
        # Perturb the fracture so that the meshing will need to move it back to the
        # grid line. We also make a copy of the original fracture.
        orig_fracture = f_set[0].copy()
        f_set[0] = f_set[0].astype(float) + 0.3

    if cart_grid:
        # The physical dimensions are set to 5 in each direction.
        if dim == 2:
            physdims = [5, 5]
            grids = structured._cart_grid_2d(f_set, physdims, physdims=physdims)
        else:
            physdims = [5, 5, 5]
            grids = structured._cart_grid_3d(f_set, physdims, physdims=physdims)
    else:
        # Still generate a Cartesian grid, but use the tensor grid function.
        grid_lines = np.linspace(0, 5, 6)
        if dim == 2:
            grids = structured._tensor_grid_2d(f_set, grid_lines, grid_lines)
        else:
            grids = structured._tensor_grid_3d(
                f_set, grid_lines, grid_lines, grid_lines
            )

    # Verify that the number of grids and cells are correct.
    for grids_this_dim in grids:
        if len(grids_this_dim) == 0:
            continue
        dim_ind = grids_this_dim[0].dim
        assert len(grids_this_dim) == expected_num_grids[dim_ind]
        for i, g in enumerate(grids_this_dim):
            assert g.num_cells == expected_num_cells[dim_ind][i]

    if perturb:
        g = grids[1][0]
        # Check that the fracture was moved to the coordinate lines before meshing,
        # so that the minimal and maximal coordinates are the same as for the
        # original fracture.
        assert np.allclose(g.nodes.min(axis=1)[:dim], orig_fracture.min(axis=1))
        assert np.allclose(g.nodes.max(axis=1)[:dim], orig_fracture.max(axis=1))

        # Reset the perturbed fracture back again.
        f_set[0] = orig_fracture


@pytest.mark.parametrize(
    "f_set, domain_size, f_p_shape_true",
    [
        (
            [
                np.array([[1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 3, 3]]),
                np.array([[1, 1, 1, 1], [1, 3, 3, 1], [1, 1, 3, 3]]),
            ],
            [3, 3, 3],
            [0, 0, 0, 8],
        ),
        (
            [
                np.array([[0, 2, 2, 0], [1, 1, 1, 1], [0, 0, 2, 2]]),
                np.array([[1, 1, 1, 1], [0, 2, 2, 0], [0, 0, 2, 2]]),
            ],
            [2, 2, 2],
            [0, 0, 2, 8],
        ),
        (
            [
                np.array([[1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 3, 3]]),
                np.array([[1, 1, 1, 1], [1, 3, 3, 1], [1, 1, 3, 3]]),
                np.array([[1, 3, 3, 1], [1, 1, 3, 3], [1, 1, 1, 1]]),
            ],
            [3, 3, 3],
            [0, 0, 0, 12],
        ),
    ],
)
def test_g_frac_pairs(f_set: list[np.ndarray], domain_size: list, f_p_shape_true: list):
    """Test that the correct number of fracture pairs are found.

    Parameters:
        f_set: List of fractures.
        domain_size: Size of the domain.
        f_p_shape_true: Number of fracture pairs in each dimension.

    """
    mdg = pp.meshing.cart_grid(f_set, domain_size)
    mdg.compute_geometry()

    # Use np.hstack as a quick way to get all grids in one array.
    g_all = np.hstack([mdg.subdomains(dim=i) for i in range(4)])

    for g in g_all:
        f_p = g.frac_pairs
        # Check that the correct number of fracture pairs are found
        assert f_p.shape[1] == f_p_shape_true[g.dim]
        assert np.allclose(g.face_centers[:, f_p[0]], g.face_centers[:, f_p[1]])
