"""The module contains tests of match_grids.py. For the moment, only function
match_grids_1d() is tested.

"""

import numpy as np
import scipy.sparse as sps

import porepy as pp

"""IMPLEMENTATION NOTE: While all tests are based on the same grid, most of them also
    do a perturbation of the grid. To document the logic behind individual tests, it was
    decided not to use a fixture or parametrization, but to centralize the grid creation
    and comparison of mappings.
"""


def test_match_grids_1d_no_perturbation():
    """Test the mapping between two identical grids. This should give a unit mapping."""
    g, h = _grid_1d(), _grid_1d()
    known_data = np.ones(2)
    known_row = np.arange(2)
    known_col = np.arange(2)
    _compare_matched_grids_with_known_values(g, h, known_data, known_row, known_col)


def test_match_grids_1d_perturbation():
    """Perturb one node in the new grid. This gives a non-trivial mapping."""
    g, h = _grid_1d(), _grid_1d()
    # Perturb one node in the first grid
    g.nodes[0, 1] = 0.5
    # To be careful, recompute geometry of the perturbed grid
    g.compute_geometry()

    known_data = np.array([1, 1.0 / 3, 2.0 / 3])
    known_row = np.array([0, 1, 1])
    known_col = np.array([0, 0, 1])

    _compare_matched_grids_with_known_values(g, h, known_data, known_row, known_col)


def test_match_grids_1d_reversed_node_order():
    """Reverse the order of nodes in the new grid. This gives a non-trivial mapping."""
    g, h = _grid_1d(), _grid_1d()
    # Reverse the order of the nodes in the second grid. Since every face consists of a
    # single node, this also reverses the order of the faces, so that the two grids are:
    # g: 0--1--2
    # h: 2--1--0
    # Thus, cell 0 (defined by faces 0 and 1 in both grids) spans the interval [0, 1]
    # in g, but [1, 2] in h, and vice versa.
    h.nodes = h.nodes[:, ::-1]
    h.compute_geometry()

    known_data = np.array([1, 1])
    known_row = np.array([0, 1])
    known_col = np.array([1, 0])
    _compare_matched_grids_with_known_values(g, h, known_data, known_row, known_col)


def test_match_grids_1d_complex():
    """Define more complex grids, by combining two simple grids, and test the mappings."""

    def _create_combined_grid(g1, g2):
        # Helper function to create a combined grid from two grids.
        g_nodes = np.hstack((g1.nodes, g2.nodes))
        g_face_nodes = sps.block_diag((g1.face_nodes, g2.face_nodes), "csc")
        g_cell_faces = sps.block_diag((g1.cell_faces, g2.cell_faces), "csc")
        g = pp.Grid(1, g_nodes, g_face_nodes, g_cell_faces, "pp.TensorGrid")
        g.compute_geometry()
        return g

    g = _create_combined_grid(
        pp.TensorGrid(np.linspace(0, 2, 2)), pp.TensorGrid(np.linspace(2, 4, 2))
    )
    h = _create_combined_grid(
        pp.TensorGrid(np.linspace(0, 2, 3)), pp.TensorGrid(np.linspace(2, 4, 3))
    )

    # First create mappings from h to g.
    known_data_h_2_g = np.array([1.0, 1.0, 1.0, 1.0])
    known_row_h_2_g = np.array([0, 1, 2, 3])
    known_col_h_2_g = np.array([0, 0, 1, 1])
    _compare_matched_grids_with_known_values(
        h, g, known_data_h_2_g, known_row_h_2_g, known_col_h_2_g
    )

    # Next, make a map from g to h. In this case, the cells in h are split in two
    # thus the weight is 0.5.
    known_data_g_2_h = np.array([0.5, 0.5, 0.5, 0.5])
    known_row_g_2_h = np.array([0, 0, 1, 1])
    known_col_g_2_h = np.array([0, 1, 2, 3])
    _compare_matched_grids_with_known_values(
        g, h, known_data_g_2_h, known_row_g_2_h, known_col_g_2_h
    )


## Helper functions below


def _grid_1d():
    """Helper function to create a 1d grid used in most tests.

    Returns:
        g: A 1d grid with two cells.
    """
    g = pp.TensorGrid(np.arange(3))
    g.compute_geometry()
    return g


def _compare_matched_grids_with_known_values(
    g: pp.TensorGrid,
    h: pp.TensorGrid,
    data: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
):
    """Compared the mapping from g to h with known values.

    Parameters:
        g: Grid to map from.
        h: Grid to map to.
        data: Known values of the mapping, as computed from match_grids_1d.
        row: Known row indices of the mapping, as computed from match_grids_1d.
        col: Known column indices of the mapping, as computed from match_grids_1d.

    """

    # Construct a map from h to g
    mat_h_2_g = pp.match_grids.match_1d(g, h, tol=1e-4, scaling="averaged")
    mat_h_2_g.eliminate_zeros()
    # Convert to coo format for easy comparison
    mat_h_2_g = mat_h_2_g.tocoo()
    # A failure here means the mapping is not correct
    assert np.allclose(mat_h_2_g.data, data)
    assert np.allclose(mat_h_2_g.row, row)
    assert np.allclose(mat_h_2_g.col, col)
