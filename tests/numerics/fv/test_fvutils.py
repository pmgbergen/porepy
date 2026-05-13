"""Unit tests for the fvutils module.

Currently tested are:
    * The subcell topology for 2d Cartesian and simplex grids.
    * The determination of the eta parameter.
    * The helper function computing a diagonal scaling matrix.
    * The function for obtaining indices of cells and faces for a partial update of the
    discretization stencil.

"""

from __future__ import division

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.applications.test_utils import reference_dense_arrays
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.numerics.fv import _fvutils


@pytest.fixture
def g_2d():
    return pp.CartGrid([5, 5])


@pytest.fixture
def g_3d():
    return pp.CartGrid([3, 3, 3])


def test_subcell_topology_2d_cart():
    # Verify that the subcell topology is correct for a 2d Cartesian grid.
    x = np.ones(2, dtype=int)
    g = pp.CartGrid(x)
    subcell_topology = _fvutils.SubcellTopology(g)

    assert np.all(subcell_topology.cno == 0)

    ncum = np.bincount(subcell_topology.nno, weights=np.ones(subcell_topology.nno.size))
    assert np.all(ncum == 2)

    fcum = np.bincount(subcell_topology.fno, weights=np.ones(subcell_topology.fno.size))
    assert np.all(fcum == 2)

    # There is only one cell, thus only unique subfno
    usubfno = np.unique(subcell_topology.subfno)
    assert usubfno.size == subcell_topology.subfno.size

    assert np.all(np.isin(subcell_topology.subfno, subcell_topology.subhfno))


def test_subcell_mapping_2d_simplex():
    # Verify that the subcell mapping is correct for a 2d simplex grid.
    p = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    g = pp.TriangleGrid(p)

    subcell_topology = _fvutils.SubcellTopology(g)

    ccum = np.bincount(subcell_topology.cno, weights=np.ones(subcell_topology.cno.size))
    assert np.all(ccum == 6)

    ncum = np.bincount(subcell_topology.nno, weights=np.ones(subcell_topology.nno.size))
    assert ncum[0] == 2
    assert ncum[1] == 4
    assert ncum[2] == 2
    assert ncum[3] == 4

    fcum = np.bincount(subcell_topology.fno, weights=np.ones(subcell_topology.fno.size))
    assert np.sum(fcum == 4) == 1
    assert np.sum(fcum == 2) == 4

    subfcum = np.bincount(
        subcell_topology.subfno, weights=np.ones(subcell_topology.subfno.size)
    )
    assert np.sum(subfcum == 2) == 2
    assert np.sum(subfcum == 1) == 8


@pytest.mark.parametrize(
    "grid, expected_eta",
    [(pp.StructuredTriangleGrid([1, 1]), 1 / 3), (pp.CartGrid([1, 1]), 0)],
)
def test_determine_eta(grid, expected_eta):
    # Test that the automatic computation of the pressure continuity point is
    # correct for a Cartesian and simplex grid.
    assert _fvutils.determine_eta(grid) == expected_eta


"""IMPLEMENTATION NOTE:
Below are tests that considers the definition of computational stencils under a partial
update of the FV discretization.

The logic of the tests is:
    1) Construct a grid, and pick a cell quantity (node, face, cell) as the trigger for
       refinement.
    2) Compute the update stencil.
    3) Compare the stencil with a known value, hard-coded based on knowledge of the grid
       and the ordering of the cell quantities.

"""


def test_cell_and_face_indices_from_node_indices_2d(g_2d):
    # Nodes of cell 12 (middle one) - from counting
    n = np.array([14, 15, 20, 21])

    known_cells = np.array([6, 7, 8, 11, 12, 13, 16, 17, 18])
    known_faces = np.array([14, 15, 42, 47])

    cell_ind, face_ind = _fvutils.cell_ind_for_partial_update(g_2d, nodes=n)
    assert compare_arrays(known_cells, cell_ind)
    assert compare_arrays(known_faces, face_ind)


def test_cell_and_face_indices_from_node_indices_2d_boundary(g_2d):
    # Nodes of cell 1
    n = np.array([1, 2, 7, 8])
    known_cells = np.array([0, 1, 2, 5, 6, 7])
    known_faces = np.array([1, 2, 31, 36])

    cell_ind, face_ind = _fvutils.cell_ind_for_partial_update(g_2d, nodes=n)

    assert compare_arrays(known_cells, cell_ind)
    assert compare_arrays(known_faces, face_ind)


def test_cell_and_face_indices_from_node_indices_3d(g_3d):
    # Nodes of cell 13 (middle one) - from counting
    n = np.array([21, 22, 25, 26, 37, 38, 41, 42])

    known_cells = np.arange(27)
    known_faces = np.array([17, 18, 52, 55, 85, 94])

    cell_ind, face_ind = _fvutils.cell_ind_for_partial_update(g_3d, nodes=n)

    assert compare_arrays(known_cells, cell_ind)
    assert compare_arrays(known_faces, face_ind)


def test_cell_and_face_indices_from_node_indices_3d_boundary(g_3d):
    # Nodes of cell 1
    n = np.array([1, 2, 5, 6, 17, 18, 21, 22])
    known_cells = np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14])
    known_faces = np.array([1, 2, 37, 40, 73, 82])

    cell_ind, face_ind = _fvutils.cell_ind_for_partial_update(g_3d, nodes=n)

    assert compare_arrays(known_cells, cell_ind)
    assert compare_arrays(known_faces, face_ind)


def test_cell_and_face_indices_from_cell_indices_2d(g_2d):
    c = np.array([12])
    known_cells = np.setdiff1d(np.arange(25), np.array([0, 4, 20, 24]))
    known_faces = np.array([8, 9, 14, 15, 20, 21, 41, 42, 43, 46, 47, 48])

    cell_ind, face_ind = _fvutils.cell_ind_for_partial_update(g_2d, cells=c)

    assert compare_arrays(known_cells, cell_ind)
    assert compare_arrays(known_faces, face_ind)


def test_cell_and_face_indices_from_cell_indices_3d(g_3d):
    # Use cell 13 (middle one)
    c = np.array([13])
    known_cells = np.arange(27)
    fx = np.hstack(
        (
            np.array([1, 2, 5, 6, 9, 10]),
            np.array([1, 2, 5, 6, 9, 10]) + 12,
            np.array([1, 2, 5, 6, 9, 10]) + 24,
        )
    )
    fy = 36 + np.hstack(
        (
            np.array([3, 4, 5, 6, 7, 8]),
            np.array([3, 4, 5, 6, 7, 8]) + 12,
            np.array([3, 4, 5, 6, 7, 8]) + 24,
        )
    )
    fz = 72 + np.hstack((np.arange(9) + 9, np.arange(9) + 18))
    known_faces = np.hstack((fx, fy, fz))

    cell_ind, face_ind = _fvutils.cell_ind_for_partial_update(g_3d, cells=c)

    assert compare_arrays(known_cells, cell_ind)
    assert compare_arrays(known_faces, face_ind)


def test_cell_and_face_indices_from_cell_indices_3d_boundary(g_3d):
    c = np.array([1])
    known_cells = np.arange(27)
    fx = np.array([1, 2, 5, 6, 13, 14, 17, 18])
    fy = 36 + np.array([0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17])
    fz = 72 + np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14])
    known_faces = np.hstack((fx, fy, fz))
    cell_ind, face_ind = _fvutils.cell_ind_for_partial_update(g_3d, cells=c)

    assert compare_arrays(known_cells, cell_ind)
    assert compare_arrays(known_faces, face_ind)


def test_cell_and_face_indices_from_face_indices_2d(g_2d):
    # Use face between cells 11 and 12
    f = np.array([14])

    known_cells = np.arange(g_2d.num_cells)
    known_faces = np.array([8, 14, 20, 41, 42, 46, 47])
    cell_ind, face_ind = _fvutils.cell_ind_for_partial_update(g_2d, faces=f)

    assert compare_arrays(known_cells, cell_ind)
    assert compare_arrays(known_faces, face_ind)


def test_cell_and_face_indices_from_face_indices_2d_boundary(g_2d):
    # Face between cell 1 and 2
    f = np.array([2])
    known_cells = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    )
    known_faces = np.array([2, 8, 31, 32, 36, 37])
    cell_ind, face_ind = _fvutils.cell_ind_for_partial_update(g_2d, faces=f)

    assert compare_arrays(known_cells, cell_ind)
    assert compare_arrays(known_faces, face_ind)


def test_bound_exclusion():
    """Test the exclusion of boundaries for a 2d grid.

    The domain consists of a 3x3 mesh. Bottom faces are Dirichlet, while left and
    right faces are rolling along y (dir_x and neu_y). Top mid face is rolling along
    x (neu_x and dir_y) and op left and right faces are Neumann.
    """

    g = pp.CartGrid([3, 3])
    g.compute_geometry()
    nd = g.dim

    dir_x = np.array([0, 3, 4, 7, 8, 11])
    dir_y = np.array([22])
    dir_both = np.array([12, 13, 14])

    bound = pp.BoundaryConditionVectorial(g)

    bound.is_dir[0, dir_x] = True
    bound.is_neu[0, dir_x] = False
    bound.is_dir[1, dir_y] = True
    bound.is_neu[1, dir_y] = False
    bound.is_dir[:, dir_both] = True
    bound.is_neu[:, dir_both] = False

    subcell_topology = _fvutils.SubcellTopology(g)
    # Move the boundary conditions to sub-faces
    bound.is_dir = bound.is_dir[:, subcell_topology.fno_unique]
    bound.is_rob = bound.is_rob[:, subcell_topology.fno_unique]
    bound.is_neu = bound.is_neu[:, subcell_topology.fno_unique]
    bound.robin_weight = bound.robin_weight[:, :, subcell_topology.fno_unique]
    bound.basis = bound.basis[:, :, subcell_topology.fno_unique]

    # Obtain the face number for each coordinate
    subfno = subcell_topology.subfno_unique
    subfno_nd = np.tile(subfno, (nd, 1)) * nd + np.atleast_2d(np.arange(0, nd)).T

    bound_exclusion = _fvutils.ExcludeBoundaries(subcell_topology, bound, nd)

    # Expand the indices
    # Define right hand side for Neumann boundary conditions
    # First row indices in rhs matrix
    # Pick out the subface indices
    subfno_neu = bound_exclusion.exclude_robin_dirichlet(subfno_nd.ravel("C")).ravel(
        "F"
    )

    # Pick out the Neumann boundary
    is_neu_nd = (
        bound_exclusion.exclude_robin_dirichlet(bound.is_neu.ravel("C"))
        .ravel("F")
        .astype(bool)
    )

    neu_ind = np.argsort(subfno_neu)
    neu_ind = neu_ind[is_neu_nd[neu_ind]]
    reference = reference_dense_arrays.test_fvutils["test_bound_exclusion"]
    assert np.all(neu_ind == reference["neu_inds"])

    subfno_dir = bound_exclusion.exclude_neumann_robin(subfno_nd.ravel("C")).ravel("F")
    is_dir_nd = (
        bound_exclusion.exclude_neumann_robin(bound.is_dir.ravel("C"))
        .ravel("F")
        .astype(bool)
    )

    dir_ind = np.argsort(subfno_dir)
    dir_ind = dir_ind[is_dir_nd[dir_ind]]

    assert np.all(dir_ind == reference["dir_inds"])


def test_adjust_eta_length_uniform_faces():
    """Check that adjust_eta_length correctly extracts subface eta values when all faces
    have the same number of nodes (standard 2d Cartesian grid).
    """
    # 2x1 Cartesian grid: 3 vertical faces (2 nodes each) + 4 horizontal faces
    # (2 nodes each) = 7 faces, 14 entries in the global face_nodes CSC indptr.
    sd = pp.CartGrid([2, 1])
    sd.compute_geometry()

    # Build a global eta vector: one distinct value per subface slot.
    n_global_subfaces = sd.face_nodes.tocsc().indptr[-1]
    eta = np.arange(n_global_subfaces, dtype=float)

    # Select a subset of faces (e.g. face 0 and face 2).
    l2g_faces = np.array([0, 2])

    # Call with a trivial sub_sd (not used in the new implementation).
    loc_eta = _fvutils.adjust_eta_length(eta=eta, sub_sd=sd, l2g_faces=l2g_faces, sd=sd)

    # Face 0 occupies slots 0-1, face 2 occupies slots 4-5 in the indptr
    # (each face has 2 nodes, so face f starts at slot 2*f).
    assert np.array_equal(loc_eta, np.array([0.0, 1.0, 4.0, 5.0]))


def test_adjust_eta_length_mixed_nodes_per_face():
    """adjust_eta_length works correctly when faces have different numbers of nodes.

    In 3D, faces can be e.g. triangles (3 nodes) or quadrilaterals (4 nodes). This test
    uses a triangular prism, which has 2 triangular faces and 3 quadrilateral faces (as
    would be the result of a grid extrusion type 3d grid).

    This test verifies that adjust_eta_length correctly gathers the subface eta values
    for a subset of faces, even when the faces have different numbers of nodes (and thus
    different numbers of subface slots in the global eta vector).
    """
    # Triangular prism: 6 nodes, 5 faces, 1 cell.
    #   Bottom triangle: nodes 0, 1, 2
    #   Top triangle:    nodes 3, 4, 5
    nodes = np.array(
        [[0, 1, 0.5, 0, 1, 0.5], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 1, 1]],
        dtype=float,
    )

    # face_nodes (CSC, 6 nodes x 5 faces):
    #   f0 (bottom triangle): nodes [0, 1, 2]    -> 3 entries
    #   f1 (top triangle):    nodes [3, 4, 5]    -> 3 entries
    #   f2 (front quad):      nodes [0, 1, 3, 4] -> 4 entries
    #   f3 (right quad):      nodes [1, 2, 4, 5] -> 4 entries
    #   f4 (left quad):       nodes [0, 2, 3, 5] -> 4 entries
    fn_indptr = np.array([0, 3, 6, 10, 14, 18])
    fn_indices = np.array([0, 1, 2, 3, 4, 5, 0, 1, 3, 4, 1, 2, 4, 5, 0, 2, 3, 5])
    face_nodes = sps.csc_matrix(
        (np.ones(18, dtype=bool), fn_indices, fn_indptr), shape=(6, 5)
    )

    # cell_faces: all 5 faces belong to the single cell.
    cell_faces = sps.csc_matrix(
        (np.ones(5), (np.arange(5), np.zeros(5, dtype=int))), shape=(5, 1)
    )

    sd = pp.Grid(3, nodes, face_nodes, cell_faces, "prism")

    # Assign a distinct eta value per global subface (node-face incidence).
    # Total: 3 + 3 + 4 + 4 + 4 = 18 entries.
    eta = np.arange(18, dtype=float)

    # Request f0 (triangle, 3 nodes) and f2 (quad, 4 nodes).
    l2g_faces = np.array([0, 2])

    loc_eta = _fvutils.adjust_eta_length(eta=eta, sub_sd=sd, l2g_faces=l2g_faces, sd=sd)

    # fn_indptr = [0, 3, 6, 10, 14, 18], so:
    #   f0 (triangle) occupies slots 0-2  -> eta values [0, 1, 2]
    #   f2 (quad)     occupies slots 6-9  -> eta values [6, 7, 8, 9]
    assert np.array_equal(loc_eta, np.array([0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 9.0]))
