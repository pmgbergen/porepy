""" Various tests related to grids:
* The main grid class, including geometry computation.
* Specific tests for Simplex and Structured Grids
* Tests for the mortar grid.
"""

import os
import pickle

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.applications.test_utils import reference_dense_arrays
from porepy.grids import simplex, structured
from porepy.utils import setmembership
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data


@pytest.mark.parametrize(
    "grid, expected_diameter",
    [
        (
            pp.CartGrid(np.array([3, 2]), np.array([1, 1])),
            np.sqrt(0.5**2 + 1.0 / 3.0**2),
        ),
        (pp.CartGrid(np.array([3, 2, 1])), np.sqrt(3)),
    ],
)
def test_cell_diameters(grid, expected_diameter):
    # The test is run for a 2d grid and a 3d grid.
    cell_diameters = grid.cell_diameters()
    known = np.repeat(expected_diameter, grid.num_cells)
    assert np.allclose(cell_diameters, known)


def test_repr():
    # Call repr, just to see that it works
    g = pp.CartGrid(np.array([1, 1]))
    g.__repr__()


def test_str():
    # Call str, just to see that it works
    g = pp.CartGrid(np.array([1, 1]))
    g.__str__()


# ----- Tests of function to find the closest cell ----- #


def test_closest_cell_in_plane():
    # 2d grid, points also in plane
    g = pp.CartGrid(np.array([3, 3]))
    g.compute_geometry()
    p = np.array([[0.5, -0.5], [0.5, 0.5]])
    ind = g.closest_cell(p)
    assert np.allclose(ind, 0 * ind)


def test_closest_cell_out_of_plane():
    g = pp.CartGrid(np.array([2, 2]))
    g.compute_geometry()
    p = np.array([[0.1], [0.1], [1]])
    ind = g.closest_cell(p)
    assert ind[0] == 0
    assert ind.size == 1


@pytest.mark.parametrize(
    "g",
    [
        pp.PointGrid(np.array([0, 0, 0])),
        pp.CartGrid(np.array([2])),
        pp.CartGrid(np.array([2, 2])),
        pp.CartGrid(np.array([2, 2, 2])),
        pp.StructuredTriangleGrid(np.array([2, 2])),
        pp.StructuredTetrahedralGrid(np.array([2, 2, 2])),
    ],
)
def test_cell_faces_as_dense(g: pp.Grid):
    """Test that the cell_faces_as_dense method works as expected.

    The test is based on constructing the dense version of the cell-face relation, using
    the relevant method in the grid class, and reconstructing the sparse version from
    the dense version using the known (or supposed, in case of bugs) structure of the
    dense version.

    Parameters:
        g: The grid for which the test should be run.

    """
    # Get the sparse and dense versions of the cell-face relation.
    cf_sparse = g.cell_faces
    cf_dense = g.cell_faces_as_dense()

    # Number of faces in the grid.
    nf, nc = g.num_faces, g.num_cells

    # All (possible) face indices. All of these will occur at least once (or else
    # something is wrong, but the test should still work).
    face_ind = np.arange(nf)

    # In the dense cell-face relation, there are as many columns as there are faces, and
    # the elements in the faces are the indices of the cells neighboring the face. The
    # cells in the first row have positive sign in the sparse cell-face relation (that
    # is, the face normal vectors point out of the cells), while the row contains cells
    # with negative sign. A negative number in the dense cell-face relation signifies
    # that the face is on the boundary, and that there is no neighboring cell on that
    # side of the face.

    # Logical array to identify actual cells in the first and second row
    internal_cell_0 = cf_dense[0] >= 0
    internal_cell_1 = cf_dense[1] >= 0

    # Form the rows, columns and data for the reconstructed sparse cell-face relation.
    # The rows are the indices of those faces that are internal. Pick indices from the
    # first row of the dense relation, then the second.
    rows = np.concatenate([face_ind[internal_cell_0], face_ind[internal_cell_1]])
    # The columns are the indices of the cells that are neighbors to the faces. Again do
    # filtering to only catch actual cells.
    cols = np.concatenate([cf_dense[0, internal_cell_0], cf_dense[1, internal_cell_1]])
    # The data is 1 for the first row, and -1 for the second.
    data = np.concatenate(
        [np.ones(np.sum(internal_cell_0)), -np.ones(np.sum(internal_cell_1))]
    )

    # Reconstruct the sparse cell-face relation.
    reconstructed = sps.coo_matrix((data, (rows, cols)), shape=(nf, nc)).tocsr()
    # The original and reconstructed cell-face relations should be the same.
    assert np.allclose(reconstructed.toarray(), cf_sparse.toarray())


class TestDivergenceTrace:

    def create_grids(self) -> list[pp.Grid]:
        """Create grids used for testing the divergence and trace operators.

        The grids are:
            1. 2d Cartesian grid with 2x2 cells.
            2. 3d Cartesian grid with 2x2x2 cells.
            3. 2d structured triangle grid with 2x2 cells.
            4. 3d structured tetrahedral grid with 2x2x2 cells.
            5. A 1d grid with two cells, and with the middle node split.

        Returns:
            A list of grids.

        """
        g1 = pp.CartGrid([2, 2], [1, 1])
        g2 = pp.CartGrid([2, 2, 2], [1, 1, 1])
        g3 = pp.StructuredTriangleGrid([2, 2], [1, 1])
        g4 = pp.StructuredTetrahedralGrid([2, 2, 2], [1, 1, 1])

        x = np.array([[0, 0.5, 0.5, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        fn = sps.identity(4, format="csr")
        cf = sps.csc_matrix(np.array([[-1, 1, 0, 0], [0, 0, -1, 1]]).T)
        g5 = pp.Grid(dim=1, nodes=x, face_nodes=fn, cell_faces=cf, name="")

        grid_list = [g1, g2, g3, g4, g5]

        for g in grid_list:
            g.compute_geometry()
        return grid_list

    def _boundary_faces(self, g: pp.Grid) -> np.ndarray:
        """Find the boundary faces of the grid.

        The boundary faces are identified by geometrically checking if the face center
        is at the boundary of the domain, which is assumed to be of unit size. For 1d
        grids, we also know that faces with center at x=0.5 are boundary faces. This
        construction is not general, but it allows us to find the boundary faces without
        using boundary tags and similar, and thus provide a real check of the
        implementation of the divergence and trace operators.

        Parameters:
            g: The grid for which the boundary faces should be found.

        Returns:
            np.ndarray: The indices of the boundary faces.

        """

        bf = np.where(
            np.logical_or(
                np.any(g.face_centers[: g.dim] == 0, axis=0),
                np.any(g.face_centers[: g.dim] == 1, axis=0),
            )
        )[0]

        if g.dim == 1:
            bf = np.hstack((bf, np.where(g.face_centers[0] == 0.5)[0]))

        bf.sort()
        return bf

    def _bound_int_ind(self, g: pp.Grid, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """Find the indices correspodning to  boundary and internal faces of a grid.

        The indices are expanded to account for the dimension 'dim' of an operator (e.g.
        the divergence operator).

        Parameters:
            g: The grid for which the indices should be found.
            dim: The dimension of the operator.

        Returns:
            tuple with the indices corresponding to the boundary and internal faces.

        """

        # Faces on the boundary, and the indices of these faces in the divergence
        # matrix. These will be identical if dim=1, but for dim > 1, there will be dim
        # boundary indices for each boundary face.
        bound_faces = self._boundary_faces(g)
        bound_ind = pp.fvutils.expand_indices_nd(bound_faces, dim)
        # The indices corresponding to internal faces are the complement of the boundary
        # indices.
        int_face_ind = np.setdiff1d(np.arange(g.num_faces * dim), bound_ind)

        return bound_ind, int_face_ind

    @pytest.mark.parametrize("dim", [1, 2])
    def test_divergence(self, dim: int) -> None:
        """Test the divergence operator for a number of grids.

        See the delow documenation for details on the test.

        Parameters:
            dim: The dimension of the divergence operator.

        """
        for g in self.create_grids():

            # Divergence operator to be tested.
            div = g.divergence(dim=dim).toarray()
            # First a simple shape check.
            assert div.shape == (dim * g.num_cells, dim * g.num_faces)

            # Indices of boundary and internal faces.
            bound_ind, int_face_ind = self._bound_int_ind(g, dim)

            # Absolute value of the divergence, used to count the number of occurrences.
            abs_div = np.abs(div)
            # There should be exactly two occurrences of each internal face in the
            # divergence matrix. They should have opposite signs, thus their sum should
            # be zero. No similar test for the boundary faces, as this is also tested
            # below (num_occ_bound). This is actually also partly true for the internal
            # faces (in that case we need both num_occ_ind and the geometric test
            # below), but we do overlapping tests since debugging a problem with
            # opposite signs is much easier here than for the geometric test.
            assert np.allclose(np.sum(abs_div[:, int_face_ind], axis=0), 2)
            assert np.allclose(np.sum(div[:, int_face_ind], axis=0), 0)

            # Find non-zero elements in the divergence matrix.
            ci, fi, sgn = sps.find(div)

            # There shoud be exactly one occurrence of each boundary face in the
            # divergence matrix.
            num_occ_bound = np.array([np.sum(i == fi) for i in bound_ind])
            assert np.allclose(num_occ_bound, 1)
            # There should be exactly two occurrences of each internal face in the
            # divergence matrix.
            num_occ_int = np.array([np.sum(i == fi) for i in int_face_ind])
            assert np.allclose(num_occ_int, 2)

            # As a final check, the sign of the divergence should be such that the
            # divergence is positive if the face normal points out of the cell, and
            # negative if it points into the cell.

            # Vector from cell center to face center, and face normal. All are expanded
            # (repeated) to match the repeats in ci and fi, if dim > 1 (each cell-face
            # combination occurs more than once).
            fc_cc = (
                np.repeat(g.face_centers, dim, axis=1)[:, fi]
                - np.repeat(g.cell_centers, dim, axis=1)[:, ci]
            )
            n = np.repeat(g.face_normals, dim, axis=1)[:, fi]

            # Norm of the vector from cell center to face center.
            nrm_fc_cc = np.linalg.norm(fc_cc, axis=0)
            # Norm of the same vector, prolonged a little bit in the direction of the
            # face normal.
            nrm_fc_cc_n = np.linalg.norm(fc_cc + 1e-3 * n, axis=0)

            # The normal vector points out of the cell if the prolonged vector is longer
            # than the original vector. Convert this binary information into 1 and -1
            # for True and False.
            outward = 2 * (nrm_fc_cc_n > nrm_fc_cc) - 1
            # The sign of the divergence should be positive for outward pointing
            # normals, and negative for inward pointing normals.
            assert np.allclose(sgn, outward)

    @pytest.mark.parametrize("dim", [1, 2])
    def test_trace(self, dim):
        """Test the trace operator for a number of grids.

        Parameters:
            dim: The dimension of the trace operator.

        """
        for g in self.create_grids():
            trace = g.trace(dim=dim).toarray()
            # Indices of boundary and internal faces.
            bound_ind, _ = self._bound_int_ind(g, dim)

            # Non-zero elements in the trace matrix.
            fi, ci, sgn = sps.find(trace)

            # The trace matrix should be a mapping from cells to faces, expanded to
            # deal with 'dim' quantities.
            assert trace.shape == (dim * g.num_faces, dim * g.num_cells)

            # The trace matrix should be mapping values without altering them, hence
            # all non-zero values should be 1.
            assert np.all(sgn == 1)

            # Each face index should occur at most once.
            assert fi.size == np.unique(fi).size

            # All boundary indices should be in the trace matrix, and the trace matrix
            # should only contain boundary indices. This implicitly checks that the
            # trace matrix contains no internal indices.
            assert np.all(np.isin(fi, bound_ind))
            assert np.all(np.isin(bound_ind, fi))

            # Finally try to verify that the trace matrix maps from the correct cell to
            # the correct face. Ideally this should have been tested using geometric
            # arguments, as this would have provided independent verification from the
            # topology-based information used to construct the trace matrix. However,
            # EK's attempts at doing so failed, so we have to rely on the topology-based
            # test below. This is not ideal, but it is better than nothing.

            # Construct a cell-face relation matrix expanded 'dim' times, and eliminate
            # zeros.
            cf = sps.kron(g.cell_faces, sps.eye(dim)).tocsr()
            cf.eliminate_zeros()

            # Loop over the boundary indices.
            for face_ind in bound_ind:
                # Find the cells that neighbor the face.
                cells = cf[face_ind].indices
                # There should be exactly one cell that neighbors the face.
                assert cells.size == 1
                # The trace matrix should contain the cell indices for the face.
                assert np.all(ci[fi == face_ind] == cells[0])


# ----- Boundary tests ----- #


def test_boundary_node_cart():
    g = pp.CartGrid(np.array([2, 2]))
    bound_ind = g.get_boundary_nodes()
    known_bound = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    assert np.allclose(np.sort(bound_ind), known_bound)


def test_boundary_faces_cart():
    g = pp.CartGrid(np.array([2, 1]))
    int_faces = g.get_internal_faces()
    assert int_faces.size == 1
    assert int_faces[0] == 1


def test_get_internal_nodes_cart():
    g = pp.CartGrid(np.array([2, 2]))
    int_nodes = g.get_internal_nodes()
    assert int_nodes.size == 1
    assert int_nodes[0] == 4


def test_get_internal_nodes_empty():
    g = pp.CartGrid(np.array([2, 1]))
    int_nodes = g.get_internal_nodes()
    assert int_nodes.size == 0


def test_bounding_box():
    sd = pp.CartGrid(np.array([1, 1]))
    sd.nodes = np.random.random((sd.dim, sd.num_nodes))
    bmin, bmax = pp.domain.grid_minmax_coordinates(sd)
    assert np.allclose(bmin, sd.nodes.min(axis=1))
    assert np.allclose(bmax, sd.nodes.max(axis=1))


# ----- Tests of the compute_geometry function ----- #

expected_geometry_1 = (
    np.ones(4),
    np.ones(1),
    np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]).T,
    np.array([[0, 0.5, 0], [1, 0.5, 0], [0.5, 0, 0], [0.5, 1, 0]]).T,
    np.array([[0.5, 0.5, 0]]).T,
)
expected_geometry_2 = (
    5000 * np.ones(4),
    5000**2 * np.ones(1),
    np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]).T * 5000,
    np.array([[0, 0.5, 0], [1, 0.5, 0], [0.5, 0, 0], [0.5, 1, 0]]).T * 5000,
    np.array([[0.5, 0.5, 0]]).T * 5000,
)
params = [
    (pp.CartGrid(np.array([1, 1])), expected_geometry_1),  # Unit Cartesian grid
    (
        pp.CartGrid(
            np.array([1, 1]), np.array([5000, 5000])
        ),  # Very large Cartesian grid
        expected_geometry_2,
    ),
]


@pytest.mark.parametrize("grid, expected_geometry", params)
def test_compute_geometry_cart_2d(grid, expected_geometry):
    grid.compute_geometry()
    known_areas = expected_geometry[0]
    known_volumes = expected_geometry[1]
    known_normals = expected_geometry[2]
    known_face_center = expected_geometry[3]
    known_cell_center = expected_geometry[4]
    assert np.allclose(grid.face_areas, known_areas)
    assert np.allclose(grid.cell_volumes, known_volumes)
    assert np.allclose(grid.face_normals, known_normals)
    assert np.allclose(grid.face_centers, known_face_center)
    assert np.allclose(grid.cell_centers, known_cell_center)


# ----- Test compute_geometry for various challenging grids ----- #

# This grid should trigger is_oriented = False, and compute_geometry should fall
# back on the implementation based on convex cells.
grid_1 = pp.Grid(
    2,
    np.array([[0, 1, 0, 1], [0, 0, 1, 1], np.zeros(4)]),
    sps.csc_matrix(
        (np.ones(8), np.array([0, 1, 2, 3, 0, 2, 1, 3]), np.arange(0, 9, 2))
    ),
    sps.csc_matrix(np.ones((4, 1))),
    "inconsistent",
)
# Known quadrilateral for which the convexity assumption fails.
grid_2 = pp.Grid(
    2,
    np.array([[0, 0.5, 1, 0.5], [0, 0.5, 0, 1], np.zeros(4)]),
    sps.csc_matrix(
        (np.ones(8), np.array([0, 1, 1, 2, 2, 3, 3, 0]), np.arange(0, 9, 2))
    ),
    sps.csc_matrix(np.ones((4, 1))),
    "concave",
)
# Known quadrilateral for which the convexity assumption fails
# and the centroid is external.
grid_3 = pp.Grid(
    2,
    np.array([[0, 0.5, 1, 0.5], [0, 0.75, 0, 1], np.zeros(4)]),
    sps.csc_matrix(
        (np.ones(8), np.array([0, 1, 1, 2, 2, 3, 3, 0]), np.arange(0, 9, 2))
    ),
    sps.csc_matrix(np.ones((4, 1))),
    "concave",
)
# Small mesh consisting of two concave and one convex cell.
grid_4 = pp.Grid(
    2,
    np.array([[0, 0.5, 1, 0.5, 0.5, 0.5], [0, 0.5, 0, 1, -0.5, -1], np.zeros(6)]),
    sps.csc_matrix(
        (
            np.ones(16),
            np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 5, 5, 2, 2, 4, 4, 0]),
            np.arange(0, 17, 2),
        )
    ),
    sps.csc_matrix(
        (
            np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1]),
            (
                np.array([0, 1, 2, 3, 7, 6, 1, 0, 4, 5, 6, 7]),
                np.repeat(np.arange(3), 4),
            ),
        )
    ),
    "concave",
)
expected_geometry_1 = (
    1,
    np.array([[0.5], [0.5], [0]]),
    np.array([[0.0, -0.0, -1.0, 1.0], [-1.0, 1.0, -0.0, 0.0], [0.0, -0.0, -0.0, 0.0]]),
)
expected_geometry_2 = (
    0.25,
    np.array([[0.5], [0.5], [0]]),
    np.array([[0.5, -0.5, 1.0, -1.0], [-0.5, -0.5, 0.5, 0.5], [0.0, 0.0, -0.0, 0.0]]),
)
expected_geometry_3 = (
    0.125,
    np.array([[0.5], [1.75 / 3], [0]]),
    np.array([[0.75, -0.75, 1.0, -1.0], [-0.5, -0.5, 0.5, 0.5], [0.0, 0.0, -0.0, 0.0]]),
)
expected_geometry_4 = (
    [0.25, 0.5, 0.25],
    np.array([[0.5, 0.5, 0.5], [0.5, 0.0, -0.5], [0.0, 0.0, 0.0]]),
    np.array(
        [
            [0.5, -0.5, 1.0, -1.0, -1.0, 1.0, -0.5, 0.5],
            [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
            [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0],
        ]
    ),
)

# The final parameter determines whether the
# cell volumes should be checked for all cells (True) or just the first one (False).
params = [
    (grid_1, expected_geometry_1, False),
    (grid_2, expected_geometry_2, False),
    (grid_3, expected_geometry_3, False),
    (grid_4, expected_geometry_4, True),
]


@pytest.mark.parametrize("grid, expected_geometry, check_all_cell_volumes", params)
def test_compute_geometry_challenging_grids(
    grid, expected_geometry, check_all_cell_volumes
):
    grid.compute_geometry()
    if check_all_cell_volumes:
        cell_volumes_to_compare = grid.cell_volumes
    else:
        cell_volumes_to_compare = grid.cell_volumes[0]
    assert np.allclose(cell_volumes_to_compare, expected_geometry[0])
    assert np.allclose(grid.cell_centers, expected_geometry[1])
    assert np.allclose(grid.face_normals, expected_geometry[2])


# ----- Tests for a 2d unperturbed grid ----- #


@pytest.fixture
def grid_2d_unperturbed():
    nc = np.array([2, 3])
    g = structured.CartGrid(nc)
    g.compute_geometry()
    return g


def test_geometry_2d_unperturbed(grid_2d_unperturbed):
    # Expected node coordinates
    xn = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    yn = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    # Expected face areas
    areas = 1 * np.ones(grid_2d_unperturbed.num_faces)
    # Expected cell volumes
    volumes = 1 * np.ones(grid_2d_unperturbed.num_cells)
    assert np.isclose(xn, grid_2d_unperturbed.nodes[0]).all()
    assert np.isclose(yn, grid_2d_unperturbed.nodes[1]).all()
    assert np.isclose(areas, grid_2d_unperturbed.face_areas).all()
    assert np.isclose(volumes, grid_2d_unperturbed.cell_volumes).all()


# ----- Tests for a 2d perturbed grid ----- #


@pytest.fixture
def grid_2d_perturbed():
    nc = np.array([2, 2])
    g = structured.CartGrid(nc)
    g.nodes[0, 4] = 1.5
    g.compute_geometry()
    return g


def test_geometry_2d_perturbed(grid_2d_perturbed):
    # Expected node coordinates
    xn = np.array([0, 1, 2, 0, 1.5, 2, 0, 1, 2])
    yn = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    # Expected face areas
    areas = np.array([1, np.sqrt(1.25), 1, 1, np.sqrt(1.25), 1, 1, 1, 1.5, 0.5, 1, 1])
    # Expected cell volumes
    volumes = np.array([1.25, 0.75, 1.25, 0.75])
    # Expected face normals
    nx = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    ny = np.array([0, -0.5, 0, 0, 0.5, 0, 1, 1, 1.5, 0.5, 1, 1])
    fn = grid_2d_perturbed.face_normals
    assert np.isclose(xn, grid_2d_perturbed.nodes[0]).all()
    assert np.isclose(yn, grid_2d_perturbed.nodes[1]).all()
    assert np.isclose(areas, grid_2d_perturbed.face_areas).all()
    assert np.isclose(volumes, grid_2d_perturbed.cell_volumes).all()
    assert np.isclose(nx, fn[0]).all()
    assert np.isclose(ny, fn[1]).all()


# ----- Tests for a 3d unperturbed grid ----- #


@pytest.fixture
def grid_3d_unperturbed():
    nc = 2 * np.ones(3, dtype=int)
    g = structured.CartGrid(nc)
    g.compute_geometry()
    return g


def test_geometry_3d_unperturbed(grid_3d_unperturbed):
    # Expected node coordinates
    x = reference_dense_arrays.test_grid["test_geometry_3d_unperturbed"]["x"]
    y = reference_dense_arrays.test_grid["test_geometry_3d_unperturbed"]["y"]
    z = reference_dense_arrays.test_grid["test_geometry_3d_unperturbed"]["z"]

    # Expected face areas
    areas = 1 * np.ones(grid_3d_unperturbed.num_faces)

    face_per_dim = 3 * 2 * 2
    ones = np.ones(face_per_dim)
    zeros = np.zeros(face_per_dim)
    # Expected face normals
    nx = np.hstack((ones, zeros, zeros))
    ny = np.hstack((zeros, ones, zeros))
    nz = np.hstack((zeros, zeros, ones))
    # Expected cell volumes
    volumes = 1 * np.ones(grid_3d_unperturbed.num_cells)
    # Expected cell centers
    cx = np.array([0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
    cy = np.array([0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
    cz = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])
    assert np.isclose(x, grid_3d_unperturbed.nodes[0]).all()
    assert np.isclose(y, grid_3d_unperturbed.nodes[1]).all()
    assert np.isclose(z, grid_3d_unperturbed.nodes[2]).all()
    assert np.isclose(areas, grid_3d_unperturbed.face_areas).all()
    assert np.isclose(nx, grid_3d_unperturbed.face_normals[0]).all()
    assert np.isclose(ny, grid_3d_unperturbed.face_normals[1]).all()
    assert np.isclose(nz, grid_3d_unperturbed.face_normals[2]).all()
    assert np.isclose(volumes, grid_3d_unperturbed.cell_volumes).all()
    assert np.isclose(cx, grid_3d_unperturbed.cell_centers[0]).all()
    assert np.isclose(cy, grid_3d_unperturbed.cell_centers[1]).all()
    assert np.isclose(cz, grid_3d_unperturbed.cell_centers[2]).all()


# ----- Tests for a perturbed 3d grid consisting of one cell ----- #


@pytest.fixture
def grid_3d_perturbed():
    nc = np.ones(3, dtype=int)
    g = structured.CartGrid(nc)
    g.nodes[:, -1] = [2, 2, 2]
    g.compute_geometry()
    return g


def test_geometry_3d_perturbed(grid_3d_perturbed):
    # Expected node coordinates
    x = np.array([0, 1, 0, 1, 0, 1, 0, 2])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 2])
    z = np.array([0, 0, 0, 0, 1, 1, 1, 2])
    # Expected face areas
    a1 = 1
    a2 = 2.159875808805010  # Face area computed another place
    areas = np.array([a1, a2, a1, a2, a1, a2])
    # Expected face normals
    nx = np.array([1, 2, 0, -0.5, 0, -0.5])
    ny = np.array([0, -0.5, 1, 2, 0, -0.5])
    nz = np.array([0, -0.5, 0, -0.5, 1, 2])
    # Expected face centers
    f1 = 1.294658198738520
    f2 = 0.839316397477041
    fx = np.array([0, f1, 0.5, f2, 0.5, f2])
    fy = np.array([0.5, f2, 0, f1, 0.5, f2])
    fz = np.array([0.5, f2, 0.5, f2, 0, f1])
    # Expected cell volume
    volume = 1.75
    # Expected cell center
    cx = 0.717261904761905
    cy = cx
    cz = cx
    assert np.isclose(x, grid_3d_perturbed.nodes[0]).all()
    assert np.isclose(y, grid_3d_perturbed.nodes[1]).all()
    assert np.isclose(z, grid_3d_perturbed.nodes[2]).all()
    assert np.isclose(areas, grid_3d_perturbed.face_areas).all()
    assert np.isclose(nx, grid_3d_perturbed.face_normals[0]).all()
    assert np.isclose(ny, grid_3d_perturbed.face_normals[1]).all()
    assert np.isclose(nz, grid_3d_perturbed.face_normals[2]).all()
    assert np.isclose(fx, grid_3d_perturbed.face_centers[0]).all()
    assert np.isclose(fy, grid_3d_perturbed.face_centers[1]).all()
    assert np.isclose(fz, grid_3d_perturbed.face_centers[2]).all()
    assert np.isclose(grid_3d_perturbed.cell_volumes, volume)
    assert np.isclose(cx, grid_3d_perturbed.cell_centers[0]).all()
    assert np.isclose(cy, grid_3d_perturbed.cell_centers[1]).all()
    assert np.isclose(cz, grid_3d_perturbed.cell_centers[2]).all()


# ----- Tests for a structured triangle grid ----- #


@pytest.fixture
def triangle_grid():
    g = simplex.StructuredTriangleGrid(np.array([1, 1]))
    g.compute_geometry()
    return g


def test_geometry_triangle_grid(triangle_grid):
    # Expected node coordinates
    nodes = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
    # Expected face centers
    fc = np.array([[0.5, 0, 0.5, 1, 0.5], [0, 0.5, 0.5, 0.5, 1], [0, 0, 0, 0, 0]])
    # Expected cell centers
    cc = np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3], [0, 0]])
    # Expected cell volumes
    cv = np.array([0.5, 0.5])
    # Expected face areas
    fa = np.array([1, 1, np.sqrt(2), 1, 1])
    # Expected face normals
    fn = np.array([[0, 1, 1, 1, 0], [-1, 0, -1, 0, -1], [0, 0, 0, 0, 0]])
    assert np.allclose(triangle_grid.nodes, nodes)
    assert np.allclose(triangle_grid.face_centers, fc)
    assert np.allclose(triangle_grid.cell_centers, cc)
    assert np.allclose(triangle_grid.cell_volumes, cv)
    assert np.allclose(triangle_grid.face_areas, fa)
    assert np.allclose(triangle_grid.face_normals, fn)


# ----- Tests for a structured tetrahedral grid ----- #


@pytest.fixture
def tetrahedral_grid():
    g = simplex.StructuredTetrahedralGrid(np.array([1, 1, 1]))
    g.compute_geometry()
    return g


def test_geometry_tetrahedral_grid(tetrahedral_grid):
    # Expected node coordinates
    nodes = np.array(
        [
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]
    )
    # Expected cell centers
    cc = np.array(
        [
            [1 / 4, 1 / 4, 1 / 4],
            [1 / 4, 1 / 2, 1 / 2],
            [1 / 2, 1 / 4, 3 / 4],
            [1 / 2, 3 / 4, 1 / 4],
            [3 / 4, 1 / 2, 1 / 2],
            [3 / 4, 3 / 4, 3 / 4],
        ]
    ).T
    # Expected cell volumes
    cv = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])

    # The ordering of faces may differ depending on the test system (presumably
    # version of scipy or similar). Below are hard-coded combination of face-nodes,
    # and the corresponding faces and face_areas.
    # Expected face-node mapping
    fn = np.array(
        [
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 5],
            [1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 3, 4, 5, 5, 6, 5, 6],
            [2, 4, 4, 3, 4, 6, 5, 6, 5, 6, 6, 6, 6, 6, 7, 7, 6, 7],
        ]
    )
    # Expected face areas
    fa = np.array(
        [
            0.5,
            0.5,
            0.5,
            0.5,
            0.8660254,
            0.70710678,
            0.5,
            0.70710678,
            0.5,
            0.70710678,
            0.70710678,
            0.5,
            0.5,
            0.8660254,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
    )
    # Expected face centers - pick from reference file
    fc = reference_dense_arrays.test_grid["test_geometry_tetrahedral_grid"][
        "face_centers"
    ]
    assert np.allclose(tetrahedral_grid.nodes, nodes)
    assert np.allclose(tetrahedral_grid.cell_centers, cc)
    assert np.allclose(tetrahedral_grid.cell_volumes, cv)

    face_nodes = tetrahedral_grid.face_nodes.indices.reshape(
        (3, tetrahedral_grid.num_faces), order="F"
    )
    ismem, ind_map = setmembership.ismember_rows(fn, face_nodes)
    assert np.all(ismem)
    assert np.allclose(tetrahedral_grid.face_centers[:, ind_map], fc)
    assert np.allclose(tetrahedral_grid.face_areas[ind_map], fa)


# Test that verifies that the tesselation covers the whole domain.
@pytest.mark.parametrize(
    "grid, domain",
    [
        (
            simplex.StructuredTriangleGrid(np.array([3, 5]), np.array([2, 3])),
            np.array([2, 3]),
        ),
        (
            simplex.StructuredTetrahedralGrid(
                np.array([3, 5, 2]), np.array([2, 3, 0.7])
            ),
            np.array([2, 3, 0.7]),
        ),
    ],
)
def test_tesselation_coverage(grid, domain):
    grid.compute_geometry()
    assert np.abs(domain.prod() - np.sum(grid.cell_volumes)) < 1e-10


def test_different_grids():
    """Test that different consecutively created grids have consecutive counters."""
    g1 = simplex.StructuredTriangleGrid(np.array([1, 3]))
    g2 = simplex.StructuredTriangleGrid(np.array([1, 3]))
    g3 = simplex.StructuredTetrahedralGrid(np.array([1, 1, 1]))
    g4 = simplex.StructuredTetrahedralGrid(np.array([1, 1, 1]))
    g5 = pp.CartGrid(np.array([1, 1, 1]))
    g6 = pp.TensorGrid(np.array([1, 1, 1]))

    assert g1.id == g2.id - 1
    assert g2.id == g3.id - 1
    assert g3.id == g4.id - 1
    assert g4.id == g5.id - 1
    assert g5.id == g6.id - 1

    # Same for mortar grids
    g1.compute_geometry()
    g2.compute_geometry()
    mg1 = pp.MortarGrid(g1.dim, {0: g1, 1: g1})
    mg2 = pp.MortarGrid(g2.dim, {0: g2, 1: g2})
    assert mg1.id == mg2.id - 1


def test_copy_grids():
    """Test that the id is not copied when copying a grid."""
    g1 = simplex.StructuredTriangleGrid(np.array([1, 1]))
    g2 = g1.copy()
    assert g1.id == g2.id - 1


def test_merge_single_grid():
    """
    Test coupling from one grid to itself. An example setting:
                    |--|--|--| ( grid )
                    0  1  2  3
     (left_coupling) \      / (right coupling)
                      \    /
                        * (mortar grid)

    with a coupling from the left face (0) to the right face (1).
    The mortar grid will just be a point
    """

    face_faces = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    face_faces = sps.csc_matrix(face_faces)
    left_side = pp.PointGrid(np.array([0, 0, 0]).T)
    left_side.compute_geometry()

    intf = pp.grids.mortar_grid.MortarGrid(0, {"0": left_side}, face_faces)

    assert intf.num_cells == 1
    assert intf.num_sides() == 1
    assert np.all(intf.primary_to_mortar_avg().toarray() == [1, 0, 0])
    assert np.all(intf.primary_to_mortar_int().toarray() == [1, 0, 0])
    assert np.all(intf.secondary_to_mortar_avg().toarray() == [0, 0, 1])
    assert np.all(intf.secondary_to_mortar_int().toarray() == [0, 0, 1])


def test_merge_two_grids():
    """
    Test coupling from one grid of three faces to grid of two faces.
    An example setting:
                    0  1  2
                    |--|--| ( left grid)
                       |    (left coupling)
                       *    (mortar_grid
                        \   (right coupling)
                      |--|  (right grid)
                      0  1
    """
    face_faces = np.array([[0, 0, 0], [0, 1, 0]])
    face_faces = sps.csc_matrix(face_faces)
    left_side = pp.PointGrid(np.array([2, 0, 0]).T)
    left_side.compute_geometry()

    intf = pp.grids.mortar_grid.MortarGrid(0, {"0": left_side}, face_faces)

    assert intf.num_cells == 1
    assert intf.num_sides() == 1
    assert np.all(intf.primary_to_mortar_avg().toarray() == [0, 1, 0])
    assert np.all(intf.primary_to_mortar_int().toarray() == [0, 1, 0])
    assert np.all(intf.secondary_to_mortar_avg().toarray() == [0, 1])
    assert np.all(intf.secondary_to_mortar_int().toarray() == [0, 1])


def test_boundary_grid():
    """Test that the boundary grid is created correctly."""
    # First make a standard grid and its derived BoundaryGrid
    g = pp.CartGrid(np.array([2, 2]))
    g.compute_geometry()

    boundary_grid = pp.BoundaryGrid(g)
    boundary_grid.set_projections()

    # Hardcoded value for the number of cells
    assert boundary_grid.num_cells == 8

    proj = boundary_grid.projection()

    assert proj.shape == (8, 12)

    rows, cols, _ = sparse_array_to_row_col_data(proj)
    assert np.allclose(np.sort(rows), np.arange(8))
    # Hardcoded values based on the known ordering of faces in a Cartesian grid
    assert np.allclose(np.sort(cols), np.array([0, 2, 3, 5, 6, 7, 10, 11]))

    # Next, mark all faces in the grid as not on the domain boundary.
    # The boundary grid should then be empty.
    g.tags["domain_boundary_faces"] = np.zeros(g.num_faces, dtype=bool)
    boundary_grid = pp.BoundaryGrid(g)
    boundary_grid.set_projections()
    assert boundary_grid.num_cells == 0
    assert boundary_grid.projection().shape == (0, 12)


@pytest.mark.parametrize(
    "g",
    [
        pp.PointGrid(np.array([0, 0, 0])),
        pp.CartGrid(np.array([2])),
        pp.CartGrid(np.array([2, 2])),
        pp.CartGrid(np.array([2, 2, 2])),
        pp.StructuredTriangleGrid(np.array([2, 2])),
        pp.StructuredTetrahedralGrid(np.array([1, 1, 1])),
    ],
)
def test_pickle_grid(g):
    """Test that grids can be pickled. Write, read and compare."""
    fn = "tmp.grid"
    pickle.dump(g, open(fn, "wb"))

    g_read = pickle.load(open(fn, "rb"))

    pp.test_utils.grids.compare_grids(g, g_read)
    os.unlink(fn)
