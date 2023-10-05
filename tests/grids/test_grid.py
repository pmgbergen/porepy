""" Various tests related to grids:
* The main grid class, including geometry computation.
* Specific tests for Simplex and Structured Grids
* Tests for the mortar grid.
"""
import pickle
import unittest

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.grids import simplex, structured
from porepy.utils import setmembership
from tests import test_utils


def set_tol():
    return 1e-8


def test_cell_diameters_2d():
    g = pp.CartGrid(np.array([3, 2]), np.array([1, 1]))
    cell_diameters = g.cell_diameters()
    known = np.repeat(np.sqrt(0.5**2 + 1.0 / 3.0**2), g.num_cells)
    assert np.allclose(cell_diameters, known)


def test_cell_diameters_3d():
    g = pp.CartGrid(np.array([3, 2, 1]))
    cell_diameters = g.cell_diameters()
    known = np.repeat(np.sqrt(3), g.num_cells)
    assert np.allclose(cell_diameters, known)


def test_repr():
    # Call repr, just to see that it works
    g = pp.CartGrid(np.array([1, 1]))
    g.__repr__()


def test_str():
    # Call str, just to see that it works
    g = pp.CartGrid(np.array([1, 1]))
    g.__str__()


# Tests of function to find the closest cell

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


def test_cell_face_as_dense():
    g = pp.CartGrid(np.array([2, 1]))
    cf = g.cell_face_as_dense()
    known = np.array([[-1, 0, 1, -1, -1, 0, 1], [0, 1, -1, 0, 1, -1, -1]])
    assert np.allclose(cf, known)

# Boundary tests


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


# Tests of the compute_geometry function

def test_compute_geometry_unit_cart_2d():
    g = pp.CartGrid(np.array([1, 1]))
    g.compute_geometry()
    known_areas = np.ones(4)
    known_volumes = np.ones(1)
    known_normals = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]).T
    known_f_centr = np.array([[0, 0.5, 0], [1, 0.5, 0], [0.5, 0, 0], [0.5, 1, 0]]).T
    known_c_centr = np.array([[0.5, 0.5, 0]]).T
    assert np.allclose(g.face_areas, known_areas)
    assert np.allclose(g.cell_volumes, known_volumes)
    assert np.allclose(g.face_normals, known_normals)
    assert np.allclose(g.face_centers, known_f_centr)
    assert np.allclose(g.cell_centers, known_c_centr)


def test_compute_geometry_huge_cart_2d():
    g = pp.CartGrid(np.array([1, 1]), np.array([5000, 5000]))
    g.compute_geometry()
    known_areas = 5000 * np.ones(4)
    known_volumes = 5000**2 * np.ones(1)
    known_normals = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]).T * 5000
    known_f_centr = (
        np.array([[0, 0.5, 0], [1, 0.5, 0], [0.5, 0, 0], [0.5, 1, 0]]).T * 5000
    )
    known_c_centr = np.array([[0.5, 0.5, 0]]).T * 5000
    assert np.allclose(g.face_areas, known_areas)
    assert np.allclose(g.cell_volumes, known_volumes)
    assert np.allclose(g.face_normals, known_normals)
    assert np.allclose(g.face_centers, known_f_centr)
    assert np.allclose(g.cell_centers, known_c_centr)


def test_compute_geometry_inconsistent_orientation_2d():
    # Test to trigger is_oriented = False such that compute_geometry falls
    # back on the implementation based on convex cells.

    nodes = np.array([[0, 1, 0, 1], [0, 0, 1, 1], np.zeros(4)])
    indices = np.array([0, 1, 2, 3, 0, 2, 1, 3])
    face_nodes = sps.csc_matrix((np.ones(8), indices, np.arange(0, 9, 2)))
    cell_faces = sps.csc_matrix(np.ones((4, 1)))

    sd = pp.Grid(2, nodes, face_nodes, cell_faces, "inconsistent")
    sd.compute_geometry()

    assert np.isclose(sd.cell_volumes[0], 1)
    assert np.allclose(sd.cell_centers, np.array([[0.5], [0.5], [0]]))
    known_face_normals = np.array(
        [[0.0, -0.0, -1.0, 1.0], [-1.0, 1.0, -0.0, 0.0], [0.0, -0.0, -0.0, 0.0]]
    )
    assert np.allclose(sd.face_normals, known_face_normals)


def test_compute_geometry_concave_quad():
    # Known quadrilateral for which the convexity assumption fails.

    nodes = np.array([[0, 0.5, 1, 0.5], [0, 0.5, 0, 1], np.zeros(4)])
    indices = np.array([0, 1, 1, 2, 2, 3, 3, 0])
    face_nodes = sps.csc_matrix((np.ones(8), indices, np.arange(0, 9, 2)))
    cell_faces = sps.csc_matrix(np.ones((4, 1)))

    sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
    sd.compute_geometry()

    assert np.isclose(sd.cell_volumes[0], 0.25)
    assert np.allclose(sd.cell_centers, np.array([[0.5], [0.5], [0]]))
    known_face_normals = np.array(
        [[0.5, -0.5, 1.0, -1.0], [-0.5, -0.5, 0.5, 0.5], [0.0, 0.0, -0.0, 0.0]]
    )
    assert np.allclose(sd.face_normals, known_face_normals)


def test_compute_geometry_exterior_cell_center():
    # Known quadrilateral for which the convexity assumption fails
    # and the centroid is external.

    nodes = np.array([[0, 0.5, 1, 0.5], [0, 0.75, 0, 1], np.zeros(4)])
    indices = np.array([0, 1, 1, 2, 2, 3, 3, 0])
    face_nodes = sps.csc_matrix((np.ones(8), indices, np.arange(0, 9, 2)))
    cell_faces = sps.csc_matrix(np.ones((4, 1)))

    sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
    sd.compute_geometry()

    assert np.isclose(sd.cell_volumes[0], 0.125)
    assert np.allclose(sd.cell_centers, np.array([[0.5], [1.75 / 3], [0]]))
    known_face_normals = np.array(
        [[0.75, -0.75, 1.0, -1.0], [-0.5, -0.5, 0.5, 0.5], [0.0, 0.0, -0.0, 0.0]]
    )
    assert np.allclose(sd.face_normals, known_face_normals)


def test_compute_geometry_multiple_concave_quads():
    # Small mesh consisting of two concave and one convex cell.

    nodes = np.array(
        [[0, 0.5, 1, 0.5, 0.5, 0.5], [0, 0.5, 0, 1, -0.5, -1], np.zeros(6)]
    )
    indices = np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 5, 5, 2, 2, 4, 4, 0])
    face_nodes = sps.csc_matrix((np.ones(16), indices, np.arange(0, 17, 2)))
    cell_faces_j = np.repeat(np.arange(3), 4)
    cell_faces_i = np.array([0, 1, 2, 3, 7, 6, 1, 0, 4, 5, 6, 7])
    cell_faces_v = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1])
    cell_faces = sps.csc_matrix((cell_faces_v, (cell_faces_i, cell_faces_j)))

    sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
    sd.compute_geometry()

    assert np.allclose(sd.cell_volumes, [0.25, 0.5, 0.25])
    known_cell_centers = np.array(
        [[0.5, 0.5, 0.5], [0.5, 0.0, -0.5], [0.0, 0.0, 0.0]]
    )
    assert np.allclose(sd.cell_centers, known_cell_centers)
    known_face_normals = np.array(
        [
            [0.5, -0.5, 1.0, -1.0, -1.0, 1.0, -0.5, 0.5],
            [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
            [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0],
        ]
    )
    assert np.allclose(sd.face_normals, known_face_normals)


# Tests for a 2d unperturbed grid

@pytest.fixture
def grid_2d_unperturbed():
    nc = np.array([2, 3])
    g = structured.CartGrid(nc)
    g.compute_geometry()
    return g


def test_node_coord_2d_unperturbed(grid_2d_unperturbed):
    xn = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    yn = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    assert np.isclose(xn, grid_2d_unperturbed.nodes[0]).all()
    assert np.isclose(yn, grid_2d_unperturbed.nodes[1]).all()


def test_faces_areas_2d_unperturbed(grid_2d_unperturbed):
    areas = 1 * np.ones(grid_2d_unperturbed.num_faces)
    assert np.isclose(areas, grid_2d_unperturbed.face_areas).all()


def test_cell_volumes_2d_unperturbed(grid_2d_unperturbed):
    volumes = 1 * np.ones(grid_2d_unperturbed.num_cells)
    assert np.isclose(volumes, grid_2d_unperturbed.cell_volumes).all()


# Tests for a 2d perturbed grid.

@pytest.fixture
def grid_2d_perturbed():
    nc = np.array([2, 2])
    g = structured.CartGrid(nc)
    g.nodes[0, 4] = 1.5
    g.compute_geometry()
    return g


def test_node_coord_2d_perturbed(grid_2d_perturbed):
    xn = np.array([0, 1, 2, 0, 1.5, 2, 0, 1, 2])
    yn = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert np.isclose(xn, grid_2d_perturbed.nodes[0]).all()
    assert np.isclose(yn, grid_2d_perturbed.nodes[1]).all()


def test_faces_areas_2d_perturbed(grid_2d_perturbed):
    areas = np.array(
        [1, np.sqrt(1.25), 1, 1, np.sqrt(1.25), 1, 1, 1, 1.5, 0.5, 1, 1]
    )
    assert np.isclose(areas, grid_2d_perturbed.face_areas).all()


def test_cell_volumes_2d_perturbed(grid_2d_perturbed):
    volumes = np.array([1.25, 0.75, 1.25, 0.75])
    assert np.isclose(volumes, grid_2d_perturbed.cell_volumes).all()


def test_face_normals_2d_perturbed(grid_2d_perturbed):
    nx = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    ny = np.array([0, -0.5, 0, 0, 0.5, 0, 1, 1, 1.5, 0.5, 1, 1])
    fn = grid_2d_perturbed.face_normals
    assert np.isclose(nx, fn[0]).all()
    assert np.isclose(ny, fn[1]).all()


# Tests for a 3d unperturbed grid

@pytest.fixture
def grid_3d_unperturbed():
    nc = 2 * np.ones(3, dtype=int)
    g = structured.CartGrid(nc)
    g.compute_geometry()
    return g


def test_node_coord_3d_unperturbed(grid_3d_unperturbed):
    x = np.array(
        [
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,  # z == 0
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,  # z == 1
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
        ]
    )  # z == 2
    y = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            2,
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            2,
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            2,
        ]
    )
    z = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
    )

    assert np.isclose(x, grid_3d_unperturbed.nodes[0]).all()
    assert np.isclose(y, grid_3d_unperturbed.nodes[1]).all()
    assert np.isclose(z, grid_3d_unperturbed.nodes[2]).all()


def test_faces_areas_3d_unperturbed(grid_3d_unperturbed):
    areas = 1 * np.ones(grid_3d_unperturbed.num_faces)
    assert np.isclose(areas, grid_3d_unperturbed.face_areas).all()


def test_face_normals_3d_unperturbed(grid_3d_unperturbed):
    face_per_dim = 3 * 2 * 2
    ones = np.ones(face_per_dim)
    zeros = np.zeros(face_per_dim)

    nx = np.hstack((ones, zeros, zeros))
    ny = np.hstack((zeros, ones, zeros))
    nz = np.hstack((zeros, zeros, ones))

    fn = grid_3d_unperturbed.face_normals
    assert np.isclose(nx, fn[0]).all()
    assert np.isclose(ny, fn[1]).all()
    assert np.isclose(nz, fn[2]).all()


def test_cell_volumes_3d_unperturbed(grid_3d_unperturbed):
    volumes = 1 * np.ones(grid_3d_unperturbed.num_cells)
    assert np.isclose(volumes, grid_3d_unperturbed.cell_volumes).all()


def test_cell_centers_3d_unperturbed(grid_3d_unperturbed):
    cx = np.array([0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
    cy = np.array([0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
    cz = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])

    cc = grid_3d_unperturbed.cell_centers

    assert np.isclose(cx, cc[0]).all()
    assert np.isclose(cy, cc[1]).all()
    assert np.isclose(cz, cc[2]).all()


class TestCartGridGeometry1CellPert3D(unittest.TestCase):
    def setUp(self):
        nc = np.ones(3, dtype=int)
        g = structured.CartGrid(nc)
        g.nodes[:, -1] = [2, 2, 2]
        g.compute_geometry()
        self.g = g

    def node_coord(self):
        x = np.array([0, 1, 0, 1, 0, 1, 0, 2])
        y = np.array([0, 0, 1, 1, 0, 0, 1, 2])
        z = np.array([0, 0, 0, 0, 1, 1, 1, 2])

        self.assertTrue(np.isclose(x, self.g.nodes[0]).all())
        self.assertTrue(np.isclose(y, self.g.nodes[1]).all())
        self.assertTrue(np.isclose(z, self.g.nodes[2]).all())

    def test_faces_areas(self):
        a1 = 1
        a2 = 2.159875808805010  # Face area computed another place
        areas = np.array([a1, a2, a1, a2, a1, a2])
        self.assertTrue(np.isclose(areas, self.g.face_areas).all())

    def test_face_normals(self):
        nx = np.array([1, 2, 0, -0.5, 0, -0.5])
        ny = np.array([0, -0.5, 1, 2, 0, -0.5])
        nz = np.array([0, -0.5, 0, -0.5, 1, 2])

        fn = self.g.face_normals
        self.assertTrue(np.isclose(nx, fn[0]).all())
        self.assertTrue(np.isclose(ny, fn[1]).all())
        self.assertTrue(np.isclose(nz, fn[2]).all())

    def test_face_centers(self):
        f1 = 1.294658198738520
        f2 = 0.839316397477041

        fx = np.array([0, f1, 0.5, f2, 0.5, f2])
        fy = np.array([0.5, f2, 0, f1, 0.5, f2])
        fz = np.array([0.5, f2, 0.5, f2, 0, f1])

        fc = self.g.face_centers

        self.assertTrue(np.isclose(fx, fc[0]).all())
        self.assertTrue(np.isclose(fy, fc[1]).all())
        self.assertTrue(np.isclose(fz, fc[2]).all())

    def test_cell_volumes(self):
        self.assertTrue(np.isclose(self.g.cell_volumes, 1.75))

    def test_cell_centers(self):
        # Center coordinate imported from other places
        cx = 0.717261904761905
        cy = cx
        cz = cx
        cc = self.g.cell_centers
        self.assertTrue(np.isclose(cx, cc[0]).all())
        self.assertTrue(np.isclose(cy, cc[1]).all())
        self.assertTrue(np.isclose(cz, cc[2]).all())


class TestStructuredTriangleGridGeometry(unittest.TestCase):
    """Create simplest possible configuration, test sanity of geometric
    quantities.
    """

    def setUp(self):
        g = simplex.StructuredTriangleGrid(np.array([1, 1]))
        g.compute_geometry()
        self.g = g

    def test_node_coords(self):
        nodes = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
        self.assertTrue(np.allclose(self.g.nodes, nodes))

    def test_face_centers(self):
        fc = np.array([[0.5, 0, 0.5, 1, 0.5], [0, 0.5, 0.5, 0.5, 1], [0, 0, 0, 0, 0]])
        self.assertTrue(np.allclose(self.g.face_centers, fc))

    def test_cell_centers(self):
        cc = np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3], [0, 0]])
        self.assertTrue(np.allclose(self.g.cell_centers, cc))

    def test_cell_volumes(self):
        cv = np.array([0.5, 0.5])
        self.assertTrue(np.allclose(self.g.cell_volumes, cv))

    def test_face_areas(self):
        fa = np.array([1, 1, np.sqrt(2), 1, 1])
        self.assertTrue(np.allclose(self.g.face_areas, fa))

    def test_face_normals(self):
        fn = np.array([[0, 1, 1, 1, 0], [-1, 0, -1, 0, -1], [0, 0, 0, 0, 0]])
        self.assertTrue(np.allclose(self.g.face_normals, fn))


class TestStructuredTetrahedralGrid(unittest.TestCase):
    def setUp(self):
        g = simplex.StructuredTetrahedralGrid(np.array([1, 1, 1]))
        g.compute_geometry()
        self.g = g

        # The ordering of faces may differ depending on the test system (presumably
        # version of scipy or similar). Below are hard-coded combination of face-nodes,
        # and the corresponding faces and face_areas.
        self.fn = np.array(
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 5],
                [1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 3, 4, 5, 5, 6, 5, 6],
                [2, 4, 4, 3, 4, 6, 5, 6, 5, 6, 6, 6, 6, 6, 7, 7, 6, 7],
            ]
        )
        self.face_areas = np.array(
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
        self.face_center = np.array(
            [
                [
                    0.33333333,
                    0.33333333,
                    0.0,
                    0.66666667,
                    0.33333333,
                    0.33333333,
                    1.0,
                    0.66666667,
                    0.66666667,
                    0.33333333,
                    0.66666667,
                    0.33333333,
                    0.0,
                    0.66666667,
                    1.0,
                    0.66666667,
                    0.33333333,
                    0.66666667,
                ],
                [
                    0.33333333,
                    0.0,
                    0.33333333,
                    0.66666667,
                    0.33333333,
                    0.66666667,
                    0.33333333,
                    0.66666667,
                    0.0,
                    0.33333333,
                    0.33333333,
                    1.0,
                    0.66666667,
                    0.66666667,
                    0.66666667,
                    1.0,
                    0.33333333,
                    0.66666667,
                ],
                [
                    0.0,
                    0.33333333,
                    0.33333333,
                    0.0,
                    0.33333333,
                    0.33333333,
                    0.33333333,
                    0.33333333,
                    0.66666667,
                    0.66666667,
                    0.66666667,
                    0.33333333,
                    0.66666667,
                    0.66666667,
                    0.66666667,
                    0.66666667,
                    1.0,
                    1.0,
                ],
            ]
        )

    def test_face_centers_areas(self):
        face_nodes = self.g.face_nodes.indices.reshape((3, self.g.num_faces), order="F")
        ismem, ind_map = setmembership.ismember_rows(self.fn, face_nodes)
        self.assertTrue(np.all(ismem))

        self.assertTrue(np.allclose(self.face_areas, self.g.face_areas[ind_map]))
        self.assertTrue(np.allclose(self.face_center, self.g.face_centers[:, ind_map]))

    def test_node_coords(self):
        nodes = np.array(
            [
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ]
        )
        self.assertTrue(np.allclose(self.g.nodes, nodes))

    def test_cell_centers(self):
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
        self.assertTrue(np.allclose(self.g.cell_centers, cc))

    def test_cell_volumes(self):
        cv = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        self.assertTrue(np.allclose(self.g.cell_volumes, cv))


class TestStructuredSimplexGridCoverage(unittest.TestCase):
    """Verify that the tessalation covers the whole domain."""

    def test_coverage_triangles(self):
        domain = np.array([2, 3])
        grid_size = np.array([3, 5])
        g = simplex.StructuredTriangleGrid(grid_size, domain)
        g.compute_geometry()
        self.assertTrue(np.abs(domain.prod() - np.sum(g.cell_volumes)) < 1e-10)

    def test_coverage_tets(self):
        domain = np.array([2, 3, 0.7])
        grid_size = np.array([3, 5, 2])
        g = simplex.StructuredTetrahedralGrid(grid_size, domain)
        g.compute_geometry()
        self.assertTrue(np.abs(domain.prod() - np.sum(g.cell_volumes)) < 1e-10)


class TestGridCounter(unittest.TestCase):
    def test_different_grids(self):
        """Test that different consecutively created grids have consecutive counters."""
        g1 = simplex.StructuredTriangleGrid(np.array([1, 3]))
        g2 = simplex.StructuredTriangleGrid(np.array([1, 3]))
        g3 = simplex.StructuredTetrahedralGrid(np.array([1, 1, 1]))
        g4 = simplex.StructuredTetrahedralGrid(np.array([1, 1, 1]))
        g5 = pp.CartGrid(np.array([1, 1, 1]))
        g6 = pp.TensorGrid(np.array([1, 1, 1]))

        self.assertTrue(g1.id == g2.id - 1)
        self.assertTrue(g2.id == g3.id - 1)
        self.assertTrue(g3.id == g4.id - 1)
        self.assertTrue(g4.id == g5.id - 1)
        self.assertTrue(g5.id == g6.id - 1)

        # Same for mortar grids
        g1.compute_geometry()
        g2.compute_geometry()
        mg1 = pp.MortarGrid(g1.dim, {0: g1, 1: g1})
        mg2 = pp.MortarGrid(g2.dim, {0: g2, 1: g2})
        self.assertTrue(mg1.id == mg2.id - 1)

    def test_copy_grids(self):
        """Test that the id is not copied when copying a grid."""
        g1 = simplex.StructuredTriangleGrid(np.array([1, 1]))
        g2 = g1.copy()
        self.assertTrue(g1.id == g2.id - 1)


class TestGridMappings1d(unittest.TestCase):
    def test_merge_single_grid(self):
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

        self.assertTrue(intf.num_cells == 1)
        self.assertTrue(intf.num_sides() == 1)
        self.assertTrue(np.all(intf.primary_to_mortar_avg().A == [1, 0, 0]))
        self.assertTrue(np.all(intf.primary_to_mortar_int().A == [1, 0, 0]))
        self.assertTrue(np.all(intf.secondary_to_mortar_avg().A == [0, 0, 1]))
        self.assertTrue(np.all(intf.secondary_to_mortar_int().A == [0, 0, 1]))

    def test_merge_two_grids(self):
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

        self.assertTrue(intf.num_cells == 1)
        self.assertTrue(intf.num_sides() == 1)
        self.assertTrue(np.all(intf.primary_to_mortar_avg().A == [0, 1, 0]))
        self.assertTrue(np.all(intf.primary_to_mortar_int().A == [0, 1, 0]))
        self.assertTrue(np.all(intf.secondary_to_mortar_avg().A == [0, 1]))
        self.assertTrue(np.all(intf.secondary_to_mortar_int().A == [0, 1]))


def test_boundary_grid():
    """Test that the boundary grid is created correctly."""
    # First make a standard grid and its derived BoundaryGrid
    g = pp.CartGrid([2, 2])
    g.compute_geometry()

    boundary_grid = pp.BoundaryGrid(g)
    boundary_grid.set_projections()

    # Hardcoded value for the number of cells
    assert boundary_grid.num_cells == 8

    proj = boundary_grid.projection

    assert proj.shape == (8, 12)

    rows, cols, _ = sps.find(proj)
    assert np.allclose(np.sort(rows), np.arange(8))
    # Hardcoded values based on the known ordering of faces in a Cartesian grid
    assert np.allclose(np.sort(cols), np.array([0, 2, 3, 5, 6, 7, 10, 11]))

    # Next, mark all faces in the grid as not on the domain boundary.
    # The boundary grid should then be empty.
    g.tags["domain_boundary_faces"] = np.zeros(g.num_faces, dtype=bool)
    boundary_grid = pp.BoundaryGrid(g)
    boundary_grid.set_projections()
    assert boundary_grid.num_cells == 0
    assert boundary_grid.projection.shape == (0, 12)


@pytest.mark.parametrize(
    "g",
    [
        pp.PointGrid([0, 0, 0]),
        pp.CartGrid([2]),
        pp.CartGrid([2, 2]),
        pp.CartGrid([2, 2, 2]),
        pp.StructuredTriangleGrid([2, 2]),
        pp.StructuredTetrahedralGrid(np.array([1, 1, 1])),
    ],
)
def test_pickle_grid(g):
    """Test that grids can be pickled. Write, read and compare."""
    fn = "tmp.grid"
    pickle.dump(g, open(fn, "wb"))

    g_read = pickle.load(open(fn, "rb"))

    test_utils.compare_grids(g, g_read)

    test_utils.delete_file(fn)


@pytest.mark.parametrize(
    "g",
    [
        pp.PointGrid([0, 0, 0]),
        pp.CartGrid([2]),
        pp.CartGrid([2, 2]),
        pp.StructuredTriangleGrid([2, 2]),
    ],
)
def test_pickle_mortar_grid(g):
    fn = "tmp.grid"
    g.compute_geometry()
    intf = pp.MortarGrid(g.dim, {0: g, 1: g})

    pickle.dump(intf, open(fn, "wb"))
    intf_read = pickle.load(open(fn, "rb"))

    test_utils.compare_mortar_grids(intf, intf_read)

    intf_one_sided = pp.MortarGrid(g.dim, {0: g})

    pickle.dump(intf, open(fn, "wb"))
    intf_read = pickle.load(open(fn, "rb"))

    test_utils.compare_mortar_grids(intf_one_sided, intf_read)

    test_utils.delete_file(fn)


if __name__ == "__main__":
    unittest.main()
