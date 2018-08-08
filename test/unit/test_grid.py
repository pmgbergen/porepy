""" Various tests related to the main Grid class.
"""

import numpy as np
import unittest

import porepy as pp

# ------------------------------------------------------------------------------#


class TestDiameterComputation(unittest.TestCase):
    def test_cell_diameters_2d(self):
        g = pp.CartGrid([3, 2], [1, 1])
        cell_diameters = g.cell_diameters()
        known = np.repeat(np.sqrt(0.5 ** 2 + 1. / 3. ** 2), g.num_cells)
        assert np.allclose(cell_diameters, known)

    def test_cell_diameters_3d(self):
        g = pp.CartGrid([3, 2, 1])
        cell_diameters = g.cell_diameters()
        known = np.repeat(np.sqrt(3), g.num_cells)
        assert np.allclose(cell_diameters, known)


class TestReprAndStr(unittest.TestCase):
    def test_repr(self):
        # Call repr, just to see that it works
        g = pp.CartGrid([1, 1])
        g.__repr__()

    def test_str(self):
        # Call repr, just to see that it works
        g = pp.CartGrid([1, 1])
        g.__str__()


class TestClosestCell(unittest.TestCase):
    # Test of function to find closest cell

    def test_in_plane(self):
        # 2d grid, points also in plane
        g = pp.CartGrid([3, 3])
        g.compute_geometry()

        p = np.array([[0.5, -0.5], [0.5, 0.5]])
        ind = g.closest_cell(p)
        assert np.allclose(ind, 0 * ind)

    def test_out_of_plane(self):
        g = pp.CartGrid([2, 2])
        g.compute_geometry()
        p = np.array([[.1], [.1], [1]])
        ind = g.closest_cell(p)
        assert ind[0] == 0
        assert ind.size == 1


class TestCellFaceAsDense(unittest.TestCase):
    def test_cart_grid(self):
        g = pp.CartGrid([2, 1])
        cf = g.cell_face_as_dense()
        known = np.array([[-1, 0, 1, -1, -1, 0, 1], [0, 1, -1, 0, 1, -1, -1]])
        assert np.allclose(cf, known)


class TestBoundaries(unittest.TestCase):
    def test_bounary_node_cart(self):
        g = pp.CartGrid([2, 2])
        bound_ind = g.get_boundary_nodes()
        known_bound = np.array([0, 1, 2, 3, 5, 6, 7, 8])
        assert np.allclose(np.sort(bound_ind), known_bound)

    def test_boundary_faces_cart(self):
        g = pp.CartGrid([2, 1])
        int_faces = g.get_internal_faces()
        assert int_faces.size == 1
        assert int_faces[0] == 1

    def test_get_internal_nodes_cart(self):
        g = pp.CartGrid([2, 2])
        int_nodes = g.get_internal_nodes()
        assert int_nodes.size == 1
        assert int_nodes[0] == 4

    def test_get_internal_nodes_empty(self):
        g = pp.CartGrid([2, 1])
        int_nodes = g.get_internal_nodes()
        assert int_nodes.size == 0

    def test_bounding_box(self):
        g = pp.CartGrid([1, 1])
        g.nodes = np.random.random((g.dim, g.num_nodes))

        bmin, bmax = g.bounding_box()
        assert np.allclose(bmin, g.nodes.min(axis=1))
        assert np.allclose(bmax, g.nodes.max(axis=1))


if __name__ == "__main__":
    unittest.main()
