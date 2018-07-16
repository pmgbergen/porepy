import numpy as np
import unittest

import porepy as pp


class TestCartGridFrac(unittest.TestCase):
    def test_tripple_x_intersection_3d(self):
        """
        Create a cartesian grid in the unit cube, and insert three fractures.
        """

        f1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [.5, .5, .5, .5]])
        f2 = np.array([[.5, .5, .5, .5], [0, 1, 1, 0], [0, 0, 1, 1]])
        f3 = np.array([[0, 1, 1, 0], [.5, .5, .5, .5], [0, 0, 1, 1]])

        gb = pp.meshing.cart_grid([f1, f2, f3], [2, 2, 2], physdims=[1, 1, 1])
        g_3 = gb.grids_of_dimension(3)
        g_2 = gb.grids_of_dimension(2)
        g_1 = gb.grids_of_dimension(1)
        g_0 = gb.grids_of_dimension(0)

        assert len(g_3) == 1
        assert len(g_2) == 3
        assert len(g_1) == 6
        assert len(g_0) == 1

        assert np.all([g.num_cells == 4 for g in g_2])
        assert np.all([g.num_faces == 16 for g in g_2])
        assert np.all([g.num_cells == 1 for g in g_1])
        assert np.all([g.num_faces == 2 for g in g_1])

        g_all = np.hstack([g_3, g_2, g_1, g_0])
        for g in g_all:
            d = np.all(np.abs(g.nodes - np.array([[.5], [.5], [.5]])) < 1e-6, axis=0)
            assert any(d)
