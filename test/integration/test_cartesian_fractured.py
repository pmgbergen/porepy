import numpy as np
import unittest

import porepy as pp


class TestCartGridFrac(unittest.TestCase):
    def test_tripple_x_intersection_3d(self):
        """
        Create a cartesian grid in the unit cube, and insert three fractures.
        """

        f1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        f2 = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
        f3 = np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])

        gb = pp.meshing.cart_grid([f1, f2, f3], [2, 2, 2], physdims=[1, 1, 1])
        g_3 = gb.grids_of_dimension(3)
        g_2 = gb.grids_of_dimension(2)
        g_1 = gb.grids_of_dimension(1)
        g_0 = gb.grids_of_dimension(0)

        self.assertTrue(len(g_3) == 1)
        self.assertTrue(len(g_2) == 3)
        self.assertTrue(len(g_1) == 6)
        self.assertTrue(len(g_0) == 1)

        self.assertTrue(np.all([g.num_cells == 4 for g in g_2]))
        self.assertTrue(np.all([g.num_faces == 16 for g in g_2]))
        self.assertTrue(np.all([g.num_cells == 1 for g in g_1]))
        self.assertTrue(np.all([g.num_faces == 2 for g in g_1]))

        g_all = np.hstack([g_3, g_2, g_1, g_0])
        for g in g_all:
            d = np.all(np.abs(g.nodes - np.array([[0.5], [0.5], [0.5]])) < 1e-6, axis=0)
            self.assertTrue(any(d))

    def test_x_intersection_2d(self):
        f_1 = np.array([[2, 6], [5, 5]])
        f_2 = np.array([[4, 4], [2, 7]])
        f = [f_1, f_2]
        gb = pp.meshing.cart_grid(f, [10, 10], physdims=[10, 10])

    def test_T_intersection_2d(self):
        f_1 = np.array([[2, 6], [5, 5]])
        f_2 = np.array([[4, 4], [2, 5]])
        f = [f_1, f_2]
        gb = pp.meshing.cart_grid(f, [10, 10], physdims=[10, 10])

    def test_L_intersection_2d(self):
        f_1 = np.array([[2, 6], [5, 5]])
        f_2 = np.array([[6, 6], [2, 5]])
        f = [f_1, f_2]

        gb = pp.meshing.cart_grid(f, [10, 10], physdims=[10, 10])

    def test_x_intersection_3d(self):
        f_1 = np.array([[2, 5, 5, 2], [2, 2, 5, 5], [5, 5, 5, 5]])
        f_2 = np.array([[2, 2, 5, 5], [5, 5, 5, 5], [2, 5, 5, 2]])
        f = [f_1, f_2]
        gb = pp.meshing.cart_grid(f, np.array([10, 10, 10]))

    def test_several_intersections_3d(self):
        f_1 = np.array([[2, 5, 5, 2], [2, 2, 5, 5], [5, 5, 5, 5]])
        f_2 = np.array([[2, 2, 5, 5], [5, 5, 5, 5], [2, 5, 5, 2]])
        f_3 = np.array([[4, 4, 4, 4], [1, 1, 8, 8], [1, 8, 8, 1]])
        f_4 = np.array([[3, 3, 6, 6], [3, 3, 3, 3], [3, 7, 7, 3]])
        f = [f_1, f_2, f_3, f_4]
        gb = pp.meshing.cart_grid(f, np.array([8, 8, 8]))


if __name__ == "__main__":
    unittest.main()
