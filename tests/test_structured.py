import unittest
import numpy as np

from gridding.fractured import structured


class TestStructured(unittest.TestCase):
    def test_x_intersection_2d(self):
        """ Check that no error messages are created in the process of creating a
        split_fracture.
        """

        f_1 = np.array([[0, 2], [1, 1]])
        f_2 = np.array([[1, 1], [0, 2]])

        f_set = [f_1, f_2]
        nx = [3, 3]

        grids = structured.cart_grid_2d(f_set, nx, physdims=nx)

        num_grids = [1, 2, 1]
        for i, g in enumerate(grids):
            assert len(g) == num_grids[i]

        g_2d = grids[0][0]
        g_1d_1 = grids[1][0]
        g_1d_2 = grids[1][1]
        g_0d = grids[2][0]

        f_nodes_1 = [4, 5, 6]
        f_nodes_2 = [1, 5, 9]
        f_nodes_0 = [5]
        glob_1 = np.sort(g_1d_1.global_point_ind)
        glob_2 = np.sort(g_1d_2.global_point_ind)
        glob_0 = np.sort(g_0d.global_point_ind)
        assert np.all(f_nodes_1 == glob_1)
        assert np.all(f_nodes_2 == glob_2)
        assert np.all(f_nodes_0 == glob_0)

    def test_tripple_x_intersection_3d(self):
        """ Check that no error messages are created in the process of creating a
        split_fracture.
        """
        f_1 = np.array([[1, 4, 4, 1], [3, 3, 3, 3], [1, 1, 4, 4]])
        f_2 = np.array([[3, 3, 3, 3], [1, 4, 4, 1], [1, 1, 5, 5]])
        f_3 = np.array([[1, 1, 4, 4], [1, 4, 4, 1], [3, 3, 3, 3]])

        f_set = [f_1, f_2, f_3]
        nx = np.array([6, 6, 6])

        grids = structured.cart_grid_3d(f_set, nx, physdims=nx)

        num_grids = [1, 3, 6, 1]
        print(grids)
        for i, g in enumerate(grids):
            print(len(g))
            assert len(g) == num_grids[i]

        g_3d = grids[0][0]
        for g_loc in grids[1:]:
            for g in g_loc:
                assert np.allclose(g.node, g_3d.nodes[:, g.global_point_ind])
