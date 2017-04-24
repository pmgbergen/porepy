import unittest
import numpy as np

from gridding.fractured import meshing


class TestMeshing(unittest.TestCase):
    def test_x_intersection_3d(self):
        """
        Create a x-intersection in 3D
        """

        f_1 = np.array([[1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 3, 3]])
        f_2 = np.array([[1, 1, 1, 1], [1, 3, 3, 1], [1, 1, 3, 3]])
        f_set = [f_1, f_2]

        bucket = meshing.cart_grid(f_set, [3, 3, 3])
        bucket.compute_geometry()

        g_3 = bucket.grids_of_dimension(3)
        g_2 = bucket.grids_of_dimension(2)
        g_1 = bucket.grids_of_dimension(1)
        g_0 = bucket.grids_of_dimension(0)

        assert len(g_3) == 1
        assert len(g_2) == 2
        assert len(g_1) == 1
        assert len(g_0) == 0

    def test_tripple_intersection_3d(self):
        """
        Create a x-intersection in 3D
        """

        f_1 = np.array([[1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 3, 3]])
        f_2 = np.array([[1, 1, 1, 1], [1, 3, 3, 1], [1, 1, 3, 3]])
        f_3 = np.array([[1, 3, 3, 1], [1, 1, 3, 3], [1, 1, 1, 1]])
        f_set = [f_1, f_2, f_3]

        bucket = meshing.cart_grid(f_set, [3, 3, 3])
        bucket.compute_geometry()

        g_3 = bucket.grids_of_dimension(3)
        g_2 = bucket.grids_of_dimension(2)
        g_1 = bucket.grids_of_dimension(1)
        g_0 = bucket.grids_of_dimension(0)

        assert len(g_3) == 1
        assert len(g_2) == 3
        assert len(g_1) == 3
        assert len(g_0) == 1
