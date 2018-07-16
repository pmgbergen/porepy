import unittest
import numpy as np

import porepy as pp


class TestMeshing(unittest.TestCase):
    def test_L_intersection_3d(self):
        """
        Create a L-intersection in 3D
        """

        f_1 = np.array([[1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 3, 3]])
        f_2 = np.array([[1, 1, 1, 1], [1, 3, 3, 1], [1, 1, 3, 3]])
        f_set = [f_1, f_2]

        bucket = pp.meshing.cart_grid(f_set, [3, 3, 3])
        bucket.compute_geometry()

        g_3 = bucket.grids_of_dimension(3)
        g_2 = bucket.grids_of_dimension(2)
        g_1 = bucket.grids_of_dimension(1)
        g_0 = bucket.grids_of_dimension(0)

        assert len(g_3) == 1
        assert len(g_2) == 2
        assert len(g_1) == 1
        assert len(g_0) == 0
        g_all = np.hstack([g_3, g_2, g_1, g_0])
        for g in g_all:
            f_p = g.frac_pairs
            if g.dim == 3:
                f_p_shape_true = 8
            else:
                f_p_shape_true = 0
            assert f_p.shape[1] == f_p_shape_true
            assert np.allclose(g.face_centers[:, f_p[0]], g.face_centers[:, f_p[1]])

    def test_X_intersection_3d(self):
        """
        Create a x-intersection in 3D
        """

        f_1 = np.array([[0, 2, 2, 0], [1, 1, 1, 1], [0, 0, 2, 2]])
        f_2 = np.array([[1, 1, 1, 1], [0, 2, 2, 0], [0, 0, 2, 2]])
        f_set = [f_1, f_2]

        bucket = pp.meshing.cart_grid(f_set, [2, 2, 2])
        bucket.compute_geometry()

        g_3 = bucket.grids_of_dimension(3)
        g_2 = bucket.grids_of_dimension(2)
        g_1 = bucket.grids_of_dimension(1)
        g_0 = bucket.grids_of_dimension(0)

        assert len(g_3) == 1
        assert len(g_2) == 2
        assert len(g_1) == 1
        assert len(g_0) == 0
        g_all = np.hstack([g_3, g_2, g_1, g_0])
        for g in g_all:
            f_p = g.frac_pairs
            if g.dim == 3:
                f_p_shape_true = 8
            elif g.dim == 2:
                f_p_shape_true = 2
            else:
                f_p_shape_true = 0
            assert f_p.shape[1] == f_p_shape_true
            assert np.allclose(g.face_centers[:, f_p[0]], g.face_centers[:, f_p[1]])

    def test_tripple_intersection_3d(self):
        """
        Create a tripple L-intersection in 3D
        """

        f_1 = np.array([[1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 3, 3]])
        f_2 = np.array([[1, 1, 1, 1], [1, 3, 3, 1], [1, 1, 3, 3]])
        f_3 = np.array([[1, 3, 3, 1], [1, 1, 3, 3], [1, 1, 1, 1]])
        f_set = [f_1, f_2, f_3]

        bucket = pp.meshing.cart_grid(f_set, [3, 3, 3])
        bucket.compute_geometry()

        g_3 = bucket.grids_of_dimension(3)
        g_2 = bucket.grids_of_dimension(2)
        g_1 = bucket.grids_of_dimension(1)
        g_0 = bucket.grids_of_dimension(0)

        assert len(g_3) == 1
        assert len(g_2) == 3
        assert len(g_1) == 3
        assert len(g_0) == 1
        g_all = np.hstack([g_3, g_2, g_1, g_0])
        for g in g_all:
            f_p = g.frac_pairs
            if g.dim == 3:
                f_p_shape_true = 12
            else:
                f_p_shape_true = 0
            assert f_p.shape[1] == f_p_shape_true
            assert np.allclose(g.face_centers[:, f_p[0]], g.face_centers[:, f_p[1]])

        assert np.allclose(g_0[0].nodes, np.array([1, 1, 1]))
