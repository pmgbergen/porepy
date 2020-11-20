import unittest
import numpy as np

import porepy as pp


class TestNormalVector(unittest.TestCase):
    """ Test computation of compute_normal().
    """

    def test_axis_normal(self):
        """
        Test that we get the correct normal in x, y and z direction
        """
        # pts in xy-plane
        pts_xy = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]).T
        # pts in xz-plane
        pts_xz = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]]).T
        # pts in yz-plane
        pts_yz = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]]).T

        nz = pp.map_geometry.compute_normal(pts_xy)
        ny = pp.map_geometry.compute_normal(pts_xz)
        nx = pp.map_geometry.compute_normal(pts_yz)

        # The method does not guarantee normal direction. Should it?
        self.assertTrue(
            np.allclose(nz, np.array([0, 0, 1]))
            or np.allclose(nz, -np.array([0, 0, 1]))
        )
        self.assertTrue(
            np.allclose(ny, np.array([0, 1, 0]))
            or np.allclose(ny, -np.array([0, 1, 0]))
        )
        self.assertTrue(
            np.allclose(nx, np.array([1, 0, 0]))
            or np.allclose(nx, -np.array([1, 0, 0]))
        )

    def test_colinear_points(self):
        """
        Test that giving colinear points throws an assertion
        """
        # pts in xy-plane
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]).T

        with self.assertRaises(RuntimeError):
            _ = pp.map_geometry.compute_normal(pts)
