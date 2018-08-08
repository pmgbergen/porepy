import numpy as np
import unittest

from porepy.utils import comp_geom as cg

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_compute_normal_2d(self):
        pts = np.array([[0., 2., -1.], [0., 4., 2.], [0., 0., 0.]])
        normal = cg.compute_normal(pts)
        normal_test = np.array([0., 0., 1.])
        pt = pts[:, 0]

        assert np.allclose(np.linalg.norm(normal), 1.)
        assert np.allclose(
            [np.dot(normal, p - pt) for p in pts[:, 1:].T], np.zeros(pts.shape[1] - 1)
        )
        assert np.allclose(normal, normal_test)

    # ------------------------------------------------------------------------------#

    def test_compute_normal_3d(self):
        pts = np.array([[2., 0., 1., 1.], [1., -2., -1., 1.], [-1., 0., 2., -8.]])
        normal_test = np.array([7., -5., -1.])
        normal_test = normal_test / np.linalg.norm(normal_test)
        normal = cg.compute_normal(pts)
        pt = pts[:, 0]

        assert np.allclose(np.linalg.norm(normal), 1.)
        assert np.allclose(
            [np.dot(normal, p - pt) for p in pts[:, 1:].T], np.zeros(pts.shape[1] - 1)
        )
        assert np.allclose(normal, normal_test) or np.allclose(
            normal, -1. * normal_test
        )

    # ------------------------------------------------------------------------------#

    def test_is_planar_2d(self):
        pts = np.array([[0., 2., -1.], [0., 4., 2.], [2., 2., 2.]])
        assert cg.is_planar(pts)

    # ------------------------------------------------------------------------------#

    def test_is_planar_3d(self):
        pts = np.array(
            [
                [0., 1., 0., 4. / 7.],
                [0., 1., 1., 0.],
                [5. / 8., 7. / 8., 7. / 4., 1. / 8.],
            ]
        )
        assert cg.is_planar(pts)

    # ------------------------------------------------------------------------------#

    def test_project_plane(self):
        pts = np.array([[2., 0., 1., 1.], [1., -2., -1., 1.], [-1., 0., 2., -8.]])
        R = cg.project_plane_matrix(pts)
        P_pts = np.dot(R, pts)

        assert np.allclose(P_pts[2, :], -1.15470054 * np.ones(4))


# ------------------------------------------------------------------------------#
