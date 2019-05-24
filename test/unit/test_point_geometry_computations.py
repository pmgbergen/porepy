""" Test of geometry operations on point clouds: normal computations, planarity.
"""

import numpy as np
import unittest

import porepy as pp


class BasicsTest(unittest.TestCase):

    def test_compute_normal_2d(self):
        pts = np.array([[0.0, 2.0, -1.0], [0.0, 4.0, 2.0], [0.0, 0.0, 0.0]])
        normal = pp.map_geometry.compute_normal(pts)
        normal_test = np.array([0.0, 0.0, 1.0])
        pt = pts[:, 0]

        self.assertTrue(np.allclose(np.linalg.norm(normal), 1.0))
        self.assertTrue(
            np.allclose(
                [np.dot(normal, p - pt) for p in pts[:, 1:].T],
                np.zeros(pts.shape[1] - 1),
            )
        )
        self.assertTrue(np.allclose(normal, normal_test))

    def test_compute_normal_3d(self):
        pts = np.array(
            [[2.0, 0.0, 1.0, 1.0], [1.0, -2.0, -1.0, 1.0], [-1.0, 0.0, 2.0, -8.0]]
        )
        normal_test = np.array([7.0, -5.0, -1.0])
        normal_test = normal_test / np.linalg.norm(normal_test)
        normal = pp.map_geometry.compute_normal(pts)
        pt = pts[:, 0]

        self.assertTrue(np.allclose(np.linalg.norm(normal), 1.0))
        self.assertTrue(
            np.allclose(
                [np.dot(normal, p - pt) for p in pts[:, 1:].T],
                np.zeros(pts.shape[1] - 1),
            )
        )
        self.assertTrue(
            np.allclose(normal, normal_test) or np.allclose(normal, -1.0 * normal_test)
        )

    def test_is_planar_2d(self):
        pts = np.array([[0.0, 2.0, -1.0], [0.0, 4.0, 2.0], [2.0, 2.0, 2.0]])
        self.assertTrue(pp.geometry_property_checks.points_are_planar(pts))

    def test_is_planar_3d(self):
        pts = np.array(
            [
                [0.0, 1.0, 0.0, 4.0 / 7.0],
                [0.0, 1.0, 1.0, 0.0],
                [5.0 / 8.0, 7.0 / 8.0, 7.0 / 4.0, 1.0 / 8.0],
            ]
        )
        self.assertTrue(pp.geometry_property_checks.points_are_planar(pts))

    def test_project_plane(self):
        pts = np.array(
            [[2.0, 0.0, 1.0, 1.0], [1.0, -2.0, -1.0, 1.0], [-1.0, 0.0, 2.0, -8.0]]
        )
        R = pp.map_geometry.project_plane_matrix(pts)
        P_pts = np.dot(R, pts)

        self.assertTrue(np.allclose(P_pts[2, :], -1.15470054 * np.ones(4)))


if __name__ == "__main__":
    unittest.main()
