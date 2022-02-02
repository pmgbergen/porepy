import unittest

import numpy as np

import porepy as pp


class BasicsTest(unittest.TestCase):
    def test_planar_square(self):
        pts = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=np.float)

        pt = np.array([0.2, 0.3, 0])
        self.assertTrue(pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([1.1, 0.5, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([-0.1, 0.5, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([0.5, 1.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([0.5, -0.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([1.1, 1.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([-0.1, 1.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([-0.1, -0.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([1.1, -0.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

    def test_planar_square_1(self):
        pts = np.array(
            [[0, 0.5, 1, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]], dtype=np.float
        )

        pt = np.array([0.2, 0.3, 0])
        self.assertTrue(pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([1.1, 0.5, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([-0.1, 0.5, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([0.5, 1.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([0.5, -0.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([1.1, 1.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([-0.1, 1.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([-0.1, -0.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([1.1, -0.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

    def test_planar_convex(self):
        pts = np.array(
            [[0, 1, 1.2, 0.5, -0.2], [0, 0, 1, 1.2, 1], [0, 0, 0, 0, 0]], dtype=np.float
        )

        pt = np.array([0.2, 0.3, 0])
        self.assertTrue(pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([0.5, 1, 0])
        self.assertTrue(pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([1.3, 0.5, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([-0.1, 0.5, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([1.1, -0.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

    def test_planar_convex_1(self):
        pts = np.array(
            [[0, 0.5, 1, 1.2, 0.5, -0.2], [0, 0, 0, 1, 1.2, 1], [0, 0, 0, 0, 0, 0]],
            dtype=np.float,
        )

        pt = np.array([0.2, 0.3, 0])
        self.assertTrue(pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([0.5, 1, 0])
        self.assertTrue(pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([1.3, 0.5, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([-0.1, 0.5, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([1.1, -0.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

    def test_planar_concave(self):
        pts = np.array(
            [[0, 0.5, 1, 0.4, 0.5, -0.2], [0, 0, 0, 1, 1.2, 1], [0, 0, 0, 0, 0, 0]],
            dtype=np.float,
        )

        pt = np.array([0.2, 0.3, 0])
        self.assertTrue(pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([0.5, 1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([1.3, 0.5, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([-0.1, 0.5, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))

        pt = np.array([1.1, -0.1, 0])
        self.assertTrue(not pp.geometry_property_checks.point_in_cell(pts, pt))


if __name__ == "__main__":
    unittest.main()
