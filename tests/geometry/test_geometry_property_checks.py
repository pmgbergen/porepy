import unittest

import numpy as np

import porepy as pp


class TestInsidePolygon(unittest.TestCase):
    def setup(self):
        return np.array([[0, 1, 1, 0], [0, 0, 1, 1]])

    def test_inside(self):
        poly = self.setup()

        p = np.array([0.5, 0.5])
        self.assertTrue(np.all(pp.geometry_property_checks.point_in_polygon(poly, p)))

    def test_outside(self):
        poly = self.setup()
        p = np.array([2, 2])

        inside = pp.geometry_property_checks.point_in_polygon(poly, p)

        self.assertTrue(not inside[0])

    def test_on_line(self):
        # Point on the line, but not the segment, of the polygon
        poly = self.setup()
        p = np.array([2, 0])

        inside = pp.geometry_property_checks.point_in_polygon(poly, p)
        self.assertTrue(not inside[0])

    def test_on_boundary(self):
        poly = self.setup()
        p = np.array([0, 0.5])

        inside = pp.geometry_property_checks.point_in_polygon(poly, p)
        self.assertTrue(not inside[0])

    def test_just_inside(self):
        poly = self.setup()
        p = np.array([0.5, 1e-6])
        self.assertTrue(pp.geometry_property_checks.point_in_polygon(poly, p))

    def test_multiple_points(self):
        poly = self.setup()
        p = np.array([[0.5, 0.5], [0.5, 1.5]])

        inside = pp.geometry_property_checks.point_in_polygon(poly, p)

        self.assertTrue(inside[0])
        self.assertTrue(not inside[1])

    def test_large_polygon(self):
        a = np.array(
            [
                [
                    -2.04462568e-01,
                    -1.88898782e-01,
                    -1.65916617e-01,
                    -1.44576869e-01,
                    -7.82444375e-02,
                    -1.44018389e-16,
                    7.82444375e-02,
                    1.03961083e-01,
                    1.44576869e-01,
                    1.88898782e-01,
                    2.04462568e-01,
                    1.88898782e-01,
                    1.44576869e-01,
                    7.82444375e-02,
                    1.44018389e-16,
                    -7.82444375e-02,
                    -1.44576869e-01,
                    -1.88898782e-01,
                ],
                [
                    -1.10953147e-16,
                    7.82444375e-02,
                    1.19484803e-01,
                    1.44576869e-01,
                    1.88898782e-01,
                    2.04462568e-01,
                    1.88898782e-01,
                    1.76059749e-01,
                    1.44576869e-01,
                    7.82444375e-02,
                    1.31179355e-16,
                    -7.82444375e-02,
                    -1.44576869e-01,
                    -1.88898782e-01,
                    -2.04462568e-01,
                    -1.88898782e-01,
                    -1.44576869e-01,
                    -7.82444375e-02,
                ],
            ]
        )
        b = np.array([[0.1281648, 0.04746067], [-0.22076491, 0.16421546]])
        inside = pp.geometry_property_checks.point_in_polygon(a, b)
        self.assertTrue(not inside[0])
        self.assertTrue(inside[1])


class TestPointInPolyhedron(unittest.TestCase):
    def setUp(self):
        west = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        east = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
        south = np.array([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1]])
        north = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
        bottom = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        top = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        self.cart_polyhedron = [west, east, south, north, bottom, top]

        south_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 0, 0], [0, 0.5, 1, 1]])
        south_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 0, 0], [0.5, 0, 1, 1]])
        north_w = np.array([[0, 0.5, 0.5, 0], [1, 1, 1, 1], [0, 0.5, 1, 1]])
        north_e = np.array([[0.5, 1, 1, 0.5], [1, 1, 1, 1], [0.5, 0, 1, 1]])
        bottom_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 1], [0, 0.5, 0.5, 0]])
        bottom_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 1, 1], [0.5, 0.0, 0, 0.5]])
        top_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        top_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 1, 1], [1, 1, 1, 1]])
        self.non_convex_polyhedron = [
            west,
            east,
            south_w,
            south_e,
            north_w,
            north_e,
            bottom_w,
            bottom_e,
            top_w,
            top_e,
        ]

    def test_point_inside_box(self):
        p = np.array([0.3, 0.5, 0.5])
        is_inside = pp.geometry_property_checks.point_in_polyhedron(
            self.cart_polyhedron, p
        )
        self.assertTrue(is_inside.size == 1)
        self.assertTrue(is_inside[0] == 1)

    def test_two_points_inside_box(self):
        p = np.array([[0.3, 0.5, 0.5], [0.5, 0.5, 0.5]]).T
        is_inside = pp.geometry_property_checks.point_in_polyhedron(
            self.cart_polyhedron, p
        )
        self.assertTrue(is_inside.size == 2)
        self.assertTrue(np.all(is_inside[0] == 1))

    def test_point_outside_box(self):
        p = np.array([1.5, 0.5, 0.5])
        is_inside = pp.geometry_property_checks.point_in_polyhedron(
            self.cart_polyhedron, p
        )
        self.assertTrue(is_inside.size == 1)
        self.assertTrue(is_inside[0] == 0)

    def test_point_inside_non_convex(self):
        p = np.array([0.5, 0.5, 0.7])
        is_inside = pp.geometry_property_checks.point_in_polyhedron(
            self.non_convex_polyhedron, p
        )
        self.assertTrue(is_inside.size == 1)
        self.assertTrue(is_inside[0] == 1)

    def test_point_outside_non_convex_inside_box(self):
        p = np.array([0.5, 0.5, 0.3])
        is_inside = pp.geometry_property_checks.point_in_polyhedron(
            self.non_convex_polyhedron, p
        )
        self.assertTrue(is_inside.size == 1)
        self.assertTrue(is_inside[0] == 0)


if __name__ == "__main__":
    unittest.main()
