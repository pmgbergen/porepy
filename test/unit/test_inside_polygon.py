import numpy as np
import unittest

from porepy.utils import comp_geom as cg


class TestInsidePolygon(unittest.TestCase):
    def setup(self):
        return np.array([[0, 1, 1, 0], [0, 0, 1, 1]])

    def test_inside(self):
        poly = self.setup()

        p = np.array([0.5, 0.5])
        assert np.all(cg.is_inside_polygon(poly, p))

    def test_outside(self):
        poly = self.setup()
        p = np.array([2, 2])

        inside = cg.is_inside_polygon(poly, p)

        assert not inside[0]

    def test_on_line(self):
        # Point on the line, but not the segment, of the polygon
        poly = self.setup()
        p = np.array([2, 0])

        inside = cg.is_inside_polygon(poly, p)
        assert not inside[0]

    def test_on_boundary(self):
        poly = self.setup()
        p = np.array([0, 0.5])

        inside = cg.is_inside_polygon(poly, p)
        assert not inside[0]

    def test_just_inside(self):
        poly = self.setup()
        p = np.array([0.5, 1e-6])
        assert cg.is_inside_polygon(poly, p)

    def test_multiple_points(self):
        poly = self.setup()
        p = np.array([[0.5, 0.5], [0.5, 1.5]])

        inside = cg.is_inside_polygon(poly, p)

        assert inside[0]
        assert not inside[1]

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
        inside = cg.is_inside_polygon(a, b)
        assert not inside[0]
        assert inside[1]

    if __name__ == "__main__":
        unittest.main()
