import numpy as np
import unittest

from porepy_new.src.porepy.utils import comp_geom as cg


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

    if __name__ == '__main__':
        unittest.main()

