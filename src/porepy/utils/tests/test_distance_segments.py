import unittest
import numpy as np

from compgeom import basics

class TestSegmentDistance(unittest.TestCase):
    def setup_2d_unit_square(self):
        p00 = np.array([0, 0])
        p10 = np.array([1, 0])
        p01 = np.array([0, 1])
        p11 = np.array([1, 1])
        return p00, p10, p11, p01

    def test_segment_no_intersect_2d(self):
        p00, p10, p11, p01 = self.setup_2d_unit_square()
        d = basics.distance_segment_segment(p00, p01, p11, p10)
        assert d == 1

    def test_segment_intersect_2d(self):
        p00, p10, p11, p01 = self.setup_2d_unit_square()
        d = basics.distance_segment_segment(p00, p11, p10, p01)
        assert d == 0

    def test_line_passing(self):
        # Lines not crossing
        p1 = np.array([0, 0])
        p2 = np.array([1, 0])
        p3 = np.array([2, -1])
        p4 = np.array([2, 1])
        d = basics.distance_segment_segment(p1, p2, p3, p4)
        assert d == 1

    def test_share_point(self):
        # Two lines share a point
        p1 = np.array([0, 0])
        p2 = np.array([0, 1])
        p3 = np.array([1, 1])
        d = basics.distance_segment_segment(p1, p2, p2, p3)
        assert d == 0

    def test_intersection_3d(self):
        p000 = np.array([0, 0, 0])
        p111 = np.array([1, 1, 1])
        p100 = np.array([1, 0, 0])
        p011 = np.array([0, 1, 1])
        d = basics.distance_segment_segment(p000, p111, p100, p011)
        assert d == 0

    def test_changed_order_3d(self):
        # The order of the start and endpoints of the segments should not matter
        dim = 3
        p1 = np.random.rand(1, 3)[0]
        p2 = np.random.rand(1, 3)[0]
        p3 = np.random.rand(1, 3)[0]
        p4 = np.random.rand(1, 3)[0]
        d1 = basics.distance_segment_segment(p1, p2, p3, p4)
        d2 = basics.distance_segment_segment(p2, p1, p3, p4)
        d3 = basics.distance_segment_segment(p1, p2, p4, p3)
        d4 = basics.distance_segment_segment(p2, p1, p4, p3)
        d5 = basics.distance_segment_segment(p4, p3, p2, p1)
        assert np.allclose(d1, d2)
        assert np.allclose(d1, d3)
        assert np.allclose(d1, d4)
        assert np.allclose(d1, d5)

    if __name__ == '__main__':
        unittest.main()

