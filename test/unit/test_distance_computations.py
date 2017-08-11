import unittest
import numpy as np

from porepy.utils import comp_geom as cg

class TestSegmentDistance(unittest.TestCase):
    def setup_2d_unit_square(self):
        p00 = np.array([0, 0])
        p10 = np.array([1, 0])
        p01 = np.array([0, 1])
        p11 = np.array([1, 1])
        return p00, p10, p11, p01

    def test_segment_no_intersect_2d(self):
        p00, p10, p11, p01 = self.setup_2d_unit_square()
        d = cg.dist_segment_segment(p00, p01, p11, p10)
        assert d == 1

    def test_segment_intersect_2d(self):
        p00, p10, p11, p01 = self.setup_2d_unit_square()
        d = cg.dist_segment_segment(p00, p11, p10, p01)
        assert d == 0

    def test_line_passing(self):
        # Lines not crossing
        p1 = np.array([0, 0])
        p2 = np.array([1, 0])
        p3 = np.array([2, -1])
        p4 = np.array([2, 1])
        d = cg.dist_segment_segment(p1, p2, p3, p4)
        assert d == 1

    def test_share_point(self):
        # Two lines share a point
        p1 = np.array([0, 0])
        p2 = np.array([0, 1])
        p3 = np.array([1, 1])
        d = cg.dist_segment_segment(p1, p2, p2, p3)
        assert d == 0

    def test_intersection_3d(self):
        p000 = np.array([0, 0, 0])
        p111 = np.array([1, 1, 1])
        p100 = np.array([1, 0, 0])
        p011 = np.array([0, 1, 1])
        d = cg.dist_segment_segment(p000, p111, p100, p011)
        assert d == 0

    def test_changed_order_3d(self):
        # The order of the start and endpoints of the segments should not matter
        dim = 3
        p1 = np.random.rand(1, 3)[0]
        p2 = np.random.rand(1, 3)[0]
        p3 = np.random.rand(1, 3)[0]
        p4 = np.random.rand(1, 3)[0]
        d1 = cg.dist_segment_segment(p1, p2, p3, p4)
        d2 = cg.dist_segment_segment(p2, p1, p3, p4)
        d3 = cg.dist_segment_segment(p1, p2, p4, p3)
        d4 = cg.dist_segment_segment(p2, p1, p4, p3)
        d5 = cg.dist_segment_segment(p4, p3, p2, p1)
        assert np.allclose(d1, d2)
        assert np.allclose(d1, d3)
        assert np.allclose(d1, d4)
        assert np.allclose(d1, d5)

    if __name__ == '__main__':
        unittest.main()


class TestDistancePointSet(unittest.TestCase):

    def test_unit_square(self):
        p = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
        d = cg.dist_pointset(p)
        s2 = np.sqrt(2)
        known = np.array([[0, 1, s2, 1],
                          [1, 0, 1, s2],
                          [s2, 1, 0, 1],
                          [1, s2, 1, 0]])
        assert np.allclose(d, known)

    def test_3d(self):
        p = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]).T
        d = cg.dist_pointset(p)
        known = np.array([[0, 1, 1],
                          [1, 0, np.sqrt(2)],
                          [1, np.sqrt(2), 0]])
        assert np.allclose(d, known)

    def test_zero_diagonal(self):
        sz = 5
        p = np.random.rand(3, sz)
        d = cg.dist_pointset(p)
        assert np.allclose(np.diagonal(d), np.zeros(sz))

    def test_symmetry(self):
        p = np.random.rand(3, 7)
        d = cg.dist_pointset(p)
        assert np.allclose(d, d.T)

    def test_single_point(self):
        p = np.random.rand(2)
        d = cg.dist_pointset(p)
        assert d.shape == (1, 1)
        assert d[0, 0] == 0

    if __name__ == '__main__':
        unittest.main()


class TestDistancePointSegments(unittest.TestCase):

    def test_single_point_and_segment(self):
        p = np.array([0, 0])
        start = np.array([1, 0])
        end = np.array([1, 1])

        d = cg.dist_points_segments(p, start, end)

        assert d[0, 0] == 1

    def test_many_points_single_segment(self):
        p = np.array([[0, 1], [1, 1], [1.5, 1], [2, 1], [3, 1]]).T
        start = np.array([1, 0])
        end = np.array([2, 0])

        d = cg.dist_points_segments(p, start, end)

        assert d.shape[0] == 5
        assert d.shape[1] == 1

        known = np.array([np.sqrt(2), 1, 1, 1, np.sqrt(2)]).reshape((-1, 1))
        assert np.allclose(d, known)

    def test_single_point_many_segments(self):
        p = np.array([1, 1])
        start = np.array([[0, 0], [0, 1], [0, 2]]).T
        end = np.array([[2, 0], [2, 1], [2, 2]]).T

        d = cg.dist_points_segments(p, start, end)

        assert d.shape[0] == 1
        assert d.shape[1] == 3

        known = np.array([1, 0, 1])
        assert np.allclose(d[0], known)

    def test_many_points_and_segments(self):
        p = np.array([[0, 0, 0], [1, 0, 0]]).T

        start = np.array([[0, 0, -1], [0, 1, 1]]).T
        end = np.array([[0, 0, 1], [1, 1, 1]]).T

        d = cg.dist_points_segments(p, start, end)

        assert d.shape[0] == 2
        assert d.shape[1] == 2

        known = np.array([[0, np.sqrt(2)], [1, np.sqrt(2)]])
        assert np.allclose(d, known)

    def test_point_closest_segment_end(self):
        p = np.array([0, 0])
        start = np.array([[1, 0], [1, 1]]).T
        end = np.array([[2, 0], [2, 1]]).T

        d = cg.dist_points_segments(p, start, end)

        known = np.array([1, np.sqrt(2)])
        assert np.allclose(d[0], known)

    def test_flipped_lines(self):
        p = np.array([0, 0])
        start = np.array([[1, 0], [1, 1]]).T
        end = np.array([[1, 1], [1, 0]])

        d = cg.dist_points_segments(p, start, end)
        known = np.array([1, 1])
        assert np.allclose(d[0], known)

    if __name__ == '__main__':
        unittest.main()
