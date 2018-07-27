import numpy as np
import unittest

from porepy.utils import comp_geom as cg


class TestSegmentSegmentIntersection(unittest.TestCase):
    def test_intersection_origin(self):
        # 3D lines cross in the origin
        p_1 = np.array([0, -1, -1])
        p_2 = np.array([0, 1, 1])
        p_3 = np.array([-1, 0, 1])
        p_4 = np.array([1, 0, -1])

        p_i = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        assert np.allclose(p_i, np.zeros(3))

    def test_argument_order_arbitrary(self):
        # Order of input arguments should be arbitrary
        p_1 = np.array([0, -1, -1])
        p_2 = np.array([0, 1, 1])
        p_3 = np.array([-1, 0, 1])
        p_4 = np.array([1, 0, -1])

        p_known = np.zeros(3)

        p_i_1 = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        p_i_2 = cg.segments_intersect_3d(p_2, p_1, p_3, p_4)
        p_i_3 = cg.segments_intersect_3d(p_1, p_2, p_4, p_3)
        p_i_4 = cg.segments_intersect_3d(p_2, p_1, p_4, p_3)

        assert np.allclose(p_i_1, p_known)
        assert np.allclose(p_i_2, p_known)
        assert np.allclose(p_i_3, p_known)
        assert np.allclose(p_i_4, p_known)

    def test_pass_in_z_coord(self):
        # The (x,y) coordinates gives intersection in origin, but z coordinates
        # do not match
        p_1 = np.array([-1, -1, -1])
        p_2 = np.array([1, 1, -1])
        p_3 = np.array([1, -1, 1])
        p_4 = np.array([-1, 1, 1])

        p_i = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        assert p_i is None

    def test_lines_cross_segments_not(self):
        p_1 = np.array([-1, 0, -1])
        p_2 = np.array([0, 0, 0])
        p_3 = np.array([1, -1, 1])
        p_4 = np.array([1, 1, 1])

        p_i = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        assert p_i is None

    def test_parallel_lines(self):
        p_1 = np.zeros(3)
        p_2 = np.array([1, 0, 0])
        p_3 = np.array([0, 1, 0])
        p_4 = np.array([1, 1, 0])

        p_i = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        assert p_i is None

    def test_L_intersection(self):
        p_1 = np.zeros(3)
        p_2 = np.random.rand(3)
        p_3 = np.random.rand(3)

        p_i = cg.segments_intersect_3d(p_1, p_2, p_2, p_3)
        assert np.allclose(p_i, p_2.reshape((-1, 1)))

    def test_equal_lines_segments_not_overlapping(self):
        p_1 = np.ones(3)
        p_2 = 0 * p_1
        p_3 = 2 * p_1
        p_4 = 3 * p_1

        p_int = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        assert p_int is None

    def test_both_aligned_with_axis(self):
        # Both lines are aligned an axis,
        p_1 = np.array([-1, -1, 0])
        p_2 = np.array([-1, 1, 0])
        p_3 = np.array([-1, 0, -1])
        p_4 = np.array([-1, 0, 1])

        p_int = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        p_known = np.array([-1, 0, 0]).reshape((-1, 1))
        assert np.allclose(p_int, p_known)

    def test_segment_fully_overlapped(self):
        # One line is fully covered by another
        p_1 = np.ones(3)
        p_2 = 2 * p_1
        p_3 = 0 * p_1
        p_4 = 3 * p_1

        p_int = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        p_known_1 = p_1.reshape((-1, 1))
        p_known_2 = p_2.reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_int - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int - p_known_2), axis=0)) < 1e-8

    def test_segments_overlap_input_order(self):
        # Test the order of inputs
        p_1 = np.ones(3)
        p_2 = 2 * p_1
        p_3 = 0 * p_1
        p_4 = 3 * p_1

        p_int_1 = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        p_int_2 = cg.segments_intersect_3d(p_2, p_1, p_3, p_4)
        p_int_3 = cg.segments_intersect_3d(p_1, p_2, p_4, p_3)
        p_int_4 = cg.segments_intersect_3d(p_2, p_1, p_4, p_3)

        p_known_1 = p_1.reshape((-1, 1))
        p_known_2 = p_2.reshape((-1, 1))

        assert np.min(np.sum(np.abs(p_int_1 - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_1 - p_known_2), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_2 - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_2 - p_known_2), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_3 - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_3 - p_known_2), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_4 - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_4 - p_known_2), axis=0)) < 1e-8

    def test_segments_partly_overlap(self):
        p_1 = np.ones(3)
        p_2 = 3 * p_1
        p_3 = 0 * p_1
        p_4 = 2 * p_1

        p_int = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        p_known_1 = p_1.reshape((-1, 1))
        p_known_2 = p_4.reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_int - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int - p_known_2), axis=0)) < 1e-8

    def test_random_incline(self):
        p_1 = np.random.rand(3)
        p_2 = 3 * p_1
        p_3 = 0 * p_1
        p_4 = 2 * p_1

        p_int = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        p_known_1 = p_1.reshape((-1, 1))
        p_known_2 = p_4.reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_int - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int - p_known_2), axis=0)) < 1e-8

    def test_segments_aligned_with_axis(self):
        p_1 = np.array([0, 1, 1])
        p_2 = 3 * p_1
        p_3 = 0 * p_1
        p_4 = 2 * p_1

        p_int = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        p_known_1 = p_1.reshape((-1, 1))
        p_known_2 = p_4.reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_int - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int - p_known_2), axis=0)) < 1e-8

    def test_constant_y_axis(self):
        p_1 = np.array([1, 0, -1])
        p_2 = np.array([1, 0, 1])
        p_3 = np.array([1.5, 0, 0])
        p_4 = np.array([0, 0, 1.5])

        p_int = cg.segments_intersect_3d(p_1, p_2, p_3, p_4)
        p_known = np.array([1, 0, 0.5]).reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_int - p_known), axis=0)) < 1e-8

    if __name__ == "__main__":
        unittest.main()
