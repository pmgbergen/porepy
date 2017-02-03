import numpy as np
import unittest

from compgeom import basics

class TestSegmentSegmentIntersection(unittest.TestCase):

    def test_intersection_origin(self):
        # 3D lines cross in the origin
        p_1 = np.array([0, -1, -1])
        p_2 = np.array([0, 1, 1])
        p_3 = np.array([-1, 0, 1])
        p_4 = np.array([1, 0, -1])

        p_i = basics.segments_intersect_3d(p_1, p_2, p_3, p_4)
        assert np.allclose(p_i, np.zeros(3))

    def test_argument_order_arbitrary(self):
        # Order of input arguments should be arbitrary
        p_1 = np.array([0, -1, -1])
        p_2 = np.array([0, 1, 1])
        p_3 = np.array([-1, 0, 1])
        p_4 = np.array([1, 0, -1])

        p_known = np.zeros(3)

        p_i_1 = basics.segments_intersect_3d(p_1, p_2, p_3, p_4)
        p_i_2 = basics.segments_intersect_3d(p_2, p_1, p_3, p_4)
        p_i_3 = basics.segments_intersect_3d(p_1, p_2, p_4, p_3)
        p_i_4 = basics.segments_intersect_3d(p_2, p_1, p_4, p_3)

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

        p_i = basics.segments_intersect_3d(p_1, p_2, p_3, p_4)
        assert p_i is None


    def test_lines_cross_segments_not(self):
        p_1 = np.array([-1, 0, -1])
        p_2 = np.array([0, 0, 0])
        p_3 = np.array([1, -1, 1])
        p_4 = np.array([1, 1, 1])

        p_i = basics.segments_intersect_3d(p_1, p_2, p_3, p_4)
        assert p_i is None

    def test_parallel_lines(self):
        p_1 = np.zeros(3)
        p_2 = np.array([1, 0, 0])
        p_3 = np.array([0, 1, 0])
        p_4 = np.array([1, 1, 0])

        p_i = basics.segments_intersect_3d(p_1, p_2, p_3, p_4)
        assert p_i is None

    def test_L_intersection(self):
        p_1 = np.zeros(3)
        p_2 = np.random.rand(3)
        p_3 = np.random.rand(3)

        p_i = basics.segments_intersect_3d(p_1, p_2, p_2, p_3)
        assert np.allclose(p_i, p_2)

    if __name__ == '__main__':
        unittest.main()
