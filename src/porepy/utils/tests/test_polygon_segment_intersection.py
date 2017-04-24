import numpy as np
from compgeom import basics
import unittest


class PolygonSegmentIntersectionTest(unittest.TestCase):

    def setup_polygons(self):
        p_1 = np.array([[-1, 1, 1, -1 ], [0, 0, 0, 0], [-1, -1, 1, 1]])
        p_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1 ], [-.7, -.7, .8, .8]])
        p_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1 ], [.5, .5, 1.5, 1.5]])
        p_4 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1 ], [-1, -1, 1, 1]])
        return p_1, p_2, p_3, p_4


    def test_one_intersection(self):
        p_1, p_2, *rest = self.setup_polygons()

        # First intersection of 1 by edges of 2. It should be two of these
        p_1_2 = basics.polygon_segment_intersect(p_1, p_2)
        p_i_known_1 = np.array([0, 0, -0.7]).reshape((-1, 1))
        p_i_known_2 = np.array([0, 0, 0.8]).reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_1_2 - p_i_known_1), axis=0)) < 1e-5
        assert np.min(np.sum(np.abs(p_1_2 - p_i_known_2), axis=0)) < 1e-5

        # Then intersection of plane of 2 by edges of 1. This should be empty
        p_2_1 = basics.polygon_segment_intersect(p_2, p_1)
        assert p_2_1 is None

    def test_mutual_intersection(self):
        p1, _, p3, *rest = self.setup_polygons()

        # First intersection of 1 by edges of 3
        p_1_3 = basics.polygon_segment_intersect(p1, p3)
        p_i_known_1 = np.array([0, 0, 0.5]).reshape((-1, 1))
        p_i_known_2 = np.array([0, 0, 1.0]).reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_1_3 - p_i_known_1), axis=0)) < 1e-5

        # Then intersection of plane of 3 by edges of 1. 
        p_3_1 = basics.polygon_segment_intersect(p3, p1)
        p_i_known_2 = np.array([0, 0, 1.0]).reshape((-1, 1))

        assert np.min(np.sum(np.abs(p_3_1 - p_i_known_2), axis=0)) < 1e-5

    def test_mutual_intersection_not_at_origin(self):
        p1, _, p3, *rest = self.setup_polygons()

        incr = np.array([1, 2, 3]).reshape((-1, 1))
        p1 += incr
        p3 += incr

        # First intersection of 1 by edges of 3
        p_1_3 = basics.polygon_segment_intersect(p1, p3)
        p_i_known_1 = np.array([0, 0, 0.5]).reshape((-1, 1)) + incr
        assert np.min(np.sum(np.abs(p_1_3 - p_i_known_1), axis=0)) < 1e-5

        # Then intersection of plane of 3 by edges of 1. 
        p_3_1 = basics.polygon_segment_intersect(p3, p1)
        p_i_known_2 = np.array([0, 0, 1.0]).reshape((-1, 1)) + incr

        assert np.min(np.sum(np.abs(p_3_1 - p_i_known_2), axis=0)) < 1e-5

    def test_parallel_planes(self):
        p_1, *rest = self.setup_polygons()
        p_2 = p_1 + np.array([0, 1, 0]).reshape((-1, 1))
        isect = basics.polygon_segment_intersect(p_1, p_2)
        assert isect is None

    def test_extension_would_intersect(self):
        # The extension of p_2 would intersect, but should detect nothing
        p_1, p_2, *rest = self.setup_polygons()
        p_2 += np.array([2, 0, 0]).reshape((-1, 1))
        isect = basics.polygon_segment_intersect(p_1, p_2)
        assert isect is None

    def test_segments_intersect(self):
        # Test where the planes intersect in a way where segments only touches
        # segments, which does not qualify as intersection.
        p_1, _, _, p_4, *rest = self.setup_polygons()
        isect = basics.polygon_segment_intersect(p_1, p_4)
        assert isect is None
        # Also try the other way around
        isect = basics.polygon_segment_intersect(p_4, p_1)
        assert isect is None

    def test_segments_same_plane_no_isect(self):
        # Polygons in the same plane, but no intersection
        p_1, *rest = self.setup_polygons()
        p_2 = p_1 + np.array([3, 0, 0]).reshape((-1, 1))
        isect = basics.polygon_segment_intersect(p_1, p_2)
        assert isect is None
        isect = basics.polygon_segment_intersect(p_2, p_1)
        assert isect is None

    def test_segments_same_plane_isect(self):
        # Polygons in the same plane, and intersection. Should raise an
        # exception.
        p_1, *rest = self.setup_polygons()
        p_2 = p_1 + np.array([1, 0, 0]).reshape((-1, 1))
        caught_exp = False
        try:
            isect = basics.polygon_segment_intersect(p_1, p_2)
        except NotImplementedError:
            caught_exp = True
        assert caught_exp

    if __name__ == '__main__':
        unittest.main()

