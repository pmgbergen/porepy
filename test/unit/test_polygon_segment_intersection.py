from __future__ import division
import numpy as np
import unittest

from porepy.utils import comp_geom as cg


class PolygonSegmentIntersectionTest(unittest.TestCase):
    def setup_polygons(self):
        p_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        p_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-.7, -.7, .8, .8]])
        p_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [.5, .5, 1.5, 1.5]])
        p_4 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        return p_1, p_2, p_3, p_4

    def test_one_intersection(self):
        p_1, p_2, _, _ = self.setup_polygons()

        # First intersection of 1 by edges of 2. It should be two of these
        p_1_2 = cg.polygon_segment_intersect(p_1, p_2)
        p_i_known_1 = np.array([0, 0, -0.7]).reshape((-1, 1))
        p_i_known_2 = np.array([0, 0, 0.8]).reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_1_2 - p_i_known_1), axis=0)) < 1e-5
        assert np.min(np.sum(np.abs(p_1_2 - p_i_known_2), axis=0)) < 1e-5

        # Then intersection of plane of 2 by edges of 1. This should be empty
        p_2_1 = cg.polygon_segment_intersect(p_2, p_1)
        assert p_2_1 is None

    def test_mutual_intersection(self):
        p1, _, p3, _ = self.setup_polygons()

        # First intersection of 1 by edges of 3
        p_1_3 = cg.polygon_segment_intersect(p1, p3)
        p_i_known_1 = np.array([0, 0, 0.5]).reshape((-1, 1))
        p_i_known_2 = np.array([0, 0, 1.0]).reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_1_3 - p_i_known_1), axis=0)) < 1e-5

        # Then intersection of plane of 3 by edges of 1.
        p_3_1 = cg.polygon_segment_intersect(p3, p1)
        p_i_known_2 = np.array([0, 0, 1.0]).reshape((-1, 1))

        assert np.min(np.sum(np.abs(p_3_1 - p_i_known_2), axis=0)) < 1e-5

    def test_mutual_intersection_not_at_origin(self):
        p1, _, p3, _ = self.setup_polygons()

        incr = np.array([1, 2, 3]).reshape((-1, 1))
        p1 += incr
        p3 += incr

        # First intersection of 1 by edges of 3
        p_1_3 = cg.polygon_segment_intersect(p1, p3)
        p_i_known_1 = np.array([0, 0, 0.5]).reshape((-1, 1)) + incr
        assert np.min(np.sum(np.abs(p_1_3 - p_i_known_1), axis=0)) < 1e-5

        # Then intersection of plane of 3 by edges of 1.
        p_3_1 = cg.polygon_segment_intersect(p3, p1)
        p_i_known_2 = np.array([0, 0, 1.0]).reshape((-1, 1)) + incr

        assert np.min(np.sum(np.abs(p_3_1 - p_i_known_2), axis=0)) < 1e-5

    def test_parallel_planes(self):
        p_1, _, _, _ = self.setup_polygons()
        p_2 = p_1 + np.array([0, 1, 0]).reshape((-1, 1))
        isect = cg.polygon_segment_intersect(p_1, p_2)
        assert isect is None

    def test_extension_would_intersect(self):
        # The extension of p_2 would intersect, but should detect nothing
        p_1, p_2, _, _ = self.setup_polygons()
        p_2 += np.array([2, 0, 0]).reshape((-1, 1))
        isect = cg.polygon_segment_intersect(p_1, p_2)
        assert isect is None

    def test_segments_intersect(self):
        # Test where the planes intersect in a way where vertex only touches
        # vertex. This is now updated to count as an intersection.
        p_1, _, _, p_4 = self.setup_polygons()

        isect = cg.polygon_segment_intersect(p_1, p_4)

        isect_known_1 = np.array([[0, 0, -1]]).T
        isect_known_2 = np.array([[0, 0, 1]]).T

        assert np.min(np.sum(np.abs(isect - isect_known_1), axis=0)) < 1e-5
        assert np.min(np.sum(np.abs(isect - isect_known_2), axis=0)) < 1e-5

        # Also try the other way around
        isect = cg.polygon_segment_intersect(p_4, p_1)
        assert np.min(np.sum(np.abs(isect - isect_known_1), axis=0)) < 1e-5
        assert np.min(np.sum(np.abs(isect - isect_known_2), axis=0)) < 1e-5

    def test_issue_16(self):
        # Test motivated from debuging Issue #16 (GitHub)
        # After updates of the code, we should find both intersection at vertex,
        # and internal to segments of both polygons
        frac1 = np.array([[1, 2, 4], [1, 4, 1], [2, 2, 2]])

        frac2 = np.array([[2, 2, 2], [2, 4, 1], [1, 2, 4]])

        # Segment
        isect_known_1 = np.array([[2, 5 / 3, 2]]).T
        isect_known_2 = np.array([[2, 4, 2]]).T
        isect = cg.polygon_segment_intersect(frac1, frac2)
        assert np.min(np.sum(np.abs(isect - isect_known_1), axis=0)) < 1e-5
        assert np.min(np.sum(np.abs(isect - isect_known_2), axis=0)) < 1e-5

        isect = cg.polygon_segment_intersect(frac1[:, [0, 2, 1]], frac2)
        assert np.min(np.sum(np.abs(isect - isect_known_1), axis=0)) < 1e-5
        assert np.min(np.sum(np.abs(isect - isect_known_2), axis=0)) < 1e-5

    def test_segment_in_polygon_plane(self):
        # One segment lies fully within a plane
        p1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        p2 = np.array([[0.3, 0.5, 0], [0.7, 0.5, 0], [0.7, 0.5, 1], [0.3, 0.5, 1]]).T
        isect_known = np.array([[0.3, 0.5, 0], [0.7, 0.5, 0]]).T
        isect = cg.polygon_segment_intersect(p1, p2)
        assert np.allclose(isect, isect_known)

    def test_segment_in_plane_but_not_in_polygon(self):
        p1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        p2 = np.array([[1.3, 0.5, 0], [1.7, 0.5, 0], [1.7, 0.5, 1], [1.3, 0.5, 1]]).T
        isect = cg.polygon_segment_intersect(p1, p2)
        assert isect is None

    def test_segment_partly_in_polygon(self):
        # Segment in plane, one point inside, one point outside polygon
        p1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        p2 = np.array([[0.3, 0.5, 0], [1.7, 0.5, 0], [0.7, 0.5, 1], [0.3, 0.5, 1]]).T
        isect_known = np.array([[0.3, 0.5, 0], [1, 0.5, 0]]).T
        isect = cg.polygon_segment_intersect(p1, p2)
        assert np.allclose(isect, isect_known)

    if __name__ == "__main__":
        unittest.main()
