import unittest
import numpy as np

from porepy.utils import comp_geom as cg


class TestPolygonBoundariesIntersect(unittest.TestCase):
    def setup_polygons(self):
        p_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        p_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-.7, -.7, .8, .8]])
        p_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [.5, .5, 1.5, 1.5]])
        p_4 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        return p_1, p_2, p_3, p_4

    def test_point_segment_intersection(self):
        # Polygons on top of each other, cross at a single point
        p_1, _, _, p_4 = self.setup_polygons()
        p_4 += np.array([0, 0, 2]).reshape((-1, 1))
        isect = cg.polygon_boundaries_intersect(p_1, p_4)
        p_known = np.array([0, 0, 1]).reshape((-1, 1))
        assert len(isect) == 1
        assert np.min(np.sum(np.abs(isect[0][2] - p_known), axis=0)) < 1e-5

    def test_segment_plane_intersection(self):
        # One intersection in a segment. Another in the interior, but should not be detected
        p_1, p_2, _, _ = self.setup_polygons()
        p_2 -= np.array([0, 0, 0.3]).reshape((-1, 1))
        isect = cg.polygon_boundaries_intersect(p_1, p_2)
        p_known = np.array([0, 0, -1]).reshape((-1, 1))
        assert len(isect) == 1
        assert np.min(np.sum(np.abs(isect[0][2] - p_known), axis=0)) < 1e-5

    def test_overlapping_segments(self):
        # The function should find the segment (1, 0, [-1,1])
        # In addition, each the points (1, 0, +-1) will be found twice (they are corners of both polygons)
        p_1, _, _, _ = self.setup_polygons()
        p_2 = p_1 + np.array([2, 0, 0]).reshape((-1, 1))
        isect = cg.polygon_boundaries_intersect(p_1, p_2)
        p_int = isect[0]

        p_known_1 = np.array([1, 0, -1]).reshape((-1, 1))
        p_known_2 = np.array([1, 0, 1]).reshape((-1, 1))

        found_1 = 0
        found_2 = 0
        found_1_2 = 0
        for i in isect:
            p_int = i[2]
            eq_p_1 = np.sum(np.sum(np.abs(p_int - p_known_1), axis=0) < 1e-8)
            eq_p_2 = np.sum(np.sum(np.abs(p_int - p_known_2), axis=0) < 1e-8)

            if eq_p_1 == 2:
                found_1 += 1
            if eq_p_2 == 2:
                found_2 += 1
            if eq_p_1 == 1 and eq_p_2 == 1:
                found_1_2 += 1

        assert found_1 == 1
        assert found_2 == 1
        assert found_1_2 == 1

    def test_one_segment_crosses_two(self):
        # A segment of one polygon crosses two segments of the other (and also its interior)
        p_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        p_2 = np.array([[1.5, 1.5, 0], [0, 0, 0], [0, 1.5, 1.5]])

        isect = cg.polygon_boundaries_intersect(p_1, p_2)

        p_known_1 = np.array([1, 0, 0.5]).reshape((-1, 1))
        p_known_2 = np.array([0.5, 0, 1]).reshape((-1, 1))

        found_1 = 0
        found_2 = 0
        for i in isect:
            p_int = i[2]
            eq_p_1 = np.sum(np.sum(np.abs(p_int - p_known_1), axis=0) < 1e-8)
            eq_p_2 = np.sum(np.sum(np.abs(p_int - p_known_2), axis=0) < 1e-8)

            if eq_p_1 == 1:
                found_1 += 1
            if eq_p_2 == 1:
                found_2 += 1

        assert found_1 == 1
        assert found_2 == 1

    if __name__ == "__main__":
        unittest.main()
