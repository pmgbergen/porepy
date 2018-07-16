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

    def test_segments_same_plane_no_isect(self):
        # Polygons in the same plane, but no intersection
        p_1, _, _, _ = self.setup_polygons()
        p_2 = p_1 + np.array([3, 0, 0]).reshape((-1, 1))
        isect = cg.polygon_segment_intersect(p_1, p_2)
        assert isect is None
        isect = cg.polygon_segment_intersect(p_2, p_1)
        assert isect is None

    def test_segments_same_plane_isect(self):
        # Polygons in the same plane, and intersection. Should raise an
        # exception.
        p_1, _, _, _ = self.setup_polygons()
        p_2 = p_1 + np.array([1, 0, 0]).reshape((-1, 1))
        caught_exp = False
        try:
            isect = cg.polygon_segment_intersect(p_1, p_2)
        except NotImplementedError:
            caught_exp = True
        assert caught_exp

    if __name__ == "__main__":
        unittest.main()
