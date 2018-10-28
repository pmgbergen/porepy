import numpy as np
import unittest

from porepy.utils import comp_geom as cg


class BasicTest(unittest.TestCase):
    """
    Various tests of intersect_polygon_lines.

    """

    def test_0(self):
        # convex polygon
        polygon = np.array([[0., 1., 1., 0.],
                            [0., 0., 1., 1.]])
        pts = np.array([[0., 1., 0., 2., 0.5, 0., -0.5, 0.3, -1, 0.],
                        [0., 1., 0., 2., 1.,  0., -0.5, 0.6, -1, 0.]])
        lines = np.array([[0, 2, 4, 6, 8],
                          [1, 3, 5, 7, 9]])

        new_pts, new_lines = cg.intersect_polygon_lines(polygon, pts, lines)

        pts_known = np.array([[0., 1., 0., 1., 0.5, 0., 0.,   0.3],
                              [0., 1., 0., 1., 1.,  0., 3/16, 0.6]])
        lines_known = np.array([[0, 2, 4, 6],
                                [1, 3, 5, 7]])

        self.assertTrue(np.allclose(new_pts, pts_known))
        self.assertTrue(np.allclose(new_lines, lines_known))

    def test_1(self):
        # non-convex polygon
        polygon = np.array([[0., 0.5, 0.75, 1., 1.5, 1.5, 0],
                            [0., 0.,  0.25, 0., 0,   1,   1]])
        pts = np.array([[0., 1., 0., 2., 0.5, 0., -0.5, 0.3, -1, 0., 0.,  2],
                        [0., 1., 0., 2., 1.,  0., -0.5, 0.6, -1, 0., 0.2, 0.2]])
        lines = np.array([[0, 2, 4, 6, 8, 10],
                          [1, 3, 5, 7, 9, 11]])

        new_pts, new_lines = cg.intersect_polygon_lines(polygon, pts, lines)
        pts_known = np.array([[0., 1., 0., 1., 0.5, 0., 0.,   0.3, 0.,  0.7, 0.8, 1.5],
                              [0., 1., 0., 1., 1.,  0., 3/16, 0.6, 0.2, 0.2, 0.2, 0.2]])
        lines_known = np.array([[0, 2, 4, 6, 8, 10],
                                [1, 3, 5, 7, 9, 11]])

        self.assertTrue(np.allclose(new_pts, pts_known))
        self.assertTrue(np.allclose(new_lines, lines_known))


if __name__ == "__main__":
    unittest.main()
