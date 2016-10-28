import numpy as np
import unittest

from compgeom import basics


class SplitIntersectingLines2DTest(unittest.TestCase):

    def test_lines_crossing_origin(self):
        p = np.array([[-1, 1, 0, 0],
                      [0, 0, -1, 1]])
        lines = np.array([[0, 2],
                          [1, 3],
                          [1, 2],
                          [3, 4]])
        box = np.array([[2], [2]])

        new_pts, new_lines = basics.remove_edge_crossings(p, lines, box)

        p_known = np.hstack((p, np.array([[0], [0]])))
        p_known = basics.snap_to_grid(p_known, box)

        lines_known = np.array([[0, 4, 2, 4],
                                [4, 1, 4, 3],
                                [1, 1, 2, 2],
                                [3, 3, 4, 4]])

        assert np.allclose(new_pts, p_known)
        assert np.allclose(new_lines, lines_known)

    def test_lines_no_crossing(self):
        p = np.array([[-1, 1, 0, 0],
                      [0, 0, -1, 1]])

        lines = np.array([[0, 1],
                          [2, 3]])
        box = np.array([[2], [2]])
        new_pts, new_lines = basics.remove_edge_crossings(p, lines, box)
        assert np.allclose(new_pts, p)
        assert np.allclose(new_lines, lines)
        
    def test_three_lines_no_crossing(self):
        # This test gave an error at some point
        p = np.array([[ 0., 0.,   0.3, 1.,  1.,   0.5],
                      [2/3, 1/.7, 0.3, 2/3, 1/.7, 0.5]])
        lines = np.array([[0, 3], [1, 4], [2, 5]]).T
        box = np.array([[1], [2]])

        new_pts, new_lines = basics.remove_edge_crossings(p, lines, box)
        print(new_pts)
        p_known = basics.snap_to_grid(p, box)
        assert np.allclose(new_pts, p_known)
        assert np.allclose(new_lines, lines)

        
    if __name__ == '__main__':
        unittest.main()


class SnapToGridTest(unittest.TestCase):

    def setUp(self):
        self.box = np.array([1, 1])
        self.anisobox = np.array([2, 1])
        self.p = np.array([0.6, 0.6])

    def test_snapping(self):
        p_snapped = basics.snap_to_grid(self.p, self.box, precision=1)
        assert np.allclose(p_snapped, np.array([1, 1]))

    def test_aniso_snapping(self):
        p_snapped = basics.snap_to_grid(self.p, self.anisobox, precision=1)
        assert np.allclose(p_snapped, np.array([0, 1]))

    if __name__ == '__main__':
        unittest.main()