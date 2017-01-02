import numpy as np
import unittest

from compgeom import basics


class SplitIntersectingLines2DTest(unittest.TestCase):
    """
    Various tests of remove_edge_crossings.

    Note that since this function in itself uses several subfunctions, this is
    somewhat against the spirit of unit testing. The subfunctions are also
    fairly well covered by unit tests, in the form of doctests.

    """
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
        p_known = basics.snap_to_grid(p, box)
        assert np.allclose(new_pts, p_known)
        assert np.allclose(new_lines, lines)
        
    def test_three_lines_one_crossing(self):
        # This test gave an error at some point
        p = np.array([[ 0., 0.5,   0.3, 1.,  0.3,  0.5],
                      [2/3, 0.3, 0.3, 2/3, 0.5, 0.5]])
        lines = np.array([[0, 3], [2, 5], [1, 4]]).T
        box = np.array([[1], [2]])

        new_pts, new_lines = basics.remove_edge_crossings(p, lines, box)
        p_known = np.hstack((p, np.array([[0.4], [0.4]])))
        p_known = basics.snap_to_grid(p_known, box)
        lines_known = np.array([[0, 3], [2, 6], [6, 5], [1, 6], [6, 4]]).T
        assert np.allclose(new_pts, p_known)
        assert np.allclose(new_lines, lines_known)
    

        
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
        

class LinesIntersectTest(unittest.TestCase):
    
    def test_lines_intersect_segments_do_not(self):
        s0 = np.array([0.3, 0.3])
        e0 = np.array([0.5, 0.5])
        s1 = np.array([0, 2/3])
        e1 = np.array([1, 2/3])
        pi = basics.lines_intersect(s0, e0, s1, e1)
        assert(pi is None or len(pi)==0)
        
    if __name__ == '__main__':
        unittest.main()