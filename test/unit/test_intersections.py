import numpy as np
import unittest

from porepy.utils import comp_geom as cg


class SplitIntersectingLines2DTest(unittest.TestCase):
    """
    Various tests of remove_edge_crossings.

    Note that since this function in itself uses several subfunctions, this is
    somewhat against the spirit of unit testing. The subfunctions are also
    fairly well covered by unit tests, in the form of doctests.

    """

    def test_lines_crossing_origin(self):
        p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])
        lines = np.array([[0, 2], [1, 3], [1, 2], [3, 4]])
        box = np.array([[2], [2]])

        new_pts, new_lines = cg.remove_edge_crossings(p, lines, box=box)

        p_known = np.hstack((p, np.array([[0], [0]])))
        p_known = cg.snap_to_grid(p_known, box=box)

        lines_known = np.array([[0, 4, 2, 4], [4, 1, 4, 3], [1, 1, 2, 2], [3, 3, 4, 4]])

        assert np.allclose(new_pts, p_known)
        assert np.allclose(new_lines, lines_known)

    def test_lines_no_crossing(self):
        p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

        lines = np.array([[0, 1], [2, 3]])
        box = np.array([[2], [2]])
        new_pts, new_lines = cg.remove_edge_crossings(p, lines, box=box)
        assert np.allclose(new_pts, p)
        assert np.allclose(new_lines, lines)

    def test_three_lines_no_crossing(self):
        # This test gave an error at some point
        p = np.array(
            [[0., 0., 0.3, 1., 1., 0.5], [2 / 3, 1 / .7, 0.3, 2 / 3, 1 / .7, 0.5]]
        )
        lines = np.array([[0, 3], [1, 4], [2, 5]]).T
        box = np.array([[1], [2]])

        new_pts, new_lines = cg.remove_edge_crossings(p, lines, box=box)
        p_known = cg.snap_to_grid(p, box=box)
        assert np.allclose(new_pts, p_known)
        assert np.allclose(new_lines, lines)

    def test_three_lines_one_crossing(self):
        # This test gave an error at some point
        p = np.array([[0., 0.5, 0.3, 1., 0.3, 0.5], [2 / 3, 0.3, 0.3, 2 / 3, 0.5, 0.5]])
        lines = np.array([[0, 3], [2, 5], [1, 4]]).T
        box = np.array([[1], [2]])

        new_pts, new_lines = cg.remove_edge_crossings(p, lines, box=box)
        p_known = np.hstack((p, np.array([[0.4], [0.4]])))
        p_known = cg.snap_to_grid(p_known, box=box)
        lines_known = np.array([[0, 3], [2, 6], [6, 5], [1, 6], [6, 4]]).T
        assert np.allclose(new_pts, p_known)
        assert np.allclose(new_lines, lines_known)

    def test_split_segment_partly_overlapping(self):
        p = np.array([[0, 1, 2, 3], [0, 0, 0, 0]])
        lines = np.array([[0, 2], [1, 3]]).T
        box = np.array([[1], [1]])

        new_pts, new_lines = cg.remove_edge_crossings(p, lines, box=box)

        p_known = cg.snap_to_grid(p, box=box)
        lines_known = np.array([[0, 1], [1, 2], [2, 3]]).T

        assert np.allclose(new_pts, p_known)
        assert np.allclose(new_lines, lines_known)

    def test_split_segment_partly_overlapping_switched_order(self):
        # Same partly overlapping test, but switch order of edge-point
        # connection. Should make no difference
        p = np.array([[0, 1, 2, 3], [0, 0, 0, 0]])
        lines = np.array([[0, 2], [3, 1]]).T
        box = np.array([[1], [1]])

        new_pts, new_lines = cg.remove_edge_crossings(p, lines, box=box)

        new_lines = np.sort(new_lines, axis=0)
        p_known = cg.snap_to_grid(p, box=box)
        lines_known = np.array([[0, 1], [1, 2], [2, 3]]).T

        assert np.allclose(new_pts, p_known)
        assert np.allclose(new_lines, lines_known)

    def test_split_segment_fully_overlapping(self):
        p = np.array([[0, 1, 2, 3], [0, 0, 0, 0]])
        lines = np.array([[0, 3], [1, 2]]).T
        box = np.array([[1], [1]])

        new_pts, new_lines = cg.remove_edge_crossings(p, lines, box=box)

        new_lines = np.sort(new_lines, axis=0)
        p_known = cg.snap_to_grid(p, box=box)
        lines_known = np.array([[0, 1], [1, 2], [2, 3]]).T

        assert np.allclose(new_pts, p_known)
        assert np.allclose(new_lines, lines_known)

    def test_split_segment_fully_overlapping_switched_order(self):
        p = np.array([[0, 1, 2, 3], [0, 0, 0, 0]])
        lines = np.array([[0, 3], [2, 1]]).T
        box = np.array([[1], [1]])

        new_pts, new_lines = cg.remove_edge_crossings(p, lines, box=box)
        new_lines = np.sort(new_lines, axis=0)

        p_known = cg.snap_to_grid(p, box=box)
        lines_known = np.array([[0, 1], [1, 2], [2, 3]]).T

        assert np.allclose(new_pts, p_known)
        assert np.allclose(new_lines, lines_known)

    if __name__ == "__main__":
        unittest.main()


class SnapToGridTest(unittest.TestCase):
    def setUp(self):
        self.box = np.array([1, 1])
        self.anisobox = np.array([2, 1])
        self.p = np.array([0.6, 0.6])

    def test_snapping(self):
        p_snapped = cg.snap_to_grid(self.p, box=self.box, tol=1)
        assert np.allclose(p_snapped, np.array([1, 1]))

    def test_aniso_snapping(self):
        p_snapped = cg.snap_to_grid(self.p, box=self.anisobox, tol=1)
        assert np.allclose(p_snapped, np.array([0, 1]))

    if __name__ == "__main__":
        unittest.main()


class LinesIntersectTest(unittest.TestCase):
    def test_lines_intersect_segments_do_not(self):
        s0 = np.array([0.3, 0.3])
        e0 = np.array([0.5, 0.5])
        s1 = np.array([0, 2 / 3])
        e1 = np.array([1, 2 / 3])
        pi = cg.lines_intersect(s0, e0, s1, e1)
        assert pi is None or len(pi) == 0

    def test_parallel_not_colinear(self):
        s0 = np.array([0, 0])
        e0 = np.array([1, 0])
        s1 = np.array([0, 1])
        e1 = np.array([1, 1])

        pi = cg.lines_intersect(s0, e0, s1, e1)
        assert pi is None

    def test_colinear_not_intersecting(self):
        s0 = np.array([0, 0])
        e0 = np.array([1, 0])
        s1 = np.array([2, 0])
        e1 = np.array([3, 0])

        pi = cg.lines_intersect(s0, e0, s1, e1)
        assert pi is None

    def test_partly_overlapping_segments(self):
        s0 = np.array([0, 0])
        e0 = np.array([2, 0])
        s1 = np.array([1, 0])
        e1 = np.array([3, 0])

        pi = cg.lines_intersect(s0, e0, s1, e1)
        assert (pi[0, 0] == 1 and pi[0, 1] == 2) or (pi[0, 0] == 2 and pi[0, 1] == 1)
        assert np.allclose(pi[1], 0)

        # Then test order of arguments
        pi = cg.lines_intersect(e0, s0, s1, e1)
        assert (pi[0, 0] == 1 and pi[0, 1] == 2) or (pi[0, 0] == 2 and pi[0, 1] == 1)
        assert np.allclose(pi[1], 0)

        pi = cg.lines_intersect(s0, e0, e1, s1)
        assert (pi[0, 0] == 1 and pi[0, 1] == 2) or (pi[0, 0] == 2 and pi[0, 1] == 1)
        assert np.allclose(pi[1], 0)

        pi = cg.lines_intersect(e0, s0, e1, s1)
        assert (pi[0, 0] == 1 and pi[0, 1] == 2) or (pi[0, 0] == 2 and pi[0, 1] == 1)
        assert np.allclose(pi[1], 0)

    def test_fully_overlapping_segments(self):
        s0 = np.array([0, 0])
        e0 = np.array([3, 0])
        s1 = np.array([1, 0])
        e1 = np.array([2, 0])

        pi = cg.lines_intersect(s0, e0, s1, e1)
        assert (pi[0, 0] == 1 and pi[0, 1] == 2) or (pi[0, 0] == 2 and pi[0, 1] == 1)
        assert np.allclose(pi[1], 0)

    def test_meeting_in_point(self):
        s0 = np.array([0, 0])
        e0 = np.array([1, 0])
        s1 = np.array([1, 0])
        e1 = np.array([2, 0])

        pi = cg.lines_intersect(s0, e0, s1, e1)
        assert pi[0, 0] == 1 and pi[1, 0] == 0

    if __name__ == "__main__":
        unittest.main()
