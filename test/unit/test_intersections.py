import numpy as np
import unittest

from test import test_utils

import porepy as pp


class SplitIntersectingLines2DTest(unittest.TestCase):
    """
    Various tests of remove_edge_crossings.

    Note that since this function in itself uses several subfunctions, this is
    somewhat against the spirit of unit testing. The subfunctions are also
    fairly well covered by unit tests, in the form of doctests.

    """

    def test_lines_no_crossing(self):
        p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

        lines = np.array([[0, 1], [2, 3]])
        new_pts, new_lines = pp.intersections.split_intersecting_segments_2d(p, lines)
        self.assertTrue(np.allclose(new_pts, p))
        self.assertTrue(np.allclose(new_lines, lines))

    def test_three_lines_no_crossing(self):
        # This test gave an error at some point
        p = np.array(
            [[0.0, 0.0, 0.3, 1.0, 1.0, 0.5], [2 / 3, 1 / 0.7, 0.3, 2 / 3, 1 / 0.7, 0.5]]
        )
        lines = np.array([[0, 3], [1, 4], [2, 5]]).T

        new_pts, new_lines = pp.intersections.split_intersecting_segments_2d(p, lines)

        self.assertTrue(np.allclose(new_pts, p))
        self.assertTrue(test_utils.compare_arrays(new_lines, lines))

    def test_three_lines_one_crossing(self):
        # This test gave an error at some point
        p = np.array(
            [[0.0, 0.5, 0.3, 1.0, 0.3, 0.5], [2 / 3, 0.3, 0.3, 2 / 3, 0.5, 0.5]]
        )
        lines = np.array([[0, 3], [2, 5], [1, 4]]).T

        new_pts, new_lines = pp.intersections.split_intersecting_segments_2d(p, lines)
        p_known = np.hstack((p, np.array([[0.4], [0.4]])))
        lines_known = np.array([[0, 3], [2, 6], [6, 5], [1, 6], [6, 4]]).T
        self.assertTrue(np.allclose(new_pts, p_known))
        self.assertTrue(test_utils.compare_arrays(new_lines, lines_known))

    def test_split_segment_partly_overlapping(self):
        p = np.array([[0, 1, 2, 3], [0, 0, 0, 0]])
        lines = np.array([[0, 2], [1, 3]]).T

        new_pts, new_lines = pp.intersections.split_intersecting_segments_2d(p, lines)

        lines_known = np.array([[0, 1], [1, 2], [2, 3]]).T

        self.assertTrue(np.allclose(new_pts, p))
        self.assertTrue(np.allclose(new_lines, lines_known))

    def test_split_segment_partly_overlapping_switched_order(self):
        # Same partly overlapping test, but switch order of edge-point
        # connection. Should make no difference
        p = np.array([[0, 1, 2, 3], [0, 0, 0, 0]])
        lines = np.array([[0, 2], [3, 1]]).T

        new_pts, new_lines = pp.intersections.split_intersecting_segments_2d(p, lines)

        new_lines = np.sort(new_lines, axis=0)
        lines_known = np.array([[0, 1], [1, 2], [2, 3]]).T

        self.assertTrue(np.allclose(new_pts, p))
        self.assertTrue(np.allclose(new_lines, lines_known))

    def test_split_segment_fully_overlapping(self):
        p = np.array([[0, 1, 2, 3], [0, 0, 0, 0]])
        lines = np.array([[0, 3], [1, 2]]).T

        new_pts, new_lines = pp.intersections.split_intersecting_segments_2d(p, lines)

        new_lines = np.sort(new_lines, axis=0)
        lines_known = np.array([[0, 1], [1, 2], [2, 3]]).T

        self.assertTrue(np.allclose(new_pts, p))
        self.assertTrue(test_utils.compare_arrays(new_lines, lines_known))

    def test_split_segment_fully_overlapping_common_start(self):
        p = np.array([[0, 1, 2], [0, 0, 0]])
        lines = np.array([[0, 2], [0, 1]]).T

        new_pts, new_lines = pp.intersections.split_intersecting_segments_2d(p, lines)

        new_lines = np.sort(new_lines, axis=0)
        lines_known = np.array([[0, 1], [1, 2]]).T

        self.assertTrue(np.allclose(new_pts, p))
        self.assertTrue(test_utils.compare_arrays(new_lines, lines_known))

    def test_split_segment_fully_overlapping_common_endt(self):
        p = np.array([[0, 1, 2], [0, 0, 0]])
        lines = np.array([[0, 2], [1, 2]]).T

        new_pts, new_lines = pp.intersections.split_intersecting_segments_2d(p, lines)

        new_lines = np.sort(new_lines, axis=0)
        lines_known = np.array([[0, 1], [1, 2]]).T

        self.assertTrue(np.allclose(new_pts, p))
        self.assertTrue(test_utils.compare_arrays(new_lines, lines_known))

    def test_split_segment_fully_overlapping_switched_order(self):
        p = np.array([[0, 1, 2, 3], [0, 0, 0, 0]])
        lines = np.array([[0, 3], [2, 1]]).T

        new_pts, new_lines = pp.intersections.split_intersecting_segments_2d(p, lines)
        new_lines = np.sort(new_lines, axis=0)

        lines_known = np.array([[0, 1], [1, 2], [2, 3]]).T

        self.assertTrue(np.allclose(new_pts, p))
        self.assertTrue(test_utils.compare_arrays(new_lines, lines_known))


class LinesIntersectTest(unittest.TestCase):
    def test_lines_intersect_segments_do_not(self):
        s0 = np.array([0.3, 0.3])
        e0 = np.array([0.5, 0.5])
        s1 = np.array([0, 2 / 3])
        e1 = np.array([1, 2 / 3])
        pi = pp.intersections.segments_2d(s0, e0, s1, e1)
        self.assertTrue(pi is None or len(pi) == 0)

    def test_parallel_not_colinear(self):
        s0 = np.array([0, 0])
        e0 = np.array([1, 0])
        s1 = np.array([0, 1])
        e1 = np.array([1, 1])

        pi = pp.intersections.segments_2d(s0, e0, s1, e1)
        self.assertTrue(pi is None)

    def test_colinear_not_intersecting(self):
        s0 = np.array([0, 0])
        e0 = np.array([1, 0])
        s1 = np.array([2, 0])
        e1 = np.array([3, 0])

        pi = pp.intersections.segments_2d(s0, e0, s1, e1)
        self.assertTrue(pi is None)

    def test_partly_overlapping_segments(self):
        s0 = np.array([0, 0])
        e0 = np.array([2, 0])
        s1 = np.array([1, 0])
        e1 = np.array([3, 0])

        pi = pp.intersections.segments_2d(s0, e0, s1, e1)
        self.assertTrue(
            (pi[0, 0] == 1 and pi[0, 1] == 2) or (pi[0, 0] == 2 and pi[0, 1] == 1)
        )
        self.assertTrue(np.allclose(pi[1], 0))

        # Then test order of arguments
        pi = pp.intersections.segments_2d(e0, s0, s1, e1)
        self.assertTrue(
            (pi[0, 0] == 1 and pi[0, 1] == 2) or (pi[0, 0] == 2 and pi[0, 1] == 1)
        )
        self.assertTrue(np.allclose(pi[1], 0))

        pi = pp.intersections.segments_2d(s0, e0, e1, s1)
        self.assertTrue(
            (pi[0, 0] == 1 and pi[0, 1] == 2) or (pi[0, 0] == 2 and pi[0, 1] == 1)
        )
        self.assertTrue(np.allclose(pi[1], 0))

        pi = pp.intersections.segments_2d(e0, s0, e1, s1)
        self.assertTrue(
            (pi[0, 0] == 1 and pi[0, 1] == 2) or (pi[0, 0] == 2 and pi[0, 1] == 1)
        )
        self.assertTrue(np.allclose(pi[1], 0))

    def test_fully_overlapping_segments(self):
        s0 = np.array([0, 0])
        e0 = np.array([3, 0])
        s1 = np.array([1, 0])
        e1 = np.array([2, 0])

        pi = pp.intersections.segments_2d(s0, e0, s1, e1)
        self.assertTrue(
            (pi[0, 0] == 1 and pi[0, 1] == 2) or (pi[0, 0] == 2 and pi[0, 1] == 1)
        )
        self.assertTrue(np.allclose(pi[1], 0))

    def test_meeting_in_point(self):
        s0 = np.array([0, 0])
        e0 = np.array([1, 0])
        s1 = np.array([1, 0])
        e1 = np.array([2, 0])

        pi = pp.intersections.segments_2d(s0, e0, s1, e1)
        self.assertTrue(pi[0, 0] == 1 and pi[1, 0] == 0)


class LineTesselation(unittest.TestCase):
    def test_tesselation_do_not(self):
        p1 = np.array([[0.3, 0.3, 0], [0.5, 0.5, 0], [0.9, 0.9, 0]]).T
        p2 = np.array([[0.4, 0.4, 0.1], [1.0, 1.0, 0.1]]).T
        l1 = np.array([[0, 1], [1, 2]]).T
        l2 = np.array([[0, 1]]).T
        intersect = pp.intersections.line_tesselation(p1, p2, l1, l2)
        self.assertTrue(len(intersect) == 0)

    def test_tesselation_do(self):
        p1 = np.array([[0.0, 0.0, 0], [0.5, 0.5, 0], [1.0, 1.0, 0]]).T
        p2 = np.array([[0.25, 0.25, 0], [1.0, 1.0, 0]]).T
        l1 = np.array([[0, 1], [1, 2]]).T
        l2 = np.array([[0, 1]]).T
        intersections = pp.intersections.line_tesselation(p1, p2, l1, l2)
        for inter in intersections:
            if inter[0] == 0:
                if inter[1] == 0:
                    self.assertTrue(inter[2] == np.sqrt(0.25 ** 2 + 0.25 ** 2))
                    continue
            elif inter[0] == 1:
                if inter[1] == 1:
                    self.assertTrue(inter[2] == np.sqrt(0.5 ** 2 + 0.5 ** 2))
            else:
                self.assertTrue(False)


class SurfaceTessalation(unittest.TestCase):
    # The naming is a bit confusing here, the sets of polygons do not cover the same
    # areas, thus they are not tessalations, but the tests serve their purpose.

    def test_two_tessalations_one_cell_each(self):
        # Two triangles, partly overlapping.
        p1 = [np.array([[0, 1, 0], [0, 0, 1]])]
        p2 = [np.array([[0, 1, 0], [0, 1, 1]])]

        isect, mappings = pp.intersections.surface_tessalations([p1, p2])

        known_isect = np.array([[0, 0.5, 0], [0, 0.5, 1]])

        # Mappings are both identity mappings

        self.assertTrue(test_utils.compare_arrays(isect[0], known_isect))
        for i in range(2):
            self.assertTrue(mappings[i].shape == (1, 1))
            self.assertTrue(mappings[i].toarray() == np.array([[1]]))

    def test_two_tessalations_no_overlap(self):
        # Two triangles, partly overlapping.
        p1 = [np.array([[0, 1, 0], [0, 0, 1]])]
        p2 = [np.array([[0, 1, 0], [1, 1, 2]])]

        isect, mappings = pp.intersections.surface_tessalations([p1, p2])

        # Mappings are both identity mappings

        self.assertTrue(len(isect) == 0)
        for i in range(2):
            self.assertTrue(mappings[i].shape == (0, 1))

    def test_two_tessalations_one_quad(self):
        # Quad and triangle. Partly overlapping
        p1 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]])]
        p2 = [np.array([[0, 1, 0], [0, 1, 2]])]

        isect, mappings = pp.intersections.surface_tessalations([p1, p2])

        known_isect = np.array([[0, 1, 0], [0, 1, 1]])

        # Mappings are both identity mappings

        self.assertTrue(test_utils.compare_arrays(isect[0], known_isect))
        for i in range(2):
            self.assertTrue(mappings[i].shape == (1, 1))
            self.assertTrue(mappings[i].toarray() == np.array([[1]]))

    def test_two_tessalations_non_convex_intersection(self):
        # Quad and triangle. Partly overlapping
        p1 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]])]
        p2 = [np.array([[0, 1, 1, 0.5, 0], [0, 0, 1, 0.5, 1]])]

        isect, mappings = pp.intersections.surface_tessalations([p1, p2])

        known_isect = p2[0]

        # Mappings are both identity mappings

        self.assertTrue(test_utils.compare_arrays(isect[0], known_isect))
        for i in range(2):
            self.assertTrue(mappings[i].shape == (1, 1))
            self.assertTrue(mappings[i].toarray() == np.array([[1]]))

    def test_two_tessalations_one_with_two_cells(self):
        # First consists of quad + triangle
        p1 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]]), np.array([[0, 1, 0], [1, 1, 2]])]
        # second is a single triangle
        p2 = [np.array([[0, 1, 0], [0, 1, 2]])]

        isect, mappings = pp.intersections.surface_tessalations([p1, p2])

        # intersection is split in two
        known_isect = [
            np.array([[0, 1, 0], [0, 1, 1]]),
            np.array([[0, 1, 0], [1, 1, 2]]),
        ]

        self.assertTrue(mappings[0].shape == (2, 2))
        # To find the order of triangles in isect relative to the polygons in p1,
        # we consider the mapping for p1
        if mappings[0][0, 0] == 1:
            self.assertTrue(test_utils.compare_arrays(isect[0], known_isect[0]))
            self.assertTrue(test_utils.compare_arrays(isect[1], known_isect[1]))
            self.assertTrue(
                np.allclose(mappings[0].toarray(), np.array([[1, 0], [0, 1]]))
            )
        else:
            self.assertTrue(test_utils.compare_arrays(isect[0], known_isect[1]))
            self.assertTrue(test_utils.compare_arrays(isect[1], known_isect[0]))
            self.assertTrue(
                np.allclose(mappings[0].toarray(), np.array([[0, 1], [1, 0]]))
            )

        # p2 is much simpler
        self.assertTrue(mappings[1].shape == (2, 1))
        self.assertTrue(np.allclose(mappings[1].toarray(), np.array([[1], [1]])))

    def test_two_tessalations_two_cells_each_one_no_overlap(self):
        # First consists of quad + triangle
        p1 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]]), np.array([[0, 1, 0], [1, 1, 2]])]
        # second has two triangles, one of which has no overlap with the first tessalation
        p2 = [
            np.array([[0, 1, 0], [0, 1, 2]]),
            np.array([[0, -1, 0], [0, 1, 2]]),
        ]

        isect, mappings = pp.intersections.surface_tessalations([p1, p2])
        # No need to test intersection points, they are identical with
        # self.test_two_tessalations_one_with_two_cells()

        self.assertTrue(mappings[1].shape == (2, 2))
        self.assertTrue(np.allclose(mappings[1].toarray(), np.array([[1, 0], [1, 0]])))

    def test_three_tessalations(self):
        # First consists of quad + triangle
        p1 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]]), np.array([[0, 1, 0], [1, 1, 2]])]
        # second is a single triangle
        p2 = [np.array([[0, 1, 0], [0, 1, 2]])]
        # A third, with a single triangle
        p3 = [np.array([[0, 1, 0], [0, 0, 1]])]

        isect, mappings = pp.intersections.surface_tessalations([p1, p2, p3])

        self.assertTrue(len(isect) == 1)
        self.assertTrue(
            test_utils.compare_arrays(isect[0], np.array([[0, 0.5, 0], [0, 0.5, 1]]))
        )

        self.assertTrue(len(mappings) == 3)
        self.assertTrue(mappings[0].shape == (1, 2))
        self.assertTrue(np.allclose(mappings[0].toarray(), np.array([[1, 0]])))

        self.assertTrue(mappings[1].shape == (1, 1))
        self.assertTrue(np.allclose(mappings[1].toarray(), np.array([[1]])))
        self.assertTrue(mappings[2].shape == (1, 1))
        self.assertTrue(np.allclose(mappings[2].toarray(), np.array([[1]])))


if __name__ == "__main__":
    unittest.main()
