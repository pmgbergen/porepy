import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.grids.standard_grids.utils import unit_domain


class TestSplitIntersectingLines2D:
    """
    Various tests of remove_edge_crossings.

    Note that since this function in itself uses several subfunctions, this is somewhat
    against the spirit of unit testing. The subfunctions are also fairly well covered by
    unit tests, in the form of doctests.

    """

    def test_lines_no_crossing(self):
        p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

        lines = np.array([[0, 1], [2, 3]])
        new_pts, new_lines, _ = pp.intersections.split_intersecting_segments_2d(
            p, lines
        )
        assert compare_arrays(new_pts, p)
        assert compare_arrays(new_lines, lines)

    def test_three_lines_no_crossing(self):
        # This test gave an error at some point
        p = np.array(
            [[0.0, 0.0, 0.3, 1.0, 1.0, 0.5], [2 / 3, 1 / 0.7, 0.3, 2 / 3, 1 / 0.7, 0.5]]
        )
        lines = np.array([[0, 3], [1, 4], [2, 5]]).T

        new_pts, new_lines, _ = pp.intersections.split_intersecting_segments_2d(
            p, lines
        )

        assert compare_arrays(new_pts, p)
        assert compare_arrays(new_lines, lines)

    def test_three_lines_one_crossing(self):
        # This test gave an error at some point
        p = np.array(
            [[0.0, 0.5, 0.3, 1.0, 0.3, 0.5], [2 / 3, 0.3, 0.3, 2 / 3, 0.5, 0.5]]
        )
        lines = np.array([[0, 3], [2, 5], [1, 4]]).T

        new_pts, new_lines, _ = pp.intersections.split_intersecting_segments_2d(
            p, lines
        )
        p_known = np.hstack((p, np.array([[0.4], [0.4]])))
        lines_known = np.array([[0, 3], [2, 6], [6, 5], [1, 6], [6, 4]]).T
        assert compare_arrays(new_pts, p_known)
        assert compare_arrays(new_lines, lines_known)

    def test_split_segment_partly_overlapping(self):
        p = np.array([[0, 1, 2, 3], [0, 0, 0, 0]])
        lines = np.array([[0, 2], [1, 3]]).T

        new_pts, new_lines, _ = pp.intersections.split_intersecting_segments_2d(
            p, lines
        )

        lines_known = np.array([[0, 1], [1, 2], [2, 3]]).T

        assert compare_arrays(new_pts, p)
        assert compare_arrays(new_lines, lines_known)

    # The rest of this class contains various versions of the same test. First we define
    # tuples with points, connections between them, and the known new lines after
    # splitting.

    # Data for the same partly overlapping test as above
    # (test_split_segment_partly_overlapping), but switch order of edge-point
    # connection. Should make no difference
    split_segment_partly_overlapping_switched_order = (
        np.array([[0, 1, 2, 3], [0, 0, 0, 0]]),
        np.array([[0, 2], [3, 1]]).T,
        np.array([[0, 1], [1, 2], [2, 3]]).T,
    )

    split_segment_fully_overlapping = (
        np.array([[0, 1, 2, 3], [0, 0, 0, 0]]),
        np.array([[0, 3], [1, 2]]).T,
        np.array([[0, 1], [1, 2], [2, 3]]).T,
    )

    split_segment_fully_overlapping_common_start = (
        np.array([[0, 1, 2], [0, 0, 0]]),
        np.array([[0, 2], [0, 1]]).T,
        np.array([[0, 1], [1, 2]]).T,
    )

    split_segment_fully_overlapping_switched_order = (
        np.array([[0, 1, 2, 3], [0, 0, 0, 0]]),
        np.array([[0, 3], [2, 1]]).T,
        np.array([[0, 1], [1, 2], [2, 3]]).T,
    )

    split_segment_fully_overlapping_common_end = (
        np.array([[0, 1, 2], [0, 0, 0]]),
        np.array([[0, 2], [1, 2]]).T,
        np.array([[0, 1], [1, 2]]).T,
    )

    @pytest.mark.parametrize(
        "p, lines, lines_known",
        [
            split_segment_partly_overlapping_switched_order,
            split_segment_fully_overlapping,
            split_segment_fully_overlapping_common_start,
            split_segment_fully_overlapping_switched_order,
            split_segment_fully_overlapping_common_end,
        ],
    )
    def test_split_segments(self, p, lines, lines_known):
        """The actual testing of spliting segments."""
        new_pts, new_lines, _ = pp.intersections.split_intersecting_segments_2d(
            p, lines
        )
        new_lines = np.sort(new_lines, axis=0)

        assert compare_arrays(new_pts, p)
        assert compare_arrays(new_lines, lines_known)


class TestLinesIntersect:
    def test_lines_intersect_segments_do_not(self):
        s0 = np.array([0.3, 0.3])
        e0 = np.array([0.5, 0.5])
        s1 = np.array([0, 2 / 3])
        e1 = np.array([1, 2 / 3])
        pi = pp.intersections.segments_2d(s0, e0, s1, e1)
        assert pi is None or len(pi) == 0

    def test_parallel_not_colinear(self):
        s0 = np.array([0, 0])
        e0 = np.array([1, 0])
        s1 = np.array([0, 1])
        e1 = np.array([1, 1])

        pi = pp.intersections.segments_2d(s0, e0, s1, e1)
        assert pi is None

    def test_colinear_not_intersecting(self):
        s0 = np.array([0, 0])
        e0 = np.array([1, 0])
        s1 = np.array([2, 0])
        e1 = np.array([3, 0])

        pi = pp.intersections.segments_2d(s0, e0, s1, e1)
        assert pi is None

    s_0 = np.array([0, 0])
    e_0 = np.array([2, 0])
    s_1 = np.array([1, 0])
    e_1 = np.array([3, 0])

    @pytest.mark.parametrize(
        "s0, e0, s1, e1",
        [
            # Partially overlapping segments
            (s_0, e_0, s_1, e_1),
            (e_0, s_0, s_1, e_1),
            (s_0, e_0, e_1, s_1),
            (e_0, s_0, e_1, s_1),
            # Fully overlapping segments
            (np.array([0, 0]), np.array([3, 0]), np.array([1, 0]), np.array([2, 0])),
        ],
    )
    def test_overlapping_segments(self, s0, e0, s1, e1):
        pi = pp.intersections.segments_2d(s0, e0, s1, e1)
        assert (pi[0, 0] == 1 and pi[0, 1] == 2) or (pi[0, 0] == 2 and pi[0, 1] == 1)
        assert np.allclose(pi[1], 0)

    def test_meeting_in_point(self):
        s0 = np.array([0, 0])
        e0 = np.array([1, 0])
        s1 = np.array([1, 0])
        e1 = np.array([2, 0])

        pi = pp.intersections.segments_2d(s0, e0, s1, e1)
        assert pi[0, 0] == 1 and pi[1, 0] == 0

    def test_segments_polygon_inside(self):
        s = np.array([0.5, 0.5, -0.5])
        e = np.array([0.5, 0.5, 0.5])

        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        is_cross, pt = pp.intersections.segments_polygon(s, e, p)
        assert is_cross[0] and compare_arrays(pt[:, 0], [0.5, 0.5, 0.0])

    def test_segments_polygon_outside(self):
        s = np.array([1.5, 0.5, -0.5])
        e = np.array([1.5, 0.5, 0.5])

        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        is_cross, pt = pp.intersections.segments_polygon(s, e, p)
        assert not is_cross[0]

    def test_segments_polygon_corner(self):
        s = np.array([1, 1, -0.5])
        e = np.array([1, 1, 0.5])

        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        is_cross, pt = pp.intersections.segments_polygon(s, e, p)
        assert not is_cross[0]

    def test_segments_polygon_edge(self):
        s = np.array([1, 0.5, -0.5])
        e = np.array([1, 0.5, 0.5])

        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        is_cross, pt = pp.intersections.segments_polygon(s, e, p)
        assert not is_cross[0]

    def test_segments_polygon_one_node_on(self):
        s = np.array([0.5, 0.5, 0])
        e = np.array([0.5, 0.5, 0.5])

        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        is_cross, pt = pp.intersections.segments_polygon(s, e, p)
        assert is_cross[0] and compare_arrays(pt[:, 0], [0.5, 0.5, 0.0])

    def test_segments_polygon_immersed(self):
        s = np.array([0.25, 0.25, 0])
        e = np.array([0.75, 0.75, 0])

        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        is_cross, _ = pp.intersections.segments_polygon(s, e, p)
        assert not is_cross[0]

    def test_segments_polyhedron_fully_inside(self):
        """Test for a segment with both extrema immersed in the polyhedron (cube)"""
        s = np.array([0.5, 0.5, 0.25])
        e = np.array([0.5, 0.5, 0.75])

        p = np.array(
            [
                [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            ]
        )

        pts, s_in, e_in, perc = pp.intersections.segments_polyhedron(s, e, p)
        assert pts[0].size == 0 and s_in[0] and e_in[0] and perc[0] == 1

    def test_segments_polyhedron_fully_outside(self):
        """Test for a segment with both extrema outside the polyhedron (cube)"""
        s = np.array([0.5, 0.5, 1.25])
        e = np.array([0.5, 0.5, 1.75])

        p = np.array(
            [
                [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            ]
        )

        pts, s_in, e_in, perc = pp.intersections.segments_polyhedron(s, e, p)
        assert pts[0].size == 0 and (not s_in[0]) and (not e_in[0]) and perc[0] == 0

    def test_segments_polyhedron_one_inside_one_outside(self):
        """Test for a segment with one extrema inside and one outside the polyhedron
        (cube)"""
        s = np.array([0.5, 0.5, 0.5])
        e = np.array([0.5, 0.5, 1.5])

        p = np.array(
            [
                [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            ]
        )

        pts, s_in, e_in, perc = pp.intersections.segments_polyhedron(s, e, p)
        assert (
            compare_arrays(pts[0].T, [0.5, 0.5, 1])
            and s_in[0]
            and (not e_in[0])
            and perc[0] == 0.5
        )

    def test_segments_polyhedron_edge(self):
        """Test for a segment that partially overlap one of the edge (face boundary) of the
        polyhedron (cube)"""
        s = np.array([1, 0, 0.5])
        e = np.array([1, 0, 1.5])

        p = np.array(
            [
                [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            ]
        )

        pts, s_in, e_in, perc = pp.intersections.segments_polyhedron(s, e, p)
        assert pts[0].size == 0 and (not s_in[0]) and (not e_in[0]) and perc[0] == 0

    def test_segments_polyhedron_face(self):
        """Test for a segment that partially overlap a face of the polyhedron (cube)"""
        s = np.array([0.5, 0, 0.5])
        e = np.array([0.5, 0, 1.5])

        p = np.array(
            [
                [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            ]
        )

        pts, s_in, e_in, perc = pp.intersections.segments_polyhedron(s, e, p)
        assert pts[0].size == 0 and (not s_in[0]) and (not e_in[0]) and perc[0] == 0


class TestLineTesselation:
    def test_tesselation_do_not(self):
        p1 = np.array([[0.3, 0.3, 0], [0.5, 0.5, 0], [0.9, 0.9, 0]]).T
        p2 = np.array([[0.4, 0.4, 0.1], [1.0, 1.0, 0.1]]).T
        l1 = np.array([[0, 1], [1, 2]]).T
        l2 = np.array([[0, 1]]).T
        intersect = pp.intersections.line_tessellation(p1, p2, l1, l2)
        assert len(intersect) == 0

    def test_tesselation_do(self):
        p1 = np.array([[0.0, 0.0, 0], [0.5, 0.5, 0], [1.0, 1.0, 0]]).T
        p2 = np.array([[0.25, 0.25, 0], [1.0, 1.0, 0]]).T
        l1 = np.array([[0, 1], [1, 2]]).T
        l2 = np.array([[0, 1]]).T
        intersections = pp.intersections.line_tessellation(p1, p2, l1, l2)
        for inter in intersections:
            if inter[0] == 0:
                if inter[1] == 0:
                    assert inter[2] == np.sqrt(0.25**2 + 0.25**2)
                    continue
            elif inter[0] == 1:
                if inter[1] == 1:
                    assert inter[2] == np.sqrt(0.5**2 + 0.5**2)
            else:
                assert False


class TestSurfaceTesselation:
    # The naming is a bit confusing here, the sets of polygons do not cover the same
    # areas, thus they are not tessalations, but the tests serve their purpose.

    def test_two_tessalations_one_cell_each(self):
        # Two triangles, partly overlapping.
        p1 = [np.array([[0, 1, 0], [0, 0, 1]])]
        p2 = [np.array([[0, 1, 0], [0, 1, 1]])]

        isect, mappings = pp.intersections.surface_tessellations([p1, p2])

        known_isect = np.array([[0, 0.5, 0], [0, 0.5, 1]])

        # Mappings are both identity mappings

        assert compare_arrays(isect[0], known_isect)
        for i in range(2):
            assert mappings[i].shape == (1, 1)
            assert mappings[i].toarray() == np.array([[1]])

    def test_two_tessalations_no_overlap(self):
        # Two triangles, partly overlapping.
        p1 = [np.array([[0, 1, 0], [0, 0, 1]])]
        p2 = [np.array([[0, 1, 0], [1, 1, 2]])]

        isect, mappings = pp.intersections.surface_tessellations([p1, p2])

        # Mappings are both identity mappings

        assert len(isect) == 0
        for i in range(2):
            assert mappings[i].shape == (0, 1)

    def test_two_tessalations_one_quad(self):
        # Quad and triangle. Partly overlapping
        p1 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]])]
        p2 = [np.array([[0, 1, 0], [0, 1, 2]])]

        isect, mappings = pp.intersections.surface_tessellations([p1, p2])

        known_isect = np.array([[0, 1, 0], [0, 1, 1]])

        # Mappings are both identity mappings

        assert compare_arrays(isect[0], known_isect)
        for i in range(2):
            assert mappings[i].shape == (1, 1)
            assert mappings[i].toarray() == np.array([[1]])

    def test_two_tessalations_non_convex_intersection(self):
        # Quad and triangle. Partly overlapping
        p1 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]])]
        p2 = [np.array([[0, 1, 1, 0.5, 0], [0, 0, 1, 0.5, 1]])]

        isect, mappings = pp.intersections.surface_tessellations([p1, p2])

        known_isect = p2[0]

        # Mappings are both identity mappings

        assert compare_arrays(isect[0], known_isect)
        for i in range(2):
            assert mappings[i].shape == (1, 1)
            assert mappings[i].toarray() == np.array([[1]])

    def test_two_tessalations_one_with_two_cells(self):
        # First consists of quad + triangle
        p1 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]]), np.array([[0, 1, 0], [1, 1, 2]])]
        # second is a single triangle
        p2 = [np.array([[0, 1, 0], [0, 1, 2]])]

        isect, mappings = pp.intersections.surface_tessellations([p1, p2])

        # intersection is split in two
        known_isect = [
            np.array([[0, 1, 0], [0, 1, 1]]),
            np.array([[0, 1, 0], [1, 1, 2]]),
        ]

        assert mappings[0].shape == (2, 2)
        # To find the order of triangles in isect relative to the polygons in p1,
        # we consider the mapping for p1
        if mappings[0][0, 0] == 1:
            assert compare_arrays(isect[0], known_isect[0])
            assert compare_arrays(isect[1], known_isect[1])
            assert compare_arrays(mappings[0].toarray(), np.array([[1, 0], [0, 1]]))
        else:
            assert compare_arrays(isect[0], known_isect[1])
            assert compare_arrays(isect[1], known_isect[0])
            assert compare_arrays(mappings[0].toarray(), np.array([[0, 1], [1, 0]]))

        # p2 is much simpler
        assert mappings[1].shape == (2, 1)
        assert compare_arrays(mappings[1].toarray(), np.array([[1], [1]]))

    def test_two_tessalations_two_cells_each_one_no_overlap(self):
        # First consists of quad + triangle
        p1 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]]), np.array([[0, 1, 0], [1, 1, 2]])]
        # second has two triangles, one of which has no overlap with the first tessalation
        p2 = [
            np.array([[0, 1, 0], [0, 1, 2]]),
            np.array([[0, -1, 0], [0, 1, 2]]),
        ]

        isect, mappings = pp.intersections.surface_tessellations([p1, p2])
        # No need to test intersection points, they are identical with
        # self.test_two_tessalations_one_with_two_cells()

        assert mappings[1].shape == (2, 2)
        assert compare_arrays(mappings[1].toarray(), np.array([[1, 0], [1, 0]]))

    def test_three_tessalations(self):
        # First consists of quad + triangle
        p1 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]]), np.array([[0, 1, 0], [1, 1, 2]])]
        # second is a single triangle
        p2 = [np.array([[0, 1, 0], [0, 1, 2]])]
        # A third, with a single triangle
        p3 = [np.array([[0, 1, 0], [0, 0, 1]])]

        isect, mappings = pp.intersections.surface_tessellations([p1, p2, p3])

        assert len(isect) == 1
        assert compare_arrays(isect[0], np.array([[0, 0.5, 0], [0, 0.5, 1]]))

        assert len(mappings) == 3
        assert mappings[0].shape == (1, 2)
        assert compare_arrays(mappings[0].toarray(), np.array([[1, 0]]))

        assert mappings[1].shape == (1, 1)
        assert compare_arrays(mappings[1].toarray(), np.array([[1]]))
        assert mappings[2].shape == (1, 1)
        assert compare_arrays(mappings[2].toarray(), np.array([[1]]))

    ## Tests of the simplex tessalation of the subdivision
    def test_return_simplex(self):
        # First is unit square, split into two
        p1 = [
            np.array([[0, 1, 1, 0], [0, 0, 0.5, 0.5]]),
            np.array([[0, 1, 1, 0], [0.5, 0.5, 1, 1]]),
        ]
        # Second is unit square
        p2 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]])]

        isect, mappings = pp.intersections.surface_tessellations(
            [p1, p2], return_simplexes=True
        )

        # Intersection is split into eight, with the
        known_isect = [
            np.array([[0, 0.5, 0], [0, 0.25, 0.5]]),
            np.array([[0, 0.5, 1], [0, 0.25, 0]]),
            np.array([[0, 0.5, 1], [0.5, 0.25, 0.5]]),
            np.array([[1, 0.5, 1], [0, 0.25, 0.5]]),
            np.array([[0, 0.5, 0], [0.5, 0.75, 1]]),
            np.array([[0, 0.5, 1], [0.5, 0.75, 0.5]]),
            np.array([[0, 0.5, 1], [1, 0.75, 1]]),
            np.array([[1, 0.5, 1], [0.5, 0.75, 1]]),
        ]
        num_known = len(known_isect)
        num_isect = len(isect)
        assert num_known == num_isect

        found_isect = np.zeros(num_isect, dtype=bool)
        found_known_isect = np.zeros(num_known, dtype=bool)

        for i in range(num_isect):
            for k in range(num_known):
                if compare_arrays(isect[i], known_isect[k]):
                    assert not (found_isect[i])
                    found_isect[i] = True
                    assert not (found_known_isect[k])
                    found_known_isect[k] = True

                    # Also check that the mapping is updated correctly for the first
                    # polygon (the second is trivial)
                    if k < 4:  # The lower quad
                        assert mappings[0][i, 0] == 1
                        assert mappings[0][i, 1] == 0
                    else:
                        assert mappings[0][i, 1] == 1
                        assert mappings[0][i, 0] == 0

        assert np.all(found_isect)
        assert np.all(found_known_isect)

    def test_return_simplex_non_convex_intersection_raise_error(self):
        # Non-convex intersection. Should return an error
        p1 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]])]
        p2 = [np.array([[0, 1, 1, 0.5, 0], [0, 0, 1, 0.5, 1]])]

        with pytest.raises(NotImplementedError):
            pp.intersections.surface_tessellations([p1, p2], return_simplexes=True)


class TestSegmentSegmentIntersection:
    p_1 = np.array([0, -1, -1])
    p_2 = np.array([0, 1, 1])
    p_3 = np.array([-1, 0, 1])
    p_4 = np.array([1, 0, -1])

    @pytest.mark.parametrize(
        "p1, p2, p3, p4",
        [
            # 3D line crossing in origin:
            (p_1, p_2, p_3, p_4),
            # Arbitrary order of the points:
            (p_2, p_1, p_3, p_4),
            (p_1, p_2, p_4, p_3),
            (p_2, p_1, p_4, p_3),
        ],
    )
    def test_argument_order_arbitrary(self, p1, p2, p3, p4):
        # Order of input arguments should be arbitrary.
        p_known = np.zeros(3)
        p_i = pp.intersections.segments_3d(p1, p2, p3, p4)
        assert np.allclose(p_i, p_known)

    def test_pass_in_z_coord(self):
        # The (x,y) coordinates gives intersection in origin, but z coordinates
        # do not match
        p_1 = np.array([-1, -1, -1])
        p_2 = np.array([1, 1, -1])
        p_3 = np.array([1, -1, 1])
        p_4 = np.array([-1, 1, 1])

        p_i = pp.intersections.segments_3d(p_1, p_2, p_3, p_4)
        assert p_i is None

    def test_lines_cross_segments_not(self):
        p_1 = np.array([-1, 0, -1])
        p_2 = np.array([0, 0, 0])
        p_3 = np.array([1, -1, 1])
        p_4 = np.array([1, 1, 1])

        p_i = pp.intersections.segments_3d(p_1, p_2, p_3, p_4)
        assert p_i is None

    def test_parallel_lines(self):
        p_1 = np.zeros(3)
        p_2 = np.array([1, 0, 0])
        p_3 = np.array([0, 1, 0])
        p_4 = np.array([1, 1, 0])

        p_i = pp.intersections.segments_3d(p_1, p_2, p_3, p_4)
        assert p_i is None

    def test_L_intersection(self):
        p_1 = np.zeros(3)
        p_2 = np.random.rand(3)
        p_3 = np.random.rand(3)

        p_i = pp.intersections.segments_3d(p_1, p_2, p_2, p_3)
        assert compare_arrays(p_i, p_2.reshape((-1, 1)))

    def test_equal_lines_segments_not_overlapping(self):
        p_1 = np.ones(3)
        p_2 = 0 * p_1
        p_3 = 2 * p_1
        p_4 = 3 * p_1

        p_int = pp.intersections.segments_3d(p_1, p_2, p_3, p_4)
        assert p_int is None

    def test_both_aligned_with_axis(self):
        # Both lines are aligned an axis,
        p_1 = np.array([-1, -1, 0])
        p_2 = np.array([-1, 1, 0])
        p_3 = np.array([-1, 0, -1])
        p_4 = np.array([-1, 0, 1])

        p_int = pp.intersections.segments_3d(p_1, p_2, p_3, p_4)
        p_known = np.array([-1, 0, 0]).reshape((-1, 1))
        assert compare_arrays(p_int, p_known)

    def test_segment_fully_overlapped(self):
        # One line is fully covered by another
        p_1 = np.ones(3)
        p_2 = 2 * p_1
        p_3 = 0 * p_1
        p_4 = 3 * p_1

        p_int = pp.intersections.segments_3d(p_1, p_2, p_3, p_4)
        p_known_1 = p_1.reshape((-1, 1))
        p_known_2 = p_2.reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_int - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int - p_known_2), axis=0)) < 1e-8

    def test_segments_overlap_input_order(self):
        # Test the order of inputs
        p_1 = np.ones(3)
        p_2 = 2 * p_1
        p_3 = 0 * p_1
        p_4 = 3 * p_1

        p_int_1 = pp.intersections.segments_3d(p_1, p_2, p_3, p_4)
        p_int_2 = pp.intersections.segments_3d(p_2, p_1, p_3, p_4)
        p_int_3 = pp.intersections.segments_3d(p_1, p_2, p_4, p_3)
        p_int_4 = pp.intersections.segments_3d(p_2, p_1, p_4, p_3)

        p_known_1 = p_1.reshape((-1, 1))
        p_known_2 = p_2.reshape((-1, 1))

        assert np.min(np.sum(np.abs(p_int_1 - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_1 - p_known_2), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_2 - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_2 - p_known_2), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_3 - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_3 - p_known_2), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_4 - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int_4 - p_known_2), axis=0)) < 1e-8

    @pytest.fixture
    def assert_equal_segments_3d(self):
        def _assert_equal_segments_3d(p_1, p_2, p_3, p_4):
            p_int = pp.intersections.segments_3d(p_1, p_2, p_3, p_4)
            p_known_1 = p_1.reshape((-1, 1))
            p_known_2 = p_4.reshape((-1, 1))
            assert np.min(np.sum(np.abs(p_int - p_known_1), axis=0)) < 1e-8
            assert np.min(np.sum(np.abs(p_int - p_known_2), axis=0)) < 1e-8

        return _assert_equal_segments_3d

    def test_segments_partly_overlap(self, assert_equal_segments_3d):
        p_1 = np.ones(3)
        p_2 = 3 * p_1
        p_3 = 0 * p_1
        p_4 = 2 * p_1

        assert_equal_segments_3d(p_1, p_2, p_3, p_4)

    def test_random_incline(self, assert_equal_segments_3d):
        p_1 = np.random.rand(3)
        p_2 = 3 * p_1
        p_3 = 0 * p_1
        p_4 = 2 * p_1

        assert_equal_segments_3d(p_1, p_2, p_3, p_4)

    def test_segments_aligned_with_axis(self, assert_equal_segments_3d):
        p_1 = np.array([0, 1, 1])
        p_2 = 3 * p_1
        p_3 = 0 * p_1
        p_4 = 2 * p_1

        assert_equal_segments_3d(p_1, p_2, p_3, p_4)

    def test_constant_y_axis(self):
        p_1 = np.array([1, 0, -1])
        p_2 = np.array([1, 0, 1])
        p_3 = np.array([1.5, 0, 0])
        p_4 = np.array([0, 0, 1.5])

        p_int = pp.intersections.segments_3d(p_1, p_2, p_3, p_4)
        p_known = np.array([1, 0, 0.5]).reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_int - p_known), axis=0)) < 1e-8


class TestSweepAndPrune:
    def pairs_contain(self, pairs, a):
        for pi in range(pairs.shape[1]):
            if a[0] == pairs[0, pi] and a[1] == pairs[1, pi]:
                return True
        return False

    def test_no_intersection(self):
        x_min = np.array([0, 2])
        x_max = np.array([1, 3])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)
        assert pairs.size == 0

    @pytest.fixture
    def assert_pairs_of_overlapping_intervals(self):
        """Helper function for asserting pairs of overlapping intervals.

        Used in seven of the tests below.

        """

        def _assert_pairs_of_overlapping_intervals(x_min, x_max):
            pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

            assert pairs.size == 2
            assert pairs[0, 0] == 0
            assert pairs[1, 0] == 1

        return _assert_pairs_of_overlapping_intervals

    def test_intersection_two_lines(self, assert_pairs_of_overlapping_intervals):
        x_min = np.array([0, 1])
        x_max = np.array([2, 3])

        assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_two_lines_switched_order(
        self, assert_pairs_of_overlapping_intervals
    ):
        x_min = np.array([1, 0])
        x_max = np.array([3, 2])

        assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_two_lines_same_start(
        self, assert_pairs_of_overlapping_intervals
    ):
        x_min = np.array([0, 0])
        x_max = np.array([3, 2])

        assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_two_lines_same_end(
        self, assert_pairs_of_overlapping_intervals
    ):
        x_min = np.array([0, 1])
        x_max = np.array([3, 3])

        assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_two_lines_same_start_and_end(
        self, assert_pairs_of_overlapping_intervals
    ):
        x_min = np.array([0, 0])
        x_max = np.array([3, 3])

        assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_two_lines_one_is_point_no_intersection(self):
        x_min = np.array([0, 1])
        x_max = np.array([0, 2])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        assert pairs.size == 0

    def test_intersection_two_lines_one_is_point_intersection(
        self, assert_pairs_of_overlapping_intervals
    ):
        x_min = np.array([1, 0])
        x_max = np.array([1, 2])

        assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_three_lines_two_intersect(
        self, assert_pairs_of_overlapping_intervals
    ):
        x_min = np.array([1, 0, 3])
        x_max = np.array([2, 2, 4])

        assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_three_lines_all_intersect(self):
        x_min = np.array([1, 0, 1])
        x_max = np.array([2, 2, 3])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        assert pairs.size == 6
        assert self.pairs_contain(pairs, [0, 1])
        assert self.pairs_contain(pairs, [0, 2])
        assert self.pairs_contain(pairs, [1, 2])

    def test_intersection_three_lines_pairs_intersect(self):
        x_min = np.array([0, 0, 2])
        x_max = np.array([1, 3, 3])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        assert pairs.size == 4
        assert self.pairs_contain(pairs, [0, 1])
        assert self.pairs_contain(pairs, [1, 2])


class TestBoundingBoxIntersection:
    # For all cases, run both 1d search + intersection, and 2d search.
    # They should be equivalent.
    # Note: The tests are only between the bounding boxes of the fractures,
    # not the fractures themselves

    def test_no_intersection(self):
        # Use same coordinates for x and y, that is, the fractures are
        # on the line x = y.
        x_min = np.array([0, 2])
        x_max = np.array([1, 3])

        x_pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)
        y_pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)
        pairs_1 = pp.intersections._intersect_pairs(x_pairs, y_pairs)
        assert pairs_1.size == 0

        combined_pairs = pp.intersections._identify_overlapping_rectangles(
            x_min, x_max, x_min, x_max
        )
        assert combined_pairs.size == 0

    def test_intersection_x_not_y(self):
        # The points are overlapping on the x-axis but not on the y-axis
        x_min = np.array([0, 0])
        x_max = np.array([2, 2])

        y_min = np.array([0, 5])
        y_max = np.array([2, 7])

        x_pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)
        y_pairs = pp.intersections._identify_overlapping_intervals(y_min, y_max)
        pairs_1 = pp.intersections._intersect_pairs(x_pairs, y_pairs)
        assert pairs_1.size == 0

        combined_pairs = pp.intersections._identify_overlapping_rectangles(
            x_min, x_max, y_min, y_max
        )
        assert combined_pairs.size == 0

    def test_intersection_x_and_y(self):
        # The points are overlapping on the x-axis but not on the y-axis
        x_min = np.array([0, 0])
        x_max = np.array([2, 2])

        y_min = np.array([0, 1])
        y_max = np.array([2, 3])

        x_pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)
        y_pairs = pp.intersections._identify_overlapping_intervals(y_min, y_max)
        pairs_1 = np.sort(pp.intersections._intersect_pairs(x_pairs, y_pairs), axis=0)
        assert pairs_1.size == 2

        combined_pairs = np.sort(
            pp.intersections._identify_overlapping_rectangles(
                x_min, x_max, y_min, y_max
            ),
            axis=0,
        )
        assert combined_pairs.size == 2

        assert compare_arrays(pairs_1, combined_pairs)

    def test_lines_in_square(self):
        # Lines in square, all should overlap
        x_min = np.array([0, 1, 0, 0])
        x_max = np.array([1, 1, 1, 0])

        y_min = np.array([0, 0, 1, 0])
        y_max = np.array([0, 1, 1, 1])

        x_pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)
        y_pairs = pp.intersections._identify_overlapping_intervals(y_min, y_max)
        pairs_1 = np.sort(pp.intersections._intersect_pairs(x_pairs, y_pairs), axis=0)
        assert pairs_1.shape[1] == 4

        combined_pairs = np.sort(
            pp.intersections._identify_overlapping_rectangles(
                x_min, x_max, y_min, y_max
            ),
            axis=0,
        )
        assert combined_pairs.shape[1] == 4

        assert compare_arrays(pairs_1, combined_pairs)


class TestFractureIntersectionRemoval:
    """Tests for functions used to remove intersections between 1d fractures."""

    def test_lines_crossing_origin(self):
        p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])
        lines = np.array([[0, 2], [1, 3], [1, 2], [3, 4]])

        x_min, x_max, y_min, y_max = pp.intersections._axis_aligned_bounding_box_2d(
            p, lines
        )

        new_pts, new_lines, _ = pp.intersections.split_intersecting_segments_2d(
            p, lines
        )

        p_known = np.hstack((p, np.array([[0], [0]])))

        lines_known = np.array([[0, 4, 2, 4], [4, 1, 4, 3], [1, 1, 2, 2], [3, 3, 4, 4]])

        assert compare_arrays(new_pts, p_known)
        assert compare_arrays(new_lines, lines_known)

    def test_lines_no_crossing(self):
        p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

        lines = np.array([[0, 1], [2, 3]])
        new_pts, new_lines, _ = pp.intersections.split_intersecting_segments_2d(
            p, lines
        )
        assert compare_arrays(new_pts, p)
        assert compare_arrays(new_lines, lines)

    def test_three_lines_no_crossing(self):
        # This test gave an error at some point
        p = np.array(
            [[0.0, 0.0, 0.3, 1.0, 1.0, 0.5], [2 / 3, 1 / 0.7, 0.3, 2 / 3, 1 / 0.7, 0.5]]
        )
        lines = np.array([[0, 3], [1, 4], [2, 5]]).T

        new_pts, new_lines, _ = pp.intersections.split_intersecting_segments_2d(
            p, lines
        )
        p_known = p
        assert compare_arrays(new_pts, p_known)
        assert compare_arrays(new_lines, lines)

    def test_three_lines_one_crossing(self):
        # This test gave an error at some point
        p = np.array(
            [[0.0, 0.5, 0.3, 1.0, 0.3, 0.5], [2 / 3, 0.3, 0.3, 2 / 3, 0.5, 0.5]]
        )
        lines = np.array([[0, 3], [2, 5], [1, 4]]).T

        new_pts, new_lines, _ = pp.intersections.split_intersecting_segments_2d(
            p, lines
        )
        p_known = np.hstack((p, np.array([[0.4], [0.4]])))
        lines_known = np.array([[0, 3], [2, 6], [6, 5], [1, 6], [6, 4]]).T
        assert compare_arrays(new_pts, p_known)
        assert compare_arrays(new_lines, lines_known)

    def test_overlapping_lines(self):
        p = np.array([[-0.6, 0.4, 0.4, -0.6, 0.4], [-0.5, -0.5, 0.5, 0.5, 0.0]])
        lines = np.array([[0, 0, 1, 1, 2], [1, 3, 2, 4, 3]])
        new_pts, new_lines, _ = pp.intersections.split_intersecting_segments_2d(
            p, lines
        )

        lines_known = np.array([[0, 1], [0, 3], [1, 4], [2, 4], [2, 3]]).T
        assert compare_arrays(new_pts, p)
        assert compare_arrays(new_lines, lines_known)


class TestFractureBoundaryIntersection:
    """
    Test of algorithm for constraining a fracture a bounding box.

    Since that algorithm uses fracture intersection methods, the tests functions as
    partial test for the wider fracture intersection framework as well. Full tests of
    the latter are too time consuming to fit into a unit test.

    Now the boundary is defined as set of "fake" fractures, all fracture network have
    2*dim additional fractures (hence the + 6 in the assertions)
    """

    def setup(self):
        self.f_1 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]]),
            check_convexity=False,
        )
        self.domain = unit_domain(3)

    def _a_in_b(self, a, b, tol=1e-5):
        for i in range(a.shape[1]):
            if not np.any(np.abs(a[:, i].reshape((-1, 1)) - b).max(axis=0) < tol):
                return False
        return True

    def _arrays_equal(self, a, b):
        return self._a_in_b(a, b) and self._a_in_b(b, a)

    def test_completely_outside_lower(self):
        self.setup()
        f = self.f_1
        f.pts[0] -= 2
        network = pp.create_fracture_network([f])
        network.impose_external_boundary(self.domain)
        assert len(network.fractures) == (0 + 6)

    def test_outside_west_bottom(self):
        self.setup()
        f = self.f_1
        f.pts[0] -= 0.5
        f.pts[2] -= 1.5
        network = pp.create_fracture_network([f])
        network.impose_external_boundary(self.domain)
        assert len(network.fractures) == (0 + 6)

    def test_intersect_one(self):
        self.setup()
        f = self.f_1
        f.pts[0] -= 0.5
        f.pts[2, :] = [0.2, 0.2, 0.8, 0.8]
        network = pp.create_fracture_network([f])
        network.impose_external_boundary(self.domain)
        p_known = np.array(
            [[0.0, 0.5, 0.5, 0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )
        assert len(network.fractures) == (1 + 6)
        p_comp = network.fractures[0].pts
        assert self._arrays_equal(p_known, p_comp)

    def test_intersect_two_same(self):
        self.setup()
        f = self.f_1
        f.pts[0, :] = [-0.5, 1.5, 1.5, -0.5]
        f.pts[2, :] = [0.2, 0.2, 0.8, 0.8]
        network = pp.create_fracture_network([f])
        network.impose_external_boundary(self.domain)
        p_known = np.array([[0.0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
        assert len(network.fractures) == (1 + 6)
        p_comp = network.fractures[0].pts
        assert self._arrays_equal(p_known, p_comp)

    def test_incline_in_plane(self):
        self.setup()
        f = self.f_1
        f.pts[0] -= 0.5
        f.pts[2, :] = [0, -0.5, 0.5, 1]
        network = pp.create_fracture_network([f])
        network.impose_external_boundary(self.domain)
        p_known = np.array(
            [[0.0, 0.5, 0.5, 0], [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.5, 0.75]]
        )
        assert len(network.fractures) == (1 + 6)
        p_comp = network.fractures[0].pts
        assert self._arrays_equal(p_known, p_comp)

    def test_full_incline(self):
        self.setup()
        p = np.array([[-0.5, 0.5, 0.5, -0.5], [0.5, 0.5, 1.5, 1.5], [-0.5, -0.5, 1, 1]])
        f = pp.PlaneFracture(p, check_convexity=False)
        network = pp.create_fracture_network([f])
        network.impose_external_boundary(self.domain)
        p_known = np.array(
            [[0.0, 0.5, 0.5, 0], [5.0 / 6, 5.0 / 6, 1, 1], [0.0, 0.0, 0.25, 0.25]]
        )
        assert len(network.fractures) == (1 + 6)
        p_comp = network.fractures[0].pts
        assert self._arrays_equal(p_known, p_comp)
