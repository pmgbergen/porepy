"""This file contains testing of funcionality found within
porepy.geometry.intersections.py.
"""

import numpy as np
import pytest

import porepy as pp
from porepy import intersections
from porepy.applications.md_grids.domains import unit_cube_domain as unit_domain
from porepy.applications.test_utils.arrays import compare_arrays


class TestSplitIntersectingLines2D:
    """
    Various tests of split_intersecting_segments_2d.

    """

    def test_lines_no_crossing(self):
        p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

        lines = np.array([[0, 1], [2, 3]])
        new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)
        assert compare_arrays(new_pts, p)
        assert compare_arrays(new_lines, lines)

    def test_three_lines_no_crossing(self):
        p = np.array(
            [[0.0, 0.0, 0.3, 1.0, 1.0, 0.5], [2 / 3, 1 / 0.7, 0.3, 2 / 3, 1 / 0.7, 0.5]]
        )
        lines = np.array([[0, 3], [1, 4], [2, 5]]).T

        new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)

        assert compare_arrays(new_pts, p)
        assert compare_arrays(new_lines, lines)

    def test_three_lines_one_crossing(self):
        p = np.array(
            [[0.0, 0.5, 0.3, 1.0, 0.3, 0.5], [2 / 3, 0.3, 0.3, 2 / 3, 0.5, 0.5]]
        )
        lines = np.array([[0, 3], [2, 5], [1, 4]]).T

        new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)
        p_known = np.hstack((p, np.array([[0.4], [0.4]])))
        lines_known = np.array([[0, 3], [2, 6], [6, 5], [1, 6], [6, 4]]).T
        assert compare_arrays(new_pts, p_known)
        assert compare_arrays(new_lines, lines_known)

    def test_split_segment_partly_overlapping(self):
        p = np.array([[0, 1, 2, 3], [0, 0, 0, 0]])
        lines = np.array([[0, 2], [1, 3]]).T

        new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)

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
        new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)
        new_lines = np.sort(new_lines, axis=0)

        assert compare_arrays(new_pts, p)
        assert compare_arrays(new_lines, lines_known)


class TestLinesIntersect:
    """Class for testing of segments_2d."""

    def test_lines_intersect_segments_do_not(self):
        s0 = np.array([0.3, 0.3])
        e0 = np.array([0.5, 0.5])
        s1 = np.array([0, 2 / 3])
        e1 = np.array([1, 2 / 3])
        pi = intersections.segments_2d(s0, e0, s1, e1)
        assert pi is None or len(pi) == 0

    def test_parallel_not_colinear(self):
        s0 = np.array([0, 0])
        e0 = np.array([1, 0])
        s1 = np.array([0, 1])
        e1 = np.array([1, 1])

        pi = intersections.segments_2d(s0, e0, s1, e1)
        assert pi is None

    def test_colinear_not_intersecting(self):
        s0 = np.array([0, 0])
        e0 = np.array([1, 0])
        s1 = np.array([2, 0])
        e1 = np.array([3, 0])

        pi = intersections.segments_2d(s0, e0, s1, e1)
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
        pi = intersections.segments_2d(s0, e0, s1, e1)
        assert (pi[0, 0] == 1 and pi[0, 1] == 2) or (pi[0, 0] == 2 and pi[0, 1] == 1)
        assert np.allclose(pi[1], 0)

    def test_meeting_in_point(self):
        s0 = np.array([0, 0])
        e0 = np.array([1, 0])
        s1 = np.array([1, 0])
        e1 = np.array([2, 0])

        pi = intersections.segments_2d(s0, e0, s1, e1)
        assert pi[0, 0] == 1 and pi[1, 0] == 0

    def test_segments_polygon_inside(self):
        s = np.array([0.5, 0.5, -0.5])
        e = np.array([0.5, 0.5, 0.5])

        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        is_cross, pt = intersections.segments_polygon(s, e, p)
        assert is_cross[0] and compare_arrays(pt[:, 0], [0.5, 0.5, 0.0])

    def test_segments_polygon_outside(self):
        s = np.array([1.5, 0.5, -0.5])
        e = np.array([1.5, 0.5, 0.5])

        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        is_cross, pt = intersections.segments_polygon(s, e, p)
        assert not is_cross[0]

    def test_segments_polygon_corner(self):
        s = np.array([1, 1, -0.5])
        e = np.array([1, 1, 0.5])

        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        is_cross, pt = intersections.segments_polygon(s, e, p)
        assert not is_cross[0]

    def test_segments_polygon_edge(self):
        s = np.array([1, 0.5, -0.5])
        e = np.array([1, 0.5, 0.5])

        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        is_cross, pt = intersections.segments_polygon(s, e, p)
        assert not is_cross[0]

    def test_segments_polygon_one_node_on(self):
        s = np.array([0.5, 0.5, 0])
        e = np.array([0.5, 0.5, 0.5])

        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        is_cross, pt = intersections.segments_polygon(s, e, p)
        assert is_cross[0] and compare_arrays(pt[:, 0], [0.5, 0.5, 0.0])

    def test_segments_polygon_immersed(self):
        s = np.array([0.25, 0.25, 0])
        e = np.array([0.75, 0.75, 0])

        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        is_cross, _ = intersections.segments_polygon(s, e, p)
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

        pts, s_in, e_in, perc = intersections.segments_polyhedron(s, e, p)
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

        pts, s_in, e_in, perc = intersections.segments_polyhedron(s, e, p)
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

        pts, s_in, e_in, perc = intersections.segments_polyhedron(s, e, p)
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

        pts, s_in, e_in, perc = intersections.segments_polyhedron(s, e, p)
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

        pts, s_in, e_in, perc = intersections.segments_polyhedron(s, e, p)
        assert pts[0].size == 0 and (not s_in[0]) and (not e_in[0]) and perc[0] == 0


class TestLineTesselation:
    def test_tesselation_do_not_intersect(self):
        p1 = np.array([[0.3, 0.3, 0], [0.5, 0.5, 0], [0.9, 0.9, 0]]).T
        p2 = np.array([[0.4, 0.4, 0.1], [1.0, 1.0, 0.1]]).T
        l1 = np.array([[0, 1], [1, 2]]).T
        l2 = np.array([[0, 1]]).T
        intersect = intersections.line_tessellation(p1, p2, l1, l2)
        assert len(intersect) == 0

    def test_tesselation_do_intersect(self):
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

        isect, mappings = intersections.surface_tessellations([p1, p2])

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

        isect, mappings = intersections.surface_tessellations([p1, p2])

        # Mappings are both identity mappings

        assert len(isect) == 0
        for i in range(2):
            assert mappings[i].shape == (0, 1)

    def test_two_tessalations_one_quad(self):
        # Quad and triangle. Partly overlapping
        p1 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]])]
        p2 = [np.array([[0, 1, 0], [0, 1, 2]])]

        isect, mappings = intersections.surface_tessellations([p1, p2])

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

        isect, mappings = intersections.surface_tessellations([p1, p2])

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

        isect, mappings = intersections.surface_tessellations([p1, p2])

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

        isect, mappings = intersections.surface_tessellations([p1, p2])
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

        isect, mappings = intersections.surface_tessellations([p1, p2, p3])

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
    def test_return_simplexes_in_surface_tessellations(self):
        # First is unit square, split into two
        p1 = [
            np.array([[0, 1, 1, 0], [0, 0, 0.5, 0.5]]),
            np.array([[0, 1, 1, 0], [0.5, 0.5, 1, 1]]),
        ]
        # Second is unit square
        p2 = [np.array([[0, 1, 1, 0], [0, 0, 1, 1]])]

        isect, mappings = intersections.surface_tessellations(
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
            intersections.surface_tessellations([p1, p2], return_simplexes=True)


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
        p_i = intersections.segments_3d(p1, p2, p3, p4)
        assert np.allclose(p_i, p_known)

    def test_pass_in_z_coord(self):
        # The (x,y) coordinates gives intersection in origin, but z coordinates
        # do not match
        p_1 = np.array([-1, -1, -1])
        p_2 = np.array([1, 1, -1])
        p_3 = np.array([1, -1, 1])
        p_4 = np.array([-1, 1, 1])

        p_i = intersections.segments_3d(p_1, p_2, p_3, p_4)
        assert p_i is None

    def test_lines_cross_segments_not(self):
        p_1 = np.array([-1, 0, -1])
        p_2 = np.array([0, 0, 0])
        p_3 = np.array([1, -1, 1])
        p_4 = np.array([1, 1, 1])

        p_i = intersections.segments_3d(p_1, p_2, p_3, p_4)
        assert p_i is None

    def test_parallel_lines(self):
        p_1 = np.zeros(3)
        p_2 = np.array([1, 0, 0])
        p_3 = np.array([0, 1, 0])
        p_4 = np.array([1, 1, 0])

        p_i = intersections.segments_3d(p_1, p_2, p_3, p_4)
        assert p_i is None

    def test_L_intersection(self):
        p_1 = np.zeros(3)
        p_2 = np.random.rand(3)
        p_3 = np.random.rand(3)

        p_i = intersections.segments_3d(p_1, p_2, p_2, p_3)
        assert compare_arrays(p_i, p_2.reshape((-1, 1)))

    def test_equal_lines_segments_not_overlapping(self):
        p_1 = np.ones(3)
        p_2 = 0 * p_1
        p_3 = 2 * p_1
        p_4 = 3 * p_1

        p_int = intersections.segments_3d(p_1, p_2, p_3, p_4)
        assert p_int is None

    def test_both_aligned_with_axis(self):
        # Both lines are aligned an axis,
        p_1 = np.array([-1, -1, 0])
        p_2 = np.array([-1, 1, 0])
        p_3 = np.array([-1, 0, -1])
        p_4 = np.array([-1, 0, 1])

        p_int = intersections.segments_3d(p_1, p_2, p_3, p_4)
        p_known = np.array([-1, 0, 0]).reshape((-1, 1))
        assert compare_arrays(p_int, p_known)

    def test_segment_fully_overlapped(self):
        # One line is fully covered by another
        p_1 = np.ones(3)
        p_2 = 2 * p_1
        p_3 = 0 * p_1
        p_4 = 3 * p_1

        p_int = intersections.segments_3d(p_1, p_2, p_3, p_4)
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

        p_int_1 = intersections.segments_3d(p_1, p_2, p_3, p_4)
        p_int_2 = intersections.segments_3d(p_2, p_1, p_3, p_4)
        p_int_3 = intersections.segments_3d(p_1, p_2, p_4, p_3)
        p_int_4 = intersections.segments_3d(p_2, p_1, p_4, p_3)

        p_known_1 = p_1.reshape((-1, 1))
        p_known_2 = p_2.reshape((-1, 1))

        l1 = np.array(
            [p_int_1, p_int_1, p_int_2, p_int_2, p_int_3, p_int_3, p_int_4, p_int_4]
        )
        l2 = np.array(
            [
                p_known_1,
                p_known_2,
                p_known_1,
                p_known_2,
                p_known_1,
                p_known_2,
                p_known_1,
                p_known_2,
            ]
        )

        for p_int, p_known in zip(l1, l2):
            assert np.min(np.sum(np.abs(p_int - p_known), axis=0)) < 1e-8

    def assert_equal_segments_3d(self, p_1, p_2, p_3, p_4):
        """Helper function for asserting equal segments.

        Used in three of the tests below.

        """
        p_int = intersections.segments_3d(p_1, p_2, p_3, p_4)
        p_known_1 = p_1.reshape((-1, 1))
        p_known_2 = p_4.reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_int - p_known_1), axis=0)) < 1e-8
        assert np.min(np.sum(np.abs(p_int - p_known_2), axis=0)) < 1e-8

    def test_segments_partly_overlap(self):
        p_1 = np.ones(3)
        p_2 = 3 * p_1
        p_3 = 0 * p_1
        p_4 = 2 * p_1

        self.assert_equal_segments_3d(p_1, p_2, p_3, p_4)

    def test_random_incline(self):
        p_1 = np.random.rand(3)
        p_2 = 3 * p_1
        p_3 = 0 * p_1
        p_4 = 2 * p_1

        self.assert_equal_segments_3d(p_1, p_2, p_3, p_4)

    def test_segments_aligned_with_axis(self):
        p_1 = np.array([0, 1, 1])
        p_2 = 3 * p_1
        p_3 = 0 * p_1
        p_4 = 2 * p_1

        self.assert_equal_segments_3d(p_1, p_2, p_3, p_4)

    def test_constant_y_axis(self):
        p_1 = np.array([1, 0, -1])
        p_2 = np.array([1, 0, 1])
        p_3 = np.array([1.5, 0, 0])
        p_4 = np.array([0, 0, 1.5])

        p_int = intersections.segments_3d(p_1, p_2, p_3, p_4)
        p_known = np.array([1, 0, 0.5]).reshape((-1, 1))
        assert np.min(np.sum(np.abs(p_int - p_known), axis=0)) < 1e-8


class TestIdentifyOverlappingIntervals:
    def pairs_contain(self, pairs, a):
        for pi in range(pairs.shape[1]):
            if a[0] == pairs[0, pi] and a[1] == pairs[1, pi]:
                return True
        return False

    def test_no_intersection(self):
        x_min = np.array([0, 2])
        x_max = np.array([1, 3])

        pairs = intersections._identify_overlapping_intervals(x_min, x_max)
        assert pairs.size == 0

    def assert_pairs_of_overlapping_intervals(self, x_min, x_max):
        """Helper function for asserting pairs of overlapping intervals.

        Used in seven of the tests below.

        """
        pairs = intersections._identify_overlapping_intervals(x_min, x_max)

        assert pairs.size == 2
        assert pairs[0, 0] == 0
        assert pairs[1, 0] == 1

    def test_intersection_two_lines(self):
        x_min = np.array([0, 1])
        x_max = np.array([2, 3])

        self.assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_two_lines_switched_order(self):
        x_min = np.array([1, 0])
        x_max = np.array([3, 2])

        self.assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_two_lines_same_start(self):
        x_min = np.array([0, 0])
        x_max = np.array([3, 2])

        self.assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_two_lines_same_end(self):
        x_min = np.array([0, 1])
        x_max = np.array([3, 3])

        self.assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_two_lines_same_start_and_end(self):
        x_min = np.array([0, 0])
        x_max = np.array([3, 3])

        self.assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_two_lines_one_is_point_no_intersection(self):
        x_min = np.array([0, 1])
        x_max = np.array([0, 2])

        pairs = intersections._identify_overlapping_intervals(x_min, x_max)

        assert pairs.size == 0

    def test_intersection_two_lines_one_is_point_intersection(self):
        x_min = np.array([1, 0])
        x_max = np.array([1, 2])

        self.assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_three_lines_two_intersect(self):
        x_min = np.array([1, 0, 3])
        x_max = np.array([2, 2, 4])

        self.assert_pairs_of_overlapping_intervals(x_min, x_max)

    def test_intersection_three_lines_all_intersect(self):
        x_min = np.array([1, 0, 1])
        x_max = np.array([2, 2, 3])

        pairs = intersections._identify_overlapping_intervals(x_min, x_max)

        assert pairs.size == 6
        assert self.pairs_contain(pairs, [0, 1])
        assert self.pairs_contain(pairs, [0, 2])
        assert self.pairs_contain(pairs, [1, 2])

    def test_intersection_three_lines_pairs_intersect(self):
        x_min = np.array([0, 0, 2])
        x_max = np.array([1, 3, 3])

        pairs = intersections._identify_overlapping_intervals(x_min, x_max)

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

        x_pairs = intersections._identify_overlapping_intervals(x_min, x_max)
        y_pairs = intersections._identify_overlapping_intervals(x_min, x_max)
        pairs_1 = intersections._intersect_pairs(x_pairs, y_pairs)
        assert pairs_1.size == 0

        combined_pairs = intersections._identify_overlapping_rectangles(
            x_min, x_max, x_min, x_max
        )
        assert combined_pairs.size == 0

    def test_intersection_x_not_y(self):
        # The points are overlapping on the x-axis but not on the y-axis
        x_min = np.array([0, 0])
        x_max = np.array([2, 2])

        y_min = np.array([0, 5])
        y_max = np.array([2, 7])

        x_pairs = intersections._identify_overlapping_intervals(x_min, x_max)
        y_pairs = intersections._identify_overlapping_intervals(y_min, y_max)
        pairs_1 = intersections._intersect_pairs(x_pairs, y_pairs)
        assert pairs_1.size == 0

        combined_pairs = intersections._identify_overlapping_rectangles(
            x_min, x_max, y_min, y_max
        )
        assert combined_pairs.size == 0

    def test_intersection_x_and_y(self):
        # The points are overlapping on the x-axis but not on the y-axis
        x_min = np.array([0, 0])
        x_max = np.array([2, 2])

        y_min = np.array([0, 1])
        y_max = np.array([2, 3])

        x_pairs = intersections._identify_overlapping_intervals(x_min, x_max)
        y_pairs = intersections._identify_overlapping_intervals(y_min, y_max)
        pairs_1 = np.sort(intersections._intersect_pairs(x_pairs, y_pairs), axis=0)
        assert pairs_1.size == 2

        combined_pairs = np.sort(
            intersections._identify_overlapping_rectangles(x_min, x_max, y_min, y_max),
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

        x_pairs = intersections._identify_overlapping_intervals(x_min, x_max)
        y_pairs = intersections._identify_overlapping_intervals(y_min, y_max)
        pairs_1 = np.sort(intersections._intersect_pairs(x_pairs, y_pairs), axis=0)
        assert pairs_1.shape[1] == 4

        combined_pairs = np.sort(
            intersections._identify_overlapping_rectangles(x_min, x_max, y_min, y_max),
            axis=0,
        )
        assert combined_pairs.shape[1] == 4

        assert compare_arrays(pairs_1, combined_pairs)


class TestFractureIntersectionRemoval:
    """Tests for functions used to remove intersections between 1d fractures."""

    def test_lines_crossing_origin(self):
        p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])
        lines = np.array([[0, 2], [1, 3], [1, 2], [3, 4]])

        x_min, x_max, y_min, y_max = intersections._axis_aligned_bounding_box_2d(
            p, lines
        )

        new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)

        p_known = np.hstack((p, np.array([[0], [0]])))

        lines_known = np.array([[0, 4, 2, 4], [4, 1, 4, 3], [1, 1, 2, 2], [3, 3, 4, 4]])

        assert compare_arrays(new_pts, p_known)
        assert compare_arrays(new_lines, lines_known)

    def test_lines_no_crossing(self):
        p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

        lines = np.array([[0, 1], [2, 3]])
        new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)
        assert compare_arrays(new_pts, p)
        assert compare_arrays(new_lines, lines)

    def test_three_lines_no_crossing(self):
        p = np.array(
            [[0.0, 0.0, 0.3, 1.0, 1.0, 0.5], [2 / 3, 1 / 0.7, 0.3, 2 / 3, 1 / 0.7, 0.5]]
        )
        lines = np.array([[0, 3], [1, 4], [2, 5]]).T

        new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)
        p_known = p
        assert compare_arrays(new_pts, p_known)
        assert compare_arrays(new_lines, lines)

    def test_three_lines_one_crossing(self):
        p = np.array(
            [[0.0, 0.5, 0.3, 1.0, 0.3, 0.5], [2 / 3, 0.3, 0.3, 2 / 3, 0.5, 0.5]]
        )
        lines = np.array([[0, 3], [2, 5], [1, 4]]).T

        new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)
        p_known = np.hstack((p, np.array([[0.4], [0.4]])))
        lines_known = np.array([[0, 3], [2, 6], [6, 5], [1, 6], [6, 4]]).T
        assert compare_arrays(new_pts, p_known)
        assert compare_arrays(new_lines, lines_known)

    def test_overlapping_lines(self):
        p = np.array([[-0.6, 0.4, 0.4, -0.6, 0.4], [-0.5, -0.5, 0.5, 0.5, 0.0]])
        lines = np.array([[0, 0, 1, 1, 2], [1, 3, 2, 4, 3]])
        new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)

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


class TestIntersectionPolygonsEmbeddedIn3d:
    def test_single_polygon(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d([f_1])
        assert new_pt.size == 0
        assert isect_pt.size == 1
        assert len(isect_pt[0]) == 0
        assert on_bound.size == 1
        assert len(on_bound[0]) == 0

        assert seg_vert.size == 1
        assert len(seg_vert[0]) == 0

    def test_two_intersecting_polygons(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])

        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert compare_arrays(np.sort(isect_pt[0]), [0, 1])
        assert compare_arrays(np.sort(isect_pt[1]), [0, 1])
        assert on_bound.size == 2
        assert not np.all(on_bound[0])
        assert not np.all(on_bound[1])

        known_points = np.array([[0, 0, -0.7], [0, 0, 0.8]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert.size == 2
        assert len(seg_vert[0]) == 2
        for i in range(len(seg_vert[0])):
            assert len(seg_vert[0][i]) == 0

        assert len(seg_vert[1]) == 2
        assert seg_vert[1][0] == (0, True)
        assert seg_vert[1][1] == (2, True)

    def test_three_intersecting_polygons(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        f_3 = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [0, 0, 0, 0]])
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2, f_3]
        )
        assert new_pt.shape[1] == 6
        assert isect_pt.size == 3
        assert len(isect_pt[0]) == 4
        assert len(isect_pt[1]) == 4
        assert len(isect_pt[2]) == 4
        assert on_bound.size == 3
        assert not np.all(on_bound[0])
        assert not np.all(on_bound[1])
        assert not np.all(on_bound[2])

        known_points = np.array(
            [[0, 0, -0.7], [0, 0, 0.8], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        ).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert.size == 3

        counter = np.zeros(3, dtype=int)

        for p in pairs:
            if p[0] == 0 and p[1] == 1:
                assert len(seg_vert[0][counter[0]]) == 0
                assert len(seg_vert[0][counter[0] + 1]) == 0
                assert seg_vert[1][counter[1]] == (0, True)
                assert seg_vert[1][counter[1] + 1] == (2, True)
            elif p[0] == 0 and p[1] == 2:
                assert seg_vert[0][counter[0]] == (1, True)
                assert seg_vert[0][counter[0] + 1] == (3, True)
                assert seg_vert[2][counter[2]] == (1, True)
                assert seg_vert[2][counter[2] + 1] == (3, True)
            else:  # p[0] == 1 and p[1] == 2
                assert seg_vert[1][counter[1]] == (3, True)
                assert seg_vert[1][counter[1] + 1] == (1, True)
                assert seg_vert[2][counter[2]] == (0, True)
                assert seg_vert[2][counter[2] + 1] == (2, True)

            counter[p[0]] += 2
            counter[p[1]] += 2

    def test_three_intersecting_polygons_one_intersected_by_two(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        f_3 = f_2 + np.array([0.5, 0, 0]).reshape((-1, 1))
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2, f_3]
        )
        assert new_pt.shape[1] == 4
        assert isect_pt.size == 3
        assert len(isect_pt[0]) == 4
        assert len(isect_pt[1]) == 2
        assert len(isect_pt[2]) == 2
        assert on_bound.size == 3
        assert not np.all(on_bound[0])
        assert not np.all(on_bound[1])
        assert not np.all(on_bound[2])

        known_points = np.array(
            [[0, 0, -0.7], [0, 0, 0.8], [0.5, 0.0, -0.7], [0.5, 0.0, 0.8]]
        ).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert.size == 3

        counter = np.zeros(3, dtype=int)

        assert len(pairs) == 2

        for p in pairs:
            if p[0] == 0 and p[1] == 1:
                assert len(seg_vert[0][counter[0]]) == 0
                assert len(seg_vert[0][counter[0] + 1]) == 0
                assert seg_vert[1][counter[1]] == (0, True)
                assert seg_vert[1][counter[1] + 1] == (2, True)
            elif p[0] == 0 and p[1] == 2:
                assert len(seg_vert[0][counter[0]]) == 0
                assert len(seg_vert[0][counter[0] + 1]) == 0
                assert seg_vert[2][counter[2]] == (0, True)
                assert seg_vert[2][counter[2] + 1] == (2, True)

            counter[p[0]] += 2
            counter[p[1]] += 2

    def test_three_intersecting_polygons_sharing_segment(self):
        # Polygon along y=0
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        # Polygon along x=y
        f_2 = np.array([[-1, 1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        # Polygon along x=0
        f_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2, f_3]
        )
        assert new_pt.shape[1] == 6
        assert isect_pt.size == 3
        assert len(isect_pt[0]) == 4
        assert len(isect_pt[1]) == 4
        assert len(isect_pt[2]) == 4
        assert on_bound.size == 3
        assert not np.all(on_bound[0])
        assert not np.all(on_bound[1])
        assert not np.all(on_bound[2])

        known_points = np.array(
            [
                [0, 0, -1],
                [0, 0, 1],
                [0.0, 0.0, -1],
                [0.0, 0.0, 1],
                [0.0, 0.0, -1],
                [0.0, 0.0, 1],
            ]
        ).T
        assert compare_arrays(new_pt, known_points)

        assert len(pairs) == 3

        counter = np.zeros(3, dtype=int)

        for p in pairs:
            if p[0] == 0 and p[1] == 1:
                assert seg_vert[0][counter[0]] == (0, True)
                assert seg_vert[0][counter[0] + 1] == (2, True)
                assert seg_vert[1][counter[1]] == (0, True)
                assert seg_vert[1][counter[1] + 1] == (2, True)
            elif p[0] == 0 and p[1] == 2:
                assert seg_vert[0][counter[0]] == (0, True)
                assert seg_vert[0][counter[0] + 1] == (2, True)
                assert seg_vert[2][counter[2]] == (0, True)
                assert seg_vert[2][counter[2] + 1] == (2, True)
            else:  # p[0] == 1 and p[1] == 2
                assert seg_vert[1][counter[1]] == (0, True)
                assert seg_vert[1][counter[1] + 1] == (2, True)
                assert seg_vert[2][counter[2]] == (0, True)
                assert seg_vert[2][counter[2] + 1] == (2, True)

            counter[p[0]] += 2
            counter[p[1]] += 2

    def test_three_intersecting_polygons_split_segment(self):
        """
        Three polygons that all intersect along the same line, but with the intersection
        between two of them forming an extension of the intersection of all three.
        """
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-2, -2, 2, 2]])
        f_3 = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [0, 0, 0, 0]])
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2, f_3]
        )
        assert new_pt.shape[1] == 6
        assert isect_pt.size == 3
        assert len(isect_pt[0]) == 4
        assert len(isect_pt[1]) == 4
        assert len(isect_pt[2]) == 4
        assert on_bound.size == 3
        assert not np.all(on_bound[0])
        assert not np.all(on_bound[1])
        assert not np.all(on_bound[2])

        known_points = np.array(
            [
                [-1, 0, 0],
                [1, 0, 0],
                [-0.5, 0.0, 0],
                [0.5, 0.0, 0],
                [-0.5, 0.0, 0],
                [0.5, 0.0, 0],
            ]
        ).T
        assert compare_arrays(new_pt, known_points)

        counter = np.zeros(3, dtype=int)

        for p in pairs:
            if p[0] == 0 and p[1] == 1:
                assert len(seg_vert[0][counter[0]]) == 0
                assert len(seg_vert[0][counter[0] + 1]) == 0
                assert seg_vert[1][counter[1]] == (1, True)
                assert seg_vert[1][counter[1] + 1] == (3, True)
            elif p[0] == 0 and p[1] == 2:
                assert seg_vert[0][counter[0]] == (1, True)
                assert seg_vert[0][counter[0] + 1] == (3, True)
                assert seg_vert[2][counter[2]] == (1, True)
                assert seg_vert[2][counter[2] + 1] == (3, True)
            else:  # p[0] == 1 and p[1] == 2
                assert seg_vert[1][counter[1]] == (1, True)
                assert seg_vert[1][counter[1] + 1] == (3, True)
                assert len(seg_vert[2][counter[2]]) == 0
                assert len(seg_vert[2][counter[2] + 1]) == 0

            counter[p[0]] += 2
            counter[p[1]] += 2

    def test_two_points_in_plane_of_other_polygon(self):
        """
        Two polygons. One has two (non-consecutive) vertexes in the plane of another
        polygon.
        """
        f_1 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert not np.all(on_bound[0])
        assert not np.all(on_bound[1])

        known_points = np.array([[0, -1, -1], [0, 1, 1]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[0][0] == (0, True)
        assert seg_vert[0][1] == (2, True)

        assert seg_vert[1][0] == (0, False)
        assert seg_vert[1][1] == (2, False)

    def test_two_points_in_plane_of_other_polygon_order_reversed(self):
        """
        Two polygons. One has two (non-consecutive) vertexes in the plane of another
        polygon. Order of polygons is reversed compared to similar test
        """
        f_1 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = intersections.polygons_3d(
            [f_2, f_1]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert not np.all(on_bound[0])
        assert not np.all(on_bound[1])

        known_points = np.array([[0, -1, -1], [0, 1, 1]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[1][0] == (0, True)
        assert seg_vert[1][1] == (2, True)
        assert seg_vert[0][0] == (0, False)
        assert seg_vert[0][1] == (2, False)

    def test_one_point_in_plane_of_other_polygon(self):
        """
        Two polygons. One has one vertexes in the plane of another polygon.
        """
        f_1 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 2, 1]])
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert not np.all(on_bound[0])
        assert not np.all(on_bound[1])

        known_points = np.array([[0, -1, -1], [0, 1, 1]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[0][0] == (0, True)
        assert seg_vert[0][1] == (2, True)

        assert seg_vert[1][0] == (0, False)
        assert seg_vert[1][1] == (1, True)

    def test_one_point_in_plane_of_other_polygon_order_reversed(self):
        """
        Two polygons. One has one vertexes in the plane of another polygon.
        """
        f_1 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 2, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d(
            [f_2, f_1]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert not np.all(on_bound[0])
        assert not np.all(on_bound[1])

        known_points = np.array([[0, -1, -1], [0, 1, 1]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[1][0] == (0, True)
        assert seg_vert[1][1] == (2, True)

        assert seg_vert[0][0] == (0, False)
        assert seg_vert[0][1] == (1, True)

    def test_point_contact_1(self):
        f_1 = np.array(
            [[0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 1.0, 1.0]]
        )
        f_2 = np.array(
            [[0.5, 1.0, 1.0, 0.5], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        )
        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 0
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 0
        assert len(isect_pt[1]) == 0

    def test_L_intersection(self):
        """
        Two polygons, L-intersection.
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 0.7, 0.7, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 1
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[0, 0.3, 0], [0, 0.7, 0]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[0][0] == (0, False)
        assert seg_vert[0][1] == (3, True)

        assert seg_vert[1][0] == (0, False)
        assert seg_vert[1][1] == (1, False)

    def test_L_intersection_reverse_order(self):
        """
        Two polygons, L-intersection.
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 0.7, 0.7, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d(
            [f_2, f_1]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 1
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[0, 0.3, 0], [0, 0.7, 0]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[1][0] == (3, True)
        assert seg_vert[1][1] == (3, True)

        assert seg_vert[0][0] == (0, False)
        assert seg_vert[0][1] == (1, False)

    def test_L_intersection_one_node_common(self):
        """
        Two polygons, L-intersection, one common node.
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 1.0, 1, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 1
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[0, 0.3, 0], [0, 1, 0]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[0][0] == (0, False)
        assert seg_vert[0][1] == (3, False)

        assert seg_vert[1][0] == (0, False)
        assert seg_vert[1][1] == (1, False)

    def test_L_intersection_extends_beyond_each_other(self):
        """
        Two polygons, L-intersection, partly overlapping segments.
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 1.5, 1.5, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 1
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[0, 0.3, 0], [0, 1, 0]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[0][0] == (3, False)
        assert seg_vert[0][1] == (0, False)

        assert seg_vert[1][0] == (0, True)
        assert seg_vert[1][1] == (0, False)

    def test_T_intersection_within_polygon(self):
        """
        Two polygons, T-intersection, segment contained within the other polygon.
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, 0.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 0
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[0.5, 0.5, 0], [0.5, 0.9, 0]]).T
        assert compare_arrays(new_pt, known_points)

        assert len(seg_vert[0][0]) == 0
        assert len(seg_vert[0][1]) == 0

        assert seg_vert[1][0] == (1, False)
        assert seg_vert[1][1] == (2, False)

    def test_T_intersection_one_outside_polygon(self):
        """
        Two polygons, L-intersection, partly overlapping segments.
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, 1.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 0
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[0.5, 0.5, 0], [0.5, 1.0, 0]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[0][0] == (2, True)
        assert len(seg_vert[0][1]) == 0

        assert seg_vert[1][0] == (1, True)
        assert seg_vert[1][1] == (1, False)

    def test_T_intersection_both_on_polygon(self):
        f_1 = np.array([[-2, -2, 2, 2], [-2, -2, 1, 1], [-2, 2, 2, -2]])
        f_2 = np.array(
            [[2.0, 2.0, 2.0, 2.0], [-2.0, -2.0, 2.0, 2.0], [2.0, -2.0, -2.0, 2.0]]
        )

        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 1
        assert np.sum(on_bound[1]) == 0

        known_points = np.array([[2, 1, 2], [2, 1.0, -2]]).T
        assert compare_arrays(new_pt, known_points)

    def test_T_intersection_one_outside_one_on_polygon(self):
        """
        Two polygons, L-intersection, partly overlapping segments
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.0, 0], [0.5, 1.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 0
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[0.5, 0.0, 0], [0.5, 1.0, 0]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[0][0] == (2, True)
        assert seg_vert[0][1] == (0, True)

        assert seg_vert[1][0] == (1, True)
        assert seg_vert[1][1] == (1, False)

    def test_T_intersection_one_outside_one_on_polygon_reverse_order(self):
        """
        Two polygons, L-intersection, partly overlapping segments.
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.0, 0], [0.5, 1.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d(
            [f_2, f_1]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 1
        assert np.sum(on_bound[1]) == 0

        known_points = np.array([[0.5, 0.0, 0], [0.5, 1.0, 0]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[1][0], (0, True)
        assert seg_vert[1][1], (2, True)

        assert seg_vert[0][0], (1, False)
        assert seg_vert[0][1], (1, True)

    def test_T_intersection_both_on_boundary(self):
        """
        Two polygons, L-intersection, partly overlapping segments.
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.0, 0], [0.5, 1.0, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert, _ = intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 0
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[0.5, 0.0, 0], [0.5, 1.0, 0]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[0][0] == (0, True)
        assert seg_vert[0][1] == (2, True)

        assert seg_vert[1][0] == (1, False)
        assert seg_vert[1][1] == (2, False)

    ### Tests involving polygons sharing a the same plane

    def test_same_plane_no_intersections(self):
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[2, 3, 3, 2], [0, 0, 1, 1], [0, 0, 0, 0]])
        new_pt, isect_pt, on_bound, *_ = intersections.polygons_3d([f_1, f_2])
        assert new_pt.shape[1] == 0

    def test_same_plane_intersections(self):
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[2, 2, 0], [0, 2, 1.5], [0, 0, 0]])
        new_pt, isect_pt, on_bound, *_ = intersections.polygons_3d([f_1, f_2])
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 0
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[2.0 / 3, 1, 0], [1, 3.0 / 4, 0]]).T
        assert compare_arrays(new_pt, known_points)

    def test_same_plane_shared_segment_1(self):
        # Shared segment and two vertexes
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[1, 1, 2], [0, 2, 1], [0, 0, 0]])
        new_pt, isect_pt, on_bound, *_ = intersections.polygons_3d([f_1, f_2])
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 1
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[1, 1, 0], [1, 0, 0]]).T
        assert compare_arrays(new_pt, known_points)

    def test_same_plane_shared_segment_2(self):
        # Shared segment and one vertex. Of the common segments, the second polygon
        # has the longest extension.
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[1, 1, 2], [0, 1, 1.5], [0, 0, 0]])
        new_pt, isect_pt, on_bound, *_ = intersections.polygons_3d([f_1, f_2])
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 1
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[1, 1, 0], [1, 0, 0]]).T
        assert compare_arrays(new_pt, known_points)

    def test_same_plane_shared_segment_3(self):
        # Shared segment, no common vertex.
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[1, 1, 2], [0.5, 0.9, 1.5], [0, 0, 0]])
        new_pt, isect_pt, on_bound, *_ = intersections.polygons_3d([f_1, f_2])
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 1
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[1, 0.5, 0], [1, 0.9, 0]]).T
        assert compare_arrays(new_pt, known_points)

    def test_same_plane_shared_segment_4(self):
        # Shared segment and a vertex. The first polygon segment extends beyond the
        # second
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[1, 2, 1], [0, 2, 0.5], [0, 0, 0]])
        new_pt, isect_pt, on_bound, *_ = intersections.polygons_3d([f_1, f_2])
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.sum(on_bound[0]) == 1
        assert np.sum(on_bound[1]) == 1

        known_points = np.array([[1, 0.5, 0], [1, 0, 0]]).T
        assert compare_arrays(new_pt, known_points)

    def test_segment_intersection_identification(self):
        # This configuration turned out to produce a nasty bug

        f_1 = np.array([[1.5, 1.0, 1.0], [0.5, 0.5, 0.5], [0.5, 0.5, 1.0]])

        f_2 = np.array(
            [[0.7, 1.4, 1.4, 0.7], [0.4, 0.4, 1.4, 1.4], [0.6, 0.6, 0.6, 0.6]]
        )

        new_pt, isect_pt, _, _, seg_vert, _ = intersections.polygons_3d([f_1, f_2])

        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2

        known_points = np.array([[1, 0.5, 0.6], [1.4, 0.5, 0.6]]).T
        assert compare_arrays(new_pt, known_points)

        # Find which of the new points have x index 1 and 1.4
        if new_pt[0, 0] == 1:
            x_1_ind = 0
            x_14_ind = 1
        else:
            x_1_ind = 1
            x_14_ind = 0

        # The x1 intersection point should be on the second segment of f_1, and
        # on its boundary
        assert seg_vert[0][x_1_ind][0] == 1
        assert seg_vert[0][x_1_ind][1]

        # The x1 intersection point should be on the final segment of f_1, and
        # in its interior
        assert len(seg_vert[1][x_1_ind]) == 0

        # The x14 intersection point should be on the second segment of f_1, and
        # on its boundary
        assert seg_vert[0][x_14_ind][0] == 2
        assert seg_vert[0][x_14_ind][1]

        # The x1 intersection point should be on the second segment of f_1, and
        # in its interior
        assert seg_vert[1][x_14_ind][0] == 1
        assert seg_vert[1][x_14_ind][1]

    def test_segment_intersection_identification_reverse_order(self):
        # This configuration turned out to produce a nasty bug

        f_1 = np.array([[1.5, 1.0, 1.0], [0.5, 0.5, 0.5], [0.5, 0.5, 1.0]])

        f_2 = np.array(
            [[0.7, 1.4, 1.4, 0.7], [0.4, 0.4, 1.4, 1.4], [0.6, 0.6, 0.6, 0.6]]
        )

        new_pt, isect_pt, _, _, seg_vert, _ = intersections.polygons_3d([f_2, f_1])

        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2

        known_points = np.array([[1, 0.5, 0.6], [1.4, 0.5, 0.6]]).T
        assert compare_arrays(new_pt, known_points)

        # Find which of the new points have x index 1 and 1.4
        if new_pt[0, 0] == 1:
            x_1_ind = 0
            x_14_ind = 1
        else:
            x_1_ind = 1
            x_14_ind = 0

        # The x1 intersection point should be on the second segment of f_1, and
        # on its boundary
        assert seg_vert[1][x_1_ind][0] == 1
        assert seg_vert[1][x_1_ind][1]

        # The x1 intersection point should be on the final segment of f_1, and
        # in its interior
        assert len(seg_vert[0][x_1_ind]) == 0

        # The x14 intersection point should be on the second segment of f_1, and
        # on its boundary
        assert seg_vert[1][x_14_ind][0] == 2
        assert seg_vert[1][x_14_ind][1]

        # The x1 intersection point should be on the second segment of f_1, and
        # in its interior
        assert seg_vert[0][x_14_ind][0] == 1
        assert seg_vert[0][x_14_ind][1]

    def test_single_point_intersection_in_interior(self):
        f_1 = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        f_2 = np.array([[0, 1, 1], [0.5, 0.5, 0.5], [0.5, 0, 1]])

        new_pt, isect_pt, _, _, seg_vert, point_contact = intersections.polygons_3d(
            [f_1, f_2]
        )
        # Also intersect in reverse order
        (
            new_pt2,
            isect_pt,
            _,
            _,
            seg_vert2,
            point_contact2,
        ) = intersections.polygons_3d([f_2, f_1])

        known_point = np.array([[0], [0.5], [0.5]])
        assert compare_arrays(new_pt, known_point)
        assert compare_arrays(new_pt2, known_point)

        for arr in point_contact:
            assert all(arr)
        for arr in point_contact2:
            assert all(arr)

        # Intersection is internal to f_1
        assert len(seg_vert[0][0]) == 0
        # Intersection on f_2 is on the first vertex
        assert seg_vert[1][0][0] == 0
        assert not (seg_vert[1][0][1])

        assert len(seg_vert2[1][0]) == 0
        assert seg_vert2[0][0][0] == 0
        assert not (seg_vert2[0][0][1])

    def test_single_point_intersection_on_boundary(self):
        f_1 = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        f_2 = np.array([[0, 1, 1], [0.5, 0.5, 0.5], [0.0, 0, 1]])

        new_pt, isect_pt, _, _, seg_vert, point_contact = intersections.polygons_3d(
            [f_1, f_2]
        )
        # Also intersect in reverse order
        (
            new_pt2,
            isect_pt,
            _,
            _,
            seg_vert2,
            point_contact2,
        ) = intersections.polygons_3d([f_2, f_1])
        known_point = np.array([[0], [0.5], [0.0]])
        assert compare_arrays(new_pt, known_point)
        assert compare_arrays(new_pt2, known_point)

        for arr in point_contact:
            assert all(arr)
        for arr in point_contact2:
            assert all(arr)

        # Intersection is on the first segment of f_1
        assert seg_vert[0][0][0] == 0
        assert seg_vert[0][0][1]
        # Intersection on f_2 is on the first vertex
        assert seg_vert[1][0][0] == 0
        assert not (seg_vert[1][0][0])

        assert seg_vert2[1][0][0] == 0
        assert seg_vert2[1][0][1]
        assert seg_vert2[0][0][0] == 0
        assert not (seg_vert2[0][0][1])

    def test_single_point_intersection_on_vertex(self):
        f_1 = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        f_2 = np.array([[0, 0.5, 1], [0, 0.5, 1], [0, 0, 1]])

        new_pt, isect_pt, _, _, seg_vert, point_contact = intersections.polygons_3d(
            [f_1, f_2]
        )
        # Also intersect in reverse order
        (
            new_pt2,
            isect_pt,
            _,
            _,
            seg_vert2,
            point_contact2,
        ) = intersections.polygons_3d([f_2, f_1])
        known_point = np.array([[0], [0], [0.0]])
        assert compare_arrays(new_pt, known_point)
        assert compare_arrays(new_pt2, known_point)

        for arr in point_contact:
            assert all(arr)
        for arr in point_contact2:
            assert all(arr)

        # Intersection is on the first segment of f_1s
        assert seg_vert[0][0][0] == 0
        assert not (seg_vert[0][0][1])
        # Intersection on f_2 is on the first vertex
        assert seg_vert[1][0][0] == 0
        assert not (seg_vert[1][0][0])

        assert seg_vert2[0][0][0] == 0
        assert not (seg_vert2[0][0][1])
        # Intersection on f_2 is on the first vertex
        assert seg_vert2[1][0][0] == 0
        assert not (seg_vert2[1][0][0])


class TestPolygonPolyhedronIntersection:
    def setUp(self):
        west = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        east = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
        south = np.array([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1]])
        north = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
        bottom = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        top = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        self.cart_polyhedron = [west, east, south, north, bottom, top]

        south_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 0, 0], [0, 0.5, 1, 1]])
        south_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 0, 0], [0.5, 0, 1, 1]])
        north_w = np.array([[0, 0.5, 0.5, 0], [1, 1, 1, 1], [0, 0.5, 1, 1]])
        north_e = np.array([[0.5, 1, 1, 0.5], [1, 1, 1, 1], [0.5, 0, 1, 1]])
        bottom_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 1], [0, 0.5, 0.5, 0]])
        bottom_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 1, 1], [0.5, 0.0, 0, 0.5]])
        top_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        top_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 1, 1], [1, 1, 1, 1]])
        self.non_convex_polyhedron = [
            west,
            east,
            south_w,
            south_e,
            north_w,
            north_e,
            bottom_w,
            bottom_e,
            top_w,
            top_e,
        ]

        west_bottom = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        west_top = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]])
        east = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
        south = np.array([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1]])
        north = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
        bottom = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        top = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        self.cart_polyhedron_hanging_node = [
            west_bottom,
            west_top,
            east,
            south,
            north,
            bottom,
            top,
        ]

    def test_poly_inside_no_intersections(self):
        self.setUp()
        poly = np.array(
            [[0.2, 0.8, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.cart_polyhedron
        )
        assert len(constrained_poly) == 1
        assert np.allclose(constrained_poly[0], poly)

    def test_poly_outside_no_intersections(self):
        self.setUp()
        poly = np.array(
            [[1.2, 1.8, 1.8, 1.2], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.cart_polyhedron
        )
        assert len(constrained_poly) == 0

    def test_poly_intersects_all_sides(self):
        # Polygon extends outside on all sides
        self.setUp()
        poly = np.array(
            [[-0.2, 1.8, 1.8, -0.2], [0.5, 0.5, 0.5, 0.5], [-0.2, -0.2, 1.8, 1.8]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.cart_polyhedron
        )

        known_constrained_poly = np.array(
            [[0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]]
        )

        assert len(constrained_poly) == 1
        assert compare_arrays(constrained_poly[0], known_constrained_poly)

    def test_poly_intersects_one_side(self):
        self.setUp()
        # Polygon extends outside on all sides, except x=0
        poly = np.array(
            [[0.2, 1.8, 1.8, 0.2], [0.5, 0.5, 0.5, 0.5], [-0.2, -0.2, 1.8, 1.8]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.cart_polyhedron
        )

        known_constrained_poly = np.array(
            [[0.2, 1, 1, 0.2], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]]
        )

        assert len(constrained_poly) == 1
        assert compare_arrays(constrained_poly[0], known_constrained_poly)

    def test_poly_intersects_two_sides(self):
        self.setUp()
        # Polygon extends outside on x-planes, but not on z
        poly = np.array(
            [[-0.2, 1.8, 1.8, -0.2], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.cart_polyhedron
        )

        known_constrained_poly = np.array(
            [[0.0, 1, 1, 0.0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )

        assert len(constrained_poly) == 1
        assert compare_arrays(constrained_poly[0], known_constrained_poly)

    def test_two_poly_one_intersects(self):
        # Combination of two sides
        self.setUp()
        poly_1 = np.array(
            [[-0.2, 1.8, 1.8, -0.2], [0.5, 0.5, 0.5, 0.5], [-0.2, -0.2, 1.8, 1.8]]
        )
        poly_2 = np.array(
            [[0.2, 0.8, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            [poly_1, poly_2], self.cart_polyhedron
        )

        known_constrained_poly_1 = np.array(
            [[0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]]
        )

        assert len(constrained_poly) == 2
        assert compare_arrays(constrained_poly[0], known_constrained_poly_1)
        assert compare_arrays(constrained_poly[1], poly_2)
        assert np.allclose(inds, np.arange(2))

    def test_intersection_on_segment_and_vertex(self):
        # Polygon has one vertex at the polyhedron boundary.
        # Permute the vertexes of the boundary - this used to be an issue.
        self.setUp()

        poly = np.array([[0.5, 1.5, 1.5], [0.5, 0.5, 0.5], [1.0, 0.5, 1.5]])
        poly_2 = np.array([[1.5, 1.5, 0.5], [0.5, 0.5, 0.5], [1.5, 0.5, 1.0]])
        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            [poly], self.cart_polyhedron
        )
        constrained_poly_2, inds = pp.constrain_geometry.polygons_by_polyhedron(
            [poly_2], self.cart_polyhedron
        )

        known_constrained_poly = np.array([[0.5, 1, 1], [0.5, 0.5, 0.5], [1, 0.75, 1]])
        assert compare_arrays(constrained_poly[0], known_constrained_poly)
        assert compare_arrays(constrained_poly_2[0], known_constrained_poly)
        assert np.allclose(inds, np.arange(1))

    def test_one_poly_non_convex_domain(self):
        # Polygon is intersected by polyhedron, cut, but still in one piece.
        self.setUp()
        poly = np.array([[-1, 2, 2, -1], [0.5, 0.5, 0.5, 0.5], [-1, -1, 2, 2]])

        known_constrained_poly = np.array(
            [[0, 1, 1, 0, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1, 0.5]]
        )
        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.non_convex_polyhedron
        )
        assert len(constrained_poly) == 1
        assert compare_arrays(constrained_poly[0], known_constrained_poly)
        assert inds[0] == 0

    def test_poly_split_by_non_convex_domain(self):
        self.setUp()
        # Polygon is split into two pieces. No internal vertexes.
        poly = np.array([[-1, 2, 2, -1], [0.5, 0.5, 0.5, 0.5], [-1, -1, 0.3, 0.3]])

        known_constrained_poly_1 = np.array(
            [[0, 0.3, 0], [0.5, 0.5, 0.5], [0, 0.3, 0.3]]
        )
        known_constrained_poly_2 = np.array(
            [[0.7, 1, 1], [0.5, 0.5, 0.5], [0.3, 0.0, 0.3]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.non_convex_polyhedron
        )

        assert len(constrained_poly) == 2
        assert compare_arrays(
            constrained_poly[0], known_constrained_poly_1
        ) or compare_arrays(constrained_poly[0], known_constrained_poly_2)
        assert compare_arrays(
            constrained_poly[1], known_constrained_poly_1
        ) or compare_arrays(constrained_poly[1], known_constrained_poly_2)
        assert np.all(inds == 0)

    def test_poly_split_by_non_convex_domain_2(self):
        # Polygon is split into two pieces. The polygon does not extend outside the
        # bounding box of the domain, and there are segment crossing the domain bounadry
        # twice.
        self.setUp()
        poly = np.array(
            [[0.1, 0.9, 0.9, 0.1], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]]
        )

        known_constrained_poly_1 = np.array(
            [[0.1, 0.2, 0.4, 0.1], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]]
        )
        known_constrained_poly_2 = np.array(
            [[0.8, 0.9, 0.9, 0.6], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.non_convex_polyhedron
        )

        assert len(constrained_poly) == 2
        assert compare_arrays(
            constrained_poly[0], known_constrained_poly_1
        ) or compare_arrays(constrained_poly[0], known_constrained_poly_2)
        assert compare_arrays(
            constrained_poly[1], known_constrained_poly_1
        ) or compare_arrays(constrained_poly[1], known_constrained_poly_2)
        assert np.all(inds == 0)

    def test_poly_split_by_non_convex_domain_3(self):
        # Polygon is split into two pieces. The polygon partly extends outside the
        # bounding box of the domain; there is one point on the domain boundary.
        # and there are segment crossing the domain bounadry twice.
        self.setUp()
        poly = np.array(
            [[-0.1, 0.9, 0.9, 0.1], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]]
        )

        known_constrained_poly_1 = np.array(
            [
                [0.0, 0.2, 0.4, 0.1, 0.0],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.2, 0.2, 0.4, 0.4, 0.3],
            ]
        )
        known_constrained_poly_2 = np.array(
            [[0.8, 0.9, 0.9, 0.6], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.non_convex_polyhedron
        )

        assert len(constrained_poly) == 2
        assert compare_arrays(
            constrained_poly[0], known_constrained_poly_1
        ) or compare_arrays(constrained_poly[0], known_constrained_poly_2)
        assert compare_arrays(
            constrained_poly[1], known_constrained_poly_1
        ) or compare_arrays(constrained_poly[1], known_constrained_poly_2)
        assert np.all(inds == 0)

    def test_poly_split_by_non_convex_domain_4(self):
        # Polygon is split into two pieces. The polygon partly extends outside the
        # bounding box of the domain; there is one point on the domain boundary.
        # and there are segment crossing the domain bounadry twice.
        self.setUp()
        poly = np.array(
            [[-0.1, 1.1, 1.1, -0.1], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]]
        )

        known_constrained_poly_1 = np.array(
            [[0.0, 0.2, 0.4, 0.0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]]
        )
        known_constrained_poly_2 = np.array(
            [[0.8, 1.0, 1.0, 0.6], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.non_convex_polyhedron
        )

        assert len(constrained_poly) == 2
        assert compare_arrays(
            constrained_poly[0], known_constrained_poly_1
        ) or compare_arrays(constrained_poly[0], known_constrained_poly_2)
        assert compare_arrays(
            constrained_poly[1], known_constrained_poly_1
        ) or compare_arrays(constrained_poly[1], known_constrained_poly_2)
        assert np.all(inds == 0)

    def test_poly_split_by_non_convex_domain_5(self):
        # Polygon is split into two pieces. The polygon partly extends outside the
        # bounding box of the domain; there is one point on the domain boundary.
        # and there are segment crossing the domain bounadry twice.
        self.setUp()
        poly = np.array(
            [[0.0, 1.1, 1.1, 0.0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]]
        )

        known_constrained_poly_1 = np.array(
            [[0.0, 0.2, 0.4, 0.0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]]
        )
        known_constrained_poly_2 = np.array(
            [[0.8, 1.0, 1.0, 0.6], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.non_convex_polyhedron
        )

        assert len(constrained_poly) == 2
        assert compare_arrays(
            constrained_poly[0], known_constrained_poly_1
        ) or compare_arrays(constrained_poly[0], known_constrained_poly_2)
        assert compare_arrays(
            constrained_poly[1], known_constrained_poly_1
        ) or compare_arrays(constrained_poly[1], known_constrained_poly_2)
        assert np.all(inds == 0)

    def test_fully_internal_segments_1(self):
        self.setUp()

        f = np.array(
            [
                [0.5, 0.8, 0.5, 0.2, 0.2],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.5, 1.5, 0.5, 0.3],
            ]
        )
        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            f, self.non_convex_polyhedron
        )

        # TODO: Fix known poitns
        known_constrained_poly = np.array(
            [
                [0.5, 0.6875, 0.8, 0.65, 0.35, 0.2, 0.2, 0.25],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.3125, 0.5, 1.0, 1.0, 0.5, 0.3, 0.25],
            ]
        )

        assert len(constrained_poly) == 1
        assert compare_arrays(constrained_poly[0], known_constrained_poly)
        assert inds[0] == 0

    def test_fully_internal_segments_2(self):
        # Issue that showed up while running the function on a fracture network
        f_1 = np.array([[-1, 2, 2, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        #        f_2 = np.array([[-1, -1, 2, 2], [-1, 1, 1, -1], [0, 0, 0, 0]])

        polyhedron = [
            np.array([[-2, -2, -2, -2], [-2, 2, 2, -2], [-2, -2, 2, 2]]),
            np.array([[2, 2, 2, 2], [-2, 2, 2, -2], [-2, -2, 2, 2]]),
            np.array([[-2, 2, 2, -2], [-2, -2, -2, -2], [-2, -2, 2, 2]]),
            np.array([[-2, 2, 2, -2], [2, 2, 2, 2], [-2, -2, 2, 2]]),
            np.array([[-2, 2, 2, -2], [-2, -2, 2, 2], [-2, -2, -2, -2]]),
            np.array([[-2, 2, 2, -2], [-2, -2, 2, 2], [2, 2, 2, 2]]),
        ]

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            f_1, polyhedron
        )
        assert compare_arrays(f_1, constrained_poly[0])

    def test_fully_internal_segments_3(self):
        # Issue that showed up while running the function on a fracture network
        f_1 = np.array([[-1, 3, 3, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])

        polyhedron = [
            np.array([[-2, -2, -2, -2], [-2, 2, 2, -2], [-2, -2, 2, 2]]),
            np.array([[2, 2, 2, 2], [-2, 2, 2, -2], [-2, -2, 2, 2]]),
            np.array([[-2, 2, 2, -2], [-2, -2, -2, -2], [-2, -2, 2, 2]]),
            np.array([[-2, 2, 2, -2], [2, 2, 2, 2], [-2, -2, 2, 2]]),
            np.array([[-2, 2, 2, -2], [-2, -2, 2, 2], [-2, -2, -2, -2]]),
            np.array([[-2, 2, 2, -2], [-2, -2, 2, 2], [2, 2, 2, 2]]),
        ]

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            f_1, polyhedron
        )
        known_poly = np.array([[-1, 2, 2, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])

        assert compare_arrays(constrained_poly[0], known_poly)

    def test_poly_hits_oposing_boundaries(self):
        # Issue that showed up while running the function on a fracture network
        f_1 = np.array([[-2, 2, 2, -2], [0, 0, 0, 0], [-1, -1, 1, 1]])

        polyhedron = [
            np.array([[-2, -2, -2, -2], [-2, 2, 2, -2], [-2, -2, 2, 2]]),
            np.array([[2, 2, 2, 2], [-2, 2, 2, -2], [-2, -2, 2, 2]]),
            np.array([[-2, 2, 2, -2], [-2, -2, -2, -2], [-2, -2, 2, 2]]),
            np.array([[-2, 2, 2, -2], [2, 2, 2, 2], [-2, -2, 2, 2]]),
            np.array([[-2, 2, 2, -2], [-2, -2, 2, 2], [-2, -2, -2, -2]]),
            np.array([[-2, 2, 2, -2], [-2, -2, 2, 2], [2, 2, 2, 2]]),
        ]

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            f_1, polyhedron
        )
        known_poly = f_1

        assert compare_arrays(constrained_poly[0], known_poly)

    def test_polyhedron_boundaries_in_same_plane_hanging_node(self):
        # Split one of the boundary planes in two, so that the polygon will get a
        # hanging node that must be treated
        f_1 = np.array([[-1, 2, 2, -1], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])

        west_bottom = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        west_top = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]])
        east = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
        south = np.array([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1]])
        north = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
        bottom = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        top = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        polyhedron = [west_bottom, west_top, east, south, north, bottom, top]

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            f_1, polyhedron
        )
        known_poly = np.array(
            [[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )

        assert compare_arrays(constrained_poly[0], known_poly)

    def test_polyhedron_in_same_plane_overlapping_segment(self):
        f_1 = np.array(
            [[0.0, 0.3, 0.3, 0.0], [1.5, 1.5, 1.5, 1.5], [0.8, 0.8, 0.2, 0.2]]
        )
        f_1 = np.array(
            [[0.0, 0.3, 0.3, 0.0], [1.5, 1.5, 1.5, 1.5], [0.2, 0.2, 0.8, 0.8]]
        )

        polyhedron = [
            # The first four surfaces form a pyradim with top at (0.5, 1.5, 0.5) and
            # base in the yz-plane with corners y=(1, 2), z=(0, 1) (then combine)
            np.array([[0.5, 0.0, 0.0], [1.5, 1.0, 2.0], [0.5, 0.0, 0.0]]),
            np.array([[0.5, 0.0, 0.0], [1.5, 2.0, 2.0], [0.5, 0.0, 1.0]]),
            np.array([[0.5, 0.0, 0.0], [1.5, 2.0, 1.0], [0.5, 1.0, 1.0]]),
            np.array([[0.5, 0.0, 0.0], [1.5, 1.0, 1.0], [0.5, 1.0, 0.0]]),
            # The last surfaces cut the base in two
            np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 1.0]]),
        ]

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            f_1, polyhedron
        )
        known_poly = np.array(
            [
                [0.0, 0.2, 0.3, 0.3, 0.2, 0.0],
                [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                [0.2, 0.2, 0.3, 0.7, 0.8, 0.8],
            ]
        )

        assert compare_arrays(constrained_poly[0], known_poly)

    def test_point_intersection_fully_inside_box(self):
        self.setUp()

        # f_1 has one intersection along a surface, point intersection in the third vertex.
        f_1 = np.array([[0, 0, 1], [0.5, 0.5, 0.5], [0.2, 0.8, 0]])
        # f_2 has three point intersections on different vertexes.
        f_2 = np.array([[0, 0.5, 1], [0.5, 0.0, 0.5], [0.2, 0.8, 1]])

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            [f_1, f_2], self.cart_polyhedron
        )

        assert compare_arrays(constrained_poly[0], f_1)
        assert compare_arrays(constrained_poly[1], f_2)

    def test_point_intersection_fully_outside_box(self):
        self.setUp()

        # f_1 has one point intersection outside, rest is outside
        f_1 = np.array([[-1, -1, 0], [0.5, 0.5, 0.5], [0.2, 0.8, 0]])

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            [f_1], self.cart_polyhedron
        )

        assert len(constrained_poly) == 0

    def test_point_intersection_crossing_on_other_side(self):
        self.setUp()

        # Constrained polygon formed by point contact and two standard intersections.
        f_1 = np.array([[-1, -1, 1], [0.5, 0.5, 0.5], [0.0, 1, 0.5]])

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            [f_1], self.cart_polyhedron
        )

        known_poly = np.array([[0, 1, 0], [0.5, 0.5, 0.5], [0.25, 0.5, 0.75]])

        assert compare_arrays(constrained_poly[0], known_poly)

    def test_point_intersection_rest_of_polygon_outside(self):
        # This used to be a problem.
        polyhedron = [
            np.array([[0.1, 0.0, 0.0], [0.5, 0.4, 0.6], [0.1, 0.0, 0.0]]),
            np.array([[0.1, 0.0, 0.0], [0.5, 0.6, 0.6], [0.1, 0.0, 0.2]]),
            np.array([[0.1, 0.0, 0.0], [0.5, 0.6, 0.4], [0.1, 0.2, 0.2]]),
            np.array([[0.1, 0.0, 0.0], [0.5, 0.4, 0.4], [0.1, 0.2, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.4, 0.6, 0.6], [0.0, 0.0, 0.2]]),
            np.array([[0.0, 0.0, 0.0], [0.4, 0.6, 0.4], [0.0, 0.2, 0.2]]),
        ]

        poly = np.array(
            [[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.9, 0.9], [0.0, 0.9, 0.9, 0.0]]
        )

        _, inds = pp.constrain_geometry.polygons_by_polyhedron([poly], polyhedron)
        assert inds.size == 0


# ---------- Testing triangulations ----------


def test_triangulations_identical_triangles():
    p = np.array([[0, 1, 0], [0, 0, 1]])
    t = np.array([[0, 1, 2]]).T

    triangulation = intersections.triangulations(p, p, t, t)
    assert len(triangulation) == 1
    assert triangulation[0][0] == 0
    assert triangulation[0][1] == 0
    assert triangulation[0][2] == 0.5


def test_triangulations_two_and_one():
    p1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    t1 = np.array([[0, 1, 3], [1, 2, 3]]).T

    p2 = np.array([[0, 1, 0], [0, 1, 1]])
    t2 = np.array([[0, 1, 2]]).T

    triangulation = intersections.triangulations(p1, p2, t1, t2)
    assert len(triangulation) == 2
    assert triangulation[0][0] == 0
    assert triangulation[0][1] == 0
    assert triangulation[0][2] == 0.25
    assert triangulation[1][0] == 1
    assert triangulation[1][1] == 0
    assert triangulation[1][2] == 0.25


def test_triangulations_one_and_two():
    p1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    t1 = np.array([[0, 1, 3], [1, 2, 3]]).T

    p2 = np.array([[0, 1, 0], [0, 1, 1]])
    t2 = np.array([[0, 1, 2]]).T

    triangulation = intersections.triangulations(p2, p1, t2, t1)
    assert len(triangulation) == 2
    assert triangulation[0][0] == 0
    assert triangulation[0][1] == 0
    assert triangulation[0][2] == 0.25
    assert triangulation[1][1] == 1
    assert triangulation[1][0] == 0
    assert triangulation[1][2] == 0.25


# ---------- Testing _identify_overlapping_intervals ----------


def check_pairs_contain(pairs: np.ndarray, a: np.ndarray) -> bool:
    for pi in range(pairs.shape[1]):
        if a[0] == pairs[0, pi] and a[1] == pairs[1, pi]:
            return True
    return False


@pytest.mark.parametrize(
    "points",
    [
        ([0, 2], [1, 3]),  # No intersection
        ([0, 1], [0, 2]),  # One line is a point, no intersection
    ],
)
def test_identify_overlapping_intervals_no_intersection(points):
    x_min = np.array(points[0])
    x_max = np.array(points[1])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)
    assert pairs.size == 0


@pytest.mark.parametrize(
    "points",
    [
        ([0, 1], [2, 3]),  # Intersection two lines
        ([1, 0], [3, 2]),  # Two lines switched order
        ([0, 0], [3, 2]),  # Two lines same start
        ([0, 1], [3, 3]),  # Two lines same end
        ([0, 0], [3, 3]),  # Two lines same start same end
        ([1, 0], [1, 2]),  # Two lines, one is a point, intersection
        ([1, 0, 3], [2, 2, 4]),  # Three lines, two intersections
    ],
)
def test_identify_overlapping_intervals_intersection(points):
    x_min = np.array(points[0])
    x_max = np.array(points[1])
    pairs = intersections._identify_overlapping_intervals(x_min, x_max)
    assert pairs.size == 2
    assert pairs[0, 0] == 0
    assert pairs[1, 0] == 1


def test_identify_overlapping_intervals_three_lines_all_intersect():
    x_min = np.array([1, 0, 1])
    x_max = np.array([2, 2, 3])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)

    assert pairs.size == 6
    assert check_pairs_contain(pairs, [0, 1])
    assert check_pairs_contain(pairs, [0, 2])
    assert check_pairs_contain(pairs, [1, 2])


def test_identify_overlapping_intervals_three_lines_pairs_intersect():
    x_min = np.array([0, 0, 2])
    x_max = np.array([1, 3, 3])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)

    assert pairs.size == 4
    assert check_pairs_contain(pairs, [0, 1])
    assert check_pairs_contain(pairs, [1, 2])


# ---------- Testing _intersect_pairs, _identify_overlapping_rectangles ----------


@pytest.mark.parametrize(
    "xmin_xmax_ymin_ymax",
    [
        # Use same coordinates for x and y, that is, the fractures are on the line x=y.
        ([0, 2], [1, 3], [0, 2], [1, 3]),
        # The points are overlapping on the x-axis but not on the y-axis.
        ([0, 0], [2, 2], [0, 5], [2, 7]),
        # The points are overlapping on the x-axis and the y-axis.
        ([0, 0], [2, 2], [0, 1], [2, 3]),
        # Lines in square, all should overlap
        ([0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 0], [0, 1, 1, 1]),
    ],
)
def test_identify_overlapping_rectangles(xmin_xmax_ymin_ymax):
    """We run both 1d search + intersection, and 2d search. They should be equivalent.

    Note: The tests are only between the bounding boxes of the fractures, not the
        fractures themselves.

    """
    x_min = np.array(xmin_xmax_ymin_ymax[0])
    x_max = np.array(xmin_xmax_ymin_ymax[1])

    y_min = np.array(xmin_xmax_ymin_ymax[2])
    y_max = np.array(xmin_xmax_ymin_ymax[3])

    x_pairs = intersections._identify_overlapping_intervals(x_min, x_max)
    y_pairs = intersections._identify_overlapping_intervals(y_min, y_max)
    pairs_expected = intersections._intersect_pairs(x_pairs, y_pairs)

    combined_pairs = intersections._identify_overlapping_rectangles(
        x_min, x_max, y_min, y_max
    )
    assert combined_pairs.size == pairs_expected.size
    assert np.allclose(pairs_expected, combined_pairs)


# ---------- Testing split_intersecting_segments_2d ----------
# Tests for function used to remove intersections between 1d fractures.


@pytest.mark.parametrize(
    "points_lines",
    [
        # Two lines no crossing.
        (
            # points
            [[-1, 1, 0, 0], [0, 0, -1, 1]],
            # lines
            [[0, 1], [2, 3]],
        ),
        # Three lines no crossing.
        (
            # points
            [
                [0.0, 0.0, 0.3, 1.0, 1.0, 0.5],
                [2 / 3, 1 / 0.7, 0.3, 2 / 3, 1 / 0.7, 0.5],
            ],
            # lines
            [[0, 1, 2], [3, 4, 5]],
        ),
    ],
)
def test_split_intersecting_segments_2d_no_crossing(points_lines):
    points = np.array(points_lines[0])
    lines = np.array(points_lines[1])
    new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(points, lines)
    assert np.allclose(new_pts, points)
    assert np.allclose(new_lines, lines)


@pytest.mark.parametrize(
    "points_lines_expected",
    [
        # Two lines crossing origin.
        (
            # points
            [[-1, 1, 0, 0], [0, 0, -1, 1]],
            # lines
            [[0, 2], [1, 3], [1, 2], [3, 4]],
            # expected_points (to be appended)
            [[0], [0]],
            # expected_lines
            [[0, 4, 2, 4], [4, 1, 4, 3], [1, 1, 2, 2], [3, 3, 4, 4]],
        ),
        # Three lines one crossing.
        (
            # points
            [[0.0, 0.5, 0.3, 1.0, 0.3, 0.5], [2 / 3, 0.3, 0.3, 2 / 3, 0.5, 0.5]],
            # lines
            [[0, 2, 1], [3, 5, 4]],
            # expected_points (to be appended)
            [[0.4], [0.4]],
            # expected_lines
            [[0, 2, 6, 1, 6], [3, 6, 5, 6, 4]],
        ),
        # Overlapping lines
        (
            # points
            [[-0.6, 0.4, 0.4, -0.6, 0.4], [-0.5, -0.5, 0.5, 0.5, 0.0]],
            # lines
            [[0, 0, 1, 1, 2], [1, 3, 2, 4, 3]],
            # expected_points (to be appended)
            [[], []],
            # expected_lines
            [[0, 0, 1, 2, 2], [1, 3, 4, 4, 3]],
        ),
    ],
)
def test_split_intersecting_segments_2d_crossing(points_lines_expected):
    points = np.array(points_lines_expected[0])
    lines = np.array(points_lines_expected[1])
    expected_points = np.hstack([points, np.array(points_lines_expected[2])])
    expected_lines = np.array(points_lines_expected[3])

    new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(points, lines)

    assert np.allclose(new_pts, expected_points)
    assert compare_arrays(new_lines, expected_lines)
