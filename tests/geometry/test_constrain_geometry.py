import pytest

import numpy as np

import porepy as pp
from tests.test_utils import compare_arrays


class BasicTest:
    """
    Various tests of intersect_polygon_lines.

    """

    def test_convex_polygon(self):
        # convex polygon
        polygon = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        pts = np.array(
            [
                [0.0, 1.0, 0.0, 2.0, 0.5, 0.0, -0.5, 0.3],
                [0.0, 1.0, 0.0, 2.0, 1.0, 0.0, -0.5, 0.6],
            ]
        )
        lines = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])

        new_pts, new_lines, lines_kept = pp.constrain_geometry.lines_by_polygon(
            polygon, pts, lines
        )

        pts_known = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.3],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 3 / 16, 0.6],
            ]
        )
        lines_known = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])
        kept_known = np.arange(lines_known.shape[1])

        assert compare_arrays(new_pts, pts_known)
        assert compare_arrays(new_lines, lines_known)

        assert compare_arrays(lines_kept, kept_known)

    def test_convex_polygon_line_outside(self):
        # convex polygon
        polygon = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])

        # The last line is completely outside, and will be kicked out
        pts = np.array(
            [
                [0.0, 1.0, 0.0, 2.0, 0.5, 0.0, -0.5, 0.3, -1, 0.0],
                [0.0, 1.0, 0.0, 2.0, 1.0, 0.0, -0.5, 0.6, -1, 0.0],
            ]
        )
        lines = np.array([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])

        new_pts, new_lines, lines_kept = pp.constrain_geometry.lines_by_polygon(
            polygon, pts, lines
        )

        pts_known = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.3],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 3 / 16, 0.6],
            ]
        )
        lines_known = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])
        kept_known = np.array([0, 1, 2, 3])

        assert compare_arrays(new_pts, pts_known)
        assert compare_arrays(new_lines, lines_known)
        assert compare_arrays(lines_kept, kept_known)

    def test_non_convex_polygon(self):
        # non-convex polygon
        polygon = np.array(
            [[0.0, 0.5, 0.75, 1.0, 1.5, 1.5, 0], [0.0, 0.0, 0.25, 0.0, 0, 1, 1]]
        )
        pts = np.array(
            [
                [0.0, 1.0, 0.0, 2.0, 0.5, 0.0, -0.5, 0.3, -1, 0.0, 0.0, 2.0, -0.1, 1.1],
                [0.0, 1.0, 0.0, 2.0, 1.0, 0.0, -0.5, 0.6, -1, 0.0, 0.2, 0.2, 0.0, 0.0],
            ]
        )
        lines = np.array([[0, 2, 4, 6, 8, 10, 12], [1, 3, 5, 7, 9, 11, 13]])

        new_pts, new_lines, lines_kept = pp.constrain_geometry.lines_by_polygon(
            polygon, pts, lines
        )
        pts_known = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.7, 0.8, 1.5],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 3 / 16, 0.6, 0.2, 0.2, 0.2, 0.2],
            ]
        )
        lines_known = np.array([[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]])

        kept_known = np.array([0, 1, 2, 3, 5, 5])

        assert compare_arrays(new_pts, pts_known)
        assert compare_arrays(new_lines, lines_known)
        assert compare_arrays(lines_kept, kept_known)


class TestIntersectionPolygonsEmbeddedIn3d:
    def test_single_fracture(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d([f_1])
        assert new_pt.size == 0
        assert isect_pt.size == 1
        assert len(isect_pt[0]) == 0
        assert on_bound.size == 1
        assert len(on_bound[0]) == 0

        assert seg_vert.size == 1
        assert len(seg_vert[0]) == 0

    def test_two_intersecting_fractures(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])

        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert compare_arrays(np.sort(isect_pt[0]), [0, 1])
        assert compare_arrays(np.sort(isect_pt[1]), [0, 1])
        assert on_bound.size == 2
        assert np.all(on_bound[0]) == False
        assert np.all(on_bound[1]) == False

        known_points = np.array([[0, 0, -0.7], [0, 0, 0.8]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert.size == 2
        assert len(seg_vert[0]) == 2
        for i in range(len(seg_vert[0])):
            assert len(seg_vert[0][i]) == 0

        assert len(seg_vert[1]) == 2
        assert seg_vert[1][0] == (0, True)
        assert seg_vert[1][1] == (2, True)

    def test_three_intersecting_fractures(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        f_3 = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [0, 0, 0, 0]])
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = pp.intersections.polygons_3d(
            [f_1, f_2, f_3]
        )
        assert new_pt.shape[1] == 6
        assert isect_pt.size == 3
        assert len(isect_pt[0]) == 4
        assert len(isect_pt[1]) == 4
        assert len(isect_pt[2]) == 4
        assert on_bound.size == 3
        assert np.all(on_bound[0]) == False
        assert np.all(on_bound[1]) == False
        assert np.all(on_bound[2]) == False

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

    def test_three_intersecting_fractures_one_intersected_by_two(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        f_3 = f_2 + np.array([0.5, 0, 0]).reshape((-1, 1))
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = pp.intersections.polygons_3d(
            [f_1, f_2, f_3]
        )
        assert new_pt.shape[1] == 4
        assert isect_pt.size == 3
        assert len(isect_pt[0]) == 4
        assert len(isect_pt[1]) == 2
        assert len(isect_pt[2]) == 2
        assert on_bound.size == 3
        assert np.all(on_bound[0]) == False
        assert np.all(on_bound[1]) == False
        assert np.all(on_bound[2]) == False

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

    def test_three_intersecting_fractures_sharing_segment(self):
        # Fracture along y=0
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        # fracture along x=y
        f_2 = np.array([[-1, 1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        # fracture along x=0
        f_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = pp.intersections.polygons_3d(
            [f_1, f_2, f_3]
        )
        assert new_pt.shape[1] == 6
        assert isect_pt.size == 3
        assert len(isect_pt[0]) == 4
        assert len(isect_pt[1]) == 4
        assert len(isect_pt[2]) == 4
        assert on_bound.size == 3
        assert np.all(on_bound[0]) == False
        assert np.all(on_bound[1]) == False
        assert np.all(on_bound[2]) == False

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

    def test_three_intersecting_fractures_split_segment(self):
        """
        Three fractures that all intersect along the same line, but with the
        intersection between two of them forming an extension of the intersection
        of all three.
        """
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-2, -2, 2, 2]])
        f_3 = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [0, 0, 0, 0]])
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = pp.intersections.polygons_3d(
            [f_1, f_2, f_3]
        )
        assert new_pt.shape[1] == 6
        assert isect_pt.size == 3
        assert len(isect_pt[0]) == 4
        assert len(isect_pt[1]) == 4
        assert len(isect_pt[2]) == 4
        assert on_bound.size == 3
        assert np.all(on_bound[0]) == False
        assert np.all(on_bound[1]) == False
        assert np.all(on_bound[2]) == False

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

    def test_two_points_in_plane_of_other_fracture(self):
        """
        Two fractures. One has two (non-consecutive) vertexes in the plane
        of another fracture
        """
        f_1 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.all(on_bound[0]) == False
        assert np.all(on_bound[1]) == False

        known_points = np.array([[0, -1, -1], [0, 1, 1]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[0][0] == (0, True)
        assert seg_vert[0][1] == (2, True)

        assert seg_vert[1][0] == (0, False)
        assert seg_vert[1][1] == (2, False)

    def test_two_points_in_plane_of_other_fracture_order_reversed(self):
        """
        Two fractures. One has two (non-consecutive) vertexes in the plane
        of another fracture. Order of polygons is reversed compared to similar test
        """
        f_1 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = pp.intersections.polygons_3d(
            [f_2, f_1]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.all(on_bound[0]) == False
        assert np.all(on_bound[1]) == False

        known_points = np.array([[0, -1, -1], [0, 1, 1]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[1][0] == (0, True)
        assert seg_vert[1][1] == (2, True)
        assert seg_vert[0][0] == (0, False)
        assert seg_vert[0][1] == (2, False)

    def test_one_point_in_plane_of_other_fracture(self):
        """
        Two fractures. One has one vertexes in the plane
        of another fracture
        """
        f_1 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 2, 1]])
        new_pt, isect_pt, on_bound, pairs, seg_vert, _ = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.all(on_bound[0]) == False
        assert np.all(on_bound[1]) == False

        known_points = np.array([[0, -1, -1], [0, 1, 1]]).T
        assert compare_arrays(new_pt, known_points)

        assert seg_vert[0][0] == (0, True)
        assert seg_vert[0][1] == (2, True)

        assert seg_vert[1][0] == (0, False)
        assert seg_vert[1][1] == (1, True)

    def test_one_point_in_plane_of_other_fracture_order_reversed(self):
        """
        Two fractures. One has one vertexes in the plane
        of another fracture
        """
        f_1 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 2, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d(
            [f_2, f_1]
        )
        assert new_pt.shape[1] == 2
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 2
        assert len(isect_pt[1]) == 2
        assert on_bound.size == 2
        assert np.all(on_bound[0]) == False
        assert np.all(on_bound[1]) == False

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
        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        assert new_pt.shape[1] == 0
        assert isect_pt.size == 2
        assert len(isect_pt[0]) == 0
        assert len(isect_pt[1]) == 0

    def test_L_intersection(self):
        """
        Two fractures, L-intersection.
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 0.7, 0.7, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d(
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
        Two fractures, L-intersection.
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 0.7, 0.7, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d(
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
        Two fractures, L-intersection, one common node.
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 1.0, 1, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d(
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
        Two fractures, L-intersection, partly overlapping segments
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 1.5, 1.5, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d(
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
        Two fractures, T-intersection, segment contained within the other polygon.
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, 0.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d(
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
        Two fractures, L-intersection, partly overlapping segments
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, 1.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d(
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

        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d(
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
        Two fractures, L-intersection, partly overlapping segments
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.0, 0], [0.5, 1.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d(
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
        Two fractures, L-intersection, partly overlapping segments
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.0, 0], [0.5, 1.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d(
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

        assert (seg_vert[1][0], (0, True))
        assert (seg_vert[1][1], (2, True))

        assert (seg_vert[0][0], (1, False))
        assert (seg_vert[0][1], (1, True))

    def test_T_intersection_both_on_boundary(self):
        """
        Two fractures, L-intersection, partly overlapping segments
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.0, 0], [0.5, 1.0, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert, _ = pp.intersections.polygons_3d(
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
        new_pt, isect_pt, on_bound, *_ = pp.intersections.polygons_3d([f_1, f_2])
        assert new_pt.shape[1] == 0

    def test_same_plane_intersections(self):
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[2, 2, 0], [0, 2, 1.5], [0, 0, 0]])
        new_pt, isect_pt, on_bound, *_ = pp.intersections.polygons_3d([f_1, f_2])
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
        new_pt, isect_pt, on_bound, *_ = pp.intersections.polygons_3d([f_1, f_2])
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
        new_pt, isect_pt, on_bound, *_ = pp.intersections.polygons_3d([f_1, f_2])
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
        new_pt, isect_pt, on_bound, *_ = pp.intersections.polygons_3d([f_1, f_2])
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
        new_pt, isect_pt, on_bound, *_ = pp.intersections.polygons_3d([f_1, f_2])
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

        new_pt, isect_pt, _, _, seg_vert, _ = pp.intersections.polygons_3d([f_1, f_2])

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

        new_pt, isect_pt, _, _, seg_vert, _ = pp.intersections.polygons_3d([f_2, f_1])

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

        new_pt, isect_pt, _, _, seg_vert, point_contact = pp.intersections.polygons_3d(
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
        ) = pp.intersections.polygons_3d([f_2, f_1])

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

        new_pt, isect_pt, _, _, seg_vert, point_contact = pp.intersections.polygons_3d(
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
        ) = pp.intersections.polygons_3d([f_2, f_1])
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

        new_pt, isect_pt, _, _, seg_vert, point_contact = pp.intersections.polygons_3d(
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
        ) = pp.intersections.polygons_3d([f_2, f_1])
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
