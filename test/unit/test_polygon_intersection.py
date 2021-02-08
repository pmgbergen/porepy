import unittest
from test import test_utils

import numpy as np

import porepy as pp


class BasicTest(unittest.TestCase):
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

        self.assertTrue(np.allclose(new_pts, pts_known))
        self.assertTrue(np.allclose(new_lines, lines_known))

        self.assertTrue(np.allclose(lines_kept, kept_known))

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

        self.assertTrue(np.allclose(new_pts, pts_known))
        self.assertTrue(np.allclose(new_lines, lines_known))

        self.assertTrue(np.allclose(lines_kept, kept_known))

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

        self.assertTrue(np.allclose(new_pts, pts_known))
        self.assertTrue(np.allclose(new_lines, lines_known))
        self.assertTrue(np.allclose(lines_kept, kept_known))


class TestIntersectionPolygonsEmbeddedIn3d(unittest.TestCase):
    def test_single_fracture(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d([f_1])
        self.assertTrue(new_pt.size == 0)
        self.assertTrue(isect_pt.size == 1)
        self.assertTrue(len(isect_pt[0]) == 0)
        self.assertTrue(on_bound.size == 1)
        self.assertTrue(len(on_bound[0]) == 0)

        self.assertTrue(seg_vert.size == 1)
        self.assertTrue(len(seg_vert[0]) == 0)

    def test_two_intersecting_fractures(self):

        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])

        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(np.allclose(np.sort(isect_pt[0]), [0, 1]))
        self.assertTrue(np.allclose(np.sort(isect_pt[1]), [0, 1]))
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.all(on_bound[0]) == False)
        self.assertTrue(np.all(on_bound[1]) == False)

        known_points = np.array([[0, 0, -0.7], [0, 0, 0.8]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertTrue(seg_vert.size == 2)
        self.assertTrue(len(seg_vert[0]) == 2)
        for i in range(len(seg_vert[0])):
            self.assertTrue(len(seg_vert[0][i]) == 0)

        self.assertTrue(len(seg_vert[1]) == 2)
        self.assertEqual(seg_vert[1][0], (0, True))
        self.assertEqual(seg_vert[1][1], (2, True))

    def test_three_intersecting_fractures(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        f_3 = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [0, 0, 0, 0]])
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2, f_3]
        )
        self.assertTrue(new_pt.shape[1] == 6)
        self.assertTrue(isect_pt.size == 3)
        self.assertTrue(len(isect_pt[0]) == 4)
        self.assertTrue(len(isect_pt[1]) == 4)
        self.assertTrue(len(isect_pt[2]) == 4)
        self.assertTrue(on_bound.size == 3)
        self.assertTrue(np.all(on_bound[0]) == False)
        self.assertTrue(np.all(on_bound[1]) == False)
        self.assertTrue(np.all(on_bound[2]) == False)

        known_points = np.array(
            [[0, 0, -0.7], [0, 0, 0.8], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        ).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertTrue(seg_vert.size == 3)

        counter = np.zeros(3, dtype=int)

        for p in pairs:
            if p[0] == 0 and p[1] == 1:
                self.assertTrue(len(seg_vert[0][counter[0]]) == 0)
                self.assertTrue(len(seg_vert[0][counter[0] + 1]) == 0)
                self.assertEqual(seg_vert[1][counter[1]], (0, True))
                self.assertEqual(seg_vert[1][counter[1] + 1], (2, True))
            elif p[0] == 0 and p[1] == 2:
                self.assertEqual(seg_vert[0][counter[0]], (1, True))
                self.assertEqual(seg_vert[0][counter[0] + 1], (3, True))
                self.assertEqual(seg_vert[2][counter[2]], (1, True))
                self.assertEqual(seg_vert[2][counter[2] + 1], (3, True))
            else:  # p[0] == 1 and p[1] == 2
                self.assertEqual(seg_vert[1][counter[1]], (3, True))
                self.assertEqual(seg_vert[1][counter[1] + 1], (1, True))
                self.assertEqual(seg_vert[2][counter[2]], (0, True))
                self.assertEqual(seg_vert[2][counter[2] + 1], (2, True))

            counter[p[0]] += 2
            counter[p[1]] += 2

    def test_three_intersecting_fractures_one_intersected_by_two(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        f_3 = f_2 + np.array([0.5, 0, 0]).reshape((-1, 1))
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2, f_3]
        )
        self.assertTrue(new_pt.shape[1] == 4)
        self.assertTrue(isect_pt.size == 3)
        self.assertTrue(len(isect_pt[0]) == 4)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(len(isect_pt[2]) == 2)
        self.assertTrue(on_bound.size == 3)
        self.assertTrue(np.all(on_bound[0]) == False)
        self.assertTrue(np.all(on_bound[1]) == False)
        self.assertTrue(np.all(on_bound[2]) == False)

        known_points = np.array(
            [[0, 0, -0.7], [0, 0, 0.8], [0.5, 0.0, -0.7], [0.5, 0.0, 0.8]]
        ).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertTrue(seg_vert.size == 3)

        counter = np.zeros(3, dtype=int)

        self.assertTrue(len(pairs) == 2)

        for p in pairs:
            if p[0] == 0 and p[1] == 1:
                self.assertTrue(len(seg_vert[0][counter[0]]) == 0)
                self.assertTrue(len(seg_vert[0][counter[0] + 1]) == 0)
                self.assertEqual(seg_vert[1][counter[1]], (0, True))
                self.assertEqual(seg_vert[1][counter[1] + 1], (2, True))
            elif p[0] == 0 and p[1] == 2:
                self.assertTrue(len(seg_vert[0][counter[0]]) == 0)
                self.assertTrue(len(seg_vert[0][counter[0] + 1]) == 0)
                self.assertEqual(seg_vert[2][counter[2]], (0, True))
                self.assertEqual(seg_vert[2][counter[2] + 1], (2, True))

            counter[p[0]] += 2
            counter[p[1]] += 2

    def test_three_intersecting_fractures_sharing_segment(self):
        # Fracture along y=0
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        # fracture along x=y
        f_2 = np.array([[-1, 1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        # fracture along x=0
        f_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2, f_3]
        )
        self.assertTrue(new_pt.shape[1] == 6)
        self.assertTrue(isect_pt.size == 3)
        self.assertTrue(len(isect_pt[0]) == 4)
        self.assertTrue(len(isect_pt[1]) == 4)
        self.assertTrue(len(isect_pt[2]) == 4)
        self.assertTrue(on_bound.size == 3)
        self.assertTrue(np.all(on_bound[0]) == False)
        self.assertTrue(np.all(on_bound[1]) == False)
        self.assertTrue(np.all(on_bound[2]) == False)

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
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertTrue(len(pairs) == 3)

        counter = np.zeros(3, dtype=int)

        for p in pairs:
            if p[0] == 0 and p[1] == 1:
                self.assertEqual(seg_vert[0][counter[0]], (0, True))
                self.assertEqual(seg_vert[0][counter[0] + 1], (2, True))
                self.assertEqual(seg_vert[1][counter[1]], (0, True))
                self.assertEqual(seg_vert[1][counter[1] + 1], (2, True))
            elif p[0] == 0 and p[1] == 2:
                self.assertEqual(seg_vert[0][counter[0]], (0, True))
                self.assertEqual(seg_vert[0][counter[0] + 1], (2, True))
                self.assertEqual(seg_vert[2][counter[2]], (0, True))
                self.assertEqual(seg_vert[2][counter[2] + 1], (2, True))
            else:  # p[0] == 1 and p[1] == 2
                self.assertEqual(seg_vert[1][counter[1]], (0, True))
                self.assertEqual(seg_vert[1][counter[1] + 1], (2, True))
                self.assertEqual(seg_vert[2][counter[2]], (0, True))
                self.assertEqual(seg_vert[2][counter[2] + 1], (2, True))

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
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2, f_3]
        )
        self.assertTrue(new_pt.shape[1] == 6)
        self.assertTrue(isect_pt.size == 3)
        self.assertTrue(len(isect_pt[0]) == 4)
        self.assertTrue(len(isect_pt[1]) == 4)
        self.assertTrue(len(isect_pt[2]) == 4)
        self.assertTrue(on_bound.size == 3)
        self.assertTrue(np.all(on_bound[0]) == False)
        self.assertTrue(np.all(on_bound[1]) == False)
        self.assertTrue(np.all(on_bound[2]) == False)

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
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        counter = np.zeros(3, dtype=int)

        for p in pairs:
            if p[0] == 0 and p[1] == 1:
                self.assertTrue(len(seg_vert[0][counter[0]]) == 0)
                self.assertTrue(len(seg_vert[0][counter[0] + 1]) == 0)
                self.assertEqual(seg_vert[1][counter[1]], (1, True))
                self.assertEqual(seg_vert[1][counter[1] + 1], (3, True))
            elif p[0] == 0 and p[1] == 2:
                self.assertEqual(seg_vert[0][counter[0]], (1, True))
                self.assertEqual(seg_vert[0][counter[0] + 1], (3, True))
                self.assertEqual(seg_vert[2][counter[2]], (1, True))
                self.assertEqual(seg_vert[2][counter[2] + 1], (3, True))
            else:  # p[0] == 1 and p[1] == 2
                self.assertEqual(seg_vert[1][counter[1]], (1, True))
                self.assertEqual(seg_vert[1][counter[1] + 1], (3, True))
                self.assertTrue(len(seg_vert[2][counter[2]]) == 0)
                self.assertTrue(len(seg_vert[2][counter[2] + 1]) == 0)

            counter[p[0]] += 2
            counter[p[1]] += 2

    def test_two_points_in_plane_of_other_fracture(self):
        """
        Two fractures. One has two (non-consecutive) vertexes in the plane
        of another fracture
        """
        f_1 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.all(on_bound[0]) == False)
        self.assertTrue(np.all(on_bound[1]) == False)

        known_points = np.array([[0, -1, -1], [0, 1, 1]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(seg_vert[0][0], (0, True))
        self.assertEqual(seg_vert[0][1], (2, True))

        self.assertEqual(seg_vert[1][0], (0, False))
        self.assertEqual(seg_vert[1][1], (2, False))

    def test_two_points_in_plane_of_other_fracture_order_reversed(self):
        """
        Two fractures. One has two (non-consecutive) vertexes in the plane
        of another fracture. Order of polygons is reversed compared to similar test
        """
        f_1 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.intersections.polygons_3d(
            [f_2, f_1]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.all(on_bound[0]) == False)
        self.assertTrue(np.all(on_bound[1]) == False)

        known_points = np.array([[0, -1, -1], [0, 1, 1]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(seg_vert[1][0], (0, True))
        self.assertEqual(seg_vert[1][1], (2, True))
        self.assertEqual(seg_vert[0][0], (0, False))
        self.assertEqual(seg_vert[0][1], (2, False))

    def test_one_point_in_plane_of_other_fracture(self):
        """
        Two fractures. One has one vertexes in the plane
        of another fracture
        """
        f_1 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 2, 1]])
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.all(on_bound[0]) == False)
        self.assertTrue(np.all(on_bound[1]) == False)

        known_points = np.array([[0, -1, -1], [0, 1, 1]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(seg_vert[0][0], (0, True))
        self.assertEqual(seg_vert[0][1], (2, True))

        self.assertEqual(seg_vert[1][0], (0, False))
        self.assertEqual(seg_vert[1][1], (1, True))

    def test_one_point_in_plane_of_other_fracture_order_reversed(self):
        """
        Two fractures. One has one vertexes in the plane
        of another fracture
        """
        f_1 = np.array([[-0.5, 0.5, 0.5, -0.5], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 2, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d(
            [f_2, f_1]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.all(on_bound[0]) == False)
        self.assertTrue(np.all(on_bound[1]) == False)

        known_points = np.array([[0, -1, -1], [0, 1, 1]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(seg_vert[1][0], (0, True))
        self.assertEqual(seg_vert[1][1], (2, True))

        self.assertEqual(seg_vert[0][0], (0, False))
        self.assertEqual(seg_vert[0][1], (1, True))

    def test_point_contact_1(self):
        f_1 = np.array(
            [[0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 1.0, 1.0]]
        )
        f_2 = np.array(
            [[0.5, 1.0, 1.0, 0.5], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        )
        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        self.assertTrue(new_pt.shape[1] == 0)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 0)
        self.assertTrue(len(isect_pt[1]) == 0)

    def test_L_intersection(self):
        """
        Two fractures, L-intersection.
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 0.7, 0.7, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 1)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[0, 0.3, 0], [0, 0.7, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(seg_vert[0][0], (3, True))
        self.assertEqual(seg_vert[0][1], (3, True))

        self.assertEqual(seg_vert[1][0], (0, False))
        self.assertEqual(seg_vert[1][1], (1, False))

    def test_L_intersection_reverse_order(self):
        """
        Two fractures, L-intersection.
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 0.7, 0.7, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d(
            [f_2, f_1]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 1)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[0, 0.3, 0], [0, 0.7, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(seg_vert[1][0], (3, True))
        self.assertEqual(seg_vert[1][1], (3, True))

        self.assertEqual(seg_vert[0][0], (0, False))
        self.assertEqual(seg_vert[0][1], (1, False))

    def test_L_intersection_one_node_common(self):
        """
        Two fractures, L-intersection, one common node.
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 1.0, 1, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 1)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[0, 0.3, 0], [0, 1, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(seg_vert[0][0], (3, True))
        self.assertEqual(seg_vert[0][1], (3, False))

        self.assertEqual(seg_vert[1][0], (0, False))
        self.assertEqual(seg_vert[1][1], (1, False))

    def test_L_intersection_extends_beyond_each_other(self):
        """
        Two fractures, L-intersection, partly overlapping segments
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 1.5, 1.5, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 1)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[0, 0.3, 0], [0, 1, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(seg_vert[0][0], (3, False))
        self.assertEqual(seg_vert[0][1], (3, True))

        self.assertEqual(seg_vert[1][0], (0, True))
        self.assertEqual(seg_vert[1][1], (0, False))

    def test_T_intersection_within_polygon(self):
        """
        Two fractures, T-intersection, segment contained within the other polygon.
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, 0.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 0)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[0.5, 0.5, 0], [0.5, 0.9, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(len(seg_vert[0][0]), 0)
        self.assertEqual(len(seg_vert[0][1]), 0)

        self.assertEqual(seg_vert[1][0], (1, False))
        self.assertEqual(seg_vert[1][1], (2, False))

    def test_T_intersection_one_outside_polygon(self):
        """
        Two fractures, L-intersection, partly overlapping segments
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, 1.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 0)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[0.5, 0.5, 0], [0.5, 1.0, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(seg_vert[0][0], (2, True))
        self.assertEqual(len(seg_vert[0][1]), 0)

        self.assertEqual(seg_vert[1][0], (1, True))
        self.assertEqual(seg_vert[1][1], (1, False))

    def test_T_intersection_both_on_polygon(self):

        f_1 = np.array([[-2, -2, 2, 2], [-2, -2, 1, 1], [-2, 2, 2, -2]])
        f_2 = np.array(
            [[2.0, 2.0, 2.0, 2.0], [-2.0, -2.0, 2.0, 2.0], [2.0, -2.0, -2.0, 2.0]]
        )

        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 1)
        self.assertTrue(np.sum(on_bound[1]) == 0)

        known_points = np.array([[2, 1, 2], [2, 1.0, -2]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

    def test_T_intersection_one_outside_one_on_polygon(self):
        """
        Two fractures, L-intersection, partly overlapping segments
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.0, 0], [0.5, 1.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 0)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[0.5, 0.0, 0], [0.5, 1.0, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(seg_vert[0][0], (2, True))
        self.assertEqual(seg_vert[0][1], (0, True))

        self.assertEqual(seg_vert[1][0], (1, True))
        self.assertEqual(seg_vert[1][1], (1, False))

    def test_T_intersection_one_outside_one_on_polygon_reverse_order(self):
        """
        Two fractures, L-intersection, partly overlapping segments
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.0, 0], [0.5, 1.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d(
            [f_2, f_1]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 1)
        self.assertTrue(np.sum(on_bound[1]) == 0)

        known_points = np.array([[0.5, 0.0, 0], [0.5, 1.0, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(seg_vert[1][0], (0, True))
        self.assertEqual(seg_vert[1][1], (2, True))

        self.assertEqual(seg_vert[0][0], (1, False))
        self.assertEqual(seg_vert[0][1], (1, True))

    def test_T_intersection_both_on_boundary(self):
        """
        Two fractures, L-intersection, partly overlapping segments
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.0, 0], [0.5, 1.0, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert = pp.intersections.polygons_3d(
            [f_1, f_2]
        )
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 0)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[0.5, 0.0, 0], [0.5, 1.0, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        self.assertEqual(seg_vert[0][0], (0, True))
        self.assertEqual(seg_vert[0][1], (2, True))

        self.assertEqual(seg_vert[1][0], (1, False))
        self.assertEqual(seg_vert[1][1], (2, False))

    ### Tests involving polygons sharing a the same plane

    def test_same_plane_no_intersections(self):

        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[2, 3, 3, 2], [0, 0, 1, 1], [0, 0, 0, 0]])
        new_pt, isect_pt, on_bound, _, _ = pp.intersections.polygons_3d([f_1, f_2])
        self.assertTrue(new_pt.shape[1] == 0)

    def test_same_plane_intersections(self):

        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[2, 2, 0], [0, 2, 1.5], [0, 0, 0]])
        new_pt, isect_pt, on_bound, _, _ = pp.intersections.polygons_3d([f_1, f_2])
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 0)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[2.0 / 3, 1, 0], [1, 3.0 / 4, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

    def test_same_plane_shared_segment_1(self):
        # Shared segment and two vertexes
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[1, 1, 2], [0, 2, 1], [0, 0, 0]])
        new_pt, isect_pt, on_bound, _, _ = pp.intersections.polygons_3d([f_1, f_2])
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 1)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[1, 1, 0], [1, 0, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

    def test_same_plane_shared_segment_2(self):
        # Shared segment and one vertex. Of the common segments, the second polygon
        # has the longest extension.
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[1, 1, 2], [0, 1, 1.5], [0, 0, 0]])
        new_pt, isect_pt, on_bound, _, _ = pp.intersections.polygons_3d([f_1, f_2])
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 1)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[1, 1, 0], [1, 0, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

    def test_same_plane_shared_segment_3(self):
        # Shared segment, no common vertex.
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[1, 1, 2], [0.5, 0.9, 1.5], [0, 0, 0]])
        new_pt, isect_pt, on_bound, _, _ = pp.intersections.polygons_3d([f_1, f_2])
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 1)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[1, 0.5, 0], [1, 0.9, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

    def test_same_plane_shared_segment_4(self):
        # Shared segment and a vertex. The first polygon segment extends beyond the
        # second
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[1, 2, 1], [0, 2, 0.5], [0, 0, 0]])
        new_pt, isect_pt, on_bound, _, _ = pp.intersections.polygons_3d([f_1, f_2])
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 1)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[1, 0.5, 0], [1, 0, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

    def test_segment_intersection_identification(self):
        # This configuration turned out to produce a nasty bug

        f_1 = np.array([[1.5, 1.0, 1.0], [0.5, 0.5, 0.5], [0.5, 0.5, 1.0]])

        f_2 = np.array(
            [[0.7, 1.4, 1.4, 0.7], [0.4, 0.4, 1.4, 1.4], [0.6, 0.6, 0.6, 0.6]]
        )

        new_pt, isect_pt, _, _, seg_vert = pp.intersections.polygons_3d([f_1, f_2])

        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)

        known_points = np.array([[1, 0.5, 0.6], [1.4, 0.5, 0.6]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        # Find which of the new points have x index 1 and 1.4
        if new_pt[0, 0] == 1:
            x_1_ind = 0
            x_14_ind = 1
        else:
            x_1_ind = 1
            x_14_ind = 0

        # The x1 intersection point should be on the second segment of f_1, and
        # on its boundary
        self.assertTrue(seg_vert[0][x_1_ind][0] == 1)
        self.assertTrue(seg_vert[0][x_1_ind][1])

        # The x1 intersection point should be on the final segment of f_1, and
        # in its interior
        self.assertTrue(len(seg_vert[1][x_1_ind]) == 0)

        # The x14 intersection point should be on the second segment of f_1, and
        # on its boundary
        self.assertTrue(seg_vert[0][x_14_ind][0] == 2)
        self.assertTrue(seg_vert[0][x_14_ind][1])

        # The x1 intersection point should be on the second segment of f_1, and
        # in its interior
        self.assertTrue(seg_vert[1][x_14_ind][0] == 1)
        self.assertTrue(seg_vert[1][x_14_ind][1])

    def test_segment_intersection_identification_reverse_order(self):
        # This configuration turned out to produce a nasty bug

        f_1 = np.array([[1.5, 1.0, 1.0], [0.5, 0.5, 0.5], [0.5, 0.5, 1.0]])

        f_2 = np.array(
            [[0.7, 1.4, 1.4, 0.7], [0.4, 0.4, 1.4, 1.4], [0.6, 0.6, 0.6, 0.6]]
        )

        new_pt, isect_pt, _, _, seg_vert = pp.intersections.polygons_3d([f_2, f_1])

        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)

        known_points = np.array([[1, 0.5, 0.6], [1.4, 0.5, 0.6]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))

        # Find which of the new points have x index 1 and 1.4
        if new_pt[0, 0] == 1:
            x_1_ind = 0
            x_14_ind = 1
        else:
            x_1_ind = 1
            x_14_ind = 0

        # The x1 intersection point should be on the second segment of f_1, and
        # on its boundary
        self.assertTrue(seg_vert[1][x_1_ind][0] == 1)
        self.assertTrue(seg_vert[1][x_1_ind][1])

        # The x1 intersection point should be on the final segment of f_1, and
        # in its interior
        self.assertTrue(len(seg_vert[0][x_1_ind]) == 0)

        # The x14 intersection point should be on the second segment of f_1, and
        # on its boundary
        self.assertTrue(seg_vert[1][x_14_ind][0] == 2)
        self.assertTrue(seg_vert[1][x_14_ind][1])

        # The x1 intersection point should be on the second segment of f_1, and
        # in its interior
        self.assertTrue(seg_vert[0][x_14_ind][0] == 1)
        self.assertTrue(seg_vert[0][x_14_ind][1])


class TestPolygonPolyhedronIntersection(unittest.TestCase):
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
        poly = np.array(
            [[0.2, 0.8, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.cart_polyhedron
        )
        self.assertTrue(len(constrained_poly) == 1)
        self.assertTrue(np.allclose(constrained_poly[0], poly))

    def test_poly_outside_no_intersections(self):
        poly = np.array(
            [[1.2, 1.8, 1.8, 1.2], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.cart_polyhedron
        )
        self.assertTrue(len(constrained_poly) == 0)

    def test_poly_intersects_all_sides(self):
        # Polygon extends outside on all sides

        poly = np.array(
            [[-0.2, 1.8, 1.8, -0.2], [0.5, 0.5, 0.5, 0.5], [-0.2, -0.2, 1.8, 1.8]]
        )

        constrained_poly, inds = pp.constrain_geometry.polygons_by_polyhedron(
            poly, self.cart_polyhedron
        )

        known_constrained_poly = np.array(
            [[0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]]
        )

        self.assertTrue(len(constrained_poly) == 1)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly)
        )

    def test_poly_intersects_one_side(self):
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

        self.assertTrue(len(constrained_poly) == 1)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly)
        )

    def test_poly_intersects_two_sides(self):
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

        self.assertTrue(len(constrained_poly) == 1)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly)
        )

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

        self.assertTrue(len(constrained_poly) == 2)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly_1)
        )
        self.assertTrue(test_utils.compare_arrays(constrained_poly[1], poly_2))
        self.assertTrue(np.allclose(inds, np.arange(2)))

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
        self.assertTrue(len(constrained_poly) == 1)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly)
        )
        self.assertTrue(inds[0] == 0)

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

        self.assertTrue(len(constrained_poly) == 2)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly_1)
            or test_utils.compare_arrays(constrained_poly[0], known_constrained_poly_2)
        )
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[1], known_constrained_poly_1)
            or test_utils.compare_arrays(constrained_poly[1], known_constrained_poly_2)
        )
        self.assertTrue(np.all(inds == 0))

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

        self.assertTrue(len(constrained_poly) == 2)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly_1)
            or test_utils.compare_arrays(constrained_poly[0], known_constrained_poly_2)
        )
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[1], known_constrained_poly_1)
            or test_utils.compare_arrays(constrained_poly[1], known_constrained_poly_2)
        )
        self.assertTrue(np.all(inds == 0))

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

        self.assertTrue(len(constrained_poly) == 2)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly_1)
            or test_utils.compare_arrays(constrained_poly[0], known_constrained_poly_2)
        )
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[1], known_constrained_poly_1)
            or test_utils.compare_arrays(constrained_poly[1], known_constrained_poly_2)
        )
        self.assertTrue(np.all(inds == 0))

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

        self.assertTrue(len(constrained_poly) == 2)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly_1)
            or test_utils.compare_arrays(constrained_poly[0], known_constrained_poly_2)
        )
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[1], known_constrained_poly_1)
            or test_utils.compare_arrays(constrained_poly[1], known_constrained_poly_2)
        )
        self.assertTrue(np.all(inds == 0))

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
        import pdb

        #      pdb.set_trace()

        self.assertTrue(len(constrained_poly) == 2)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly_1)
            or test_utils.compare_arrays(constrained_poly[0], known_constrained_poly_2)
        )
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[1], known_constrained_poly_1)
            or test_utils.compare_arrays(constrained_poly[1], known_constrained_poly_2)
        )
        self.assertTrue(np.all(inds == 0))

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

        self.assertTrue(len(constrained_poly) == 1)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly)
        )
        self.assertTrue(inds[0] == 0)

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
        self.assertTrue(test_utils.compare_arrays(f_1, constrained_poly[0]))

    #        self.assertTrue(test_utils.compare_arrays(f_2, constrained_poly[1]))

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

        self.assertTrue(test_utils.compare_arrays(constrained_poly[0], known_poly))

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

        self.assertTrue(test_utils.compare_arrays(constrained_poly[0], known_poly))

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

        self.assertTrue(test_utils.compare_arrays(constrained_poly[0], known_poly))

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

        self.assertTrue(test_utils.compare_arrays(constrained_poly[0], known_poly))


if __name__ == "__main__":
    unittest.main()
