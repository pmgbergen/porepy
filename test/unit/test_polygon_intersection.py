import numpy as np
import unittest

import porepy as pp
from test import test_utils


class BasicTest(unittest.TestCase):
    """
    Various tests of intersect_polygon_lines.

    """

    def test_0(self):
        # convex polygon
        polygon = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        pts = np.array(
            [
                [0.0, 1.0, 0.0, 2.0, 0.5, 0.0, -0.5, 0.3, -1, 0.0],
                [0.0, 1.0, 0.0, 2.0, 1.0, 0.0, -0.5, 0.6, -1, 0.0],
            ]
        )
        lines = np.array([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])

        new_pts, new_lines = pp.cg.constrain_lines_by_polygon(polygon, pts, lines)

        pts_known = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.3],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 3 / 16, 0.6],
            ]
        )
        lines_known = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])

        self.assertTrue(np.allclose(new_pts, pts_known))
        self.assertTrue(np.allclose(new_lines, lines_known))

    def test_1(self):
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

        new_pts, new_lines = pp.cg.constrain_lines_by_polygon(polygon, pts, lines)
        pts_known = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.7, 0.8, 1.5],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 3 / 16, 0.6, 0.2, 0.2, 0.2, 0.2],
            ]
        )
        lines_known = np.array([[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]])

        self.assertTrue(np.allclose(new_pts, pts_known))
        self.assertTrue(np.allclose(new_lines, lines_known))


class TestIntersectionPolygonsEmbeddedIn3d(unittest.TestCase):
    def test_single_fracture(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert = pp.cg.intersect_polygons_3d([f_1])
        self.assertTrue(new_pt.size == 0)
        self.assertTrue(isect_pt.size == 1)
        self.assertTrue(len(isect_pt[0]) == 0)
        self.assertTrue(on_bound.size == 1)
        self.assertTrue(len(on_bound[0]) == 0)

        self.assertTrue(seg_vert.size == 1)
        self.assertTrue(seg_vert[0].size == 0)

    def test_two_intersecting_fractures(self):

        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])

        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.cg.intersect_polygons_3d(
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
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.cg.intersect_polygons_3d(
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

        counter = np.zeros(3, dtype=np.int)

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
                self.assertEqual(seg_vert[1][counter[1]], (1, True))
                self.assertEqual(seg_vert[1][counter[1] + 1], (3, True))
                self.assertEqual(seg_vert[2][counter[2]], (0, True))
                self.assertEqual(seg_vert[2][counter[2] + 1], (2, True))

            counter[p[0]] += 2
            counter[p[1]] += 2



    def test_three_intersecting_fractures_one_intersected_by_two(self):
        f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
        f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        f_3 = f_2 + np.array([0.5, 0, 0]).reshape((-1, 1))
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.cg.intersect_polygons_3d(
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

        counter = np.zeros(3, dtype=np.int)

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
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.cg.intersect_polygons_3d([f_1, f_2, f_3])
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

        counter = np.zeros(3, dtype=np.int)

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
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.cg.intersect_polygons_3d([f_1, f_2, f_3])
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

        counter = np.zeros(3, dtype=np.int)

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
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.cg.intersect_polygons_3d([f_1, f_2])
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
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.cg.intersect_polygons_3d([f_2, f_1])
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
        new_pt, isect_pt, on_bound, pairs, seg_vert = pp.cg.intersect_polygons_3d(
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
        new_pt, isect_pt, on_bound, _, seg_vert = pp.cg.intersect_polygons_3d([f_2, f_1])
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

    def test_L_intersection(self):
        """
        Two fractures, L-intersection.
        """
        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[0, 0, 0, 0], [0.3, 0.7, 0.7, 0.3], [0, 0, 1, 1]])
        new_pt, isect_pt, on_bound, _, seg_vert = pp.cg.intersect_polygons_3d([f_1, f_2])
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
        new_pt, isect_pt, on_bound, _, seg_vert = pp.cg.intersect_polygons_3d([f_2, f_1])
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
        new_pt, isect_pt, on_bound, _, seg_vert = pp.cg.intersect_polygons_3d([f_1, f_2])
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
        new_pt, isect_pt, on_bound, _, seg_vert = pp.cg.intersect_polygons_3d([f_1, f_2])
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
        self.assertEqual(seg_vert[0][1], (0, True))

        self.assertEqual(seg_vert[1][0], (0, True))
        self.assertEqual(seg_vert[1][1], (0, False))

    def test_T_intersection_within_polygon(self):
        """
        Two fractures, T-intersection, segment contained within the other polygon.
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, 0.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert = pp.cg.intersect_polygons_3d([f_1, f_2])
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

        new_pt, isect_pt, on_bound, _, seg_vert = pp.cg.intersect_polygons_3d([f_1, f_2])
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

    def test_T_intersection_one_outside_one_on_polygon(self):
        """
        Two fractures, L-intersection, partly overlapping segments
        """
        f_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f_2 = np.array([[0.5, 0.5, 1], [0.5, 0.0, 0], [0.5, 1.9, 0.0]]).T

        new_pt, isect_pt, on_bound, _, seg_vert = pp.cg.intersect_polygons_3d([f_1, f_2])
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

        new_pt, isect_pt, on_bound, _, seg_vert = pp.cg.intersect_polygons_3d([f_2, f_1])
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

        new_pt, isect_pt, on_bound, _, seg_vert = pp.cg.intersect_polygons_3d([f_1, f_2])
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
        new_pt, isect_pt, on_bound, _, _ = pp.cg.intersect_polygons_3d([f_1, f_2])
        self.assertTrue(new_pt.shape[1] == 0)

    def test_same_plane_intersections(self):

        f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f_2 = np.array([[2, 2, 0], [0, 2, 1.5], [0, 0, 0]])
        new_pt, isect_pt, on_bound, _, _ = pp.cg.intersect_polygons_3d([f_1, f_2])
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
        new_pt, isect_pt, on_bound, _, _ = pp.cg.intersect_polygons_3d([f_1, f_2])
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
        new_pt, isect_pt, on_bound, _, _ = pp.cg.intersect_polygons_3d([f_1, f_2])
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
        new_pt, isect_pt, on_bound, _, _ = pp.cg.intersect_polygons_3d([f_1, f_2])
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
        new_pt, isect_pt, on_bound, _, _ = pp.cg.intersect_polygons_3d([f_1, f_2])
        self.assertTrue(new_pt.shape[1] == 2)
        self.assertTrue(isect_pt.size == 2)
        self.assertTrue(len(isect_pt[0]) == 2)
        self.assertTrue(len(isect_pt[1]) == 2)
        self.assertTrue(on_bound.size == 2)
        self.assertTrue(np.sum(on_bound[0]) == 1)
        self.assertTrue(np.sum(on_bound[1]) == 1)

        known_points = np.array([[1, 0.5, 0], [1, 0, 0]]).T
        self.assertTrue(test_utils.compare_arrays(new_pt, known_points))


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
        self.non_convex_polyhedron = [
            west,
            east,
            south_w,
            south_e,
            north_w,
            north_e,
            bottom_w,
            bottom_e,
            top,
        ]

    def test_poly_inside_no_intersections(self):
        poly = np.array(
            [[0.2, 0.8, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )

        constrained_poly, inds = pp.cg.constrain_polygons_by_polyhedron(
            poly, self.cart_polyhedron
        )
        self.assertTrue(len(constrained_poly) == 1)
        self.assertTrue(np.allclose(constrained_poly[0], poly))

    def test_poly_outside_no_intersections(self):
        poly = np.array(
            [[1.2, 1.8, 1.8, 1.2], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )

        constrained_poly, inds = pp.cg.constrain_polygons_by_polyhedron(
            poly, self.cart_polyhedron
        )
        self.assertTrue(len(constrained_poly) == 0)

    def test_poly_intersects_all_sides(self):

        poly = np.array(
            [[-0.2, 1.8, 1.8, -0.2], [0.5, 0.5, 0.5, 0.5], [-0.2, -0.2, 1.8, 1.8]]
        )

        constrained_poly, inds = pp.cg.constrain_polygons_by_polyhedron(
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

        poly = np.array(
            [[0.2, 1.8, 1.8, 0.2], [0.5, 0.5, 0.5, 0.5], [-0.2, -0.2, 1.8, 1.8]]
        )

        constrained_poly, inds = pp.cg.constrain_polygons_by_polyhedron(
            poly, self.cart_polyhedron
        )

        known_constrained_poly = np.array(
            [[0.2, 1, 1, 0.2], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]]
        )

        self.assertTrue(len(constrained_poly) == 1)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly)
        )

    def test_two_poly_one_intersects(self):
        poly_1 = np.array(
            [[-0.2, 1.8, 1.8, -0.2], [0.5, 0.5, 0.5, 0.5], [-0.2, -0.2, 1.8, 1.8]]
        )
        poly_2 = np.array(
            [[0.2, 0.8, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )

        constrained_poly, inds = pp.cg.constrain_polygons_by_polyhedron(
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
        poly = np.array([[-1, 2, 2, 1], [0.5, 0.5, 0.5, 0.5], [-1, -1, 1, 1]])

        known_constrained_poly = np.array(
            [[0, 0.5, 1, 1, 0], [0.5, 0.5, 0.5, 0.5, 0.5], [0, 0.5, 0, 1, 1]]
        )
        constrained_poly, inds = pp.cg.constrain_polygons_by_polyhedron(
            poly, self.non_convex_polyhedron
        )

        self.assertTrue(len(constrained_poly) == 1)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly)
        )
        self.assertTrue(inds[0] == 0)

    def test_poly_split_by_non_convex_domain(self):
        poly = np.array([[-1, 2, 2, -1], [0.5, 0.5, 0.5, 0.5], [-1, -1, 0.3, 0.3]])

        known_constrained_poly_1 = np.array(
            [[0, 0.3, 0], [0.5, 0.5, 0.5], [0, 0.3, 0.3]]
        )
        known_constrained_poly_2 = np.array(
            [[0.7, 1, 1], [0.5, 0.5, 0.5], [0.3, 0.0, 0.3]]
        )

        constrained_poly, inds = pp.cg.constrain_polygons_by_polyhedron(
            poly, self.non_convex_polyhedron
        )

        self.assertTrue(len(constrained_poly) == 2)
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[0], known_constrained_poly_1)
            or test_utils.compare_arrays(
                constrained_poly[0], known_constrained_poly_2
            )
        )
        self.assertTrue(
            test_utils.compare_arrays(constrained_poly[1], known_constrained_poly_1)
            or test_utils.compare_arrays(
                constrained_poly[1], known_constrained_poly_2
            )
        )
        self.assertTrue(np.all(inds == 0))

if __name__ == "__main__":

    #TestPolygonPolyhedronIntersection().test_poly_split_by_non_convex_domain()
    unittest.main()
