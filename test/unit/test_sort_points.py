import unittest
from test import test_utils

import numpy as np

from porepy.utils import sort_points


class SortLinePairTest(unittest.TestCase):
    def test_quad(self):
        p = np.array([[1, 2], [5, 1], [2, 7], [7, 5]]).T
        sp, sort_ind = sort_points.sort_point_pairs(p)
        # Use numpy arrays to ease comparison of points
        known_lines = np.array([[1, 2], [2, 7], [7, 5], [5, 1]]).T
        known_sort_ind = np.array([0, 2, 3, 1])

        self.assertTrue(np.allclose(known_lines, sp))
        self.assertTrue(np.allclose(known_sort_ind, sort_ind))

    def test_not_circular_1(self):
        # The points are not circular, but the isolated points are contained in the
        # first and last column, thus no rearrangement is needed
        p = np.array([[1, 0], [1, 3], [3, 2]]).T
        sp, sort_ind = sort_points.sort_point_pairs(p, is_circular=False)

        known_lines = np.array([[0, 1], [1, 3], [3, 2]]).T
        known_sort_ind = np.array([0, 1, 2])

        self.assertTrue(test_utils.compare_arrays(known_lines, sp))
        self.assertTrue(np.allclose(known_sort_ind, sort_ind))

    def test_not_circular_2(self):
        # The points are not circular, but the isolated points are contained in the
        # first column, thus re-arrangement should be automatic
        p = np.array([[1, 0], [3, 2], [1, 3]]).T
        sp, sort_ind = sort_points.sort_point_pairs(p, is_circular=False)

        known_lines = np.array([[0, 1], [1, 3], [3, 2]]).T
        known_sort_ind = np.array([0, 2, 1])

        self.assertTrue(test_utils.compare_arrays(sp, known_lines))
        self.assertTrue(np.allclose(known_sort_ind, sort_ind))

    def test_not_circular_3(self):
        # The points are not circular, and the isolated points are not contained in the
        # first column, thus re-arrangement is needed
        p = np.array([[1, 3], [3, 2], [1, 0]]).T
        sp, sort_ind = sort_points.sort_point_pairs(p, is_circular=False)

        known_lines = np.array([[2, 3], [3, 1], [1, 0]]).T
        known_sort_ind = np.array([1, 0, 2])

        self.assertTrue(test_utils.compare_arrays(sp, known_lines))
        self.assertTrue(np.allclose(known_sort_ind, sort_ind))


class TestSortPointPlane(unittest.TestCase):
    def test_points_already_in_xy_plane(self):
        p = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
        center = np.array([[0.5], [0.5], [0]])
        sp = sort_points.sort_point_plane(p, center)
        known_ordering = np.array([0, 1, 3, 2])
        self.assertTrue(test_utils.compare_arrays(sp, known_ordering))

    def test_points_already_in_yz_plane(self):
        p = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]])
        center = np.array([[0.5], [0.5], [0]])
        sp = sort_points.sort_point_plane(p, center)
        known_ordering = np.array([0, 1, 3, 2])
        self.assertTrue(test_utils.compare_arrays(sp, known_ordering))

    def test_points_to_be_rotated(self):
        p = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]])
        center = np.array([[0.5], [0.5], [0]])
        sp = sort_points.sort_point_plane(p, center)
        known_ordering = np.array([0, 1, 3, 2])
        self.assertTrue(test_utils.compare_arrays(sp, known_ordering))


class SortTriangleEdges(unittest.TestCase):
    def test_no_sorting(self):

        t = np.array([[0, 1, 2], [2, 1, 3]]).T
        sorted_t = sort_points.sort_triangle_edges(t)

        truth = t
        self.assertTrue(np.allclose(sorted_t, truth))

    def test_sort_one(self):

        t = np.array([[0, 1, 2], [1, 2, 3]]).T
        sorted_t = sort_points.sort_triangle_edges(t)

        truth = np.array([[0, 1, 2], [2, 1, 3]]).T
        self.assertTrue(np.allclose(sorted_t, truth))

    def test_sort_one_2(self):

        t = np.array([[1, 3, 0], [3, 2, 1]]).T
        sorted_t = sort_points.sort_triangle_edges(t)

        truth = np.array([[1, 3, 0], [1, 2, 3]]).T
        self.assertTrue(np.allclose(sorted_t, truth))

    def test_two_fracs_sort_second_not_third(self):
        # The first will lead the second to be sorted, which again remove the need to sort the third
        t = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]).T

        sorted_t = sort_points.sort_triangle_edges(t)

        truth = np.array([[0, 1, 2], [2, 1, 3], [2, 3, 4]]).T
        self.assertTrue(np.allclose(sorted_t, truth))

    def test_two_fracs_sort_second_then_third(self):
        # The first will lead the second to be sorted, which again remove the need to sort the third
        t = np.array([[0, 1, 2], [1, 2, 3], [3, 2, 4]]).T

        sorted_t = sort_points.sort_triangle_edges(t)

        truth = np.array([[0, 1, 2], [2, 1, 3], [2, 3, 4]]).T
        self.assertTrue(np.allclose(sorted_t, truth))

    def test_four_fracs_last_sorted_automatically(self):
        t = np.array([[0, 1, 2], [1, 2, 3], [3, 2, 4], [0, 4, 2]]).T
        sorted_t = sort_points.sort_triangle_edges(t)

        truth = np.array([[0, 1, 2], [2, 1, 3], [2, 3, 4], [2, 4, 0]]).T
        self.assertTrue(np.allclose(sorted_t, truth))

    def test_issue_1(self):
        # Bug found while using the code
        t = np.array(
            [
                [2, 1, 0, 5, 1, 0, 1, 6, 4, 7, 4, 5],
                [3, 3, 5, 3, 5, 5, 6, 7, 3, 3, 7, 4],
                [1, 0, 3, 4, 6, 1, 2, 2, 7, 2, 6, 6],
            ]
        )
        sorted_t = sort_points.sort_triangle_edges(t.copy())
        truth = np.array(
            [
                [2, 3, 1],
                [1, 3, 0],
                [3, 5, 0],
                [5, 3, 4],
                [1, 5, 6],
                [0, 5, 1],
                [1, 6, 2],
                [6, 7, 2],
                [4, 3, 7],
                [7, 3, 2],
                [4, 7, 6],
                [5, 4, 6],
            ]
        ).T
        self.assertTrue(np.allclose(sorted_t, truth))


if __name__ == "__main__":
    unittest.main()
