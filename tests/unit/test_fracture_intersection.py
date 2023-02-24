""" Various tests for intersections between fractures, and between
fractures and boundaries.

"""

import unittest

import numpy as np

import porepy as pp
from tests import test_utils


class TestSweepAndPrune(unittest.TestCase):
    def pairs_contain(self, pairs, a):
        for pi in range(pairs.shape[1]):
            if a[0] == pairs[0, pi] and a[1] == pairs[1, pi]:
                return True
        return False

    def test_no_intersection(self):
        x_min = np.array([0, 2])
        x_max = np.array([1, 3])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)
        self.assertTrue(pairs.size == 0)

    def test_intersection_two_lines(self):
        x_min = np.array([0, 1])
        x_max = np.array([2, 3])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        self.assertTrue(pairs.size == 2)
        self.assertTrue(pairs[0, 0] == 0)
        self.assertTrue(pairs[1, 0] == 1)

    def test_intersection_two_lines_switched_order(self):
        x_min = np.array([1, 0])
        x_max = np.array([3, 2])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        self.assertTrue(pairs.size == 2)
        self.assertTrue(pairs[0, 0] == 0)
        self.assertTrue(pairs[1, 0] == 1)

    def test_intersection_two_lines_same_start(self):
        x_min = np.array([0, 0])
        x_max = np.array([3, 2])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        self.assertTrue(pairs.size == 2)
        self.assertTrue(pairs[0, 0] == 0)
        self.assertTrue(pairs[1, 0] == 1)

    def test_intersection_two_lines_same_end(self):
        x_min = np.array([0, 1])
        x_max = np.array([3, 3])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        self.assertTrue(pairs.size == 2)
        self.assertTrue(pairs[0, 0] == 0)
        self.assertTrue(pairs[1, 0] == 1)

    def test_intersection_two_lines_same_start_and_end(self):
        x_min = np.array([0, 0])
        x_max = np.array([3, 3])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        self.assertTrue(pairs.size == 2)
        self.assertTrue(pairs[0, 0] == 0)
        self.assertTrue(pairs[1, 0] == 1)

    def test_intersection_two_lines_one_is_point_no_intersection(self):
        x_min = np.array([0, 1])
        x_max = np.array([0, 2])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        self.assertTrue(pairs.size == 0)

    def test_intersection_two_lines_one_is_point_intersection(self):
        x_min = np.array([1, 0])
        x_max = np.array([1, 2])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        self.assertTrue(pairs.size == 2)
        self.assertTrue(pairs[0, 0] == 0)
        self.assertTrue(pairs[1, 0] == 1)

    def test_intersection_three_lines_two_intersect(self):
        x_min = np.array([1, 0, 3])
        x_max = np.array([2, 2, 4])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        self.assertTrue(pairs.size == 2)
        self.assertTrue(pairs[0, 0] == 0)
        self.assertTrue(pairs[1, 0] == 1)

    def test_intersection_three_lines_all_intersect(self):
        x_min = np.array([1, 0, 1])
        x_max = np.array([2, 2, 3])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        self.assertTrue(pairs.size == 6)
        self.assertTrue(self.pairs_contain(pairs, [0, 1]))
        self.assertTrue(self.pairs_contain(pairs, [0, 2]))
        self.assertTrue(self.pairs_contain(pairs, [1, 2]))

    def test_intersection_three_lines_pairs_intersect(self):
        x_min = np.array([0, 0, 2])
        x_max = np.array([1, 3, 3])

        pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)

        self.assertTrue(pairs.size == 4)
        self.assertTrue(self.pairs_contain(pairs, [0, 1]))
        self.assertTrue(self.pairs_contain(pairs, [1, 2]))


class TestBoundingBoxIntersection(unittest.TestCase):
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
        self.assertTrue(pairs_1.size == 0)

        combined_pairs = pp.intersections._identify_overlapping_rectangles(
            x_min, x_max, x_min, x_max
        )
        self.assertTrue(combined_pairs.size == 0)

    def test_intersection_x_not_y(self):
        # The points are overlapping on the x-axis but not on the y-axis
        x_min = np.array([0, 0])
        x_max = np.array([2, 2])

        y_min = np.array([0, 5])
        y_max = np.array([2, 7])

        x_pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)
        y_pairs = pp.intersections._identify_overlapping_intervals(y_min, y_max)
        pairs_1 = pp.intersections._intersect_pairs(x_pairs, y_pairs)
        self.assertTrue(pairs_1.size == 0)

        combined_pairs = pp.intersections._identify_overlapping_rectangles(
            x_min, x_max, y_min, y_max
        )
        self.assertTrue(combined_pairs.size == 0)

    def test_intersection_x_and_y(self):
        # The points are overlapping on the x-axis but not on the y-axis
        x_min = np.array([0, 0])
        x_max = np.array([2, 2])

        y_min = np.array([0, 1])
        y_max = np.array([2, 3])

        x_pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)
        y_pairs = pp.intersections._identify_overlapping_intervals(y_min, y_max)
        pairs_1 = np.sort(pp.intersections._intersect_pairs(x_pairs, y_pairs), axis=0)
        self.assertTrue(pairs_1.size == 2)

        combined_pairs = np.sort(
            pp.intersections._identify_overlapping_rectangles(
                x_min, x_max, y_min, y_max
            ),
            axis=0,
        )
        self.assertTrue(combined_pairs.size == 2)

        self.assertTrue(np.allclose(pairs_1, combined_pairs))

    def test_lines_in_square(self):
        # Lines in square, all should overlap
        x_min = np.array([0, 1, 0, 0])
        x_max = np.array([1, 1, 1, 0])

        y_min = np.array([0, 0, 1, 0])
        y_max = np.array([0, 1, 1, 1])

        x_pairs = pp.intersections._identify_overlapping_intervals(x_min, x_max)
        y_pairs = pp.intersections._identify_overlapping_intervals(y_min, y_max)
        pairs_1 = np.sort(pp.intersections._intersect_pairs(x_pairs, y_pairs), axis=0)
        self.assertTrue(pairs_1.shape[1] == 4)

        combined_pairs = np.sort(
            pp.intersections._identify_overlapping_rectangles(
                x_min, x_max, y_min, y_max
            ),
            axis=0,
        )
        self.assertTrue(combined_pairs.shape[1] == 4)

        self.assertTrue(np.allclose(pairs_1, combined_pairs))


class TestFractureIntersectionRemoval(unittest.TestCase):
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

        self.assertTrue(np.allclose(new_pts, p_known))
        self.assertTrue(test_utils.compare_arrays(new_lines, lines_known))

    def test_lines_no_crossing(self):
        p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

        lines = np.array([[0, 1], [2, 3]])
        box = np.array([[2], [2]])
        new_pts, new_lines, _ = pp.intersections.split_intersecting_segments_2d(
            p, lines
        )
        self.assertTrue(np.allclose(new_pts, p))
        self.assertTrue(np.allclose(new_lines, lines))

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
        self.assertTrue(np.allclose(new_pts, p_known))
        self.assertTrue(np.allclose(new_lines, lines))

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
        self.assertTrue(np.allclose(new_pts, p_known))
        self.assertTrue(test_utils.compare_arrays(new_lines, lines_known))

    def test_overlapping_lines(self):
        p = np.array([[-0.6, 0.4, 0.4, -0.6, 0.4], [-0.5, -0.5, 0.5, 0.5, 0.0]])
        lines = np.array([[0, 0, 1, 1, 2], [1, 3, 2, 4, 3]])
        new_pts, new_lines, _ = pp.intersections.split_intersecting_segments_2d(
            p, lines
        )

        lines_known = np.array([[0, 1], [0, 3], [1, 4], [2, 4], [2, 3]]).T
        self.assertTrue(np.allclose(new_pts, p))
        self.assertTrue(test_utils.compare_arrays(new_lines, lines_known))


class TestFractureBoundaryIntersection(unittest.TestCase):
    """
    Test of algorithm for constraining a fracture a bounding box.

    Since that algorithm uses fracture intersection methods, the tests functions as
    partial test for the wider fracture intersection framework as well. Full tests
    of the latter are too time consuming to fit into a unit test.

    Now the boundary is defined as set of "fake" fractures, all fracture network
    have 2*dim additional fractures (hence the + 6 in the assertions)
    """

    def setup(self):
        self.f_1 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]]),
            check_convexity=False,
        )
        self.domain = pp.Domain(
            {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        )

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
        network = pp.FractureNetwork3d([f])
        network.impose_external_boundary(self.domain)
        self.assertTrue(len(network.fractures) == (0 + 6))

    def test_outside_west_bottom(self):
        self.setup()
        f = self.f_1
        f.pts[0] -= 0.5
        f.pts[2] -= 1.5
        network = pp.FractureNetwork3d([f])
        network.impose_external_boundary(self.domain)
        self.assertTrue(len(network.fractures) == (0 + 6))

    def test_intersect_one(self):
        self.setup()
        f = self.f_1
        f.pts[0] -= 0.5
        f.pts[2, :] = [0.2, 0.2, 0.8, 0.8]
        network = pp.FractureNetwork3d([f])
        network.impose_external_boundary(self.domain)
        p_known = np.array(
            [[0.0, 0.5, 0.5, 0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )
        self.assertTrue(len(network.fractures) == (1 + 6))
        p_comp = network.fractures[0].pts
        self.assertTrue(self._arrays_equal(p_known, p_comp))

    def test_intersect_two_same(self):
        self.setup()
        f = self.f_1
        f.pts[0, :] = [-0.5, 1.5, 1.5, -0.5]
        f.pts[2, :] = [0.2, 0.2, 0.8, 0.8]
        network = pp.FractureNetwork3d([f])
        network.impose_external_boundary(self.domain)
        p_known = np.array([[0.0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
        self.assertTrue(len(network.fractures) == (1 + 6))
        p_comp = network.fractures[0].pts
        self.assertTrue(self._arrays_equal(p_known, p_comp))

    def test_incline_in_plane(self):
        self.setup()
        f = self.f_1
        f.pts[0] -= 0.5
        f.pts[2, :] = [0, -0.5, 0.5, 1]
        network = pp.FractureNetwork3d([f])
        network.impose_external_boundary(self.domain)
        p_known = np.array(
            [[0.0, 0.5, 0.5, 0], [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.5, 0.75]]
        )
        self.assertTrue(len(network.fractures) == (1 + 6))
        p_comp = network.fractures[0].pts
        self.assertTrue(self._arrays_equal(p_known, p_comp))

    def test_full_incline(self):
        self.setup()
        p = np.array([[-0.5, 0.5, 0.5, -0.5], [0.5, 0.5, 1.5, 1.5], [-0.5, -0.5, 1, 1]])
        f = pp.PlaneFracture(p, check_convexity=False)
        network = pp.FractureNetwork3d([f])
        network.impose_external_boundary(self.domain)
        p_known = np.array(
            [[0.0, 0.5, 0.5, 0], [5.0 / 6, 5.0 / 6, 1, 1], [0.0, 0.0, 0.25, 0.25]]
        )
        self.assertTrue(len(network.fractures) == (1 + 6))
        p_comp = network.fractures[0].pts
        self.assertTrue(self._arrays_equal(p_known, p_comp))


if __name__ == "__main__":
    unittest.main()
