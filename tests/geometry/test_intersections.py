import numpy as np

from porepy import intersections
from porepy.applications.test_utils.arrays import compare_arrays

# ---------- Testing triangulations ----------


def test_identical_triangles():
    p = np.array([[0, 1, 0], [0, 0, 1]])
    t = np.array([[0, 1, 2]]).T

    triangulation = intersections.triangulations(p, p, t, t)
    assert len(triangulation) == 1
    assert triangulation[0][0] == 0
    assert triangulation[0][1] == 0
    assert triangulation[0][2] == 0.5


def test_two_and_one():
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


def test_one_and_two():
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


def test_no_intersection():
    x_min = np.array([0, 2])
    x_max = np.array([1, 3])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)
    assert pairs.size == 0


def test_intersection_two_lines():
    x_min = np.array([0, 1])
    x_max = np.array([2, 3])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)

    assert pairs.size == 2
    assert pairs[0, 0] == 0
    assert pairs[1, 0] == 1


def test_intersection_two_lines_switched_order():
    x_min = np.array([1, 0])
    x_max = np.array([3, 2])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)

    assert pairs.size == 2
    assert pairs[0, 0] == 0
    assert pairs[1, 0] == 1


def test_intersection_two_lines_same_start():
    x_min = np.array([0, 0])
    x_max = np.array([3, 2])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)

    assert pairs.size == 2
    assert pairs[0, 0] == 0
    assert pairs[1, 0] == 1


def test_intersection_two_lines_same_end():
    x_min = np.array([0, 1])
    x_max = np.array([3, 3])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)

    assert pairs.size == 2
    assert pairs[0, 0] == 0
    assert pairs[1, 0] == 1


def test_intersection_two_lines_same_start_and_end():
    x_min = np.array([0, 0])
    x_max = np.array([3, 3])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)

    assert pairs.size == 2
    assert pairs[0, 0] == 0
    assert pairs[1, 0] == 1


def test_intersection_two_lines_one_is_point_no_intersection():
    x_min = np.array([0, 1])
    x_max = np.array([0, 2])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)

    assert pairs.size == 0


def test_intersection_two_lines_one_is_point_intersection():
    x_min = np.array([1, 0])
    x_max = np.array([1, 2])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)

    assert pairs.size == 2
    assert pairs[0, 0] == 0
    assert pairs[1, 0] == 1


def test_intersection_three_lines_two_intersect():
    x_min = np.array([1, 0, 3])
    x_max = np.array([2, 2, 4])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)

    assert pairs.size == 2
    assert pairs[0, 0] == 0
    assert pairs[1, 0] == 1


def test_intersection_three_lines_all_intersect():
    x_min = np.array([1, 0, 1])
    x_max = np.array([2, 2, 3])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)

    assert pairs.size == 6
    assert check_pairs_contain(pairs, [0, 1])
    assert check_pairs_contain(pairs, [0, 2])
    assert check_pairs_contain(pairs, [1, 2])


def test_intersection_three_lines_pairs_intersect():
    x_min = np.array([0, 0, 2])
    x_max = np.array([1, 3, 3])

    pairs = intersections._identify_overlapping_intervals(x_min, x_max)

    assert pairs.size == 4
    assert check_pairs_contain(pairs, [0, 1])
    assert check_pairs_contain(pairs, [1, 2])


# ---------- Testing _intersect_pairs ----------
# We run both 1d search + intersection, and 2d search. They should be equivalent.
# Note: The tests are only between the bounding boxes of the fractures, not the
# fractures themselves.


def test_no_intersection_intersect_pairs():
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


def test_intersection_x_not_y():
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


def test_intersection_x_and_y():
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

    assert np.allclose(pairs_1, combined_pairs)


def test_lines_in_square():
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

    assert np.allclose(pairs_1, combined_pairs)


# ---------- Testing split_intersecting_segments_2d ----------
# Tests for function used to remove intersections between 1d fractures.


def test_lines_crossing_origin():
    p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])
    lines = np.array([[0, 2], [1, 3], [1, 2], [3, 4]])

    new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)

    p_known = np.hstack((p, np.array([[0], [0]])))

    lines_known = np.array([[0, 4, 2, 4], [4, 1, 4, 3], [1, 1, 2, 2], [3, 3, 4, 4]])

    assert np.allclose(new_pts, p_known)
    assert compare_arrays(new_lines, lines_known)


def test_lines_no_crossing():
    p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

    lines = np.array([[0, 1], [2, 3]])
    new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)
    assert np.allclose(new_pts, p)
    assert np.allclose(new_lines, lines)


def test_three_lines_no_crossing():
    # This test gave an error at some point
    p = np.array(
        [[0.0, 0.0, 0.3, 1.0, 1.0, 0.5], [2 / 3, 1 / 0.7, 0.3, 2 / 3, 1 / 0.7, 0.5]]
    )
    lines = np.array([[0, 3], [1, 4], [2, 5]]).T

    new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)
    p_known = p
    assert np.allclose(new_pts, p_known)
    assert np.allclose(new_lines, lines)


def test_three_lines_one_crossing():
    # This test gave an error at some point
    p = np.array([[0.0, 0.5, 0.3, 1.0, 0.3, 0.5], [2 / 3, 0.3, 0.3, 2 / 3, 0.5, 0.5]])
    lines = np.array([[0, 3], [2, 5], [1, 4]]).T

    new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)
    p_known = np.hstack((p, np.array([[0.4], [0.4]])))
    lines_known = np.array([[0, 3], [2, 6], [6, 5], [1, 6], [6, 4]]).T
    assert np.allclose(new_pts, p_known)
    assert compare_arrays(new_lines, lines_known)


def test_overlapping_lines():
    p = np.array([[-0.6, 0.4, 0.4, -0.6, 0.4], [-0.5, -0.5, 0.5, 0.5, 0.0]])
    lines = np.array([[0, 0, 1, 1, 2], [1, 3, 2, 4, 3]])
    new_pts, new_lines, _ = intersections.split_intersecting_segments_2d(p, lines)

    lines_known = np.array([[0, 1], [0, 3], [1, 4], [2, 4], [2, 3]]).T
    assert np.allclose(new_pts, p)
    assert compare_arrays(new_lines, lines_known)
