import numpy as np
import pytest

from porepy import intersections
from porepy.applications.test_utils.arrays import compare_arrays

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
        # Three lines no crossing (this test gave an error at some point).
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
        # Three lines one crossing (this test gave an error at some point).
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
