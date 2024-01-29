"""Testing the sorting of points for multiple cases."""
import numpy as np
import pytest

from porepy.utils import sort_points
from porepy.applications.test_utils.arrays import compare_arrays


@pytest.mark.parametrize(
    "points,target_lines,target_sort_idx",
    [
        (
            np.array([[1, 2], [5, 1], [2, 7], [7, 5]]).T,
            np.array([[1, 2], [2, 7], [7, 5], [5, 1]]).T,
            np.array([0, 2, 3, 1]),
        ),
    ],
)
def test_circular(points, target_lines, target_sort_idx):
    sp, sort_idx = sort_points.sort_point_pairs(points)
    assert np.allclose(target_lines, sp)
    assert np.allclose(target_sort_idx, sort_idx)


@pytest.mark.parametrize(
    "points,target_lines,target_sort_idx",
    [
        # The points are not circular, but the isolated points are contained in the
        # first and last column, thus no rearrangement is needed
        (
            np.array([[1, 0], [1, 3], [3, 2]]).T,
            np.array([[0, 1], [1, 3], [3, 2]]).T,
            np.array([0, 1, 2]),
        ),
    ],
)
def test_not_circular(points, target_lines, target_sort_idx):
    sp, sort_idx = sort_points.sort_point_pairs(points, is_circular=False)
    assert np.allclose(target_lines, sp)
    assert np.allclose(target_sort_idx, sort_idx)


@pytest.mark.parametrize(
    "points,target_lines,target_sort_idx",
    [
        # The points are not circular, but the isolated points are contained in the
        # first column, thus re-arrangement should be automatic
        (
            np.array([[1, 0], [3, 2], [1, 3]]).T,
            np.array([[0, 1], [1, 3], [3, 2]]).T,
            np.array([0, 2, 1]),
        ),
        # The points are not circular, and the isolated points are not contained in the
        # first column, thus re-arrangement is needed
        (
            np.array([[1, 3], [3, 2], [1, 0]]).T,
            np.array([[2, 3], [3, 1], [1, 0]]).T,
            np.array([1, 0, 2]),
        ),
    ],
)
def test_not_circular_permuted(points, target_lines, target_sort_idx):
    sp, sort_idx = sort_points.sort_point_pairs(points, is_circular=False)
    assert compare_arrays(sp, target_lines)
    assert np.allclose(target_sort_idx, sort_idx)


@pytest.mark.parametrize(
    "points,center,target_ordering",
    [
        # points already in xy plane
        (
            np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]]),
            np.array([[0.5], [0.5], [0]]),
            np.array([0, 1, 3, 2]),
        ),
        # points already in yz plane
        (
            np.array([[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]]),
            np.array([[0.5], [0.5], [0]]),
            np.array([0, 1, 3, 2]),
        ),
        # points to be rotated
        (
            np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]]),
            np.array([[0.5], [0.5], [0]]),
            np.array([0, 1, 3, 2]),
        ),
    ],
)
def test_sort_points_in_plane(points, center, target_ordering):
    sp = sort_points.sort_point_plane(points, center)
    assert compare_arrays(sp, target_ordering)


@pytest.mark.parametrize(
    "points,target_sorting",
    [
        # no sorting
        (np.array([[0, 1, 2], [2, 1, 3]]).T, np.array([[0, 1, 2], [2, 1, 3]]).T),
        # sort one
        (np.array([[0, 1, 2], [1, 2, 3]]).T, np.array([[0, 1, 2], [2, 1, 3]]).T),
        (np.array([[1, 3, 0], [3, 2, 1]]).T, np.array([[1, 3, 0], [1, 2, 3]]).T),
        # two fracs sorted, second not third
        (
            np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]).T,
            np.array([[0, 1, 2], [2, 1, 3], [2, 3, 4]]).T,
        ),
        # two fracs sorted, second then third
        (
            np.array([[0, 1, 2], [1, 2, 3], [3, 2, 4]]).T,
            np.array([[0, 1, 2], [2, 1, 3], [2, 3, 4]]).T,
        ),
        # four fracs, last sorted automatically
        (
            np.array([[0, 1, 2], [1, 2, 3], [3, 2, 4], [0, 4, 2]]).T,
            np.array([[0, 1, 2], [2, 1, 3], [2, 3, 4], [2, 4, 0]]).T,
        ),
        # issue #1 (here the points where copied during passing as arg)
        (
            np.array(
                [
                    [2, 1, 0, 5, 1, 0, 1, 6, 4, 7, 4, 5],
                    [3, 3, 5, 3, 5, 5, 6, 7, 3, 3, 7, 4],
                    [1, 0, 3, 4, 6, 1, 2, 2, 7, 2, 6, 6],
                ]
            ),
            np.array(
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
            ).T,
        ),
    ],
)
def test_sorting_triangle_edge(points, target_sorting):
    sorted_t = sort_points.sort_triangle_edges(points)
    assert np.allclose(sorted_t, target_sorting)
