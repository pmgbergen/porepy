""" """

from __future__ import annotations

import numpy as np
import pytest

from porepy import array_operations

coord_1 = [
    [[np.array([1])], np.array([42])],
    [[np.array([1]), np.array([0])], np.array([2, 3])],
    [[np.array([1]), np.array([0]), np.array([0])], np.array([2, 3, 4])],
]

coord_2 = [
    [[np.array([1, 0])], np.array([42])],
    [[np.array([1, 0]), np.array([0, 3])], np.array([2, 3])],
    [  # Array with two points that have similar coordinates, but with the axis swapped.
        # An error from here would indicate confusion of axis, possibly involving a sort
        # along dimension 0 of the coordinate axis.
        [np.array([1, 0]), np.array([0, 1])],
        np.array([2, 3]),
    ],
    [  # Array with both a true duplicate and equal coordinates but swapped axes.
        [np.array([1, 0]), np.array([0, 1]), np.array([0, 1])],
        np.array([2, 3, 4]),
    ],
]

coord_3 = [
    [[np.array([1, 0, 2])], np.array([42])],
    [[np.array([1, 0, 1]), np.array([0, 1, 3])], np.array([2, 3])],
    [  # Array with two points that have similar coordinates, but with the axis swapped.
        # An error from here would indicate confusion of axis, possibly involving a sort
        # along dimension 0 of the coordinate axis.
        [np.array([1, 0, 1]), np.array([0, 1, 1])],
        np.array([2, 3]),
    ],
    [  # Array with both a true duplicate and equal coordinates but swapped axes.
        [np.array([1, 0, 1]), np.array([0, 1, 1]), np.array([0, 1, 1])],
        np.array([2, 3, 4]),
    ],
]
# An array with 2d values.
coord_2d_values = [
    [[np.array([0]), np.array([1]), np.array([0])], np.array([[42, 2, 3], [4, 43, 7]])]
]

coordinates = coord_1 + coord_2 + coord_3 + coord_2d_values


@pytest.mark.parametrize("coords_values", coordinates)
def test_sparse_nd_array(coords_values: list[np.ndarray]):
    """Checks of the sparse array data class.

    The following checks are done:
        1) Additive mode, add all coordinates to the array at the same time.
        2) Overwrite mode, add all coordinates to the array at the same time.
        Together these tests verify that duplicate coordinates in new data is correctly
        handled (of course, other parts of the code are also tested at the same time).
        3) Additive mode, add one data point at a time.
        4) Overwrite mode, add one data point at a time.
        Tests 3 and 4 verify that coordinates already present in the array are correctly
        handled.
        5) Check that an error message is raised when coordinates not present in the
        array are retrieved.

    """
    # Split the data and values, infer dimension from the coordinates.
    coords, values = coords_values
    dim = coords[0].size

    # Force all values to be 2d (this will be done inside the array as data is added,
    # doing it here simplifies comparison and indexing).
    values = np.atleast_2d(values)
    # Dimension of the values to be represented
    value_dim = values.shape[0]

    all_coords = np.vstack(coords).T

    _, all_2_unique, unique_2_all = np.unique(
        all_coords, return_index=True, return_inverse=True, axis=1
    )
    # Get the maximum coordinates, we will use this to create a coordinate not present
    # in the coordinate array.
    max_coords = np.max(all_coords, axis=1).reshape((-1, 1))

    # Find the expected values for the coordinates. For coordinates with duplicates,
    # this will differ between additive  and overwrite modes.
    expected_values_additive = np.zeros(values.shape)
    expected_values_overwrite = np.zeros_like(expected_values_additive)

    # Loop over all the inidivual coordinates, compare to the array of merged
    # coordinates to find one (itself) or several (if duplicates) hits. If duplicates
    # are found, we handle the overwriting or adding correctly.
    for ind, c in enumerate(coords):
        # Comparison of points.
        # The threshold for point comparison here assumes that the coordinates are all
        # integers, as should be the case for sparse arrays.
        all_occurrences = np.where(
            np.sum((c.reshape((-1, 1)) - all_coords) ** 2, axis=0) < 0.5
        )[0]
        # In additive mode, do a sum
        expected_values_additive[:, ind] = np.sum(values[:, all_occurrences], axis=1)
        # For overwrite mode, find the last occurence and use that.
        last_written = all_occurrences[-1]
        expected_values_overwrite[:, ind] = values[:, last_written]

    # All tests are now run in the same way: Make an array, add values (either in one
    # bath or one by one), retrieve the values and check that the result is as expected.

    # Additive, add all at the same time
    arr_additive = array_operations.SparseNdArray(dim, value_dim=value_dim)
    arr_additive.add(coords, values, additive=True)
    retrieved_values_additive = arr_additive.get(coords)
    assert np.allclose(retrieved_values_additive, expected_values_additive)

    # Overwrite, all at the same time
    arr_overwrite = array_operations.SparseNdArray(dim, value_dim=value_dim)
    arr_overwrite.add(coords, values)
    retrieved_values_overwrite = arr_overwrite.get(coords)
    assert np.allclose(retrieved_values_overwrite, expected_values_overwrite)

    # Additive, add one by one
    arr_additive_incremental = array_operations.SparseNdArray(dim, value_dim=value_dim)
    for ind, coord in enumerate(coords):
        arr_additive_incremental.add(
            [coord], values[:, ind].reshape((-1, 1)), additive=True
        )
    retrieved_values_additive_incremental = arr_additive_incremental.get(coords)
    # The expeted value is the same as when all coordinates are added at the same time
    assert np.allclose(retrieved_values_additive_incremental, expected_values_additive)

    # Overwrite, one at a time
    arr_overwrite_incremental = array_operations.SparseNdArray(dim, value_dim=value_dim)
    for ind, coord in enumerate(coords):
        arr_overwrite_incremental.add(
            [coord], values[:, ind].reshape((-1, 1)), additive=False
        )
    retrieved_values_overwrite_incremental = arr_overwrite_incremental.get(coords)
    assert np.allclose(
        retrieved_values_overwrite_incremental, expected_values_overwrite
    )

    # Final test: Try to access a coordiinates that has not been added, this should
    # raise a ValueError.
    with pytest.raises(ValueError):
        _ = arr_overwrite.get(2 * max_coords)


# Data for intersection between sets
data_intersect_test = [
    # Non-intersecting 1d
    [
        np.array([1]),
        np.array([2]),
        0.1,
        np.array([], dtype=int),
        np.array([], dtype=int),
        [[]],
    ],
    # Intersecting 1d
    [np.array([1, 2]), np.array([2]), 0.1, np.array([1]), np.array([0]), [[], [0]]],
    # Complex intersection 1d
    [
        np.array([1, 2]),
        np.array([2, 3, 1]),
        0.1,
        np.array([0, 1]),
        np.array([0, 2]),
        [[2], [], [0]],
    ],
    # Intersecting because of high tolerance
    [np.array([1]), np.array([2]), 1.5, np.array([0]), np.array([0]), [[0]]],
    # Non-intersecting 2d, single dimension
    [
        np.array([[1]]),
        np.array([[2]]),
        0.1,
        np.array([], dtype=int),
        np.array([], dtype=int),
        [[]],
    ],
    # Non-intersecting 2d, several dimensions and items
    [
        np.array([[1], [2]]),
        np.array([[3], [4]]),
        0.1,
        np.array([], dtype=int),
        np.array([], dtype=int),
        [[]],
    ],
    # Simplest intersecting in 2d
    [
        np.array([[1], [2]]),
        np.array([[1], [2]]),
        0.1,
        np.array([0]),
        np.array([0]),
        [[0]],
    ],
    # Same numbers, but rows switched. Failure signifies sorting along 0-axis
    [
        np.array([[1], [2]]),
        np.array([[2], [1]]),
        0.1,
        np.array([], dtype=int),
        np.array([], dtype=int),
        [[]],
    ],
    # All a are in b, but not in reverse. The columns in b are permuted relative to a.
    [
        np.array([[1, 3], [2, 4]]),
        np.array([[3, 5, 1], [4, 6, 2]]),
        0.1,
        np.array([0, 1]),
        np.array([0, 2]),
        [[2], [0]],
    ],
    # elements a[:, 0] and a[:, 2] both correspond to b[:, 2].
    # Similarly, b[:, 0] and b[:, 1] correspond to a[:, 1].
    [
        np.array([[1, 3, 1], [2, 4, 2]]),
        np.array([[3, 3, 1], [4, 4, 2]]),
        0.1,
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
        [[2], [0, 1], [2]],
    ],
]


@pytest.mark.parametrize("data", data_intersect_test)
def test_intersect_sets(data):
    """Test intersection of sets.

    The setup assumes the provided data contains both sets to be
    intersected and the known values that should result.

    """
    a, b, tol, ia_known, ib_known, a_2_b_known = data

    ia, ib, a_in_b, a_2_b = array_operations.intersect_sets(a, b, tol)

    assert np.allclose(ia, np.sort(ia_known))
    assert np.allclose(ib, np.sort(ib_known))

    # The known values from a_in_b must be inferred from the known values.
    a_in_b_known = np.zeros(a.shape[-1])
    a_in_b_known[ia_known] = True
    assert np.allclose(a_in_b, a_in_b_known)

    # a_2_b is a list of list, check one by one.
    for e1, e2 in zip(a_2_b, a_2_b_known):
        assert np.allclose(e1, e2)
