""" """

from __future__ import annotations

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.arrays import compare_arrays


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
    arr_additive = pp.array_operations.SparseNdArray(dim, value_dim=value_dim)
    arr_additive.add(coords, values, additive=True)
    retrieved_values_additive = arr_additive.get(coords)
    assert np.allclose(retrieved_values_additive, expected_values_additive)

    # Overwrite, all at the same time
    arr_overwrite = pp.array_operations.SparseNdArray(dim, value_dim=value_dim)
    arr_overwrite.add(coords, values)
    retrieved_values_overwrite = arr_overwrite.get(coords)
    assert np.allclose(retrieved_values_overwrite, expected_values_overwrite)

    # Additive, add one by one
    arr_additive_incremental = pp.array_operations.SparseNdArray(
        dim, value_dim=value_dim
    )
    for ind, coord in enumerate(coords):
        arr_additive_incremental.add(
            [coord], values[:, ind].reshape((-1, 1)), additive=True
        )
    retrieved_values_additive_incremental = arr_additive_incremental.get(coords)
    # The expeted value is the same as when all coordinates are added at the same time
    assert np.allclose(retrieved_values_additive_incremental, expected_values_additive)

    # Overwrite, one at a time
    arr_overwrite_incremental = pp.array_operations.SparseNdArray(
        dim, value_dim=value_dim
    )
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

    ia, ib, a_in_b, a_2_b = pp.array_operations.intersect_sets(a, b, tol)

    assert np.allclose(ia, np.sort(ia_known))
    assert np.allclose(ib, np.sort(ib_known))

    # The known values from a_in_b must be inferred from the known values.
    a_in_b_known = np.zeros(a.shape[-1])
    a_in_b_known[ia_known] = True
    assert np.allclose(a_in_b, a_in_b_known)

    # a_2_b is a list of list, check one by one.
    for e1, e2 in zip(a_2_b, a_2_b_known):
        assert np.allclose(e1, e2)


@pytest.mark.parametrize(
    "params",
    [
        # Array with only unique rows.
        {
            "input": np.array(
                [
                    [3, 1, 20, 19],
                    [12, 15, 22, 5],
                    [20, 21, 14, 1],
                    [1, 12, 15, 20],
                ]
            ),
            # Note different ordering in the output array vs. the input array. TODO:
            # Include preserve_order or not? This also affects the unique_columns test,
            # which does have a preserve_order argument.
            "expected": (
                np.array(
                    [
                        [1, 12, 15, 20],
                        [3, 1, 20, 19],
                        [12, 15, 22, 5],
                        [20, 21, 14, 1],
                    ]
                ),
                np.array([3, 0, 1, 2]),
                np.array([1, 2, 3, 0]),
            ),
        },
        # Array with duplicate rows.
        {
            "input": np.array(
                [
                    [12, 15, 22, 5],
                    [3, 1, 20, 19],
                    [12, 15, 22, 5],
                    [3, 1, 20, 19],
                    [1, 12, 15, 20],
                ]
            ),
            "expected": (
                np.array(
                    [
                        [1, 12, 15, 20],
                        [3, 1, 20, 19],
                        [12, 15, 22, 5],
                    ]
                ),
                np.array([4, 1, 0]),
                np.array([2, 1, 2, 1, 0]),
            ),
        },
    ],
)
def test_unique_rows(params):
    """Tests the function unique_rows.

        Tests the following different cases:
        1) An array with only unique rows.
        2) An array with two duplicate rows.

    Parameters:
        params: Dictionary which contains two keys: "input" and "expected". The "input"
            key contains the 2D input array, while the "expected" key contains a tuple
            of the expected outputs of the function unique_columns (array of unique
            rows, mapping from new to old array and mapping from old to new array).

    """
    input_array = params["input"]
    result = pp.array_operations.unique_rows(input_array)

    assert np.all(result[0] == params["expected"][0])
    assert np.all(result[1] == params["expected"][1])
    assert np.all(result[2] == params["expected"][2])


@pytest.mark.parametrize(
    "params",
    [
        # Array with only unique columns.
        {
            "input": np.array([[16, 15, 18], [5, 16, 25]]),
            "expected": (
                np.array([[16, 15, 18], [5, 16, 25]]),
                np.array([0, 1, 2]),
                np.array([0, 1, 2]),
            ),
        },
        # Array with one duplicate column.
        {
            "input": np.array([[3, 13, 15, 3, 12, 13], [15, 13, 14, 15, 21, 14]]),
            "expected": (
                np.array([[3, 13, 15, 12, 13], [15, 13, 14, 21, 14]]),
                np.array([0, 1, 2, 4, 5]),
                np.array([0, 1, 2, 0, 3, 4]),
            ),
        },
        # Array with several duplicate columns.
        {
            "input": np.array([[1, 2, 3, 1, 3, 4], [2, 3, 4, 2, 4, 5]]),
            "expected": (
                np.array([[1, 2, 3, 4], [2, 3, 4, 5]]),
                np.array([0, 1, 2, 5]),
                np.array([0, 1, 2, 0, 2, 3]),
            ),
        },
        # Edge case: no points.
        {"input": np.array([]), "expected": (np.array([]), np.array([]), np.array([]))},
    ],
)
def test_unique_columns(params):
    """Tests the function unique_columns.

    Tests the following different cases:
        1) An array with only unique columns.
        2) An array with one duplicate column.
        3) An array with two duplicate columns.

    Parameters:
        params: Dictionary which contains two keys: "input" and "expected". The "input"
            key contains the 2D input array, while the "expected" key contains a tuple
            of the expected outputs of the function unique_columns (array of unique
            columns, mapping from new to old array and mapping from old to new array).

    """
    input_array = params["input"]
    result_ord = pp.array_operations.unique_columns(input_array, preserve_order=True)

    assert np.all(result_ord[0] == params["expected"][0])
    assert np.all(result_ord[1] == params["expected"][1])
    assert np.all(result_ord[2] == params["expected"][2])

    # Test that unique_vectors_tol produce the same result.
    result_tol = pp.array_operations.unique_vectors_tol(input_array, tol=0.1)
    assert np.all(result_tol[0] == params["expected"][0])
    assert np.all(result_tol[1] == params["expected"][1])
    assert np.all(result_tol[2] == params["expected"][2])

    # Test that unordered produces the same result up to some permutation.
    result_unord = pp.array_operations.unique_columns(input_array, preserve_order=False)
    unique_unord, new_2_old_unord, old_2_new_unord = result_unord
    assert compare_arrays(result_unord[0], params["expected"][0])
    assert np.all(input_array[:, new_2_old_unord] == unique_unord)
    assert compare_arrays(input_array[:, new_2_old_unord], params["expected"][1])
    assert np.all(unique_unord[:, old_2_new_unord], input_array)


@pytest.mark.parametrize(
    "a, b, ma_known, ia_known, sort",
    [
        (
            np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]]),
            np.array([[3, 1, 3, 5, 3], [3, 3, 2, 1, 2]]),
            np.array([1, 1, 1, 1, 0], dtype=bool),
            np.array([1, 0, 2, 1]),
            True,
        ),
        # unequal sizes
        (
            np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]]),
            np.array([[3, 1, 2, 5], [3, 3, 3, 1]]),
            np.array([1, 1, 1, 1, 0], dtype=bool),
            np.array([1, 0, 2, 1]),
            True,
        ),
        (
            np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]]),
            np.array([[3, 1, 2, 5, 3, 4, 7], [3, 3, 3, 1, 9, 9, 9]]),
            np.array([1, 1, 1, 1, 0], dtype=bool),
            np.array([1, 0, 2, 1]),
            True,
        ),
        # double occurences
        (
            np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]]),
            np.array([[3, 2, 5], [3, 3, 1]]),
            np.array([0, 1, 1, 0, 0], dtype=bool),
            np.array([0, 1]),
            True,
        ),
        (
            np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]]),
            np.array([[3, 1, 2, 5, 3], [3, 3, 3, 1, 1]]),
            np.array([1, 1, 1, 1, 0], dtype=bool),
            np.array([1, 0, 2, 1]),
            True,
        ),
        # 1D
        (
            np.array([0, 2, 1, 3, 0]),
            np.array([2, 4, 3]),
            np.array([0, 1, 0, 1, 0], dtype=bool),
            np.array([0, 2]),
            True,
        ),
        (
            np.array([0, 2, 1, 13, 0]),
            np.array([2, 4, 13, 0]),
            np.array([1, 1, 0, 1, 1], dtype=bool),
            np.array([3, 0, 2, 3]),
            True,
        ),
        # test issue #123
        (
            np.array(
                [
                    [0.0, 1.0, 1.0, 0.0, 0.2, 0.8, 0.5, 0.5, 0.5],
                    [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.8, 0.2, 0.5],
                ]
            ),
            np.array([[0, 1, 1, 0], [0, 0, 1, 1]]),
            np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool),
            np.array([0, 1, 2, 1]),
            True,
        ),
        # Test no sorting:
        (
            np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]]),
            np.array([[3, 1, 2, 5, 3], [3, 3, 3, 1, 1]]),
            np.array([1, 1, 0, 1, 0], dtype=bool),
            np.array([1, 0, 1]),
            False,
        ),
    ],
)
def test_ismember_rows(a, b, ma_known, ia_known, sort):
    ma, ia = pp.array_operations.ismember_rows(a, b, sort=sort)
    assert np.allclose(ma, ma_known)
    assert np.allclose(ia, ia_known)


@pytest.mark.parametrize(
    "params",
    [
        # Expand indices with C order.
        {
            "input": ([0, 1, 3], 3, "C"),
            "expected": [0, 3, 9, 1, 4, 10, 2, 5, 11],
        },
        # Expand indices with non-specified order (should correspond to "F", which is
        # default.)
        {
            "input": ([0, 1, 3], 3, None),
            "expected": [0, 1, 2, 3, 4, 5, 9, 10, 11],
        },
        # Array with indices that are not sorted.
        {
            "input": ([9, 14, 7, 18, 9, 4], 2, "F"),
            "expected": [18, 19, 28, 29, 14, 15, 36, 37, 18, 19, 8, 9],
        },
        # Array with decreasing indices.
        {
            "input": ([25, 21, 18, 1], 3, "F"),
            "expected": [75, 76, 77, 63, 64, 65, 54, 55, 56, 3, 4, 5],
        },
    ],
)
def test_expand_indices_nd(params):
    """Tests the function expand_indices_nd.

    Tests the following different cases:
        1) Expand array with C order.
        2) Expand the array with the default order ("F").
        3) Expand array with indices which are not sorted.
        4) Expand array with decreasing indices.

    Parameters:
        params: Dictionary which contains two keys: "input" and "expected". The "input"
        is a tuple containing the array we want to expand, the dimension we want to
        expand it with and the order (Fortran or C). "expected" contains the array which
        is expected to be returned by the function.

    """
    inputs = params.get("input")
    ind = np.array(inputs[0])
    nd = inputs[1]
    order = params["input"][2]
    if order is None:
        result = pp.array_operations.expand_indices_nd(ind=ind, nd=nd)
    else:
        result = pp.array_operations.expand_indices_nd(ind=ind, nd=nd, order=order)
    expected = np.array(params["expected"])

    assert np.all(result == expected)


@pytest.mark.parametrize(
    "params",
    [
        # Repeat array 3 times and add increment of 200.
        {
            "input": ([0, 1, 3], 3, 200),
            "expected": [0, 200, 400, 1, 201, 401, 3, 203, 403],
        },
        # No increment is added. The original array is thus just expanded.
        {
            "input": ([0, 1, 3], 4, 0),
            "expected": [0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3],
        },
        # Repeat only once, but add increment. This should result in the original array.
        {
            "input": ([0, 1, 3], 1, 5),
            "expected": [0, 1, 3],
        },
    ],
)
def test_expand_indices_add_increment(params):
    """Tests the function expand_indices_add_increment.

    Tests the following different cases:
        1) Repeat/Expand array three times with an added increment value of 200 for each
           repetition.
        2) Repeating/Expanding array with a zero increment value.
        3) Repeating an array only once.

    Parameters:
        params: Dictionary which contains two keys: "input" and "expected". The "input"
        is a tuple containing the array we want to expand, the dimension we want to
        expand it with and the order (Fortran or C). The key "expected" contains the
        array which we expect to be returned from the function.

    """
    input_array = np.array(params["input"][0])
    n = params["input"][1]
    increment = params["input"][2]

    result = pp.array_operations.expand_indices_add_increment(
        x=input_array, n=n, increment=increment
    )
    expected = np.array(params["expected"])

    assert np.all(result == expected)


@pytest.mark.parametrize(
    "params",
    [
        # Simple comparison of input (2 arrays) and expected output of expand_index_pointers.
        {
            "input": ([1, 2], [3, 4]),
            "expected": [1, 2, 2, 3],
        },
        # Zero size result.
        {
            "input": ([1, 2], [1, 2]),
            "expected": [],
        },
        # Equal middle point. Motivated by GitHub issue #11.
        {
            "input": ([1, 5, 5], [5, 5, 6]),
            "expected": [1, 2, 3, 4, 5],
        },
        # Equal last point. Motivated by GitHub issue #11.
        {
            "input": ([1, 5], [5, 5]),
            "expected": [1, 2, 3, 4],
        },
        # Different sizes of high and low bounds.
        {
            "input": ([1, 2], [2, 3, 4]),
            "raises": True,
        },
        # Test type conversion.
        {
            "input": (
                np.array([1, 3], dtype=np.int8),
                np.array([2, 4], dtype=np.int16),
            ),
            "expected_dtype": int,
        },
    ],
)
def test_expand_index_pointers(params: dict):
    a = np.array(params["input"][0])
    b = np.array(params["input"][1])

    if params.get("raises", False):
        with pytest.raises(ValueError):
            pp.array_operations.expand_index_pointers(a, b)
        return
    result = pp.array_operations.expand_index_pointers(a, b)

    expected = np.array(params["expected"]) if params.get("expected") else None
    if expected is not None:
        assert np.all(result == expected)

    expected_dtype = params.get("expected_dtype")
    if expected_dtype is not None:
        assert result.dtype == expected_dtype


@pytest.mark.parametrize(
    "params",
    [
        # Test that the ordering is preserved and some non-unique values are filtered.
        {
            "input": [[1, 1], [0, 0], [0.5, 0], [0, 0], [0, 0.5], [0, 0], [0.5, 0]],
            "tol": 1e-2,
            "expected_unique": [[1, 1], [0, 0], [0.5, 0], [0, 0.5]],
        },
        # Edge case for the values close to the tolerance.
        {
            "input": np.array([np.linspace(0, 1, num=101)[::-1], [0] * 101]).T,
            "tol": 1e-1,
            "expected_unique": [
                [1, 0.97, 0.86, 0.75, 0.64, 0.53, 0.43, 0.32, 0.21, 0.1],
                [0] * 10,
            ],
        },
    ],
)
def test_unique_vectors_tol(params: dict):
    input = np.array(params["input"]).T
    tol = params["tol"]
    unique_vectors, new_2_old, old_2_new = pp.array_operations.unique_vectors_tol(
        vectors=input, tol=tol
    )
    expected_unique = np.array(params["expected_unique"]).T
    assert np.allclose(unique_vectors, expected_unique, atol=tol)
    assert np.allclose(input[:, new_2_old], expected_unique, atol=tol)
    assert np.allclose(unique_vectors[:, old_2_new], input, atol=tol)
