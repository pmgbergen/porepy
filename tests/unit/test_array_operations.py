"""

"""
import pytest
import numpy as np

import porepy as pp

# Data for test of intersection between sets
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

    a, b, tol, ia_known, ib_known, a_2_b_known = data

    ia, ib, a_in_b, a_2_b = pp.array_operations.intersect_sets(a, b, tol)

    assert np.allclose(ia, np.sort(ia_known))
    assert np.allclose(ib, np.sort(ib_known))
    a_in_b_known = np.zeros(a.shape[-1])
    a_in_b_known[ia_known] = True

    assert np.allclose(a_in_b, a_in_b_known)

    for (e1, e2) in zip(a_2_b, a_2_b_known):
        assert np.allclose(e1, e2)
