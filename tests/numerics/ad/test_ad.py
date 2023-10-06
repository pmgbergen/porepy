"""Tests of grid_operator classes, currently coveringe
"""

import numpy as np
import scipy.sparse as sps

from porepy.numerics.ad._ad_utils import concatenate_ad_arrays
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays
from porepy.numerics.ad.functions import exp


def test_quadratic_function():
    x, y = initAdArrays([np.array([1]), np.array([2])])
    z = 1 * x + 2 * y + 3 * x * y + 4 * x * x + 5 * y * y
    val = 35
    assert (z.val == val and np.all(z.jac.A == [15, 25]))

def test_vector_quadratic():
    x, y = initAdArrays([np.array([1, 1]), np.array([2, 3])])
    z = 1 * x + 2 * y + 3 * x * y + 4 * x * x + 5 * y * y
    val = np.array([35, 65])
    J = np.array([[15, 0, 25, 0], [0, 18, 0, 35]])

    assert (np.all(z.val == val) and np.sum(z.jac != J) == 0)

def test_mapping_m_to_n():
    x, y = initAdArrays([np.array([1, 1, 3]), np.array([2, 3])])
    A = sps.csc_matrix(np.array([[1, 2, 1], [2, 3, 4]]))

    z = y * (A @ x)
    val = np.array([12, 51])
    J = np.array([[2, 4, 2, 6, 0], [6, 9, 12, 0, 17]])

    assert (np.all(z.val == val) and np.sum(z.jac != J) == 0)

def test_merge_two_equations():
    x, y = initAdArrays([np.array([1]), np.array([2])])
    z1 = exp(x) + y
    z2 = exp(y) + x

    z = concatenate_ad_arrays((z1, z2))

    val = np.array([np.exp(1) + 2, np.exp(2) + 1])
    J = np.array([[np.exp(1), 1], [1, np.exp(2)]])

    assert (np.allclose(z.val, val) and np.allclose(z.jac.A, J))
