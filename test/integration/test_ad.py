import numpy as np
import scipy.sparse as sps
import unittest

from porepy.ad.forward_mode import Ad_array, initAdArrays
from porepy.ad.utils import concatenate
from porepy.ad.functions import exp


class AdTest(unittest.TestCase):
    def test_quadratic_function(self):
        x, y = initAdArrays([np.array(1), np.array(2)])
        z = 1 * x + 2 * y + 3 * x * y + 4 * x * x + 5 * y * y
        val = 35
        assert z.val == val and np.all(z.jac.A == [15, 25])

    def test_vector_quadratic(self):
        x, y = initAdArrays([np.array([1, 1]), np.array([2, 3])])
        z = 1 * x + 2 * y + 3 * x * y + 4 * x * x + 5 * y * y
        val = np.array([35, 65])
        J = np.array([[15, 0, 25, 0], [0, 18, 0, 35]])

        assert np.all(z.val == val) and np.sum(z.full_jac() != J) == 0

    def test_mapping_m_to_n(self):
        x, y = initAdArrays([np.array([1, 1, 3]), np.array([2, 3])])
        A = sps.csc_matrix(np.array([[1, 2, 1], [2, 3, 4]]))

        z = y * (A * x)
        val = np.array([12, 51])
        J = np.array([[2, 4, 2, 6, 0], [6, 9, 12, 0, 17]])

        assert np.all(z.val == val) and np.sum(z.full_jac() != J) == 0

    def test_merge_two_equations(self):
        x, y = initAdArrays([np.array([1]), np.array([2])])
        z1 = exp(x) + y
        z2 = exp(y) + x

        z = concatenate((z1, z2))

        val = np.array([np.exp(1) + 2, np.exp(2) + 1])
        J = np.array([[np.exp(1), 1], [1, np.exp(2)]])

        assert np.allclose(z.val, val) and np.allclose(z.full_jac().A, J)
