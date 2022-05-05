"""Module contains tests of arithmetic operations on forward Ad arrays.

The tests cover initiation of Ad_array (both of single arrays and joint initiation of
multiple dependent variables). The test also partly cover the arithmetic operations
implemented for Ad_arrays, e.g., __add__, __sub__, etc., however, coverage of these
is only partial at the moment.
"""
import unittest
import warnings

import numpy as np
import scipy.sparse as sps

from porepy.numerics.ad.forward_mode import initAdArrays
from porepy.numerics.ad import functions as af

warnings.simplefilter("ignore", sps.SparseEfficiencyWarning)


class AdInitTest(unittest.TestCase):
    def _compare(self, arr, known_val, known_jac):
        self.assertTrue(np.allclose(arr.val, known_val))
        self.assertTrue(np.allclose(arr.jac.A, known_jac))

    def test_add_two_ad_variables_init(self):
        a, b = initAdArrays([np.array(1), np.array(-10)])
        c = a + b
        self.assertTrue(c.val == -9 and np.all(c.jac.A == [1, 1]))
        self.assertTrue(a.val == 1 and np.all(a.jac.A == [1, 0]))
        self.assertTrue(b.val == -10 and np.all(b.jac.A == [0, 1]))

    def test_add_var_init_with_scal(self):
        a = initAdArrays(3)
        b = 3
        c = a + b
        self.assertTrue(np.allclose(c.val, 6) and np.allclose(c.jac.A, 1))
        self.assertTrue(a.val == 3 and np.allclose(a.jac.A, 1))
        self.assertTrue(b == 3)

    def test_sub_scal_with_var_init(self):
        a = initAdArrays(3)
        b = 3
        c = b - a
        self.assertTrue(np.allclose(c.val, 0) and np.allclose(c.jac.A, -1))
        self.assertTrue(a.val == 3 and a.jac.A == 1)
        self.assertTrue(b == 3)

    def test_sub_var_init_with_var_init(self):
        a, b = initAdArrays([np.array(3), np.array(2)])
        c = b - a
        self.assertTrue(np.allclose(c.val, -1) and np.all(c.jac.A == [-1, 1]))
        self.assertTrue(a.val == 3 and np.all(a.jac.A == [1, 0]))
        self.assertTrue(b.val == 2 and np.all(b.jac.A == [0, 1]))

    def test_add_scal_with_var_init(self):
        a = initAdArrays(3)
        b = 3
        c = b + a
        self.assertTrue(np.allclose(c.val, 6) and np.allclose(c.jac.A, 1))
        self.assertTrue(a.val == 3 and np.allclose(a.jac.A, 1))
        self.assertTrue(b == 3)

    def test_mul_ad_var_init(self):
        a, b = initAdArrays([np.array(3), np.array(2)])
        c = a * b
        self.assertTrue(c.val == 6 and np.all(c.jac.A == [2, 3]))
        self.assertTrue(a.val == 3 and np.all(a.jac.A == [1, 0]))
        self.assertTrue(b.val == 2 and np.all(b.jac.A == [0, 1]))

    def test_mul_scal_ad_var_init(self):
        a, b = initAdArrays([np.array(3), np.array(2)])
        d = 3
        c = d * a
        self.assertTrue(c.val == 9 and np.all(c.jac.A == [3, 0]))
        self.assertTrue(a.val == 3 and np.all(a.jac.A == [1, 0]))
        self.assertTrue(b.val == 2 and np.all(b.jac.A == [0, 1]))

    def test_mul_sps_advar_init(self):
        x = initAdArrays(np.array([1, 2, 3]))
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        f = A * x
        self.assertTrue(np.all(f.val == [14, 32, 50]))
        self.assertTrue(np.all((f.jac == A).A))

    def test_advar_init_diff_len(self):
        a, b = initAdArrays([np.array([1, 2, 3]), np.array([1, 2])])
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        B = sps.csc_matrix(np.array([[1, 2], [4, 5]]))

        f = A * a
        g = B * b
        zero_32 = sps.csc_matrix((3, 2))
        zero_23 = sps.csc_matrix((2, 3))

        jac_f = sps.hstack((A, zero_32))
        jac_g = sps.hstack((zero_23, B))
        self.assertTrue(np.all(f.val == [14, 32, 50]))
        self.assertTrue(np.all((f.jac == jac_f).A))
        self.assertTrue(np.all(g.val == [5, 14]))
        self.assertTrue(np.all((g.jac == jac_g).A))

    def test_advar_init_cross_jacobi(self):
        x, y = initAdArrays([np.array([-1, 4]), np.array([1, 5])])

        z = x * y
        J = np.array([[1, 0, -1, 0], [0, 5, 0, 4]])
        self.assertTrue(np.all(z.val == [-1, 20]))
        self.assertTrue(np.all((z.jac == J).A))

    def test_exp_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a, _, _ = initAdArrays([val, val, val])
        c = 2
        b = af.exp(c * a)

        zero = sps.csc_matrix((3, 3))
        jac = sps.hstack([c * sps.diags(np.exp(c * val)) * J, zero, zero])
        jac_a = sps.hstack([J, zero, zero])
        self.assertTrue(
            np.allclose(b.val, np.exp(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == jac_a.A))

    def test_pow(self):
        # Tests of Ad_array.__pow__()

        # First run an initial test that covers the basic arithmetics of powers.
        # This is sort of a legacy test.
        a, b = initAdArrays([np.array(4.0), np.array(-8)])

        c = a**b
        jac = np.array([-8 * (4**-9), 4**-8 * np.log(4)])

        self.assertTrue(np.allclose(c.val, 4**-8))
        self.assertTrue(np.all(np.abs(c.jac.A - jac) < 1e-6))

        # Now, systematic tests of arrays with different data formats.
        # The tests raises Ad arrays to powers of Ad_arrays, numpy arrays and scalars.
        # Both ints and floats are considered, since the former requires some special
        # treatment in the implementation of __pow__() (it seems this can be traced back
        # to the underlying implementation in numpy).

        # Numpy arrays, int and float versions of the same numbers
        array_int = np.array([2, 3])
        array_float = np.array([2, 3], dtype=float)
        # Ad_array versions of the numpy arrays
        ad_int = initAdArrays(array_int)
        ad_float = initAdArrays(array_float)
        # Also some scalars
        scalar_int = 2
        scalar_float = 2.0

        # First, raise arrays to the power of positive scalars
        known_val = np.array([4, 9])
        # The derivatives will be (x**2)' = 2x
        known_jac = np.array([[4, 0], [0, 6]])
        self._compare(ad_int**scalar_int, known_val, known_jac)
        self._compare(ad_int**scalar_float, known_val, known_jac)
        self._compare(ad_float**scalar_int, known_val, known_jac)
        self._compare(ad_float**scalar_float, known_val, known_jac)

        # Raise arrays to the power of negative scalars. This will trigger some special
        # handling in __pow__().
        known_val = np.array([1 / 4, 1 / 9])
        # The derivatives will be (x^-2)' = -2/x^3
        known_jac = np.array([[-1 / 4, 0], [0, -2 / 27]])
        self._compare(ad_int ** (-scalar_int), known_val, known_jac)
        self._compare(ad_int ** (-scalar_float), known_val, known_jac)
        self._compare(ad_float ** (-scalar_int), known_val, known_jac)
        self._compare(ad_float ** (-scalar_float), known_val, known_jac)

        # Next raise the Ad arrays to other arrays of positive integers
        known_val = np.array([4, 27])
        # The derivatives are of the form n x ^(n-1)
        known_jac = np.array([[4, 0], [0, 27]])
        self._compare(ad_int**array_int, known_val, known_jac)
        self._compare(ad_int**array_float, known_val, known_jac)
        self._compare(ad_float**array_int, known_val, known_jac)
        self._compare(ad_float**array_float, known_val, known_jac)

        # Raise an Ad_array to the negative power of numpy arrays
        # Next raise the Ad arrays to other arrays of positive integers
        known_val = np.array([1 / 4, 1 / 27])
        # The derivatives of x^-n = -nx^(-n-1)
        known_jac = np.array([[-1 / 4, 0], [0, -1 / 27]])
        self._compare(ad_int ** (-array_int), known_val, known_jac)
        self._compare(ad_int ** (-array_float), known_val, known_jac)
        self._compare(ad_float ** (-array_int), known_val, known_jac)
        self._compare(ad_float ** (-array_float), known_val, known_jac)

        # Finally, raise an array to the power of another array.
        # For this, create new arrays with coupled derivatives.
        # Also, test only int-int and float-float combinations (the expected problem
        # is powers that are negative integers, which is well covered).
        a_int, b_int = initAdArrays([np.array([2]), np.array([3])])
        a_float, b_float = initAdArrays([np.array([2]), np.array([3])])

        # First positive powers
        known_val = np.array([2**3])
        known_jac = np.array([[3 * 2 ** (3 - 1), 2**3 * np.log(2)]])
        self._compare(a_int**b_int, known_val, known_jac)
        self._compare(a_float**b_float, known_val, known_jac)

        # Then negative
        known_val = np.array([2 ** (-3)])
        known_jac = np.array([[-3 * 2 ** (-3 - 1), 2 ** (-3) * np.log(2) * (-1)]])
        self._compare(a_int ** (-b_int), known_val, known_jac)
        self._compare(a_float ** (-b_float), known_val, known_jac)

    def test_truediv(self):
        # Tests of Ad_array.__truediv__().
        # Division, a/b, is implemented as a* (b**-1), and, since numpy has some rules
        # for not taking negative integer powers, we need to check that various special
        # cases have beed correctly dealt with.

        val_int = np.array([2])
        val_float = np.array([4]).astype(float)

        a_int, a_float = initAdArrays([val_int, val_float])

        scalar_int = 2
        scalar_float = 2.0

        # First combine Ad arrays with scalars
        self._compare(a_int / scalar_int, np.array([1]), np.array([[1 / 2, 0]]))
        self._compare(a_int / scalar_float, np.array([1]), np.array([[1 / 2, 0]]))

        # Divide Ad arrays by numpy arrays
        self._compare(a_int / val_int, np.array([1]), np.array([[1 / 2, 0]]))
        self._compare(a_int / val_float, np.array([1 / 2]), np.array([[1 / 4, 0]]))
        self._compare(a_float / val_int, np.array([2]), np.array([[0, 1 / 2]]))
        self._compare(a_float / val_float, np.array([1]), np.array([[0, 1 / 4]]))

        # Finally, divide two Ad_arrays
        self._compare(a_int / a_float, np.array([1 / 2]), np.array([[1 / 4, -1 / 8]]))
        self._compare(a_float / a_int, np.array([2]), np.array([[-1, 1 / 2]]))
