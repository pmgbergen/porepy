"""Tests of the forward part of the AD framework, that is Ad_arrays and operations on these.

The tests are grouped into three categories (using unittest classes for historical reasons):
    AdInitTests: tests of the initialization of the Ad arrays.
    AdArrays: Test fundamental operation on Ad_arrays, such as addition, multiplication, etc.
    AdFunctions: Tests of the functions that are defined on Ad_arrays, such as exp, log, etc.

"""
import unittest
import warnings

import numpy as np
import scipy.sparse as sps

from porepy.numerics.ad import functions as af
from porepy.numerics.ad import Ad_array, initAdArrays


warnings.simplefilter("ignore", sps.SparseEfficiencyWarning)

class AdInitTest(unittest.TestCase):
    """
    The tests cover initiation of Ad_array (both of single arrays and joint initiation of
    multiple dependent variables). The test also partly cover the arithmetic operations
    implemented for Ad_arrays, e.g., __add__, __sub__, etc., however, coverage of these
    is only partial at the moment.
    """
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

class AdArrays(unittest.TestCase):
    """Tests for the implementation of the main Ad array class,
    that is, the functionality needed for the forward Ad operations.
    """

    def test_add_two_scalars(self):
        a = Ad_array(1, 0)
        b = Ad_array(-10, 0)
        c = a + b
        self.assertTrue(c.val == -9 and c.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)
        self.assertTrue(b.val == -10 and b.jac == 0)

    def test_add_two_ad_variables(self):
        a = Ad_array(4, 1.0)
        b = Ad_array(9, 3)
        c = a + b
        self.assertTrue(np.allclose(c.val, 13) and np.allclose(c.jac, 4.0))
        self.assertTrue(a.val == 4 and np.allclose(a.jac, 1.0))
        self.assertTrue(b.val == 9 and b.jac == 3)

    def test_add_var_with_scal(self):
        a = Ad_array(3, 2)
        b = 3
        c = a + b
        self.assertTrue(np.allclose(c.val, 6) and np.allclose(c.jac, 2))
        self.assertTrue(a.val == 3 and np.allclose(a.jac, 2))
        self.assertTrue(b == 3)

    def test_add_scal_with_var(self):
        a = Ad_array(3, 2)
        b = 3
        c = b + a
        self.assertTrue(np.allclose(c.val, 6) and np.allclose(c.jac, 2))
        self.assertTrue(a.val == 3 and a.jac == 2)
        self.assertTrue(b == 3)

    def test_sub_two_scalars(self):
        a = Ad_array(1, 0)
        b = Ad_array(3, 0)
        c = a - b
        self.assertTrue(c.val == -2 and c.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)
        self.assertTrue(b.val == 3 and a.jac == 0)

    def test_sub_two_ad_variables(self):
        a = Ad_array(4, 1.0)
        b = Ad_array(9, 3)
        c = a - b
        self.assertTrue(np.allclose(c.val, -5) and np.allclose(c.jac, -2))
        self.assertTrue(a.val == 4 and np.allclose(a.jac, 1.0))
        self.assertTrue(b.val == 9 and b.jac == 3)

    def test_sub_var_with_scal(self):
        a = Ad_array(3, 2)
        b = 3
        c = a - b
        self.assertTrue(np.allclose(c.val, 0) and np.allclose(c.jac, 2))
        self.assertTrue(a.val == 3 and a.jac == 2)
        self.assertTrue(b == 3)

    def test_sub_scal_with_var(self):
        a = Ad_array(3, 2)
        b = 3
        c = b - a
        self.assertTrue(np.allclose(c.val, 0) and np.allclose(c.jac, -2))
        self.assertTrue(a.val == 3 and a.jac == 2)
        self.assertTrue(b == 3)

    def test_mul_scal_ad_scal(self):
        a = Ad_array(3, 0)
        b = Ad_array(2, 0)
        c = a * b
        self.assertTrue(c.val == 6 and c.jac == 0)
        self.assertTrue(a.val == 3 and a.jac == 0)
        self.assertTrue(b.val == 2 and b.jac == 0)

    def test_mul_ad_var_ad_scal(self):
        a = Ad_array(3, 3)
        b = Ad_array(2, 0)
        c = a * b
        self.assertTrue(c.val == 6 and c.jac == 6)
        self.assertTrue(a.val == 3 and a.jac == 3)
        self.assertTrue(b.val == 2 and b.jac == 0)

    def test_mul_ad_var_ad_var(self):
        a = Ad_array(3, 3)
        b = Ad_array(2, -4)
        c = a * b
        self.assertTrue(c.val == 6 and c.jac == -6)
        self.assertTrue(a.val == 3 and a.jac == 3)
        self.assertTrue(b.val == 2 and b.jac == -4)

    def test_mul_ad_var_scal(self):
        a = Ad_array(3, 3)
        b = 3
        c = a * b
        self.assertTrue(c.val == 9 and c.jac == 9)
        self.assertTrue(a.val == 3 and a.jac == 3)
        self.assertTrue(b == 3)

    def test_mul_scar_ad_var(self):
        a = Ad_array(3, 3)
        b = 3
        c = b * a
        self.assertTrue(c.val == 9 and c.jac == 9)
        self.assertTrue(a.val == 3 and a.jac == 3)
        self.assertTrue(b == 3)

    def test_mul_ad_var_mat(self):
        x = Ad_array(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        f = x * A
        sol = np.array([30, 36, 42])
        jac = np.diag([3, 2, 1]) * A

        self.assertTrue(np.all(f.val == sol) and np.all(f.jac == jac))
        self.assertTrue(
            np.all(x.val == np.array([1, 2, 3])) and np.all(x.jac == np.diag([3, 2, 1]))
        )
        self.assertTrue(np.all(A == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))

    def test_advar_mul_vec(self):
        x = Ad_array(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
        A = np.array([1, 3, 10])
        f = x * A
        sol = np.array([1, 6, 30])
        jac = np.diag([3, 6, 10])

        self.assertTrue(np.all(f.val == sol) and np.all(f.jac == jac))
        self.assertTrue(
            np.all(x.val == np.array([1, 2, 3])) and np.all(x.jac == np.diag([3, 2, 1]))
        )

    def test_advar_m_mul_vec_n(self):
        x = Ad_array(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
        vec = np.array([1, 2])
        R = sps.csc_matrix(np.array([[1, 0, 1], [0, 1, 0]]))
        y = R * x
        z = y * vec
        Jy = np.array([[1, 0, 3], [0, 2, 0]])
        Jz = np.array([[1, 0, 3], [0, 4, 0]])
        self.assertTrue(np.all(y.val == [4, 2]))
        self.assertTrue(np.sum(y.full_jac().A - Jy) == 0)
        self.assertTrue(np.all(z.val == [4, 4]))
        self.assertTrue(np.sum(z.full_jac().A - Jz) == 0)

    def test_mul_sps_advar(self):
        J = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
        x = Ad_array(np.array([1, 2, 3]), J)
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        f = A * x

        self.assertTrue(np.all(f.val == [14, 32, 50]))
        self.assertTrue(np.all(f.jac == A * J.A))

    def test_mul_advar_vectors(self):
        Ja = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
        Jb = sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        a = Ad_array(np.array([1, 2, 3]), Ja)
        b = Ad_array(np.array([1, 1, 1]), Jb)
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        f = A * a + b

        self.assertTrue(np.all(f.val == [15, 33, 51]))
        self.assertTrue(np.sum(f.full_jac() != A * Ja + Jb) == 0)
        self.assertTrue(
            np.sum(Ja != sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]])))
            == 0
        )
        self.assertTrue(
            np.sum(Jb != sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
            == 0
        )

    def test_power_advar_scalar(self):
        a = Ad_array(2, 3)
        b = a**2
        self.assertTrue(b.val == 4 and b.jac == 12)

    def test_power_advar_advar(self):
        a = Ad_array(4, 4)
        b = Ad_array(-8, -12)
        c = a**b
        jac = -(2 + 3 * np.log(4)) / 16384
        self.assertTrue(np.allclose(c.val, 4**-8) and np.allclose(c.jac, jac))

    def test_rpower_advar_scalar(self):
        # Make an Ad_array with value 2 and derivative 3.
        a = Ad_array(2, 3)
        b = 2**a
        self.assertTrue(b.val == 4 and b.jac == 12 * np.log(2))

        c = 2 ** (-a)
        self.assertTrue(c.val == 1 / 4 and c.jac == 2 ** (-2) * np.log(2) * (-3))

    def test_rpower_advar_vector_scalar(self):
        J = sps.csc_matrix(np.array([[1, 2], [2, 3], [0, 1]]))
        a = Ad_array(np.array([1, 2, 3]), J)
        b = 3**a
        bJac = np.array(
            [
                [3 * np.log(3) * 1, 3 * np.log(3) * 2],
                [9 * np.log(3) * 2, 9 * np.log(3) * 3],
                [27 * np.log(3) * 0, 27 * np.log(3) * 1],
            ]
        )

        self.assertTrue(np.all(b.val == [3, 9, 27]))
        self.assertTrue(np.all(b.jac.A == bJac))

    def test_div_advar_scalar(self):
        a = Ad_array(10, 6)
        b = 2
        c = a / b
        self.assertTrue(c.val == 5, c.jac == 2)

    def test_div_advar_advar(self):
        # a = x ^ 3: b = x^2: x = 2
        a = Ad_array(8, 12)
        b = Ad_array(4, 4)
        c = a / b
        self.assertTrue(c.val == 2 and np.allclose(c.jac, 1))

    def test_full_jac(self):
        J = np.array(
            [
                [1, 3, 5, 1, 2],
                [1, 5, 1, 2, 5],
                [6, 2, 4, 6, 0],
                [2, 4, 1, 9, 9],
                [6, 2, 1, 45, 2],
            ]
        )

        a = Ad_array(np.array([1, 2, 3, 4, 5]), J.copy())  # np.array([J1, J2]))

        self.assertTrue(np.sum(a.full_jac() != J) == 0)

    def test_copy_scalar(self):
        a = Ad_array(1, 0)
        b = a.copy()
        self.assertTrue(a.val == b.val)
        self.assertTrue(a.jac == b.jac)
        a.val = 2
        a.jac = 3
        self.assertTrue(b.val == 1)
        self.assertTrue(b.jac == 0)

    def test_copy_vector(self):
        a = Ad_array(np.ones((3, 1)), np.ones((3, 1)))
        b = a.copy()
        self.assertTrue(np.allclose(a.val, b.val))
        self.assertTrue(np.allclose(a.jac, b.jac))
        a.val[0] = 3
        a.jac[2] = 4
        self.assertTrue(np.allclose(b.val, np.ones((3, 1))))
        self.assertTrue(np.allclose(b.jac, np.ones((3, 1))))

class AdFunctionTest(unittest.TestCase):

    # Function: exp
    def test_exp_scalar(self):
        a = Ad_array(1, 0)
        b = af.exp(a)
        self.assertTrue(b.val == np.exp(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_exp_advar(self):
        a = Ad_array(2, 3)
        b = af.exp(a)
        self.assertTrue(b.val == np.exp(2) and b.jac == 3 * np.exp(2))
        self.assertTrue(a.val == 2 and a.jac == 3)

    def test_exp_vector(self):
        val = np.array([1, 2, 3])
        J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.exp(a)
        jac = np.dot(np.diag(np.exp(val)), J)

        self.assertTrue(np.all(b.val == np.exp(val)) and np.all(b.jac == jac))
        self.assertTrue(np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])))

    def test_exp_sparse_jac(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.exp(a)
        jac = np.dot(np.diag(np.exp(val)), J.A)
        self.assertTrue(np.all(b.val == np.exp(val)) and np.all(b.jac == jac))

    def test_exp_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a = Ad_array(val, J)
        c = 2
        b = af.exp(c * a)
        jac = c * sps.diags(np.exp(c * val)) * J

        self.assertTrue(
            np.allclose(b.val, np.exp(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: log
    def test_log_scalar(self):
        a = Ad_array(2, 0)
        b = af.log(a)
        self.assertTrue(b.val == np.log(2) and b.jac == 0)
        self.assertTrue(a.val == 2 and a.jac == 0)

    def test_log_advar(self):
        a = Ad_array(2, 3)
        b = af.log(a)
        self.assertTrue(b.val == np.log(2) and b.jac == 1 / 2 * 3)
        self.assertTrue(a.val == 2 and a.jac == 3)

    def test_log_vector(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.log(a)
        jac = sps.diags(1 / val) * J

        self.assertTrue(np.all(b.val == np.log(val)) and np.all(b.jac.A == jac))

    def test_log_sparse_jac(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.log(a)
        jac = np.dot(np.diag(1 / val), J.A)
        self.assertTrue(np.all(b.val == np.log(val)) and np.all(b.jac == jac))

    def test_log_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a = Ad_array(val, J)
        c = 2
        b = af.log(c * a)
        jac = sps.diags(1 / val) * J

        self.assertTrue(
            np.allclose(b.val, np.log(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: sign
    def test_sign_no_advar(self):
        a = np.array([1, -10, 3, -np.pi])
        sign = af.sign(a)
        self.assertTrue(np.all(sign == [1, -1, 1, -1]))

    def test_sign_advar(self):
        a = Ad_array(np.array([1, -10, 3, -np.pi]), np.eye(4))
        sign = af.sign(a)
        self.assertTrue(np.all(sign == [1, -1, 1, -1]))
        self.assertTrue(
            np.allclose(a.val, [1, -10, 3, -np.pi]) and np.allclose(a.jac, np.eye(4))
        )

    # Function: abs
    def test_abs_no_advar(self):
        a = np.array([1, -10, 3, -np.pi])
        a_abs = af.abs(a)
        self.assertTrue(np.allclose(a_abs, [1, 10, 3, np.pi]))
        self.assertTrue(np.allclose(a, [1, -10, 3, -np.pi]))

    def test_abs_advar(self):
        J = np.array(
            [[1, -1, -np.pi, 3], [0, 0, 0, 0], [1, 2, -3.2, 4], [4, 2, 300000, 1]]
        )
        a = Ad_array(np.array([1, -10, 3, -np.pi]), sps.csc_matrix(J))
        a_abs = af.abs(a)
        J_abs = np.array(
            [[1, -1, -np.pi, 3], [0, 0, 0, 0], [1, 2, -3.2, 4], [-4, -2, -300000, -1]]
        )

        self.assertTrue(
            np.allclose(
                J,
                np.array(
                    [
                        [1, -1, -np.pi, 3],
                        [0, 0, 0, 0],
                        [1, 2, -3.2, 4],
                        [4, 2, 300000, 1],
                    ]
                ),
            )
        )
        self.assertTrue(np.allclose(a_abs.val, [1, 10, 3, np.pi]))
        self.assertTrue(np.allclose(a_abs.jac.A, J_abs))

    # Function: sin
    def test_sin_scalar(self):
        a = Ad_array(1, 0)
        b = af.sin(a)
        self.assertTrue(b.val == np.sin(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_sin_advar(self):
        a = Ad_array(2, 3)
        b = af.sin(a)
        self.assertTrue(b.val == np.sin(2) and b.jac == np.cos(2) * 3)
        self.assertTrue(a.val == 2 and a.jac == 3)

    def test_sin_vector(self):
        val = np.array([1, 2, 3])
        J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.sin(a)
        jac = np.dot(np.diag(np.cos(val)), J)

        self.assertTrue(np.all(b.val == np.sin(val)) and np.all(b.jac == jac))
        self.assertTrue(np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])))

    def test_sin_sparse_jac(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.sin(a)
        jac = np.dot(np.diag(np.cos(val)), J.A)
        self.assertTrue(np.all(b.val == np.sin(val)) and np.all(b.jac == jac))

    def test_sin_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a = Ad_array(val, J)
        c = 2
        b = af.sin(c * a)
        jac = c * sps.diags(np.cos(c * val)) * J

        self.assertTrue(
            np.allclose(b.val, np.sin(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: cos
    def test_cos_scalar(self):
        a = Ad_array(1, 0)
        b = af.cos(a)
        self.assertTrue(b.val == np.cos(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_cos_advar(self):
        a = Ad_array(2, 3)
        b = af.cos(a)
        self.assertTrue(b.val == np.cos(2) and b.jac == -np.sin(2) * 3)
        self.assertTrue(a.val == 2 and a.jac == 3)

    def test_cos_vector(self):
        val = np.array([1, 2, 3])
        J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.cos(a)
        jac = np.dot(-np.diag(np.sin(val)), J)

        self.assertTrue(np.all(b.val == np.cos(val)) and np.all(b.jac == jac))
        self.assertTrue(np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])))

    def test_cos_sparse_jac(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.cos(a)
        jac = np.dot(-np.diag(np.sin(val)), J.A)
        self.assertTrue(np.all(b.val == np.cos(val)) and np.all(b.jac == jac))

    def test_cos_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a = Ad_array(val, J)
        c = 2
        b = af.cos(c * a)
        jac = -c * sps.diags(np.sin(c * val)) * J

        self.assertTrue(
            np.allclose(b.val, np.cos(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: tan
    def test_tan_scalar(self):
        a = Ad_array(1, 0)
        b = af.tan(a)
        self.assertTrue(b.val == np.tan(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_tan_advar(self):
        a = Ad_array(2, 3)
        b = af.tan(a)
        self.assertTrue(b.val == np.tan(2) and b.jac == 1 / (np.cos(2) ** 2) * 3)
        self.assertTrue(a.val == 2 and a.jac == 3)

    def test_tan_vector(self):
        val = np.array([1, 2, 3])
        J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.tan(a)
        jac = np.dot(np.diag((np.cos(val) ** 2) ** (-1)), J)

        self.assertTrue(np.all(b.val == np.tan(val)) and np.all(b.jac == jac))
        self.assertTrue(np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])))

    def test_tan_sparse_jac(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.tan(a)
        jac = np.dot(np.diag((np.cos(val) ** 2) ** (-1)), J.A)
        self.assertTrue(np.all(b.val == np.tan(val)) and np.all(b.jac == jac))

    def test_tan_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a = Ad_array(val, J)
        c = 2
        b = af.tan(c * a)
        jac = c * sps.diags((np.cos(c * val) ** 2) ** (-1)) * J

        self.assertTrue(
            np.allclose(b.val, np.tan(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: arcsin
    def test_arcsin_scalar(self):
        a = Ad_array(0.5, 0)
        b = af.arcsin(a)
        self.assertTrue(b.val == np.arcsin(0.5) and b.jac == 0)
        self.assertTrue(a.val == 0.5 and a.jac == 0)

    def test_arcsin_advar(self):
        a = Ad_array(0.2, 0.3)
        b = af.arcsin(a)
        self.assertTrue(
            b.val == np.arcsin(0.2) and b.jac == (1 - 0.2**2) ** (-0.5) * 0.3
        )
        self.assertTrue(a.val == 0.2 and a.jac == 0.3)

    def test_arcsin_vector(self):
        val = np.array([0.1, 0.2, 0.3])
        J = np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.arcsin(a)
        jac = np.dot(np.diag((1 - val**2) ** (-0.5)), J)

        self.assertTrue(np.all(b.val == np.arcsin(val)) and np.all(b.jac == jac))
        self.assertTrue(
            np.all(J == np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))
        )

    def test_arcsin_sparse_jac(self):
        val = np.array([0.1, 0.2, 0.3])
        J = sps.csc_matrix(
            np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
        )
        a = Ad_array(val, J)
        b = af.arcsin(a)
        jac = np.dot(np.diag((1 - val**2) ** (-0.5)), J.A)
        self.assertTrue(np.all(b.val == np.arcsin(val)) and np.all(b.jac == jac))

    def test_arcsin_scalar_times_ad_var(self):
        val = np.array([0.1, 0.2, 0.3])
        J = sps.diags(np.array([0.1, 0.1, 0.1]))
        a = Ad_array(val, J)
        c = 0.2
        b = af.arcsin(c * a)
        jac = sps.diags(c * (1 - (c * val) ** 2) ** (-0.5)) * J

        self.assertTrue(
            np.allclose(b.val, np.arcsin(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [0.1, 0.2, 0.3]) and np.all(a.jac.A == J.A))

    # Function: arccos
    def test_arccos_scalar(self):
        a = Ad_array(0.5, 0)
        b = af.arccos(a)
        self.assertTrue(b.val == np.arccos(0.5) and b.jac == 0)
        self.assertTrue(a.val == 0.5 and a.jac == 0)

    def test_arccos_advar(self):
        a = Ad_array(0.2, 0.3)
        b = af.arccos(a)
        self.assertTrue(
            b.val == np.arccos(0.2) and b.jac == -((1 - 0.2**2) ** (-0.5)) * 0.3
        )
        self.assertTrue(a.val == 0.2 and a.jac == 0.3)

    def test_arccos_vector(self):
        val = np.array([0.1, 0.2, 0.3])
        J = np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.arccos(a)
        jac = np.dot(-np.diag((1 - val**2) ** (-0.5)), J)

        self.assertTrue(np.all(b.val == np.arccos(val)) and np.all(b.jac == jac))
        self.assertTrue(
            np.all(J == np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))
        )

    def test_arccos_sparse_jac(self):
        val = np.array([0.1, 0.2, 0.3])
        J = sps.csc_matrix(
            np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
        )
        a = Ad_array(val, J)
        b = af.arccos(a)
        jac = np.dot(-np.diag((1 - val**2) ** (-0.5)), J.A)
        self.assertTrue(np.all(b.val == np.arccos(val)) and np.all(b.jac == jac))

    def test_arccos_scalar_times_ad_var(self):
        val = np.array([0.1, 0.2, 0.3])
        J = sps.diags(np.array([0.1, 0.1, 0.1]))
        a = Ad_array(val, J)
        c = 0.2
        b = af.arccos(c * a)
        jac = -sps.diags(c * (1 - (c * val) ** 2) ** (-0.5)) * J

        self.assertTrue(
            np.allclose(b.val, np.arccos(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [0.1, 0.2, 0.3]) and np.all(a.jac.A == J.A))

    # Function: arctan
    def test_arctan_scalar(self):
        a = Ad_array(0.5, 0)
        b = af.arctan(a)
        self.assertTrue(b.val == np.arctan(0.5) and b.jac == 0)
        self.assertTrue(a.val == 0.5 and a.jac == 0)

    def test_arctan_advar(self):
        a = Ad_array(0.2, 0.3)
        b = af.arctan(a)
        self.assertTrue(
            b.val == np.arctan(0.2) and b.jac == (1 + 0.2**2) ** (-1) * 0.3
        )
        self.assertTrue(a.val == 0.2 and a.jac == 0.3)

    def test_arctan_vector(self):
        val = np.array([0.1, 0.2, 0.3])
        J = np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.arctan(a)
        jac = np.dot(np.diag((1 + val**2) ** (-1)), J)

        self.assertTrue(np.all(b.val == np.arctan(val)) and np.all(b.jac == jac))
        self.assertTrue(
            np.all(J == np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))
        )

    def test_arctan_sparse_jac(self):
        val = np.array([0.1, 0.2, 0.3])
        J = sps.csc_matrix(
            np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
        )
        a = Ad_array(val, J)
        b = af.arctan(a)
        jac = np.dot(np.diag((1 + val**2) ** (-1)), J.A)
        self.assertTrue(np.all(b.val == np.arctan(val)) and np.all(b.jac == jac))

    def test_arctan_scalar_times_ad_var(self):
        val = np.array([0.1, 0.2, 0.3])
        J = sps.diags(np.array([0.1, 0.1, 0.1]))
        a = Ad_array(val, J)
        c = 0.2
        b = af.arctan(c * a)
        jac = sps.diags(c * (1 + (c * val) ** 2) ** (-1)) * J

        self.assertTrue(
            np.allclose(b.val, np.arctan(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [0.1, 0.2, 0.3]) and np.all(a.jac.A == J.A))

    # Function: sinh
    def test_sinh_scalar(self):
        a = Ad_array(1, 0)
        b = af.sinh(a)
        self.assertTrue(b.val == np.sinh(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_sinh_advar(self):
        a = Ad_array(2, 3)
        b = af.sinh(a)
        self.assertTrue(b.val == np.sinh(2) and b.jac == np.cosh(2) * 3)
        self.assertTrue(a.val == 2 and a.jac == 3)

    def test_sinh_vector(self):
        val = np.array([1, 2, 3])
        J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.sinh(a)
        jac = np.dot(np.diag(np.cosh(val)), J)

        self.assertTrue(np.all(b.val == np.sinh(val)) and np.all(b.jac == jac))
        self.assertTrue(np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])))

    def test_sinh_sparse_jac(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.sinh(a)
        jac = np.dot(np.diag(np.cosh(val)), J.A)
        self.assertTrue(np.all(b.val == np.sinh(val)) and np.all(b.jac == jac))

    def test_sinh_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a = Ad_array(val, J)
        c = 2
        b = af.sinh(c * a)
        jac = c * sps.diags(np.cosh(c * val)) * J

        self.assertTrue(
            np.allclose(b.val, np.sinh(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: cosh
    def test_cosh_scalar(self):
        a = Ad_array(1, 0)
        b = af.cosh(a)
        self.assertTrue(b.val == np.cosh(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_cosh_advar(self):
        a = Ad_array(2, 3)
        b = af.cosh(a)
        self.assertTrue(b.val == np.cosh(2) and b.jac == np.sinh(2) * 3)
        self.assertTrue(a.val == 2 and a.jac == 3)

    def test_cosh_vector(self):
        val = np.array([1, 2, 3])
        J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.cosh(a)
        jac = np.dot(np.diag(np.sinh(val)), J)

        self.assertTrue(np.all(b.val == np.cosh(val)) and np.all(b.jac == jac))
        self.assertTrue(np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])))

    def test_cosh_sparse_jac(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.cosh(a)
        jac = np.dot(np.diag(np.sinh(val)), J.A)
        self.assertTrue(np.all(b.val == np.cosh(val)) and np.all(b.jac == jac))

    def test_cosh_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a = Ad_array(val, J)
        c = 2
        b = af.cosh(c * a)
        jac = c * sps.diags(np.sinh(c * val)) * J

        self.assertTrue(
            np.allclose(b.val, np.cosh(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: tanh
    def test_tanh_scalar(self):
        a = Ad_array(1, 0)
        b = af.tanh(a)
        self.assertTrue(b.val == np.tanh(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_tanh_advar(self):
        a = Ad_array(2, 3)
        b = af.tanh(a)
        self.assertTrue(b.val == np.tanh(2) and b.jac == np.cosh(2) ** (-2) * 3)
        self.assertTrue(a.val == 2 and a.jac == 3)

    def test_tanh_vector(self):
        val = np.array([1, 2, 3])
        J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.tanh(a)
        jac = np.dot(np.diag((np.cosh(val) ** 2) ** (-1)), J)

        self.assertTrue(np.all(b.val == np.tanh(val)) and np.all(b.jac == jac))
        self.assertTrue(np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])))

    def test_tanh_sparse_jac(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.tanh(a)
        jac = np.dot(np.diag((np.cosh(val) ** 2) ** (-1)), J.A)
        self.assertTrue(np.all(b.val == np.tanh(val)) and np.all(b.jac == jac))

    def test_tanh_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a = Ad_array(val, J)
        c = 2
        b = af.tanh(c * a)
        jac = c * sps.diags((np.cosh(c * val) ** 2) ** (-1)) * J

        self.assertTrue(
            np.allclose(b.val, np.tanh(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: arcsinh
    def test_arcsinh_scalar(self):
        a = Ad_array(0.5, 0)
        b = af.arcsinh(a)
        self.assertTrue(b.val == np.arcsinh(0.5) and b.jac == 0)
        self.assertTrue(a.val == 0.5 and a.jac == 0)

    def test_arcsinh_advar(self):
        a = Ad_array(0.2, 0.3)
        b = af.arcsinh(a)
        self.assertTrue(
            b.val == np.arcsinh(0.2) and b.jac == (1 + 0.2**2) ** (-0.5) * 0.3
        )
        self.assertTrue(a.val == 0.2 and a.jac == 0.3)

    def test_arcsinh_vector(self):
        val = np.array([0.1, 0.2, 0.3])
        J = np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.arcsinh(a)
        jac = np.dot(np.diag((1 + val**2) ** (-0.5)), J)

        self.assertTrue(np.all(b.val == np.arcsinh(val)) and np.all(b.jac == jac))
        self.assertTrue(
            np.all(J == np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))
        )

    def test_arcsinh_sparse_jac(self):
        val = np.array([0.1, 0.2, 0.3])
        J = sps.csc_matrix(
            np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
        )
        a = Ad_array(val, J)
        b = af.arcsinh(a)
        jac = np.dot(np.diag((1 + val**2) ** (-0.5)), J.A)
        self.assertTrue(np.all(b.val == np.arcsinh(val)) and np.all(b.jac == jac))

    def test_arcsinh_scalar_times_ad_var(self):
        val = np.array([0.1, 0.2, 0.3])
        J = sps.diags(np.array([0.1, 0.1, 0.1]))
        a = Ad_array(val, J)
        c = 0.2
        b = af.arcsinh(c * a)
        jac = sps.diags(c * (1 + (c * val) ** 2) ** (-0.5)) * J

        self.assertTrue(
            np.allclose(b.val, np.arcsinh(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [0.1, 0.2, 0.3]) and np.all(a.jac.A == J.A))

    # Function: arccosh
    def test_arccosh_scalar(self):
        a = Ad_array(2, 0)
        b = af.arccosh(a)
        self.assertTrue(b.val == np.arccosh(2) and b.jac == 0)
        self.assertTrue(a.val == 2 and a.jac == 0)

    def test_arccosh_advar(self):
        a = Ad_array(2, 3)
        b = af.arccosh(a)
        self.assertTrue(
            b.val == np.arccosh(2)
            and b.jac == (2 - 1) ** (-0.5) * (2 + 1) ** (-0.5) * 3
        )
        self.assertTrue(a.val == 2 and a.jac == 3)

    def test_arccosh_vector(self):
        val = np.array([1, 2, 3])
        J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.arccosh(a)
        jac = np.dot(np.diag((val - 1) ** (-0.5) * (val + 1) ** (-0.5)), J)

        self.assertTrue(np.all(b.val == np.arccosh(val)) and np.all(b.jac == jac))
        self.assertTrue(np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])))

    def test_arccosh_sparse_jac(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.arccosh(a)
        jac = np.dot(np.diag((val - 1) ** (-0.5) * (val + 1) ** (-0.5)), J.A)
        self.assertTrue(np.all(b.val == np.arccosh(val)) and np.all(b.jac == jac))

    def test_arccosh_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a = Ad_array(val, J)
        c = 2
        b = af.arccosh(c * a)
        jac = sps.diags(c * (c * val - 1) ** (-0.5) * (c * val + 1) ** (-0.5)) * J

        self.assertTrue(
            np.allclose(b.val, np.arccosh(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: arctanh
    def test_arctanh_scalar(self):
        a = Ad_array(0.5, 0)
        b = af.arctanh(a)
        self.assertTrue(b.val == np.arctanh(0.5) and b.jac == 0)
        self.assertTrue(a.val == 0.5 and a.jac == 0)

    def test_arctanh_advar(self):
        a = Ad_array(0.2, 0.3)
        b = af.arctanh(a)
        self.assertTrue(
            b.val == np.arctanh(0.2) and b.jac == (1 - 0.2**2) ** (-1) * 0.3
        )
        self.assertTrue(a.val == 0.2 and a.jac == 0.3)

    def test_arctanh_vector(self):
        val = np.array([0.1, 0.2, 0.3])
        J = np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.arctanh(a)
        jac = np.dot(np.diag((1 - val**2) ** (-1)), J)

        self.assertTrue(np.all(b.val == np.arctanh(val)) and np.all(b.jac == jac))
        self.assertTrue(
            np.all(J == np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))
        )

    def test_arctanh_sparse_jac(self):
        val = np.array([0.1, 0.2, 0.3])
        J = sps.csc_matrix(
            np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
        )
        a = Ad_array(val, J)
        b = af.arctanh(a)
        jac = np.dot(np.diag((1 - val**2) ** (-1)), J.A)
        self.assertTrue(np.all(b.val == np.arctanh(val)) and np.all(b.jac == jac))

    def test_arctanh_scalar_times_ad_var(self):
        val = np.array([0.1, 0.2, 0.3])
        J = sps.diags(np.array([0.1, 0.1, 0.1]))
        a = Ad_array(val, J)
        c = 0.2
        b = af.arctanh(c * a)
        jac = sps.diags(c * (1 - (c * val) ** 2) ** (-1)) * J

        self.assertTrue(
            np.allclose(b.val, np.arctanh(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [0.1, 0.2, 0.3]) and np.all(a.jac.A == J.A))

    # Function: heaviside_smooth
    def test_heaviside_smooth_scalar(self):
        a = Ad_array(0.5, 0)
        b = af.heaviside_smooth(a)
        val = 0.5 * (1 + 2 / np.pi * np.arctan(0.5 / 1e-3))
        self.assertTrue(b.val == val and b.jac == 0)
        self.assertTrue(a.val == 0.5 and a.jac == 0)

    def test_heaviside_smooth_advar(self):
        a = Ad_array(0.2, 0.3)
        b = af.heaviside_smooth(a)
        val = 0.5 * (1 + (2 / np.pi) * np.arctan(0.2 / 1e-3))
        der = (1 / np.pi) * (1e-3 / (1e-3**2 + 0.2**2))
        self.assertTrue(np.isclose(b.val, val) and np.isclose(b.jac, der * 0.3))
        self.assertTrue(a.val == 0.2 and a.jac == 0.3)

    def test_heaviside_smooth_vector(self):
        val = np.array([1, -2, 3])
        J = np.array([[3, -2, 1], [-5, 6, 1], [2, 3, -5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.heaviside_smooth(a)

        true_val = 0.5 * (1 + 2 * np.pi ** (-1) * np.arctan(val * 1e3))
        true_jac = np.dot(
            np.diag(np.pi ** (-1) * (1e-3 * (1e-3**2 + val**2) ** (-1))), J
        )

        self.assertTrue(np.allclose(b.val, true_val) and np.allclose(b.jac.A, true_jac))
        self.assertTrue(np.all(J == np.array([[3, -2, 1], [-5, 6, 1], [2, 3, -5]])))

    def test_heaviside_smooth_sparse_jac(self):
        val = np.array([1, -2, 3])
        J = sps.csc_matrix(np.array([[3, -2, 1], [-5, 6, 1], [2, 3, -5]]))
        a = Ad_array(val, J)
        b = af.heaviside_smooth(a)

        true_val = 0.5 * (1 + 2 * np.pi ** (-1) * np.arctan(val * 1e3))
        true_jac = np.dot(
            np.diag(np.pi ** (-1) * (1e-3 * (1e-3**2 + val**2) ** (-1))), J.A
        )

        self.assertTrue(np.allclose(b.val, true_val) and np.allclose(b.jac.A, true_jac))

    def test_heaviside_smooth_times_ad_var(self):
        val = np.array([1, -2, -3])
        J = sps.diags(np.array([0.1, 0.1, 0.1]))
        a = Ad_array(val, J)
        c = 0.2
        b = af.heaviside_smooth(c * a)

        true_val = 0.5 * (1 + 2 * np.pi ** (-1) * np.arctan(c * val * 1e3))
        true_jac = (
            sps.diags(c * np.pi ** (-1) * (1e-3 * (1e-3**2 + (c * val) ** 2) ** (-1)))
            * J
        )

        self.assertTrue(
            np.allclose(b.val, true_val) and np.allclose(b.jac.A, true_jac.A)
        )
        self.assertTrue(np.all(a.val == [1, -2, -3]) and np.all(a.jac.A == J.A))
