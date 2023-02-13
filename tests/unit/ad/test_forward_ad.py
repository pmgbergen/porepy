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

from porepy.numerics.ad import Ad_array
from porepy.numerics.ad import functions as af
from porepy.numerics.ad import initAdArrays

warnings.simplefilter("ignore", sps.SparseEfficiencyWarning)


class AdArrayInitAndCombination(unittest.TestCase):
    """
    The tests cover initiation of Ad_array (joint initiation of multiple dependent
    variables). The test also partly cover the arithmetic operations implemented for
    Ad_arrays, e.g., __add__, __sub__, etc., but these are also tested in different
    tests.
    """

    def _compare(self, arr, known_val, known_jac):
        self.assertTrue(np.allclose(arr.val, known_val))
        self.assertTrue(np.allclose(arr.jac.A, known_jac))

    def test_add_two_ad_variables_init(self):
        a, b = initAdArrays([np.array([1]), np.array([-10])])
        c = a + b
        self.assertTrue(c.val == -9 and np.all(c.jac.A == [1, 1]))
        self.assertTrue(a.val == 1 and np.all(a.jac.A == [1, 0]))
        self.assertTrue(b.val == -10 and np.all(b.jac.A == [0, 1]))

    def test_sub_var_init_with_var_init(self):
        a, b = initAdArrays([np.array([3]), np.array([2])])
        c = b - a
        self.assertTrue(np.allclose(c.val, -1) and np.all(c.jac.A == [-1, 1]))
        self.assertTrue(a.val == 3 and np.all(a.jac.A == [1, 0]))
        self.assertTrue(b.val == 2 and np.all(b.jac.A == [0, 1]))

    def test_mul_ad_var_init(self):
        a, b = initAdArrays([np.array([3]), np.array([2])])
        c = a * b
        self.assertTrue(c.val == 6 and np.all(c.jac.A == [2, 3]))
        self.assertTrue(a.val == 3 and np.all(a.jac.A == [1, 0]))
        self.assertTrue(b.val == 2 and np.all(b.jac.A == [0, 1]))

    def test_mul_scal_ad_var_init(self):
        a, b = initAdArrays([np.array([3]), np.array([2])])
        d = 3.0
        c = d * a
        self.assertTrue(c.val == 9 and np.all(c.jac.A == [3, 0]))
        self.assertTrue(a.val == 3 and np.all(a.jac.A == [1, 0]))
        self.assertTrue(b.val == 2 and np.all(b.jac.A == [0, 1]))

    def test_mul_sps_advar_init(self):
        x = initAdArrays([np.array([1, 2, 3])])[0]
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        f = A @ x
        self.assertTrue(np.all(f.val == [14, 32, 50]))
        self.assertTrue(np.all((f.jac == A).A))

    def test_advar_init_diff_len(self):
        a, b = initAdArrays([np.array([1, 2, 3]), np.array([1, 2])])
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        B = sps.csc_matrix(np.array([[1, 2], [4, 5]]))

        f = A @ a
        g = B @ b
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
        c = 2.0
        b = af.exp(c * a)

        zero = sps.csc_matrix((3, 3))
        jac = sps.hstack([c * sps.diags(np.exp(c * val)) * J, zero, zero])
        jac_a = sps.hstack([J, zero, zero])
        self.assertTrue(
            np.allclose(b.val, np.exp(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == jac_a.A))

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
        y = R @ x
        z = y * vec
        Jy = np.array([[3, 0, 1], [0, 2, 0]])
        Jz = np.array([[1, 0, 3], [0, 4, 0]])
        self.assertTrue(np.all(y.val == [4, 2]))
        self.assertTrue(np.sum(y.jac.A - Jy) == 0)
        self.assertTrue(np.all(z.val == [4, 4]))
        self.assertTrue(np.sum(z.jac.A - Jz) == 0)

    def test_mul_sps_advar(self):
        J = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
        x = Ad_array(np.array([1, 2, 3]), J)
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        f = A @ x

        self.assertTrue(np.all(f.val == [14, 32, 50]))
        self.assertTrue(np.all(f.jac == A * J.A))

    def test_mul_advar_vectors(self):
        Ja = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
        Jb = sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        a = Ad_array(np.array([1, 2, 3]), Ja)
        b = Ad_array(np.array([1, 1, 1]), Jb)
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        f = A @ a + b

        self.assertTrue(np.all(f.val == [15, 33, 51]))
        self.assertTrue(np.sum(f.jac.A != A * Ja + Jb) == 0)
        self.assertTrue(
            np.sum(Ja != sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]])))
            == 0
        )
        self.assertTrue(
            np.sum(Jb != sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
            == 0
        )

    def test_copy_scalar(self):
        a = Ad_array(np.array([1]), sps.csr_matrix([[0]]))
        b = a.copy()
        self.assertTrue(a.val == b.val)
        self.assertTrue(a.jac == b.jac)
        a.val = 2
        a.jac = 3
        self.assertTrue(b.val == 1)
        self.assertTrue(b.jac == 0)

    def test_copy_vector(self):
        a = Ad_array(np.ones(3), sps.csr_matrix(np.diag(np.ones((3)))))
        b = a.copy()
        self.assertTrue(np.allclose(a.val, b.val))
        self.assertTrue(np.allclose(a.jac.A, b.jac.A))
        a.val[0] = 3
        a.jac[2] = 4
        self.assertTrue(np.allclose(b.val, np.ones(3)))
        self.assertTrue(np.allclose(b.jac.A, sps.csr_matrix(np.diag(np.ones((3)))).A))


class AdFunctionTest(unittest.TestCase):

    # Function: exp
    def test_exp_scalar(self):
        a = Ad_array(np.array([1]), sps.csr_matrix(np.array([0])))
        b = af.exp(a)
        self.assertTrue(b.val == np.exp(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_exp_advar(self):
        a = Ad_array(np.array([2]), sps.csr_matrix(np.array([3])))
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
        c = 2.0
        b = af.exp(c * a)
        jac = c * sps.diags(np.exp(c * val)) * J

        self.assertTrue(
            np.allclose(b.val, np.exp(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: log
    def test_log_scalar(self):
        a = Ad_array(np.array([2]), sps.csr_matrix(np.array([0])))
        b = af.log(a)
        self.assertTrue(b.val == np.log(2) and b.jac == 0)
        self.assertTrue(a.val == 2 and a.jac == 0)

    def test_log_advar(self):
        a = Ad_array(np.array([2]), sps.csr_matrix(np.array([3])))
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
        c = 2.0
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
        a = Ad_array(np.array([1]), sps.csr_matrix(np.array([0])))
        b = af.sin(a)
        self.assertTrue(b.val == np.sin(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_sin_advar(self):
        a = Ad_array(np.array([2]), sps.csr_matrix(np.array([3])))
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
        c = 2.0
        b = af.sin(c * a)
        jac = c * sps.diags(np.cos(c * val)) * J

        self.assertTrue(
            np.allclose(b.val, np.sin(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: cos
    def test_cos_scalar(self):
        a = Ad_array(np.array([1]), sps.csr_matrix(np.array([0])))
        b = af.cos(a)
        self.assertTrue(b.val == np.cos(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_cos_advar(self):
        a = Ad_array(np.array([2]), sps.csr_matrix(np.array([3])))
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
        c = 2.0
        b = af.cos(c * a)
        jac = -c * sps.diags(np.sin(c * val)) * J

        self.assertTrue(
            np.allclose(b.val, np.cos(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: tan
    def test_tan_scalar(self):
        a = Ad_array(np.array([1]), sps.csr_matrix(np.array([0])))
        b = af.tan(a)
        self.assertTrue(b.val == np.tan(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_tan_advar(self):
        a = Ad_array(np.array([2]), sps.csr_matrix(np.array([3])))
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
        c = 2.0
        b = af.tan(c * a)
        jac = c * sps.diags((np.cos(c * val) ** 2) ** (-1)) * J

        self.assertTrue(
            np.allclose(b.val, np.tan(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: arcsin
    def test_arcsin_scalar(self):
        a = Ad_array(np.array([0.5]), sps.csr_matrix(np.array([0])))
        b = af.arcsin(a)
        self.assertTrue(b.val == np.arcsin(0.5) and b.jac == 0)
        self.assertTrue(a.val == 0.5 and a.jac == 0)

    def test_arcsin_advar(self):
        a = Ad_array(np.array([0.2]), sps.csr_matrix(np.array([0.3])))
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
        a = Ad_array(np.array([0.5]), sps.csr_matrix(np.array([0])))
        b = af.arccos(a)
        self.assertTrue(b.val == np.arccos(0.5) and b.jac == 0)
        self.assertTrue(a.val == 0.5 and a.jac == 0)

    def test_arccos_advar(self):
        a = Ad_array(np.array([0.2]), sps.csr_matrix(np.array([0.3])))
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
        a = Ad_array(np.array([0.5]), sps.csr_matrix(np.array([0])))
        b = af.arctan(a)
        self.assertTrue(b.val == np.arctan(0.5) and b.jac == 0)
        self.assertTrue(a.val == 0.5 and a.jac == 0)

    def test_arctan_advar(self):
        a = Ad_array(np.array([0.2]), sps.csr_matrix(np.array([0.3])))
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
        a = Ad_array(np.array([1]), sps.csr_matrix(np.array([0])))
        b = af.sinh(a)
        self.assertTrue(b.val == np.sinh(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_sinh_advar(self):
        a = Ad_array(np.array([2]), sps.csr_matrix(np.array([3])))
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
        c = 2.0
        b = af.sinh(c * a)
        jac = c * sps.diags(np.cosh(c * val)) * J

        self.assertTrue(
            np.allclose(b.val, np.sinh(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: cosh
    def test_cosh_scalar(self):
        a = Ad_array(np.array([1]), sps.csr_matrix(np.array([0])))
        b = af.cosh(a)
        self.assertTrue(b.val == np.cosh(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_cosh_advar(self):
        a = Ad_array(np.array([2]), sps.csr_matrix(np.array([3])))
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
        c = 2.0
        b = af.cosh(c * a)
        jac = c * sps.diags(np.sinh(c * val)) * J

        self.assertTrue(
            np.allclose(b.val, np.cosh(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: tanh
    def test_tanh_scalar(self):
        a = Ad_array(np.array([1]), sps.csr_matrix(np.array([0])))
        b = af.tanh(a)
        self.assertTrue(b.val == np.tanh(1) and b.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)

    def test_tanh_advar(self):
        a = Ad_array(np.array([2]), sps.csr_matrix(np.array([3])))
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
        c = 2.0
        b = af.tanh(c * a)
        jac = c * sps.diags((np.cosh(c * val) ** 2) ** (-1)) * J

        self.assertTrue(
            np.allclose(b.val, np.tanh(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: arcsinh
    def test_arcsinh_scalar(self):
        a = Ad_array(np.array([0.5]), sps.csr_matrix(np.array([0])))
        b = af.arcsinh(a)
        self.assertTrue(b.val == np.arcsinh(0.5) and b.jac == 0)
        self.assertTrue(a.val == 0.5 and a.jac == 0)

    def test_arcsinh_advar(self):
        a = Ad_array(np.array([0.2]), sps.csr_matrix(np.array([0.3])))
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
        a = Ad_array(np.array([2]), sps.csr_matrix(np.array([0])))
        b = af.arccosh(a)
        self.assertTrue(b.val == np.arccosh(2) and b.jac == 0)
        self.assertTrue(a.val == 2 and a.jac == 0)

    def test_arccosh_advar(self):
        a = Ad_array(np.array([2]), sps.csr_matrix(np.array([3])))
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
        c = 2.0
        b = af.arccosh(c * a)
        jac = sps.diags(c * (c * val - 1) ** (-0.5) * (c * val + 1) ** (-0.5)) * J

        self.assertTrue(
            np.allclose(b.val, np.arccosh(c * val)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))

    # Function: arctanh
    def test_arctanh_scalar(self):
        a = Ad_array(np.array([0.5]), sps.csr_matrix(np.array([0])))
        b = af.arctanh(a)
        self.assertTrue(b.val == np.arctanh(0.5) and b.jac == 0)
        self.assertTrue(a.val == 0.5 and a.jac == 0)

    def test_arctanh_advar(self):
        a = Ad_array(np.array([0.2]), sps.csr_matrix(np.array([0.3])))
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
        a = Ad_array(np.array([0.5]), sps.csr_matrix(np.array([0])))
        b = af.heaviside_smooth(a)
        val = 0.5 * (1 + 2 / np.pi * np.arctan(0.5 / 1e-3))
        self.assertTrue(b.val == val and b.jac == 0)
        self.assertTrue(a.val == 0.5 and a.jac == 0)

    def test_heaviside_smooth_advar(self):
        a = Ad_array(np.array([0.2]), sps.csr_matrix(np.array([0.3])))
        b = af.heaviside_smooth(a)
        val = 0.5 * (1 + (2 / np.pi) * np.arctan(0.2 / 1e-3))
        der = (1 / np.pi) * (1e-3 / (1e-3**2 + 0.2**2))
        self.assertTrue(np.isclose(b.val, val) and np.isclose(b.jac.A, der * 0.3))
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
