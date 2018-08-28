import numpy as np
import scipy.sparse as sps
import unittest

from porepy.ad.forward_mode import Ad_array
from porepy.ad import functions as af


class AdFunctionTest(unittest.TestCase):
    def test_exp_scalar(self):
        a = Ad_array(1, 0)
        b = af.exp(a)
        assert b.val == np.exp(1) and b.jac == 0
        assert a.val == 1 and a.jac == 0

    def test_exp_advar(self):
        a = Ad_array(2, 3)
        b = af.exp(a)
        assert b.val == np.exp(2) and b.jac == 3 * np.exp(2)
        assert a.val == 2 and a.jac == 3

    def test_exp_vector(self):
        val = np.array([1, 2, 3])
        J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.exp(a)
        jac = np.dot(np.diag(np.exp(val)), J)

        assert np.all(b.val == np.exp(val)) and np.all(b.jac == jac)
        assert np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))

    def test_exp_sparse_jac(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.exp(a)
        jac = np.dot(np.diag(np.exp(val)), J.A)
        assert np.all(b.val == np.exp(val)) and np.all(b.jac == jac)

    def test_exp_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a = Ad_array(val, J)
        c = 2
        b = af.exp(c * a)
        jac = c * sps.diags(np.exp(c * val)) * J

        assert np.allclose(b.val, np.exp(c * val)) and np.allclose(b.jac.A, jac.A)
        assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A)

    def test_log_scalar(self):
        a = Ad_array(2, 0)
        b = af.log(a)
        assert b.val == np.log(2) and b.jac == 0
        assert a.val == 2 and a.jac == 0

    def test_log_advar(self):
        a = Ad_array(2, 3)
        b = af.log(a)
        assert b.val == np.log(2) and b.jac == 1 / 2 * 3
        assert a.val == 2 and a.jac == 3

    def test_log_vector(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.log(a)
        jac = sps.diags(1 / val) * J

        assert np.all(b.val == np.log(val)) and np.all(b.jac.A == jac)

    def test_log_sparse_jac(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.log(a)
        jac = np.dot(np.diag(1 / val), J.A)
        assert np.all(b.val == np.log(val)) and np.all(b.jac == jac)

    def test_log_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a = Ad_array(val, J)
        c = 2
        b = af.log(c * a)
        jac = sps.diags(1 / val) * J

        assert np.allclose(b.val, np.log(c * val)) and np.allclose(b.jac.A, jac.A)
        assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A)

    def test_sign_no_advar(self):
        a = np.array([1, -10, 3, -np.pi])
        sign = af.sign(a)
        assert np.all(sign == [1, -1, 1, -1])

    def test_sign_advar(self):
        a = Ad_array(np.array([1, -10, 3, -np.pi]), np.eye(4))
        sign = af.sign(a)
        assert np.all(sign == [1, -1, 1, -1])
        assert np.allclose(a.val, [1, -10, 3, -np.pi]) and np.allclose(a.jac, np.eye(4))

    def test_abs_no_advar(self):
        a = np.array([1, -10, 3, -np.pi])
        a_abs = af.abs(a)
        assert np.allclose(a_abs, [1, 10, 3, np.pi])
        assert np.allclose(a, [1, -10, 3, -np.pi])

    def test_abs_advar(self):
        J = np.array(
            [[1, -1, -np.pi, 3], [0, 0, 0, 0], [1, 2, -3.2, 4], [4, 2, 300000, 1]]
        )
        a = Ad_array(np.array([1, -10, 3, -np.pi]), sps.csc_matrix(J))
        a_abs = af.abs(a)
        J_abs = np.array(
            [[1, -1, -np.pi, 3], [0, 0, 0, 0], [1, 2, -3.2, 4], [-4, -2, -300000, -1]]
        )

        assert np.allclose(
            J,
            np.array(
                [[1, -1, -np.pi, 3], [0, 0, 0, 0], [1, 2, -3.2, 4], [4, 2, 300000, 1]]
            ),
        )
        assert np.allclose(a_abs.val, [1, 10, 3, np.pi])
        assert np.allclose(a_abs.jac.A, J_abs)
