import numpy as np
import scipy.sparse as sps
import unittest
import warnings

from porepy.ad.forward_mode import initAdArrays
from porepy.ad import functions as af

warnings.simplefilter("ignore", sps.SparseEfficiencyWarning)


class AdInitTest(unittest.TestCase):
    def test_add_two_ad_variables_init(self):
        a, b = initAdArrays([np.array(1), np.array(-10)])
        c = a + b
        assert c.val == -9 and np.all(c.jac.A == [1, 1])
        assert a.val == 1 and np.all(a.jac.A == [1, 0])
        assert b.val == -10 and np.all(b.jac.A == [0, 1])

    def test_add_var_init_with_scal(self):
        a = initAdArrays(3)
        b = 3
        c = a + b
        assert np.allclose(c.val, 6) and np.allclose(c.jac.A, 1)
        assert a.val == 3 and np.allclose(a.jac.A, 1)
        assert b == 3

    def test_sub_scal_with_var_init(self):
        a = initAdArrays(3)
        b = 3
        c = b - a
        assert np.allclose(c.val, 0) and np.allclose(c.jac.A, -1)
        assert a.val == 3 and a.jac.A == 1
        assert b == 3

    def test_sub_var_init_with_var_init(self):
        a, b = initAdArrays([np.array(3), np.array(2)])
        c = b - a
        assert np.allclose(c.val, -1) and np.all(c.jac.A == [-1, 1])
        assert a.val == 3 and np.all(a.jac.A == [1, 0])
        assert b.val == 2 and np.all(b.jac.A == [0, 1])

    def test_add_scal_with_var_init(self):
        a = initAdArrays(3)
        b = 3
        c = b + a
        assert np.allclose(c.val, 6) and np.allclose(c.jac.A, 1)
        assert a.val == 3 and np.allclose(a.jac.A, 1)
        assert b == 3

    def test_mul_ad_var_init(self):
        a, b = initAdArrays([np.array(3), np.array(2)])
        c = a * b
        assert c.val == 6 and np.all(c.jac.A == [2, 3])
        assert a.val == 3 and np.all(a.jac.A == [1, 0])
        assert b.val == 2 and np.all(b.jac.A == [0, 1])

    def test_mul_scal_ad_var_init(self):
        a, b = initAdArrays([np.array(3), np.array(2)])
        d = 3
        c = d * a
        assert c.val == 9 and np.all(c.jac.A == [3, 0])
        assert a.val == 3 and np.all(a.jac.A == [1, 0])
        assert b.val == 2 and np.all(b.jac.A == [0, 1])

    def test_mul_sps_advar_init(self):
        x = initAdArrays(np.array([1, 2, 3]))
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        f = A * x
        assert np.all(f.val == [14, 32, 50])
        assert np.all((f.jac == A).A)

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
        assert np.all(f.val == [14, 32, 50])
        assert np.all((f.jac == jac_f).A)
        assert np.all(g.val == [5, 14])
        assert np.all((g.jac == jac_g).A)

    def test_advar_init_cross_jacobi(self):
        x, y = initAdArrays([np.array([-1, 4]), np.array([1, 5])])

        z = x * y
        J = np.array([[1, 0, -1, 0], [0, 5, 0, 4]])
        assert np.all(z.val == [-1, 20])
        assert np.all((z.jac == J).A)

    def test_power_advar_advar_init(self):
        a, b = initAdArrays([np.array(4.), np.array(-8)])

        c = a ** b
        jac = np.array([-8 * (4 ** -9), 4 ** -8 * np.log(4)])

        assert np.allclose(c.val, 4 ** -8)
        assert np.all(np.abs(c.jac.A - jac) < 1e-6)

    def test_exp_scalar_times_ad_var(self):
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a, _, _ = initAdArrays([val, val, val])
        c = 2
        b = af.exp(c * a)

        zero = sps.csc_matrix((3, 3))
        jac = sps.hstack([c * sps.diags(np.exp(c * val)) * J, zero, zero])
        jac_a = sps.hstack([J, zero, zero])
        assert np.allclose(b.val, np.exp(c * val)) and np.allclose(b.jac.A, jac.A)
        assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == jac_a.A)
