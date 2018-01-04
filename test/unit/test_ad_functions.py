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
        assert a.val==1 and a.jac == 0

    def test_exp_advar(self):
        a = Ad_array(2,3)
        b = af.exp(a)
        assert b.val == np.exp(2) and b.jac == 3 * np.exp(2)
        assert a.val==2 and a.jac == 3

    def test_exp_vector(self):
        val = np.array([1, 2, 3])
        J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
        a = Ad_array(val, J)
        b = af.exp(a)
        jac = np.dot(np.diag(np.exp(val)), J)
        
        assert np.all(b.val == np.exp(val)) and np.all(b.jac == jac)

    def test_exp_sparse_jac(self):
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.exp(a)
        jac = np.dot(np.diag(np.exp(val)),J.A)
        assert np.all(b.val == np.exp(val)) and np.all(b.jac == jac)

    def teste_exp_scalar_times_ad_var(self):
        val = np.array([1,2,3])
        J = sps.diags(np.array([1,1,1]))
        a = Ad_array(val, J)
        c = 2
        b = af.exp(c * a)
        jac = c * sps.diags(np.exp(c * val)) * J
        
        assert np.allclose(b.val, np.exp(c * val)) and np.allclose(b.jac.A, jac.A)
        assert np.all(a.val == [1,2,3]) and np.all(a.jac.A == J.A)
