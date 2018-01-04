import numpy as np
import scipy.sparse as sps
import unittest

from porepy.ad.forward_mode import Ad_array

class AdTest(unittest.TestCase):

    def test_add_two_scalars(self):
        a = Ad_array(1, 0)
        b = Ad_array(-10, 0)
        c = a + b
        assert c.val == -9 and c.jac == 0
        assert a.val == 1 and a.jac == 0
        assert b.val == -10 and b.jac==0

    def test_add_two_ad_variables(self):
        a = Ad_array(4, 1.0)
        b = Ad_array(9, 3)
        c = a + b
        assert np.allclose(c.val, 13) and np.allclose(c.jac, 4.0)
        assert a.val == 4 and np.allclose(a.jac, 1.0)
        assert b.val == 9 and b.jac==3
        
    def test_add_var_with_scal(self):
        a = Ad_array(3, 2)
        b = 3
        c = a + b
        assert np.allclose(c.val, 6) and np.allclose(c.jac, 2)
        assert a.val == 3 and np.allclose(a.jac, 2)
        assert b == 3

    def test_add_scal_with_var(self):
        a = Ad_array(3, 2)
        b = 3
        c = b + a
        assert np.allclose(c.val, 6) and np.allclose(c.jac, 2)
        assert a.val == 3 and a.jac == 2
        assert b == 3

    def test_sub_two_scalars(self):
        a = Ad_array(1, 0)
        b = Ad_array(3, 0)
        c = a - b
        assert c.val == -2 and c.jac == 0
        assert a.val == 1 and a.jac == 0
        assert b.val == 3 and a.jac == 0

    def test_sub_two_ad_variables(self):
        a = Ad_array(4, 1.0)
        b = Ad_array(9, 3)
        c = a - b
        assert np.allclose(c.val, -5) and np.allclose(c.jac, -2)
        assert a.val == 4 and np.allclose(a.jac, 1.0)
        assert b.val == 9 and b.jac == 3

    def test_sub_var_with_scal(self):
        a = Ad_array(3, 2)
        b = 3
        c = a - b
        assert np.allclose(c.val, 0) and np.allclose(c.jac, 2)
        assert a.val == 3 and a.jac == 2
        assert b == 3

    def test_sub_scal_with_var(self):
        a = Ad_array(3, 2)
        b = 3
        c = b - a
        assert np.allclose(c.val, 0) and np.allclose(c.jac, -2)
        assert a.val == 3 and a.jac ==2
        assert b == 3

    def test_mul_scal_ad_scal(self):
        a = Ad_array(3, 0)
        b = Ad_array(2, 0)
        c = a * b
        assert c.val == 6 and c.jac == 0
        assert a.val == 3 and a.jac == 0
        assert b.val == 2 and b.jac == 0

    def test_mul_ad_var_ad_scal(self):
        a = Ad_array(3, 3)
        b = Ad_array(2, 0)
        c = a * b
        assert c.val == 6 and c.jac == 6
        assert a.val == 3 and a.jac == 3
        assert b.val == 2 and b.jac == 0

    def test_mul_ad_var_ad_var(self):
        a = Ad_array(3, 3)
        b = Ad_array(2, -4)
        c = a * b
        assert c.val == 6 and c.jac == -6
        assert a.val == 3 and a.jac == 3
        assert b.val == 2 and b.jac == -4

    def test_mul_ad_var_scal(self):
        a = Ad_array(3, 3)
        b = 3
        c = a * b
        assert c.val == 9 and c.jac == 9
        assert a.val == 3 and a.jac == 3
        assert b == 3

    def test_mul_scar_ad_var(self):
        a = Ad_array(3, 3)
        b = 3
        c = b * a 
        assert c.val == 9 and c.jac == 9
        assert a.val == 3 and a.jac == 3
        assert b == 3

    def test_mul_ad_var_mat(self):
        x = Ad_array(np.array([1,2,3]), np.diag([3,2,1]))
        A = np.array([[1,2,3],[4,5,6],[7,8,9]])
        f = x * A
        sol = np.array([[1,4,9],[4,10,18],[7,16,27]])
        jac = np.diag([3, 2, 1]) * A
        assert np.all(f.val == sol) and np.all(f.jac == jac)
        assert np.all(x.val == np.array([1,2,3])) and np.all(x.jac == np.diag([3,2,1]))
        assert np.all(A == np.array([[1,2,3],[4,5,6],[7,8,9]]))

    def test_mul_sps_advar(self):
        J = np.array([[1,3,1],[5,0,0],[5,1,2]])
        x = Ad_array(np.array([1,2,3]), J)
        A = sps.csc_matrix(np.array([[1,2,3],[4,5,6],[7,8,9]]))
        f = A * x
        assert np.all(f.val == [14,32,50])
        assert np.all(f.jac == A*J)

    def test_mul_advar_vectors(self):
        Ja = sps.csc_matrix(np.array([[1,3,1],[5,0,0],[5,1,2]]))
        Jb = np.array([[1,0,0],[0,1,0],[0,0,1]])
        a = Ad_array(np.array([1,2,3]), Ja)
        b = Ad_array(np.array([1,1,1]), Jb)
        A = sps.csc_matrix(np.array([[1,2,3],[4,5,6],[7,8,9]]))
        f = A*a + b

        jac  = A*Ja + Jb
        assert np.all(f.val == [15, 33, 51])
        assert np.all(A*Ja + Jb)
        
    def test_power_advar_scalar(self):
        a = Ad_array(2, 3)
        b = a**2
        assert b.val==4 and b.jac==12

    def test_power_advar_advar(self):
        a = Ad_array(4, 4)
        b = Ad_array(-8, -12 )
        c = a**b
        jac = -(2 + 3*np.log(4)) / 16384
        assert np.allclose(c.val,4**-8) and np.allclose(c.jac, jac)

    def test_div_advar_scalar(self):
        a = Ad_array(10, 6)
        b = 2
        c = a / b
        assert c.val == 5, c.jac == 2

    def test_div_advar_advar(self):
        # a = x ^ 3: b = x^2: x = 2
        a = Ad_array(8, 12)
        b = Ad_array(4, 4)
        c = a / b
        assert c.val == 2 and np.allclose(c.jac, 1)

    def test_copy_scalar(self):
        a = Ad_array(1, 0)
        b = a.copy()
        assert a.val==b.val
        assert a.jac==b.jac
        a.val = 2
        a.jac = 3
        assert b.val == 1
        assert b.jac == 0

    def test_copy_vector(self):
        a = Ad_array(np.ones((3,1)), np.ones((3,1)))
        b = a.copy()
        assert np.allclose(a.val, b.val)
        assert np.allclose(a.jac, b.jac)
        a.val[0] = 3
        a.jac[2] = 4
        assert np.allclose(b.val, np.ones((3,1)))
        assert np.allclose(b.jac, np.ones((3,1)))
