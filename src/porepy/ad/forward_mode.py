import numpy as np
import scipy.sparse as sps


class Ad_array():

    def __init__(self, val=1.0, jac=0.0):
        self.val = val
        self.jac = jac

    def __add__(self, other):
        b = _cast(other)
        c = Ad_array()
        c.val = self.val + b.val
        c.jac = self.jac + b.jac
        return c

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        b = _cast(other).copy()
        b.val = -b.val
        b.jac = -b.jac
        return self + b

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        if not isinstance(other, Ad_array): # other is scalar
            val = self.val * other
            jac = self.jac * other
        else:
            val = self.val * other.val
            jac = self.l_jac_mul(other.val) + other.l_jac_mul(self.val)
        return Ad_array(val, jac)

    def __rmul__(self, other):
        if not isinstance(other, Ad_array): # other is scalar
            val = other * self.val
            jac = other * self.jac
            return Ad_array(val, jac)
        val =  other.val * self.val
        jac = other.val * self.jac + self.val * other.jac
        return Ad_array(val, jac)

    def __pow__(self, other):
        if not isinstance(other, Ad_array):
            val = self.val**other
            jac = self.l_jac_mul(other * self.val**(other - 1))
        else:
            val = self.val**other.val
            jac = self.l_jac_mul(other.val * self.val**(other.val - 1)) \
                + other.l_jac_mul(self.val **other.val * np.log(self.val))
        return Ad_array(val, jac)

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        b = self.copy()
        b.val = -b.val
        b.jac = -b.jac
        return b

    def copy(self):
        b = Ad_array()
        try:
            b.val = self.val.copy()
        except AttributeError:
            b.val = self.val
        try:
            b.jac = self.jac.copy()
        except AttributeError:
            b.jac = self.jac
        return b

    def l_jac_mul(self, a):
        try:
            A = sps.diags(a)
        except TypeError:
            A = a
        return A * self.jac

    def r_jac_mul(self, a):
        try:
            A = sps.diags(a)
        except TypeError:
            A = a
        return self.jac * A
    

def _cast(variables):
    if isinstance(variables, list):
        out_var = []
        for var in variables:
            if isinstance(var, Ad_array):
                out_var.append(var)
            else:
                out_var.append(Ad_array(var))
    else:
        if isinstance(variables, Ad_array):
            out_var = variables
        else:
            out_var = Ad_array(variables)
    return out_var
        

a = Ad_array(1,1)
b = 2
c = a + b
