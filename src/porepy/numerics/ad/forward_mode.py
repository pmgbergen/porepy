import numpy as np
import scipy.sparse as sps

import porepy as pp

__all__ = ["initAdArrays", "Ad_array"]

module_sections = ["assembly", "numerics"]


@pp.time_logger(sections=module_sections)
def initAdArrays(variables):
    if not isinstance(variables, list):
        try:
            num_val = variables.size
        except AttributeError:
            num_val = 1
        return Ad_array(variables, sps.diags(np.ones(num_val)).tocsc())

    num_val = [v.size for v in variables]
    ad_arrays = []
    for i, val in enumerate(variables):
        # initiate zero jacobian
        n = num_val[i]
        jac = [sps.csc_matrix((n, m)) for m in num_val]
        # set jacobian of variable i to I
        jac[i] = sps.diags(np.ones(num_val[i])).tocsc()
        # initiate Ad_array
        jac = sps.bmat([jac])
        ad_arrays.append(Ad_array(val, jac))

    return ad_arrays


class Ad_array:
    @pp.time_logger(sections=module_sections)
    def __init__(self, val=1.0, jac=0.0):
        self.val = val
        self.jac = jac

    def __repr__(self) -> str:
        s = f"Ad array of size {self.val.size}\n"
        s += f"Jacobian is of size {self.jac.shape} and has {self.jac.data.size} elements"
        return s

    @pp.time_logger(sections=module_sections)
    def __add__(self, other):
        b = _cast(other)
        c = Ad_array()
        c.val = self.val + b.val
        c.jac = self.jac + b.jac
        return c

    @pp.time_logger(sections=module_sections)
    def __radd__(self, other):
        return self.__add__(other)

    @pp.time_logger(sections=module_sections)
    def __sub__(self, other):
        b = _cast(other).copy()
        b.val = -b.val
        b.jac = -b.jac
        return self + b

    @pp.time_logger(sections=module_sections)
    def __rsub__(self, other):
        return -self.__sub__(other)

    @pp.time_logger(sections=module_sections)
    def __lt__(self, other):
        return self.val < _cast(other).val

    @pp.time_logger(sections=module_sections)
    def __le__(self, other):
        return self.val <= _cast(other).val

    @pp.time_logger(sections=module_sections)
    def __gt__(self, other):
        return self.val > _cast(other).val

    @pp.time_logger(sections=module_sections)
    def __ge__(self, other):
        return self.val >= _cast(other).val

    @pp.time_logger(sections=module_sections)
    def __eq__(self, other):
        return self.val == _cast(other).val

    @pp.time_logger(sections=module_sections)
    def __mul__(self, other):
        if not isinstance(other, Ad_array):  # other is scalar
            val = self.val * other
            if isinstance(other, np.ndarray):
                jac = self.diagvec_mul_jac(other)
            else:
                jac = self._jac_mul_other(other)
        else:
            val = self.val * other.val
            jac = self.diagvec_mul_jac(other.val) + other.diagvec_mul_jac(self.val)
        return Ad_array(val, jac)

    @pp.time_logger(sections=module_sections)
    def __rmul__(self, other):
        if isinstance(other, Ad_array):
            # other is Ad_var, so should have called __mul__
            raise RuntimeError("Somthing went horrible wrong")
        val = other * self.val
        jac = self._other_mul_jac(other)
        return Ad_array(val, jac)

    @pp.time_logger(sections=module_sections)
    def __pow__(self, other):
        if not isinstance(other, Ad_array):
            val = self.val ** other
            jac = self.diagvec_mul_jac(other * self.val ** (other - 1))
        else:
            val = self.val ** other.val
            jac = self.diagvec_mul_jac(
                other.val * self.val ** (other.val - 1)
            ) + other.diagvec_mul_jac(self.val ** other.val * np.log(self.val))
        return Ad_array(val, jac)

    @pp.time_logger(sections=module_sections)
    def __rpow__(self, other):
        if isinstance(other, Ad_array):
            raise ValueError(
                "Somthing went horrible wrong, should" "have called __pow__"
            )

        val = other ** self.val
        jac = self.diagvec_mul_jac(other ** self.val * np.log(other))
        return Ad_array(val, jac)

    @pp.time_logger(sections=module_sections)
    def __truediv__(self, other):
        return self * other ** -1

    @pp.time_logger(sections=module_sections)
    def __neg__(self):
        b = self.copy()
        b.val = -b.val
        b.jac = -b.jac
        return b

    @pp.time_logger(sections=module_sections)
    def __len__(self):
        return len(self.val)

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def diagvec_mul_jac(self, a):
        try:
            A = sps.diags(a)
        except TypeError:
            A = a

        if isinstance(self.jac, np.ndarray):
            return np.array([A * J for J in self.jac])
        else:
            return A * self.jac

    @pp.time_logger(sections=module_sections)
    def jac_mul_diagvec(self, a):
        try:
            A = sps.diags(a)
        except TypeError:
            A = a
        if isinstance(self.jac, np.ndarray):
            return np.array([J * A for J in self.jac])
        else:
            return self.jac * A

    @pp.time_logger(sections=module_sections)
    def full_jac(self):
        return self.jac

    #        return sps.hstack(self.jac[:])

    @pp.time_logger(sections=module_sections)
    def _other_mul_jac(self, other):
        return other * self.jac

    #        return np.array([other * J for J in self.jac])

    @pp.time_logger(sections=module_sections)
    def _jac_mul_other(self, other):
        return self.jac * other


#        return np.array([J * other for J in self.jac])


@pp.time_logger(sections=module_sections)
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
