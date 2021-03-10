import numpy as np
import scipy.sparse as sps

from porepy.numerics.ad.forward_mode import Ad_array

__all__ = ["initLocalAdArrays", "Local_Ad_array"]


def initLocalAdArrays(variables):
    if not isinstance(variables, list):
        try:
            num_val = variables.size
        except AttributeError:
            num_val = 1
        return Local_Ad_array(variables, np.ones(num_val))

    num_val = [v.size for v in variables]
    ad_arrays = []
    for i, val in enumerate(variables):
        # initiate zero jacobian
        jac = [np.zeros(num_val[i]) for m in num_val]
        # set jacobian of variable i to I
        jac[i] = np.ones(num_val[i])
        # initiate Ad_array
        jac = np.array(
            jac
        )  # TODO use some more efficient data structure? There is many zeros.
        ad_arrays.append(Local_Ad_array(val, jac))

    return ad_arrays


class Local_Ad_array:
    """ Special case of Ad_array. Essentially with a diagonal Jacobian."""

    def __init__(self, val=1.0, jac=0.0):
        self.val = val
        self.jac = jac

    def __repr__(self) -> str:
        s = f"Ad array of size {self.val.size}\n"
        s += f"Jacobian is of size {self.jac.shape} and has {self.jac.size} elements"
        return s

    def __add__(self, other):
        if isinstance(other, Ad_array):
            raise RuntimeError(
                "Operations between (normal) Ad_array and Local_Ad_array not implemented."
            )
        if isinstance(other, Local_Ad_array):
            val = self.val + other.val
            jac = self.jac + other.jac
        else:
            val = self.val + other
            jac = self.jac
        return Local_Ad_array(val, jac)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Ad_array):
            raise RuntimeError(
                "Operations between (normal) Ad_array and Local_Ad_array not implemented."
            )
        b = Local_Ad_array()
        if isinstance(other, Local_Ad_array):
            b.val = -other.val
            b.jac = -other.jac
            return self + b
        else:
            b.val = self.val - other
            b.jac = self.jac
            return b

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Ad_array):
            raise RuntimeError(
                "Operations between (normal) Ad_array and Local_Ad_array not implemented."
            )
        if not isinstance(other, Local_Ad_array):  # other is scalar
            val = self.val * other
            jac = self.jac * other
        else:
            val = self.val * other.val
            jac = self.val * other.jac + other.val * self.jac
        return Local_Ad_array(val, jac)

    def __rmul__(self, other):
        if isinstance(other, Ad_array):
            raise RuntimeError(
                "Operations between (normal) Ad_array and Local_Ad_array not implemented."
            )
        if isinstance(other, Local_Ad_array):
            # other is Ad_var, so should have called __mul__
            raise RuntimeError("Somthing went horrible wrong")
        val = other * self.val
        jac = other * self.jac
        return Local_Ad_array(val, jac)

    def __pow__(self, other):
        if isinstance(other, Ad_array):
            raise RuntimeError(
                "Operations between (normal) Ad_array and Local_Ad_array not implemented."
            )
        if not isinstance(other, Local_Ad_array):
            val = self.val ** other
            jac = other * self.val ** (other - 1) * self.jac
        else:
            val = self.val ** other.val
            jac = other.val * self.val ** (other.val - 1) * self.jac
            +self.val ** other.val * np.log(self.val) * other.jac
        return Local_Ad_array(val, jac)

    def __rpow__(self, other):
        if isinstance(other, Ad_array):
            raise RuntimeError(
                "Operations between (normal) Ad_array and Local_Ad_array not implemented."
            )
        if isinstance(other, Local_Ad_array):
            raise ValueError(
                "Somthing went horrible wrong, should" "have called __pow__"
            )

        val = other ** self.val
        jac = other ** self.val * np.log(other) * self.jac
        return Local_Ad_array(val, jac)

    def __truediv__(self, other):
        if isinstance(other, Ad_array):
            raise RuntimeError(
                "Operations between (normal) Ad_array and Local_Ad_array not implemented."
            )
        return self * other ** -1

    def __neg__(self):
        b = self.copy()
        b.val = -b.val
        b.jac = -b.jac
        return b

    def __getitem__(self, key):
        return Local_Ad_array(self.val[key], self.jac[key])

    def copy(self):
        b = Local_Ad_array()
        try:
            b.val = self.val.copy()
        except AttributeError:
            b.val = self.val
        try:
            b.jac = self.jac.copy()
        except AttributeError:
            b.jac = self.jac
        return b

    def full_jac(self):
        return self.jac

    def makeAd_array(self):
        return Ad_array(self.val, sps.diags(self.jac).tocsc())
