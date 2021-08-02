from typing import Callable

import numpy as np

from porepy.numerics.ad.forward_mode import Ad_array

import porepy  # noqa isort: skip


__all__ = [
    "exp",
    "log",
    "sign",
    "abs",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "heaviside",
    "heaviside_smooth",
    "RegularizedHeaviside",
]


# %% Exponential and logarithmic functions
def exp(var):
    if isinstance(var, Ad_array):
        val = np.exp(var.val)
        der = var.diagvec_mul_jac(np.exp(var.val))
        return Ad_array(val, der)
    else:
        return np.exp(var)


def log(var):
    if isinstance(var, Ad_array):
        val = np.log(var.val)
        der = var.diagvec_mul_jac(1 / var.val)
        return Ad_array(val, der)
    else:
        return np.log(var)


# %% Sign and absolute value functions
def sign(var):
    if not isinstance(var, Ad_array):
        return np.sign(var)
    else:
        return np.sign(var.val)


def abs(var):
    if isinstance(var, Ad_array):
        val = np.abs(var.val)
        jac = var.diagvec_mul_jac(sign(var))
        return Ad_array(val, jac)
    else:
        return np.abs(var)


# %% Trigonometric functions
def sin(var):
    if isinstance(var, Ad_array):
        val = np.sin(var.val)
        jac = var.diagvec_mul_jac(np.cos(var.val))
        return Ad_array(val, jac)
    else:
        return np.sin(var)


def cos(var):
    if isinstance(var, Ad_array):
        val = np.cos(var.val)
        jac = var.diagvec_mul_jac(-np.sin(var.val))
        return Ad_array(val, jac)
    else:
        return np.cos(var)


def tan(var):
    if isinstance(var, Ad_array):
        val = np.tan(var.val)
        jac = var.diagvec_mul_jac((np.cos(var.val) ** 2) ** (-1))
        return Ad_array(val, jac)
    else:
        return np.tan(var)


def arcsin(var):
    if isinstance(var, Ad_array):
        val = np.arcsin(var.val)
        jac = var.diagvec_mul_jac((1 - var.val ** 2) ** (-0.5))
        return Ad_array(val, jac)
    else:
        return np.arcsin(var)


def arccos(var):
    if isinstance(var, Ad_array):
        val = np.arccos(var.val)
        jac = var.diagvec_mul_jac(-((1 - var.val ** 2) ** (-0.5)))
        return Ad_array(val, jac)
    else:
        return np.arccos(var)


def arctan(var):
    if isinstance(var, Ad_array):
        val = np.arctan(var.val)
        jac = var.diagvec_mul_jac((var.val ** 2 + 1) ** (-1))
        return Ad_array(val, jac)
    else:
        return np.arctan(var)


# %% Hyperbolic functions
def sinh(var):
    if isinstance(var, Ad_array):
        val = np.sinh(var.val)
        jac = var.diagvec_mul_jac(np.cosh(var.val))
        return Ad_array(val, jac)
    else:
        return np.sinh(var)


def cosh(var):
    if isinstance(var, Ad_array):
        val = np.cosh(var.val)
        jac = var.diagvec_mul_jac(np.sinh(var.val))
        return Ad_array(val, jac)
    else:
        return np.cosh(var)


def tanh(var):
    if isinstance(var, Ad_array):
        val = np.tanh(var.val)
        jac = var.diagvec_mul_jac(np.cosh(var.val) ** (-2))
        return Ad_array(val, jac)
    else:
        return np.tanh(var)


def arcsinh(var):
    if isinstance(var, Ad_array):
        val = np.arcsinh(var.val)
        jac = var.diagvec_mul_jac((var.val ** 2 + 1) ** (-0.5))
        return Ad_array(val, jac)
    else:
        return np.arcsinh(var)


def arccosh(var):
    if isinstance(var, Ad_array):
        val = np.arccosh(var.val)
        den1 = (var.val - 1) ** (-0.5)
        den2 = (var.val + 1) ** (-0.5)
        jac = var.diagvec_mul_jac(den1 * den2)
        return Ad_array(val, jac)
    else:
        return np.arccosh(var)


def arctanh(var):
    if isinstance(var, Ad_array):
        val = np.arctanh(var.val)
        jac = var.diagvec_mul_jac((1 - var.val ** 2) ** (-1))
        return Ad_array(val, jac)
    else:
        return np.arctanh(var)


# %% Step and Heaviside functions
def heaviside(var, zerovalue: float = 0.5):
    if isinstance(var, Ad_array):
        return np.heaviside(var.val, zerovalue)
    else:
        return np.heaviside(var, zerovalue)


def heaviside_smooth(var, eps: float = 1e-3):
    """
    Smooth (regularized) version of the Heaviside function.

    Parameters
    ----------
    var : Ad_array or ndarray
        Input array.
    eps : float, optional
        Regularization parameter. The default is 1E-3. The function will
        convergence to the Heaviside function in the limit when eps --> 0

    Returns
    -------
    Ad_array or ndarray (depending on the input)
        Regularized heaviside function (and its Jacobian if applicable).

    Note
    _____
    The analytical expression for the smooth version Heaviside function reads:
        H_eps(x) = (1/2) * (1 + (2/pi) * arctan(x/eps)),
    with its derivative smoothly approximating the Dirac delta function:
        d(H(x))/dx = delta_eps = (1/pi) * (eps / (eps^2 + x^2)).

    Reference: https://ieeexplore.ieee.org/document/902291

    """
    if isinstance(var, Ad_array):
        val = 0.5 * (1 + 2 * np.pi ** (-1) * np.arctan(var.val * eps ** (-1)))
        jac = var.diagvec_mul_jac(
            np.pi ** (-1) * eps * (eps ** 2 + var.val ** 2) ** (-1)
        )
        return Ad_array(val, jac)
    else:
        return 0.5 * (1 + 2 * np.pi ** (-1) * np.arctan(var * eps ** (-1)))


class RegularizedHeaviside:
    def __init__(self, regularization: Callable):
        self._regularization = regularization

    def __call__(self, var, zerovalue: float = 0.5):
        if isinstance(var, Ad_array):
            val = np.heaviside(var.val, 0.0)
            regularization = self._regularization(var)
            jac = regularization.jac
            return Ad_array(val, jac)
        else:
            return np.heaviside(var)  # type: ignore
