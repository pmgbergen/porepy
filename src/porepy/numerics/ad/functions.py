from typing import Callable

import numpy as np
import scipy.sparse as sps

from porepy.numerics.ad.forward_mode import Ad_array
from porepy.numerics.ad.local_forward_mode import Local_Ad_array

import porepy  # noqa isort: skip


__all__ = [
    "exp",
    "log",
    "sign",
    "abs",
    "sin",
    "cos",
    "tanh",
    "heaviside",
    "RegularizedHeaviside",
]


def exp(var):
    if isinstance(var, Ad_array):
        val = np.exp(var.val)
        der = var.diagvec_mul_jac(np.exp(var.val))
        return Ad_array(val, der)
    elif isinstance(var, Local_Ad_array):
        val = np.exp(var.val)
        der = np.exp(var.val) * var.jac
        return Local_Ad_array(val, der)
    else:
        return np.exp(var)


def log(var):
    if isinstance(var, Ad_array):
        val = np.log(var.val)
        der = var.diagvec_mul_jac(1 / var.val)
        return Ad_array(val, der)
    elif isinstance(var, Local_Ad_array):
        val = np.log(var.val)
        der = (1 / var.val) * var.jac
        return Local_Ad_array(val, der)
    else:
        return np.log(var)


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
    elif isinstance(var, Local_Ad_array):
        val = np.abs(var.val)
        jac = sign(var) * var.jac
        return Local_Ad_array(val, jac)
    else:
        return np.abs(var)


def sin(var):
    if isinstance(var, Ad_array):
        val = np.sin(var.val)
        # TODO use capabilties offered by forward_mode.py
        jac = sps.diags(np.cos(var.val)).tocsc() * var.jac
        return Ad_array(val, jac)
    elif isinstance(var, Local_Ad_array):
        val = np.sin(var.val)
        jac = np.cos(var.val) * var.jac
        return Local_Ad_array(val, jac)
    else:
        return np.sin(var)


def cos(var):
    if isinstance(var, Ad_array):
        val = np.cos(var.val)
        # TODO use capabilties offered by forward_mode.py
        jac = -sps.diags(np.sin(var.val)).tocsc() * var.jac
        return Ad_array(val, jac)
    elif isinstance(var, Local_Ad_array):
        val = np.cos(var.val)
        jac = -np.sin(var.val) * var.jac
        return Local_Ad_array(val, jac)
    else:
        return np.cos(var)


def tanh(var):
    if isinstance(var, Ad_array):
        val = np.tanh(var.val)
        jac = sps.diags(1 - np.tanh(var.val) ** 2).tocsc() * var.jac
        return Ad_array(val, jac)
    elif isinstance(var, Local_Ad_array):
        val = np.tanh(var.val)
        jac = (1 - np.tanh(var.val) ** 2) * var.jac
        return Local_Ad_array(val, jac)
    else:
        return np.tanh(var)


def heaviside(var, zerovalue: float = 0.5):
    if isinstance(var, Ad_array) or isinstance(var, Local_Ad_array):
        return np.heaviside(var.val, zerovalue)
    else:
        return np.heaviside(var, zerovalue)


class RegularizedHeaviside:
    def __init__(self, regularization: Callable):
        self._regularization = regularization

    def __call__(self, var, zerovalue: float = 0.5):
        if isinstance(var, Ad_array):
            val = np.heaviside(var.val, 0.0)
            regularization = self._regularization(var)
            jac = regularization.jac
            return Ad_array(val, jac)
        elif isinstance(var, Local_Ad_array):
            val = np.heaviside(var.val, 0.0)
            regularization = self._regularization(var)
            jac = regularization.jac
            return Local_Ad_array(val, jac)
        else:
            return np.heaviside(var)
