"""This module contains utility functions for the composite subpackage.

It also contains some older code kept until ready for removal.

"""

from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np

import porepy as pp

__all__ = [
    "safe_sum",
    "truncexp",
    "trunclog",
    "CompositeModellingError",
]


NumericType = Union[pp.number, np.ndarray, pp.ad.AdArray]
"""DEPRECATED: Delete with finalization of flash and eos update to numba. TODO"""


def truncexp(var):
    if isinstance(var, pp.ad.AdArray):
        trunc = var > 700
        val = np.exp(var.val, where=(~trunc))
        val[trunc] = np.exp(700)
        der = var._diagvec_mul_jac(val)
        return pp.ad.AdArray(val, der)
    else:
        trunc = var > 700
        val = np.exp(var, where=(~trunc))
        val[trunc] = np.exp(700)
        return val


def trunclog(var, eps):
    if isinstance(var, pp.ad.AdArray):
        trunc_val = np.maximum(var.val, eps)
        val = np.log(trunc_val)
        der = var._diagvec_mul_jac(1 / trunc_val)
        return pp.ad.AdArray(val, der)
    else:
        return np.log(np.maximum(var, eps))


def safe_sum(x: Sequence[Any]) -> Any:
    """Auxiliary method to safely sum the elements, without creating
    a first addition with 0 (important for AD operators to avoid overhead)."""
    if len(x) >= 1:
        sum_ = x[0]
        # need to deep copy to avoid change by reference
        if isinstance(sum_, (np.ndarray, pp.ad.AdArray)):
            sum_ = sum_.copy()
        for i in range(1, len(x)):
            sum_ = sum_ + x[i]
        return sum_
    else:
        return 0


class CompositeModellingError(Exception):
    """Custom exception class to alert the user when using the composite framework
    logically inconsistent."""
