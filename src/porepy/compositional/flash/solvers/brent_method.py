"""Module implementing Brent's method, a derivative-free method to find the root of a
scalar function."""

from __future__ import annotations

from typing import Callable, Literal, TypeAlias

import numba as nb
import numpy as np

from ._core import SOLVER_PARAMETERS_TYPE

__all__ = [
    "DEFAULT_BRENT_PARAMS",
    "brent",
]


_BRENT_PARAMS_KEYS: TypeAlias = Literal["brent_max_iterations", "brent_tolerance"]
"""Keys (names) for brent method parameters."""


DEFAULT_BRENT_PARAMS: dict[_BRENT_PARAMS_KEYS, float] = {
    "brent_max_iterations": 100,
    "brent_tolerance": 1e-16,
}
"""Default parameters for :func:`brent`.

- ``'brent_max_iterations': 100.`` maximal number of iterations.
- ``'brent_tolerance': 1e-16``: Convergence criterion for root finding.

"""


BRENT_METHOD_SIGNATURE = nb.types.Tuple((nb.f8, nb.i4, nb.i4))(
    nb.typeof(nb.cfunc("f8(f8)")(lambda x: x)),
    nb.f8,
    nb.f8,
    SOLVER_PARAMETERS_TYPE,
)
"""Numba-signature for the brent method for compilation."""


@nb.njit(BRENT_METHOD_SIGNATURE, cache=True)
def brent(
    f: Callable[[float], float], a: float, b: float, params: dict[str, float]
) -> tuple[float, int, int]:
    """Classical Brent method to find a root of ``f``, bracketed by ``a`` and ``b``.

    Implementation is taken from reference below.

    For more information on the numba types of the signature, see
    :data:`BRENT_METHOD_SIGNATURE`.

    References:
        Press, William H. 2007. Numerical Recipes : The Art of Scientific Computing.
        3rd ed. Cambridge: Cambridge University Press.

    Parameters:
        f: Scalar function with ``f(a)`` and ``f(b)`` having opposite signs.
        a: Left bracket to the supposed root.
        b: Right bracket to the supposed root.
        params: A dictionary containing ``'max_iterations'`` and ``'tolerance'``.

    Raises:
        ValueError: If ``f(a)`` and ``f(b)`` have the same sign.

    Returns:
        A 3-tuple containing

        - a float, the root
        - an integer, the failure flag (0 means success)
        - an integer, the number of iterations.

    """

    max_iter = int(params["brent_max_iterations"])
    tol = float(params["brent_tolerance"])

    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f must have opposite signs at interval endpoints [a, b].")

    c = b
    fc = fb

    # declaration of intermediate values
    d: float
    e: float
    p: float
    q: float
    r: float
    s: float
    tol1: float
    min1: float
    min2: float
    xm: float

    # default return values (failure == 1 means failure by reaching max iter)
    # root is assumed to be stored in b
    iter_num = 0
    failure = 1

    for i in range(max_iter):
        iter_num = i
        # Adjusting bounding intervals
        if fb * fc > 0.0:
            c = a
            fc = fa
            e = b - a
            d = e
        if np.abs(fc) < np.abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        # tolerance for convergence check
        tol1 = 2.0 * np.finfo(np.float64).eps * np.abs(b) + 0.5 * tol
        # tol1 = tol
        xm = 0.5 * (c - b)
        # Check convergence
        if np.abs(xm) <= tol1 or fb == 0.0:
            failure = 0
            break
        # Inverse quadratic interpolation
        if np.abs(e) >= tol1 and np.abs(fa) > np.abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            # Check if in bounds
            if p > 0.0:
                q = -q
            p = np.abs(p)
            min1 = 3.0 * xm * q - np.abs(tol1 * q)
            min2 = np.abs(e * q)  # TODO
            # Accepting interpolation
            if 2.0 * p < np.minimum(min1, min2):
                e = d
                d = p / q
            # Interpolation failed, use bisection
            else:
                d = xm
                e = d
        # Bounds decrease too slowly, use bisection
        else:
            d = xm
            e = d

        a = b
        fa = fb
        if np.abs(d) > tol1:
            b += d
        else:
            b += np.sign(xm) * tol1
        fb = f(b)

    # for loop finished
    return b, failure, iter_num
