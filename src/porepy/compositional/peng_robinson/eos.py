"""This module contains compiled versions of the Peng-Robinson EoS functionalities,
and related functions.

The functions provided here are building on lambdified expressions in
:mod:`~porepy.compositional.peng_robinson.eos_symbolic`.

Important:
    Importing this module for the first time triggers numba compilation with static
    signatures for a multitude of functions.

    This takes a considerable amount of time.

"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Sequence, TypeVar

import numba as nb
import numpy as np

from .._core import NUMBA_CACHE, NUMBA_FAST_MATH, NUMBA_PARALLEL
from ..eos_compiler import EoSCompiler, ScalarFunction, VectorFunction
from ..materials import FluidComponent
from . import eos_symbolic

# Import explicitely to avoid some issues in numba (referencing vars internally).
from .eos_symbolic import A_CRIT, B_CRIT
from .utils import thd_function_type

__all__ = [
    "characteristic_residual",
    "get_root_case",
    "is_extended_root",
    "critical_line",
    "widom_line",
    "compressibility_factor",
    "PengRobinsonCompiler",
]


logger = logging.getLogger(__name__)


_NUMBA_STATIC_COMPILE_KWARGS: dict[str, Any] = {
    "fastmath": NUMBA_FAST_MATH,
    "cache": NUMBA_CACHE,
}
"""Shortcut to frequently used numba-compiler flags."""


_T = TypeVar("_T", float, np.ndarray)
"""Type variable for functions where the operations are analogous for floats and arrays.
"""


_import_msg: str = "(import peng_robinson/eos.py)"
_import_start = time.time()

logger.info(f"{_import_msg} Compiling Peng-Robinson EoS from symbolic source ..")


# region Functions related to the characteristic polynomial and its roots


logger.debug(f"{_import_msg} Compiling cubic polynomial ..")


coeff_0: Callable[[_T, _T], _T] = nb.njit(
    [
        nb.f8(nb.f8, nb.f8),
        nb.f8[:](nb.f8[:], nb.f8[:]),
    ],
    **_NUMBA_STATIC_COMPILE_KWARGS,
)(eos_symbolic.coeff_0)
"""NJIT-ed version of :func:`~porepy.compositional.peng_robinson.eos_symbolic.coeff_0`.

Signature: ``(float64, float64) -> float64``. Accepts vectorized input.

"""


coeff_1: Callable[[_T, _T], _T] = nb.njit(
    [
        nb.f8(nb.f8, nb.f8),
        nb.f8[:](nb.f8[:], nb.f8[:]),
    ],
    **_NUMBA_STATIC_COMPILE_KWARGS,
)(eos_symbolic.coeff_1)
"""NJIT-ed version of :func:`~porepy.compositional.peng_robinson.eos_symbolic.coeff_1`.

Signature: ``(float64, float64) -> float64``. Accepts vectorized input.

"""


coeff_2: Callable[[_T, _T], _T] = nb.njit(
    [
        nb.f8(nb.f8, nb.f8),
        nb.f8[:](nb.f8[:], nb.f8[:]),
    ],
    **_NUMBA_STATIC_COMPILE_KWARGS,
)(eos_symbolic.coeff_2)
"""NJIT-ed version of :func:`~porepy.compositional.peng_robinson.eos_symbolic.coeff_2`.

Signature: ``(float64, float64) -> float64``. Accepts vectorized input.

"""


# NOTE: Due to internal usage of already compiled function for non-reduced coefficients
# the formulas for the reduced coefficients must be re-implemented and we cannot
# directly compile the respective functions.
@nb.njit(
    [
        nb.f8(nb.f8, nb.f8),
        nb.f8[:](nb.f8[:], nb.f8[:]),
    ],
    **_NUMBA_STATIC_COMPILE_KWARGS,
)
def reduced_coeff_0(A: _T, B: _T) -> _T:
    """NJIT-ed version of :func:`~porepy.compositional.peng_robinson.eos_symbolic.
    reduced_coeff_0`.

    Signature: ``(float64, float64) -> float64``. Accepts vectorized input.

    """
    c2 = coeff_2(A, B)
    return c2**3 * (2.0 / 27.0) - c2 * coeff_1(A, B) * (1.0 / 3.0) + coeff_0(A, B)


@nb.njit(
    [
        nb.f8(nb.f8, nb.f8),
        nb.f8[:](nb.f8[:], nb.f8[:]),
    ],
    **_NUMBA_STATIC_COMPILE_KWARGS,
)
def reduced_coeff_1(A: _T, B: _T) -> _T:
    """NJIT-ed version of :func:`~porepy.compositional.peng_robinson.eos_symbolic.
    reduced_coeff_1`.

    Signature: ``(float64, float64) -> float64``. Accepts vectorized input.

    """
    return coeff_1(A, B) - coeff_2(A, B) ** 2 * (1.0 / 3.0)


discriminant: Callable[[_T, _T], _T] = nb.njit(
    [
        nb.f8(nb.f8, nb.f8),
        nb.f8[:](nb.f8[:], nb.f8[:]),
    ],
    **_NUMBA_STATIC_COMPILE_KWARGS,
)(eos_symbolic.discriminant)
"""NJIT-ed version of :func:`~porepy.compositional.peng_robinson.eos_symbolic.discriminant`.

Signature: ``(float64, float64) -> float64``. Accepts vectorized input.

"""


@nb.njit(nb.i1(nb.f8, nb.f8, nb.f8), **_NUMBA_STATIC_COMPILE_KWARGS)
def _get_root_case(A: float, B: float, eps: float) -> int:
    """Internal, scalar function for :data:`get_root_case`.

    Note:
        Due to logical operations performed, it is easier (for numba) to split the
        scalar and vector functions into two entities.

    """
    q = reduced_coeff_0(A, B)
    r = reduced_coeff_1(A, B)
    d = discriminant(q, r)

    # if discriminant is positive, the polynomial has one real root
    if d > eps:
        return 1
    # if discriminant is negative, the polynomial has three distinct real roots
    if d < -eps:
        return 3
    # if discrimant is zero, we are in the degenerate case
    else:
        # if first reduced coefficient is zero, the polynomial has a triple root
        # the critical point is a known triple root
        if np.abs(r) < eps or (np.abs(A - A_CRIT) < eps and np.abs(B - B_CRIT) < eps):
            return 0
        # if first reduced coefficient is not zero, the polynomial has 2 real roots
        # one with multiplicity 2
        # the zero point (A=B=0) is one such case.
        else:
            return 2


@nb.njit(
    nb.i1[:](nb.f8[:], nb.f8[:], nb.f8),
    parallel=NUMBA_PARALLEL,
    **_NUMBA_STATIC_COMPILE_KWARGS,
)
def get_root_case(A: np.ndarray, B: np.ndarray, eps: float) -> np.ndarray:
    """A piece-wise constant function dependent on non-dimensional cohesion and covolume,
    representing the number of roots of the characteristic polynomial in terms of
    cohesion and covolume.

    Function with signature ``(float64, float64, float64) -> int8``.
    Can be called with vectorized input for ``A,B``.

    See also:

        `Cardano formula <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_ .

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.
        eps: Numerical zero to detect degenerate polynomials (zero discriminant), e.g.
            ``1e-14``.

    Returns:
        An integer indicating the root case

        - 0 : triple root
        - 1 : 1 real root, 2 complex-conjugated roots
        - 2 : 2 real roots, one with multiplicity 2
        - 3 : 3 distinct real roots

    """
    assert A.shape == B.shape
    root_cases = np.empty(A.shape, dtype=np.int8)
    for i in nb.prange(A.shape[0]):
        root_cases[i] = _get_root_case(A[i], B[i], eps)
    return root_cases


@nb.njit(
    [
        nb.f8(nb.f8, nb.f8, nb.f8),
        nb.f8[:](nb.f8[:], nb.f8[:], nb.f8[:]),
    ],
    **_NUMBA_STATIC_COMPILE_KWARGS,
)
def characteristic_residual(Z: _T, A: _T, B: _T) -> _T:
    r"""Computes the residual of the PR cubic polynomial, for a given root ``Z`` and
    the parametrization in terms of cohesion and co-volume.

    Parameters:
        Z: A supposed root.
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The residual of the characteristic polynomial
        :math:`Z^3 + c_2(A, B) Z^2 + c_1(A, B) Z + c_0(A, B)`.

        If ``Z`` is an actual root, the residual is 0.

    """
    c2 = coeff_2(A, B)
    c1 = coeff_1(A, B)
    c0 = coeff_0(A, B)

    return Z**3 + c2 * Z**2 + c1 * Z + c0


# endregion


# region Functions related to the A-B space


logger.debug(f"{_import_msg} Compiling A-B space functions ..")


@nb.njit(
    [
        nb.f8(nb.f8),
        nb.f8[:](nb.f8[:]),
    ],
    **_NUMBA_STATIC_COMPILE_KWARGS,
)
def critical_line(A: _T) -> _T:
    r"""Parametrization of the critical line for the PR EoS in the A-B space.

    Parameters:
        A: Non-dimensional cohesion.

    Returns:
        The critical line parametrized as ``B(A)``

        .. math::

            \\frac{B_{crit}}{A_{crit}} A

    """
    return (B_CRIT / A_CRIT) * A


@nb.njit(
    [
        nb.f8(nb.f8),
        nb.f8[:](nb.f8[:]),
    ],
    **_NUMBA_STATIC_COMPILE_KWARGS,
)
def widom_line(A: _T) -> _T:
    r"""Parametrization of the Widom-line for the PR EoS in the A-B space.

    Parameters:
        A: Non-dimensional cohesion.

    Returns:
        The Widom-line parametrized as ``B(A)`` in the A-B space

        .. math::

            B_{crit} + 0.8 \cdot 0.3381965009398633 \cdot \left(A - A_{crit}\right)

    """
    return B_CRIT + 0.8 * 0.3381965009398633 * (A - A_CRIT)


@nb.njit(
    [
        nb.f8(
            nb.types.Array(nb.f8, 1, "C", readonly=False),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
        ),
        nb.f8(
            nb.types.Array(nb.f8, 1, "C", readonly=False),
            nb.types.Array(nb.f8, 1, "C", readonly=False),
            nb.types.Array(nb.f8, 1, "C", readonly=False),
        ),
    ],
    cache=True,
)
def point_to_line_distance(p: np.ndarray, lp1: np.ndarray, lp2: np.ndarray) -> float:
    """Computes the distance between a 2-D point and a line spanned by two points.

    NJIT-ed function with signature ``(float64[:], float64[:], float64[:]) -> float64``.

    Parameters:
        p: ``shape=(2,n)``

            Point(s) in 2D space.
        lp1: ``shape=(2,)``

            First point spanning the line.
        lp2: ``shape=(2,)``

            Second point spanning the line.

    Returns:
        Normal distance between ``p`` and the spanned line.

    """

    d = np.sqrt((lp2[0] - lp1[0]) ** 2 + (lp2[1] - lp1[1]) ** 2)
    n = np.abs(
        (lp2[0] - lp1[0]) * (lp1[1] - p[1]) - (lp1[0] - p[0]) * (lp2[1] - lp1[1])
    )
    return n / d


B_CRIT_LINE_POINTS: tuple[np.ndarray, np.ndarray] = (
    np.array([0.0, B_CRIT], dtype=np.float64),
    np.array([A_CRIT, B_CRIT], dtype=np.float64),
)
r"""Two 2D points characterizing the line ``B=B_CRIT`` in the A-B space, namely

.. math::

    (0, B_{crit}),~(A_{crit},B_{crit})

See :data:`B_CRIT`, data:`A_CRIT`.

"""


S_CRIT_LINE_POINTS: tuple[np.ndarray, np.ndarray] = (
    np.zeros(2, dtype=np.float64),
    np.array([A_CRIT, B_CRIT], dtype=np.float64),
)
r"""Two 2D points characterizing the super-critical line in the A-B space, namely

.. math::

    (0,0),~(A_{crit},B_{crit})

See :data:`B_CRIT`, data:`A_CRIT`.

"""


W_LINE_POINTS: tuple[np.ndarray, np.ndarray] = (
    np.array([0.0, widom_line(0)], dtype=np.float64),
    np.array([A_CRIT, widom_line(A_CRIT)], dtype=np.float64),
)
r"""Two 2D points characterizing the Widom-line for water.

The points are created by using :func:`widom_line` for :math:`A\in\{0, A_{crit}\}`.

See :data:`~porepy.compositional.peng_robinson.eos.A_CRIT`.

"""


# endregion


# region Functions related to the compressibility factor
# NJIT compilation of lambdified expressions
# TODO sympy.lambdified functions are source-less and cannot be cached
# find solution for this
# see https://github.com/sympy/sympy/issues/18432
# https://github.com/numba/numba/issues/5128
def _compile_dZ(d_Z_: eos_symbolic.dZ_TYPE) -> Callable[[float, float], np.ndarray]:
    """Helper function to wrap derivatives of compressibility factors into arrays.

    Parameters:
        d_Z_: Callable of derivative of root, dependent on two floats
            (cohesion and covolume).

    Returns:
        NJIT-compiled function with signature ``(float64, float64) -> float64[:]``.

    """
    # make internal function, and external which wraps list into array
    f = nb.njit(d_Z_, cache=False, fastmath=NUMBA_FAST_MATH)

    @nb.njit(nb.f8[:](nb.f8, nb.f8), cache=False, fastmath=NUMBA_FAST_MATH)
    def inner(a, b):
        return np.array(f(a, b), dtype=np.float64)

    return inner


def _compile_Z(Z_: eos_symbolic.Z_TYPE) -> eos_symbolic.Z_TYPE:
    """Helper function to compile expressions representing compressibility factors as
    roots.

    Parameters:
        Z_: Callable for root calculation, dependent on two floats
            (cohesion and covolume).

    Returns:
        NJIT-compiled function with signature ``(float64, float64) -> float64``.

    """
    return nb.njit(nb.f8(nb.f8, nb.f8), cache=False, fastmath=NUMBA_FAST_MATH)(Z_)


logger.debug(f"{_import_msg} Compiling compressibility factors ..")


# Standard compressibility factors and their derivatives

Z_triple = _compile_Z(eos_symbolic.Z_triple_f)
dZ_triple = _compile_dZ(eos_symbolic.dZ_triple_f)

Z_one = _compile_Z(eos_symbolic.Z_one_f)
dZ_one = _compile_dZ(eos_symbolic.dZ_one_f)

Z_double_g = _compile_Z(eos_symbolic.Z_double_g_f)
dZ_double_g = _compile_dZ(eos_symbolic.dZ_double_g_f)
Z_double_l = _compile_Z(eos_symbolic.Z_double_l_f)
dZ_double_l = _compile_dZ(eos_symbolic.dZ_double_l_f)

Z_three_g = _compile_Z(eos_symbolic.Z_three_g_f)
dZ_three_g = _compile_dZ(eos_symbolic.dZ_three_g_f)
Z_three_l = _compile_Z(eos_symbolic.Z_three_l_f)
dZ_three_l = _compile_dZ(eos_symbolic.dZ_three_l_f)
Z_three_i = _compile_Z(eos_symbolic.Z_three_i_f)
dZ_three_i = _compile_dZ(eos_symbolic.dZ_three_i_f)

# extended compressibility factors and their derivatives


@nb.njit(nb.f8(nb.f8, nb.f8), **_NUMBA_STATIC_COMPILE_KWARGS)
def W_subcrit(Z: float, B: float) -> float:
    """Extended compressibility factor in the sub-critical area (Ben Gharbia 2021).

    Parameters:
        Z: The 1 real root.
        B: Dimensionless co-volume.

    Returns:
        :math:`\\frac{1 - B - Z}{2}`

    """
    return (1 - B - Z) * 0.5


@nb.njit(nb.f8[:](nb.f8[:]), **_NUMBA_STATIC_COMPILE_KWARGS)
def dW_subcrit(d_Z: np.ndarray) -> np.ndarray:
    """
    Parameters:
        d_Z: ``shape=(2,)``

            The derivatives of ``Z`` w.r.t. to cohesion and co-volume.

    Returns:
        The derivative of :meth:`W_subcrit` w.r.t. the cohesion and covolume.

    """
    return -0.5 * np.array([d_Z[0], 1 + d_Z[1]])


@nb.njit(nb.f8(nb.f8, nb.f8), **_NUMBA_STATIC_COMPILE_KWARGS)
def W_supercrit_l(Z: float, B: float) -> float:
    """Extended liquid-like compressibility factor in the super-critical region, where
    the gas-like phase is flagged as present.

    Parameters:
        Z: Existing, gas-like compressibility factor.
        B: Dimensionless co-volume.

    Returns:
        :math:`Z + \\frac{B - Z}{2}`

    """

    return Z + (B - Z) * 0.5


@nb.njit(nb.f8[:](nb.f8[:]), **_NUMBA_STATIC_COMPILE_KWARGS)
def dW_supercrit_l(d_Z: np.ndarray) -> np.ndarray:
    """
    Parameters:
        d_Z: ``shape=(2,)``

            The derivatives of ``Z`` w.r.t. to cohesion and co-volume.

    Returns:
        The derivative of :meth:`W_supercrit_l` w.r.t. the cohesion and covolume.

    """
    return 0.5 * np.array([d_Z[0], 1 + d_Z[1]])


@nb.njit(nb.f8(nb.f8, nb.f8), **_NUMBA_STATIC_COMPILE_KWARGS)
def W_supercrit_g(Z: float, B: float) -> float:
    """Extended gas-like compressibility factor in the super-critical region, where
    the liquid-like phase is flagged as present.

    Parameters:
        Z: Existing, liquid-like compressibility factor.
        B: Dimensionless co-volume.

    Returns:
        :math:`B + \\frac{1 - B - Z}{2}`

    """

    return B + (1 - B - Z) * 0.5


@nb.njit(nb.f8[:](nb.f8[:]), **_NUMBA_STATIC_COMPILE_KWARGS)
def dW_supercrit_g(d_Z: np.ndarray) -> np.ndarray:
    """
    Parameters:
        d_Z: ``shape=(2,)``

            The derivatives of ``Z`` w.r.t. to cohesion and co-volume.

    Returns:
        The derivative of :meth:`W_supercrit_g` w.r.t. the cohesion and covolume.

    """
    return -0.5 * np.array([d_Z[0], d_Z[1] - 1])


logger.debug(f"{_import_msg} Compiling general compressibility factor ..")


@nb.njit(nb.i1(nb.f8, nb.f8, nb.i1, nb.f8), **_NUMBA_STATIC_COMPILE_KWARGS)
def _is_extended_root(A: float, B: float, gaslike: int, eps: float) -> int:
    """Internal, scalar function for :data:`is_extended_root`."""
    nroot = _get_root_case(A, B, eps)
    # super critical check
    is_supercritical = B >= critical_line(A) or B >= B_CRIT
    # below widom -> gas-like root is extended
    is_below_widom = B < widom_line(A)

    is_extended = 0  # default return value is 0, actual root.

    # Classical extension case where only 1 root is present
    if nroot == 1:
        # in the supercritical area, the Water widom line is currently used
        if is_supercritical:
            # below the widom line, the gas-like root is asymmetrically extended
            if is_below_widom and gaslike:
                is_extended = 1
            # above the Widom line, the liquid-like root is asymmetrically extended
            elif not is_below_widom and not gaslike:
                is_extended = 1
        # in the sub-critical area, the approach by Ben Gharbia is used
        # smaller root is liquid-like
        else:
            z = Z_one(A, B)
            w = W_subcrit(z, B)
            if w < z and not gaslike:
                is_extended = 1
            elif w >= z and gaslike:
                is_extended = 1
    # special case for 3-root region outside the sub-critical area:
    # liquid always extended
    elif nroot == 3:
        if is_supercritical and not gaslike:
            is_extended = 1

    return is_extended


@nb.njit(
    nb.i1[:](nb.f8[:], nb.f8[:], nb.i1, nb.f8),
    parallel=NUMBA_PARALLEL,
    **_NUMBA_STATIC_COMPILE_KWARGS,
)
def is_extended_root(
    A: np.ndarray, B: np.ndarray, gaslike: int, eps: float
) -> np.ndarray:
    """Checks if a configuration of gas-like flag, cohesion and covolume would lead to
    an extended root.

    If True, an extension procedure was applied, i.e. the compressibility factor
    is not an actual root of the characteristic polynomial.

    Numpy-universal function with signature ``(float64, float64, int8, float64) -> int8``.
    Can be called with vectorized input for ``A,B``.

    Note:
        Argument ``gaslike`` must be also vectorized, if ``A`` and ``B`` are vectorized.
        This is due to some numba-related peculiarities.
        ``eps`` doesn't have to be vectorized.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.
        gaslike: 1 if a gas-like root is assumed, 0 otherwise.
        eps: Numerical zero, used to determine the root case (see :data:`get_root_case`).

    Returns:
        1, if the root is an extended root of the polynomial, 0 if it is an actual root.

    """
    assert A.shape == B.shape
    Z = np.empty(A.shape, dtype=np.int8)
    for i in nb.prange(A.shape[0]):
        Z[i] = _is_extended_root(A[i], B[i], gaslike, eps)
    return Z


@nb.njit(nb.f8(nb.f8, nb.f8, nb.f8, nb.f8), cache=True)
def _gas_smoother(Z_L: float, Z_i: float, Z_G: float, s: float) -> float:
    """Auxiliary function to compute the weight for convex-combination of gas root
    and the average of gas and intermediate root in the physical 2-phase region."""

    # proximity:
    # If close to 1, intermediate root is close to gas root.
    # If close to 0, intermediate root is close to liquid root.
    # values bound by [0,1]
    d = (Z_i - Z_L) / (Z_G - Z_L)

    w = (d - (1 - 2 * s)) / s
    w = w**2 * (3 - 2 * w)
    if d >= 1 - s:
        w = 1.0
    elif d <= 1 - 2 * s:
        w = 0.0

    return w


@nb.njit(nb.f8(nb.f8, nb.f8, nb.f8, nb.f8), cache=True)
def _liq_smoother(Z_L: float, Z_i: float, Z_G: float, s: float) -> float:
    """Auxiliary function to compute the weight for convex-combination of liquid root
    and the average of liquid and intermediate root in the physical 2-phase region."""
    # NOTE See gas smoother for explanation
    d = (Z_i - Z_L) / (Z_G - Z_L)

    w = (d - s) / s
    w = -(w**2) * (3 - 2 * w) + 1.0
    if d <= s:
        w = 1.0
    elif d >= 2 * s:
        w = 0.0

    return w


@nb.njit(
    [
        nb.f8(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8),
        nb.f8[:](nb.f8[:], nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8),
    ],
    cache=NUMBA_CACHE,
)
def _smooth_asymmetric_liq_extension(
    W_scl: float | np.ndarray,
    W_scg: float | np.ndarray,
    W_sub: float | np.ndarray,
    A: float,
    B: float,
    s: float,
) -> float | np.ndarray:
    """Resolves the representation of the asymetrically extended liquid-like root, or
    its derivatives."""

    AB_point = np.array([A, B])

    W = W_scl  # use the assymetric extension by default

    # computing distance to Widom-line,
    # which separates gas and liquid in supercrit area
    d = point_to_line_distance(
        AB_point,
        W_LINE_POINTS[0],
        W_LINE_POINTS[1],
    )
    if d < s and B >= B_CRIT:
        d_n = d / s
        W = W_scg * (1 - d_n) + W * d_n

    # Computing distance to supercritical line,
    # which separates sub- and supercritical liquid extension
    d = point_to_line_distance(
        AB_point,
        S_CRIT_LINE_POINTS[0],
        S_CRIT_LINE_POINTS[1],
    )
    if d < s and B < B_CRIT:
        d_n = d / s
        W = W_sub * (1 - d_n) + W * d_n

    return W


@nb.njit(
    [
        nb.f8(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8),
        nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8),
    ],
    cache=NUMBA_CACHE,
)
def _smooth_asymmetric_gas_extension(
    W_scg: float | np.ndarray,
    W_sub: float | np.ndarray,
    A: float,
    B: float,
    s: float,
) -> float | np.ndarray:
    """Resolves the representation of the asymetrically extended gas-like root, or
    its derivatives."""
    AB_point = np.array([A, B])

    W = W_scg  # use the assymetric extension by default

    d = point_to_line_distance(
        AB_point,
        B_CRIT_LINE_POINTS[0],
        B_CRIT_LINE_POINTS[1],
    )
    if d < s:
        d_n = d / s
        W = W_sub * (1 - d_n) + W * d_n

    return W


@nb.njit(nb.f8(nb.f8, nb.f8, nb.i1, nb.f8, nb.f8, nb.f8), cache=True)
def _Z_from_AB(
    A: float,
    B: float,
    gaslike: int,
    eps: float,
    smooth_e: float,
    smooth_3: float,
) -> float:
    """Internal, scalar function for :func:`compressibility_factor`."""

    # determine number of roots
    nroot = _get_root_case(A, B, eps)
    # sub critical area as defined by Ben Gharbia et al. (2021)
    is_subcritical = (B < B_CRIT) and (B < critical_line(A))

    if nroot == 1:
        Z_1_real = Z_one(A, B)
        # Extension procedure according Ben Gharbia et al.
        if is_subcritical:
            W = W_subcrit(Z_1_real, B)
            # gas like roots are always given by the bigger one
            if gaslike:
                return W if W >= Z_1_real else Z_1_real
            else:
                return W if W < Z_1_real else Z_1_real
        # Asymetric extension in super-critical area
        else:
            # For asymetric extension in supercritical area:
            # below widom -> gas-like root is extended
            # NOTE TODO This holds only for water-like mixtures
            if B < widom_line(A):
                if gaslike:
                    W = W_supercrit_g(Z_1_real, B)

                    if smooth_e > 0.0:
                        W_sub = W_subcrit(Z_1_real, B)
                        W = _smooth_asymmetric_gas_extension(W, W_sub, A, B, smooth_e)

                    return W
                else:
                    return Z_1_real
            else:
                if gaslike:
                    return Z_1_real
                else:
                    W = W_supercrit_l(Z_1_real, B)

                    if smooth_e > 0.0:
                        W_scg = W_supercrit_g(Z_1_real, B)
                        W_sub = W_subcrit(Z_1_real, B)
                        W = _smooth_asymmetric_liq_extension(
                            W, W_scg, W_sub, A, B, smooth_e
                        )

                return W
    elif nroot == 2:
        return Z_double_g(A, B) if gaslike else Z_double_l(A, B)
    elif nroot == 3:
        # Physical 2-phase region with 3 roots fulfilling lower bound by B
        if is_subcritical:
            # smoothing according Ben Gharbia et al., in physical 2-phase region
            if smooth_3 > 0.0:
                Z_l = Z_three_l(A, B)
                Z_i = Z_three_i(A, B)
                Z_g = Z_three_g(A, B)

                # gas root smoothing
                if gaslike:
                    v_g = _gas_smoother(Z_l, Z_i, Z_g, smooth_3)
                    return Z_g * (1 - v_g) + (Z_i + Z_g) * 0.5 * v_g
                # liquid root smoothing
                else:
                    v_l = _liq_smoother(Z_l, Z_i, Z_g, smooth_3)
                    return Z_l * (1 - v_l) + (Z_i + Z_l) * 0.5 * v_l
            else:
                return Z_three_g(A, B) if gaslike else Z_three_l(A, B)
        # There is a super-critical region with 3 roots, where the liquid-like root
        # violates the lower bound by B -> asymetric extension of liquid-like root
        else:
            Z_gas = Z_three_g(A, B)
            if gaslike:
                return Z_gas
            else:
                W = W_supercrit_l(Z_gas, B)

                if smooth_e > 0.0:
                    W_scg = W_supercrit_g(Z_gas, B)
                    W_sub = W_subcrit(Z_gas, B)
                    W = _smooth_asymmetric_liq_extension(
                        W, W_scg, W_sub, A, B, smooth_e
                    )

                return W
    else:
        return Z_triple(A, B)


@nb.njit(
    nb.f8[:](nb.f8[:], nb.f8[:], nb.i1, nb.f8, nb.f8, nb.f8),
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_CACHE,
)
def compressibility_factor(
    A: np.ndarray,
    B: np.ndarray,
    phasetype: int,
    eps: float,
    smooth_e: float,
    smooth_3: float,
) -> np.ndarray:
    """Root-case insensitive computation of the (extended) compressibility factor
    depending on A and B.

    It determins the root case, depending on A and B, and applies the correct formula to
    obtain the root. It also computes the extended root, if it turns out to be required.

    To check if a root is extended, see :func:`is_extended`.

    Numpy-universal function with signature
    ``(float64, float64, int8, float64, float64, float64) -> float64``.
    Can be called with vectorized input for ``A,B``.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.
        gaslike: 0 if the computation should return the liquid-like root, 1 for the
            gas-like root.
        eps: Numerical zero, used to determine the root case
            (see :func:`get_root_case`).
        smooth_e: Width of smoothing area around borders between areas of different
            extension procedures, e.g. ``1e-2``.
            Set to 0. to turn of this moothing.
        smooth_3: Width of area in the subcritical 2-phase/ 3-root region for smoothing
            according Ben Gharbia et al. (2021) (e.g. ``1e-4``).
            Set to 0. to turn it of.

    Returns:
        The (extended) compressibility factor.

    """
    assert A.shape == B.shape
    Z = np.empty_like(A)
    for i in nb.prange(A.shape[0]):
        Z[i] = _Z_from_AB(A[i], B[i], phasetype, eps, smooth_e, smooth_3)
    return Z


@nb.njit(nb.f8[:](nb.f8, nb.f8, nb.i1, nb.f8, nb.f8, nb.f8), cache=True)
def _dZ_dAB(
    A: float,
    B: float,
    gaslike: int,
    eps: float,
    smooth_e: float,
    smooth_3: float,
) -> np.ndarray:
    """Analogoues to :func:`_Z_from_AB`, only returns the derivatives of the
    compressibility factor w.r.t. ``A`` and ``B`` in an array."""

    # determine number of roots
    nroot = _get_root_case(A, B, eps)
    # sub critical area as defined by Ben Gharbia et al. (2021)
    is_subcritical = (B < B_CRIT) and (B < critical_line(A))

    if nroot == 1:
        d_Z_1_real = dZ_one(A, B)
        Z_1_real = Z_one(A, B)
        # Extension procedure according Ben Gharbia et al.
        if is_subcritical:
            d_W = dW_subcrit(d_Z_1_real)
            W = W_subcrit(Z_1_real, B)
            # gas like roots are always given by the bigger one
            if gaslike:
                return d_W if W >= Z_1_real else d_Z_1_real
            else:
                return d_W if W < Z_1_real else d_Z_1_real
        # Asymetric extension in super-critical area
        else:
            # For asymetric extension in supercritical area:
            # below widom -> gas-like root is extended
            # NOTE TODO This holds only for water-like mixtures
            if B < widom_line(A):
                if gaslike:
                    d_W = dW_supercrit_g(d_Z_1_real)

                    if smooth_e > 0.0:
                        d_W_sub = dW_subcrit(d_Z_1_real)
                        d_W = _smooth_asymmetric_gas_extension(
                            d_W, d_W_sub, A, B, smooth_e
                        )

                    return d_W
                else:
                    return d_Z_1_real
            else:
                if gaslike:
                    return d_Z_1_real
                else:
                    d_W = dW_supercrit_l(d_Z_1_real)

                    if smooth_e > 0.0:
                        d_W_scg = dW_supercrit_g(d_Z_1_real)
                        d_W_sub = dW_subcrit(d_Z_1_real)
                        d_W = _smooth_asymmetric_liq_extension(
                            d_W, d_W_scg, d_W_sub, A, B, smooth_e
                        )

                    return d_W
    elif nroot == 2:
        return dZ_double_g(A, B) if gaslike else dZ_double_l(A, B)
    elif nroot == 3:
        # Physical 2-phase region with 3 roots fulfilling lower bound by B
        if is_subcritical:
            # smoothing according Ben Gharbia et al., in physical 2-phase region
            if smooth_3 > 0.0:
                Z_l = Z_three_l(A, B)
                Z_i = Z_three_i(A, B)
                Z_g = Z_three_g(A, B)

                d_Z_l = dZ_three_l(A, B)
                d_Z_i = dZ_three_i(A, B)
                d_Z_g = dZ_three_g(A, B)

                # gas root smoothing
                if gaslike:
                    v_g = _gas_smoother(Z_l, Z_i, Z_g, smooth_3)
                    return d_Z_g * (1 - v_g) + (d_Z_i + d_Z_g) * 0.5 * v_g
                # liquid root smoothing
                else:
                    v_l = _liq_smoother(Z_l, Z_i, Z_g, smooth_3)
                    return d_Z_l * (1 - v_l) + (d_Z_i + d_Z_l) * 0.5 * v_l
            else:
                return dZ_three_g(A, B) if gaslike else dZ_three_l(A, B)
        # There is a super-critical region with 3 roots, where the liquid-like root
        # violates the lower bound by B -> asymetric extension of liquid-like root
        else:
            d_Z_gas = dZ_three_g(A, B)
            if gaslike:
                return d_Z_gas
            else:
                d_W = dW_supercrit_l(d_Z_gas)

                if smooth_e > 0.0:
                    d_W_scg = dW_supercrit_g(d_Z_gas)
                    d_W_sub = dW_subcrit(d_Z_gas)
                    d_W = _smooth_asymmetric_liq_extension(
                        d_W, d_W_scg, d_W_sub, A, B, smooth_e
                    )

                return d_W
    else:
        return dZ_triple(A, B)


# endregion


def _compile_fugacities(
    phis: Callable[[float, float, np.ndarray, float, float, float], np.ndarray],
) -> Callable[[float, float, np.ndarray, float, float, float], np.ndarray]:
    """Helper function to compile the vector of fugacity coefficients.

    It needs an additional reduction of shape from ``(num_comp, 1)`` to ``(num_comp,)``
    because of the usage of a symbolic, vector-valued function."""
    f = nb.njit(phis)

    @nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8[:], nb.f8, nb.f8, nb.f8))
    def inner(p_, T_, X_, A_, B_, Z_):
        phi_ = f(p_, T_, X_, A_, B_, Z_)
        return phi_[:, 0]

    return inner


def _compile_thd_function_derivatives(
    thd_df: Callable[[float, float, np.ndarray], list[float]],
) -> Callable[[float, float, np.ndarray], np.ndarray]:
    """Helper function to compile the gradient of a thermodynamic function.

    Functions are supposed to take pressure, temperature and a vector of
    fractions as arguments.

    This helper function ensures that the return value is wrapped in an array, and not
    a list (as by default returned when using sympy.lambdify).

    It also enforces a signature ``(float64, float64, float64[:]) -> float64[:]``

    """
    df = nb.njit(thd_df, fastmath=NUMBA_FAST_MATH)

    @nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8[:]), fastmath=NUMBA_FAST_MATH)
    def inner(p_, T_, X_):
        return np.array(df(p_, T_, X_), dtype=np.float64)

    return inner


def _compile_extended_thd_function_derivatives(
    ext_thd_df: Callable[[float, float, np.ndarray, float, float, float], list[float]],
) -> Callable[[float, float, np.ndarray, float, float, float], np.ndarray]:
    """Helper function to compile the gradient of an extended thermodynamic function.

    Functions are supposed to take pressure, temperature, a vector of
    fractions, and the EoS specific terms cohesion, covolume and
    compressibility factor as arguments.

    This helper function ensures that the return value is wrapped in an array, and not
    a list (as by default returned when using sympy.lambdify).

    It also enforces a signature
    ``(float64, float64, float64[:], float64, float64, float64) -> float64[:]``

    """
    df = nb.njit(ext_thd_df)

    @nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8[:], nb.f8, nb.f8, nb.f8))
    def inner(p_, T_, X_, A_, B_, Z_):
        return np.array(df(p_, T_, X_, A_, B_, Z_), dtype=np.float64)

    return inner


def _compile_volume_derivative(
    dv: Callable[[float, float, float], list[float]],
) -> Callable[[float, float, float], np.ndarray]:
    """Helper function to compile the gradient of the specific volume.

    Required to wrap the result in an array.

    It also enforces a signature ``(float64, float64, float64) -> float64[:]``.

    """

    dv_ = nb.njit(fastmath=NUMBA_FAST_MATH)(dv)

    @nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8))
    def inner(p_, T_, Z_):
        return np.array(dv_(p_, T_, Z_), dtype=np.float64)

    return inner


# NOTE notation: The suffix *_c is used for dynamically compiled function variables to
# avoid confusion.
class PengRobinsonCompiler(EoSCompiler):
    """Class providing compiled computations of thermodynamic quantities for the
    Peng-Robinson EoS.

    Parameters:
        components: A list of ``num_comp`` component instances.
        ideal_enthalpies: A list of ``num_comp`` callables representing the ideal
            enthalpies of individual components in ``components``.
        bip_matrix: A 2D array containing BIPs for ``components``. Note that only the
            upper triangle of this matrix is used due to expected symmetry.

    """

    def __init__(
        self,
        components: Sequence[FluidComponent],
        ideal_enthalpies: Sequence[thd_function_type],
        bip_matrix: np.ndarray,
        params: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__(components)

        self._cfuncs: dict[str, Callable] = dict()
        """A collection of internally required, compiled callables"""

        default_params: dict[str, float] = {
            "smoothing_multiphase": 1e-4,
            "smoothing_extension": 1e-2,
            "eps": 1e-14,
        }
        if params is None:
            params = {}
        default_params.update(params)

        self.params: dict[str, float] = default_params
        """Parameters for the equation of state.

        Once set, the parameters are not changable after compilation.

        List of parameters:

        - ``'eps'``: Numerical tolerance for zero. Applied in search for roots of the
          cubic polynomial.
        - ``'smoothing_factor'``: smoothing factor for compressibility factors when
          approaching phase transition. If zero, no smoothing is performed.

        """

        self.symbolic = eos_symbolic.PengRobinsonSymbolic(
            components, ideal_enthalpies, bip_matrix
        )
        """Symbolic representation of the EoS, providing expressions and derivatives
        for properties, which are turned into functions and compiled."""

    def compile(self) -> None:
        """Child method compiles essential functions from symbolic part before calling
        the parent class compiler"""

        logger.info("Compiling Peng-Robinson EoS ..")
        start = time.time()

        B_c = nb.njit(
            nb.f8(nb.f8, nb.f8, nb.f8[:]),
            fastmath=NUMBA_FAST_MATH,
        )(self.symbolic.B_f)
        logger.debug("Compiling symbolic functions 1/12")
        dB_c = _compile_thd_function_derivatives(self.symbolic.dB_f)
        logger.debug("Compiling symbolic functions 2/12")

        A_c = nb.njit(nb.f8(nb.f8, nb.f8, nb.f8[:]))(self.symbolic.A_f)
        logger.debug("Compiling symbolic functions 3/12")
        dA_c = _compile_thd_function_derivatives(self.symbolic.dA_f)
        logger.debug("Compiling symbolic functions 4/12")

        phi_c = _compile_fugacities(self.symbolic.phi_f)
        logger.debug("Compiling symbolic functions 5/12")
        dphi_c = nb.njit(nb.f8[:, :](nb.f8, nb.f8, nb.f8[:], nb.f8, nb.f8, nb.f8))(
            self.symbolic.dphi_f
        )
        logger.debug("Compiling symbolic functions 6/12")

        h_dep_c = nb.njit(nb.f8(nb.f8, nb.f8, nb.f8[:], nb.f8, nb.f8, nb.f8))(
            self.symbolic.h_dep_f
        )
        logger.debug("Compiling symbolic functions 7/12")
        h_ideal_c = nb.njit(nb.f8(nb.f8, nb.f8, nb.f8[:]))(self.symbolic.h_ideal_f)
        logger.debug("Compiling symbolic functions 8/12")
        dh_dep_c = _compile_extended_thd_function_derivatives(self.symbolic.dh_dep_f)
        logger.debug("Compiling symbolic functions 9/12")
        dh_ideal_c = _compile_thd_function_derivatives(self.symbolic.dh_ideal_f)
        logger.debug("Compiling symbolic functions 10/12")

        rho_c = nb.njit(
            nb.f8(nb.f8, nb.f8, nb.f8),
            fastmath=NUMBA_FAST_MATH,
        )(self.symbolic.rho_f)
        logger.debug("Compiling symbolic functions 11/12")
        drho_c = _compile_volume_derivative(self.symbolic.drho_f)
        logger.debug("Compiling symbolic functions 12/12")

        self._cfuncs.update(
            {
                "A": A_c,
                "B": B_c,
                "dA": dA_c,
                "dB": dB_c,
                "phi": phi_c,
                "dphi": dphi_c,
                "h_dep": h_dep_c,
                "h_ideal": h_ideal_c,
                "dh_dep": dh_dep_c,
                "dh_ideal": dh_ideal_c,
                "rho": rho_c,
                "drho": drho_c,
            }
        )

        super().compile()

        logger.info(
            f"{self._nc}-component Peng-Robinson EoS compiled"
            + " (elapsed time: %.5f (s))." % (time.time() - start)
        )

    def get_prearg_for_values(self) -> VectorFunction:
        A_c = self._cfuncs["A"]
        B_c = self._cfuncs["B"]

        eps = self.params["eps"]
        s_m = self.params["smoothing_multiphase"]
        s_e = self.params["smoothing_extension"]

        @nb.njit(nb.f8[:](nb.i1, nb.f8, nb.f8, nb.f8[:]))
        def prearg_val_c(
            phasetype: int, p: float, T: float, xn: np.ndarray
        ) -> np.ndarray:
            prearg = np.empty((3,), dtype=np.float64)
            A = A_c(p, T, xn)
            B = B_c(p, T, xn)

            prearg[0] = A_c(p, T, xn)
            prearg[1] = B_c(p, T, xn)
            prearg[2] = _Z_from_AB(A, B, phasetype, eps, s_e, s_m)

            return prearg

        return prearg_val_c

    def get_prearg_for_derivatives(self) -> VectorFunction:
        A_c = self._cfuncs["A"]
        B_c = self._cfuncs["B"]
        dA_c = self._cfuncs["dA"]
        dB_c = self._cfuncs["dB"]
        # number of derivatives for A, B, Z (p, T, and per component fraction)
        d = 2 + self._nc

        eps = self.params["eps"]
        s_m = self.params["smoothing_multiphase"]
        s_e = self.params["smoothing_extension"]

        @nb.njit(nb.f8[:](nb.i1, nb.f8, nb.f8, nb.f8[:]))
        def prearg_jac_c(
            phasetype: int, p: float, T: float, xn: np.ndarray
        ) -> np.ndarray:
            # the pre-arg for the jacobian contains the derivatives of A, B, Z
            # w.r.t. p, T, and fractions.
            prearg = np.empty((3 * d,), dtype=np.float64)

            A = A_c(p, T, xn)
            B = B_c(p, T, xn)

            dA = dA_c(p, T, xn)
            dB = dB_c(p, T, xn)
            dZ_ = _dZ_dAB(A, B, phasetype, eps, s_e, s_m)
            dZ = dZ_[0] * dA + dZ_[1] * dB

            prearg[0:d] = dA
            prearg[d : 2 * d] = dB
            prearg[2 * d : 3 * d] = dZ

            return prearg

        return prearg_jac_c

    def get_fugacity_function(self) -> VectorFunction:
        phi_c = self._cfuncs["phi"]

        @nb.njit(nb.f8[:](nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def phi_mix_c(
            prearg: np.ndarray, p: float, T: float, xn: np.ndarray
        ) -> np.ndarray:
            return phi_c(p, T, xn, prearg[0], prearg[1], prearg[2])

        return phi_mix_c

    def get_fugacity_derivative_function(self) -> VectorFunction:
        dphi_c = self._cfuncs["dphi"]
        # number of derivatives
        d = 2 + self._nc

        @nb.njit(nb.f8[:, :](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def dphi_mix_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            # computation of phis dependent on A_j, B_j, Z_j
            d_phis = dphi_c(p, T, xn, prearg_val[0], prearg_val[1], prearg_val[2])
            # derivatives of A_j, B_j, Z_j w.r.t. p, T, and X_j
            dA = prearg_jac[0:d]
            dB = prearg_jac[d : 2 * d]
            dZ = prearg_jac[2 * d : 3 * d]
            # expansion of derivatives (chain rule)
            return (
                d_phis[:, :-3]
                + np.outer(d_phis[:, -3], dA)
                + np.outer(d_phis[:, -2], dB)
                + np.outer(d_phis[:, -1], dZ)
            )

        return dphi_mix_c

    def get_enthalpy_function(self) -> ScalarFunction:
        h_dep_c = self._cfuncs["h_dep"]
        h_ideal_c = self._cfuncs["h_ideal"]

        @nb.njit(nb.f8(nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def h_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> np.ndarray:
            return h_ideal_c(p, T, xn) + h_dep_c(
                p, T, xn, prearg[0], prearg[1], prearg[2]
            )

        return h_c

    def get_enthalpy_derivative_function(self) -> VectorFunction:
        d = 2 + self._nc
        dh_dep_c = self._cfuncs["dh_dep"]
        dh_ideal_c = self._cfuncs["dh_ideal"]

        @nb.njit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def dh_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            d_h_ideal = dh_ideal_c(p, T, xn)
            d_h_dep = dh_dep_c(p, T, xn, prearg_val[0], prearg_val[1], prearg_val[2])
            # derivatives of A_j, B_j, Z_j w.r.t. p, T, and X_j
            dA = prearg_jac[0:d]
            dB = prearg_jac[d : 2 * d]
            dZ = prearg_jac[2 * d : 3 * d]
            # expansion of derivatives of departure enthalpy (chain rule)
            d_h_dep = (
                d_h_dep[:-3] + d_h_dep[-3] * dA + d_h_dep[-2] * dB + d_h_dep[-1] * dZ
            )
            return d_h_ideal + d_h_dep

        return dh_c

    def get_density_function(self) -> ScalarFunction:
        rho_c_ = self._cfuncs["rho"]

        @nb.njit(nb.f8(nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def rho_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> np.ndarray:
            return rho_c_(p, T, prearg[2])

        return rho_c

    def get_density_derivative_function(self) -> VectorFunction:
        d = 2 + self._nc
        drho_c_ = self._cfuncs["drho"]

        @nb.njit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def drho_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            d_rho_ = drho_c_(p, T, prearg_val[2])
            # derivatives of Z_j w.r.t. p, T, and X_j
            dZ = prearg_jac[2 * d : 3 * d]
            # expansion of derivatives (chain rule)
            d_rho = d_rho_[-1] * dZ
            d_rho[:2] += d_rho_[:2]  # contribution of p, T derivatives
            return d_rho

        return drho_c

    # TODO need models for below functions
    def get_viscosity_function(self) -> ScalarFunction:
        @nb.njit(nb.f8(nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def mu_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            return 1.0

        return mu_c

    def get_viscosity_derivative_function(self) -> VectorFunction:
        @nb.njit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def dmu_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            return np.zeros(2 + xn.shape[0], dtype=np.float64)

        return dmu_c

    def get_conductivity_function(self) -> ScalarFunction:
        @nb.njit(nb.f8(nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def kappa_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            return 1.0

        return kappa_c

    def get_conductivity_derivative_function(self) -> VectorFunction:
        @nb.njit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def dkappa_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            return np.zeros(2 + xn.shape[0], dtype=np.float64)

        return dkappa_c


logger.info(
    f"{_import_msg} Done" + " (elapsed time: %.5f (s))." % (time.time() - _import_start)
)

del _import_start, _import_msg
