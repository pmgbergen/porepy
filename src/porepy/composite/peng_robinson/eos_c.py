"""This module contains compiled versions of the Peng-Robinson EoS functionalities,
and related, required functions.

The functions provided here are building on lambdified expressions in
:mod:`~porepy.composite.peng_robinson.eos_s`.

The naming convention introduced there is extended here:

- ``_c``: NJIT compiled callable with a specific signature.
- ``_u``: Numpy-universal version (ufunc) of the respective ``_c`` callable, for
  vectorized evaluation.

To extend the example based on the compressibility factor ``Z``:

- ``Z_c`` denotes a compiled, scalar callable with a signature reflexting the dependency
  of ``Z_e``.
- ``Z_u`` denotes the numpy-universal function, a vectorized version of ``Z_c`` since
  a fast and efficient evaluation of the compressibility factor is deemed necessary
  (e.g. in flow problems to compute fluid properties)
- ``d_Z_u`` denotes the numpy universal function of the total derivative of ``Z``,
  w.r.t. to the dependencies given in the call signature of ``Z_c``
- ``dp_Z_c`` denotes a compiled function returning the derivative of ``Z`` w.r.t.
  pressure ``p``, with the same call signature as ``Z_c``.

Important:
    Importing this module for the first time triggers numba NJIT compilation with
    static signatures for a multitude of functions.

    This takes a considerable amount of time.

    Several functions are also implemented as only Python Callables, and then NJIT-ed.

Note:
    - We decided to append ``_c`` to the name of the compiled version of a function,
    which takes non-vectorized input.
    - ``_u`` denotes a numpy-universal function. These versions have been implemented
    for some quantities, where fast, vectorized evaluation is deemed required.
    Not all functions have a ``_u`` version.

"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numba
import numpy as np
from numpy import ndarray

from .._core import NUMBA_CACHE
from ..eos_compiler import EoSCompiler
from .eos_s import (
    A_CRIT,
    B_CRIT,
    B_CRIT_LINE_POINTS,
    S_CRIT_LINE_POINTS,
    W_LINE_POINTS,
    PengRobinsonSymbolic,
    Z_double_g_f,
    Z_double_l_f,
    Z_ext_scg_f,
    Z_ext_scl_f,
    Z_ext_sub_f,
    Z_one_f,
    Z_three_g_f,
    Z_three_i_f,
    Z_three_l_f,
    Z_triple_f,
    coeff_0,
    coeff_1,
    coeff_2,
    critical_line,
    d_Z_double_g_f,
    d_Z_double_l_f,
    d_Z_ext_scg_f,
    d_Z_ext_scl_f,
    d_Z_ext_sub_f,
    d_Z_one_f,
    d_Z_three_g_f,
    d_Z_three_i_f,
    d_Z_three_l_f,
    d_Z_triple_f,
    discriminant,
    point_to_line_distance,
    widom_line,
)
from .pr_components import ComponentPR

__all__ = ["PengRobinsonCompiler"]


logger = logging.getLogger(__name__)


_STATIC_FAST_COMPILE_ARGS: dict[str, Any] = {
    "fastmath": True,
    "cache": NUMBA_CACHE,
}


_import_msg: str = "(import peng_robinson/eos_c.py)"
_import_start = time.time()


# region Functions related to the characteristic polynomial and its roots


logger.debug(f"{_import_msg} Compiling cubic polynomial ..")


coeff_0_c: Callable[[float, float], float] = numba.njit(
    "float64(float64, float64)", **_STATIC_FAST_COMPILE_ARGS
)(coeff_0)
"""NJIT-ed version of :func:`coeff_0`.

Signature: ``(float64, float64) -> float64``

"""


coeff_1_c: Callable[[float, float], float] = numba.njit(
    "float64(float64, float64)", **_STATIC_FAST_COMPILE_ARGS
)(coeff_1)
"""NJIT-ed version of :func:`coeff_1`.

Signature: ``(float64, float64) -> float64``

"""


coeff_2_c: Callable[[float, float], float] = numba.njit(
    "float64(float64, float64)", **_STATIC_FAST_COMPILE_ARGS
)(coeff_2)
"""NJIT-ed version of :func:`coeff_2`.

Signature: ``(float64, float64) -> float64``

"""


@numba.njit("float64(float64, float64)", **_STATIC_FAST_COMPILE_ARGS)
def red_coeff_0_c(A: float, B: float) -> float:
    """NJIT-ed version of :func:`red_coeff_0`.

    Signature: ``(float64, float64) -> float64``

    """
    c2 = coeff_2_c(A, B)
    return c2**3 * (2.0 / 27.0) - c2 * coeff_1_c(A, B) * (1.0 / 3.0) + coeff_0_c(A, B)


@numba.njit("float64(float64, float64)", **_STATIC_FAST_COMPILE_ARGS)
def red_coeff_1_c(A: float, B: float) -> float:
    """NJIT-ed version of :func:`red_coeff_1`.

    Signature: ``(float64, float64) -> float64``

    """
    return coeff_1_c(A, B) - coeff_2_c(A, B) ** 2 * (1.0 / 3.0)


discriminant_c: Callable[[float, float], float] = numba.njit(
    "float64(float64, float64)", **_STATIC_FAST_COMPILE_ARGS
)(discriminant)
"""NJIT-ed version of :func:`discriminant`.

Signature: ``(float64, float64) -> float64``

"""


@numba.njit("int8(float64, float64, float64)", **_STATIC_FAST_COMPILE_ARGS)
def get_root_case_c(A, B, eps=1e-14):
    """A piece-wise cosntant function dependent on
    non-dimensional cohesion and covolume, representing the number of roots
    of the characteristic polynomial in terms of cohesion and covolume.

    NJIT-ed function with signature ``(float64, float64, float64) -> int8``.

    :data:`red_coeff_0_c`, :data:`red_coeff_1_c` and :data:`discriminant_c` are used to
    compute and determine the root case.

    For more information,
    `see here <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_ .


    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.
        eps: ``default=1e-14``

            Numerical zero to detect degenerate polynomials (zero discriminant).

    Returns:
        An integer indicating the root case

        - 0 : triple root
        - 1 : 1 real root, 2 complex-conjugated roots
        - 2 : 2 real roots, one with multiplicity 2
        - 3 : 3 distinct real roots

    """
    q = red_coeff_0_c(A, B)
    r = red_coeff_1_c(A, B)
    d = discriminant_c(q, r)

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


get_root_case_u = numba.vectorize(
    [numba.int8(numba.float64, numba.float64, numba.float64)],
    nopython=True,
    **_STATIC_FAST_COMPILE_ARGS,
)(get_root_case_c)
"""Numpy-universial version of :func:`get_root_case_c`.

Important:
    ``eps`` is not optional any more. Can be made so with a simple wrapper.

"""


@numba.njit("float64(float64,float64,float64)", **_STATIC_FAST_COMPILE_ARGS)
def characteristic_residual_c(Z, A, B):
    r"""NJIT-ed function with signature ``(float64, float64, float64) -> float64``.

    Parameters:
        Z: A supposed root.
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The residual of the characteristic polynomial
        :math:`Z^3 + c_2(A, B) Z^2 + c_1(A, B) Z + c_0(A, B)`.

        If ``Z`` is an actual root, the residual is 0.

    """
    c2 = coeff_2_c(A, B)
    c1 = coeff_1_c(A, B)
    c0 = coeff_0_c(A, B)

    return Z**3 + c2 * Z**2 + c1 * Z + c0


characteristic_residual_u = numba.vectorize(
    [numba.float64(numba.float64, numba.float64, numba.float64)],
    nopython=True,
    **_STATIC_FAST_COMPILE_ARGS,
)(characteristic_residual_c)
"""Numpy-universial version of :func:`characteristic_residual_c`."""


# endregion


# region Functions related to the A-B space

logger.debug(f"{_import_msg} Compiling A-B space functions ..")

critical_line_c: Callable[[float], float] = numba.njit(
    "float64(float64)", **_STATIC_FAST_COMPILE_ARGS
)(critical_line)
"""NJIT-ed version of :func:`critical_line`.

Signature: ``(float64) -> float64``

"""


critical_line_u = numba.vectorize(
    [numba.float64(numba.float64)],
    nopython=True,
    fastmath=True,
    cache=NUMBA_CACHE,
)(critical_line_c)
"""Numpy-universal version of :func:`critical_line_c`."""


widom_line_c: Callable[[float], float] = numba.njit(
    "float64(float64)", fastmath=True, cache=NUMBA_CACHE
)(widom_line)
"""NJIT-ed version of :func:`widom_line`.

Signature: ``(float64) -> float64``

"""


widom_line_u = numba.vectorize(
    [numba.float64(numba.float64)],
    nopython=True,
    fastmath=True,
    cache=NUMBA_CACHE,
)(widom_line_c)
"""Numpy-universal version of :func:`widom_line_c`."""


point_to_line_distance_c = numba.njit(
    [
        numba.float64(
            numba.types.Array(numba.float64, 1, "C", readonly=False),
            numba.types.Array(numba.float64, 1, "C", readonly=True),
            numba.types.Array(numba.float64, 1, "C", readonly=True),
        ),
        numba.float64(
            numba.types.Array(numba.float64, 1, "C", readonly=False),
            numba.types.Array(numba.float64, 1, "C", readonly=False),
            numba.types.Array(numba.float64, 1, "C", readonly=False),
        ),
    ],
    cache=NUMBA_CACHE,  # NOTE no fastmath because of sqrt and abs for small numbers
)(point_to_line_distance)
"""NJIT-ed version of :func:`point_to_line_distance`.

Signatures:

- ``(float64[:], float64[:], float64[:]) -> float64``.
- ``(float64[:], readonly float64[:], readonly float64[:]) -> float64``.

Second signature is for the case the line is constant and defined somewhere.

Important:
    Compared to the original function, this one is only meant for a single point
    given as a 1D array of two floats.

"""


# endregion


# region Functions related to the compressibility factor
# NJIT compilation of lambdified expressions
# TODO sympy.lambdified functions are source-less and cannot be cached
# find solution for this
# see https://github.com/sympy/sympy/issues/18432
# https://github.com/numba/numba/issues/5128
def _compile_d_Z(
    d_Z_: Callable[[float, float], list[float]], fastmath: bool = False
) -> Callable[[float, float], np.ndarray]:
    """Helper function to wrap derivatives of compressibility factors into arrays.

    Parameters:
        d_Z_: Callable of derivative of root, dependent on two floats
            (cohesion and covolume).
        fastmath: for numba (use only for simple expressions)

    Returns:
        NJIT-compiled function with signature ``(float64, float64) -> float64[:]``.

    """
    # make internal function, and external which wraps list into array
    f = numba.njit(d_Z_, cache=False, fastmath=fastmath)

    @numba.njit("float64[:](float64, float64)", cache=False, fastmath=fastmath)
    def inner(a, b):
        return np.array(f(a, b), dtype=np.float64)

    return inner


def _compile_Z(
    Z_: Callable[[float, float], float], fastmath: bool = False
) -> Callable[[float, float], float]:
    """Helper function to compile expressions representing compressibility factors as
    roots.

    Parameters:
        Z_: Callable for root calculation, dependent on two floats
            (cohesion and covolume).
        fastmath: for numba (use only for simple expressions)

    Returns:
        NJIT-compiled function with signature ``(float64, float64) -> float64``.

    """
    return numba.njit("float64(float64, float64)", cache=False, fastmath=fastmath)(Z_)


logger.debug(f"{_import_msg} Compiling compressibility factors ..")


Z_triple_c: Callable[[float, float], float] = _compile_Z(Z_triple_f, fastmath=True)
d_Z_triple_c: Callable[[float, float], np.ndarray] = _compile_d_Z(
    d_Z_triple_f, fastmath=True
)
Z_one_c: Callable[[float, float], float] = _compile_Z(Z_one_f)
d_Z_one_c: Callable[[float, float], np.ndarray] = _compile_d_Z(d_Z_one_f)
Z_ext_sub_c: Callable[[float, float], float] = _compile_Z(Z_ext_sub_f)
d_Z_ext_sub_c: Callable[[float, float], np.ndarray] = _compile_d_Z(d_Z_ext_sub_f)
Z_ext_scg_c: Callable[[float, float], float] = _compile_Z(Z_ext_scg_f)
d_Z_ext_scg_c: Callable[[float, float], np.ndarray] = _compile_d_Z(d_Z_ext_scg_f)
Z_ext_scl_c: Callable[[float, float], float] = _compile_Z(Z_ext_scl_f)
d_Z_ext_scl_c: Callable[[float, float], np.ndarray] = _compile_d_Z(d_Z_ext_scl_f)
Z_double_g_c: Callable[[float, float], float] = _compile_Z(Z_double_g_f)
d_Z_double_g_c: Callable[[float, float], np.ndarray] = _compile_d_Z(d_Z_double_g_f)
Z_double_l_c: Callable[[float, float], float] = _compile_Z(Z_double_l_f)
d_Z_double_l_c: Callable[[float, float], np.ndarray] = _compile_d_Z(d_Z_double_l_f)
Z_three_g_c: Callable[[float, float], float] = _compile_Z(Z_three_g_f)
d_Z_three_g_c: Callable[[float, float], np.ndarray] = _compile_d_Z(d_Z_three_g_f)
Z_three_l_c: Callable[[float, float], float] = _compile_Z(Z_three_l_f)
d_Z_three_l_c: Callable[[float, float], np.ndarray] = _compile_d_Z(d_Z_three_l_f)
Z_three_i_c: Callable[[float, float], float] = _compile_Z(Z_three_i_f)
d_Z_three_i_c: Callable[[float, float], np.ndarray] = _compile_d_Z(d_Z_three_i_f)


logger.debug(f"{_import_msg} Compiling general compressibility factor ..")


@numba.njit("int8(int8, float64, float64, float64)", cache=NUMBA_CACHE, fastmath=True)
def is_real_root(gaslike: int, A: float, B: float, eps: float = 1e-14) -> int:
    """Checks if a configuration of gas-like flag, cohesion and covolume would
    lead to an real root.

    If not, an extension procedure was applied, i.e. the compressibility factor
    is not an actual root of the characteristic polynomial.

    Parameters:
        gaslike: 1 if a gas-like root is assumed, 0 otherwise.
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.
        eps: ``default=1e-14``

            Numerical zero, used to determine the root case
            (see :func:`get_root_case_c`).

    Returns:
        1, if the root is a real root of the polynomial, 0 if it is an extended root.

    """
    nroot = get_root_case_c(A, B, eps)
    # super critical check
    is_sc = B >= critical_line_c(A)
    # below widom -> gas-like root is extended
    below_widom = B <= widom_line_c(A)

    ext = 1  # default return value is 1, real root.

    # only here can an extended representation be used
    if nroot == 1 and below_widom and gaslike:
        ext = 0
    elif nroot == 3 and is_sc and gaslike == 0:
        ext = 0

    return ext


# TODO find proper signature with or without default args
# must be done for Z_c, d_Z_c and compile_Z_mix, compile_d_Z_mix
@numba.njit(
    # "float64(int8, float64, float64, float64, float64, float64)",
    cache=NUMBA_CACHE,
)
def Z_c(
    gaslike: int,
    A: float,
    B: float,
    eps: float = 1e-14,
    smooth_e: float = 1e-2,
    smooth_3: float = 1e-3,
) -> float:
    """Computation of the (extended) compressibility factor depending on A and B.

    NJIT-ed function with signature
    ``(int8, float64, float64, float64, float64, float64) -> float64``

    It determince the root case, depending on A and B, and applies the
    correct formula to obtain the root.
    It also computes the extended root, if it turns out to be required.

    To check if a root is extended, see :func:`is_extended`

    Parameters:
        gaslike: 0 if the computation should return the liquid-like root, 1 for the
            gas-like root.
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.
        eps: ``default=1e-14``

            Numerical zero, used to determine the root case
            (see :func:`get_root_case_c`).
        smooth_e: ``default=1e-2``

            Width of smoothing area around borders between areas of different extension
            procedures.

            Set to 0. to turn of this moothing.
        smooth_3: ``default=1e-3``

            Width of area in the subcritical 2-phase/ 3-root region for smoothing
            according Ben Gharbia et al. (2021)

            Set to 0. to turn it of.

    Returns:
        The (possibly extended) compressibility factor.

    """
    # computed root
    root_out = 0.0
    AB_point = np.array([A, B])

    # super critical check
    is_sc = B >= critical_line_c(A)
    # below widom -> gas-like root is extended
    below_widom = B <= widom_line_c(A)
    # determine number of roots
    nroot = get_root_case_c(A, B, eps)

    if nroot == 1:
        Z_1_real = Z_one_c(A, B)
        # Extension procedure according Ben Gharbia et al.
        # though we use the Widom-line to distinguis between roots, not their size
        if not is_sc and B < B_CRIT:
            W = Z_ext_sub_c(A, B)
            if below_widom:
                root_out = W if gaslike else Z_1_real

            else:
                root_out = Z_1_real if gaslike else W
        # Extension procedure with asymmetric extension of gas
        elif below_widom and B >= B_CRIT:
            if gaslike:
                W = Z_ext_scg_c(A, B)

                # computing distance to border to subcritical extension
                # smooth if close
                d = point_to_line_distance_c(
                    AB_point,
                    B_CRIT_LINE_POINTS[0],
                    B_CRIT_LINE_POINTS[1],
                )
                if smooth_e > 0.0 and d < smooth_e:
                    d_n = d / smooth_e
                    W = Z_ext_sub_c(A, B) * (1 - d_n) + W * d_n

                root_out = W
            else:
                root_out = Z_1_real
        # Extension procedure with asymmetric extension of liquid
        else:
            if gaslike:
                root_out = Z_1_real
            else:
                W = Z_ext_scl_c(A, B)

                # computing distance to Widom-line,
                # which separates gas and liquid in supercrit area
                d = point_to_line_distance_c(
                    AB_point,
                    W_LINE_POINTS[0],
                    W_LINE_POINTS[1],
                )
                if smooth_e > 0.0 and d < smooth_e and B >= B_CRIT:
                    d_n = d / smooth_e
                    W = Z_ext_scg_c(A, B) * (1 - d_n) + W * d_n

                # Computing distance to supercritical line,
                # which separates sub- and supercritical liquid extension
                d = point_to_line_distance_c(
                    AB_point,
                    S_CRIT_LINE_POINTS[0],
                    S_CRIT_LINE_POINTS[1],
                )
                if smooth_e > 0.0 and d < smooth_e and B < B_CRIT:
                    d_n = d / smooth_e
                    W = Z_ext_sub_c(A, B) * (1 - d_n) + W * d_n

                root_out = W
    elif nroot == 2:
        if gaslike > 0:
            root_out = Z_double_g_c(A, B)
        else:
            root_out = Z_double_l_c(A, B)
    elif nroot == 3:
        # triple root area above the critical line is substituted with the
        # extended supercritical liquid-like root
        if is_sc:
            if gaslike:
                root_out = Z_three_g_c(A, B)
            else:
                W = Z_ext_scl_c(A, B)

                # computing distance to Widom-line,
                # which separates gas and liquid in supercrit area
                d = point_to_line_distance_c(
                    AB_point,
                    W_LINE_POINTS[0],
                    W_LINE_POINTS[1],
                )
                if smooth_e > 0.0 and d < smooth_e and B >= B_CRIT:
                    d_n = d / smooth_e
                    W = Z_ext_scg_c(A, B) * (1 - d_n) + W * d_n

                # Computing distance to supercritical line,
                # which separates sub- and supercritical liquid extension
                d = point_to_line_distance_c(
                    AB_point,
                    S_CRIT_LINE_POINTS[0],
                    S_CRIT_LINE_POINTS[1],
                )
                if smooth_e > 0.0 and d < smooth_e and B < B_CRIT:
                    d_n = d / smooth_e
                    W = Z_ext_sub_c(A, B) * (1 - d_n) + W * d_n

                root_out = W
        else:
            # smoothing according Ben Gharbia et al., in physical 2-phase region
            if smooth_3 > 0.0:
                Z_l = Z_three_l_c(A, B)
                Z_i = Z_three_i_c(A, B)
                Z_g = Z_three_g_c(A, B)

                d = (Z_i - Z_l) / (Z_g - Z_l)

                # gas root smoothing
                if gaslike:
                    # gas root smoothing weight
                    v_g = (d - (1 - 2 * smooth_3)) / smooth_3
                    v_g = v_g**2 * (3 - 2 * v_g)
                    if d >= 1 - smooth_3:
                        v_g = 1.0
                    elif d <= 1 - 2 * smooth_3:
                        v_g = 0.0

                    root_out = Z_g * (1 - v_g) + (Z_i + Z_g) * 0.5 * v_g
                # liquid root smoothing
                else:
                    v_l = (d - smooth_3) / smooth_3
                    v_l = -(v_l**2) * (3 - 2 * v_l) + 1.0
                    if d <= smooth_3:
                        v_l = 1.0
                    elif d >= 2 * smooth_3:
                        v_l = 0.0

                    root_out = Z_l * (1 - v_l) + (Z_i + Z_l) * 0.5 * v_l
            else:
                if gaslike:
                    root_out = Z_three_g_c(A, B)
                else:
                    root_out = Z_three_l_c(A, B)
    else:
        root_out = Z_triple_c(A, B)

    return root_out


@numba.guvectorize(
    ["void(int8, float64, float64, float64, float64, float64, float64[:])"],
    "(),(),(),(),(),()->()",
    target="parallel",
    nopython=True,
    cache=NUMBA_CACHE,
)
def _Z_u(
    gaslike,
    A,
    B,
    eps,
    smooth_e,
    smooth_3,
    out,
):
    """Internal ufunc because it cannot have default arguments"""
    out[0] = Z_c(gaslike, A, B, eps, smooth_e, smooth_3)


def Z_u(
    gaslike: bool | int,
    A: float | np.ndarray,
    B: float | np.ndarray,
    eps: float = 1e-14,
    smooth_e: float = 1e-2,
    smooth_3: float = 1e-3,
) -> float | np.ndarray:
    """Numpy-universal function of :func:`Z_c`.

    This is a shallow wrapper of the actual ufunc to allow default arguments.

    """
    # get correct shape independent of format of input
    out = np.empty_like(A + B)
    _Z_u(int(gaslike), A, B, eps, smooth_e, smooth_3, out)
    return out


@numba.njit(
    # "float64[:](int8, float64, float64, float64, float64, float64)",
    cache=NUMBA_CACHE,
)
def d_Z_c(
    gaslike: int,
    A: float,
    B: float,
    eps: float = 1e-14,
    smooth_e: float = 1e-2,
    smooth_3: float = 1e-3,
) -> np.ndarray:
    """Analogoues to :func:`Z_c`, only returns an array instead of the root.
    The array contains the derivative w.r.t. A and B of the root"""

    # derivatives of the root
    dz = np.zeros(2)
    AB_point = np.array([A, B])

    # super critical check
    is_sc = B >= critical_line_c(A)
    # below widom -> gas-like root is extended
    below_widom = B <= widom_line_c(A)
    # determine number of roots
    nroot = get_root_case_c(A, B, eps)

    if nroot == 1:
        d_Z_1_real = d_Z_one_c(A, B)
        # Extension procedure according Ben Gharbia et al.
        if not is_sc and B < B_CRIT:
            W = d_Z_ext_sub_c(A, B)
            if below_widom:
                dz = W if gaslike else d_Z_1_real

            else:
                dz = d_Z_1_real if gaslike else W
        # Extension procedure with asymmetric extension of gas
        elif below_widom and B >= B_CRIT:
            if gaslike:
                W = d_Z_ext_scg_c(A, B)

                # computing distance to border to subcritical extension
                # smooth if close
                d = point_to_line_distance_c(
                    AB_point,
                    B_CRIT_LINE_POINTS[0],
                    B_CRIT_LINE_POINTS[1],
                )
                if smooth_e > 0.0 and d < smooth_e:
                    d_n = d / smooth_e
                    W = d_Z_ext_sub_c(A, B) * (1 - d_n) + W * d_n

                dz = W
            else:
                dz = d_Z_1_real
        # Extension procedure with asymmetric extension of liquid
        else:
            if gaslike:
                dz = d_Z_1_real
            else:
                W = d_Z_ext_scl_c(A, B)

                # computing distance to Widom-line,
                # which separates gas and liquid in supercrit area
                d = point_to_line_distance_c(
                    AB_point,
                    W_LINE_POINTS[0],
                    W_LINE_POINTS[1],
                )
                if smooth_e > 0.0 and d < smooth_e and B >= B_CRIT:
                    d_n = d / smooth_e
                    W = d_Z_ext_scg_c(A, B) * (1 - d_n) + W * d_n

                # Computing distance to supercritical line,
                # which separates sub- and supercritical liquid extension
                d = point_to_line_distance_c(
                    AB_point,
                    S_CRIT_LINE_POINTS[0],
                    S_CRIT_LINE_POINTS[1],
                )
                if smooth_e > 0.0 and d < smooth_e and B < B_CRIT:
                    d_n = d / smooth_e
                    W = d_Z_ext_sub_c(A, B) * (1 - d_n) + W * d_n

                dz = W
    elif nroot == 2:
        if gaslike:
            dz = d_Z_double_g_c(A, B)
        else:
            dz = d_Z_double_l_c(A, B)
    elif nroot == 3:
        # triple root area above the critical line is substituted with the
        # extended supercritical liquid-like root
        if is_sc:
            if gaslike:
                dz = d_Z_three_g_c(A, B)
            else:
                W = d_Z_ext_scl_c(A, B)

                # computing distance to Widom-line,
                # which separates gas and liquid in supercrit area
                d = point_to_line_distance_c(
                    AB_point,
                    W_LINE_POINTS[0],
                    W_LINE_POINTS[1],
                )
                if smooth_e > 0.0 and d < smooth_e and B >= B_CRIT:
                    d_n = d / smooth_e
                    W = d_Z_ext_scg_c(A, B) * (1 - d_n) + W * d_n

                # Computing distance to supercritical line,
                # which separates sub- and supercritical liquid extension
                d = point_to_line_distance_c(
                    AB_point,
                    S_CRIT_LINE_POINTS[0],
                    S_CRIT_LINE_POINTS[1],
                )
                if smooth_e > 0.0 and d < smooth_e and B < B_CRIT:
                    d_n = d / smooth_e
                    W = d_Z_ext_sub_c(A, B) * (1 - d_n) + W * d_n

                dz = W
        else:
            # smoothing according Ben Gharbia et al., in physical 2-phase region
            if smooth_3 > 0.0:
                Z_l = Z_three_l_c(A, B)
                Z_i = Z_three_i_c(A, B)
                Z_g = Z_three_g_c(A, B)

                d_Z_l = d_Z_three_l_c(A, B)
                d_Z_i = d_Z_three_i_c(A, B)
                d_Z_g = d_Z_three_g_c(A, B)

                d = (Z_i - Z_l) / (Z_g - Z_l)

                # gas root smoothing
                if gaslike:
                    # gas root smoothing weight
                    v_g = (d - (1 - 2 * smooth_3)) / smooth_3
                    v_g = v_g**2 * (3 - 2 * v_g)
                    if d >= 1 - smooth_3:
                        v_g = 1.0
                    elif d <= 1 - 2 * smooth_3:
                        v_g = 0.0

                    dz = d_Z_g * (1 - v_g) + (d_Z_i + d_Z_g) * 0.5 * v_g
                # liquid root smoothing
                else:
                    v_l = (d - smooth_3) / smooth_3
                    v_l = -(v_l**2) * (3 - 2 * v_l) + 1.0
                    if d <= smooth_3:
                        v_l = 1.0
                    elif d >= 2 * smooth_3:
                        v_l = 0.0

                    dz = d_Z_l * (1 - v_l) + (d_Z_i + d_Z_l) * 0.5 * v_l
            else:
                dz = d_Z_three_g_c(A, B) if gaslike else d_Z_three_l_c(A, B)
    else:
        dz = d_Z_triple_c(A, B)

    return dz


@numba.guvectorize(
    ["void(int8,float64,float64,float64,float64,float64,float64[:],float64[:])"],
    "(),(),(),(),(),(),(m)->(m)",
    target="parallel",
    nopython=True,
    cache=NUMBA_CACHE,
)
def _d_Z_u(
    gaslike,
    A,
    B,
    eps,
    smooth_e,
    smooth_3,
    out,
    dummy,
):
    """Internal ufunc because it cannot have default arguments"""
    # dummy is required to get the dimension of of the derivative
    # per row in vectorized computations (m in layout arg to guvectorize)
    # https://stackoverflow.com/questions/66052723/bad-token-in-signature-with-numba-guvectorize
    out[:] = d_Z_c(gaslike, A, B, eps, smooth_e, smooth_3)


def d_Z_u(
    gaslike: bool | int,
    A: float | np.ndarray,
    B: float | np.ndarray,
    eps: float = 1e-14,
    smooth_e: float = 1e-2,
    smooth_3: float = 1e-3,
) -> float | np.ndarray:
    """Numpy-universal function of :func:`d_Z_c`.

    This is a shallow wrapper of the actual ufunc to allow default arguments.

    """
    # get correct shape independent of format of input
    out_ = np.empty_like(A + B)
    if out_.shape:
        out = np.empty((out_.shape[0], 2))
    else:
        out = np.empty((1, 2), dtype=np.float64)
    _d_Z_u(int(gaslike), A, B, eps, smooth_e, smooth_3, out)
    # TODO in the case of scalar input (not vectorized)
    # decide if return arg has shape (1,n) or (n,)
    return out


def compile_Z_mix(
    A_c: Callable[[float, float, np.ndarray], float],
    B_c: Callable[[float, float, np.ndarray], float],
) -> Callable[[int, float, float, np.ndarray, float, float, float], float]:
    """Compiles a mixture-specific function for the compressibility factor, based on
    given (NJIT compatible) callables for the mixture's cohesion and covolume.

    This functiones is intended to be used inside :class:`PengRobinson_c`.

    Parameters:
        A_c: NJIT-ed representation of the mixed cohesion, dependent on pressure,
            temperature and fractions of components.
        B_c: Analogous mixed covolume.

    Returns:
        A compiled callable, expressing the compressibility factor in terms of
        pressure, temperature and fractions, instead of cohesion and covolume.

        The signature of the mixture-specific compressibility factor is identical to
        :func:`Z_c`, except that instead of the arguments ``A,B``,
        the arguments ``p, T, x`` are now taken, where ``x`` is an array of length equal
        to number of components.

    """

    # TODO signature with default args
    # @numba.njit(
    #     "float64(int8, float64, float64, float64[:], float64, float64, float64)"
    # )
    @numba.njit
    def Z_mix(
        gaslike: int,
        p: float,
        T: float,
        X: np.ndarray,
        eps: float = 1e-14,
        smooth_e: float = 1e-2,
        smooth_3: float = 1e-3,
    ) -> float:
        A_ = A_c(p, T, X)
        B_ = B_c(p, T, X)
        return Z_c(gaslike, A_, B_, eps, smooth_e, smooth_3)

    return Z_mix


def compile_d_Z_mix(
    A_c: Callable[[float, float, np.ndarray], float],
    B_c: Callable[[float, float, np.ndarray], float],
    d_A_c: Callable[[float, float, np.ndarray], np.ndarray],
    d_B_c: Callable[[float, float, np.ndarray], np.ndarray],
) -> Callable[[int, float, float, np.ndarray, float, float, float], np.ndarray]:
    """Same as :func:`compile_Z_mix`, only for the derivative of Z (see :func:`d_Z_c`).

    Parameters:
        A_c: NJIT-ed representation of the mixed cohesion, dependent on pressure,
            temperature and fractions of components.
        B_c: Analogous to ``A_c``.
        d_A_c: NJIT-ed representation of the derivative of the mixed cohesion ``A_c``,
            dependent on pressure, temperature and fractions of components.
            The returned array is expected to be of ``shape=(2 + num_comp,)``.
        d_B_c: Analogous to ``d_A_c`` for ``B_c``.

    Returns:
        A compiled callable, expressing the derivative compressibility factor in terms
        of pressure, temperature and fractions, instead of cohesion and covolume.

        Also here, the signature is changed from taking ``A,B`` to ``p,T,x``
        (see :func:`d_Z_c`).
        The return value is an array with the derivatives of compressibility factor
        w.r.t. the pressure, temperature and compositions per component
        (return array ``shape=(2 + num_comp,)``).

    """

    # TODO signature with default args
    # @numba.njit(
    #     "float64[:](int8, float64, float64, float64[:], float64, float64, float64)"
    # )
    @numba.njit
    def d_Z_mix(
        gaslike: int,
        p: float,
        T: float,
        X: np.ndarray,
        eps: float = 1e-14,
        smooth_e: float = 1e-2,
        smooth_3: float = 1e-3,
    ) -> float:
        A_ = A_c(p, T, X)
        B_ = B_c(p, T, X)
        dA = d_A_c(p, T, X)
        dB = d_B_c(p, T, X)
        dz = d_Z_c(gaslike, A_, B_, eps, smooth_e, smooth_3)
        return dz[0] * dA + dz[1] * dB

    return d_Z_mix


# endregion


def _compile_fugacities(
    phis: Callable[[float, float, np.ndarray, float, float, float], np.ndarray],
) -> Callable[[float, float, np.ndarray, float, float, float], np.ndarray]:
    """Helper function to compile the vector of fugacity coefficients.

    It needs an additional reduction of shape from ``(num_comp, 1)`` to ``(num_comp,)``
    because of the usage of a symbolic, vector-valued function."""
    f = numba.njit(phis)

    @numba.njit(
        "float64[:](float64, float64, float64[:], float64, float64, float64)",
    )
    def inner(p_, T_, X_, A_, B_, Z_):
        phi_ = f(p_, T_, X_, A_, B_, Z_)
        return phi_[:, 0]

    return inner


def _compile_thd_function_derivatives(
    thd_df: Callable[[float, float, np.ndarray], list[float]],
    fastmath: bool = False,
) -> Callable[[float, float, np.ndarray], np.ndarray]:
    """Helper function to compile the gradient of a thermodynamic function.

    Functions are supposed to take pressure, temperature and a vector of
    fractions as arguments.

    This helper function ensures that the return value is wrapped in an array, and not
    a list (as by default returned when using sympy.lambdify).

    It also enforces a signature ``(float64, float64, float64[:]) -> float64[:]``

    """
    df = numba.njit(thd_df, fastmath=fastmath)

    @numba.njit(
        "float64[:](float64, float64, float64[:])",
        fastmath=fastmath,
    )
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
    df = numba.njit(ext_thd_df)

    @numba.njit(
        "float64[:](float64, float64, float64[:], float64, float64, float64)",
    )
    def inner(p_, T_, X_, A_, B_, Z_):
        return np.array(df(p_, T_, X_, A_, B_, Z_), dtype=np.float64)

    return inner


def _compile_volume_derivative(
    dv: Callable[[float, float, float], list[float]]
) -> Callable[[float, float, float], np.ndarray]:
    """Helper function to compile the gradient of the specific volume.

    Required to wrap the result in an array.

    It also enforces a signature ``(float64, float64, float64) -> float64[:]``.

    """

    dv_ = numba.njit(
        fastmath=True,
    )(dv)

    @numba.njit("float64[:](float64,float64,float64)")
    def inner(p_, T_, Z_):
        return np.array(dv_(p_, T_, Z_), dtype=np.float64)

    return inner


class PengRobinsonCompiler(EoSCompiler):
    """Class providing compiled computations of thermodynamic quantities for the
    Peng-Robinson EoS."""

    def __init__(self, components: list[ComponentPR]) -> None:
        super().__init__(components)

        self._cfuncs: dict[str, Callable] = dict()
        """A collection of internally required, compiled callables"""

        self.symbolic: PengRobinsonSymbolic = PengRobinsonSymbolic(components)

    def compile(self) -> None:
        """Child method compiles essential functions from symbolic part before calling
        the parent class compiler"""

        logger.info("Compiling symbolic functions ..")

        B_c = numba.njit(
            "float64(float64, float64, float64[:])",
            fastmath=True,
        )(self.symbolic.B_f)
        logger.debug("Compiling symbolic functions 1/14")
        d_B_c = _compile_thd_function_derivatives(self.symbolic.d_B_f, fastmath=True)
        logger.debug("Compiling symbolic functions 2/14")
        A_c = numba.njit(
            "float64(float64, float64, float64[:])",
        )(self.symbolic.A_f)
        logger.debug("Compiling symbolic functions 3/14")
        # no fastmath because of sqrt
        d_A_c = _compile_thd_function_derivatives(self.symbolic.d_A_f)
        logger.debug("Compiling symbolic functions 4/14")

        Z_mix_c = compile_Z_mix(A_c, B_c)
        logger.debug("Compiling symbolic functions 5/14")
        d_Z_mix_c = compile_d_Z_mix(A_c, B_c, d_A_c, d_B_c)
        logger.debug("Compiling symbolic functions 6/14")

        phi_c = _compile_fugacities(self.symbolic.phi_f)
        logger.debug("Compiling symbolic functions 7/14")
        d_phi_c = numba.njit(
            "float64[:,:](float64, float64, float64[:], float64, float64, float64)"
        )(self.symbolic.d_phi_f)
        logger.debug("Compiling symbolic functions 8/14")

        h_dep_c = numba.njit(
            "float64(float64, float64, float64[:], float64, float64, float64)"
        )(self.symbolic.h_dep_f)
        logger.debug("Compiling symbolic functions 9/14")
        h_ideal_c = numba.njit("float64(float64, float64, float64[:])")(
            self.symbolic.h_ideal_f
        )
        logger.debug("Compiling symbolic functions 10/14")
        d_h_dep_c = _compile_extended_thd_function_derivatives(self.symbolic.d_h_dep_f)
        logger.debug("Compiling symbolic functions 11/14")
        d_h_ideal_c = _compile_thd_function_derivatives(self.symbolic.d_h_ideal_f)
        logger.debug("Compiling symbolic functions 12/14")

        v_c = numba.njit(
            "float64(float64,float64,float64)",
            fastmath=True,
        )(self.symbolic.v_f)
        logger.debug("Compiling symbolic functions 13/14")
        d_v_c = _compile_volume_derivative(self.symbolic.d_v_f)
        logger.debug("Compiling symbolic functions 14/14")

        self._cfuncs.update(
            {
                "A": A_c,
                "B": B_c,
                "Z": Z_mix_c,
                "d_A": d_A_c,
                "d_B": d_B_c,
                "d_Z": d_Z_mix_c,
                "phi": phi_c,
                "d_phi": d_phi_c,
                "h_dep": h_dep_c,
                "h_ideal": h_ideal_c,
                "d_h_dep": d_h_dep_c,
                "d_h_ideal": d_h_ideal_c,
                "v": v_c,
                "d_v": d_v_c,
            }
        )

        return super().compile()

    def get_prearg_for_values(
        self,
    ) -> Callable[[float, float, np.ndarray], np.ndarray]:
        A_c = self._cfuncs["A"]
        B_c = self._cfuncs["B"]
        Z_c = self._cfuncs["Z"]

        @numba.njit("float64[:](int32, float64, float64, float64[:])")
        def prearg_val_c(
            phasetype: int, p: float, T: float, xn: np.ndarray
        ) -> np.ndarray:
            prearg = np.empty((3,), dtype=np.float64)

            prearg[0] = A_c(p, T, xn)
            prearg[1] = B_c(p, T, xn)
            prearg[2] = Z_c(phasetype, p, T, xn)

            return prearg

        return prearg_val_c

    def get_prearg_for_derivatives(
        self,
    ) -> Callable[[float, float, np.ndarray], np.ndarray]:
        dA_c = self._cfuncs["d_A"]
        dB_c = self._cfuncs["d_B"]
        dZ_c = self._cfuncs["d_Z"]
        # number of derivatives for A, B, Z (p, T, and per component fraction)
        d = 2 + self._nc

        @numba.njit("float64[:](int32, float64, float64, float64[:])")
        def prearg_jac_c(
            phasetype: int, p: float, T: float, xn: np.ndarray
        ) -> np.ndarray:
            # the pre-arg for the jacobian contains the derivatives of A, B, Z
            # w.r.t. p, T, and fractions.
            prearg = np.empty((3 * d,), dtype=np.float64)

            prearg[0:d] = dA_c(p, T, xn)
            prearg[d : 2 * d] = dB_c(p, T, xn)
            prearg[2 * d : 3 * d] = dZ_c(phasetype, p, T, xn)

            return prearg

        return prearg_jac_c

    def get_fugacity_function(
        self,
    ) -> Callable[[np.ndarray, float, float, np.ndarray], np.ndarray]:
        phi_c = self._cfuncs["phi"]

        @numba.njit("float64[:](float64[:], float64, float64, float64[:])")
        def phi_mix_c(
            prearg: np.ndarray, p: float, T: float, xn: np.ndarray
        ) -> np.ndarray:
            return phi_c(p, T, xn, prearg[0], prearg[1], prearg[2])

        return phi_mix_c

    def get_dpTX_fugacity_function(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float, float, np.ndarray], np.ndarray]:
        d_phi_c = self._cfuncs["d_phi"]
        # number of derivatives
        d = 2 + self._nc

        @numba.njit(
            "float64[:,:](float64[:], float64[:], float64, float64, float64[:])"
        )
        def d_phi_mix_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            # computation of phis dependent on A_j, B_j, Z_j
            d_phis = d_phi_c(p, T, xn, prearg_val[0], prearg_val[1], prearg_val[2])
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

        return d_phi_mix_c

    def get_enthalpy_function(
        self,
    ) -> Callable[[np.ndarray, float, float, np.ndarray], float]:
        h_dep_c = self._cfuncs["h_dep"]
        h_ideal_c = self._cfuncs["h_ideal"]

        @numba.njit("float64(float64[:], float64, float64, float64[:])")
        def h_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> np.ndarray:
            return h_ideal_c(p, T, xn) + h_dep_c(
                p, T, xn, prearg[0], prearg[1], prearg[2]
            )

        return h_c

    def get_dpTX_enthalpy_function(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float, float, np.ndarray], np.ndarray]:
        d = 2 + self._nc
        d_h_dep_c = self._cfuncs["d_h_dep"]
        d_h_ideal_c = self._cfuncs["d_h_ideal"]

        @numba.njit("float64[:](float64[:], float64[:], float64, float64, float64[:])")
        def d_h_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            d_h_ideal = d_h_ideal_c(p, T, xn)
            d_h_dep = d_h_dep_c(p, T, xn, prearg_val[0], prearg_val[1], prearg_val[2])
            # derivatives of A_j, B_j, Z_j w.r.t. p, T, and X_j
            dA = prearg_jac[0:d]
            dB = prearg_jac[d : 2 * d]
            dZ = prearg_jac[2 * d : 3 * d]
            # expansion of derivatives of departure enthalpy (chain rule)
            d_h_dep = (
                d_h_dep[:-3] + d_h_dep[-3] * dA + d_h_dep[-2] * dB + d_h_dep[-1] * dZ
            )
            return d_h_ideal + d_h_dep

        return d_h_c

    def get_volume_function(
        self,
    ) -> Callable[[np.ndarray, float, float, np.ndarray], float]:
        v_c_ = self._cfuncs["v"]

        @numba.njit("float64(float64[:], float64, float64, float64[:])")
        def v_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> np.ndarray:
            return v_c_(p, T, prearg[2])

        return v_c

    def get_dpTX_volume_function(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float, float, np.ndarray], np.ndarray]:
        d = 2 + self._nc
        d_v_c_ = self._cfuncs["d_v"]

        @numba.njit("float64[:](float64[:], float64[:], float64, float64, float64[:])")
        def d_v_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            d_v_ = d_v_c_(p, T, prearg_val[2])
            # derivatives of Z_j w.r.t. p, T, and X_j
            dZ = prearg_jac[2 * d : 3 * d]
            # expansion of derivatives (chain rule)
            d_v = d_v_[-1] * dZ
            d_v[:2] += d_v[:2]  # contribution of p, T derivatives
            return d_v

        return d_v_c

    # TODO need models for below functions
    def get_viscosity_function(
        self,
    ) -> Callable[[ndarray, float, float, ndarray], float]:
        @numba.njit("float64(float64[:], float64, float64, float64[:])")
        def mu_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> np.ndarray:
            return 1.0

        return mu_c

    def get_dpTX_viscosity_function(
        self,
    ) -> Callable[[ndarray, float, float, ndarray], np.ndarray]:
        @numba.njit("float64[:](float64[:], float64[:], float64, float64, float64[:])")
        def dmu_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            return np.zeros(2 + xn.shape[0], dtype=np.float64)

        return dmu_c

    def get_conductivity_function(
        self,
    ) -> Callable[[ndarray, float, float, ndarray], float]:
        @numba.njit("float64(float64[:], float64, float64, float64[:])")
        def kappa_c(
            prearg: np.ndarray, p: float, T: float, xn: np.ndarray
        ) -> np.ndarray:
            return 1.0

        return kappa_c

    def get_dpTX_conductivity_function(
        self,
    ) -> Callable[[ndarray, float, float, ndarray], np.ndarray]:
        @numba.njit("float64[:](float64[:], float64[:], float64, float64, float64[:])")
        def d_kappa_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            return np.zeros(2 + xn.shape[0], dtype=np.float64)

        return d_kappa_c


logger.debug(
    f"{_import_msg} Done (elapsed time: {time.time() - _import_start} (s)).\n\n"
)

del _import_start, _import_msg


class _PR_Compiler_tests:
    """Collection of older code for testing the compiled functions."""

    def test_compiled_functions(self, tol: float = 1e-12, n: int = 100000):
        """Performs some tests on assembled functions.

        Warning:
            This triggers numba's just-in-time compilation!

            I.e., the execution of this function takes a considerable amount of time.

        Warning:
            This method raises AssertionErrors if any test failes.

        Parameters:
            tol: ``default=1e-12``

                Tolerance for numerical zero.
            n: ``default=100000``

                Number for testing of vectorized computations.

        """

        ncomp = ...

        p_1 = 1.0
        T_1 = 1.0
        X0 = np.array([0.0] * ncomp)

        A_c = self.cfuncs["A"]
        d_A_c = self.cfuncs["d_A"]
        B_c = self.cfuncs["B"]
        d_B_c = self.cfuncs["d_B"]
        Z_c = self.cfuncs["Z"]
        d_Z_c = self.cfuncs["d_Z"]

        Z_double_g_c = self._Z_cfuncs["double-root-gas"]
        d_Z_double_g_c = self._Z_cfuncs["d-double-root-gas"]
        Z_double_l_c = self._Z_cfuncs["double-root-liq"]
        d_Z_double_l_c = self._Z_cfuncs["d-double-root-liq"]

        # if compositions are zero, A and B are zero
        assert (
            B_c(p_1, T_1, X0) < tol
        ), "Value-test of compiled call to non-dimensional covolume failed."
        assert (
            A_c(p_1, T_1, X0) < tol
        ), "Value-test of compiled call to non-dimensional cohesion failed."

        # if A,B are zero, this should give the double-root case
        z_test_g = Z_c(True, p_1, T_1, X0, eps=1e-14, smooth_e=0.0, smooth_3=0.0)
        z_test_l = Z_c(False, p_1, T_1, X0, eps=1e-14, smooth_e=0.0, smooth_3=0.0)
        assert (
            np.abs(z_test_g - Z_double_g_c(0.0, 0.0)) < tol
        ), "Value-test for compiled, gas-like compressibility factor failed."
        assert (
            np.abs(z_test_l - Z_double_l_c(0.0, 0.0)) < tol
        ), "Value-test for compiled, liquid-like compressibility factor failed."

        d_z_test_g = d_Z_c(True, p_1, T_1, X0, eps=1e-14, smooth_e=0.0, smooth_3=0.0)
        d_z_test_l = d_Z_c(False, p_1, T_1, X0, eps=1e-14, smooth_e=0.0, smooth_3=0.0)
        da_ = d_A_c(p_1, T_1, X0)
        db_ = d_B_c(p_1, T_1, X0)
        dzg_ = d_Z_double_g_c(0.0, 0.0)
        dzl_ = d_Z_double_l_c(0.0, 0.0)
        dzg_ = dzg_[0] * da_ + dzg_[1] * db_
        dzl_ = dzl_[0] * da_ + dzl_[1] * db_
        assert (
            np.linalg.norm(d_z_test_g - dzg_) < tol
        ), "Derivative-test for compiled, gas-like compressibility factor failed."
        assert (
            np.linalg.norm(d_z_test_l - dzl_) < tol
        ), "Derivative-test for compiled, liquid-like compressibility factor failed."

        # p = np.random.rand(n) * 1e6 + 1
        # T = np.random.rand(n) * 1e2 + 1
        # X = np.random.rand(n, 2)
        # X = normalize_fractions(X)
