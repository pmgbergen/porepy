"""This module contains compiled versions of the Peng-Robinson EoS functionalities,
and related, required functions.

The compilation relies on implementing symbolic expressions for various quantities using
``sympy``, then lambdifying them using :func:`sympy.lambdify`, and finally compiling
the resulting callable using ``numba`` (:func:`numba.njit`).

Several functions are also implemented as only Python Callables, and then NJIT-ed.

Important:
    Importing this module for the first time triggers numba NJIT compilation with
    static signatures for a multitude of functions.

    This takes a considerable amount of time.


Due to the magnitude of expressions and functions (and different versions),
naming conflicts are hard to avoid.
We introcued the naming convention ``<derivative>_<name>_<version>``,
where the name represents the quantity, and version how the quantity is represented.

For example, the compressibility factor has the standard symbol ``Z`` in literature.
- ``Z_s`` denotes the symbolic representation using ``sympy``.
- ``Z_c`` denotes a compiled, scalar callable with a specified signature.
- ``Z_u`` denotes the numpy-universal function, a vectorized version of ``Z_c`` since
  a fast and efficient evaluation of the compressibility factor is deemed necessary
  (e.g. in flow problems to compute fluid properties)
- ``d_Z_u`` denotes the numpy universal function of the total derivative of ``Z``,
  w.r.t. to the dependencies given in the call signature of ``Z_c``
- ``dp_Z_c`` denotes a compiled function returning the derivative of ``Z`` w.r.t.
  pressure ``p``, with the same call signature as ``Z_c``.

You can assume the equal call signature for all related quantities, in terms of
number of dependencies, ``*args`` and ``**kwargs``.

The following convention for versions is used:

- ``_c``: NJIT compiled callable with a specific signature.
- ``_u``: Numpy-universal version (ufunc) of the respective ``_c`` callable, for
  vectorized evaluation.
- ``_s``: A symbol representing either an independent quantity, or an intermediate
  quantity serving as an argument. Created using :class:`sympy.Symbol`.
- ``_e``: A symbolic expression created using some algebraic combination of symbols.

The following standard names are used for thermodynamic quantities:

- ``Z`` compressibility factor
- ``A`` non-dimensional cohesion
- ``B`` non-dimensional covolume
- ``a`` cohesion
- ``b`` covolume
- ``T`` temperature
- ``p`` pressure
- ``X`` (extended) fractions of components in a phases
- ``y`` molar phase fractions
- ``z`` overal component fractions / feed fractions
- ``s`` volumetric phase fractions (saturations)
- ``_i`` index related to a component i
- ``_j`` index related to a phase j
- ``_r`` index related to the reference phase (the first one is assumed to be r)

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
import sympy as sp

import porepy as pp

from .._core import COMPOSITIONAL_VARIABLE_SYMBOLS as SYMBOLS
from .._core import NUMBA_CACHE, R_IDEAL
from ..composite_utils import COMPOSITE_LOGGER as logger
from ..composite_utils import safe_sum
from ..eos_compiler import EoSCompiler
from .eos import (
    A_CRIT,
    B_CRIT,
    B_CRIT_LINE_POINTS,
    S_CRIT_LINE_POINTS,
    W_LINE_POINTS,
    PengRobinson,
    coef0,
    coef1,
    coef2,
    critical_line,
    discr,
    point_to_line_distance,
    red_coef0,
    red_coef1,
    widom_line,
)
from .mixing import VanDerWaals

_STATIC_FAST_COMPILE_ARGS: dict[str, Any] = {
    "fastmath": True,
    "cache": NUMBA_CACHE,
}


_import_start = time.time()


# region Functions related to the characteristic polynomial and its roots


logger.debug("(import peng_robinson/eos_compilter.py) Compiling cubic polynomial ..\n")


coef0_c: Callable[[float, float], float] = numba.njit(
    "float64(float64, float64)", **_STATIC_FAST_COMPILE_ARGS
)(coef0)
"""NJIT-ed version of :func:`coef0`.

Signature: ``(float64, float64) -> float64``

"""


coef1_c: Callable[[float, float], float] = numba.njit(
    "float64(float64, float64)", **_STATIC_FAST_COMPILE_ARGS
)(coef1)
"""NJIT-ed version of :func:`coef1`.

Signature: ``(float64, float64) -> float64``

"""


coef2_c: Callable[[float, float], float] = numba.njit(
    "float64(float64, float64)", **_STATIC_FAST_COMPILE_ARGS
)(coef2)
"""NJIT-ed version of :func:`coef2`.

Signature: ``(float64, float64) -> float64``

"""


@numba.njit("float64(float64, float64)", **_STATIC_FAST_COMPILE_ARGS)
def red_coef0_c(A: float, B: float) -> float:
    """NJIT-ed version of :func:`red_coef0`.

    Signature: ``(float64, float64) -> float64``

    """
    c2 = coef2_c(A, B)
    return c2**3 * (2.0 / 27.0) - c2 * coef1_c(A, B) * (1.0 / 3.0) + coef0_c(A, B)


@numba.njit("float64(float64, float64)", **_STATIC_FAST_COMPILE_ARGS)
def red_coef1_c(A: float, B: float) -> float:
    """NJIT-ed version of :func:`red_coef1`.

    Signature: ``(float64, float64) -> float64``

    """
    return coef1_c(A, B) - coef2_c(A, B) ** 2 * (1.0 / 3.0)


discr_c: Callable[[float, float], float] = numba.njit(
    "float64(float64, float64)", **_STATIC_FAST_COMPILE_ARGS
)(discr)
"""NJIT-ed version of :func:`discr`.

Signature: ``(float64, float64) -> float64``

"""


@numba.njit("int8(float64, float64, float64)", **_STATIC_FAST_COMPILE_ARGS)
def get_root_case_c(A, B, eps=1e-14):
    """A piece-wise cosntant function dependent on
    non-dimensional cohesion and covolume, representing the number of roots
    of the characteristic polynomial in terms of cohesion and covolume.

    NJIT-ed function with signature ``(float64, float64, float64) -> int8``.

    :data:`red_coef0_c`, :data:`red_coef1_c` and :data:`discr_c` are used to compute
    and determine the root case.

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
    q = red_coef0_c(A, B)
    r = red_coef1_c(A, B)
    d = discr_c(q, r)

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
    r"""NJIT-ed function with signature ``(float64,float64,float64) -> float64``.

    Parameters:
        Z: A supposed root.
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The residual of the characteristic polynomial
        :math:`Z^3 + c_2(A, B) Z^2 + c_1(A, B) Z + c_0(A, B)`.

        IF ``Z`` is an actual root, the residual is 0.

    """
    c2 = coef2_c(A, B)
    c1 = coef1_c(A, B)
    c0 = coef0_c(A, B)

    return Z**3 + c2 * Z**2 + c1 * Z + c0


characteristic_residual_u = numba.vectorize(
    [numba.float64(numba.float64, numba.float64, numba.float64)],
    nopython=True,
    **_STATIC_FAST_COMPILE_ARGS,
)(characteristic_residual_c)
"""Numpy-universial version of :func:`characteristic_residual_c`."""


class _cbrt(sp.Function):
    """Custom symbolic cubic root to circumvent sympy using the power expression.

    The power expression is costly and does not always work with negative numbers.
    It returns sometimes not the principle cubic root (which is always real).

    Has a custom implementation of the derivative to always return a positive, real
    number.

    For more information
    `see here <https://docs.sympy.org/latest/guides/custom-functions.html>`_.

    Warning:
        As of now, lambdified expressions using this must use
        ``[{'_cbrt': numpy.cbrt}]`` in there module argument to provide a numerical
        evaluation for this function. TODO

        This Function is temprary and experimental. TODO do tests.

    """

    def fdiff(self, argindex=1):
        a = self.args[0]
        return 1.0 / (_cbrt(a**2) * 3)


def triple_root(A: sp.Symbol, B: sp.Symbol) -> sp.Expr:
    r"""Formula for tripple root of characteristic polynomial.

    Only valid in triple root case (see :func:`get_root_case_c`).

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        `See here <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_
        in section triple-root case.

    """
    c2 = coef2(A, B)
    return -c2 / 3


def double_root(A: sp.Symbol, B: sp.Symbol, gaslike: bool) -> sp.Expr:
    """Formulae for double roots.

    Only valid in 2-root case (see :func:`get_root_case_c`).

    Important:
        This returns a piece-wise expression, selecting the bigger root for the gas-like
        case. Lambdification only with module ``'math'``.

        The lambdified expression usind module ``'numpy'`` cannot be compiled by
        numba.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.
        gaslike: Bool indicating if the bigger root should be returned, or the smaller

    Returns:
        `See here <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_
        in section 2-root case.

    """
    c2 = coef2(A, B)
    q = red_coef0(A, B)
    r = red_coef1(A, B)

    u = 3 / 2 * q / r

    z1 = 2 * u - c2 / 3
    z23 = -u - c2 / 3

    if gaslike:
        return sp.Piecewise((z1, z1 > z23), (z23, True))
    else:
        return sp.Piecewise((z23, z1 > z23), (z1, True))


def three_root(A: sp.Symbol, B: sp.Symbol, gaslike: bool) -> sp.Expr:
    """Formulae for the 3-root case using the trigonometric representation
    (Casus Irreducibilis)

    Only valid in 3-root case (see :func:`get_root_case_c`).

    Note:
        The formulae allow for a clear distinction which root is the biggest, which
        the smallest. I.e. no piecewise nature required.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.
        gaslike: Bool indicating if the biggest root should be returned, or the smallest

    Returns:
        `See here <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_
        in section 3-root case, trigonometric approach.

    """
    c2 = coef2(A, B)
    q = red_coef0(A, B)
    r = red_coef1(A, B)

    # trigonometric formula for Casus Irreducibilis
    t_2 = sp.acos(-q / 2 * sp.sqrt(-27 * r ** (-3))) / 3
    t_1 = sp.sqrt(-4 / 3 * r)

    if gaslike:
        return t_1 * sp.cos(t_2) - c2 / 3
    else:
        return -t_1 * sp.cos(t_2 - np.pi / 3) - c2 / 3


def three_root_intermediate(A: sp.Symbol, B: sp.Symbol) -> sp.Expr:
    """Formula for the intermediate root in the 3-root case using the trigonometric
    representation

    Only valid in 3-root case (see :func:`get_root_case_c`).

    Note:
        This root has no physical meaning and is only used in the smoothing procedure
        proposed by Ben Gharbia et al. (2021).

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        `See here <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_
        in section 3-root case, trigonometric approach.

    """
    c2 = coef2(A, B)
    q = red_coef0(A, B)
    r = red_coef1(A, B)

    t_2 = sp.acos(-q / 2 * sp.sqrt(-27 * r ** (-3))) / 3
    t_1 = sp.sqrt(-4 / 3 * r)

    return -t_1 * sp.cos(t_2 + np.pi / 3) - c2 / 3


def one_root(A: sp.Symbol, B: sp.Symbol) -> sp.Expr:
    """Formulae for single, real root.

    Only valid in 1-root case (see :func:`get_root_case_c`).

    Important:
        This returns a piece-wise expression, due to some special choices of root
        computation for numerical reasons.

        Lambdification only with module ``'math'``.
        The lambdified expression usind module ``'numpy'`` cannot be compiled by
        numba.

        Furthermore, a custom implementation of the cubic root is used, which
        allways returns the principal cubic root (which is always real).

        This custom implementation must be replaced upon lambdification, using
        the module argument ``[{'_cbrt': numpy.cbrt}, 'math']``

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        `See here <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_
        in section 1-root case.

    """
    c2 = coef2(A, B)
    q = red_coef0(A, B)
    r = red_coef1(A, B)
    d = discr(q, r)

    t1 = sp.sqrt(d) - q * 0.5
    t2 = -1 * (sp.sqrt(d) + q * 0.5)

    t = sp.Piecewise((t2, sp.Abs(t2) > sp.Abs(t1)), (t1, True))

    u = _cbrt(t)  # TODO potential source of error

    return u - r / (3 * u) - c2 / 3


def ext_root_gharbia(Z: sp.Expr, B: sp.Symbol) -> sp.Expr:
    r"""Formula for the Ben Gharbia extension in the case of only one real root in the
    subcritical area of the A-B space.

    Parameters:
        Z: The single, real root
        B: Non-dimensional covolume.

    Returns:
        The expression :math:`\frac{1 - B - Z}{2}`.

    """
    return (1 - B - Z) / 2


def ext_root_scg(Z: sp.Expr, B: sp.Symbol) -> sp.Expr:
    r"""Formula for the extended, gas-like root, outside the subcritical area defined
    by Ben Gharbia et al..

    Parameters:
        Z: The single, real root
        B: Non-dimensional covolume.

    Returns:
        The expression :math:`\frac{1 - B - Z}{2} + B`.

    """
    return (1 - B - Z) / 2 + B


def ext_root_scl(Z: sp.Expr, B: sp.Symbol) -> sp.Expr:
    r"""Formula for the extended, liquid-like root, outside the subcritical area defined
    by Ben Gharbia et al.

    This is the counterpart to :func:`ext_root_scg` for the case when the liquid-like
    phase needs to be extended.

    To be used also in the 3-root region above the critical line in the A-B space.

    Parameters:
        Z: The single, real root
        B: Non-dimensional covolume.

    Returns:
        The expression :math:`\frac{B - Z}{2} + Z`.

    """
    return (B - Z) / 2 + Z


# endregion
# region Functions related to the A-B space

logger.debug(
    "(import peng_robinson/eos_compilter.py) Compiling A-B space functions ..\n"
)

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
# region Functions, expressions and symbols related to the compressibility factor

Z_s: sp.Symbol = sp.Symbol("Z")
"""Symbol for the compressibility factor.

Intended use is as an intermediate, independent quantity to evaluate
complex symbolic expressions.

"""
A_s: sp.Symbol = sp.Symbol("A")
"""Symbol for non-dimensional cohesion.

Intended use is as an intermediate, independent quantity to evaluate complex
symbolic expressions.

"""
B_s: sp.Symbol = sp.Symbol("B")
"""Symbol for non-dimensional covolume.

Intended use is as an intermediate, independent quantity to evaluate complex
symbolic expressions.

"""

_AB_arg: list[sp.Symbol] = [A_s, B_s]
"""Arguments for lambdified expressions representing any root of the characteristic
polynomial."""

# Symbolic expressions for roots

logger.debug(
    "(import peng_robinson/eos_compilter.py) Assembling symbolic polynomial roots ..\n"
)

Z_triple_e: sp.Expr = triple_root(A_s, B_s)
d_Z_triple_e: list[sp.Expr] = [Z_triple_e.diff(_) for _ in _AB_arg]

Z_one_e: sp.Expr = one_root(A_s, B_s)
d_Z_one_e: list[sp.Expr] = [Z_one_e.diff(_) for _ in _AB_arg]

Z_ext_sub_e: sp.Expr = ext_root_gharbia(Z_one_e, B_s)
d_Z_ext_sub_e: list[sp.Expr] = [Z_ext_sub_e.diff(_) for _ in _AB_arg]

Z_ext_scg_e: sp.Expr = ext_root_scg(Z_one_e, B_s)
d_Z_ext_scg_e: list[sp.Expr] = [Z_ext_scg_e.diff(_) for _ in _AB_arg]

Z_ext_scl_e: sp.Expr = ext_root_scl(Z_one_e, B_s)
d_Z_ext_scl_e: list[sp.Expr] = [Z_ext_scl_e.diff(_) for _ in _AB_arg]

Z_double_g_e: sp.Expr = double_root(A_s, B_s, True)
d_Z_double_g_e: list[sp.Expr] = [Z_double_g_e.diff(_) for _ in _AB_arg]

Z_double_l_e: sp.Expr = double_root(A_s, B_s, False)
d_Z_double_l_e: list[sp.Expr] = [Z_double_l_e.diff(_) for _ in _AB_arg]

Z_three_g_e: sp.Expr = three_root(A_s, B_s, True)
d_Z_three_g_e: list[sp.Expr] = [Z_three_g_e.diff(_) for _ in _AB_arg]

Z_three_l_e: sp.Expr = three_root(A_s, B_s, False)
d_Z_three_l_e: list[sp.Expr] = [Z_three_l_e.diff(_) for _ in _AB_arg]

Z_three_i_e: sp.Expr = three_root_intermediate(A_s, B_s)
d_Z_three_i_e: list[sp.Expr] = [Z_three_i_e.diff(_) for _ in _AB_arg]


# NJIT compilation of lambdified expressions
# TODO sympy.lambdified functions are source-less and cannot be cached
# find solution for this
# see https://github.com/sympy/sympy/issues/18432
# https://github.com/numba/numba/issues/5128
def _compile_Z_diffs(
    d_Z_: list[sp.Expr], fastmath: bool = False, lambdify_mod: Any = None
) -> Callable[[float, float], np.ndarray]:
    """Helper function to wrap derivatives of compressibility factors into arrays.

    Parameters:
        d_Z_: Symbolic expression of derivative of root,
            dependent on symbols for A and B.
        fastmath: for numba (use only for simple expressions)
        lambdify_mod: module argument for :func:`sympy.lambdify`

    Returns:
        NJIT-compiled function with signature ``(float64, float64) -> float64[:]``.

    """

    f = (
        sp.lambdify(_AB_arg, d_Z_)
        if lambdify_mod is None
        else sp.lambdify(_AB_arg, d_Z_, lambdify_mod)
    )
    f_ = numba.njit(f, cache=False, fastmath=fastmath)

    @numba.njit("float64[:](float64, float64)", cache=False, fastmath=fastmath)
    def inner(a, b):
        return np.array(f_(a, b), dtype=np.float64)

    return inner


def _compile_Z(
    Z_: sp.Expr, fastmath: bool = False, lambdify_mod: Any = None
) -> Callable[[float, float], float]:
    """Helper function to compile expressions representing compressibility factors as
    roots.

    Parameters:
        Z_: Symbolic expression for root, dependent on symbols for A and B.
        fastmath: for numba (use only for simple expressions)
        lambdify_mod: module argument for :func:`sympy.lambdify`

    Returns:
        NJIT-compiled function with signature ``(float64, float64) -> float64``.

    """
    f = (
        sp.lambdify(_AB_arg, Z_)
        if lambdify_mod is None
        else sp.lambdify(_AB_arg, Z_, lambdify_mod)
    )
    return numba.njit("float64(float64, float64)", cache=False, fastmath=fastmath)(f)


logger.debug(
    "(import peng_robinson/eos_compilter.py)"
    + " Compiling generalized compressibility factor ..\n"
)

Z_triple_c: Callable[[float, float], float] = _compile_Z(Z_triple_e, fastmath=True)
d_Z_triple_c: Callable[[float, float], np.ndarray] = _compile_Z_diffs(
    d_Z_triple_e, fastmath=True
)

# because piecewise and to provide numeric evaluation of custom cubic root
_module_one_root = [{"_cbrt": np.cbrt}, "math"]

Z_one_c: Callable[[float, float], float] = _compile_Z(
    Z_one_e, lambdify_mod=_module_one_root
)
d_Z_one_c: Callable[[float, float], np.ndarray] = _compile_Z_diffs(
    d_Z_one_e, lambdify_mod=_module_one_root
)

Z_ext_sub_c: Callable[[float, float], float] = _compile_Z(
    Z_ext_sub_e, lambdify_mod=_module_one_root
)
d_Z_ext_sub_c: Callable[[float, float], np.ndarray] = _compile_Z_diffs(
    d_Z_ext_sub_e, lambdify_mod=_module_one_root
)

Z_ext_scg_c: Callable[[float, float], float] = _compile_Z(
    Z_ext_scg_e, lambdify_mod=_module_one_root
)
d_Z_ext_scg_c: Callable[[float, float], np.ndarray] = _compile_Z_diffs(
    d_Z_ext_scg_e, lambdify_mod=_module_one_root
)

Z_ext_scl_c: Callable[[float, float], float] = _compile_Z(
    Z_ext_scl_e, lambdify_mod=_module_one_root
)
d_Z_ext_scl_c: Callable[[float, float], np.ndarray] = _compile_Z_diffs(
    d_Z_ext_scl_e, lambdify_mod=_module_one_root
)

# because piece-wise
_module_double_root = "math"

Z_double_g_c: Callable[[float, float], float] = _compile_Z(
    Z_double_g_e, lambdify_mod=_module_double_root
)
d_Z_double_g_c: Callable[[float, float], np.ndarray] = _compile_Z_diffs(
    d_Z_double_g_e, lambdify_mod=_module_double_root
)

Z_double_l_c: Callable[[float, float], float] = _compile_Z(
    Z_double_l_e, lambdify_mod=_module_double_root
)
d_Z_double_l_c: Callable[[float, float], np.ndarray] = _compile_Z_diffs(
    d_Z_double_l_e, lambdify_mod=_module_double_root
)

Z_three_g_c: Callable[[float, float], float] = _compile_Z(Z_three_g_e)
d_Z_three_g_c: Callable[[float, float], np.ndarray] = _compile_Z_diffs(d_Z_three_g_e)

Z_three_l_c: Callable[[float, float], float] = _compile_Z(Z_three_l_e)
d_Z_three_l_c: Callable[[float, float], np.ndarray] = _compile_Z_diffs(d_Z_three_l_e)

Z_three_i_c: Callable[[float, float], float] = _compile_Z(Z_three_i_e)
d_Z_three_i_c: Callable[[float, float], np.ndarray] = _compile_Z_diffs(d_Z_three_i_e)


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


class PengRobinson_s:
    """Symbolic representation fo some thermodynamic quantities using the
    Peng-Robinson EoS."""

    def __init__(self, mixture: pp.composite.NonReactiveMixture) -> None:
        self.p: sp.Symbol = sp.Symbol(str(SYMBOLS["pressure"]))
        """Symbolic representation fo pressure."""
        self.T: sp.Symbol = sp.Symbol(str(SYMBOLS["temperature"]))
        """Symbolic representation fo temperature."""
        self.h: sp.Symbol = sp.Symbol(str(SYMBOLS["enthalpy"]))
        """Symbolic representation fo specific molar enthalpy."""
        self.v: sp.Symbol = sp.Symbol(str(SYMBOLS["volume"]))
        """Symbolic representation fo specific molar volume."""

        self.x_in_j: list[sp.Symbol] = [
            sp.Symbol(f"{SYMBOLS['phase_composition']}_{comp.name}_j")
            for comp in mixture.components
        ]
        """List of phase composition fractions associated with a phase.
        Length is equal to number of components, because every component is asssumed
        present in every phase in the unified setting."""

        self.thd_arg = [self.p, self.T, self.x_in_j]
        """General representation of the thermodynamic argument:
        a pressure value, a temperature value, and a sequence of fractional values."""

        # region coterms
        # covolume per component
        self.b_i_crit: list[float] = [
            PengRobinson.b_crit(comp.p_crit, comp.T_crit) for comp in mixture.components
        ]
        """List of critical covolumes per component"""

        # mixed covolume
        self.b_e: sp.Expr = VanDerWaals.covolume(self.x_in_j, self.b_i_crit)
        """Mixed covolume according to the Van der Waals mixing rule."""
        # non-dimensional covolume
        self.B_e: sp.Expr = PengRobinson.B(self.b_e, self.p, self.T)
        """Non-dimensional, mixed covolume created using :attr:`b_e`"""

        # cohesion

        # TODO this is not safe. Special BIP implementations are not reliably sympy
        # compatible as of now
        bips, _ = mixture.reference_phase.eos.compute_bips(self.T)

        self.ai_crit: list[float] = [
            PengRobinson.a_crit(comp.p_crit, comp.T_crit) for comp in mixture.components
        ]
        """List of critical cohesion values per component."""

        ki: list[float] = [
            PengRobinson.a_correction_weight(comp.omega) for comp in mixture.components
        ]
        ai_correction_e: list[sp.Expr] = [
            1 + k * (1 - sp.sqrt(self.T / comp.T_crit))
            for k, comp in zip(ki, mixture.components)
        ]

        self.ai_e: list[sp.Expr] = [
            a * corr**2 for a, corr in zip(self.ai_crit, ai_correction_e)
        ]
        """List of cohesion values per component, including a correction involving
        the critical temperature and acentric factor."""

        self.a_e: sp.Expr = VanDerWaals.cohesion_s(self.x_in_j, self.ai_e, bips)
        """Mixed cohesion according to the Van der Waals mixing rule."""
        self.A_e: sp.Expr = PengRobinson.A(self.a_e, self.p, self.T)
        """Non-dimensional, mixed cohesion created using :attr:`b_e`"""
        # endregion

        # region Fugacity coefficients
        phi_i_e: list[sp.Expr] = []

        for i in range(mixture.num_components):
            B_i_e = PengRobinson.B(self.b_i_crit[i], self.p, self.T)
            dXi_A_e = self.A_e.diff(self.x_in_j[i])
            log_phi_i = (
                B_i_e / B_s * (Z_s - 1)
                - sp.ln(Z_s - B_s)
                - A_s
                / (B_s * np.sqrt(8))
                * sp.ln((Z_s + (1 + np.sqrt(2)) * B_s) / (Z_s + (1 - np.sqrt(2)) * B_s))
                * (dXi_A_e / A_s - B_i_e / B_s)
            )
            # TODO this is numerically disadvantages
            # no truncation and usage of exp
            phi_i_e.append(sp.exp(log_phi_i))

        self.phi_e: sp.Matrix = sp.Matrix(phi_i_e)
        """A vector-valued symbolic expression containing fugacity coefficients per
        component.

        Note:
            We need to make it vector-valued to avoid looping over individual functions.
            This is not parallelizable with numba.

        """
        self.d_phi_e: sp.Matrix = self.phi_e.jacobian(
            [self.p, self.T] + self.x_in_j + [A_s, B_s, Z_s]
        )
        """The symbolic Jacobian of :attr:`phi_e` w.r.t. to thermodynamic arguments and
        :data:`A_s`, :data:`B_s`, :data:`Z_s`"""
        # endregion

        # region Enthalpy

        dT_A_e: sp.Expr = self.A_e.diff(self.T)

        self.h_dep_e: sp.Expr = (R_IDEAL / np.sqrt(8)) * (
            dT_A_e * self.T**2 + A_s * self.T
        ) / B_s * sp.ln(
            (Z_s + (1 + np.sqrt(2)) * B_s) / (Z_s + (1 - np.sqrt(2)) * B_s)
        ) + R_IDEAL * self.T * (
            Z_s - 1
        )
        """Symbolic expression for departure enthalpy."""

        self.d_h_dep_e: list[sp.Expr] = [
            self.h_dep_e.diff(_)
            for _ in [self.p, self.T] + self.x_in_j + [A_s, B_s, Z_s]
        ]
        """Symbolic gradient of :attr:`h_dep_e` w.r.t. to thermodynamic arguments and
        :data:`A_s`, :data:`B_s`, :data:`Z_s`"""

        self.h_ideal_e: sp.Expr = safe_sum(
            [
                x * comp.h_ideal(self.p, self.T)
                for x, comp in zip(self.x_in_j, mixture.components)
            ]
        )
        """Symbolic expression for the ideal enthalpy."""

        self.d_h_ideal_e: list[sp.Expr] = [
            self.h_ideal_e.diff(_) for _ in [self.p, self.T] + self.x_in_j
        ]
        """Symbolic gradient of :attr:`h_ideal_e` w.r.t. to thermodynamic arguments."""

        # endregion

        # region Volume

        self.v_e: sp.Expr = Z_s * self.p * R_IDEAL / self.T
        """Symbolic expression for specific volume, depending on pressure, temperature
        and compressibility factor."""

        self.d_v_e: list[sp.Expr] = [self.v_e.diff(_) for _ in [self.p, self.T, Z_s]]
        """Symbolic gradient of :attr:`v_e` w.r.t. pressure, temperature and
        compressibility factor."""

        # endregion


def _compile_coterms_derivatives(
    expr: sp.Expr,
    thd_arg: tuple[sp.Symbol, sp.Symbol, list[sp.Symbol]],
    fastmath: bool = False,
) -> Callable[[float, float, np.ndarray], np.ndarray]:
    """Helper function to compile the derivatives of A and B.

    Includes differentiating, wrapping into an array and a compilation with a proper
    signature.

    Note:
        No caching since functions are mixture dependent

    """
    diff = [expr.diff(thd_arg[0]), expr.diff(thd_arg[1])] + [
        expr.diff(_) for _ in thd_arg[2]
    ]
    diff_c = numba.njit(sp.lambdify(thd_arg, diff), fastmath=fastmath)

    @numba.njit("float64[:](float64, float64, float64[:])", fastmath=fastmath)
    def inner(
        p_,
        T_,
        X_,
    ):
        return np.array(diff_c(p_, T_, X_), dtype=np.float64)

    return inner


def _compile_fugacities(
    phis: sp.Matrix,
    phi_arg: tuple[
        sp.Symbol, sp.Symbol, list[sp.Symbol], sp.Symbol, sp.Symbol, sp.Symbol
    ],
) -> Callable[[float, float, np.ndarray, float, float, float], np.ndarray]:
    """Helper function to compile the vector of fugacity coefficients.

    It needs an additional reduction of shape from ``(num_comp, 1)`` to ``(num_comp,)``
    because of the usage of a symbolic, vector-valued function."""
    f = numba.njit(sp.lambdify(phi_arg, phis))

    @numba.njit(
        "float64[:](float64, float64, float64[:], float64, float64, float64)",
    )
    def inner(p_, T_, X_, A_, B_, Z_):
        phi_ = f(p_, T_, X_, A_, B_, Z_)
        return phi_[:, 0]

    return inner


def _compile_thd_function_derivatives(
    thd_df: sp.Expr,
    thd_arg: tuple[sp.Symbol, sp.Symbol, list[sp.Symbol]],
) -> Callable[[float, float, np.ndarray], np.ndarray]:
    """Helper function to compile the gradient of a thermodynamic function.

    Functions are supposed to take pressure, temperature and a vector of
    fractions as arguments.

    This helper function ensures that the return value is wrapped in an array, and not
    a list (as by default returned when using sympy.lambdify).

    """
    df = numba.njit(sp.lambdify(thd_arg, thd_df))

    @numba.njit(
        "float64[:](float64, float64, float64[:])",
    )
    def inner(p_, T_, X_):
        return np.array(df(p_, T_, X_), dtype=np.float64)

    return inner


def _compile_extended_thd_function_derivatives(
    ext_thd_df: sp.Expr,
    ext_thd_arg: tuple[
        sp.Symbol, sp.Symbol, list[sp.Symbol], sp.Symbol, sp.Symbol, sp.Symbol
    ],
) -> Callable[[float, float, np.ndarray, float, float, float], np.ndarray]:
    """Helper function to compile the gradient of an extended thermodynamic function.

    Functions are supposed to take pressure, temperature, a vector of
    fractions, and the EoS specific terms cohesion, covolume and
    compressibility factor as arguments.

    This helper function ensures that the return value is wrapped in an array, and not
    a list (as by default returned when using sympy.lambdify).

    """
    df = numba.njit(sp.lambdify(ext_thd_arg, ext_thd_df))

    @numba.njit(
        "float64[:](float64, float64, float64[:], float64, float64, float64)",
    )
    def inner(p_, T_, X_, A_, B_, Z_):
        return np.array(df(p_, T_, X_, A_, B_, Z_), dtype=np.float64)

    return inner


class PengRobinsonCompiler(EoSCompiler):
    """Class providing compiled computations of thermodynamic quantities for the
    Peng-Robinson EoS."""

    def __init__(
        self, mixture: pp.composite.NonReactiveMixture, verbosity: int = 0
    ) -> None:
        super().__init__(mixture)

        self._cfuncs: dict[str, Callable] = dict()
        """A collection of internally required, compiled callables"""

        self.symbolic: PengRobinson_s = PengRobinson_s(mixture)

        # setting logging verbosity
        if verbosity == 1:
            logger.setLevel(logging.INFO)
        elif verbosity >= 2:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        nphase, ncomp = self.npnc
        logger.info(
            f"Starting EoS compilation (phases: {nphase}, components: {ncomp}):\n"
        )
        _start = time.time()
        logger.debug("(EoS) Compiling coterms ..\n")

        B_c = numba.njit(
            "float64(float64, float64, float64[:])",
            fastmath=True,
        )(sp.lambdify(self.symbolic.thd_arg, self.symbolic.B_e))

        d_B_c = _compile_coterms_derivatives(
            self.symbolic.B_e, self.symbolic.thd_arg, fastmath=True
        )

        A_c = numba.njit(
            "float64(float64, float64, float64[:])",
        )(sp.lambdify(self.symbolic.thd_arg, self.symbolic.A_e))
        # no fastmath because of sqrt
        d_A_c = _compile_coterms_derivatives(self.symbolic.A_e, self.symbolic.thd_arg)

        logger.debug("(EoS) Compiling compressibility factor ..\n")
        Z_mix_c = compile_Z_mix(A_c, B_c)
        d_Z_mix_c = compile_d_Z_mix(A_c, B_c, d_A_c, d_B_c)

        ext_thd_arg = [
            self.symbolic.p,
            self.symbolic.T,
            self.symbolic.x_in_j,
            A_s,
            B_s,
            Z_s,
        ]

        logger.debug("(EoS) Compiling fugacity coefficients ..\n")
        phi_c = _compile_fugacities(self.symbolic.phi_e, ext_thd_arg)

        d_phi_c = numba.njit(
            "float64[:,:](float64, float64, float64[:], float64, float64, float64)"
        )(sp.lambdify(ext_thd_arg, self.symbolic.d_phi_e))

        logger.debug("(EoS) Compiling spec. mol. mixture enthalpy computation ..\n")

        h_dep_c = numba.njit(
            "float64(float64, float64, float64[:], float64, float64, float64)"
        )(sp.lambdify(ext_thd_arg, self.symbolic.h_dep_e))

        h_ideal_c = numba.njit("float64(float64, float64, float64[:])")(
            sp.lambdify(self.symbolic.thd_arg, self.symbolic.h_ideal_e)
        )

        d_h_dep_c = _compile_extended_thd_function_derivatives(
            self.symbolic.d_h_dep_e, ext_thd_arg
        )

        d_h_ideal_c = _compile_thd_function_derivatives(
            self.symbolic.d_h_ideal_e, self.symbolic.thd_arg
        )

        logger.debug("(EoS) Compiling spec. mol. volume computation ..\n")

        v_c = numba.njit(
            "float64(float64,float64,float64)",
            fastmath=True,
        )(sp.lambdify([self.symbolic.p, self.symbolic.T, Z_s], self.symbolic.v_e))

        d_v_c_inner = numba.njit(
            sp.lambdify([self.symbolic.p, self.symbolic.T, Z_s], self.symbolic.d_v_e)
        )

        @numba.njit(
            "float64[:](float64, float64, float64)",
        )
        def d_v_c(p_: float, T_: float, Z_: float) -> np.ndarray:
            return np.array(d_v_c_inner(p_, T_, Z_), dtype=np.float64)

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
        logger.debug("(EoS) Compiling generalzied u-funcs ..\n")

        _end = time.time()
        logger.info(
            f"EoS compilation compleded (elapsed time: {_end - _start} (s)).\n\n"
        )

    def get_prearg_for_values(
        self,
    ) -> Callable[[float, float, np.ndarray], np.ndarray]:
        nphase, _ = self.npnc

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
        d = 2 + self.npnc[1]

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
        _, ncomp = self.npnc
        # number of derivatives for A, B, Z
        d = 2 + ncomp

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
        _, ncomp = self.npnc
        d = 2 + ncomp
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
        _, ncomp = self.npnc
        d = 2 + ncomp
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
            # expansion of derivatives of enthalpy (chain rule)
            d_v = d_v_[-1] * dZ
            d_v[:2] += d_v[:2]  # contribution of p, T derivatives
            return d_v

        return d_v_c


_import_end = time.time()

logger.debug(
    "(import peng_robinson/eos_compilter.py)"
    + f" Done (elapsed time: {_import_end - _import_start} (s)).\n\n"
)

del _import_start, _import_end
