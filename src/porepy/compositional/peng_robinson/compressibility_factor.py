"""Module for computing the compressibility factor using the Peng-Robinson EOS.

The Peng-Robinson equation of state is a cubic equation in terms of the compressibility,
where the coefficients depend on the dimensionless cohesion and covolume parameters,
:math:`A` and :math:`B` respectively.

The cubic polynomial is given by

.. math::

    Z^3 + (B - 1) Z^2 + (A - 2B - 3B^2) Z + (B^3 + B^2 - AB) = 0,

I.e., the coefficients are a function of :math:`A` and :math:`B`.

This module wraps the functionality given in
:mod:`porepy.compositional.peng_robinson.cubic_polynomial` and expresses the solutions
and their derivatives in terms of :math:`A` and :math:`B`.

Additionally, it provides an extension procedure following the work of Ben Gharbia et
al. (2021), as well es Lipovac et al. (2024).

I.e., it provides an additional root in the 1-root area.

"""

from __future__ import annotations

import numba as nb
import numpy as np

from .._numba_interface import NUMBA_CACHE, NUMBA_FAST_MATH, njit
from .cubic_polynomial import (
    calculate_root_derivatives,
    calculate_roots,
    get_root_case,
    one_root,
)

__all__ = [
    "A_CRIT",
    "B_CRIT",
    "Z_CRIT",
    "COVOLUME_LIMIT",
    "c_from_AB",
    "dc_from_AB",
    "widom_line",
    "is_supercritical",
    "is_extended_factor",
    "get_compressibility_factor",
    "get_compressibility_factor_derivatives",
]


_COMPILER = njit
"""Decorator for compiling functions in this module.

Alternative compilers are the :obj:`numba.cfunc` call-back decorator, or future AOT
compilation.

"""


A_CRIT: float = (
    1
    / 512
    * (
        -59
        + 3 * np.cbrt(276231 - 192512 * np.sqrt(2))
        + 3 * np.cbrt(276231 + 192512 * np.sqrt(2))
    )
)
"""Critical dimensionless cohesion value in the Peng-Robinson EoS,
~ 0.457235529."""


B_CRIT: float = (
    1
    / 32
    * (-1 - 3 * np.cbrt(16 * np.sqrt(2) - 13) + 3 * np.cbrt(16 * np.sqrt(2) + 13))
)
"""Critical dimensionless covolume in the Peng-Robinson EoS, ~ 0.077796073."""


Z_CRIT: float = (
    1 / 32 * (11 + np.cbrt(16 * np.sqrt(2) - 13) - np.cbrt(16 * np.sqrt(2) + 13))
)
"""Critical compressibility factor in the Peng-Robinson EoS, ~ 0.307401308."""

dZ_CRIT: np.ndarray = np.array((0.0, -1 / 3))
"""Derivative of critical compressibility factor with respect to ``A`` and ``B``."""


CRITICAL_SLOPE: float = B_CRIT / A_CRIT
"""Slope of the critical line ``(0,0) -> (A_c, B_c)`` given by ``B_c / A_c``.

Used to parametrize the critical line in terms of the cohesion ``A``.

See also:
    :data:`B_CRIT`, :data:`A_CRIT`

"""


COVOLUME_LIMIT: float = 1e-7
""""Below this value, the covolume is considered zero.

Required to treat the limit case of B -> 0.

Note:
    This value is highly related to how the cubic root computation is implemented for
    the degenerate 2-root case. If sensitivity improves there, this can be decreased as
    well.

"""


@_COMPILER(
    nb.f8(nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def cometric(v1: np.ndarray, v2: np.ndarray) -> float:
    """Metric in AB space considering the discrepancies in size between cohesion and
    covolume.

    The dot product is equivalent to ``numpy.dot(v1, M@v2)``, with ``M`` being a
    diagonal matrix scaling the first entry using :data:`CRITICAL_SLOPE`. I.e, it
    introduces a distance in cohesion-dimension scaling the same as the covolume
    dimension.

    Parameters:
        v1: Vector of shape ``(2,)`` containing cohesion and covolume values.
        v2: Vector of same shape.

    Returns:
        The scalar product using the co-metric.

    """
    return v1[0] * v2[0] * CRITICAL_SLOPE + v1[1] * v2[1]


@_COMPILER(
    nb.f8(nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def conorm(A: float, B: float) -> float:
    """Norm induced by co-metric :func:`cometric`.

    Parameters:
        A: Dimensionless cohesion.
        B: Dimensionless covolume.

    Returns:
        Norm of the vector ``(A, B)`` using the metric.

    """
    return np.sqrt(A**2 * CRITICAL_SLOPE + B**2)


@_COMPILER(
    nb.f8[:](nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def c_from_AB(A: float, B: float) -> np.ndarray:
    """Implements the formula for the coefficients of the normalized cubic polynomial
    dependeng on cohesion and covolume.

    .. math::

        c_2 = B - 1,
        c_1 = A - 2B - 3B^2,
        c_0 = B^3 + B^2 - AB.

    Note:
        The returned array contains the coefficients as required by the cubic polynomial
        module.

    Parameters:
        A: Dimensionless cohesion.
        B: Dimensionless covolume.

    Returns:
        A ``(3,)``-array containing the coefficients :math:`c_0, c_1, c_2`.

    """
    return np.array(
        [
            B - 1.0,
            A - 2.0 * B - 3.0 * B**2,
            B**3 + B**2 - A * B,
        ]
    )


@_COMPILER(
    nb.f8[:, :](nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def dc_from_AB(A: float, B: float) -> np.ndarray:
    """Returns the Jacobian of the function implemented by :func:`c_from_AB`.

    Parameters:
        A: Dimensionless cohesion.
        B: Dimensionless covolume.

    Returns:
        A ``(3,2)``-array containing the derivatives of coefficients
        :math:`c_0, c_1, c_2` with respect to :math:`A,B`.

    """
    return np.array(
        [
            [0.0, 1.0],
            [1.0, -2.0 - 6.0 * B],
            [-B, 3.0 * B**2 + 2.0 * B - A],
        ]
    )


WIDOM_SLOPE: float = 0.8 * 0.3381965009398633
"""Slope of the Widom line in the A-B space, reverse-engineered from data available for
water."""


@_COMPILER(
    nb.f8(nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def widom_line(A: float) -> float:
    r"""Parametrization of the Widom-line for the PR EoS in the A-B space,
    reverse-engineered from data available for water.

    References:
        `Maxim, et al. (2019) <hhttps://doi.org/10.1038/s41467-019-12117-5>`_

    Parameters:
        A: Dimensionless cohesion.

    Returns:
        The Widom-line parametrized as ``B(A)`` in the A-B space

        .. math::

            B_{crit} + 0.8 \cdot 0.3381965009398633 \cdot \left(A - A_{crit}\right)

    """
    return B_CRIT + WIDOM_SLOPE * (A - A_CRIT)


@_COMPILER(
    [
        nb.f8[:](nb.f8[:], nb.f8[:, :], nb.f8[:, :]),
        nb.f8[:](
            nb.f8[:],
            nb.types.Array(nb.f8, 2, "C", readonly=True),
            nb.types.Array(nb.f8, 2, "C", readonly=True),
        ),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def project_point_to_line(p: np.ndarray, line: np.ndarray, A: np.ndarray) -> np.ndarray:
    r"""Returns the projection of a point ``p`` onto a line given by two points.

    Parameters:
        p: ``shape=(n,)``

            Point of dimension ``n``.
        line: ``shape=(2, n)``

            2D array containing points per row spanning the line.
        A: ``shape=(n, n)``

            A matrix for for the metric :math:`\langle x, Ay\rangle`. Use the identity
            to use regular Euclidean geometry.

    Returns:
        A 1D array which is the normal projection of ``p`` onto the line.

    """
    A = line[0]
    B = line[1]
    p = np.asarray(p)

    AB = B - A
    AP = p - A

    # Avoid division by zero if A == B (degenerate line)
    AB_norm_sq = np.dot(AB, AB)
    if np.isclose(AB_norm_sq, 0):
        return A.copy()

    # Scalar projection: t = (AP · AB) / ||AB||²
    t = np.dot(AP, AB) / AB_norm_sq
    # Closest point on the line
    Q = A + t * AB

    return Q


WIDOM_LINE: np.ndarray = np.array(
    [
        [0.0, widom_line(0.0)],
        [A_CRIT, widom_line(A_CRIT)],
    ],
    dtype=np.float64,
)
r"""2D array containing points per row spanning the Widom line.

The points are created by using :func:`widom_line` for :math:`A\in\{0, A_{crit}\}`.

"""

WIDOM_VEC: np.ndarray = WIDOM_LINE[1] - WIDOM_LINE[0]
"""Vector spanning the Widom line."""
ABc: np.ndarray = -np.array([A_CRIT, B_CRIT])
"""Vector spanning the super-critical line."""
THETA_WIDOM_SC = np.atan2(
    WIDOM_VEC[0] * ABc[1] - WIDOM_VEC[1] * ABc[0], np.dot(WIDOM_VEC, ABc)
)
"""Angle between critical line and Widom line."""
THETA_WIDOM_BC = np.atan2(WIDOM_VEC[1], WIDOM_VEC[0])
"""Angle between line ``B=B_CRIT`` and Widom line."""


@_COMPILER(
    nb.bool(nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def is_supercritical(A: float, B: float) -> bool:
    """Checks whether the pair of cohesion and covolume values lies in the supercritical
    area.

    I.e., covolume is on or above the critical value and the critical line
    :math:`\\frac{B_c}{A_c} * A`

    Parameters:
        A: Dimensionless cohesion parameter.
        B: Dimensionless covolume parameter.

    Returns:
        True, if it is in the supercritical area, False otherwise.

    """
    return B >= CRITICAL_SLOPE * A or B >= B_CRIT


@_COMPILER(
    [
        nb.void(nb.f8[:], nb.f8, nb.bool, nb.f8[:]),
        nb.void(nb.f8[:], nb.f8, nb.bool, nb.f8[:, :]),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _smooth_3root_region(
    z: np.ndarray, s: float, gaslike: bool, out: np.ndarray
) -> None:
    """Smoothing of roots in the physical 3-root region, close to the border where the
    intermediate root approaches either the biggest or the smallest root.

    Parameters:
        z: 1D array with shape ``(3,)`` containing the 3 real roots ordered by size.
        s: Small number saying how close the intermediate root has to be to start
            smoothing. Must be in ``(0, 0.25]``.
        gaslike: Either smoothing last/largest root (True) or the first/smallest root
            (False).
        out: Array to be smoothed. Can be the roots (i.e., equal to z), or their
            derivatives. Must be of shape ``(3,m)``.

    """
    assert z.shape == (3,), "Roots must be (3,)-array."
    assert out.shape[0] == 3, "Require at least 3 values to be smoothed."
    assert 0 < s <= 0.25, "Require s to be in (0, 0.25]"

    d = (z[1] - z[0]) / (z[2] - z[0])

    if gaslike:
        i = 2

        if d >= 1 - s:
            w = 1.0
        elif d <= 1 - 2 * s:
            w = 0.0
        else:
            w = (d - (1 - 2 * s)) / s
            w = w**2 * (3 - 2 * w)
    else:
        i = 0

        if d <= s:
            w = 1.0
        elif d >= 2 * s:
            w = 0.0
        else:
            w = (d - s) / s
            w = 1.0 - (w**2) * (3 - 2 * w)

    out[i] = out[i] * (1 - w) + (out[1] + out[i]) * 0.5 * w


@_COMPILER(
    nb.f8(nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def extended_factor(Z: float, B: float) -> float:
    """Extended compressibility factor using the real part of the complex-conjugated
    roots in areas where there is only 1 root (Ben Gharbia 2021).

    Note:
        The sum of the roots of any cubic normalized cubic polynomial is equal to
        :math:`-c_2`. That's how this is derived.

    Parameters:
        Z: The 1 real root.
        B: Dimensionless co-volume.

    Returns:
        :math:`\\frac{1 - B - Z}{2}`

    """
    return (1 - B - Z) * 0.5


@_COMPILER(
    nb.f8[:](nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def extended_factor_derivatives(dZ: np.ndarray) -> np.ndarray:
    """The derivatives of :func:`extended_factor` dependent on the derivatives of the 1
    real root.

    Parameters:
        d_Z: ``shape=(2,)``

            The derivatives of ``Z`` w.r.t. to cohesion and co-volume.

    Returns:
        The derivative of :func:`extended_factor` w.r.t. the cohesion and covolume.

    """
    return -0.5 * np.array([dZ[0], 1 + dZ[1]])


@_COMPILER(
    nb.int_(nb.f8, nb.f8, nb.bool, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def is_extended_factor(A: float, B: float, gaslike: bool, eps: float) -> int:
    """Method implementing the extension procedure logic to defining the zone and
    method of providing a value for the compressibility factor, where it is physically
    not available.

    It returns an integer encoding the extension procedure, which depends on the
    cohesion, covolume and whether the gaslike root is requested or not.

    Parameters:
        A: Dimensionless cohesion.
        B: Dimensionless covolume.
        gaslike: True, if check is performed for gaslike root, False for liquid-like.
        eps: See :meth:`~porepy.compositional.peng_robinson.compressibility_factor.
            get_root_case`

    Returns:
        One of the following codes is returned

        - 0: The indicated root (``gaslike`` or not) is not obtained through an
          extension procedure and is real. It is an actual root of the polynomial.
        - 1: The indicated root is liquidlike and in the subcritical area, where it is
          extended. The (bigger) gaslike root is real.
        - 2: The indicated root is gaslike and in the subcritical area, where it is
          extended. The (smaller) liquidlike root is real.
        - ``10 + i``, with ``i`` being the number of roots (0 for triple root):
          The indicated root is liquidlike and in the supercritical area, where it
          is extended. The gaslike root is real.
        - ``20 + i``, with ``i`` being the number of roots (0 for triple root):
          The indicated root is gaslike and in the supercritical area, where it
          is extended. The liquidlike root is real.

    """
    c = c_from_AB(A, B)
    nroot = get_root_case(c, eps)
    is_sc = is_supercritical(A, B)
    # NOTE. Supercritical line and super-critical liquid-gas border are such that the
    # halfspace below them is open.
    above_sc_lg = B >= widom_line(A)

    # Default return value is that it is not extended.
    is_extended = 0

    if is_sc:
        # Extension codes 10 - 13 for when supercritical liquid needs extension.
        if above_sc_lg and not gaslike:
            is_extended = 10 + nroot
        # Extnesion codes 20 - 23 for when supercritical gas needs extension.
        elif gaslike and not above_sc_lg:
            is_extended = 20 + nroot
    else:
        # Extension according to Ben Gharbia.
        if nroot == 1:
            Z = one_root(c)
            Wsub = extended_factor(Z[0], B)
            if not gaslike and Z > Wsub:
                is_extended = 1
            elif gaslike and Z < Wsub:
                is_extended = 2

    return is_extended


@_COMPILER(nb.f8(nb.f8, nb.f8), cache=True, fastmath=NUMBA_FAST_MATH)
def Sigmoid(t: float, k: float) -> float:
    """Logarithmic sigmoid function.

    .. math::

        S(t) = \\frac{1}{1 + e^{-k(t - \\frac{1}{2})}}

    Its normalization can be used for smoothing.

    Note:
        The derivative is simply :math:`S^{\\prime}(t) = k S(t)(1 - S(t))`.

    Parameters:
        t: (Normalized) argument.
        k: Slope.

    Returns:
        The value of above function.

    """
    return 1.0 / (1 + np.exp(-k * (t - 0.5)))


@_COMPILER(nb.f8(nb.f8, nb.f8), cache=NUMBA_CACHE, fastmath=NUMBA_FAST_MATH)
def fab(A: float, B: float) -> float:
    AB = np.array([A - A_CRIT, B - B_CRIT])

    if B <= widom_line(A):  # SCG extension
        thn = THETA_WIDOM_BC
        # Angle between AB and horizontal line: atan((1, 0) X AB, dot)
        th = np.atan2(AB[1], AB[0])
        # Shift the AB point parallel to the Widom line onto the horizontal line B=Bc.
        Asc = A + (B_CRIT - B) / WIDOM_VEC[1] * WIDOM_VEC[0]
        Bsc = B_CRIT
        Zsc = calculate_roots(c_from_AB(Asc, B_CRIT), 1e-14)[-1]
    else:  # SCL extension
        thn = THETA_WIDOM_SC
        # Angle between AcBc -> AB and AcBc -> 00.
        th = np.atan2(AB[0] * ABc[1] - AB[1] * ABc[0], np.dot(AB, ABc))
        # Rotate onto super-critical line 00 -> AcBc counter-clockwise.
        Asc = max(0.0, A_CRIT + AB[0] * np.cos(th) - AB[1] * np.sin(th))
        Bsc = CRITICAL_SLOPE * Asc
        # Evaluate value on super-critical line.
        Zsc = calculate_roots(c_from_AB(Asc, Bsc), 1e-14)[-1]
        # Weigh towards value 1 on Widom line using the angle fraction.

    f = max((1 - 3 * Bsc - Zsc) / (Zsc - Bsc) * 0.5, 1e-14)
    w = max(min(th / thn, 1.0), 0.0)
    return (1.0 - w) * f + w


@_COMPILER(nb.f8[:](nb.f8, nb.f8, nb.f8), cache=NUMBA_CACHE, fastmath=NUMBA_FAST_MATH)
def dfab(A: float, B: float, sc_reg: float) -> np.ndarray:
    AB = np.array([A - A_CRIT, B - B_CRIT])

    if B <= widom_line(A):
        thn = THETA_WIDOM_BC
        y = AB[1]
        x = AB[0]
        r = max(x**2 + y**2, sc_reg)
        th = np.atan2(y, x)
        dthdA = -y / r
        dthdB = x / r

        Asc = A + (B_CRIT - B) / WIDOM_VEC[1] * WIDOM_VEC[0]
        dAscdB = -WIDOM_VEC[0] / WIDOM_VEC[1]

        Bsc = B_CRIT
        dBscdA = 0.0
        dBscdB = 0.0

        cc = c_from_AB(Asc, B_CRIT)
        Zsc = calculate_roots(cc, 1e-14)[-1]
        dZsc = np.dot(calculate_root_derivatives(cc, 1e-14), dc_from_AB(Asc, B_CRIT))[
            -1
        ]
        dZsc = np.array((dZsc[0], dZsc[0] * dAscdB))
    else:
        thn = THETA_WIDOM_SC
        y = AB[0] * ABc[1] - AB[1] * ABc[0]
        x = np.dot(AB, ABc)
        r = max(x**2 + y**2, sc_reg)
        th = np.atan2(y, x)
        dthdy = x / r
        dthdx = -y / r
        dthdA = dthdy * ABc[1] + dthdx * ABc[0]
        dthdB = -dthdy * ABc[0] + dthdx * ABc[1]

        sth = np.sin(th)
        cth = np.cos(th)

        Asc = A_CRIT + AB[0] * cth - AB[1] * sth
        dAscdA = cth - (AB[0] * sth + AB[1] * cth) * dthdA
        dAscdB = -sth - (AB[0] * sth + AB[1] * cth) * dthdB

        Bsc = CRITICAL_SLOPE * Asc
        dBscdA = CRITICAL_SLOPE * dAscdA
        dBscdB = CRITICAL_SLOPE * dAscdB

        cc = c_from_AB(Asc, Bsc)
        Zsc = calculate_roots(cc, 1e-14)[-1]
        dZsc = np.dot(calculate_root_derivatives(cc, 1e-14), dc_from_AB(Asc, Bsc))[-1]
        dZsc = np.array(
            (dZsc[0] * dAscdA + dZsc[1] * dBscdA, dZsc[0] * dAscdB + dZsc[1] * dBscdB)
        )

    f = max((1 - 3 * Bsc - Zsc) / (Zsc - Bsc) * 0.5, 1e-14)
    df = (
        ((4 * Bsc - 1) * dZsc + (1 - 4 * Zsc) * np.array([dBscdA, dBscdB]))
        / (Zsc - Bsc) ** 2
        * 0.5
    )
    w = max(min(th / thn, 1.0), 0.0)
    dw = np.array((dthdA, dthdB)) / thn
    return (1.0 - w) * df + dw * (1 - f)


@_COMPILER(
    nb.f8(nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def W_fab(A: float, B: float, Z: float) -> float:
    return B + (Z - B) * fab(A, B)


@_COMPILER(
    nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8[:], nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def dW_fab(
    A: float,
    B: float,
    Z: float,
    dZ: np.ndarray,
    sc_reg: float,
    sc_bw: float,
    sc_ss: float,
) -> np.ndarray:
    f = fab(A, B)
    df = dfab(A, B, sc_reg)
    dW = np.array([f * dZ[0] + (Z - B) * df[0], 1 + (Z - B) * df[1] + f * (dZ[1] - 1)])

    # NOTE: Idea is to smooth derivatives towards the the values at the critical lines
    # to avoid jumps in derivatives.
    w = 1.0
    dWc = np.zeros(2)
    if B <= widom_line(A):  # SCG smoothing
        cd = abs(B - B_CRIT)
        dmax = sc_bw * B_CRIT
        if cd < dmax:
            cc = c_from_AB(A, B_CRIT)
            dZ = (calculate_root_derivatives(cc, 1e-14) @ dc_from_AB(A, B_CRIT))[-1]
            dWc = extended_factor_derivatives(dZ)
            w = max(min(cd / dmax, 1.0), 0.0)
    elif B <= B_CRIT:  # SCL smoothing
        Asc = B / CRITICAL_SLOPE
        cd = abs(A - Asc)
        dmax = 0.5 * sc_bw * A_CRIT
        if cd < dmax:
            cc = c_from_AB(Asc, B)
            dZ = (calculate_root_derivatives(cc, 1e-14) @ dc_from_AB(Asc, B))[-1]
            dWc = extended_factor_derivatives(dZ)
            w = max(min(cd / dmax, 1.0), 0.0)
    S0 = Sigmoid(0, sc_ss)
    S1 = Sigmoid(1, sc_ss)
    w = (Sigmoid(w, sc_ss) - S0) / (S1 - S0)

    # Cancel smoothing if around critical point.
    dc = conorm(A - A_CRIT, B - B_CRIT)
    w = 1.0 if dc < sc_bw else w
    dW = (1 - w) * dWc + w * dW

    return dW


@_COMPILER(
    nb.f8(nb.f8, nb.f8, nb.bool, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def get_compressibility_factor(
    A: float,
    B: float,
    gaslike: bool,
    eps: float,
) -> float:
    """Compute the compressibility factor for given :math:`A` and :math:`B`.

    Depending on the value of ``gaslike``, the largest or smallest root is returned
    in the case of three real roots.

    In the case of a single real root, that root is returned.

    An extension procedure is applied in the one-root area, among others.
    Negative values of ``A`` or ``B`` are projected to zero.
    Additionally, ``B`` is caped by :data:`COVOLUME_LIMIT`.

    See also:
        :func:`is_extended` for more information on extension procedures.

    Parameters:
        A: Dimensionless cohesion parameter.
        B: Dimensionless covolume parameter.
        gaslike: Flag indicating whether to return the gas-like (True) or liquid-like
            (False) root.
        eps: Tolerance for detection of degeneracy/two-root and triple root case.

    Returns:
        The compressibility factor.

    """
    A = 0.0 if A < 0.0 else A
    # Zero covolume limit case leads to largest root being <=B (numerically zero) for
    # A>=0.25, i.e. past the liquid-saturated curve.
    # Limit how small B can get to avoid that limit case.
    # NOTE that for limit A=0, the largest root is always numerically > B.
    B_limit_reached = False
    B_original = max(B, 0.0)
    if B < COVOLUME_LIMIT and (not gaslike or A > 0.25):
        B = COVOLUME_LIMIT
        B_limit_reached = True
        assert B != B_original, "Copy error for B original."

    c = c_from_AB(A, B)

    # NOTE: For the 3-root case, we can safe some computations by computing only root
    # we need. Not sure how much it saves.
    # NOTE: c contains the coefficients as the polynomial is read from left to right:
    # C[0] contains c_2, c[2] contains c_0
    # NOTE: Roots always ordered by size.
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return np.nan
    roots = calculate_roots(c, eps)
    assert roots[-1] > B, (
        "Expecting largest compressibility factor to be greater than covolume."
    )

    extension_case = is_extended_factor(A, B, gaslike, eps)

    match extension_case:
        case 0:  # No Extension, nothing to do.
            pass
        case 1:  # Sub-critical liquid extension
            roots[0] = extended_factor(roots[-1], B)
        case 2:  # Sub-critical gas extension.
            roots[-1] = extended_factor(roots[0], B)
        case 10 | 20:  # Only known triple root is super-critial point.
            if not np.allclose((A, B), (A_CRIT, B_CRIT)):
                raise NotImplementedError(
                    "Encountered triple root which is not critical point."
                )
        # Super-critical liquid extension.
        # There are non-physical regions with num_roots != 1, which need treatment.
        # The value of the smallest root can go below B and needs extra attention.
        # Point A, B = (0,0) for example.
        case 11 | 12 | 13:
            roots[0] = W_fab(A, B, roots[-1])
        # Super-critical gas extension.
        # Contrary to the super-critical liquid, we only know how to deal with the
        # 1-root case.
        case 21 | 22 | 23:
            roots[-1] = W_fab(A, B, roots[-1])
        case _:
            raise NotImplementedError(
                f"Uncovered extension case {extension_case} for A,B = {(A, B)}."
            )

    # Sanity check: This order must always hold, otherwise we are in an uncovered case.
    if roots[0] > roots[-1]:
        raise NotImplementedError(
            f"Encountered an A-B pair violating Zl <= Zg: A, B = {A} {B}"
        )

    raise_B_lim_error = False
    # These type of violations have so far only been observed when the super-critical
    # liquid extension is required.
    if not gaslike and roots[0] <= B and extension_case >= 10:
        # In the limit case B = 0 and close to A=B=0, all kind of violations can happen
        # We avoid the lower-B-bound violation and pay with an approximation error.
        # Compute distance to zero point.
        # NOTE: Use critical slope to get a metric in AB space which considers the
        # discrepancy in size between A and B.
        d = conorm(A, B_original)
        if B_limit_reached or d <= 5e-3:
            roots[0] = max(eps, 1.1 * B_original)
        else:
            raise_B_lim_error = True

        if roots[0] <= B_original or raise_B_lim_error:
            raise NotImplementedError(
                f"Encountered A-B pair violating B < Zl: A, B = {A} {B}"
            )

    # Since ordered by size, gaslike is largest root and liquidlike is smallest.
    if gaslike:
        return roots[-1]
    else:
        return roots[0]


@_COMPILER(
    nb.f8[:](nb.f8, nb.f8, nb.bool, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def get_compressibility_factor_derivatives(
    A: float,
    B: float,
    gaslike: bool,
    eps: float,
    sm: float,
    sc_reg: float,
    sc_bw: float,
    sc_ss: float,
) -> np.ndarray:
    """Compute the derivatives of the compressibility factor with respect to :math:`A`
    and :math:`B`.

    For more information, see :func:`Z`.

    Parameters:
        A: Dimensionless cohesion parameter.
        B: Dimensionless covolume parameter.
        gaslike: Flag indicating whether to return the gas-like (True) or liquid-like
            (False) root.
        eps: Tolerance for detection of degeneracy/two-root and triple root case.
        sm: Smoothing parameter for the three-root area.
        sc_reg: Regularization parameter for super-critical extension.
        sc_bw: Bandwidth for super-critical smoothing.
        sc_ss: Slope for super-critical smoothing.

    Returns:
        A ``(2,)``-array containing the derivatives w.r.t. cohesion and covolume.

    """

    # NOTE for more information on individual steps, see get_compressibility_factor, as
    # the computations are analogous.

    A = 0.0 if A < 0.0 else A
    B_limit_reached = False
    B_original = max(B, 0.0)
    if B < COVOLUME_LIMIT and (not gaslike or A > 0.25):
        B = COVOLUME_LIMIT
        B_limit_reached = True
        assert B != B_original, "Copy error for B original."

    c = c_from_AB(A, B)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return np.full((2,), np.nan)
    # Chainrule to obtain derivatives w.r.t. A and B.
    droots: np.ndarray = np.dot(calculate_root_derivatives(c, eps), dc_from_AB(A, B))

    roots = calculate_roots(c, eps)
    assert roots[-1] > B, (
        "Expecting largest compressibility factor to be greater than covolume."
    )

    extension_case = is_extended_factor(A, B, gaslike, eps)

    match extension_case:
        case 0:
            if droots.shape[0] == 3 and sm > 0.0:
                _smooth_3root_region(roots, sm, gaslike, droots)
        case 1:
            droots[0] = extended_factor_derivatives(droots[0])
        case 2:
            droots[-1] = extended_factor_derivatives(droots[-1])
        case 10 | 20:
            if not np.allclose((A, B), (A_CRIT, B_CRIT)):
                raise NotImplementedError(
                    "Encountered triple root which is not critical point."
                )
        case 11 | 12 | 13:
            droots[0] = dW_fab(A, B, roots[-1], droots[-1], sc_reg, sc_bw, sc_ss)
        case 21 | 22 | 23:
            droots[-1] = dW_fab(A, B, roots[-1], droots[-1], sc_reg, sc_bw, sc_ss)
        case _:
            raise NotImplementedError(
                f"Uncovered extension case {extension_case} for A,B = {(A, B)}."
            )

    if not gaslike and roots[0] <= B and extension_case >= 10:
        d = conorm(A, B_original)
        if B_limit_reached or d <= 5e-3:
            droots[0] = np.array([0.0, 1.1 if B_original > 0.0 else 0.0])

    if gaslike:
        dZ = droots[-1]
    else:
        dZ = droots[0]

    # Regularization around the critical point.
    # d = conorm(A - A_CRIT, B_CRIT)
    d = conorm(A - A_CRIT, B - B_CRIT)
    dmax = sc_bw * B_CRIT
    if d <= dmax:
        S0 = Sigmoid(0, sc_ss)
        S1 = Sigmoid(1, sc_ss)
        w = (Sigmoid(d / dmax, sc_ss) - S0) / (S1 - S0)
        dZ = (1 - w) * dZ_CRIT + w * dZ
    # if extension_case > 10:
    #     dZ = np.clip(dZ, -5.0, 5.0)
    return dZ
