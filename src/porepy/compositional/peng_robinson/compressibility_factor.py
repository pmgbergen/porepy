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
al. (2021).

I.e., it provides an additional root in the 1-root area.

"""

from __future__ import annotations

from typing import Literal

import numba as nb
import numpy as np

from .._core import NUMBA_CACHE, NUMBA_FAST_MATH, njit
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
    "critical_line",
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


ABMETRIC: np.ndarray = np.diag((B_CRIT / A_CRIT, 1.0))
"""Metric for computing distances in the AB space :meth:`\\langle x, Ay\\rangle`.

This is required to properly determin distance in the AB space, since covolume and
cohesion are of different orders.

Intended use is for arrays of shape ``(2,)``, where the first entry is a cohesion value
and the second a covolume value.
Scales the cohesion dimension down to operate on distances relevant for the covolume
dimension.

"""


COVOLUME_LIMIT: float = 1e-7
""""Below this value, the covolume is considered zero.

Required to treat the limit case of B -> 0.

"""


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
        The returned array contains the coefficients as the polynomial is read from
        left to right: ``c[0]`` contains :math:`c_2`, ``c[2]`` contains :math:`c_0`.

    Parameters:
        A: Dimensionless cohesion.
        B: Dimensionless covolume.

    Returns:
        A ``(3,)``-array containing the coefficients :math:`c_2, c_1, c_0`.

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
        :math:`c_2, c_1, c_0` with respect to :math:`A,B`.

    """
    return np.array(
        [
            [0.0, 1.0],
            [1.0, -2.0 - 6.0 * B],
            [-B, 3.0 * B**2 + 2.0 * B - A],
        ]
    )


@_COMPILER(
    nb.f8(nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def critical_line(A: float) -> float:
    r"""Parametrization of the critical line for the PR EoS in the A-B space.

    Parameters:
        A: Non-dimensional cohesion.

    Returns:
        The critical line parametrized as ``B(A)``

        .. math::

            \\frac{B_{crit}}{A_{crit}} A

    """
    return (B_CRIT / A_CRIT) * A


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
    return B_CRIT + 0.8 * 0.3381965009398633 * (A - A_CRIT)


@_COMPILER(
    nb.f8(nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def approximate_sc_lg_border(A: float) -> float:
    """Approximate border between liquid and gas in the supercritical area.

    Approximates the incline of the saturated curves at the critical point.

    Parameters:
        A: Dimensionless cohesion.

    Returns:
        Parametrization of the line sa :math:`B(A)`.

    """
    # Shift from middle of 2-phase region.
    shift = 0.25 / 2.0 + 0.01105
    return B_CRIT / (A_CRIT - shift) * A + B_CRIT - A_CRIT * B_CRIT / (A_CRIT - shift)


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


SUPERCRITICAL_LINE: np.ndarray = np.array(
    [
        [0.0, 0.0],
        [A_CRIT, B_CRIT],
    ],
    dtype=np.float64,
)
r"""2D array containing points per row spanning the super-critical line

.. math::

    (0,0),~(A_{crit},B_{crit})

See :data:`B_CRIT`, data:`A_CRIT`.

"""


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

SUPERCRITICAL_LG_LINE: np.ndarray = np.array(
    [
        [0.0, approximate_sc_lg_border(0.0)],
        [A_CRIT, approximate_sc_lg_border(A_CRIT)],
    ],
    dtype=np.float64,
)
r"""2D array containing points per row spanning the approximate gas-liquid border in the
supercritical area..

The points are created by using :func:`approximate_sc_lg_border` for
:math:`A\in\{0, A_{crit}\}`.

"""

_SC_BORDER_LINE = WIDOM_LINE
"""Shortcut to the line used to separate liquid- and gas-like supercritical roots."""

_SC_BORDER_FUNC = widom_line
"""Shortcut to the parametrization of the line separating liquid- and gas-like
supercritical roots."""


@_COMPILER(
    nb.bool(nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def is_supercritical(A: float, B: float) -> bool:
    """Checks whether the pair of cohesion and covolume values lies in the supercritical
    area.

    I.e., covolume is below the critical value and the critical line
    :math:`\\frac{B_c}{A_c} * A`

    Parameters:
        A: Dimensionless cohesion parameter.
        B: Dimensionless covolume parameter.

    Returns:
        True, if it is in the supercritical area, False otherwise.

    """
    return B >= critical_line(A) or B >= B_CRIT


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
    [
        nb.void(nb.f8, nb.f8, nb.f8, nb.f8[:]),
        nb.void(nb.f8, nb.f8, nb.f8, nb.f8[:, :]),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def _smooth_supercritical_transition(
    B: float, W: float, T: float, out: np.ndarray
) -> None:
    """Smoothing the super-critical liquid-like root when it comes close to a target
    value.

    The smoothing is performed using a cubic function which is 1 if ``W==B`` and zero
    if ``W==T``, decreasing monotonously in between, assuming ``B <= W <= T``.
    I.e., assuming a threshold ``T`` the value ``W`` is smoothed towards ``B`` once it
    falls below that threshold.

    .. math::

        d = \\lvert W - B \\rvert
        s = \\lvert T - B \\rvert
        w = 1 - (\\frac{d}{s})^2 (3 - 2 \\frac{d}{s})
        out[0] = out[0](1 - w) + out[1] w

    ``out`` is operated on by reference. This design-choice is made so that this
    function can be used to smooth both, ``W`` and potentially it's derivatives.

    Parameters:
        B: Reference value for distance.
        W: Value to be smoothed.
        T: Top or target value towards which it is smoothed.
        s: Fraction of range on which smoothing is applied
        out: An output array of shape ``(2,n)``, containing the values to be smoothed at
            index 0 and the values towards which it is smoothed at index 1.

    """
    d = np.abs((W - B))
    s = np.abs((T - B))
    assert s >= d, "Expecting the |W-B| <= |T - B|"
    assert B <= W, "Expecting B <= W"

    if d >= s:
        w = 0.0
    elif d <= 0.0:
        w = 1.0
    else:
        w = 1.0 - (d / s) ** 2 * (3 - 2 * d / s)

    out[0] = out[0] * (1 - w) + out[1] * w


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
    nb.f8(nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def extended_factor_scl(Z: float, B: float) -> float:
    """Extended compressibility factor for the liquid root in the super-critical area.

    Parameters:
        Z: The 1 real root.
        B: Dimensionless co-volume.

    Returns:
        :math:`\\frac{B + Z}{2}`

    """
    return (Z + B) * 0.5


@_COMPILER(
    nb.f8[:](nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def extended_factor_scl_derivatives(dZ: np.ndarray) -> np.ndarray:
    """The derivatives of :func:`extended_factor_scl` dependent on the derivatives of
    the 1 real root.

    Parameters:
        d_Z: ``shape=(2,)``

            The derivatives of ``Z`` w.r.t. to cohesion and co-volume.

    Returns:
        The derivative of :func:`extended_factor_scl` w.r.t. the cohesion and covolume.

    """
    return 0.5 * (dZ + np.array([0.0, 1.0]))


@_COMPILER(
    nb.f8(nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def extended_factor_scg(Z: float, B: float) -> float:
    """Extended compressibility factor for the gas root in the super-critical area.

    Parameters:
        Z: The 1 real root.
        B: Dimensionless co-volume.

    Returns:
        :math:`B + \\frac{1 - B - Z}{2}`

    """
    # return B + extended_factor(Z, B)
    return B + Z


@_COMPILER(
    nb.f8[:](nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def extended_factor_scg_derivatives(dZ: np.ndarray) -> np.ndarray:
    """The derivatives of :func:`extended_factor_scg` dependent on the derivatives of
    the 1 real root.

    Parameters:
        d_Z: ``shape=(2,)``

            The derivatives of ``Z`` w.r.t. to cohesion and co-volume.

    Returns:
        The derivative of :func:`extended_factor_scg` w.r.t. the cohesion and covolume.

    """
    # return np.array([0.0, 1.0]) + extended_factor_derivatives(dZ)
    return np.array([0.0, 1.0]) + dZ


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
          extended. The gaslike root is real.
        - 2: The indicated root is gaslike and in the subcritical area, where it is
          extended. The liquidlike root is real.
        - ``10 + i``, with ``i`` being the number of roots (0 for triple root):
          The indicated root is liquidlike and in the supercritical area, where it
          is extended. The gaslike root is real.
        - ``20 + i``, with ``i`` being the number of roots (0 for triple root):
          The indicated root is gaslike and in the supercritical area, where it
          is extended. The liquidlike root is real.

    """
    c = c_from_AB(A, B)
    nroot = get_root_case(c[0], c[1], c[2], eps)
    is_sc = is_supercritical(A, B)
    # NOTE. Supercritical line and super-critical liquid-gas border are such that the
    # halfspace below them is open.
    above_sc_lg = B >= _SC_BORDER_FUNC(A)

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
            Z = one_root(c[0], c[1], c[2])
            Wsub = extended_factor(Z[0], B)
            if not gaslike and Z > Wsub:
                is_extended = 1
            elif gaslike and Z < Wsub:
                is_extended = 2

    return is_extended


@_COMPILER(
    nb.f8(nb.f8, nb.f8, nb.bool, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def get_compressibility_factor(
    A: float,
    B: float,
    gaslike: bool,
    eps: float,
    smooth3: float,
) -> float:
    """Compute the compressibility factor for given :math:`A` and :math:`B`.

    Depending on the value of ``gaslike``, the largest or smallest root is returned
    in the case of three real roots.

    In the case of a single real root, that root is returned.

    An extension procedure is applied in the one-root area, among others.
    Negative values of ``A`` or ``B`` are projected to zero.

    See also:
        :func:`is_extended` for more information on extension procedures.

    Parameters:
        A: Dimensionless cohesion parameter.
        B: Dimensionless covolume parameter.
        gaslike: Flag indicating whether to return the gas-like (True) or liquid-like
            (False) root.
        eps: Tolerance for detection of degeneracy/two-root and triple root case.
        smooth3: Smoothing parameter for the three-root area.

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
    roots = calculate_roots(c[0], c[1], c[2], eps)
    assert roots[-1] > B, (
        "Expecting largest compressibility factor to be greater than covolume."
    )

    extension_case = is_extended_factor(A, B, gaslike, eps)

    # Shortcuts for quick switching between models.
    Wgsub = extended_factor
    Wlsub = extended_factor
    Wgsc = extended_factor_scg
    Wlsc = extended_factor_scl

    # Index for super-critical smoothing. Switches to zero or -1 indicating which root
    # needs smoothing. Using potential indices as indicators.
    smooth_sc_idx: Literal[-1, 0, 1] = 1
    # Bandwidth around borders of regions in the super-critical area for smoothing.
    smooth_sc: float = 0.07

    match extension_case:
        # No root is extended.
        case 0:
            # In the sub-critical regime, there is a physical 3-root region.
            # If requested, we smooth the roots close to the phase borders where 1 phase
            # disapears.
            if roots.size == 3 and smooth3 > 0.0:
                _smooth_3root_region(roots, smooth3, gaslike, roots)
        # Sub-critical liquid extension.
        case 1:
            assert roots.size == 1, "Expecting only 1 real root in extension cases 1."
            roots[0] = Wlsub(roots[-1], B)
        # Sub-critical gas extension.
        case 2:
            assert roots.size == 1, "Expecting only 1 real root in extension cases 2."
            roots[-1] = Wgsub(roots[0], B)
        # Super-critical liquid extension.
        # There are non-physical regions with num_roots != 1, which need treatment.
        # We cannot use the Ben Gharbia extension, as that value goes below B in the
        # supercritical area. Includes the 2 root point A,B = (0, 0)
        case 10 | 11 | 12 | 13:
            roots[0] = Wlsc(roots[-1], B)
            smooth_sc_idx = 0
        # Known super-critical triple points is the critical point Ac Bc.
        case 20:
            pass
        # Super-critical gas extension.
        # Contrary to the super-critical liquid, we only know how to deal with the
        # 1-root case.
        case 21:
            assert roots.size == 1, "Expecting only 1 real root in extension cases 21."
            roots[-1] = Wgsc(roots[0], B)
            smooth_sc_idx = -1
        # Extension case 22 and 23 are uncovered.
        case _:
            raise NotImplementedError(
                f"Uncovered extension case {extension_case} for A,B = {(A, B)}."
            )

    # If in super-critical extension case, smooth towards border lines.
    if smooth_sc_idx in [-1, 0] and smooth_sc > 0.0:
        # First, smooth towards super-critical gas-liquid border in any case
        AB = np.array([A, B])
        # Normal projection onto line.
        AB_p = project_point_to_line(AB, _SC_BORDER_LINE, ABMETRIC)
        D = AB - AB_p
        d = np.sqrt(np.dot(D, ABMETRIC @ D))
        # Avoid a conflict with the other smoothing by demanding the projected B to be
        # bigger than B_crit.
        if d < smooth_sc and AB_p[1] >= B_CRIT:
            c = c_from_AB(AB_p[0], AB_p[1])
            Z = calculate_roots(c[0], c[1], c[2], eps)
            out = np.array([roots[smooth_sc_idx], Z[-1]])
            _smooth_supercritical_transition(0, d, smooth_sc, out)
            roots[smooth_sc_idx] = out[0]

        # If gas extended, smooth towards sub-critical extension value on horizontal
        # B=Bcrit.
        if smooth_sc_idx == -1:
            d = B - B_CRIT
            # Floating point operations can cause it to be slightly negative.
            d = 0.0 if d < 0.0 else d
            if d < smooth_sc:
                c = c_from_AB(A, B_CRIT)
                Z = calculate_roots(c[0], c[1], c[2], eps)
                if Z.size > 1:
                    raise NotImplementedError(
                        "SC-smoothing has ambiguous target value."
                    )
                W = Wgsub(Z[0], B_CRIT)
                out = np.array([roots[-1], W])
                _smooth_supercritical_transition(0, d, smooth_sc, out)
                roots[-1] = out[0]
        # If liquid extended, smooth towards sub-critical extension value on critical
        # line.
        elif smooth_sc_idx == 0:
            # Normal projection onto line.
            AB_p = project_point_to_line(AB, SUPERCRITICAL_LINE, ABMETRIC)
            D = AB - AB_p
            d = np.sqrt(np.dot(D, ABMETRIC @ D))
            # d = np.linalg.norm(AB - AB_p)
            # Avoid conflicts with the SC border line smoothing by demanding B <= B_crit
            if d < smooth_sc and AB_p[1] <= B_CRIT:
                c = c_from_AB(AB_p[0], AB_p[1])
                # Near A=B=0, it can lead to more than 1 real root. In any case it is
                # the gas root which is real and which is used for the extension.
                Z = calculate_roots(c[0], c[1], c[2], eps)
                W = Wlsub(Z[-1], AB_p[1])
                out = np.array([roots[0], W])
                _smooth_supercritical_transition(0, d, smooth_sc, out)
                roots[0] = out[0]

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
        d = np.sqrt(B_original**2 + ABMETRIC[0, 0] * A**2)
        if B_limit_reached or d <= 5e-3:
            if B_original == 0.0:
                roots[0] = eps
            else:
                roots[0] = 1.1 * B_original
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
    nb.f8[:](nb.f8, nb.f8, nb.bool, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def get_compressibility_factor_derivatives(
    A: float,
    B: float,
    gaslike: bool,
    eps: float,
    smooth3: float,
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
        smooth3: Smoothing parameter for the three-root area.

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
    # Derivatives of coefficients w.r.t. A and B.
    dc_dAB = dc_from_AB(A, B)

    # Chainrule to obtain derivatives w.r.t. A and B.
    droots: np.ndarray = np.dot(
        calculate_root_derivatives(c[0], c[1], c[2], eps), dc_dAB
    )

    roots = calculate_roots(c[0], c[1], c[2], eps)
    assert roots[-1] > B, (
        "Expecting largest compressibility factor to be greater than covolume."
    )

    extension_case = is_extended_factor(A, B, gaslike, eps)

    # Shortcuts for quick switching between models.
    dWgsub = extended_factor_derivatives
    dWlsub = extended_factor_derivatives
    dWgsc = extended_factor_scg_derivatives
    dWlsc = extended_factor_scl_derivatives

    smooth_sc_idx: Literal[-1, 0, 1] = 1
    smooth_sc: float = 0.07

    match extension_case:
        case 0:
            if droots.shape[0] == 3 and smooth3 > 0.0:
                assert droots.shape[0] == 3, (
                    "Expecting shape (3,n) for derivatives of 3 roots."
                )
                assert roots.size == 3, "Expecting shape (3,) for roots for smoothing."
                _smooth_3root_region(roots, smooth3, gaslike, droots)
        case 1:
            assert droots.shape == (1, 2), (
                "Expecting shape (1, 2) of root derivatives in extension cases 1."
            )
            droots[0] = dWlsub(droots[0])
        case 2:
            assert droots.shape == (1, 2), (
                "Expecting shape (1, 2) of root derivatives in extension cases 2."
            )
            droots[-1] = dWgsub(droots[-1])
        case 10 | 11 | 12 | 13:
            droots[0] = dWlsc(droots[-1])
            smooth_sc_idx = 0
        case 20:
            pass
        case 21:
            assert droots.shape == (1, 2), (
                "Expecting shape (1, 2) of root derivatives in extension cases 21."
            )
            droots[-1] = dWgsc(droots[0])
            smooth_sc_idx = -1
        case _:
            raise NotImplementedError(
                f"Uncovered extension case {extension_case} for A,B = {(A, B)}."
            )

    if smooth_sc_idx in [-1, 0] and smooth_sc > 0.0:
        AB = np.array([A, B])
        AB_p = project_point_to_line(AB, _SC_BORDER_LINE, ABMETRIC)
        D = AB - AB_p
        d = np.sqrt(np.dot(D, ABMETRIC @ D))
        if d < smooth_sc and AB_p[1] >= B_CRIT:
            c = c_from_AB(AB_p[0], AB_p[1])
            dc_dAB = dc_from_AB(AB_p[0], AB_p[1])
            dZ = calculate_root_derivatives(c[0], c[1], c[2], eps)
            dZ = np.dot(dZ, dc_dAB)
            out = np.empty((2, 2))
            out[0] = droots[smooth_sc_idx]
            out[1] = dZ[-1]
            _smooth_supercritical_transition(0, d, smooth_sc, out)
            droots[smooth_sc_idx] = out[0]

        if smooth_sc_idx == -1:
            d = B - B_CRIT
            d = 0.0 if d < 0.0 else d
            if d < smooth_sc:
                c = c_from_AB(A, B_CRIT)
                dc_dAB = dc_from_AB(A, B_CRIT)
                dZ = calculate_root_derivatives(c[0], c[1], c[2], eps)
                dZ = np.dot(dZ, dc_dAB)
                if dZ.shape[0] > 1:
                    raise NotImplementedError(
                        "SC-smoothing has ambiguous target value."
                    )
                dW = dWgsub(dZ[0])
                out = np.empty((2, 2))
                out[0] = droots[-1]
                out[1] = dW
                _smooth_supercritical_transition(0, d, smooth_sc, out)
                droots[-1] = out[0]

        elif smooth_sc_idx == 0:
            AB_p = project_point_to_line(AB, SUPERCRITICAL_LINE, ABMETRIC)
            D = AB - AB_p
            d = np.sqrt(np.dot(D, ABMETRIC @ D))
            if d < smooth_sc and AB_p[1] <= B_CRIT:
                c = c_from_AB(AB_p[0], AB_p[1])
                c = c_from_AB(AB_p[0], AB_p[1])
                dZ = calculate_root_derivatives(c[0], c[1], c[2], eps)
                dZ = np.dot(dZ, dc_dAB)
                dW = dWlsub(dZ[-1])
                out = np.empty((2, 2))
                out[0] = droots[0]
                out[1] = dW
                _smooth_supercritical_transition(0, d, smooth_sc, out)
                droots[0] = out[0]

    if not gaslike and roots[0] <= B and extension_case >= 10:
        d = np.sqrt(B_original**2 + ABMETRIC[0, 0] * A**2)
        if B_limit_reached or d <= 5e-3:
            if B_original == 0.0:
                droots[0] = np.zeros(2)
            else:
                droots[0] = np.array([0.0, 1.1])

    if gaslike:
        return droots[-1]
    else:
        return droots[0]
