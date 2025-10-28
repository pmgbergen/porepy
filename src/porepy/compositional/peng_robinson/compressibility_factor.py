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

import numpy as np
import numba as nb

from .._core import NUMBA_FAST_MATH, NUMBA_CACHE
from .cubic_polynomial import (
    calculate_roots,
    calculate_root_derivatives,
    get_root_case,
    one_root,
)


_COMPILE_KWARGS = dict(fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
"""Keyword arguments for compiling functions in this module."""


_COMPILE_DECORATOR = nb.njit
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


@_COMPILE_DECORATOR(nb.f8[:](nb.f8, nb.f8), **_COMPILE_KWARGS)
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


@_COMPILE_DECORATOR(nb.f8[:, :](nb.f8, nb.f8), **_COMPILE_KWARGS)
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


@_COMPILE_DECORATOR(nb.f8(nb.f8), **_COMPILE_KWARGS)
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


@_COMPILE_DECORATOR(
    nb.bool(nb.f8, nb.f8),
    **_COMPILE_KWARGS,
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


@_COMPILE_DECORATOR(
    [
        nb.void(nb.f8[:], nb.f8, nb.bool, nb.f8[:]),
        nb.void(nb.f8[:], nb.f8, nb.bool, nb.f8[:, :]),
    ],
    **_COMPILE_KWARGS,
)
def _smooth_3root_region(
    z: np.ndarray, s: float, gaslike: bool, out: np.ndarray
) -> None:
    """Smoothing of roots in the physical 3-root region, close to the border where the
    intermediate root approaches either the biggest or the smallest root.

    Parameters:
        z: 1D array with shape ``(3,)`` containing the 3 real roots ordered by size.
        s: Small number saying how close the intermediate root has to be to start
            smoothing.
        gaslike: Either smoothing last/largest root (True) or the first/smallest root
            (False).
        out: Array to be smoothed. Can be the roots (i.e., equal to z), or their
            derivatives. Must be of shape ``(3,m)``.

    """
    assert z.shape == (3,), "Roots must be (3,)-array."
    assert out.shape[0] == 3, "Require at least 3 values to be smoothed."

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


@_COMPILE_DECORATOR(
    [
        nb.void(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8[:]),
        nb.void(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8[:, :]),
    ],
    **_COMPILE_KWARGS,
)
def _smooth_scl_transition(B: float, W: float, T: float, s: float, out: float) -> None:
    """Smoothing the super-critical liquid-like root when it comes close to a target
    value.

    The distance ``d`` between ``W`` and ``T`` is normalized using the distance between
    ``B`` and ``T`` as a reference.
    With ``s`` denoting a fraction of the reference distance, a convex smoothing is
    applied as ``W`` approaches ``T`` using a a weight ``w``. If ``d < s`` it holds

    .. math::

        w = 1 - (\\frac{d}{s})^2 (3 - 2 \\frac{d}{s})
        out[0] = out[0](1 - w) + out[1] w

    ``out`` is manipulated directly.

    Parameters:
        B: Reference value for distance.
        W: Value to be smoothed.
        T: Top or target value towards which it is smoothed.
        s: Fraction of range on which smoothing is applied
        out: An output array of shape ``(2,n)``, containing the values to be smoothed at
            index 0 and the values towards which it is smoothed at index 1.

    """
    assert 0 < s < 1, "Smoothing factor needs to be in (0,1)."
    d = np.abs((W - B)) / np.abs((T - B))

    if d >= s:
        w = 0.0
    else:
        w = 1.0 - (d / s) ** 2 * (3 - 2 * d / s)

    out[0] = out[0] * (1 - w) + out[1] * w


@_COMPILE_DECORATOR(nb.f8(nb.f8, nb.f8), **_COMPILE_KWARGS)
def W_sub(Z: float, B: float) -> float:
    """Extended compressibility factor in the sub-critical area (Ben Gharbia 2021).

    Note:
        The sum of the roots of any cubic normalized cubic polynomial is equal to
        :math:`-c_2`.

    Parameters:
        Z: The 1 real root.
        B: Dimensionless co-volume.

    Returns:
        :math:`\\frac{1 - B - Z}{2}`

    """
    return (1 - B - Z) * 0.5


@_COMPILE_DECORATOR(nb.f8[:](nb.f8[:]), **_COMPILE_KWARGS)
def dW_sub(dZ: np.ndarray) -> np.ndarray:
    """
    Parameters:
        d_Z: ``shape=(2,)``

            The derivatives of ``Z`` w.r.t. to cohesion and co-volume.

    Returns:
        The derivative of :func:`W_Sub` w.r.t. the cohesion and covolume.

    """
    return -0.5 * np.array([dZ[0], 1 + dZ[1]])


@_COMPILE_DECORATOR(nb.f8(nb.f8, nb.f8), **_COMPILE_KWARGS)
def W_scl(Zg: float, B: float) -> float:
    """Extended liquidlike compressibility factor in the super-critical region, where
    the real root is used for the gaslike phase.

    Parameters:
        Zg: Existing, gas-like compressibility factor. This is a real root of the
            polynomial.
        B: Dimensionless co-volume.

    Returns:
        :math:`B + \\frac{Z - B}{2}`

    """
    # return B + (Zg - B) * 0.5
    return (Zg + B) * 0.5


@_COMPILE_DECORATOR(nb.f8[:](nb.f8[:]), **_COMPILE_KWARGS)
def dW_scl(dZg: np.ndarray) -> np.ndarray:
    """
    Parameters:
        d_Z: ``shape=(2,)``

            The derivatives of ``Z`` w.r.t. to cohesion and co-volume.

    Returns:
        The derivative of :func:`W_scl` w.r.t. the cohesion and covolume.

    """
    return np.array([dZg[0], dZg[1] + 1]) * 0.5


# @_COMPILE_DECORATOR(nb.f8(nb.f8, nb.f8), **_COMPILE_KWARGS)
# def W_scg(Zl: float, B: float) -> float:
#     """Extended gas-like compressibility factor in the super-critical region, where
#     the liquid-like phase is flagged as present.

#     Parameters:
#         Z: Existing, liquid-like compressibility factor.
#         B: Dimensionless co-volume.

#     Returns:
#         :math:`B + \\frac{1 - B - Z}{2}`, i.r. the regular sub-critical extension plus
#         an extra :math:`B`.

#     """
#     return B + W_sub(Zl, B)


# @_COMPILE_DECORATOR(nb.f8[:](nb.f8[:]), **_COMPILE_KWARGS)
# def dW_scg(dZl: np.ndarray) -> np.ndarray:
#     """
#     Parameters:
#         dZ: ``shape=(2,)``

#             The derivatives of ``Z`` w.r.t. to cohesion and co-volume.

#     Returns:
#         The derivative of :func:`W_scg` w.r.t. the cohesion and covolume.

#     """
#     return np.array([0., 1.]) + dW_sub(dZl)


@_COMPILE_DECORATOR(nb.i1(nb.f8, nb.f8, nb.bool, nb.f8), **_COMPILE_KWARGS)
def is_extended_root(A: float, B: float, gaslike: bool, eps: float) -> int:
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
        One of the following codes is returned:

        - 0: The indicated root (``gaslike`` or not) is not obtained through an
          extension procedure and is real. It is an actual root of the polynomial.
        - 1: The indicated root is gaslike and in the subcritical area, where it is
          extended. The liquidlike root is real.
        - 2: The indicated root is liquidlike and in the subcritical area, where it is
          extended. The gaslike root is real.
        - 3: The indicated root is liquidlike and in the super-critical area, where it
          is extended. The gaslike root is real.
        - 4: The indicated root is liquidlike and in the super-critical area with
          3-root-regime, where the smallest root violates the constraint by B from
          below. The gaslike root is real.

    """
    c = c_from_AB(A, B)
    nroot = get_root_case(c[0], c[1], c[2], eps)
    is_sc = is_supercritical(A, B)

    # Default return value is that it is not extended.
    is_extended = 0

    if nroot == 1:
        Z = one_root(c[0], c[1], c[2])
        W = W_sub(Z[0], B)
        # Subcritical area
        if not is_sc:
            if gaslike and Z < W:
                is_extended = 1
            elif not gaslike and W < Z:
                is_extended = 2
        # Supercritical area
        else:
            if not gaslike and W <= Z:
                is_extended = 3
    # Supercritical area with where non-physical 3-root regimes can appear.
    # The smallest root (liquid-like) needs a correction as it can take values below
    # the B. That would introduce all kind of numerical instability.
    elif nroot == 3:
        if is_sc and not gaslike:
            is_extended = 4

    return is_extended


# @_COMPILE_DECORATOR(
#     nb.f8(nb.f8, nb.f8, nb.bool, nb.f8, nb.f8, nb.f8),
#     **_COMPILE_KWARGS,
# )
def Z(
    A: float,
    B: float,
    gaslike: bool,
    eps: float,
    smoothE: float,
    smooth3: float,
) -> float:
    """Compute the compressibility factor for given :math:`A` and :math:`B`.

    Depending on the value of ``gaslike``, the largest or smallest root is returned
    in the case of three real roots.

    In the case of a single real root, that root is returned.

    An extension procedure is applied in the one-root area if ``eps > 0``.

    Parameters:
        A: Dimensionless cohesion parameter.
        B: Dimensionless covolume parameter.
        gaslike: Flag indicating whether to return the gas-like (True) or liquid-like
            (False) root.
        eps: Tolerance for detection of degeneracy/two-root and triple root case.
        smoothE: Smoothing parameter for the extension procedure.
        smooth3: Smoothing parameter for the three-root area.

    Returns:
        The compressibility factor.

    """
    c = c_from_AB(A, B)

    # NOTE: For the 3-root case, we can safe some computations by computing only root
    # we need. Not sure how much it saves.
    # NOTE: c contains the coefficients as the polynomial is read from left to right:
    # C[0] contains c_2, c[2] contains c_0
    roots = calculate_roots(c[0], c[1], c[2], eps)

    extension_case = is_extended_root(A, B, gaslike, eps)

    match extension_case:
        # No root is extended.
        case 0:
            # In the sub-critical regime, there is a physical 3-root region.
            # If requested, we smooth the roots close to the phase borders where 1 phase
            # disapears.
            if roots.size == 3 and smooth3 > 0.0:
                _smooth_3root_region(roots, smooth3, gaslike, roots)
        # Root extension in the subcritical 1-root area. Note that in both cases, the
        # real root calculations contains only 1 value.
        case 1 | 2:
            assert roots.size == 1, "Expecting only 1 real root in extension cases 1,2."
            roots[0] = W_sub(roots[0], B)
        # Liquid-like root extension in the super-critical 1-root area
        case 3 | 4:
            assert roots.size in [1, 3], (
                "Expecting only 1 or 3 real roots in extension case 3, 4."
            )
            Z = roots[-1]
            W = W_scl(Z, B)

            # Extended liquid-root smoothing in the super-critical area.
            if smoothE > 0.0:
                assert smoothE <= 0.5, "Smoothing factor must be smaller than 0.5"
                # Smoothing the transition from liquid-extended-supercrit to
                # liquid-real-supercrit
                Wsub = W_sub(Z, B)
                if B > B_CRIT:
                    out = np.array([W, Z])
                    # We use the distance of W_sub to Z as a measure to the line W_sub=Z
                    _smooth_scl_transition(B, Wsub, Z, smoothE, out)
                    W = out[0]
                # Smoothing the transition from liquid-extended-supercrit to
                # liquid-extended-sub-crit
                else:
                    out = np.array([W, Wsub])
                    # We use the distance of B to the critical line, relative to B_C
                    _smooth_scl_transition(B_CRIT, B, critical_line(A), smoothE, out)
                    W = out[0]

            roots[0] = W
        case _:
            # Should never happen.
            raise NotImplementedError(f"Uncovered extension case encountered.")

    # Since ordered by size, gaslike is largest root and liquidlike is smallest.
    if gaslike:
        return roots[-1]
    else:
        return roots[0]


# @_COMPILE_DECORATOR(
#     nb.f8[:](nb.f8, nb.f8, nb.bool, nb.f8, nb.f8, nb.f8),
#     **_COMPILE_KWARGS,
# )
def dZ_dAB(
    A: float,
    B: float,
    gaslike: bool,
    eps: float,
    smoothE: float,
    smooth3: float,
) -> float:
    """Compute the derivatives of the compressibility factor with respect to :math:`A`
    and :math:`B`.

    For more information, see :func:`Z`.

    Parameters:
        A: Dimensionless cohesion parameter.
        B: Dimensionless covolume parameter.
        gaslike: Flag indicating whether to return the gas-like (True) or liquid-like
            (False) root.
        eps: Tolerance for detection of degeneracy/two-root and triple root case.
        smoothE: Smoothing parameter for the extension procedure.
        smooth3: Smoothing parameter for the three-root area.

    Returns:
        A ``(2,)``-array containing the derivatives w.r.t. cohesion and covolume.

    """

    c = c_from_AB(A, B)

    # Derivatives of coefficients w.r.t. A and B.
    dc_dAB = dc_from_AB(A, B)

    # Chainrule to obtain derivatives w.r.t. A and B.
    droots = calculate_root_derivatives(c[0], c[1], c[0], eps) @ dc_dAB

    # Since ordered by size, gaslike is largest root and liquidlike is smallest.
    if gaslike:
        return droots[-1]
    else:
        return droots[0]
