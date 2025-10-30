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

import numba as nb
import numpy as np

from .._core import NUMBA_CACHE, NUMBA_FAST_MATH
from .cubic_polynomial import (
    calculate_root_derivatives,
    calculate_roots,
    get_root_case,
    one_root,
)

_COMPILE_KWARGS = dict(fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
"""Keyword arguments for compiling functions in this module."""


_COMPILER = nb.njit
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

COVOLUME_LIMIT: float = 1e-5
""""Below this value, the covolume is considered zero.

Required to treay the limit case of B -> 0.

"""


@_COMPILER(nb.f8[:](nb.f8, nb.f8), **_COMPILE_KWARGS)
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


@_COMPILER(nb.f8[:, :](nb.f8, nb.f8), **_COMPILE_KWARGS)
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


@_COMPILER(nb.f8(nb.f8), **_COMPILE_KWARGS)
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


@_COMPILER(
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


@_COMPILER(
    [
        nb.void(nb.f8, nb.f8, nb.f8, nb.f8[:]),
        nb.void(nb.f8, nb.f8, nb.f8, nb.f8[:, :]),
    ],
    **_COMPILE_KWARGS,
)
def _smooth_scl_transition(B: float, W: float, T: float, out: np.ndarray) -> None:
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


@_COMPILER(nb.f8(nb.f8, nb.f8), **_COMPILE_KWARGS)
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


@_COMPILER(nb.f8[:](nb.f8[:]), **_COMPILE_KWARGS)
def dW_sub(dZ: np.ndarray) -> np.ndarray:
    """
    Parameters:
        d_Z: ``shape=(2,)``

            The derivatives of ``Z`` w.r.t. to cohesion and co-volume.

    Returns:
        The derivative of :func:`W_Sub` w.r.t. the cohesion and covolume.

    """
    return -0.5 * np.array([dZ[0], 1 + dZ[1]])


@_COMPILER(nb.i1(nb.f8, nb.f8, nb.bool, nb.f8), **_COMPILE_KWARGS)
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
        - 3: The indicated root is gaslike and in the super-critical area, where it is
          extended. The liquidlike root is real.
        - 4: The indicated root is liquidlike and in the super-critical area, where it
          is extended. The gaslike root is real.
        - 5: The indicated root is liquidlike and in the super-critical area with
          3-root-regime, where the smallest root violates the constraint by B from
          below. The gaslike root is real.
        - 6: The indicated root is liquidlike and in the super-critical area with
          2-root regime. This is a special case (like zero cohesion and covolume), or
          the border to case 5. In any case the smallest root (liquid) needs an
          extension procedure. The gaslike root is real.
        - 7: The indicated root is liquid-like and approaching the limit case of zero
          covolume where the root value goes to zero as well. Using the threshold of
          :data:`COVOLUME_LIMIT`, a thin stripe above A=0 is set as the zone where this
          root needs some additional work. The gaslike root can be real for ``A`` in
          (0,0.25]. This is performed in the sub-critical area.

    """
    c = c_from_AB(A, B)
    nroot = get_root_case(c[0], c[1], c[2], eps)
    is_sc = is_supercritical(A, B)

    # Default return value is that it is not extended.
    is_extended = 0

    if nroot == 1:
        Z = one_root(c[0], c[1], c[2])
        Wsub = W_sub(Z[0], B)
        # Subcritical area
        if gaslike and Z < Wsub:
            is_extended = 1
        elif not gaslike and Wsub < Z:
            is_extended = 2
        # If super-critical and we are in the extension case, shift the code by two.
        if is_sc and is_extended > 0:
            is_extended += 2

    # Supercritical area with where non-physical 3-root regimes can appear.
    # The smallest root (liquid-like) needs a correction as it can take values below
    # the B. That would introduce all kind of numerical instability.
    elif nroot == 3:
        if is_sc and not gaslike:
            is_extended = 5
    # Borderline to the 3-root regime in the super-critical area, as well as special
    # point 0,0, which is treated as super-critical since it's on the line.
    # The liquid-like root needs an extension as it violates the lower bound b.
    elif nroot == 2:
        if is_sc and not gaslike:
            is_extended = 6

    # Special area: When B is zero, the smallest real root is also zero.
    # We bind it away from zero with some theshold
    if B <= COVOLUME_LIMIT and not gaslike and not is_sc:
        is_extended = 7

    return is_extended


@_COMPILER(
    nb.f8(nb.f8, nb.f8, nb.bool, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
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
    B = 0.0 if B < 0.0 else B
    c = c_from_AB(A, B)

    # NOTE: For the 3-root case, we can safe some computations by computing only root
    # we need. Not sure how much it saves.
    # NOTE: c contains the coefficients as the polynomial is read from left to right:
    # C[0] contains c_2, c[2] contains c_0
    roots = calculate_roots(c[0], c[1], c[2], eps)

    # Limit-case zero covolume suffers from various issues, including numerical ones
    # where the root can be -1e-17, i.e. smaller than 0. This is especially true near
    # the liquid-saturated curve, where there is only 1 real root, the liquid one.
    if B > COVOLUME_LIMIT:
        assert roots[-1] >= B, "Expecting largest root >= B."
    elif roots[-1] <= B:
        assert roots.size == 1, "Expecting 1 root case if lower bound B is violated."
        assert np.abs(roots[-1]) < 1e-7, (
            "Expecting near zero root if lower bound B is violated."
        )

    extension_case = is_extended_root(A, B, gaslike, eps)

    # Extended super-critical liquid-like root must not fall below this value.
    # Add some small value to actually never let it go to zero for various limit cases.
    B_thresh = 1.1 * B + 1e-8
    # Extended super-critical liquid-like root is smoothed once it falls below this
    # value.
    B_thresh_smoothing = 2.0 * B + 2e-8

    # Extended roots in the area B > Bcrit and A > Acrit are smoothed once they reach
    # the stripe (1 +- threshold) * Z. They are smoothed towards Z, mainly to counter
    # the exploding derivatives. Derivatives of W and Z have derivatives of opposite
    # end we counter the divergence with this.
    Zsc_thresh_factor = 0.2

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
            assert roots.size == 1, "Expecting only 1 real root in extension cases 1-2."
            roots[0] = W_sub(roots[0], B)
        # We use the same procedure in the supercritical area where gas is extended
        # since Wsub is greater there than the real root, hence the extension is fine.
        # But we introduce some additional smoothing when the extended root approaches
        # the line where Wsub = Z. We do this mainly for the sake of derivatives.
        # NOTE The super-critical extension is rudimentary. Especially the border
        # demarking transion from gas-like to liquid-like requires work. Technically it
        # does not exist, but there must be one in the unified case.
        case 3:
            assert roots.size == 1, "Expecting only 1 real root in extension cases 3."
            W = W_sub(roots[0], B)
            Z = roots[0]
            assert W >= Z, "Expecting W >= Z in extension case 3."

            Z_thresh = (1 + Zsc_thresh_factor) * Z
            # If it approaches the real root, smooth the values.
            if W <= Z_thresh:
                out = np.array([W, Z])
                _smooth_scl_transition(Z, W, Z_thresh, out)
                W = out[0]

            roots[0] = W
        # Liquid-like root extension in the super-critical 1-root area
        # The idea is to use Wsub as usual, but correct it with B multiplied with some
        # threshold >1 where it becomes too small. The derivatives must go to zero in
        # that case.
        case 4:
            assert roots.size == 1, "Expecting only 1 real root in extension cases 4."
            Z = roots[-1]
            W = W_sub(Z, B)
            assert W <= Z, "Expecting W <= Z in extension case 4."

            Z_thresh = (1 - Zsc_thresh_factor) * Z

            # If it is smaller than the threshold, we just take the threshold
            if W <= B_thresh:
                W = B_thresh
            # If it is bigger, we smooth the transition for numerical purposes.
            elif B_thresh < W <= B_thresh_smoothing:
                out = np.array([W, B_thresh])
                _smooth_scl_transition(B_thresh, W, B_thresh_smoothing, out)
                W = out[0]
            # If it approaches the transition to becoming the real root, smooth it.
            elif W >= Z_thresh:
                out = np.array([W, Z])
                _smooth_scl_transition(-Z, -W, -Z_thresh, out)
                W = out[0]

            roots[0] = W
        # There are super-critical 2 and 3-root regions which are nonphysical, and the
        # smallest root (liquid) is smaller than the physically admissible value B.
        # With the smoothing in case 3 in mind, we just set the liquid root to the
        # threshold value.
        # Borderline to 3-root area, as well as point 0,0 are 2-root areas. We treat it
        # the same as case 5, and without smoothing as in case 4. This is a limit case
        # in multiple senes.
        case 5 | 6:
            assert roots.size in [2, 3], (
                "Expecting 2 or 3 real root in extension cases 5-6."
            )
            if roots[0] <= B_thresh:
                roots[0] = B_thresh
            # The 2-root area on the border to the super-critical 3-root area can have
            # root values which do not violate the constrain, very close to the point
            # (A, B) = (0, 0).
            elif roots.size == 3:
                raise NotImplementedError(
                    "Expecting smallest root <= B in extension case 5."
                )
        # Special area where the covolume approaches zero. The liquid-like root goes to
        # zero as well and needs attention. We cap it with the limit value, and use the
        # derivatives at the point (A, COVOLUME_LIMIT)
        case 7:
            c7 = c_from_AB(A, COVOLUME_LIMIT)
            # Take absolute value because close to A=0 we enter the zone where it might
            # become negative
            Zl_lim = np.abs(calculate_roots(c7[0], c7[1], c7[2], eps)[0])
            assert Zl_lim > COVOLUME_LIMIT, (
                "Limitcase liquid root expected to be greater than threshold in "
                "extension case 7."
            )
            # This never happened because Zl goes very fast to zero. But we need to make
            # sure that the extension procedure here is not the cause of other problems.
            if roots.size > 1 and Zl_lim > roots[-1]:
                raise NotImplementedError(
                    "Uncovered root order violation in extension case 7."
                )
            roots[0] = Zl_lim
        case _:
            # Should never happen.
            raise NotImplementedError(f"Uncovered extension case encountered.")

    # Sanity check: This order must always hold, otherwise we are in an uncovered case.
    if roots[0] > roots[-1]:
        raise NotImplementedError(
            f"Encountered an A-B pair violating Zl <= Zg:\n\tA = {A}\n\tB = {B}\n"
        )
    if not gaslike and roots[0] <= B:
        raise NotImplementedError(
            f"Encountered A-B pair violating B < Zl:\n\tA = {A}\n\tB = {B}\n"
        )

    # Since ordered by size, gaslike is largest root and liquidlike is smallest.
    if gaslike:
        return roots[-1]
    else:
        return roots[0]


@_COMPILER(
    nb.f8[:](nb.f8, nb.f8, nb.bool, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
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

    # NOTE for more information on individual steps, see Z(), as the computations are
    # analogous.

    A = 0.0 if A < 0.0 else A
    B = 0.0 if B < 0.0 else B
    c = c_from_AB(A, B)
    # Derivatives of coefficients w.r.t. A and B.
    dc_dAB = dc_from_AB(A, B)

    # Chainrule to obtain derivatives w.r.t. A and B.
    roots = calculate_roots(c[0], c[1], c[2], eps)
    if B > 0.0:
        assert roots[-1] >= B, "Expecting largest root >= B."
    # droots = calculate_root_derivatives(c[0], c[1], c[0], eps) @ dc_dAB
    droots: np.ndarray = np.dot(
        calculate_root_derivatives(c[0], c[1], c[2], eps), dc_dAB
    )

    extension_case = is_extended_root(A, B, gaslike, eps)

    B_thresh = 1.1 * B
    B_thresh_smoothing = 2.0 * B
    Zsc_thresh_factor = 0.2

    # NOTE At and below the threshold hold it wolds W = a*B, so we make a linear
    # extension dW = (0, a). The Taylor expansion might lose some order
    # (and consequently Newton-based flash algorithms), but only in this rather weird
    # super-critical region where the EoS has limited validity.
    dW_thresh = np.array([0.0, -1.1])

    match extension_case:
        case 0:
            if roots.size == 3 and smooth3 > 0.0:
                _smooth_3root_region(roots, smooth3, gaslike, droots)
        case 1 | 2:
            assert roots.size == 1, "Expecting only 1 real root in extension cases 1-2."
            assert droots.shape == (1, 2), (
                "Expecting shape (1, 2) of root derivatives in extension cases 1-2."
            )
            droots[0] = dW_sub(droots[0])
        case 3:
            assert roots.size == 1, "Expecting only 1 real root in extension case 3."
            assert droots.shape == (1, 2), (
                "Expecting shape (1, 2) of root derivatives in extension case 3."
            )
            Z = roots[0]
            W = W_sub(Z, B)
            dZ = droots[0]
            dW = dW_sub(dZ)

            Z_thresh = (1 + Zsc_thresh_factor) * Z
            # If it approaches the real root, smooth the values.
            if W <= Z_thresh:
                # NOTE: numba has issues with array([vec, vec]), list of arrays
                # in constructor.
                out = np.zeros((2, 2))
                out[0] = dW
                out[1] = dZ
                _smooth_scl_transition(Z, W, Z_thresh, out)
                dW = out[0]

            droots[0] = dW
        case 4:
            assert roots.size == 1, "Expecting only 1 real root in extension cases 4."
            assert droots.shape == (1, 2), (
                "Expecting shape (1, 2) of root derivatives in extension case 4."
            )
            Z = roots[-1]
            W = W_sub(Z, B)
            dZ = droots[-1]
            dW = dW_sub(dZ)

            Z_thresh = (1 - Zsc_thresh_factor) * Z

            if W <= B_thresh:
                dW = dW_thresh
            elif B_thresh < W <= B_thresh_smoothing:
                out = np.zeros((2, 2))
                out[0] = dW
                out[1] = dW_thresh
                _smooth_scl_transition(B_thresh, W, B_thresh_smoothing, out)
                dW = out[0]
            elif W >= Z_thresh:
                out = np.zeros((2, 2))
                out[0] = dW
                out[1] = dZ
                _smooth_scl_transition(-Z, -W, -Z_thresh, out)
                dW = out[0]

            droots[0] = dW
        case 5 | 6:
            assert roots.size in [2, 3], (
                "Expecting 3 or 2 real root in extension cases 5-6."
            )
            assert droots.shape in [(2, 2), (3, 2)], (
                "Expecting shape (3, 2) or (2, 2) of root derivatives in extension "
                "cases 5-6."
            )
            if roots[0] <= B_thresh:
                droots[0] = dW_thresh
            elif roots.size == 3:
                raise NotImplementedError(
                    "Expecting smallest root <= B in extension case 5."
                )
        case 7:
            c7 = c_from_AB(A, COVOLUME_LIMIT)
            dc_dAB7 = dc_from_AB(A, COVOLUME_LIMIT)
            droots7 = np.dot(
                calculate_root_derivatives(c7[0], c7[1], c7[2], eps), dc_dAB7
            )
            # Don't need much assertions here, this is just a less exact derivative.
            droots[0] = droots7[0]
        case _:
            raise NotImplementedError(f"Uncovered extension case encountered.")

    if gaslike:
        return droots[-1]
    else:
        return droots[0]
