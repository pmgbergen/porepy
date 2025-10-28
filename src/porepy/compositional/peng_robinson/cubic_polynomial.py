"""Functionality for calculating real solutions or real cubic polynomials, efficiently
and in large quantities.

This module utilizes numba to compile the core routines, which can than be wrapped in
actual applications like the Peng-Robinson equation of state.

The base formulation of the cubic polynomial is

.. math::
    z^3 + c_2 z^2 + c_1 z + c_0 = 0,

from which the reduced form is obtained as

.. math::
    z^3 + r_1 z + r_0 = 0.

The root is a function of the coefficients :math:`c_0`, :math:`c_1`, and :math:`c_2`.
Most importantly, this module implements also the derivatives of the roots with respect
to the coefficients, which are essential in many applications.

Note:
    The implementation here is valid for real coefficients only.

See also:

    - https://en.wikipedia.org/wiki/Cubic_equation
    - https://de.wikipedia.org/wiki/Kubische_Gleichung

"""

from __future__ import annotations

import numba as nb
import numpy as np

from .._core import NUMBA_FAST_MATH, NUMBA_CACHE


_COMPILE_KWARGS = dict(fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
"""Keyword arguments for compiling functions in this module."""


_COMPILE_DECORATOR = nb.njit
"""Decorator for compiling functions in this module.

Alternative compilers are the :obj:`numba.cfunc` call-back decorator, or future AOT
compilation.

"""


@_COMPILE_DECORATOR(
    nb.f8(nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def get_r1(c2: float, c1: float) -> float:
    """Calculate the reduced coefficient r1.

    .. math::

        r_1 = c_1 - \\frac{c_2^2}{3}

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.
        c1: Coefficient of the linear term in the cubic polynomial.

    Returns:
        The reduced coefficient r1.

    """
    return c1 - c2**2 / 3.0


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8),
    **_COMPILE_KWARGS,
)
def get_dr1(c2: float) -> np.ndarray:
    """Derivatives of the reduced coefficient r1 with respect to c2, c1 and c0.

    The derivative with respect to c1 is always 1.
    The derivative with respect to c0 is always 0.

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.

    Returns:
        A numpy array with ``shape=(3,)`` containing the derivatives with respect to
        ``(c2, c1, c0)``.

    """
    return np.array([-2.0 * c2 / 3.0, 1.0, 0.0])


@_COMPILE_DECORATOR(
    nb.f8(nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def get_r0(c2: float, c1: float, c0: float) -> float:
    """Calculate the reduced coefficient r0.

    .. math::

        r_0 = \\frac{2}{27} c_2^3 - \\frac{c_1 c_2}{3} + c_0

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.
        c1: Coefficient of the linear term in the cubic polynomial.
        c0: Coefficient of the constant term in the cubic polynomial.

    Returns:
        The reduced coefficient r0.

    """
    return 2.0 / 27.0 * c2**3 - (c1 * c2) / 3.0 + c0


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def get_dr0(c2: float, c1: float) -> np.ndarray:
    """Derivatives of the reduced coefficient r0 with respect to c2, c1 and c0.

    The derivative with respect to c0 is always 1.

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.
        c1: Coefficient of the linear term in the cubic polynomial.

    Returns:
        A numpy array with ``shape=(3,)`` containing the derivatives with respect to
        ``(c2, c1, c0)``.

    """
    return np.array([6.0 / 27.0 * c2**2 - c1 / 3.0, -c2 / 3.0, 1.0])


@_COMPILE_DECORATOR(
    nb.f8(nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def discriminant(r1: float, r0: float) -> float:
    """Calculate the discriminant of the reduced cubic polynomial.

    Note:
        For normalized cubic polynomials where the coefficient of the monomial
        :math:`z^3` is 1, the discriminant of the reduced polynomial is equal to
        the discriminant of the original polynomial.

    Parameters:
        r1: Reduced coefficient r1.
        r0: Reduced coefficient r0.

    Returns:
        The discriminant of the reduced cubic polynomial.

    """
    return -(4 * r1**3 + 27 * r0**2)


@_COMPILE_DECORATOR(
    nb.i4(nb.f8, nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def get_root_case(c2: float, c1: float, c0: float, eps: float) -> int:
    """Determine the case for the roots of the cubic polynomial.

    The cases are:

        - 3: Three distinct real roots.
        - 2: One real root and one root with multiplicity two.
        - 1: One real root and two complex conjugate roots.
        - 0: One real root with multiplicity three (triple root).

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.
        c1: Coefficient of the linear term in the cubic polynomial.
        c0: Coefficient of the constant term in the cubic polynomial.
        eps: Tolerance for determining whether the discriminant is zero.

    Returns:
        An integer indicating the case (0, 1, 2 or 3).

    """
    r1 = get_r1(c2, c1)
    r0 = get_r0(c2, c1, c0)

    D = discriminant(r1, r0)

    # Negative D => one real root, two complex conjugate roots.
    if D < -eps:
        return 1
    # Positve D => three distinct real roots.
    elif D > eps:
        return 3
    # D == 0 => multiple roots.
    else:
        # r_1 == 0 => triple root.
        if np.abs(r1) < eps:
            return 0
        # Else two roots, one with multiplicity two.
        else:
            return 2


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8),
    **_COMPILE_KWARGS,
)
def triple_root(c2: float) -> np.ndarray:
    """Calculate the triple root of the cubic polynomial.

    See also:
        https://en.wikipedia.org/wiki/Cubic_equation#Multiple_root

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.

    Returns:
        The triple root.

    """
    return np.array([-c2 / 3.0])


@_COMPILE_DECORATOR(
    nb.f8[:, :](nb.f8),
    **_COMPILE_KWARGS,
)
def d_triple_root(c2: float) -> np.ndarray:
    """Derivatives of the triple root with respect to c2, c1 and c0.
    This is a constant array with values (-1/3, 0, 0).

    Note:
        Though always a constant array, we keep the method format for simplicity of code
        using the root computations.
        Otherwise we would need a copy operation on a constant array, which is
        cumbersome.

    """
    return np.array([[-1.0 / 3.0, 0.0, 0.0]])


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def two_roots(c2: float, c1: float, c0: float) -> np.ndarray:
    """Compute the two roots of the cubic polynomial, in the case where one of them has
    multiplicity two.

    See also:
        https://en.wikipedia.org/wiki/Cubic_equation#Multiple_root

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.
        c1: Coefficient of the linear term in the cubic polynomial.
        c0: Coefficient of the constant term in the cubic polynomial.

    Returns:
        A numpy array with the two roots. The second entry contains the larger root.

    """

    r1 = get_r1(c2, c1)
    r0 = get_r0(c2, c1, c0)

    u = 3.0 * r0 / r1

    z1 = u
    z2 = -u / 2.0

    if z1 < z2:
        return np.array([z1, z2]) - c2 / 3.0
    else:
        return np.array([z2, z1]) - c2 / 3.0


@_COMPILE_DECORATOR(
    nb.f8[:, :](nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def d_two_roots(c2: float, c1: float, c0: float) -> np.ndarray:
    """Derivatives of the two roots with respect to c2, c1 and c0, in the case where one
    of them has multiplicity two.

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.
        c1: Coefficient of the linear term in the cubic polynomial.
        c0: Coefficient of the constant term in the cubic polynomial.

    Returns:
        A numpy array with shape (2, 3), where the first axis corresponds to the two
        roots, and the second axis to the derivatives with respect to (c2, c1, c0).
        The second row belongs to the larger root.

    """

    r1 = get_r1(c2, c1)
    r0 = get_r0(c2, c1, c0)

    dr1 = get_dr1(c2)
    dr0 = get_dr0(c2, c1)

    u = 3.0 * r0 / r1

    du = -3.0 * r0 / r1**2 * dr1 + 3.0 / r1 * dr0

    dc2 = np.array([-1.0 / 3.0, 0.0, 0.0])

    dz1_dc = du + dc2
    dz2_dc = -du / 2.0 + dc2

    if u < -u / 2.0:
        return np.vstack((dz1_dc, dz2_dc))
    else:
        return np.vstack((dz2_dc, dz1_dc))


@_COMPILE_DECORATOR(
    nb.f8(nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def _get_Gamma(r1: float, r0: float) -> float:
    """Calculate the auxiliary variable gamma, used in the trigonometric solution of
    real roots of real cubic polynomials.

    Parameters:
        r1: Reduced coefficient r1.
        r0: Reduced coefficient r0.

    Returns:
        The auxiliary variable gamma.

    """
    return -r0 / 2.0 * np.sqrt(27.0 / np.abs(r1**3))


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def _get_dGamma(r1: float, r0: float) -> float:
    """Derivatives of the auxiliary variable :math:`\\gamma` w.r.t. reduced
    coefficients.

    Parameters:
        r1: Reduced coefficient r1.
        r0: Reduced coefficient r0.

    Returns:
        A numpy array with ``shape=(2,)`` containing the derivatives with respect to
        ``(r1, r0)``.

    """
    t = np.sqrt(27.0 / np.abs(r1**3))
    return np.array(
        [
            r0 / 4.0 / t * 27.0 / np.abs(r1**3) ** 2 * np.sign(r1) * 3.0 * r1**2,
            -t / 2.0,
        ]
    )


@_COMPILE_DECORATOR(
    nb.f8(nb.f8),
    **_COMPILE_KWARGS,
)
def _get_t1(r1: float) -> float:
    """Calculate the auxiliary variable t1, used in the trigonometric solution of
    real roots of real cubic polynomials.

    Parameters:
        r1: Reduced coefficient r1.

    Returns:
        The auxiliary variable t1.

    """
    assert r1 > 0.0, "Argument for auxiliary variable t1 must be positive."
    return 2.0 * np.sqrt(r1 / 3.0)


@_COMPILE_DECORATOR(
    nb.f8(nb.f8),
    **_COMPILE_KWARGS,
)
def _get_dt1(r1: float) -> float:
    """Derivatives of the auxiliary variable t1 w.r.t. reduced coefficient r1.

    Parameters:
        r1: Reduced coefficient r1.

    Returns:
        The derivative with respect to r1.

    """
    assert r1 > 0.0, "Argument for auxiliary variable t1 must be positive."
    return np.sqrt(1.0 / (3.0 * r1))


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def one_root(c2: float, c1: float, c0: float) -> np.ndarray:
    """Calculate the single (real) root of the cubic polynomial, where applicable.

    See also:
        https://en.wikipedia.org/wiki/
        Cubic_equation#Trigonometric_and_hyperbolic_solutions

        https://de.wikipedia.org/wiki/
        Kubische_Gleichung#Die_F%C3%A4lle_mit_p_%E2%89%A0_0

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.
        c1: Coefficient of the linear term in the cubic polynomial.
        c0: Coefficient of the constant term in the cubic polynomial.

    Returns:
        The single (real) root.

    """
    r1 = get_r1(c2, c1)
    r0 = get_r0(c2, c1, c0)

    g = _get_Gamma(r1, r0)

    if r1 < 0.0:
        absg = np.abs(g)
        t1 = -_get_t1(-r1)

        # Special case for numerical stability.
        if 1.0 - 1e-14 < absg < 1.0 + 1e-14:
            t1 *= -1.0
            t2 = 1.0
        else:
            t2 = np.sign(r0) * np.cosh(np.arccosh(absg) / 3.0)

    elif r1 > 0.0:
        t1 = _get_t1(r1)
        t2 = np.sinh(np.arcsinh(g) / 3.0)
    else:
        raise ValueError("r1 cannot be zero for one real root.")

    return np.array([t1 * t2]) - c2 / 3.0


@_COMPILE_DECORATOR(
    nb.f8[:, :](nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def d_one_root(c2: float, c1: float, c0: float) -> np.ndarray:
    """Derivatives of the single (real) root with respect to c2, c1 and c0.

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.
        c1: Coefficient of the linear term in the cubic polynomial.
        c0: Coefficient of the constant term in the cubic polynomial.

    Returns:
        A numpy array with ``shape=(3,)`` containing the derivatives with respect to
        ``(c2, c1, c0)``.

    """
    r1 = get_r1(c2, c1)
    r0 = get_r0(c2, c1, c0)

    dr1 = get_dr1(c2)
    dr0 = get_dr0(c2, c1)

    g = _get_Gamma(r1, r0)
    dg = _get_dGamma(r1, r0)
    dg = dg[0] * dr1 + dg[1] * dr0

    if r1 < 0.0:
        absg = np.abs(g)

        t1 = -_get_t1(-r1)
        dt1 = _get_dt1(-r1) * dr1

        dt2 = np.zeros(3)

        # Special case for numerical stability.
        if 1.0 - 1e-14 < absg < 1.0 + 1e-14:
            t1 *= -1.0
            dt1 *= -1.0
            t2 = 1.0
        else:
            t = np.cosh(np.arccosh(absg) / 3.0)
            t2 = np.sign(r0) * t
            dt2 = (
                np.sign(r0)
                * np.sinh(np.arccosh(absg) / 3.0)
                / np.sqrt(absg**2 - 1.0)
                * np.sign(g)
                * dg
                / 3.0
            )
            if np.abs(r0) <= 1e-14:
                dt2 += t * dr0

    elif r1 > 0.0:
        t1 = _get_t1(r1)
        t2 = np.sinh(np.arcsinh(g) / 3.0)

        dt1 = _get_dt1(r1) * dr1
        dt2 = np.cosh(np.arcsinh(g) / 3.0) / np.sqrt(g**2 + 1.0) * dg / 3.0
    else:
        raise ValueError("r1 cannot be zero for one real root.")

    z = t1 * dt2 + dt1 * t2 - np.array([1.0 / 3.0, 0.0, 0.0])
    return z.reshape((1, 3))


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def three_roots(c2: float, c1: float, c0: float) -> np.ndarray:
    """Compute the three distinct real roots of the cubic polynomial using the
    trigonometric method of Vieta.

    See also:
        https://en.wikipedia.org/wiki/
        Cubic_equation#Trigonometric_and_hyperbolic_solutions

        https://de.wikipedia.org/wiki/
        Kubische_Gleichung#Die_F%C3%A4lle_mit_p_%E2%89%A0_0

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.
        c1: Coefficient of the linear term in the cubic polynomial.
        c0: Coefficient of the constant term in the cubic polynomial.

    Returns:
        A numpy array with the three roots, sorted in ascending order.

    """
    r1 = get_r1(c2, c1)
    r0 = get_r0(c2, c1, c0)

    assert r1 < 0.0, "r1 must be negative for 3 real roots."
    t1 = _get_t1(-r1)
    g = _get_Gamma(r1, r0)
    # prevening out-of-bound errors due to numerical inaccuracies
    assert np.abs(g) < 1.0, f"Gamma argument out of bounds (0,1) for 3 real roots: {g}"
    t2 = np.arccos(g) / 3.0

    z1 = -t1 * np.cos(t2 - np.pi / 3.0)
    z2 = -t1 * np.cos(t2 + np.pi / 3.0)
    z3 = t1 * np.cos(t2)

    return np.array([z1, z2, z3]) - c2 / 3.0

    # z1 = t_1 * np.cos(t_2) - c2 / 3.0
    # z2 = t_1 * np.cos(t_2 + 2.0 / 3.0* np.pi) - c2 / 3.0
    # z3 = t_1 * np.cos(t_2 + 4.0 / 3.0 * np.pi) - c2 / 3.0

    # roots = np.array([z1, z2, z3])
    # roots.sort()

    # return roots


@_COMPILE_DECORATOR(
    nb.f8[:, :](nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def d_three_roots(c2: float, c1: float, c0: float) -> np.ndarray:
    """Derivatives of the three distinct real roots with respect to c2, c1 and c0.

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.
        c1: Coefficient of the linear term in the cubic polynomial.
        c0: Coefficient of the constant term in the cubic polynomial.

    Returns:
        A numpy array with shape (3, 3), where the first axis corresponds to the three
        roots, and the second axis to the derivatives with respect to (c2, c1, c0).
        Like :func:`three_roots`, the roots are ordered by size in ascending order.

    """
    r1 = get_r1(c2, c1)
    r0 = get_r0(c2, c1, c0)

    dr1 = get_dr1(c2)
    dr0 = get_dr0(c2, c1)

    assert r1 < 0.0, "r1 must be negative for three real roots."
    t1 = _get_t1(-r1)
    g = _get_Gamma(r1, r0)
    # prevening out-of-bound errors due to numerical inaccuracies
    assert np.abs(g) < 1.0, f"acos argument out of bounds: {g}"
    t2 = np.arccos(g) / 3.0

    dg = _get_dGamma(r1, r0)
    dg = dg[0] * dr1 + dg[1] * dr0
    dt1 = -_get_dt1(-r1) * dr1
    dt2 = (-1 / np.sqrt(1.0 - g**2) * dg) / 3.0

    dc2 = np.array([-1.0 / 3.0, 0.0, 0.0])

    dz_1 = -np.cos(t2 - np.pi / 3.0) * dt1 + t1 * np.sin(t2 - np.pi / 3.0) * dt2 + dc2
    dz_2 = -np.cos(t2 + np.pi / 3.0) * dt1 + t1 * np.sin(t2 + np.pi / 3.0) * dt2 + dc2
    dz_3 = np.cos(t2) * dt1 - t1 * np.sin(t2) * dt2 + dc2
    return np.vstack((dz_1, dz_2, dz_3))


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def calculate_roots(c2: float, c1: float, c0: float, eps: float) -> np.ndarray:
    """Calculate the roots of a cubic polynomial represented by its coefficients
    :math:`c_2, c_1, c_0`.

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.
        c1: Coefficient of the linear term in the cubic polynomial.
        c0: Coefficient of the constant term in the cubic polynomial.
        eps: Tolerance for determining whether the discriminant is zero.

    Returns:
        A 1D array containing the real root(s) in ascending order.

    """
    match get_root_case(c2, c1, c0, eps):
        case 0:
            val = triple_root(c2)
        case 1:
            val = one_root(c2, c1, c0)
        case 2:
            val = two_roots(c2, c1, c0)
        case 3:
            val = three_roots(c2, c1, c0)
        case _:
            # Should never happen.
            raise NotImplementedError(f"Uncovered root case encountered.")

    return val


@_COMPILE_DECORATOR(
    nb.f8[:, :](nb.f8, nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def calculate_root_derivatives(
    c2: float, c1: float, c0: float, eps: float
) -> np.ndarray:
    """Calculate the derivative of roots of a cubic polynomial with respect to its
    coefficients :math:`c_2, c_1, c_0`.

    Parameters:
        c2: Coefficient of the quadratic term in the cubic polynomial.
        c1: Coefficient of the linear term in the cubic polynomial.
        c0: Coefficient of the constant term in the cubic polynomial.
        eps: Tolerance for determining whether the discriminant is zero.

    Returns:
        A 2D array containing the derivatives w.r.t. :math:`c_2, c_1, c_0` column wise.
        Row-wise the derivatives correspond to the root returned by
        :func:`calculate_roots`.

    """
    match get_root_case(c2, c1, c0, eps):
        case 0:
            val = d_triple_root(c2)
        case 1:
            val = d_one_root(c2, c1, c0)
        case 2:
            val = d_two_roots(c2, c1, c0)
        case 3:
            val = d_three_roots(c2, c1, c0)
        case _:
            # Should never happen.
            raise NotImplementedError(f"Uncovered root case encountered.")

    return val
