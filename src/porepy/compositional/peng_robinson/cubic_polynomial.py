"""Functionality for calculating real solutions or real cubic polynomials, efficiently
and in large quantities.

This module utilizes numba to compile the core routines, which can than be wrapped in
actual applications like the Peng-Robinson equation of state.

The base formulation of the cubic polynomial is

.. math::
    z^3 + c_0 z^2 + c_1 z + c_2 = 0,

from which the reduced form is obtained as

.. math::
    z^3 + r_0 z + r_1 = 0.

The root is a function of the coefficient array :math:`c`.
Throughout the module it holds :math:`c = [c_0, c_1, c_2]` and :math:`r=[r_0, r_1]`.

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

from .._numba_interface import NUMBA_CACHE, NUMBA_FAST_MATH, njit

__all__ = [
    "get_root_case",
    "calculate_roots",
    "calculate_root_derivatives",
]


_COMPILER = njit
"""Decorator for compiling functions in this module.

Alternative compilers are the :obj:`numba.cfunc` call-back decorator, or future AOT
compilation.

"""


@_COMPILER(nb.f8(nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=True)
def _theta_from_r(r: np.ndarray) -> float:
    """Calculate the auxiliary variable used in the trigonometric solution of
    real roots of real cubic polynomials.

    Parameters:
        r: Reduced coefficients.

    Returns:
        The auxiliary variable.

    """
    _c = 3.0 * np.sqrt(3.0) * 0.5
    return -r[1] * _c / np.sqrt(np.abs(r[0]) ** 3)


@_COMPILER(nb.f8[:](nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=True)
def _dtheta_from_r(r: np.ndarray) -> np.ndarray:
    """Gradient of :func:`_theta_from_r`.

    Parameters:
        r: Reduced coefficients.

    Returns:
        A ``(2,)`` array.

    """
    _ar = np.abs(r[0])
    _ar3 = _ar**3
    _c = 3.0 * np.sqrt(3.0) * 0.5
    return np.array(
        [
            r[1] * _c * 1.5 / (_ar3 * np.sqrt(_ar)) * r[0],
            -_c / np.sqrt(_ar3),
        ]
    )


@_COMPILER(
    [
        nb.f8[:](nb.f8[:]),
        nb.f8[:](nb.int_[:]),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def r_from_c(c: np.ndarray) -> np.ndarray:
    """Computes the coefficients of the reduced polynomial

    .. math::

        x^3 + r_0 x + r_1

    which is unique for all normalized cubic polynomials in standard form.

    Parameters:
        c: Coefficients in normal form.

    Returns:
        An array with ``shape=(2,)`` containing the ``r_0`` and ``r_1``.

    """
    _c0 = 1.0 / 3.0
    _c1 = 2.0 / 27.0
    return np.array(
        (
            c[1] - _c0 * c[0] ** 2,
            _c1 * c[0] ** 3 - _c0 * c[0] * c[1] + c[2],
        )
    ).astype(np.float64)


@_COMPILER(
    [
        nb.f8[:, :](nb.f8[:]),
        nb.f8[:, :](nb.int_[:]),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def dr_from_c(c: np.ndarray) -> np.ndarray:
    """Returns the derivatives of the reduced polynomial coefficients with respect to
    the standard coefficients (Jacobian of :func:`r_from_c`)

    Parameters:
        c: Coefficients in normal form.

    Returns:
        An array with ``shape=(2, 3)`` containing the Jacobian.

    """
    _c0 = 1.0 / 3.0
    _c1 = 2.0 / 3.0

    dr = np.zeros((2, 3))
    dr[0, 0] = -_c1 * c[0]
    dr[0, 1] = 1.0
    dr[1, 0] = _c0 * (_c1 * c[0] ** 2 - c[1])
    dr[1, 1] = -_c0 * c[0]
    dr[1, 2] = 1.0
    return dr.astype(np.float64)


@_COMPILER([nb.f8(nb.f8[:]), nb.f8(nb.int_[:])], fastmath=NUMBA_FAST_MATH, cache=True)
def discriminant(rc: np.ndarray) -> float:
    """Calculate the discriminant of the reduced cubic polynomial.

    Parameters:
        rc: Reduced or normal coefficients.

    Raises:
        ValueError: If ``rc`` is not of size 2 or 3.

    Returns:
        The discriminant of the polynomial. If positive, the polynomial has 1 real root.
        If negative, it has 3 distinct real roots. If zero, it has multiple real roots
        with at least one with higher algebraic multiplicity.

    """
    if rc.size == 2:
        r = rc.astype(np.float64)
    elif rc.size == 3:
        r = r_from_c(rc)
    else:
        raise ValueError(
            "Expecting coefficient array of size 2 or 3 (reduced or normal)."
        )
    _c0 = 1.0 / 27.0
    _c1 = 0.25
    return _c0 * r[0] ** 3 + _c1 * r[1] ** 2


@_COMPILER(
    [
        nb.int_(nb.f8[:], nb.f8),
        nb.int_(nb.int_[:], nb.f8),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def get_root_case(c: np.ndarray, eps: float) -> int:
    """Determine the case for the roots of the cubic polynomial.

    Parameters:
        c: Coefficient in normal form.
        eps: Tolerance for determining degeneracy.

    Returns:
        An integer indicating the case.

        - 3: 3 distinct real roots.
        - 2: 2 distinct real roots, one with algebraic multiplicity of 2.
        - 1: 1 real root and two complex conjugate roots.
        - 0: 1 real root with multiplicity three (triple root).

    """
    r = r_from_c(c)
    D = discriminant(r)
    absq = np.abs(r[1])

    # Fail-safe:
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        raise ValueError(
            "Root case not determinable if nans or infs present in coefficients."
        )

    # Degenerate case with triple root.
    if max(np.abs(r[0]), absq) < eps:
        return 0

    # NOTE Usage of D0 and DR is a numerically stable way of determining the
    # discriminant if r contains very large or very small values.
    _c = 1.0 / np.sqrt(27.0)
    D0 = np.abs(r[1]) * 0.5
    DR = _c * np.abs(r[0]) * np.sqrt(np.abs(r[0]))

    # Positive discriminant => one real root, two complex conjugate roots.
    if D0 > DR * (1 + eps) or D > eps:
        return 1
    # Negative discriminant => three distinct real roots.
    elif D0 < DR * (1 - eps) or D < -eps:
        # Edge cases, welcome to floating point hell.
        # r[0] must be strictly negative for 3 roots. Cancelation can lead to pitfalls.
        if absq < 1e-7 and 0 < D < eps and r[0] >= 0.0:
            return 1
        # If discriminant is numerically zero in float64, we want to return 2 roots to
        # avoid a 3-root case where 2 roots are numerically identical.
        if np.abs(D) <= np.finfo(np.float64).eps:
            return 2
        return 3
    # Degenerate case 2 roots: Numerically almost never the case, but required for
    # completeness.
    else:
        # np.finfo(np.float64).eps
        assert np.abs(D) <= eps, "Expecting degenerate discriminant."
        return 2


@_COMPILER(nb.f8[:](nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=True)
def triple_root(c: np.ndarray) -> np.ndarray:
    """Calculate the triple root of the cubic polynomial, which is always ``c_0 / 3``.

    Parameters:
        c: Coefficients in normal form.

    Returns:
        The triple root.

    """
    _c0 = 1.0 / 3.0
    return np.array([-_c0 * c[0]])


@_COMPILER(nb.f8[:, :](nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=True)
def d_triple_root(c: np.ndarray) -> np.ndarray:
    """Derivatives of the triple root with respect to the coefficients.

    Note:
        Though always a constant array, we keep the signature for compatibility with
        other root methods.

    Parameters:
        c: Coefficients in normal form.

    Returns:
        A ``(3,)`` array containing ``(-1/3, 0, 0)``.

    """
    _ds = np.array((-1.0 / 3.0, 0.0, 0.0)).reshape((1, 3))
    return _ds


@_COMPILER(nb.f8[:](nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
def two_roots(c: np.ndarray) -> np.ndarray:
    """Compute the two roots of the cubic polynomial, in the case where one of them has
    multiplicity two.

    See also:
        https://en.wikipedia.org/wiki/Cubic_equation#Multiple_root

    Important:
        Contrary to :func:`three_roots`, the ordering here happens explicitly since
        the formula does not yield always properly ordered roots. I.e., there the
        function is no smooth where the switch happens.

    Parameters:
        c: Coefficients in normal form.

    Returns:
        A ``(2,)`` array containing the roots ordered by size.

    """

    r = r_from_c(c)
    u = 3.0 * r[1] / r[0]
    z1 = u
    z2 = -u * 0.5

    if z1 < z2:
        z = np.array([z1, z2])
    else:
        z = np.array([z2, z1])

    return z - c[0] / 3.0


@_COMPILER(nb.f8[:, :](nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
def d_two_roots(c: np.ndarray) -> np.ndarray:
    """Jacobian of :func:`two_roots`.

    Parameters:
        c: Coefficients in normal form..

    Returns:
        A ``(2, 3)`` array containing the derivatives with respect to ``c`` column-wise.
        The second row belongs to the larger root (consistent order with
        :func:`two_roots`).

    """

    r = r_from_c(c)
    dr = dr_from_c(c)

    u = 3.0 * r[1] / r[0]

    du = (3.0 * dr[1] - 3.0 * r[1] / r[0] * dr[0]) / r[0]

    _ds = np.array((-1.0 / 3.0, 0.0, 0.0))

    dz1_dc = du + _ds
    dz2_dc = -du / 2.0 + _ds

    if u < -u / 2.0:
        return np.vstack((dz1_dc, dz2_dc))
    else:
        return np.vstack((dz2_dc, dz1_dc))


@_COMPILER(nb.f8[:](nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
def one_root(c: np.ndarray) -> np.ndarray:
    """Calculate the single (real) root of the cubic polynomial, where applicable.

    See also:
        https://en.wikipedia.org/wiki/
        Cubic_equation#Trigonometric_and_hyperbolic_solutions

        https://de.wikipedia.org/wiki/
        Kubische_Gleichung#Die_F%C3%A4lle_mit_p_%E2%89%A0_0

    Parameters:
        c: Coefficients in normal form.

    Returns:
        A ``(1,)`` array containing the single real root.

    """
    r = r_from_c(c)
    eps = 1e-15

    _c0 = 1.0 / 3.0
    _c1 = 2.0 / np.sqrt(3.0)

    # Edge case.
    if np.abs(r[0]) <= eps:
        t1 = _c0
        t2_ = c[0] ** 3 - 27.0 * c[2]
        t2 = np.cbrt(np.abs(t2_)) * np.sign(t2_)
    else:
        theta = _theta_from_r(r)
        t1 = np.sign(r[0]) * _c1 * np.sqrt(np.abs(r[0]))

        if r[0] < 0.0:
            absg = np.abs(theta)

            # Edge case.
            if 1.0 - eps < absg < 1.0 + eps:
                t1 *= -1.0
                t2 = 1.0
            else:
                t2 = np.sign(r[1]) * np.cosh(_c0 * np.arccosh(absg))

        elif r[0] > 0.0:
            t2 = np.sinh(_c0 * np.arcsinh(theta))

    return np.array([t1 * t2]) - _c0 * c[0]


@_COMPILER(nb.f8[:, :](nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
def d_one_root(c: np.ndarray) -> np.ndarray:
    """Jacobian of :func:`one_root`.

    Parameters:
        c: Coefficients in normal form.

    Returns:
        A ``(1,3)`` array containing the derivatives column-wise.

    """

    r = r_from_c(c)
    dr = dr_from_c(c)
    eps = 1e-15

    _c0 = 1.0 / 3.0
    _c1 = 2.0 / np.sqrt(3.0)
    _c2 = 1.0 / np.sqrt(3.0)
    _ds = np.array((-1.0 / 3.0, 0.0, 0.0))

    if np.abs(r[0]) <= eps:
        _cr = np.cbrt(np.abs(r[1]))
        t1 = 1.0
        t2 = _cr * np.sign(r[1])

        dt1 = np.zeros(3)
        dt2 = np.sign(r[1]) / _cr**2 * dr[1]
    else:
        theta = _theta_from_r(r)
        dtheta = np.dot(_dtheta_from_r(r), dr)

        _sar = np.sqrt(np.abs(r[0]))

        t1 = np.sign(r[0]) * _c1 * _sar
        dt1 = _c2 / _sar * dr[0]

        if r[0] < 0.0:
            absg = np.abs(theta)
            dt2 = np.zeros(3)

            # Special case for numerical stability.
            if 1.0 - eps < absg < 1.0 + eps:
                t1 *= -1.0
                dt1 *= -1.0
                t2 = 1.0
            else:
                acoshg = _c0 * np.arccosh(absg)
                t = np.cosh(acoshg)
                t2 = np.sign(r[1]) * t
                dt2 = (
                    np.sign(r[1])
                    * np.sinh(acoshg)
                    * _c0
                    / np.sqrt(absg**2 - 1.0)
                    * np.sign(theta)
                    * dtheta
                )

        elif r[0] > 0.0:
            asinht = _c0 * np.arcsinh(theta)
            t2 = np.sinh(asinht)
            dt2 = _c0 * np.cosh(asinht) / np.sqrt(theta**2 + 1.0) * dtheta

    z = t1 * dt2 + dt1 * t2 + _ds
    return z.reshape((1, 3))


@_COMPILER(nb.f8[:](nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
def three_roots(c: np.ndarray) -> np.ndarray:
    """Compute the three distinct real roots of the cubic polynomial using the
    trigonometric approach.

    See also:
        https://en.wikipedia.org/wiki/
        Cubic_equation#Trigonometric_and_hyperbolic_solutions

        https://de.wikipedia.org/wiki/
        Kubische_Gleichung#Die_F%C3%A4lle_mit_p_%E2%89%A0_0

    Note:
        In theory, the formula yields roots which are always ordered, but that has
        not been tested. Might be non-smooth around areas where 2 roots approach
        each other in terms of value.

    Parameters:
        c: Coefficients in normal form.

    Returns:
        A ``(3,)`` array containing the roots ordered by size.

    """
    r = r_from_c(c)

    _c0 = 1.0 / 3.0
    _c1 = 2.0 / np.sqrt(3.0)
    pi3 = np.pi / 3.0

    assert r[0] < 0.0, "r1 must be negative for 3 real roots."
    t1 = _c1 * np.sqrt(np.abs(r[0]))
    theta = _theta_from_r(r)
    theta = max(min(theta, 1.0), -1.0)  # Avoid out of bounds errors.
    t2 = _c0 * np.arccos(theta)

    z1 = -t1 * np.cos(t2 - pi3)
    z2 = -t1 * np.cos(t2 + pi3)
    z3 = t1 * np.cos(t2)

    return np.array((z1, z2, z3)) - _c0 * c[0]


@_COMPILER(nb.f8[:, :](nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
def d_three_roots(c: np.ndarray) -> np.ndarray:
    """Jacobian of :func:`two_roots`.

    Parameters:
        c: Coefficients in normal form..

    Returns:
        A ``(3, 3)`` array containing the derivatives with respect to ``c`` column-wise.

    """
    r = r_from_c(c)
    dr = dr_from_c(c)

    _c0 = 1.0 / 3.0
    _c1 = 2.0 / np.sqrt(3.0)
    _c2 = 1.0 / np.sqrt(3.0)
    pi3 = np.pi / 3.0
    _ds = np.array((-1.0 / 3.0, 0.0, 0.0))

    assert r[0] < 0.0, "r1 must be negative for three real roots."
    _sar = np.sqrt(np.abs(r[0]))
    t1 = _c1 * _sar
    dt1 = -_c2 / _sar * dr[0]
    theta = _theta_from_r(r)
    theta = max(min(theta, 1.0), -1.0)  # Avoid out of bounds errors.
    dtheta = np.dot(_dtheta_from_r(r), dr)

    t2 = _c0 * np.arccos(theta)
    dt2 = -_c0 * dtheta / np.sqrt(1.0 - theta**2)

    _ds = np.array([-1.0 / 3.0, 0.0, 0.0])

    dz_1 = -np.cos(t2 - pi3) * dt1 + t1 * np.sin(t2 - pi3) * dt2 + _ds
    dz_2 = -np.cos(t2 + pi3) * dt1 + t1 * np.sin(t2 + pi3) * dt2 + _ds
    dz_3 = np.cos(t2) * dt1 - t1 * np.sin(t2) * dt2 + _ds
    return np.vstack((dz_1, dz_2, dz_3))


@_COMPILER(
    [
        nb.f8[:](nb.f8[:], nb.f8),
        nb.f8[:](nb.int_[:], nb.f8),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def calculate_roots(c: np.ndarray, eps: float) -> np.ndarray:
    """Calculate the roots of a cubic polynomial represented by its coefficients
    :math:`c_2, c_1, c_0`.

    Parameters:
        c: Coefficient in normal form.
        eps: Tolerance for determining degeneracy.

    Returns:
        A 1D array containing the real root(s) in ascending order.

    """
    c_ = c.astype(np.float64)
    match get_root_case(c_, eps):
        case 0:
            val = triple_root(c_)
        case 1:
            val = one_root(c_)
        case 2:
            val = two_roots(c_)
        case 3:
            val = three_roots(c_)
        case _:
            # Should never happen.
            raise NotImplementedError(f"Uncovered root case encountered.")

    return val


@_COMPILER(
    [
        nb.f8[:, :](nb.f8[:], nb.f8),
        nb.f8[:, :](nb.int_[:], nb.f8),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def calculate_root_derivatives(c: np.ndarray, eps: float) -> np.ndarray:
    """Jacobian of :func:`calculate_roots`.

    Parameters:
        c: Coefficient in normal form.
        eps: Tolerance for determining degeneracy.

    Returns:
        A 2D array containing the derivatives w.r.t. ``c`` column-wise.
        Row-wise the derivatives correspond to the roots returned by
        :func:`calculate_roots`.

    """
    c_ = c.astype(np.float64)
    match get_root_case(c_, eps):
        case 0:
            val = d_triple_root(c_)
        case 1:
            val = d_one_root(c_)
        case 2:
            val = d_two_roots(c_)
        case 3:
            val = d_three_roots(c_)
        case _:
            # Should never happen.
            raise NotImplementedError(f"Uncovered root case encountered.")

    return val
