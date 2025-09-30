"""Functionality for solving cubic polynomials, efficiently and in large quantities.

This module utilizes numba to compile the core routines, which can than be wrapped in
actual applications like the Peng-Robinson equation of state.

The base formulation of the cubic polynomial is

.. math::
    Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0,

from which the reduced form is obtained as

.. math::
    z^3 + r_1 z + r_0 = 0.

The root is a function of the coefficients :math:`c_0`, :math:`c_1`, and :math:`c_2`.
Most importantly, this module implements also the derivatives of the roots with
    respect to the coefficients, which are essential in many applications.

Separate tests are available to check whether a root is a real root or a pseudo-root.

"""

from __future__ import annotations

import numba as nb
import numpy as np

from .._core import NUMBA_FAST_MATH, NUMBA_CACHE


# _COMPILE_KWARGS = dict(nopython=True, cache=True)
# _COMPILE_DECORATOR = nb.cfunc
_COMPILE_KWARGS = dict(fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
_COMPILE_DECORATOR = nb.njit


@_COMPILE_DECORATOR(
    nb.f8(nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def r1(c2: float, c1: float) -> float:
    """Calculate the reduced coefficient r1.

    Parameters:
        c2:
            Coefficient of the quadratic term in the cubic polynomial.
        c1:
            Coefficient of the linear term in the cubic polynomial.

    Returns:
        The reduced coefficient r1.

    """
    return c1 - c2**2 / 3.0


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8),
    **_COMPILE_KWARGS,
)
def dr1(c2: float) -> np.ndarray:
    """Derivatives of the reduced coefficient r1 with respect to c2, c1 and c0.

    The derivative with respect to c1 is always 1.
    The derivative with respect to c0 is always 0.

    Parameters:
        c2:
            Coefficient of the quadratic term in the cubic polynomial.

    Returns:
        A numpy array with ``shape=(3,)`` containing the derivatives with respect to
        ``(c2, c1, c0)``.

    """
    return np.array([-2.0 * c2 / 3.0, 1.0, 0.0])


@_COMPILE_DECORATOR(
    nb.f8(nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def r0(c2: float, c1: float, c0: float) -> float:
    """Calculate the reduced coefficient r0.

    Parameters:
        c2:
            Coefficient of the quadratic term in the cubic polynomial.
        c1:
            Coefficient of the linear term in the cubic polynomial.
        c0:
            Coefficient of the constant term in the cubic polynomial.

    Returns:
        The reduced coefficient r0.

    """
    return 2.0 / 27.0 * c2**3 - (c1 * c2) / 3.0 + c0


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def dr0(c2: float, c1: float) -> np.ndarray:
    """Derivatives of the reduced coefficient r0 with respect to c2, c1 and c0.

    The derivative with respect to c0 is always 1.

    Parameters:
        c2:
            Coefficient of the quadratic term in the cubic polynomial.
        c1:
            Coefficient of the linear term in the cubic polynomial.

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

    Parameters:
        r1:
            Reduced coefficient r1.
        r0:
            Reduced coefficient r0.

    Returns:
        The discriminant.

    """
    return r1**3 / 27.0 + r0**2 / 4.0


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
    r_1 = r1(c2, c1)
    r_0 = r0(c2, c1, c0)

    D = discriminant(r_1, r_0)

    # Positive D => one real root, two complex conjugate roots.
    if D > eps:
        return 1
    # Negative D => three distinct real roots.
    elif D < -eps:
        return 3
    # D == 0 => multiple roots.
    else:
        # r_1 == 0 => triple root.
        if np.abs(r_1) < eps:
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

    Parameters:
        c2:
            Coefficient of the quadratic term in the cubic polynomial.

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
    """
    return np.array([-1.0 / 3.0, 0.0, 0.0]).reshape((1, 3))


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def one_root(c2: float, c1: float, c0: float) -> np.ndarray:
    """Calculate the single (real) root of the cubic polynomial, when there is only one.

    Parameters:
        c2:
            Coefficient of the quadratic term in the cubic polynomial.
        c1:
            Coefficient of the linear term in the cubic polynomial.
        c0:
            Coefficient of the constant term in the cubic polynomial.

    Returns:
        The single (real) root.

    """
    r_1 = r1(c2, c1)
    r_0 = r0(c2, c1, c0)

    D = discriminant(r_1, r_0)
    sqrt_D = np.sqrt(D)

    t_1 = sqrt_D - r_0 / 2.0
    t_2 = -sqrt_D - r_0 / 2.0

    if np.abs(t_2) > np.abs(t_1):
        t = t_2
    else:
        t = t_1

    u = np.cbrt(t)
    v = 0.0 if u == 0.0 else -r_1 / (3.0 * u)

    # C_plus = (-r_0 / 2.0 + sqrt_D) ** (1.0 / 3.0)
    # C_minus = (-r_0 / 2.0 - sqrt_D) ** (1.0 / 3.0)
    # return C_plus + C_minus - c2 / 3.0
    return np.array([u + v - c2 / 3.0])


@_COMPILE_DECORATOR(
    nb.f8[:, :](nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def d_one_root(c2: float, c1: float, c0: float) -> np.ndarray:
    """Derivatives of the single (real) root with respect to c2, c1 and c0.

    Parameters:
        c2:
            Coefficient of the quadratic term in the cubic polynomial.
        c1:
            Coefficient of the linear term in the cubic polynomial.
        c0:
            Coefficient of the constant term in the cubic polynomial.

    Returns:
        A numpy array with ``shape=(3,)`` containing the derivatives with respect to
        ``(c2, c1, c0)``.

    """
    r_1 = r1(c2, c1)
    r_0 = r0(c2, c1, c0)

    dr_1 = dr1(c2)
    dr_0 = dr0(c2, c1)

    D = discriminant(r_1, r_0)
    sqrt_D = np.sqrt(D)

    dsqrt_D_dr1 = 3.0 * r_1**2 / 27.0 / (2.0 * sqrt_D) if sqrt_D != 0.0 else 0.0
    dsqrt_D_dr0 = r_0 / 4.0 / sqrt_D if sqrt_D != 0.0 else 0.0

    t_1 = sqrt_D - r_0 / 2.0
    t_2 = -sqrt_D - r_0 / 2.0

    if np.abs(t_2) > np.abs(t_1):
        t = t_2
        dt_dr1 = -dsqrt_D_dr1
        dt_dr0 = -0.5 - dsqrt_D_dr0
    else:
        t = t_1
        dt_dr1 = dsqrt_D_dr1
        dt_dr0 = -0.5 + dsqrt_D_dr0

    u = np.cbrt(t)

    outer = 1.0 / (3.0 * np.cbrt(t**2)) if t != 0.0 else 0.0
    du_dr1 = outer * dt_dr1
    du_dr0 = outer * dt_dr0
    du_dc = du_dr1 * dr_1 + du_dr0 * dr_0

    dv_dr0 = 0.0 if u == 0.0 else (r_1 / 3.0 / u**2 * du_dr0)
    dv_dr1 = 0.0 if u == 0.0 else (-1.0 / (3.0 * u) + r_1 / 3.0 / u**2 * du_dr1)
    dv_dc = dv_dr1 * dr_1 + dv_dr0 * dr_0

    # NOTE: d_triple_root is equivalent to the derivative of -c2/3
    return (du_dc + dv_dc + np.array([-1.0 / 3.0, 0.0, 0.0])).reshape((1, 3))


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def two_roots(c2: float, c1: float, c0: float) -> np.ndarray:
    """Compute the two roots of the cubic polynomial, in the case where one of them has
    multiplicity two.

    Parameters:
        c2:
            Coefficient of the quadratic term in the cubic polynomial.
        c1:
            Coefficient of the linear term in the cubic polynomial.
        c0:
            Coefficient of the constant term in the cubic polynomial.

    Returns:
        A numpy array with the two roots. The second entry contains the larger root.

    """

    r_1 = r1(c2, c1)
    r_0 = r0(c2, c1, c0)

    u = 3.0 / 2.0 * r_0 / r_1

    z1 = 2.0 * u - c2 / 3.0
    z2 = -u - c2 / 3.0

    if z1 < z2:
        return np.array([z1, z2])
    else:
        return np.array([z2, z1])


@_COMPILE_DECORATOR(
    nb.f8[:, :](nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def d_two_roots(c2: float, c1: float, c0: float) -> np.ndarray:
    """Derivatives of the two roots with respect to c2, c1 and c0, in the case where one
    of them has multiplicity two.

    Parameters:
        c2:
            Coefficient of the quadratic term in the cubic polynomial.
        c1:
            Coefficient of the linear term in the cubic polynomial.
        c0:
            Coefficient of the constant term in the cubic polynomial.

    Returns:
        A numpy array with shape (2, 3), where the first axis corresponds to the two
        roots, and the second axis to the derivatives with respect to (c2, c1, c0).
        The second row belongs to the larger root.

    """

    r_1 = r1(c2, c1)
    r_0 = r0(c2, c1, c0)

    dr_1 = dr1(c2)
    dr_0 = dr0(c2, c1)

    u = 3.0 / 2.0 * r_0 / r_1

    du_dr1 = -3.0 / 2.0 * r_0 / (r_1**2)
    du_dr0 = 3.0 / 2.0 / r_1

    du_dc = du_dr1 * dr_1 + du_dr0 * dr_0

    dc2 = np.array([-1.0 / 3.0, 0.0, 0.0])

    dz1_dc = 2.0 * du_dc + dc2
    dz2_dc = -du_dc + dc2

    if 2.0 * u - c2 / 3.0 < -u - c2 / 3.0:
        return np.vstack((dz1_dc, dz2_dc))
    else:
        return np.vstack((dz2_dc, dz1_dc))


@_COMPILE_DECORATOR(
    nb.f8[:](nb.f8, nb.f8, nb.f8),
    **_COMPILE_KWARGS,
)
def three_roots(c2: float, c1: float, c0: float) -> np.ndarray:
    """Compute the three distinct real roots of the cubic polynomial.

    Parameters:
        c2:
            Coefficient of the quadratic term in the cubic polynomial.
        c1:
            Coefficient of the linear term in the cubic polynomial.
        c0:
            Coefficient of the constant term in the cubic polynomial.

    Returns:
        A numpy array with the three roots, sorted in ascending order.

    """
    r_1 = r1(c2, c1)
    r_0 = r0(c2, c1, c0)

    t_2 = np.arccos(-r_0 / 2.0 * np.sqrt(-27.0 / r_1**3)) / 3.0
    t_1 = 2.0 * np.sqrt(-r_1 / 3.0)

    z1 = -t_1 * np.cos(t_2 - np.pi / 3.0)
    z2 = -t_1 * np.cos(t_2 + np.pi / 3.0)
    z3 = t_1 * np.cos(t_2)

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
    """"""
    return np.eye(3)


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
