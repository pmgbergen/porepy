"""This module contains compiled versions of the Peng-Robinson equation of state.

The functions provided here are building on lambdified expressions created using
:mod:`sympy` and then just-in-time compiled.

"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numba as nb
import numpy as np

from .._global_thermodynamic_reference_state import R_U
from .._numba_interface import NUMBA_CACHE, NUMBA_FAST_MATH, njit
from ..compiled_eos import (
    FUGACITY_COEFF_DERIVATIVE_FUNC_SIGNATURE,
    FUGACITY_COEFF_FUNC_SIGNATURE,
    PREARGUMENT_DFUNC_SIGNATURE,
    PREARGUMENT_FUNC_SIGNATURE,
    PROPERTY_DERIVATIVE_FUNC_SIGNATURE,
    PROPERTY_FUNC_SIGNATURE,
    CompiledEoS,
    PropertyFunctionDict,
    ScalarFunction,
    VectorFunction,
)
from ..ideal import IdealFluid, IdealProperty_T
from ..ideal.ideal_fluid import grad_ideal_rho, ideal_rho
from ..materials import FluidComponent
from ..states import PhysicalState
from .compressibility_factor import (
    A_CRIT,
    B_CRIT,
    get_compressibility_factor,
    get_compressibility_factor_derivatives,
)

__all__ = [
    "a_VdW",
    "CompiledPengRobinson",
]


logger = logging.getLogger(__name__)

_COMPILER = njit
"""Decorator for compiling functions in this module.

Uses :func:`~porepy.compositional._numba_interface.njit`.

"""


@_COMPILER(
    [nb.f8[:, :](nb.f8[:]), nb.f8[:](nb.f8[:, :])], fastmath=NUMBA_FAST_MATH, cache=True
)
def compact_dense_symmat(mat_arr: np.ndarray) -> np.ndarray:
    """Compact storage of symmetric, dense, square matrix by storing only (parts of)
    rows of the upper triangle matrix, concatenated into a 1D array.

    Serves also as a reverse operation (expanding 1D to 2D array.)

    Parameters:
        matt_arr: A 1D or a 2D array with ``shape=(n(n+1)/2,)`` or ``shape=(n,n)``
            respectively.

    Returns:
        If ``mat_arr`` is a 1D array, returns a symmetric 2D array.
        If ``mat_arr`` is a 2D array, returns a 1D array.

    """
    if mat_arr.ndim == 1:
        m = mat_arr.size
        n = (-1 + np.sqrt(1 + 8 * m)) / 2
        N = int(n)
        if n != N or N < 0:
            raise ValueError("Could not determine square shape of original matrix.")
    elif mat_arr.ndim == 2:
        N = mat_arr.shape[0]
        assert mat_arr.shape[1] == N, "Expecting square matrix"
    else:
        raise ValueError("Expecting either 1D or 2D array.")

    ids = np.array([0] + [n * (n + 1) / 2 for n in range(N)]).astype(np.int_)

    if mat_arr.ndim == 2:
        out = np.zeros((int(N * (N + 1) / 2)))
        for i in range(N):
            out[i * N - ids[i] : (i + 1) * N - ids[i + 1]] = mat_arr[i, i:]
    elif mat_arr.ndim == 1:
        out = np.zeros((N, N))
        for i in range(N):
            out[i, i:] = mat_arr[i * N - ids[i] : (i + 1) * N - ids[i + 1]]
            # For symmetry.
            out[i, i] /= 2.0
        out = out + out.T

    return out


@_COMPILER(nb.f8(nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def bc_component(pc: np.ndarray, Tc: np.ndarray) -> float:
    """Computes the critical covolume of a component based on critical values.

    Parameters:
        pc: Critical pressure.
        Tc: Critical temperature.

    Returns:
        :math:`B_c R \\frac{T_c}{p_c}`, with :math:`B_c` being
        :data:`~porepy.compositional.peng_robinson.compressibility_factor.B_CRIT`.

    """
    return B_CRIT * R_U * Tc / pc


@_COMPILER(nb.f8(nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def ac_component(pc: float, Tc: float) -> float:
    """Computes the critical cohesion of a component based on critical values.

    Parameters:
        pc: Critical pressure.
        Tc: Critical temperature.

    Returns:
        :math:`A_c \\frac{(R T_c)^2}{p_c}`, with :math:`A_c` being
        :data:`~porepy.compositional.peng_robinson.compressibility_factor.A_CRIT`.

    """
    RT = R_U * Tc
    return A_CRIT * RT * RT / (pc * pc)


@_COMPILER(nb.f8(nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def _k_of_omega(omega: float) -> float:
    """Returns the weight depending on the acentric factor, which is used in
    :func:`alpha` and its derivatives."""
    if omega < 0.491:
        return 0.37464 + 1.54226 * omega - 0.26992 * omega * omega
    else:
        return (
            0.379642
            + 1.48503 * omega
            - 0.164423 * omega * omega
            + 0.016666 * omega * omega * omega
        )


@_COMPILER(nb.f8(nb.f8, nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
def alpha(T: float, Tc: float, omega: float) -> float:
    """Returns the temperature-dependent weight in the cohesion of a component.

    Note:
        Modified weight :math:`k(\\omega)` is used according to
        `Zhu and Okuno (2014) <https://doi.org/10.1016/j.fluid.2014.07.003>`_ .

    Parameters:
        T: Temperature.
        Tc: Critical temperature of the component.
        omega: Acentric factor of the component.

    Returns:
        :math:`(1 + k(\\omega)(1 - \\sqrt(\\frac{T}{T_c})))^2`
    """
    Tr = max(T / Tc, 1e-15)
    salpha = 1.0 + _k_of_omega(omega) * (1.0 - np.sqrt(Tr))
    return salpha * salpha


@_COMPILER(nb.f8(nb.f8, nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
def dalpha_dT(T: float, Tc: float, omega: float) -> float:
    """Returns the derivative of :func:`alpha` with respect to temperature."""
    k = _k_of_omega(omega)
    sqrtTr = np.sqrt(max(T / Tc, 1e-15))
    return -k / Tc * ((1 + k) / sqrtTr - k)


@_COMPILER(nb.f8(nb.f8, nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=NUMBA_CACHE)
def ddalpha_dTT(T: float, Tc: float, omega: float) -> float:
    """Returns the second derivative of :func:`alpha` w.r.t. temperature."""
    k = _k_of_omega(omega)
    sqrtTr = np.sqrt(max(T / Tc, 1e-15))
    return k * (k + 1) / 2 / (Tc * Tc) / (sqrtTr * sqrtTr * sqrtTr)


@_COMPILER(
    [
        nb.f8(nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:, :]),
        nb.f8(
            nb.f8,
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 2, "C", readonly=True),
        ),
        nb.f8(
            nb.f8,
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=False),
            nb.types.Array(nb.f8, 1, "C", readonly=False),
            nb.types.Array(nb.f8, 1, "C", readonly=False),
            nb.types.Array(nb.f8, 2, "C", readonly=False),
        ),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def a_VdW(
    T: float,
    xn: np.ndarray,
    Tcs: np.ndarray,
    omegas: np.ndarray,
    acs: np.ndarray,
    bips: np.ndarray,
) -> float:
    """Van der Waals cohesion for fluid mixtures.

    Notes:
        If there is 1 component, ``xn`` is overwritten with 1.

    Parameters:
        T: Temperature.
        xn: Partial fractions per component.
        Tcs: Critical temperature per component.
        omegas: Acentric factor per component.
        acs: Critical cohesion per component.
        bip: Symmetric matrix of binary interaction coefficients.

    Returns:
        :math:`\\sum_i\\sum_j x_i x_j\\sqrt{a_i a_j}(1 - \\delta_ij)`, using
        :func:`a_component` and :math:`\\delta` denoting binary interaction parameters.

    """

    nc = xn.size
    if nc == 1:
        return alpha(T, Tcs[0], omegas[0]) * acs[0]

    a = 0.0
    for i in range(nc):
        a_i = alpha(T, Tcs[i], omegas[i]) * acs[i]
        a += xn[i] * xn[i] * a_i
        for j in range(i + 1, nc):
            a += 2.0 * (
                xn[i]
                * xn[j]
                * np.sqrt(a_i * alpha(T, Tcs[j], omegas[j]) * acs[j])
                * (1.0 - bips[i, j])
            )

    return a


@_COMPILER(
    [
        nb.f8[:](nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:, :]),
        nb.f8[:](
            nb.f8,
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 2, "C", readonly=True),
        ),
        nb.f8[:](
            nb.f8,
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=False),
            nb.types.Array(nb.f8, 1, "C", readonly=False),
            nb.types.Array(nb.f8, 1, "C", readonly=False),
            nb.types.Array(nb.f8, 2, "C", readonly=False),
        ),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def grad_a_VdW(
    T: float,
    xn: np.ndarray,
    Tcs: np.ndarray,
    omegas: np.ndarray,
    acs: np.ndarray,
    bips: np.ndarray,
) -> np.ndarray:
    """Gradient of Van der Waals cohesion for fluid mixtures with respect to
    temperature and partial fractions.

    Notes:
        If there is 1 component the returned array contains only the temperature
        derivative.

    Parameters:
        T: Temperature.
        xn: Partial fractions per component.
        Tcs: Critical temperature per component.
        omegas: Acentric factor per component.
        acs: Critical cohesion per component.
        bip: Symmetric matrix of binary interaction coefficients.

    Returns:
        A 1D array of size ``1 + xn`` containing the temperature derivative followed by
        derivatives with respect to partial fractions.

    """
    nc = xn.size
    if nc == 1:
        return np.ones(1) * dalpha_dT(T, Tcs[0], omegas[0]) * acs[0]

    dadT = 0.0
    da = np.zeros(nc + 1)

    for i in range(nc):
        dTai = acs[i] * dalpha_dT(T, Tcs[i], omegas[i])
        ai = acs[i] * alpha(T, Tcs[i], omegas[i])

        for j in range(nc):
            dTaj = acs[j] * dalpha_dT(T, Tcs[j], omegas[j])
            aj = acs[j] * alpha(T, Tcs[j], omegas[j])

            dij = 1.0 - bips[i, j]
            saij = np.sqrt(ai * aj)

            da[i + 1] += xn[j] * saij * dij

            dadT += xn[i] * xn[j] / saij * (ai * dTaj + aj * dTai) * dij

    da *= 2.0
    da[0] = dadT / 2.0
    return da


@_COMPILER(
    [
        nb.f8[:](nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:, :]),
        nb.f8[:](
            nb.f8,
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 2, "C", readonly=True),
        ),
        nb.f8[:](
            nb.f8,
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=False),
            nb.types.Array(nb.f8, 1, "C", readonly=False),
            nb.types.Array(nb.f8, 1, "C", readonly=False),
            nb.types.Array(nb.f8, 2, "C", readonly=False),
        ),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def hess_a_VdW(
    T: float,
    xn: np.ndarray,
    Tcs: np.ndarray,
    omegas: np.ndarray,
    acs: np.ndarray,
    bips: np.ndarray,
) -> np.ndarray:
    """Hessian of Van der Waals cohesion for fluid mixtures with respect to
    temperature and partial fractions.

    Note:
        If there is only 1 component, the returned array contains the second derivative
        with respect to temperature.

    Parameters:
        T: Temperature.
        xn: Partial fractions per component.
        Tcs: Critical temperature per component.
        omegas: Acentric factor per component.
        acs: Critical cohesion per component.
        bip: Symmetric matrix of binary interaction coefficients.

    Returns:
        A compact form of the Hessian, consisting of the upper triangle including
        diagonal, flattened C-style (row-major) to a 1D array (Hessian is symmetric).

    """
    nc = xn.size
    if nc == 1:
        return np.ones(1) * ddalpha_dTT(T, Tcs[0], omegas[0]) * acs[0]

    ii = 1 + nc
    grad_dTa = np.zeros(ii)
    Hess_x = np.zeros((nc, nc))
    for i in range(nc):
        xi = xn[i]
        ai = acs[i] * alpha(T, Tcs[i], omegas[i])
        dTai = acs[i] * dalpha_dT(T, Tcs[i], omegas[i])
        dTTai = acs[i] * ddalpha_dTT(T, Tcs[i], omegas[i])
        for j in range(nc):
            dij = 1 - bips[i, j]
            xj = xn[j]
            aj = acs[j] * alpha(T, Tcs[j], omegas[j])
            dTaj = acs[j] * dalpha_dT(T, Tcs[j], omegas[j])
            dTTaj = acs[j] * ddalpha_dTT(T, Tcs[j], omegas[j])

            saij = np.sqrt(max(ai * aj, 1e-15))
            dTaij = ai * dTaj + dTai * aj
            # Contribution to dTT
            grad_dTa[0] += (
                xi
                * xj
                * dij
                / 2.0
                / saij
                * (
                    (2.0 * dTai * dTaj + ai * dTTaj + dTTai * aj)
                    - dTaij * dTaij / (saij * saij) / 2.0
                )
            )
            # Contribution to dxdT.
            grad_dTa[i + 1] += xj / saij * dij * dTaij
            # dxidxj
            if j >= i:
                Hess_x[i, j] = 2.0 * saij * dij

    # Hessian is symmetric, return only upper triangle (including diag).
    hess_arr = np.zeros(int((nc + 2) * (nc + 1) / 2))
    hess_arr[:ii] = grad_dTa
    hess_arr[ii:] = compact_dense_symmat(Hess_x)
    return hess_arr


@_COMPILER(
    [
        nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8[:], nb.f8[:]),
        nb.f8[:](
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=True),
        ),
        nb.f8[:](
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=False),
        ),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def lnphis(
    A: float,
    B: float,
    Z: float,
    p: float,
    T: float,
    dadx: np.ndarray,
    bcs: np.ndarray,
) -> np.ndarray:
    """Returns the logarithm of the fugacity coefficients per component.

    Contains some adjustments for numerical stability.

    Parameters:
        A: Dimensionless cohesion.
        B: Dimensionless covolume.
        Z: Compressibility factor.
        p: Pressure.
        T: Temperature.
        dadx: Derivative of the cohesion with respect to partial fractions. Must be
            of same size as ``bc``.
        bcs: Critical covolume per component.

    Returns:
        A 1D array of size ``bc`` containing the logarithms of the fugacity
        coefficients.

    """
    nc = bcs.size
    out = np.zeros(nc)
    RT = R_U * T
    Zm = Z - 1.0
    AB = A / np.sqrt(8) / B
    # Cap numerically for stability.
    lnZB0 = np.log(max(Z - B, 1e-15))
    lnZB1 = np.log(max((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B), 1e-15))

    # Special case: 1 component
    if nc == 1:
        phi = Zm - lnZB0 - AB * lnZB1
        return np.ones(1) * phi

    for i in range(nc):
        BiB = bcs[i] * p / RT / B
        dAdxi = dadx[i] * p / (RT * RT)
        out[i] = BiB * Zm - lnZB0 + AB * (BiB - dAdxi / A) * lnZB1

    return out


@_COMPILER(
    [
        nb.f8[:, :](nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8[:], nb.f8[:]),
        nb.f8[:, :](
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=True),
        ),
        nb.f8[:, :](
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=False),
        ),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def lnphis_jac(
    A: float,
    B: float,
    Z: float,
    p: float,
    T: float,
    dadx: np.ndarray,
    bcs: np.ndarray,
) -> np.ndarray:
    """Jacobian of :func:`lnphis` with respect to it's arguments.

    ``Z, A, B`` and especially ``dadx[i]`` are intermediate values per fugacity
    coefficient depending on the mixing rule and the EoS.

    Notes:
        1. The derivatives w.r.t. ``bc`` are not taken as this is assumed to be
           constant array.
        2. The derivatives w.r.t. ``dadx`` are performed only for ``dadx[i]`` in
           row ``i`` of the ``lnphis``. Otherwise the output array would be of shape
           ``(bc.size, 5 + bc.size)``.

    Parameters:
        A: Dimensionless cohesion.
        B: Dimensionless covolume.
        Z: Compressibility factor.
        p: Pressure.
        T: Temperature.
        dadx: Derivative of the cohesion with respect to partial fractions. Must be
            of same size as ``bc``.
        bcs: Critical covolume per component.

    Returns:
        A 2D array of size ``(bc.size, 6)`` containing the derivatives column-wise.

    """
    nc = bcs.size
    out = np.zeros((nc, 6))
    RT = R_U * T

    Z_m = Z - 1.0
    AB = A / np.sqrt(8) / B
    # Cap numerically for stability.
    ZB0 = max(Z - B, 1e-15)
    denom = Z + (1 - np.sqrt(2)) * B
    ZB1 = max((Z + (1 + np.sqrt(2)) * B) / denom, 1e-15)

    dZB1dZ = -2.0 * np.sqrt(2) * B / (denom * denom)
    dZB1dB = 2.0 * np.sqrt(2) * Z / (denom * denom)

    lnZB1 = np.log(ZB1)

    # Special case: 1 component
    if nc == 1:
        dZ = 1 - 1 / np.abs(ZB0) - AB / np.abs(ZB1) * dZB1dZ
        dA = -lnZB1 / np.sqrt(8) / B
        dB = 1 / np.abs(ZB0) + AB / B * lnZB1 - AB / np.abs(ZB1) * dZB1dB
        out[0, :3] = np.array((dA, dB, dZ))
        return out

    # Derivative row-wise per dadxi[i] is the same for all.
    ddadxi_ = -AB * lnZB1 / A * p / (RT * RT)

    for i in range(nc):
        dBiBdp = bcs[i] / RT / B
        BiB = dBiBdp * p
        dAdxip = dadx[i] / (RT * RT)
        dAdxi = dAdxip * p

        dZ = BiB - 1 / np.abs(ZB0) + AB * (BiB - dAdxi / A) / np.abs(ZB1) * dZB1dZ
        dA = (BiB - dAdxi / A) * lnZB1 / np.sqrt(8) / B + AB * lnZB1 * dAdxi / (A * A)
        dB = (
            -BiB / B * Z_m
            + 1 / np.abs(ZB0)
            - AB / B * (BiB - dAdxi / A) * lnZB1
            + AB * (lnZB1 * (-BiB / B) + (BiB - dAdxi / A) / np.abs(ZB1) * dZB1dB)
        )
        dp = dBiBdp * Z_m + AB * lnZB1 * (dBiBdp - dAdxip / A)
        dT = -(BiB * Z_m + AB * lnZB1 * (BiB - 2.0 * dAdxi / A)) / T
        out[i] = np.array((dA, dB, dZ, dp, dT, ddadxi_))

    return out


@_COMPILER(
    nb.f8(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def h_dep(
    A: float,
    B: float,
    Z: float,
    T: float,
    dAdT: float,
) -> float:
    """Computes the departure enthalpy.

    Parameters:
        A: Dimensionless cohesion.
        B: Dimensionless covolume.
        Z: Compressibility factor.
        T: Temperature.
        dAdT: Derivative of dimensionless cohesion with respect to temperature.

    Returns:
        The departure enthalpy.

    """
    RT = R_U * T
    ZB1 = max((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B), 1e-15)
    lnZB1 = np.log(ZB1)
    return RT * (Z - 1.0) + RT / np.sqrt(8) / B * (T * dAdT + A) * lnZB1


@_COMPILER(
    nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def grad_h_dep(
    A: float,
    B: float,
    Z: float,
    T: float,
    dAdT: float,
) -> np.ndarray:
    """Gradient of :func:`h_dep` with respect to its arguments."""
    RT = R_U * T
    denom = Z + (1 - np.sqrt(2)) * B
    ZB1 = max((Z + (1 + np.sqrt(2)) * B) / denom, 1e-15)

    dZB1dZ = -2.0 * np.sqrt(2) * B / (denom * denom)
    dZB1dB = 2.0 * np.sqrt(2) * Z / (denom * denom)

    lnZB1 = np.log(ZB1)

    dA = RT / np.sqrt(8) / B * lnZB1
    dB = (
        RT
        / np.sqrt(8)
        * (T * dAdT + A)
        * (1 / np.abs(ZB1) * dZB1dB / B - lnZB1 / (B * B))
    )
    dZ = RT + RT / np.sqrt(8) / B * (T * dAdT + A) / np.abs(ZB1) * dZB1dZ
    dT = R_U * (Z - 1.0) + R_U / np.sqrt(8) / B * lnZB1 * (T * dAdT + A + T * dAdT)
    ddAdT = RT / np.sqrt(8) / B * T * lnZB1

    return np.array((dA, dB, dZ, dT, ddAdT))


class CompiledPengRobinson(CompiledEoS):
    """Class providing compiled computations of thermodynamic quantities for the
    Peng-Robinson EoS.

    The parameter array for the pre-argument function can have up to 3 entries
    (see also :attr:`params`).

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
        ideal_fluids: Sequence[IdealFluid],
        bip_matrix: np.ndarray,
        params: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__(components)

        self.Tcs: np.ndarray = np.array(
            [c.critical_temperature for c in components]
        ).astype(np.float64)
        """Array of critical temperatures per component."""

        self.pcs: np.ndarray = np.array(
            [c.critical_pressure for c in components]
        ).astype(np.float64)
        """Array of critical pressures per component."""

        self.bcs: np.ndarray = np.array(
            [bc_component(p, T) for p, T in zip(self.pcs, self.Tcs)]
        )
        """Critical covolume values per component."""

        self.acs: np.ndarray = np.array(
            [ac_component(p, T) for p, T in zip(self.pcs, self.Tcs)]
        )
        """Critical cohesion values per component."""

        self.bips = (bip_matrix + bip_matrix.T) / 2.0
        """Symmetric 2D array of binary interaction parameters."""

        self.omegas = np.array([c.acentric_factor for c in components])
        """Array of acentric factors per component."""

        default_params: dict[str, float] = {
            "smoothing_multiphase": 1e-4,
            "smoothing_sc": 1e-3,
            "eps": 1e-14,
        }
        if params is None:
            params = {}
        default_params.update(params)

        self.params: dict[str, float] = default_params
        """Parameters for the equation of state.

        Once set, the parameters are not changable after compilation.

        List of parameters:

        - ``'smoothing_multiphase'`` : Portion of 2-phase region used for smoothing
          roots near phase borders.
        - ``'smoothing_sc'`` : Stripes in the super-critical area to smooth transitions
          from one extension formula to the other.
        - ``'eps'``: Numerical tolerance to determine zero (root case computation).

        Warning:
            Choosing the multiphase smoothing factor too big can alter the results of
            the phase separation calculation, making the multiphase region for example
            narrower or wider. Use only small values like ``1e-4``.

            Choosing a large smoothing factor for the super-critical transitions leads
            to larger areas, where the derivatives of the compressibility factor loose
            exactness. The order subsequent approximations may drop by at least.

        """

        assert len(ideal_fluids) == len(components), (
            "Require as many ideal fluids as components."
        )
        self.ideal_fluids: Sequence[IdealFluid] = ideal_fluids

        self._ideal_funcs: PropertyFunctionDict = {}
        """Contains ideal parts for thermodynamic properties."""

    def get_prearg_for_values(self) -> VectorFunction:
        eps = self.params["eps"]
        s_m = self.params["smoothing_multiphase"]
        s_sc = self.params["smoothing_sc"]

        Tcs = self.Tcs.copy()
        bcs = self.bcs.copy()
        acs = self.acs.copy()
        omegas = self.omegas.copy()
        bips = self.bips.copy()

        @_COMPILER(PREARGUMENT_FUNC_SIGNATURE)
        def prearg_val_c(
            phase_state: PhysicalState,
            p: float,
            T: float,
            xn: np.ndarray,
            params: np.ndarray,
        ) -> np.ndarray:
            nc = xn.size
            # Avoid redundant value storage if only 1 component.
            if nc == 1:
                nc = 0
            RT = R_U * T

            # Computing dimensionless cohesion and covolume.
            a = a_VdW(T, xn, Tcs, omegas, acs, bips)
            da = grad_a_VdW(T, xn, Tcs, omegas, acs, bips)
            b = np.dot(xn, bcs)
            A = a * p / (RT * RT)
            B = b * p / RT

            # Choose default parameters, and then parse given parameters.
            # Can only be done this way because params are a sub-array of the generic
            # argument.
            s_m_ = s_m
            eps_ = eps
            s_sc_ = s_sc
            if params.size >= 1:
                s_m_ = params[0]
            if params.size >= 2:
                s_sc_ = params[1]
            if params.size >= 3:
                eps_ = params[3]

            if phase_state == PhysicalState.gas:
                gaslike = True
            elif phase_state == PhysicalState.liquid:
                gaslike = False
            else:
                raise NotImplementedError(f"Unsupported phase state: {phase_state}.")

            # Contains A, B, Z, phase state, a, b, da/dt and da/dx
            prearg = np.zeros(7 + nc, dtype=np.float64)

            prearg[0] = float(phase_state.value)
            prearg[1] = A
            prearg[2] = B
            prearg[3] = get_compressibility_factor(A, B, gaslike, eps_, s_m_, s_sc_)
            prearg[4] = a
            prearg[5] = b
            prearg[-(1 + nc) :] = da

            return prearg

        return prearg_val_c

    def get_prearg_for_derivatives(self) -> VectorFunction:
        eps = self.params["eps"]
        s_m = self.params["smoothing_multiphase"]
        s_sc = self.params["smoothing_sc"]

        Tcs = self.Tcs.copy()
        bcs = self.bcs.copy()
        acs = self.acs.copy()
        omegas = self.omegas.copy()
        bips = self.bips.copy()

        @_COMPILER(PREARGUMENT_DFUNC_SIGNATURE)
        def prearg_jac_c(
            prearg_val: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
            params: np.ndarray,
        ) -> np.ndarray:
            nc = xn.size
            if nc == 1:
                nc = 0

            dn = 2 + nc
            RT = R_U * T

            s_m_ = s_m
            eps_ = eps
            s_sc_ = s_sc
            if params.size >= 1:
                s_m_ = params[0]
            if params.size >= 2:
                s_sc_ = params[1]
            if params.size >= 3:
                eps_ = params[3]

            phase_state = int(prearg_val[0])
            A = prearg_val[1]
            B = prearg_val[2]
            a = prearg_val[4]
            b = prearg_val[5]
            da = prearg_val[-(1 + nc) :]
            dA = da * p / (RT * RT)
            hess_a = hess_a_VdW(T, xn, Tcs, omegas, acs, bips)

            if phase_state == PhysicalState.gas.value:
                gaslike = True
            elif phase_state == PhysicalState.liquid.value:
                gaslike = False
            else:
                raise NotImplementedError(f"Unsupported phase state: {phase_state}")

            # Contains dA, dB, dZ and the compacted Hessian of a
            prearg_jac = np.zeros((3 * dn + hess_a.size,), dtype=np.float64)

            # Derivatives of A w.r.t. p, T, x.
            prearg_jac[0] = a / (RT * RT)
            prearg_jac[1] = dA[0] - a * p / RT / T
            if nc > 1:
                prearg_jac[2:dn] = dA[1:]
            # Derivatives of B w.r.t. p, T, x.
            prearg_jac[dn] = b / RT
            prearg_jac[dn + 1] = -b * p / RT / T
            if nc > 1:
                prearg_jac[dn + 2 : 2 * dn] = bcs * p / RT

            # Derivatives of Z w.r.t. p, T, x.
            dZ_ = get_compressibility_factor_derivatives(
                A, B, gaslike, eps_, s_m_, s_sc_
            )
            dZ = dZ_[0] * prearg_jac[:dn] + dZ_[1] * prearg_jac[dn : 2 * dn]
            prearg_jac[2 * dn : 3 * dn] = dZ
            prearg_jac[3 * dn :] = hess_a

            return prearg_jac

        return prearg_jac_c

    def get_fugacity_function(self) -> VectorFunction:
        bs = self.bcs.copy()

        @_COMPILER(FUGACITY_COEFF_FUNC_SIGNATURE)
        def phis_c(
            prearg: np.ndarray, p: float, T: float, xn: np.ndarray
        ) -> np.ndarray:
            nc = xn.size
            if nc == 1:
                nc = 0

            A = prearg[1]
            B = prearg[2]
            Z = prearg[3]
            if xn.size > 1:
                dadx = prearg[-xn.size :]
            else:
                dadx = np.ones(1)

            return lnphis(A, B, Z, p, T, dadx, bs)

        return phis_c

    def get_fugacity_derivative_function(self) -> VectorFunction:
        bs = self.bcs.copy()

        @_COMPILER(FUGACITY_COEFF_DERIVATIVE_FUNC_SIGNATURE)
        def dphi_mix_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            A = prearg_val[1]
            B = prearg_val[2]
            Z = prearg_val[3]
            if xn.size > 1:
                dn = 2 + xn.size
                dadx = prearg_val[-xn.size :]
                hess_x_a = compact_dense_symmat(prearg_jac[3 * dn :])[1:]
            else:
                dn = 2
                dadx = np.ones(1)
                hess_x_a = np.zeros((1, 1))

            dphis = np.zeros((xn.size, dn))

            dA = prearg_jac[0:dn]
            dB = prearg_jac[dn : 2 * dn]
            dZ = prearg_jac[2 * dn : 3 * dn]

            # Raw values, need expansion.
            dphis_ = lnphis_jac(A, B, Z, p, T, dadx, bs)
            for i in range(xn.size):
                dphis[i] += dphis_[i, 0] * dA
                dphis[i] += dphis_[i, 1] * dB
                dphis[i] += dphis_[i, 2] * dZ
                dphis[i, 0] += dphis_[i, 3]
                dphis[i, 1] += dphis_[i, 4]
                dphis[i, 1:] += dphis_[i, 5] * hess_x_a[i]

            return dphis

        return dphi_mix_c

    def get_enthalpy_function(self) -> ScalarFunction:
        h_id_c = self._ideal_funcs["h"]

        @_COMPILER(PROPERTY_FUNC_SIGNATURE)
        def h_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            nc = xn.size
            h_id = h_id_c(T, xn)
            A = prearg[1]
            B = prearg[2]
            Z = prearg[3]
            a = prearg[4]
            dadT = prearg[-(1 + nc)]

            RT = R_U * T
            dAdT = p * dadT / (RT * RT) - 2.0 * a * p / RT / T
            return h_id + h_dep(A, B, Z, T, dAdT)

        return h_c

    def get_enthalpy_derivative_function(self) -> VectorFunction:
        dh_id_c = self._ideal_funcs["dh"]

        @_COMPILER(PROPERTY_DERIVATIVE_FUNC_SIGNATURE)
        def dh_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            if xn.size > 1:
                nc = xn.size
            else:
                nc = 0

            dn = 2 + nc

            dh_id = dh_id_c(T, xn)

            A = prearg_val[1]
            B = prearg_val[2]
            Z = prearg_val[3]
            a = prearg_val[4]
            dadT = prearg_val[-(1 + nc)]

            RT = R_U * T
            dAdT = p * dadT / (RT * RT) - 2.0 * a * p / (RT * T)

            dA = prearg_jac[0:dn]
            dB = prearg_jac[dn : 2 * dn]
            dZ = prearg_jac[2 * dn : 3 * dn]
            grad_dadT = compact_dense_symmat(prearg_jac[3 * dn :])[0]
            grad_dAdT = p * grad_dadT / (RT * RT)
            grad_dAdT[0] -= 2.0 * a * p / (RT * T)

            dh_dep = grad_h_dep(A, B, Z, T, dAdT)
            dh = dh_dep[0] * dA + dh_dep[1] * dB + dh_dep[2] * dZ
            dh[1] += dh_dep[3]
            dh[1:] += dh_dep[4] * grad_dAdT

            # Contribution of ideal part to derivative w.r.t. T and x
            dh[1:] += dh_id
            return dh

        return dh_c

    def get_density_function(self) -> ScalarFunction:
        @_COMPILER(PROPERTY_FUNC_SIGNATURE)
        def rho_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            # Real density is ideal density divided by compressibility factor.
            return ideal_rho(p, T) / prearg[3]

        return rho_c

    def get_density_derivative_function(self) -> VectorFunction:
        @_COMPILER(PROPERTY_DERIVATIVE_FUNC_SIGNATURE)
        def drho_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            if xn.size > 1:
                dn = 2 + xn.size
            else:
                dn = 2

            Z = prearg_val[3]
            # derivative of Z w.r.t. p, T, xn
            dZ = prearg_jac[2 * dn : 3 * dn]
            # Chain rule.
            drho = -ideal_rho(p, T) / (Z * Z) * dZ
            # Contribution of ideal pT derivative
            drho[:2] += grad_ideal_rho(p, T)

            return drho

        return drho_c

    def compile(self):
        """Compiles the ideal part of the fluid properties before continuing to parent
        method."""
        if self.is_compiled:
            return

        logger.info("Compiling ideal property functions ..")

        h_ids: list[IdealProperty_T] = []
        dh_ids: list[IdealProperty_T] = []
        u_ids: list[IdealProperty_T] = []
        du_ids: list[IdealProperty_T] = []

        for f in self.ideal_fluids:
            f.compile()
            h_ids.append(f.funcs["h"])
            dh_ids.append(f.funcs["dh"])
            u_ids.append(f.funcs["u"])
            du_ids.append(f.funcs["du"])

        h_ids = tuple(h_ids)
        u_ids = tuple(u_ids)
        dh_ids = tuple(dh_ids)
        du_ids = tuple(du_ids)

        compiler_Tx = njit(nb.f8(nb.f8, nb.f8[:]))
        compiler_gradTx = njit(nb.f8[:](nb.f8, nb.f8[:]))

        logger.info("Compiling ideal mixture property functions ..")

        # region Compiling ideal property functions

        @compiler_Tx
        def h_ideal(T: float, xn: np.ndarray) -> float:
            """Ideal enthalpy of the mixture."""
            u = 0.0
            for i in range(xn.size):
                u += xn[i] * h_ids[i](T)
            return u

        @compiler_gradTx
        def dh_ideal(T: float, xn: np.ndarray) -> np.ndarray:
            """Gradient of mixture ideal enthalpy w.r.t. temperature and fractions."""
            if xn.size == 1:
                return np.ones(1) * dh_ids[0](T)
            dh = np.zeros(xn.size + 1)
            for i in range(xn.size):
                dh[0] += xn[i] * dh_ids[i](T)
                dh[i + 1] = h_ids[i](T)
            return dh

        @compiler_Tx
        def u_ideal(T: float, xn: np.ndarray) -> float:
            """Ideal enthalpy of the mixture."""
            u = 0.0
            for i in range(xn.size):
                u += xn[i] * u_ids[i](T)
            return u

        @compiler_gradTx
        def du_ideal(T: float, xn: np.ndarray) -> np.ndarray:
            """Gradient of mixture ideal enthalpy w.r.t. temperature and fractions."""
            if xn.size == 1:
                return np.ones(1) * du_ids[0](T)
            du = np.zeros(xn.size + 1)
            for i in range(xn.size):
                du[0] += xn[i] * du_ids[i](T)
                du[i + 1] = u_ids[i](T)
            return du

        # endregion

        self._ideal_funcs = {
            "h": h_ideal,
            "dh": dh_ideal,
            "u": u_ideal,
            "du": du_ideal,
            # Ideal density is same for all.
            "rho": self.ideal_fluids[0].funcs["rho"],
            "drho": self.ideal_fluids[0].funcs["drho"],
        }
        super().compile()
