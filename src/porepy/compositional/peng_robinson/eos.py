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
    "compact_dense_symmat",
    "a_VdW",
    "grad_a_VdW",
    "hess_a_VdW",
    "a_dl",
    "grad_a_dl",
    "hess_a_dl",
    "b_dl",
    "grad_b_dl",
    "CompiledPengRobinson",
]


logger = logging.getLogger(__name__)

_COMPILER = njit
"""Decorator for compiling functions in this module.

Uses :func:`~porepy.compositional._numba_interface.njit`.

"""


@_COMPILER(nb.f8(nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def covolume_dep(Z: float, B: float) -> float:
    r"""Special treatment of departure term to remain well-defined.

    Parameters:
        Z: Compressibility factor.
        B: Dimensionless covolume.

    Returns:
        :math:`\ln{\frac{Z +  (1 + \sqrt{2}) B}{Z +  (1 + \sqrt{2}) B}}`.
        If the log-argument goes below ``c``, ``c`` is chosen,

    """
    tol = 1e-14
    _s2 = np.sqrt(2)
    _cn = 1 + _s2
    _cd = 1 - _s2
    ZB1 = max((Z + _cn * B) / (Z + _cd * B), tol)
    return np.log1p(ZB1 - 1.0)


@_COMPILER(nb.f8[:](nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def grad_covolume_dep(Z: float, B: float) -> float:
    r"""Gradient of :func:`_lnZB1`.

    Parameters:
        Z: Compressibility factor.
        B: Dimensionless covolume.

    Returns:
        Derivatives w.r.t. Z and B.

    """
    tol = 1e-14
    _s2 = np.sqrt(2)
    _cn = 1 + _s2
    _cd = 1 - _s2

    denom = Z + _cd * B
    denom2 = denom**2
    ZB1 = max((Z + _cn * B) / denom, tol)

    dZB1dZ = -2.0 * _s2 * B / denom2
    dZB1dB = 2.0 * _s2 * Z / denom2
    return np.array((dZB1dZ, dZB1dB)) / np.abs(ZB1)


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
            out[i, i] *= 0.5
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


@_COMPILER(nb.f8(nb.f8, nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def b_dl(b: float, p: float, T: float) -> float:
    """Computes the dimensionless covolume.

    Parameters:
        a: Covolume.
        p: Pressure.
        T: Temperature.

    Returns:
        :math:`\\frac{a p}{R T}`.

    """
    return b * p / (R_U * T)


@_COMPILER(
    nb.f8[:](nb.f8[:], nb.f8, nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=True
)
def grad_b_dl(grad_b: np.ndarray, b: float, p: float, T: float) -> np.ndarray:
    """Expands the gradient of the cohesion to the gradient to the dimensionless
    cohesion by chainrule.

    Note:
        If there is only 1 component, i.e. ``grad_b`` contains only the (constant)
        cohesion of the component, the resulting gradient is of shape ``(2,)``,
        containing only pressure- and temperature derivative.

    Parameters:
        grad_b: Gradient of covolume. Expecting only derivatives w.r.t. partial
            fractions.
        b: Covolume.
        p: Pressure.
        T: Temperature.

    Returns:
        A 1D array of size ``2 + grad_b.size``, pre-appending the
        pressure- and temperature-derivative.

    """
    RT = R_U * T
    dBdpT = np.array((b / RT, -b * p / (RT * T)))
    if grad_b.size == 1:
        return dBdpT
    else:
        return np.hstack((dBdpT, grad_b * p / RT))


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
    return A_CRIT * RT**2 / pc**2


@_COMPILER(nb.f8(nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def _k_of_omega(omega: float) -> float:
    """Returns the weight depending on the acentric factor, which is used in
    :func:`alpha` and its derivatives."""
    if omega < 0.491:
        return 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    else:
        return 0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.016666 * omega**3


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
    return salpha**2


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
    return k * (k + 1) / (2.0 * Tc**2 * sqrtTr**3)


@_COMPILER(
    nb.f8(nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:, :]),
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
    nb.f8[:](nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:, :]),
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
    da[0] = dadT * 0.5
    return da


@_COMPILER(
    nb.f8[:](nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:, :]),
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
                * 0.5
                / saij
                * (
                    (2.0 * dTai * dTaj + ai * dTTaj + dTTai * aj)
                    - 0.5 * (dTaij / saij) ** 2
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


@_COMPILER(nb.f8(nb.f8, nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def a_dl(a: float, p: float, T: float) -> float:
    """Computes the dimensionless cohesion.

    Parameters:
        a: Cohesion.
        p: Pressure.
        T: Temperature.

    Returns:
        :math:`\\frac{a p}{(R T)^2}`.

    """
    iR = 1.0 / R_U**2
    return iR * a * p / T**2


@_COMPILER(
    nb.f8[:](nb.f8[:], nb.f8, nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=True
)
def grad_a_dl(grad_a: np.ndarray, a: float, p: float, T: float) -> np.ndarray:
    """Expands the gradient of the cohesion to the gradient to the dimensionless
    cohesion by chainrule.

    Note:
        If there is only 1 component, i.e. ``grad_a`` contains only the temperature
        derivative, the resulting gradient is of shape ``(2,)``, containing only
        pressure- and temperature derivative.

    Parameters:
        grad_a: Gradient of cohesion. Expecting temperature derivative and possibly
            derivatives w.r.t. partial fractions.
        a: cohesion
        p: Pressure.
        T: Temperature.

    Returns:
        A 1D array of size ``1 + grad_a.size``, pre-appending the
        pressure-derivative.

    """
    RT2 = R_U**2 * T**2

    dAdp = a / RT2
    dAdTx = grad_a * p / RT2
    dAdTx[0] -= 2.0 * a * p / (RT2 * T)
    if grad_a.size == 1:
        return np.array((dAdp, dAdTx[0]))
    else:
        return np.hstack((np.ones(1) * dAdp, dAdTx))


@_COMPILER(
    nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def hess_a_dl(
    hess_a: np.ndarray, grad_a: np.ndarray, a: float, p: float, T: float
) -> np.ndarray:
    """Expands the Hessian of the cohesion to the Hessian of the dimensionless
    cohesion.

    Note:
        If there is only 1 component (``hess_a.shape=(1,)`` and ``grad_a.shape=(1,)``),
        the returned array contains the Hessian w.r.t. pressure and temperature.

    Parameters:
        hess_a: Hessian of cohesion in compact form.
        grad_a: Gradient of cohesion.
        a: cohesion
        p: Pressure.
        T: Temperature.

    Returns:
        A compact form of the Hessian, consisting of the upper triangle including
        diagonal, flattened C-style (row-major) to a 1D array (Hessian is symmetric).

    """
    RT2 = R_U**2 * T**2
    nc = grad_a.size - 1  # Should contain only 1 derivative if only 1 component.

    # dpp is zero, linear in pressure.
    dp_gradA = np.zeros(grad_a.size + 1)
    dp_gradA[1:] = grad_a / RT2
    dp_gradA[1] -= 2.0 * a / (RT2 * T)
    # dA / dTdT
    dTT = (hess_a[0] - 4.0 * grad_a[0] / T + 6.0 * a / T**2) * p / RT2

    if nc == 0:
        return np.array((dp_gradA[0], dp_gradA[1], dTT))

    # Otherwise the Hessian of A can be split in d gradA / dp (first row) and the
    # Hessian of dimensional a, scaled by the factor p/RT2.
    # The second row, d gradA / dT, needs to account for factor.
    hess_Tx_A = hess_a * p / RT2
    hess_Tx_A[0] = dTT
    hess_Tx_A[1 : 1 + nc] = (hess_a[1 : 1 + nc] - 2.0 * grad_a[1:] / T) * p / RT2

    return np.hstack((dp_gradA, hess_Tx_A))


@_COMPILER(
    nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def lnphis(
    A: float,
    B: float,
    Z: float,
    dAdx: np.ndarray,
    Bis: np.ndarray,
) -> np.ndarray:
    """Returns the logarithm of the fugacity coefficients per component.

    Contains some adjustments for numerical stability.

    Parameters:
        A: Dimensionless cohesion.
        B: Dimensionless covolume.
        Z: Compressibility factor.
        dAdx: Derivative of the cohesion with respect to partial fractions.
        Bis: Dimensionless covolume per component. Must be of same size as ``dAdx``.

    Returns:
        A 1D array of size ``dAdx`` containing the logarithms of the fugacity
        coefficients.

    """
    nc = Bis.size
    Zm = Z - 1.0
    AB = A / (np.sqrt(8) * B)
    # Cap numerically for stability.
    lnZB0 = np.log(max(Z - B, 1e-14))
    lnZB1 = covolume_dep(Z, B)

    # Special case: 1 component
    if nc == 1:
        phi = Zm - lnZB0 - AB * lnZB1
        return np.ones(1) * phi

    out = np.zeros(nc)

    for i in range(nc):
        out[i] = Bis[i] / B * Zm - lnZB0 - AB * (dAdx[i] / A - Bis[i] / B) * lnZB1

    return out


@_COMPILER(
    nb.f8[:, :](nb.f8, nb.f8, nb.f8, nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def lnphis_jac(
    A: float,
    B: float,
    Z: float,
    dAdx: np.ndarray,
    Bis: np.ndarray,
) -> np.ndarray:
    """Jacobian of :func:`lnphis` with respect to it's arguments.

    ``Z, A, B`` and especially ``dadx[i]`` are intermediate values per fugacity
    coefficient depending on the mixing rule and the EoS.

    Notes:
        The derivatives w.r.t. ``dAdx`` are performed only for ``dAdx[i]`` in
        row ``i`` of the ``lnphis``. Same for ``Bis``.

    Parameters:
        A: Dimensionless cohesion.
        B: Dimensionless covolume.
        Z: Compressibility factor.
        dAdx: Derivative of the cohesion with respect to partial fractions.
        Bis: Dimensionless covolume per component. Must be of same size as ``dAdx``.

    Returns:
        A 2D array of size ``(dAdx.size, 5)`` containing the derivatives column-wise.

    """
    nc = Bis.size

    Zm = Z - 1.0
    # Cap numerically for stability.
    ZB0 = max(Z - B, 1e-15)
    dlnZB0 = np.array((1, -1)) / np.abs(ZB0)
    lnZB1 = covolume_dep(Z, B)
    dlnZB1 = grad_covolume_dep(Z, B)
    sB = np.sqrt(8) * B
    AB = A / sB

    out = np.zeros((nc, 5))

    # Special case: 1 component
    if nc == 1:
        out[0, 0] = -lnZB1 / sB
        out[0, 1] = -dlnZB0[1] + AB / B * lnZB1 - AB * dlnZB1[1]
        out[0, 2] = 1 - dlnZB0[0] - AB * dlnZB1[0]
        return out

    # Derivative row-wise per dAdxi[i] is the same for all.
    ddAdx = -AB * lnZB1 / A
    # Derivative row-wise per Bis[i] is also the same for all.
    dBis = Zm / B + AB * lnZB1 / B

    for i in range(nc):
        D = dAdx[i] / A - Bis[i] / B
        dBi = -Bis[i] / B**2
        out[i, 0] = (AB * dAdx[i] / A**2 - D / sB) * lnZB1
        out[i, 1] = (
            dBi * Zm - dlnZB0[1] - AB * (D * dlnZB1[1] - lnZB1 * dBi - lnZB1 * D / B)
        )
        out[i, 2] = Bis[i] / B - dlnZB0[0] - AB * D * dlnZB1[0]
        out[i, 3] = ddAdx
        out[i, 4] = dBis

    return out


@_COMPILER(
    nb.f8(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def u_dep(
    A: float,
    B: float,
    Z: float,
    T: float,
    dAdT: float,
) -> float:
    """Computes the departure internal energy.

    Parameters:
        A: Dimensionless cohesion.
        B: Dimensionless covolume.
        Z: Compressibility factor.
        T: Temperature.
        dAdT: Derivative of dimensionless cohesion with respect to temperature.

    Returns:
        The departure internal energy.

    """
    _c = -R_U / np.sqrt(8)
    lnZB1 = covolume_dep(Z, B)
    iB = 1.0 / B
    return _c * T * (A + T * dAdT) * lnZB1 * iB


@_COMPILER(
    nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def grad_u_dep(
    A: float,
    B: float,
    Z: float,
    T: float,
    dAdT: float,
) -> np.ndarray:
    """Gradient of :func:`u_dep` with respect to its arguments."""
    _c = -R_U / np.sqrt(8)
    lnZB1 = covolume_dep(Z, B)
    dlnZB1 = grad_covolume_dep(Z, B)

    cAA = _c * (A + T * dAdT)
    iB = 1.0 / B

    dA = _c * T * lnZB1 * iB
    dB = cAA * T * (dlnZB1[1] - lnZB1 * iB) * iB
    dZ = cAA * T * dlnZB1[0] * iB
    dT = _c * (A + 2.0 * T * dAdT) * lnZB1 * iB
    ddAdT = T * dA

    return np.array((dA, dB, dZ, dT, ddAdT))


class CompiledPengRobinson(CompiledEoS):
    """Class providing compiled computations of thermodynamic quantities for the
    Peng-Robinson EoS.

    The parameter array for the pre-argument function can have up to 3 entries
    (see also :attr:`params`).

    Important:
        All properties implemented here are molar quantities.

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
            # Avoid redundant value storage if only 1 component.
            if xn.size == 1:
                dn = 2
            else:
                dn = 2 + xn.size

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
                eps_ = params[2]

            # Copying turns arrays into function locals, making compilation and
            # signatures easier.
            Tcs_ = Tcs.copy()
            acs_ = acs.copy()
            bcs_ = bcs.copy()
            omegas_ = omegas.copy()
            bips_ = bips.copy()

            # Computing cohesion, covolume, compressibility factor.
            a = a_VdW(T, xn, Tcs_, omegas_, acs_, bips_)
            grad_a = grad_a_VdW(T, xn, Tcs_, omegas_, acs_, bips_)
            b = np.sum(xn * bcs_)

            A = a_dl(a, p, T)
            B = b_dl(b, p, T)
            Z = get_compressibility_factor(
                A,
                B,
                True if phase_state == PhysicalState.gas else False,
                eps_,
                s_m_,
                s_sc_,
            )
            grad_A = grad_a_dl(grad_a, a, p, T)

            prearg = np.zeros(6 + dn, dtype=np.float64)

            prearg[0] = float(phase_state.value)
            prearg[1] = A
            prearg[2] = B
            prearg[3] = Z
            prearg[4] = a
            prearg[5] = b
            prearg[-dn:] = grad_A

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
            if xn.size == 1:
                dn = 2
            else:
                dn = 2 + xn.size

            s_m_ = s_m
            eps_ = eps
            s_sc_ = s_sc
            if params.size >= 1:
                s_m_ = params[0]
            if params.size >= 2:
                s_sc_ = params[1]
            if params.size >= 3:
                eps_ = params[2]

            Tcs_ = Tcs.copy()
            acs_ = acs.copy()
            bcs_ = bcs.copy()
            omegas_ = omegas.copy()
            bips_ = bips.copy()

            phase_state = int(prearg_val[0])
            A = prearg_val[1]
            B = prearg_val[2]
            a = prearg_val[4]
            b = prearg_val[5]
            grad_a = grad_a_VdW(T, xn, Tcs_, omegas_, acs_, bips_)
            hess_a = hess_a_VdW(T, xn, Tcs_, omegas_, acs_, bips_)
            hess_A = hess_a_dl(hess_a, grad_a, a, p, T)

            grad_A = prearg_val[-dn:]
            grad_B = grad_b_dl(bcs_, b, p, T)
            dZ = get_compressibility_factor_derivatives(
                A,
                B,
                True if phase_state == PhysicalState.gas.value else False,
                eps_,
                s_m_,
                s_sc_,
            )
            grad_Z = dZ[0] * grad_A + dZ[1] * grad_B

            prearg_jac = np.zeros((2 * dn + hess_A.size,), dtype=np.float64)

            prearg_jac[:dn] = grad_B
            prearg_jac[dn : 2 * dn] = grad_Z
            prearg_jac[2 * dn :] = hess_A

            return prearg_jac

        return prearg_jac_c

    def get_lnphis_function(self) -> VectorFunction:
        bs = self.bcs.copy()

        @_COMPILER(FUGACITY_COEFF_FUNC_SIGNATURE)
        def phis_c(
            prearg: np.ndarray, p: float, T: float, xn: np.ndarray
        ) -> np.ndarray:
            if xn.size == 1:
                dn = 2
            else:
                dn = 2 + xn.size

            A = prearg[1]
            B = prearg[2]
            Z = prearg[3]
            Bis = bs.copy() * p / (R_U * T)
            grad_A = prearg[-dn:]
            if xn.size > 1:
                dAdx = grad_A[2:]
            else:
                dAdx = np.ones(1)

            return lnphis(A, B, Z, dAdx, Bis)

        return phis_c

    def get_grad_lnphis_function(self) -> VectorFunction:
        bs = self.bcs.copy()

        @_COMPILER(FUGACITY_COEFF_DERIVATIVE_FUNC_SIGNATURE)
        def dphi_mix_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            bs_ = bs.copy()

            A = prearg_val[1]
            B = prearg_val[2]
            Z = prearg_val[3]
            if xn.size == 1:
                dn = 2
                dAdx = np.ones(1)
                hess_A = np.zeros((1, 1))
            else:
                dn = 2 + xn.size

            dphis = np.zeros((xn.size, dn))

            grad_A = prearg_val[-dn:]
            grad_B = prearg_jac[:dn]
            grad_Z = prearg_jac[dn : 2 * dn]
            if dn > 1:
                hess_A = compact_dense_symmat(prearg_jac[2 * dn :])
                dAdx = grad_A[2:]

            RT = R_U * T
            Bis = bs_.copy() * p / RT

            # Raw values, need expansion.
            dphis_ = lnphis_jac(A, B, Z, dAdx, Bis)
            for i in range(xn.size):
                grad_Bi = np.zeros(dn)
                grad_Bi[0] = bs_[i] / RT
                grad_Bi[1] = -bs_[i] * p / (RT * T)

                if dn > 2:
                    grad_dAx = hess_A[2 + i]
                else:
                    grad_dAx = np.zeros(dn)

                dphis[i] = (
                    dphis_[i, 0] * grad_A
                    + dphis_[i, 1] * grad_B
                    + dphis_[i, 2] * grad_Z
                    + dphis_[i, 3] * grad_dAx
                    + dphis_[i, 4] * grad_Bi
                )

            return dphis

        return dphi_mix_c

    def get_h_function(self) -> ScalarFunction:
        h_id_c = self._ideal_funcs["h"]

        @_COMPILER(PROPERTY_FUNC_SIGNATURE)
        def h_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            if xn.size == 1:
                dn = 2
            else:
                dn = 2 + xn.size
            A = prearg[1]
            B = prearg[2]
            Z = prearg[3]
            grad_A = prearg[-dn:]

            RTZ = R_U * T * (Z - 1.0)

            return h_id_c(T, xn) + u_dep(A, B, Z, T, grad_A[1]) + RTZ

        return h_c

    def get_grad_h_function(self) -> VectorFunction:
        dh_id_c = self._ideal_funcs["dh"]

        @_COMPILER(PROPERTY_DERIVATIVE_FUNC_SIGNATURE)
        def dh_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            if xn.size == 1:
                dn = 2
            else:
                dn = 2 + xn.size

            A = prearg_val[1]
            B = prearg_val[2]
            Z = prearg_val[3]

            grad_A = prearg_val[-dn:]
            grad_B = prearg_jac[:dn]
            grad_Z = prearg_jac[dn : 2 * dn]
            hess_A = compact_dense_symmat(prearg_jac[2 * dn :])

            du_dep_ = grad_u_dep(A, B, Z, T, grad_A[1])
            du_dep = (
                du_dep_[0] * grad_A
                + du_dep_[1] * grad_B
                + du_dep_[2] * grad_Z
                # grad(dAdT)
                + du_dep_[4] * hess_A[1]
            )
            du_dep[1] += du_dep_[3]
            # Contribution of ideal part to derivative w.r.t. T and x
            du_dep[1:] += dh_id_c(T, xn)

            dRTZ = T * grad_Z
            dRTZ[1] += Z - 1.0
            return du_dep + R_U * dRTZ

        return dh_c

    def get_u_function(self) -> ScalarFunction:
        u_id_c = self._ideal_funcs["u"]

        @_COMPILER(PROPERTY_FUNC_SIGNATURE)
        def u_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            if xn.size == 1:
                dn = 2
            else:
                dn = 2 + xn.size
            A = prearg[1]
            B = prearg[2]
            Z = prearg[3]
            grad_A = prearg[-dn:]

            return u_id_c(T, xn) + u_dep(A, B, Z, T, grad_A[1])

        return u_c

    def get_grad_u_function(self) -> VectorFunction:
        du_id_c = self._ideal_funcs["du"]

        @_COMPILER(PROPERTY_DERIVATIVE_FUNC_SIGNATURE)
        def du_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            if xn.size == 1:
                dn = 2
            else:
                dn = 2 + xn.size

            A = prearg_val[1]
            B = prearg_val[2]
            Z = prearg_val[3]

            grad_A = prearg_val[-dn:]
            grad_B = prearg_jac[:dn]
            grad_Z = prearg_jac[dn : 2 * dn]
            hess_A = compact_dense_symmat(prearg_jac[2 * dn :])

            du_dep_ = grad_u_dep(A, B, Z, T, grad_A[1])
            du_dep = (
                du_dep_[0] * grad_A
                + du_dep_[1] * grad_B
                + du_dep_[2] * grad_Z
                # grad(dAdT)
                + du_dep_[4] * hess_A[1]
            )
            du_dep[1] += du_dep_[3]
            # Contribution of ideal part to derivative w.r.t. T and x
            du_dep[1:] += du_id_c(T, xn)

            return du_dep

        return du_c

    def get_rho_function(self) -> ScalarFunction:
        @_COMPILER(PROPERTY_FUNC_SIGNATURE)
        def rho_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            # Real density is ideal density divided by compressibility factor.
            return ideal_rho(p, T) / prearg[3]

        return rho_c

    def get_grad_rho_function(self) -> VectorFunction:
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
            dZ = prearg_jac[dn : 2 * dn]
            # Chain rule.
            drho = -ideal_rho(p, T) / Z**2 * dZ
            # Contribution of ideal pT derivative
            drho[:2] += grad_ideal_rho(p, T) / Z

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

        # NOTE Convert to tuple, otherwise numba cannot access the functions properly.
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
