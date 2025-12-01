"""Module implementing a partial EOS class which implements the Lohrenz-Bray-Clark
viscosity model.

To be used in combination with EoS classes implementing other properties, e.g., the
Peng-Robinson EOS.

Reference:

    1. Li, Z. et al.
       Fluid Property Model for Carbon Capture and Storage by Volume-Translated
       Peng-Robinson Equation of State and Lohrenz-Bray-Clark Viscosity Correlation.
       in (OnePetro, 2023). doi:10.2118/212584-MS.

"""

from __future__ import annotations

from typing import Sequence

import numba as nb
import numpy as np

from .._numba_interface import NUMBA_CACHE, NUMBA_FAST_MATH, njit
from ..compiled_eos import (
    PROPERTY_DERIVATIVE_FUNC_SIGNATURE,
    PROPERTY_FUNC_SIGNATURE,
    CompiledEoS,
    ScalarFunction,
    VectorFunction,
)
from ..materials import FluidComponent

__all__ = [
    "LBCViscosity",
]


_COMPILER = njit
"""Decorator for compiling functions in this module.

Uses :func:`~porepy.compositional._numba_interface.njit`

"""


@_COMPILER(
    nb.f8[:](nb.f8, nb.f8[:], nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _mu_pure(T: float, Tcs: np.ndarray, pcs: np.ndarray, mws: np.ndarray) -> np.ndarray:
    """Pure component viscosities at low pressure in Centipoise.

    Parameters:
        T: Temperature in [K].
        Tcs: ``shape=(n,)``

            Critical temperatures of components in [K].
        pcs: ``shape=(n,)``

            Critical pressures of components in [Pa].
        mws: ``shape=(n,)``
            Molar weights of components in [kg/mol].

    Returns:
        A ``(n,)`` array containing the pure component viscosities at the given T.

    """
    ncomp = Tcs.size
    mus = np.zeros(ncomp)

    srmws = np.sqrt(mws)
    Pcsatms = pcs / 101325  # Conversion from Pa to atm

    for i in range(ncomp):
        cpc = np.cbrt(Pcsatms[i])
        Tr = T / Tcs[i]
        d = np.power(Tcs[i], 1.0 / 6.0) / (cpc * cpc * srmws[i])
        if Tr < 1.5:
            n = 34e-5 * np.power(Tr, 0.94)
        else:
            n = 17.78e-5 * np.power(4.58 * Tr - 1.67, 5.0 / 8.0)

        mus[i] = n / d

    return mus


@_COMPILER(
    nb.f8[:](nb.f8, nb.f8[:], nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _dmu_pure_dT(
    T: float, Tcs: np.ndarray, pcs: np.ndarray, mws: np.ndarray
) -> np.ndarray:
    """Derivative pure component viscosities at low pressure w.r.t. temperature.

    Parameters:
        T: Temperature in [K].
        Tcs: ``shape=(n,)``

            Critical temperatures of components in [K].
        pcs: ``shape=(n,)``

            Critical pressures of components in [Pa].
        mws: ``shape=(n,)``
            Molar weights of components in [kg/mol].

    Returns:
        A ``(n,)`` array containing the temperature derivatives of the values returned
        by :func:`_mu_pure`.

    """
    ncomp = Tcs.size
    mus = np.zeros(ncomp)

    srmws = np.sqrt(mws)
    Pcsatms = pcs / 101325.0  # Conversion from Pa to atm

    for i in range(ncomp):
        cpc = np.cbrt(Pcsatms[i])
        Tr = T / Tcs[i]
        d = np.power(Tcs[i], 1.0 / 6.0) / (cpc * cpc * srmws[i])
        if Tr < 1.5:
            n = (34e-5 * 0.94 / np.power(Tr, 0.06)) / Tcs[i]
        else:
            n = (
                17.78e-5
                * 5.0
                / 8.0
                / np.power(4.58 * Tr - 1.67, 3.0 / 8.0)
                * 4.58
                / Tcs[i]
            )

        mus[i] = n / d

    return mus


@_COMPILER(
    nb.f8(nb.f8[:], nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _mu_zero(xn: np.ndarray, mus: np.ndarray, mws: np.ndarray) -> float:
    """Mixture viscosity at low pressure..

    Parameters:
        xn: ``shape=(n,)``

            Mole fractions per components in [-].
        mus: ``shape=(n,)``

            Viscosities of pure components at the given T in some unit.
        mws: ``shape=(n,)``

            Molar weights of components in [kg/mol].

    Returns:
        The mixture viscosity value in the unit of ``mus``.

    """
    n = xn * np.sqrt(mws)
    return np.sum(n * mus) / np.sum(n)


@_COMPILER(
    nb.f8[:](nb.f8[:], nb.f8[:], nb.f8[:, :], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _dmu_zero(
    xn: np.ndarray, mus: np.ndarray, dmus: np.ndarray, mws: np.ndarray
) -> np.ndarray:
    """Derivative of the mixture viscosity at low pressure.

    Derivatives are performed with respect to fractions, while other derivatives are
    assumed to be part of ``dmus`` (i.e., pressure, temperature derivatives).

    Parameters:
        xn: ``shape=(n,)``

            Mole fractions per components in [-].
        mus: ``shape=(n,)``

            Viscosities of pure components at the given T in [Pa s].
        dmus: ``shape=(n, 2)``

            Pressure and temperature derivatives of pure component viscosities.
        mws: ``shape=(n,)``

            Molar weights of components in [kg/mol].

    Returns:
        A ``(2 + n,``)`` array containing the derivatives with respect to pressure,
        temperature and the fractions.

    """

    sqrtmws = np.sqrt(mws)
    ncomp = xn.size
    n = xn * sqrtmws

    dpt = np.zeros(2)
    for i in range(ncomp):
        dpt += n[i] * dmus[i, :]
    dpt /= np.sum(n)

    u = np.sum(n * mus)
    v = np.sum(n)
    du = sqrtmws * mus
    dv = sqrtmws

    dx = (du * v - u * dv) / (v * v)

    return np.hstack((dpt, dx))


@_COMPILER(
    nb.f8(nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _xi(xn: np.ndarray, Tcs: np.ndarray, pcs: np.ndarray, mws: np.ndarray) -> float:
    """Dimensionless density parameter.

    Parameters:
        xn: ``shape=(n,)``

            Mole fractions per components in [-].
        Tcs: ``shape=(n,)``

            Critical temperatures of components in [K].
        pcs: ``shape=(n,)``

            Critical pressures of components in [Pa].
        mws: ``shape=(n,)``

            Molar weights of components in [kg/mol].

    Returns:
        The dimensionless density parameter.

    """
    Pcsatms = pcs / 101325  # Conversion from Pa to atm
    n = np.power(np.sum(xn * Tcs), 1.0 / 6.0)
    cpc = np.cbrt(np.sum(xn * Pcsatms))
    d = np.sqrt(np.sum(xn * mws)) * cpc * cpc
    return n / d


@_COMPILER(
    nb.f8[:](nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _dxi(
    xn: np.ndarray, Tcs: np.ndarray, pcs: np.ndarray, mws: np.ndarray
) -> np.ndarray:
    """Derivative of the dimensionless density parameter with respect to fractions.

    Parameters:
        xn: ``shape=(n,)``

            Mole fractions per components in [-].
        Tcs: ``shape=(n,)``

            Critical temperatures of components in [K].
        pcs: ``shape=(n,)``

            Critical pressures of components in [Pa].
        mws: ``shape=(n,)``

            Molar weights of components in [kg/mol].

    Returns:
        A ``(n,)`` array containing the derivatives with respect to fractions.

    """
    Pcsatms = pcs / 101325.0  # Conversion from Pa to atm
    n = np.power(np.sum(xn * Tcs), 1.0 / 6.0)
    d1 = np.sqrt(np.sum(xn * mws))
    d2 = np.cbrt(np.sum(xn * Pcsatms))
    d = d1 * d2 * d2

    dn = Tcs / (6 * np.power(np.sum(xn * Tcs), 5.0 / 6.0))
    dd = 0.5 / d1 * mws * d2 * d2 + d1 * (2.0 / 3.0) / d2 * Pcsatms

    return (dn * d - n * dd) / (d * d)


@_COMPILER(
    nb.f8(nb.f8, nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _reduced_pseudo_density(
    rho: float,
    xn: np.ndarray,
    vcs: np.ndarray,
) -> float:
    """Reduced pseudo-density using a mixing rule to obtain pseudo-critical values for
    the specific volume of the mixture.

    Parameters:
        rho: Density in [mol / m^3].
        xn: ``shape=(n,)``

            Mole fractions per components in [-].
        vcs: ``shape=(n,)``

            Critical specific volumes of components in [m^3 / mol].

    Returns:
        The reduced pseudo-density in [-].

    """
    return rho * np.sum(xn * vcs)


@_COMPILER(
    nb.f8[:](nb.f8, nb.f8[:], nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _d_reduced_pseudo_density(
    rho: float,
    drho: np.ndarray,
    xn: np.ndarray,
    vcs: np.ndarray,
) -> float:
    """Derivative of the reduced pseudo-critical density with respect to the derivatives
    contained in the density derivative, and some additional terms from the
    pseudo-critical approximation.

    Parameters:
        rho: Density in [mol / m^3].
        drho: ``shape(2 + n,)``

            Derivatives of the density with respect to pressure, temperature and
            fractions.
        xn: ``shape=(n,)``

            Mole fractions per components in [-].
        vcs: ``shape=(n,)``

            Critical specific volumes of components in [m^3 / mol].
    """
    drho_r = drho * np.sum(xn * vcs)
    drho_r[2:] += rho * vcs
    return drho_r


@_COMPILER(
    nb.f8(nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def _mu_correction(
    rho: float,
    xn: np.ndarray,
    Tcs: np.ndarray,
    pcs: np.ndarray,
    vcs: np.ndarray,
    mws: np.ndarray,
) -> float:
    """Density correction term for viscosity.

    Parameters:
        rho: Mixture density in [mol / m^3].
        xn: ``shape=(n,)``

            Mole fractions per components in [-].
        Tcs: ``shape=(n,)``

            Critical temperatures of components in [K].
        pcs: ``shape=(n,)``

            Critical pressures of components in [Pa].
        vcs: ``shape=(n,)``

            Critical specific volumes of components in [m^3 / mol].

        mws: ``shape=(n,)``

            Molar weights of components in [kg/mol].

    Returns:
        The density correction term in Centipoise, using some pseudo-critical
        approximation of the reduced density

    """
    rho_r = _reduced_pseudo_density(rho, xn, vcs)
    xi = _xi(xn, Tcs, pcs, mws)
    rrho_r = rho_r * rho_r
    rp = (
        0.1023
        + 0.023364 * rho_r
        + 0.058533 * rrho_r
        - 0.040758 * rrho_r * rho_r
        + 0.0093324 * rrho_r * rrho_r
    )
    n = rp * rp * rp * rp - 0.0001
    return n / xi


@_COMPILER(
    nb.f8[:](nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def _dmu_correction(
    rho: float,
    drho: np.ndarray,
    xn: np.ndarray,
    Tcs: np.ndarray,
    pcs: np.ndarray,
    vcs: np.ndarray,
    mws: np.ndarray,
) -> np.ndarray:
    """pTx derivatives of density correction term for viscosity.

    Parameters:
        rho: Mixture density in [mol / m^3].
        drho: ``shape=(2 + n,)``

            Derivative of density with respect to pressure, temperature and fractions.
        xn: ``shape=(n,)``

            Mole fractions per components in [-].
        Tcs: ``shape=(n,)``

            Critical temperatures of components in [K].
        pcs: ``shape=(n,)``

            Critical pressures of components in [Pa].
        vcs: ``shape=(n,)``

            Critical specific volumes of components in [m^3 / mol].

        mws: ``shape=(n,)``

            Molar weights of components in [kg/mol].

    Returns:
        The derivative of the density correction term taking the derivatives of the
        density into account and adding additional terms according
        :func:`_mu_correction`.

    """

    rho_r = _reduced_pseudo_density(rho, xn, vcs)
    xi = _xi(xn, Tcs, pcs, mws)

    drho_r = _d_reduced_pseudo_density(rho, drho, xn, vcs)
    dxi = np.zeros(2 + xn.size)
    dxi[2:] = _dxi(xn, Tcs, pcs, mws)

    rrho_r = rho_r * rho_r
    k = (
        0.1023
        + 0.023364 * rho_r
        + 0.058533 * rrho_r
        - 0.040758 * rrho_r * rho_r
        + 0.0093324 * rrho_r * rrho_r
    )
    dk = (
        0.023364
        + 2 * 0.058533 * rho_r
        - 3 * 0.040758 * rrho_r
        + 4 * 0.0093324 * rrho_r * rho_r
    )
    kk = k * k
    n = kk * kk - 0.0001
    dn = 4 * kk * k * dk * drho_r

    return (dn * xi - n * dxi) / (xi * xi)


class LBCViscosity(CompiledEoS):
    """Partial EOS class implementing the Lohrenz-Bray-Clark viscosity model,
    returning viscosity and its derivatives with respect to pressure, temperature and
    fractions.

    Viscosity is returned in [Pa s].

    """

    def __init__(self, components: Sequence[FluidComponent], *args, **kwargs) -> None:
        super().__init__(components, *args, **kwargs)

        self.Tcs: np.ndarray = np.array(
            [c.critical_temperature for c in components]
        ).astype(np.float64)
        """Array of critical temperatures per component."""

        self.pcs: np.ndarray = np.array(
            [c.critical_pressure for c in components]
        ).astype(np.float64)
        """Array of critical pressures per component."""

        self.vcs: np.ndarray = np.array(
            [c.critical_specific_volume for c in components]
        ).astype(np.float64)
        """Array of critical specific volume per component."""

        self.mws: np.ndarray = np.array([c.molar_mass for c in components]).astype(
            np.float64
        )
        """Array of molar masses per component."""

    def get_mu_function(self) -> ScalarFunction:
        mws = self.mws.copy()
        Tcs = self.Tcs.copy()
        pcs = self.pcs.copy()
        vcs = self.vcs.copy()

        if "rho" in self.funcs:
            rho_c = self.funcs["rho"]
        else:
            rho_c = self.get_rho_function()

        @_COMPILER(PROPERTY_FUNC_SIGNATURE)
        def mu_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            # Copy to create local object, simplifies compilation, remains in memory.
            _mws = mws.copy()
            _Tcs = Tcs.copy()
            _vcs = vcs.copy()
            _pcs = pcs.copy()
            mus_pure = _mu_pure(T, _Tcs, _pcs, _mws)
            mu_zero = _mu_zero(xn, mus_pure, _mws)

            mu_correction = _mu_correction(
                rho_c(prearg, p, T, xn), xn, _Tcs, _pcs, _vcs, _mws
            )

            mu_val = mu_zero + mu_correction
            # Centipoise to Pa s
            mu_val *= 1e-3
            return mu_val

        return mu_c

    def get_grad_mu_function(self) -> VectorFunction:
        mws = self.mws.copy()
        Tcs = self.Tcs.copy()
        pcs = self.pcs.copy()
        vcs = self.vcs.copy()

        if "rho" in self.funcs:
            rho_c = self.funcs["rho"]
        else:
            rho_c = self.get_rho_function()

        if "drho" in self.funcs:
            drho_c = self.funcs["drho"]
        else:
            drho_c = self.get_grad_rho_function()

        @_COMPILER(PROPERTY_DERIVATIVE_FUNC_SIGNATURE)
        def dmu_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            _mws = mws.copy()
            _Tcs = Tcs.copy()
            _vcs = vcs.copy()
            _pcs = pcs.copy()

            mus_pure = _mu_pure(T, _Tcs, _pcs, _mws)
            dmus_pure_dpT = np.zeros((mus_pure.size, 2))
            dmus_pure_dpT[:, 1] = _dmu_pure_dT(T, _Tcs, _pcs, _mws)

            dmu_zero = _dmu_zero(xn, mus_pure, dmus_pure_dpT, _mws)
            dmu_correction = _dmu_correction(
                rho_c(prearg_val, p, T, xn),
                drho_c(prearg_val, prearg_jac, p, T, xn),
                xn,
                _Tcs,
                _pcs,
                _vcs,
                _mws,
            )

            dmu = dmu_zero + dmu_correction
            # Centipoise to Pa s
            dmu *= 1e-3
            return dmu

        return dmu_c
