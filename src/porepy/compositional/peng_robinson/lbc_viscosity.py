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

from .._core import NUMBA_CACHE, NUMBA_FAST_MATH, njit
from ..compiled_flash.eos_compiler import EoSCompiler, ScalarFunction, VectorFunction
from ..materials import FluidComponent

__all__ = [
    "LBCViscosity",
]


_COMPILER = njit
"""Decorator for compiling functions in this module.

Uses :func:`~porepy.compositional._core.njit`

"""


@_COMPILER(
    [
        nb.f8[:](nb.f8, nb.f8[:], nb.f8[:], nb.f8[:]),
        nb.f8[:](
            nb.f8,
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
        ),
    ],
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
        Tr = T / Tcs[i]
        d = Tcs[i] ** (1 / 6) / (np.cbrt(Pcsatms[i] ** 2) * srmws[i])
        if Tr < 1.5:
            n = 34e-5 * Tr**0.94
        else:
            n = 17.78e-5 * (4.58 * Tr - 1.67) ** (5 / 8)

        mus[i] = n / d

    return mus


@_COMPILER(
    [
        nb.f8[:](nb.f8, nb.f8[:], nb.f8[:], nb.f8[:]),
        nb.f8[:](
            nb.f8,
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
        ),
    ],
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
    Pcsatms = pcs / 101325  # Conversion from Pa to atm

    for i in range(ncomp):
        Tr = T / Tcs[i]
        d = Tcs[i] ** (1 / 6) / (np.cbrt(Pcsatms[i] ** 2) * srmws[i])
        if Tr < 1.5:
            n = (34e-5 * 0.94 / Tr**0.06) / Tcs[i]
        else:
            n = 17.78e-5 * 5 / 8 / (4.58 * Tr - 1.67) ** (3 / 8) * 4.58 / Tcs[i]

        mus[i] = n / d

    return mus


@_COMPILER(
    [
        nb.f8(nb.f8[:], nb.f8[:], nb.f8[:]),
        nb.f8(nb.f8[:], nb.f8[:], nb.types.Array(nb.f8, 1, "C", readonly=True)),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _mu_zero(x: np.ndarray, mus: np.ndarray, mws: np.ndarray) -> float:
    """Mixture viscosity at low pressure..

    Parameters:
        x: ``shape=(n,)``

            Mole fractions per components in [-].
        mus: ``shape=(n,)``

            Viscosities of pure components at the given T in some unit.
        mws: ``shape=(n,)``

            Molar weights of components in [kg/mol].

    Returns:
        The mixture viscosity value in the unit of ``mus``.

    """
    n = x * np.sqrt(mws)
    return np.sum(n * mus) / np.sum(n)


@_COMPILER(
    [
        nb.f8[:](nb.f8[:], nb.f8[:], nb.f8[:, :], nb.f8[:]),
        nb.f8[:](
            nb.f8[:],
            nb.f8[:],
            nb.f8[:, :],
            nb.types.Array(nb.f8, 1, "C", readonly=True),
        ),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _dmu_zero(
    x: np.ndarray, mus: np.ndarray, dmus: np.ndarray, mws: np.ndarray
) -> np.ndarray:
    """Derivative of the mixture viscosity at low pressure.

    Derivatives are performed with respect to fractions, while other derivatives are
    assumed to be part of ``dmus`` (i.e., pressure, temperature derivatives).

    Parameters:
        x: ``shape=(n,)``

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

    ncomp = x.size
    n = x * np.sqrt(mws)

    dpt = np.zeros(2)
    for i in range(ncomp):
        dpt += n[i] * dmus[i, :]
    dpt /= np.sum(n)

    dx = (mus * np.sqrt(mws) * n - np.sum(n * mus) * np.sqrt(mws)) / (np.sum(n) ** 2)

    return np.hstack((dpt, dx))


@_COMPILER(
    [
        nb.f8(nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
        nb.f8(
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
        ),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _xi(x: np.ndarray, Tcs: np.ndarray, pcs: np.ndarray, mws: np.ndarray) -> float:
    """Dimensionless density parameter.

    Parameters:
        x: ``shape=(n,)``

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
    n = np.sum(x * Tcs) ** (1 / 6)
    d = np.sqrt(np.sum(x * mws)) * np.cbrt(np.sum(x * Pcsatms) ** 2)
    return n / d


@_COMPILER(
    [
        nb.f8[:](nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
        nb.f8[:](
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
        ),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _dxi(
    x: np.ndarray, Tcs: np.ndarray, pcs: np.ndarray, mws: np.ndarray
) -> np.ndarray:
    """Derivative of the dimensionless density parameter with respect to fractions.

    Parameters:
        x: ``shape=(n,)``

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
    Pcsatms = pcs / 101325  # Conversion from Pa to atm
    n = np.sum(x * Tcs) ** (1 / 6)
    d1 = np.sqrt(np.sum(x * mws))
    d2 = np.cbrt(np.sum(x * Pcsatms))
    d = d1 * d2**2

    dn = (1 / 6) / np.sum(x * Tcs) ** (5 / 6) * Tcs
    dd = 0.5 / d1 * d2**2 * mws + (2 / 3) * d1 / d2**2 * Pcsatms

    return (dn * d - n * dd) / (d**2)


@_COMPILER(
    [
        nb.f8(nb.f8, nb.f8[:], nb.f8[:]),
        nb.f8(nb.f8, nb.f8[:], nb.types.Array(nb.f8, 1, "C", readonly=True)),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _reduced_pseudo_density(
    rho: float,
    x: np.ndarray,
    vcs: np.ndarray,
) -> float:
    """Reduced pseudo-density using a mixing rule to obtain pseudo-critical values for
    the specific volume of the mixture.

    Parameters:
        rho: Density in [mol / m^3].
        x: ``shape=(n,)``

            Mole fractions per components in [-].
        vcs: ``shape=(n,)``

            Critical specific volumes of components in [m^3 / mol].

    Returns:
        The reduced pseudo-density in [-].

    """
    return rho * np.sum(x * vcs)


@_COMPILER(
    [
        nb.f8[:](nb.f8, nb.f8[:], nb.f8[:], nb.f8[:]),
        nb.f8[:](
            nb.f8, nb.f8[:], nb.f8[:], nb.types.Array(nb.f8, 1, "C", readonly=True)
        ),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _d_reduced_pseudo_density(
    rho: float,
    drho: np.ndarray,
    x: np.ndarray,
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
        x: ``shape=(n,)``

            Mole fractions per components in [-].
        vcs: ``shape=(n,)``

            Critical specific volumes of components in [m^3 / mol].
    """
    drho_r = drho * np.sum(x * vcs)
    drho_r[2:] += rho * vcs
    return drho_r


@_COMPILER(
    [
        nb.f8(nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
        nb.f8(
            nb.f8,
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
        ),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def _mu_correction(
    rho: float,
    x: np.ndarray,
    Tcs: np.ndarray,
    pcs: np.ndarray,
    vcs: np.ndarray,
    mws: np.ndarray,
) -> float:
    """Density correction term for viscosity.

    Parameters:
        rho: Mixture density in [mol / m^3].
        x: ``shape=(n,)``

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
    rho_r = _reduced_pseudo_density(rho, x, vcs)
    xi = _xi(x, Tcs, pcs, mws)
    n = (
        0.1023
        + 0.023364 * rho_r
        + 0.058533 * rho_r**2
        - 0.040758 * rho_r**3
        + 0.0093324 * rho_r**4
    ) ** 4 - 0.0001
    return n / xi


@_COMPILER(
    [
        nb.f8[:](nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
        nb.f8[:](
            nb.f8,
            nb.f8[:],
            nb.f8[:],
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
            nb.types.Array(nb.f8, 1, "C", readonly=True),
        ),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def _dmu_correction(
    rho: float,
    drho: np.ndarray,
    x: np.ndarray,
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
        x: ``shape=(n,)``

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

    rho_r = _reduced_pseudo_density(rho, x, vcs)
    xi = _xi(x, Tcs, pcs, mws)

    drho_r = _d_reduced_pseudo_density(rho, drho, x, vcs)
    dxi = np.zeros(2 + x.size)
    dxi[2:] = _dxi(x, Tcs, pcs, mws)

    k = (
        0.1023
        + 0.023364 * rho_r
        + 0.058533 * rho_r**2
        - 0.040758 * rho_r**3
        + 0.0093324 * rho_r**4
    )
    n = k**4 - 0.0001
    dn = (
        4
        * k**3
        * (
            0.023364
            + 2 * 0.058533 * rho_r
            - 3 * 0.040758 * rho_r**2
            + 4 * 0.0093324 * rho_r**3
        )
        * drho_r
    )

    return (dn * xi - n * dxi) / (xi**2)


class LBCViscosity(EoSCompiler):
    """Partial EOS class implementing the Lohrenz-Bray-Clark viscosity model,
    returning viscosity and its derivatives with respect to pressure, temperature and
    fractions.

    Viscosity is returned in [Pa s].

    """

    def __init__(self, components: Sequence[FluidComponent], *args, **kwargs) -> None:
        super().__init__(components, *args, **kwargs)

        self._mws = np.array([c.molar_mass for c in components])
        """Molar weight per component in [kg/mol]."""
        self._tc = np.array([c.critical_temperature for c in components])
        """Critical temperature per component in [K]."""
        self._pc = np.array([c.critical_pressure for c in components])
        """Critical pressure per component in [Pa]."""
        self._vc = np.array([c.critical_specific_volume for c in components])
        """Critical specific volume per component in [m^3/mol]."""

    def get_viscosity_function(self) -> ScalarFunction:
        mws = self._mws.copy()
        tc = self._tc.copy()
        pc = self._pc.copy()
        vc = self._vc.copy()

        if "rho" in self.funcs:
            rho_c = self.funcs["rho"]
        else:
            rho_c = self.get_density_function()

        @_COMPILER(nb.f8(nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def mu_c(prearg: np.ndarray, p: float, T: float, x: np.ndarray) -> float:
            mus_pure = _mu_pure(T, tc, pc, mws)
            mu_zero = _mu_zero(x, mus_pure, mws)

            mu_correction = _mu_correction(rho_c(prearg, p, T, x), x, tc, pc, vc, mws)

            mu_val = mu_zero + mu_correction
            # Centipoise to Pa s
            mu_val *= 1e-3
            return mu_val

        return mu_c

    def get_viscosity_derivative_function(self) -> VectorFunction:
        mws = self._mws.copy()
        tc = self._tc.copy()
        pc = self._pc.copy()
        vc = self._vc.copy()

        if "rho" in self.funcs:
            rho_c = self.funcs["rho"]
        else:
            rho_c = self.get_density_function()

        if "drho" in self.funcs:
            drho_c = self.funcs["drho"]
        else:
            drho_c = self.get_density_derivative_function()

        @_COMPILER(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def dmu_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            mus_pure = _mu_pure(T, tc, pc, mws)
            dmus_pure_dpT = np.zeros((mus_pure.size, 2))
            dmus_pure_dpT[:, 1] = _dmu_pure_dT(T, tc, pc, mws)

            dmu_zero = _dmu_zero(xn, mus_pure, dmus_pure_dpT, mws)
            dmu_correction = _dmu_correction(
                rho_c(prearg_val, p, T, xn),
                drho_c(prearg_val, prearg_jac, p, T, xn),
                xn,
                tc,
                pc,
                vc,
                mws,
            )

            dmu = dmu_zero + dmu_correction
            # Centipoise to Pa s
            dmu *= 1e-3
            return dmu

        return dmu_c
