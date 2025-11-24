"""This module contains compiled versions of the Peng-Robinson equation of state.

The functions provided here are building on lambdified expressions created using
:mod:`sympy` and then just-in-time compiled.

"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Sequence

import numba as nb
import numpy as np
import sympy as sp

from .._core import COMPOSITIONAL_VARIABLE_SYMBOLS as SYMBOLS
from .._core import NUMBA_CACHE, NUMBA_FAST_MATH, R_IDEAL_MOL, PhysicalState, njit
from ..compiled_eos import (
    FUGACITY_COEFF_DERIVATIVE_FUNC_SIGNATURE,
    FUGACITY_COEFF_FUNC_SIGNATURE,
    PREARGUMENT_DFUNC_SIGNATURE,
    PREARGUMENT_FUNC_SIGNATURE,
    PROPERTY_DERIVATIVE_FUNC_SIGNATURE,
    PROPERTY_FUNC_SIGNATURE,
    CompiledEoS,
    ScalarFunction,
    VectorFunction,
)
from ..materials import FluidComponent
from ..utils import safe_sum
from .compressibility_factor import (
    A_CRIT,
    B_CRIT,
    get_compressibility_factor,
    get_compressibility_factor_derivatives,
)

# Import explicitely to avoid some issues in numba (referencing vars internally).
from .utils import VanDerWaals_cohesion, VanDerWaals_covolume, thd_function_type

__all__ = [
    "SymbolicPengRobinson",
    "CompiledPengRobinson",
]


logger = logging.getLogger(__name__)

_COMPILER = njit
"""Decorator for compiling functions in this module.

Uses :func:`~porepy.compositional._core.njit`

"""


def _compile_fugacities(
    phis: Callable[[float, float, np.ndarray, float, float, float], np.ndarray],
) -> Callable[[float, float, np.ndarray, float, float, float], np.ndarray]:
    """Helper function to compile the vector of fugacity coefficients.

    It needs an additional reduction of shape from ``(num_comp, 1)`` to ``(num_comp,)``
    because of the usage of a symbolic, vector-valued function."""
    f = _COMPILER(phis)

    @_COMPILER(nb.f8[:](nb.f8, nb.f8, nb.f8[:], nb.f8, nb.f8, nb.f8))
    def inner(p_, T_, X_, A_, B_, Z_):
        phi_ = f(p_, T_, X_, A_, B_, Z_)
        return phi_[:, 0]

    return inner


def _compile_thd_function_derivatives(
    thd_df: Callable[[float, float, np.ndarray], list[float]],
) -> Callable[[float, float, np.ndarray], np.ndarray]:
    """Helper function to compile the gradient of a thermodynamic function.

    Functions are supposed to take pressure, temperature and a vector of
    fractions as arguments.

    This helper function ensures that the return value is wrapped in an array, and not
    a list (as by default returned when using sympy.lambdify).

    It also enforces a signature ``(float64, float64, float64[:]) -> float64[:]``

    """
    df = _COMPILER(thd_df, fastmath=NUMBA_FAST_MATH)

    @_COMPILER(nb.f8[:](nb.f8, nb.f8, nb.f8[:]), fastmath=NUMBA_FAST_MATH)
    def inner(p_, T_, X_):
        return np.array(df(p_, T_, X_), dtype=np.float64)

    return inner


def _compile_extended_thd_function_derivatives(
    ext_thd_df: Callable[[float, float, np.ndarray, float, float, float], list[float]],
) -> Callable[[float, float, np.ndarray, float, float, float], np.ndarray]:
    """Helper function to compile the gradient of an extended thermodynamic function.

    Functions are supposed to take pressure, temperature, a vector of
    fractions, and the EoS specific terms cohesion, covolume and
    compressibility factor as arguments.

    This helper function ensures that the return value is wrapped in an array, and not
    a list (as by default returned when using sympy.lambdify).

    It also enforces a signature
    ``(float64, float64, float64[:], float64, float64, float64) -> float64[:]``

    """
    df = _COMPILER(ext_thd_df)

    @_COMPILER(nb.f8[:](nb.f8, nb.f8, nb.f8[:], nb.f8, nb.f8, nb.f8))
    def inner(p_, T_, X_, A_, B_, Z_):
        return np.array(df(p_, T_, X_, A_, B_, Z_), dtype=np.float64)

    return inner


def _compile_density_derivative(
    dv: Callable[[float, float, float], list[float]],
) -> Callable[[float, float, float], np.ndarray]:
    """Helper function to compile the gradient of the density.

    Required to wrap the result in an array.

    It also enforces a signature ``(float64, float64, float64) -> float64[:]``.

    """

    dv_ = _COMPILER(fastmath=NUMBA_FAST_MATH)(dv)

    @_COMPILER(nb.f8[:](nb.f8, nb.f8, nb.f8))
    def inner(p_, T_, Z_):
        return np.array(dv_(p_, T_, Z_), dtype=np.float64)

    return inner


@_COMPILER(cache=True)
def _select(condlist: list, choicelist: list, default=np.nan):
    """Intermediate function to replace the ``numpy.select`` for scalar condition and
    choice input, because numba has a lot of issues resolving ``numpy.select``.

    See also:

        https://numba.readthedocs.io/en/0.59.1/reference/numpysupported.html

    """
    assert len(condlist) == len(choicelist) == 2, "Supported selection between two."
    if condlist[0]:
        return choicelist[0]
    else:
        return choicelist[1]


class SymbolicPengRobinson:
    """A class providing functions for thermodynamic properties using the Peng-Robinson
    EoS, based on a symbolic representation using :mod:`sympy`.

    Note:
        The functions are generated using :func:`sympy.lambdify` and are *sourceless*.

    Parameters:
        components: A sequence of ``num_comp`` components.
        ideal_enthalpies: A list of ``num_comp`` callables representing the ideal
            enthalpies of individual components in ``components``.
        bip_matrix: A 2D array containing BIPs for ``components``. Note that only the
            upper triangle of this matrix is used.

    """

    p_s: sp.Symbol = sp.Symbol(str(SYMBOLS["pressure"]))
    """Symbolic representation of pressure."""

    T_s: sp.Symbol = sp.Symbol(str(SYMBOLS["temperature"]))
    """Symbolic representation of temperature."""

    A_s: sp.Symbol = sp.Symbol("A")
    """Symbolic representation of the non-dimensional cohesion."""

    B_s: sp.Symbol = sp.Symbol("B")
    """Symbolic representation of the non-dimensional covolume."""

    Z_s: sp.Symbol = sp.Symbol("Z")
    """Symbolic representation of the compressibility factor."""

    def __init__(
        self,
        components: Sequence[FluidComponent],
        ideal_enthalpies: Sequence[thd_function_type],
        bip_matrix: np.ndarray,
    ) -> None:
        self.mixing_rule: str = "van-der-waals"
        """Name of mixing rule applied to covolume and cohesion for a fluid mixture."""

        self.x_s: list[sp.Symbol] = [
            sp.Symbol(f"{SYMBOLS['phase_composition']}_{comp.name}_j")
            for comp in components
        ]
        """Symbolic representation of fractions per component in ``components`` given at
        instantiation."""

        self.thd_arg: tuple[sp.Symbol, sp.Symbol, list[sp.Symbol]] = (
            self.p_s,
            self.T_s,
            self.x_s,
        )
        """General representation of the thermodynamic argument:

        1. a pressure value,
        2. a temperature value,
        3. an array of fractions per component.

        """

        self.ext_thd_arg = [self.p_s, self.T_s, self.x_s, self.A_s, self.B_s, self.Z_s]
        """Extended thermodynamic argument (see :attr:`thd_arg`).

        The extended arguments includes:

        4. mixed non-dimensional cohesion,
        5. mixed non-dimensional covolume,
        6. compressibility factor.

        The computation and dependencies have to be split by introducing additional
        dependencies due to their complexity (compilability and efficiency).

        """

        self.T_i_crit: list[float] = [comp.critical_temperature for comp in components]
        """List of critical temperatures per component."""

        self.p_i_crit: list[float] = [comp.critical_pressure for comp in components]
        """List of critical pressures per component."""

        self.b_i_crit: list[float] = [
            B_CRIT * (R_IDEAL_MOL * T_c) / p_c
            for T_c, p_c in zip(self.T_i_crit, self.p_i_crit)
        ]
        """List of critical covolumes per component.

        :math:`B_{c}R\\frac{T_{i,c}}{p_{i,c}}`, using :data:`B_CRIT`.

        """

        self.a_i_crit: list[float] = [
            A_CRIT * (R_IDEAL_MOL**2 * T_c**2) / p_c
            for T_c, p_c in zip(self.T_i_crit, self.p_i_crit)
        ]
        """List of critical cohesion values per component.

        :math:`A_c \\frac{R^2 T_{i,c}^2}{p_{i,c}}`, using :data:`A_CRIT`.

        """

        self.k_i: list[float] = [
            self.a_correction_weight(comp.acentric_factor) for comp in components
        ]
        """List of corrective weights for cohesion terms per components."""

        self.bip_matrix: np.ndarray = bip_matrix
        """Matrix of binary interaction parameters passed at instantiation."""

        self.ideal_enthalpies: Sequence[thd_function_type] = ideal_enthalpies
        """Sequence of callables representing ideal enthalpies per component, passed at
        instantiation."""

    @property
    def b(self) -> sp.Expr:
        """Covolume of the mixture according to the set mixing rule."""

        if self.mixing_rule == "van-der-waals":
            return VanDerWaals_covolume(self.x_s, self.b_i_crit)
        else:
            raise ValueError(f"Unknown mixing rule {self.mixing_rule}.")

    @property
    def B(self) -> sp.Expr:
        """Non-dimensional, mixed covolume created using :meth:`b`.

        :math:`\\frac{b p}{R T}`.

        """
        return self.b * self.p_s / (R_IDEAL_MOL * self.T_s)

    @property
    def grad_pTx_B(self) -> list[sp.Expr]:
        """Derivatives of :meth:`B` w.r.t. pressure, temperature and component
        fractions."""
        B = self.B
        return [B.diff(self.p_s), B.diff(self.T_s)] + [B.diff(x) for x in self.x_s]

    @property
    def B_func(self) -> Callable[[float, float, np.ndarray], float]:
        """Lambdified expression :meth:`B` returning the non-dimensional covolume
        for given values of pressure, temperature and component fractions."""
        return sp.lambdify(self.thd_arg, self.B)

    @property
    def grad_pTx_B_func(self) -> Callable[[float, float, np.ndarray], list[float]]:
        """Lambdified expression :meth:`grad_pTx_B` returning a list of floats of length
        ``2 + num_comp``, representing the derivatives w.r.t. pressure, temperature and
        component fractions."""
        return sp.lambdify(self.thd_arg, self.grad_pTx_B)

    @property
    def alphas(self) -> list[sp.Expr]:
        """Corrective terms for cohesion value such that
        :math:`a_{i} = \\alpha_i^2 a_{i,c}` for a component :math:`i`."""
        return [
            1 + k * (1 - sp.sqrt(self.T_s / T_ic))
            for k, T_ic in zip(self.k_i, self.T_i_crit)
        ]

    @property
    def a(self) -> sp.Expr:
        """Cohesion of the mixture according to the set mixing rule."""

        a_i: list[sp.Expr] = [
            a * alpha**2 for a, alpha in zip(self.a_i_crit, self.alphas)
        ]

        if self.mixing_rule == "van-der-waals":
            return VanDerWaals_cohesion(
                self.x_s, a_i, self.bip_matrix, sqrt_of_any=sp.sqrt
            )
        else:
            raise ValueError(f"Unknown mixing rule {self.mixing_rule}.")

    @property
    def A(self) -> sp.Expr:
        """Non-dimensional, mixed cohesion created using :attr:`a`.

        :math:`\\frac{a p}{R^2 T^2}`.

        """
        return self.a * self.p_s / (R_IDEAL_MOL**2 * self.T_s**2)

    @property
    def grad_pTx_A(self) -> list[sp.Expr]:
        """Derivatives of :meth:`A` w.r.t. pressure, temperature and component
        fractions."""
        A = self.A
        return [A.diff(self.p_s), A.diff(self.T_s)] + [A.diff(x) for x in self.x_s]

    @property
    def A_func(self) -> Callable[[float, float, np.ndarray], float]:
        """Lambdified expression :meth:`A` returning the non-dimensional cohesion
        for given values of pressure, temperature and component fractions."""
        return sp.lambdify(self.thd_arg, self.A)

    @property
    def grad_pTx_A_func(self) -> Callable[[float, float, np.ndarray], list[float]]:
        """Lambdified expression :meth:`grad_pTx_A` returning a list of floats of length
        ``2 + num_comp``, representing the derivatives w.r.t. pressure, temperature and
        component fractions."""
        return sp.lambdify(self.thd_arg, self.grad_pTx_A)

    @property
    def rho(self) -> sp.Expr:
        """Expression for density depending on pressure, temperature and compressibility
        factor.

        :math:`\\frac{p}{RTZ}`.

        """
        return self.p_s / (self.Z_s * self.T_s * R_IDEAL_MOL)

    @property
    def grad_pTZ_rho(self) -> list[sp.Expr]:
        """Expression for gradient of :meth:`rho` containing derivatives w.r.t. pressure
        temperature and compressibility factor.

        """
        rho = self.rho
        return [rho.diff(_) for _ in [self.p_s, self.T_s, self.Z_s]]

    @property
    def rho_func(self) -> Callable[[float, float, float], float]:
        """Lambdified expression :meth:`rho` returning density for given values of
        pressure, temperature and compressibility factor."""
        return sp.lambdify([self.p_s, self.T_s, self.Z_s], self.rho)

    @property
    def grad_pTZ_rho_func(self) -> Callable[[float, float, float], list[float]]:
        """Lambdified expression :meth:`grad_pTZ_rho` returning a list of floats of
        length 3, representing the derivatives w.r.t. pressure, temperature and
        compressibility factor."""
        return sp.lambdify([self.p_s, self.T_s, self.Z_s], self.grad_pTZ_rho)

    @property
    def h_ideal(self) -> sp.Expr:
        """Expression for the ideal enthalpy based on the provided
        :attr:`ideal_enthalpies` at instantiation."""
        return safe_sum(
            [x * h(self.T_s) for x, h in zip(self.x_s, self.ideal_enthalpies)]
        )

    @property
    def grad_pTx_h_ideal(self) -> list[sp.Expr]:
        """Derivatives of :meth:`h_ideal` w.r.t. pressure, temperature and component
        fractions."""
        h_ideal = self.h_ideal
        return [h_ideal.diff(_) for _ in [self.p_s, self.T_s] + self.x_s]

    @property
    def h_ideal_func(self) -> Callable[[float, float, np.ndarray], float]:
        """Lambdified expression :attr:`h_ideal` returning the ideal enthalpy for given
        values of pressure, temperature and component fractions."""
        return sp.lambdify(self.thd_arg, self.h_ideal)

    @property
    def grad_pTx_h_ideal_func(
        self,
    ) -> Callable[[float, float, np.ndarray], list[float]]:
        """Lambdified expression :meth:`grad_pTx_h_ideal` returning a list of floats of
        length ``2 + num_comp``, representing the derivatives w.r.t. pressure,
        temperature and component fractions."""
        return sp.lambdify(self.thd_arg, self.grad_pTx_h_ideal)

    @property
    def h_departure(self) -> sp.Expr:
        r"""The departure enthalpy using the Peng-Robinson EoS, depending on pressure,
        temperature, component fraction, non-dimensional cohesion and covolume, and
        compressibility factor.

        Note:
            Due to the complexity, this quantity requires cohesion, covolume and
            compressibility factor as intermediate values.
            Numba struggles to compile otherwise.

        :math:`RT(Z-1) + \frac{R}{\sqrt{8}B}(\frac{dA}{dT} T^2 + AT)
        \ln(\frac{Z + (1 + \sqrt{2})B}{Z + (1 - \sqrt{2})B})`

        """
        T = self.T_s
        A = self.A_s
        B = self.B_s
        Z = self.Z_s
        dA_dT = self.A.diff(self.T_s)
        return R_IDEAL_MOL * T * (Z - 1) + (R_IDEAL_MOL / np.sqrt(8)) * (
            dA_dT * T**2 + A * T
        ) / B * sp.ln(
            SymbolicPengRobinson._truncate(
                (Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B)
            )
        )

    @property
    def grad_pTxABZ_h_departure(self) -> list[sp.Expr]:
        """Derivatives of :meth:`h_departure` w.r.t. pressure, temperature, component
        fractions, non-dimensional cohesion and covolume, and compressibility factor."""
        h_dep = self.h_departure
        return [
            h_dep.diff(_)
            for _ in [self.p_s, self.T_s] + self.x_s + [self.A_s, self.B_s, self.Z_s]
        ]

    @property
    def h_departure_func(
        self,
    ) -> Callable[[float, float, np.ndarray, float, float, float], float]:
        """Lambdified expression :attr:`h_departure` returning the departure enthalpy
        for given values of pressure, temperature, component fractions, non-dimensional
        cohesion and covolume, and compressibility factor.

        See also:
            :attr:`ext_thd_arg`

        """
        return sp.lambdify(
            self.ext_thd_arg, self.h_departure, modules=[{"select": _select}, "numpy"]
        )

    @property
    def grad_pTxABZ_h_departure_func(
        self,
    ) -> Callable[[float, float, np.ndarray, float, float, float], list[float]]:
        """Lambdified expression :attr:`grad_pTxABZ_h_departure` returning a list of
        floats of length ``2 + num_comp + 3``, representing the derivatives w.r.t.
        pressure, temperature, component fractions, non-dimensional cohesion and
        covolume, and compressibility factor.

        See also:
            :attr:`ext_thd_arg`

        """
        return sp.lambdify(
            self.ext_thd_arg,
            self.grad_pTxABZ_h_departure,
            modules=[{"select": _select}, "numpy"],
        )

    @property
    def phis(self) -> sp.Matrix:
        """Vector of fugacity coefficients per component, depending on pressure,
        temperature, component fraction, non-dimensional cohesion and covolume, and
        compressibility factor.

        Note:
            Due to the complexity, this quantity requires cohesion, covolume and
            compressibility factor as intermediate values.
            Numba struggles to compile otherwise.

        """

        A = self.A_s
        B = self.B_s
        Z = self.Z_s

        ZB_term = (Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B)
        A_expr = self.A

        phi_i: list[sp.Expr] = []

        for i in range(len(self.x_s)):
            B_i = self.b_i_crit[i] * self.p_s / (R_IDEAL_MOL * self.T_s)
            dA_dXi = A_expr.diff(self.x_s[i])

            # TODO fix translation issue between numba and sympy
            # (involves translation ufunc.reduce for numba, which us used by sympy)
            # See https://numba.pydata.org/numba-doc/dev/reference/pysupported.html
            # initial argument required for reduce
            log_phi_i = (
                B_i / B * (Z - 1)
                # - sp.ln(PengRobinsonSymbolic._truncate(Z - B))
                - sp.ln(Z - B)
                + A
                / (B * np.sqrt(8))
                * (B_i / B - dA_dXi / A)
                # * sp.ln(PengRobinsonSymbolic._truncate(ZB_term))
                * sp.ln(ZB_term)
            )
            phi_i.append(sp.exp(SymbolicPengRobinson._cap(log_phi_i)))

        return sp.Matrix(phi_i)

    @property
    def lnphis(self) -> sp.Matrix:
        """Vector of logarithms of fugacity coefficients per component.
        Used for numerical stability.

        """

        B = self.B_s
        Z = self.Z_s

        ZB_term = sp.ln((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B))

        da_dx = self.da_dx

        lnphis: list[sp.Expr] = []

        for i in range(len(self.x_s)):
            lnphi_i = (
                self.b_i_crit[i] / self.b * (Z - 1)
                - sp.ln(Z - B)
                + self.a
                / (np.sqrt(8) * self.b * R_IDEAL_MOL * self.T_s)
                * (self.b_i_crit[i] / self.b - da_dx[i] / self.a)
                * ZB_term
            )
            lnphis.append(lnphi_i)

        return sp.Matrix(lnphis)

    @property
    def phis_func(
        self,
    ) -> Callable[[float, float, np.ndarray, float, float, float], np.ndarray]:
        """Lambdified expression :attr:`phis` returning the fugacity coefficients as
        a vector of length ``num_components``, for given values of pressure,
        temperature, component fractions, non-dimensional cohesion and covolume, and
        compressibility factor.

        See also:
            :attr:`ext_thd_arg`

        """
        return sp.lambdify(
            self.ext_thd_arg, self.phis, modules=[{"select": _select}, "numpy"]
        )

    @property
    def jac_phis(self) -> sp.Matrix:
        """The Jacobian of :meth:`phis` w.r.t. pressure,
        temperature, component fraction, non-dimensional cohesion and covolume, and
        compressibility factor."""
        return self.phis.jacobian(
            [self.p_s, self.T_s] + self.x_s + [self.A_s, self.B_s, self.Z_s]
        )

    @property
    def jac_phis_func(
        self,
    ) -> Callable[[float, float, np.ndarray, float, float, float], np.ndarray]:
        """Lambdified expression :attr:`jac_phis` returning a 2D array of shape
        ``(num_components, num_components + 5)``, containing the derivatives of fugacity
        coefficients w.r.t. pressure, temperature, component fractions, non-dimensional
        cohesion and covolume, and compressibility factor.

        See also:
            :attr:`ext_thd_arg`

        """
        return sp.lambdify(
            self.ext_thd_arg, self.jac_phis, modules=[{"select": _select}, "numpy"]
        )

    def _truncate(x: sp.Expr, eps: float = 1e-6) -> sp.Expr:
        """Truncated expression where the value of ``eps`` is chosen if the argument
        ``x`` becomes smaller than ``eps``."""
        return sp.Piecewise((x, x > eps), (eps, True))

    def _cap(x: sp.Expr, cap: float = 650) -> sp.Expr:
        """Capped expression where the value ``cap`` is chosen if the argument
        ``x`` becomes bigger than ``cap``."""
        return sp.Piecewise((x, x < cap), (cap, True))

    @staticmethod
    def a_correction_weight(omega: float) -> float:
        """Computes the cohesion correction weight based on the acentric factor.

        References:
            `Zhu et al. (2014), Appendix A
            <https://doi.org/10.1016/j.fluid.2014.07.003>`_

        Parameters:
            omega: Acentric factor for a component.

        Returns:
            Returns the cohesion correction parameter depending on a component's
            acentric factor.

        """
        if omega < 0.491:
            return 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        else:
            return (
                0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.016666 * omega**3
            )


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
    return B_CRIT * R_IDEAL_MOL * Tc / pc


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
    return A_CRIT * (R_IDEAL_MOL * Tc) ** 2 / pc**2


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
    return (1.0 + _k_of_omega(omega) * (1.0 - np.sqrt(Tr))) ** 2


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
    return k * (k + 1) / 2 / Tc**2 / sqrtTr**3


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
        a += xn[i] ** 2 * a_i
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
        return np.array((dalpha_dT(T, Tcs[0], omegas[0]) * acs[0]))

    dadT = 0.0
    da = np.zeros(nc + 1)

    for i in range(nc):
        dadT_i = dalpha_dT(T, Tcs[i], omegas[i]) * acs[i]
        a_i = alpha(T, Tcs[i], omegas[i]) * acs[i]

        for j in range(nc):
            dadT_j = dalpha_dT(T, Tcs[j], omegas[j]) * acs[j]
            a_j = alpha(T, Tcs[j], omegas[j]) * acs[j]

            dij = 1.0 - bips[i, j]
            sij = np.sqrt(a_i * a_j)

            da[i + 1] += xn[j] * sij * dij

            dadT += xn[i] * xn[j] / sij * (a_i * dadT_j + a_j * dadT_i) * dij

    da *= 2.0
    da[0] = dadT / 2.0
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
        return np.array((ddalpha_dTT(T, Tcs[0], omegas[0])))

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
                * (
                    (2.0 * dTai * dTaj + dTai * dTTaj + dTTai * dTaj) / saij
                    - dTaij / 2 / saij**3
                )
            )
            # Contribution to dxdT.
            grad_dTa[i + 1] += xj / saij * dij * dTaij
            # dxidxj
            if j >= i:
                Hess_x[i, j] = 2.0 * saij * dij

    # Hessian is symmetric, return only upper triangle (including diag).
    hess_arr = np.zeros(int(nc * (nc + 1) / 2 + 1))
    hess_arr[:ii] = grad_dTa
    hess_arr[ii:] = Hess_x[np.triu_indices(nc)]
    return hess_arr


@_COMPILER(nb.f8[:, :](nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=True)
def expand_compact_dense_sym_mat(mat_arr: np.ndarray) -> np.ndarray:
    """Restores a compacted square dense symmetric matrix.

    Parameters:
        mat_arr: ``shape=(n(n+1)/2,)``

            1D array containing the upper triangle part of the symmetric matrix, C-style
            flattened.

    Returns:
        A symmetric, dense matrix of shape ``(n, n)``, where the first row corresponds
        to the first ``n`` entries of ``mat_arr``, the second row starting with column 1
        to the next ``n-1`` entries of ``mat_arr`` and so on.

    """
    m = mat_arr.size
    n = (-1 + np.sqrt(1 + 8 * m)) / 2
    ni = int(n)
    if n != ni or ni < 0:
        raise ValueError("Could not determine square shape of restored matrix.")

    A = np.zeros((ni, ni))
    A[np.triu_indices(ni)] = mat_arr
    return (A + A.T) / 2


@_COMPILER(
    nb.f8[:](
        nb.f8,
        nb.f8,
        nb.f8,
        nb.f8,
        nb.f8,
        nb.f8[:],
        nb.f8[:],
    ),
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
    RT = R_IDEAL_MOL * T
    Zm = Z - 1.0
    AB = A / np.sqrt(8) / B
    # Cap numerically for stability.
    lnZB0 = np.log(max(Z - B, 1e-15))
    lnZB1 = np.log(max((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B), 1e-15))

    # Special case: 1 component
    if nc == 1:
        phi = Zm - lnZB0 - AB * lnZB1
        return np.array((phi))

    for i in range(nc):
        BiB = bcs[i] * p / RT / B
        dAdxi = dadx[i] * p / RT**2
        out[i] = BiB * Zm - lnZB0 + AB * (BiB - dAdxi / A) * lnZB1

    return out


@_COMPILER(
    nb.f8[:, :](
        nb.f8,
        nb.f8,
        nb.f8,
        nb.f8,
        nb.f8,
        nb.f8[:],
        nb.f8[:],
    ),
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
    RT = R_IDEAL_MOL * T

    Z_m = Z - 1.0
    AB = A / np.sqrt(8) / B
    # Cap numerically for stability.
    ZB0 = max(Z - B, 1e-15)
    denom = Z + (1 - np.sqrt(2)) * B
    ZB1 = max((Z + (1 + np.sqrt(2)) * B) / denom, 1e-15)

    dZB1dZ = 2.0 * (Z + B) / denom**2
    dZB1dB = (
        (1 + np.sqrt(2)) * denom + (Z + (1 + np.sqrt(2)) * B) * (1 - np.sqrt(2))
    ) / denom**2

    lnZB1 = np.log(ZB1)

    # Special case: 1 component
    if nc == 1:
        dZ = 1 - 1 / np.abs(ZB0) - AB / np.abs(ZB1) * dZB1dZ
        dA = -lnZB1 / np.sqrt(8) / B
        dB = 1 / np.abs(ZB0) + AB / B * lnZB1 - AB / np.abs(ZB1) * dZB1dB
        out[0, :3] = np.array((dA, dB, dZ))
        return out

    # Derivative row-wise per dadxi[i] is the same for all.
    ddadxi_ = -AB * lnZB1 / A * p / RT**2

    for i in range(nc):
        dBiBdp = bcs[i] / RT / B
        BiB = dBiBdp * p
        dAdxip = dadx[i] / RT**2
        dAdxi = dAdxip * p

        dZ = BiB - 1 / np.abs(ZB0) + AB * (BiB - dAdxi / A) / np.abs(ZB1) * dZB1dZ
        dA = (BiB - dAdxi / A) * lnZB1 / np.sqrt(8) / B + AB * lnZB1 * dAdxi / A**2
        dB = (
            -BiB / B * Z_m
            + 1 / np.abs(ZB0)
            - AB / B * (BiB - dAdxi / A) * lnZB1
            + AB * (lnZB1 * (-BiB / B) + (BiB - dAdxi / A) / np.abs(ZB1) * dZB1dB)
        )
        dp = dBiBdp * Z_m + AB * lnZB1 * (dBiBdp - dAdxip / A)
        dT = -BiB / T * Z_m + AB * lnZB1 * (-BiB / T - 2.0 * dAdxi / T / A)
        out[i] = np.array((dA, dB, dZ, dp, dT, ddadxi_))

    return out


class CompiledPengRobinson(CompiledEoS):
    """Class providing compiled computations of thermodynamic quantities for the
    Peng-Robinson EoS.

    The parameter array for the pre-argument function can have up to 3 entries
    (see also :attr:`params`):

    1. ``'smoothing_multiphase'`` : Portion of 2-phase region used for smoothing roots
       near phase borders
    2. ``'eps'``: Numerical tolerance to determine zero (root case computation).

    Warning:
        Choosing ``smoothing_multiphase`` too big (say 0.2), can move the borders
        between single and multiphase regions, leading to wrong results! Use with care
        and only small numbers e.g., ``1e-4``.

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
        ideal_enthalpies: Sequence[thd_function_type],
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
            "eps": 1e-14,
        }
        if params is None:
            params = {}
        default_params.update(params)

        self.params: dict[str, float] = default_params
        """Parameters for the equation of state.

        Once set, the parameters are not changable after compilation.

        List of parameters:

        - ``'eps'``: Numerical tolerance for zero. Applied in search for roots of the
          cubic polynomial.
        - ``'smoothing_multiphase'``: smoothing factor for compressibility factors in
          the multiphase regime when phases are about to dissapear. If zero, no
          smoothing is performed.

        """

    def get_prearg_for_values(self) -> VectorFunction:
        eps = self.params["eps"]
        s_m = self.params["smoothing_multiphase"]

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
            RT = R_IDEAL_MOL * T

            # Computing dimensionless cohesion and covolume.
            a = a_VdW(T, xn, Tcs, omegas, acs, bips)
            da = grad_a_VdW((T, xn, Tcs, omegas, acs, bips))
            b = np.dot(xn, bcs)
            A = a * p / RT**2
            B = b * p / RT

            # Choose default parameters, and then parse given parameters.
            # Can only be done this way because params are a sub-array of the generic
            # argument.
            s_m_ = s_m
            eps_ = eps
            if params.size >= 1:
                s_m_ = params[0]
            if params.size >= 2:
                eps_ = params[1]

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
            prearg[3] = get_compressibility_factor(A, B, gaslike, eps_, s_m_)
            prearg[4] = a
            prearg[5] = b
            prearg[-(1 + nc) :] = da

            return prearg

        return prearg_val_c

    def get_prearg_for_derivatives(self) -> VectorFunction:
        eps = self.params["eps"]
        s_m = self.params["smoothing_multiphase"]

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
            RT = R_IDEAL_MOL * T

            s_m_ = s_m
            eps_ = eps
            if params.size >= 1:
                s_m_ = params[0]
            if params.size >= 2:
                eps_ = params[1]

            phase_state = int(prearg_val[0])
            A = prearg_val[1]
            B = prearg_val[2]
            a = prearg_val[4]
            b = prearg_val[5]
            da = prearg_val[-(1 + nc) :]
            dA = da * p / RT**2
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
            prearg_jac[0] = a / RT**2
            prearg_jac[1] = dA[0] - a * p / RT / T
            if nc > 1:
                prearg_jac[2:dn] = dA[1:]
            # Derivatives of B w.r.t. p, T, x.
            prearg_jac[dn] = b / RT
            prearg_jac[dn + 1] = -b * p / RT / T
            if nc > 1:
                prearg_jac[dn + 2 : 2 * dn] = bcs * p / RT

            # Derivatives of Z w.r.t. p, T, x.
            dZ_ = get_compressibility_factor_derivatives(A, B, gaslike, eps_, s_m_)
            dZ = dZ_[0] * prearg_jac[:dn] + dZ_[1] * prearg_jac[dn : 2 * dn]
            prearg_jac[2 * dn : 3 * dn] = dZ
            prearg_jac[3 * dn :] = hess_a

            return prearg_jac

        return prearg_jac_c

    def get_fugacity_function(self) -> VectorFunction:
        bcs = self.bcs.copy()

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

            return lnphis(A, B, Z, p, T, dadx, bcs)

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
            nc = xn.size
            if nc == 1:
                nc = 0

            dn = 2 + nc
            dphis = np.zeros((nc, dn))

            A = prearg_val[1]
            B = prearg_val[2]
            Z = prearg_val[3]
            if xn.size > 1:
                dadx = prearg_val[-xn.size :]
            else:
                dadx = np.ones(1)

            dA = prearg_jac[0:dn]
            dB = prearg_jac[dn : 2 * dn]
            dZ = prearg_jac[2 * dn : 3 * dn]
            hess_a = expand_compact_dense_sym_mat(prearg_jac[3 * dn :])

            # Raw values, need expansion.
            dphis_ = lnphis_jac(A, B, Z, p, T, dadx, bs)
            for i in range(nc):
                dphis[i] += dphis_[0] * dA
                dphis[i] += dphis_[1] * dB
                dphis[i] += dphis_[2] * dZ
                dphis[i, 0] += dphis_[3]
                dphis[i, 1] += dphis_[4]
                dphis[i, 1:] += dphis_[5] * hess_a[i + 1]

            return dphis

        return dphi_mix_c

    def get_enthalpy_function(self) -> ScalarFunction:
        h_dep_c = self._cfuncs["h_dep"]
        h_ideal_c = self._cfuncs["h_ideal"]

        @_COMPILER(PROPERTY_FUNC_SIGNATURE)
        def h_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            return h_ideal_c(p, T, xn) + h_dep_c(
                p, T, xn, prearg[1], prearg[2], prearg[3]
            )

        return h_c

    def get_enthalpy_derivative_function(self) -> VectorFunction:
        d = 2 + self.nc
        dh_dep_c = self._cfuncs["dh_dep"]
        dh_ideal_c = self._cfuncs["dh_ideal"]

        @_COMPILER(PROPERTY_DERIVATIVE_FUNC_SIGNATURE)
        def dh_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            d_h_ideal = dh_ideal_c(p, T, xn)
            d_h_dep = dh_dep_c(p, T, xn, prearg_val[0], prearg_val[1], prearg_val[2])
            # derivatives of A_j, B_j, Z_j w.r.t. p, T, and X_j
            dA = prearg_jac[0:d]
            dB = prearg_jac[d : 2 * d]
            dZ = prearg_jac[2 * d : 3 * d]
            # expansion of derivatives of departure enthalpy (chain rule)
            d_h_dep = (
                d_h_dep[:-3] + d_h_dep[-3] * dA + d_h_dep[-2] * dB + d_h_dep[-1] * dZ
            )
            return d_h_ideal + d_h_dep

        return dh_c

    def get_density_function(self) -> ScalarFunction:
        @_COMPILER(PROPERTY_FUNC_SIGNATURE)
        def rho_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            denom = R_IDEAL_MOL * T * prearg[3]
            return p / denom

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
            denom = R_IDEAL_MOL * T * Z

            dp = 1 / denom
            dT = -p / denom / T
            dZ = -p / denom / Z

            # Contains derivatives w.r.t. pTx.
            drho = dZ * prearg_jac[2 * dn : 3 * dn]

            # Add outer contribution for dp and dT.
            drho[0] += dp
            drho[1] += dT
            return drho

        return drho_c
