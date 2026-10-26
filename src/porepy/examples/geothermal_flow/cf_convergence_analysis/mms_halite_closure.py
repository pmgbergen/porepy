"""Analytic closure of the manufactured-solution test: liquid brine and immobile halite.

The relations are defined once symbolically; the values and exact derivatives
used by the solver are generated from them with ``sympy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp

from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_halite_constants import (
    SYMBOLS,
    HaliteMMSConstants,
    MMS_HALITE,
)


# Symbols. The primary variables of the model are (p, h, z) on every subdomain;
# s is the halite saturation, primitive in the fracture and derived in the
# matrix.
p, h, z, s = sp.symbols("p h z s", real=True)

# Constants, pulled from the shared symbol table so that the substitution map
_C = SYMBOLS
rho_liq, rho_hal = _C["rho_liq"], _C["rho_hal"]
mu_liq, cp = _C["mu_liq"], _C["cp"]
kappa_liq, kappa_hal, kappa_r = _C["kappa_liq"], _C["kappa_hal"], _C["kappa_r"]
rho_r, cp_r = _C["rho_r"], _C["cp_r"]
chi0, beta_T, beta_p = _C["chi0"], _C["beta_T"], _C["beta_p"]
T0, p0, z0 = _C["T0"], _C["p0"], _C["z0"]
phi0, K0 = _C["phi0"], _C["K0"]
a0, theta = _C["a0"], _C["theta"]

# The ordering of the thermodynamic dependencies expected throughout the PorePy
# compositional framework: derivative rows are [d/dp, d/dh, d/dz].
DEPENDENCIES = (p, h, z)


@dataclass(frozen=True)
class SymbolicClosure:
    """Symbolic closure relations in ``(p, h, s)``; :attr:`s_hal` is the inverse map in ``(p, h, z)``."""

    temperature: sp.Expr
    chi_star: sp.Expr
    chi_NaCl_liq: sp.Expr
    chi_H2O_liq: sp.Expr
    chi_NaCl_hal: sp.Expr
    chi_H2O_hal: sp.Expr
    z_NaCl: sp.Expr
    z_H2O: sp.Expr
    s_hal: sp.Expr
    rho: sp.Expr
    s_liq: sp.Expr
    k_r_liq: sp.Expr
    k_r_hal: sp.Expr
    lambda_NaCl: sp.Expr
    lambda_H2O: sp.Expr
    lambda_total: sp.Expr
    f_NaCl: sp.Expr
    f_H2O: sp.Expr
    aperture: sp.Expr
    fracture_permeability: sp.Expr
    matrix_porosity: sp.Expr
    matrix_permeability: sp.Expr

    def in_phz(self, expr: sp.Expr) -> sp.Expr:
        """Rewrite an ``s``-dependent expression in ``(p, h, z)`` through the inverse map.

        Parameters:
            expr: Closure expression, possibly in ``s``.

        Returns:
            The expression in ``(p, h, z)``.

        """
        return expr.subs(s, self.s_hal)

    def thermal_conductivity(self, porosity: sp.Expr) -> sp.Expr:
        """Effective thermal conductivity for a given porosity.

        Parameters:
            porosity: Porosity of the subdomain.

        Returns:
            The scalar effective conductivity.

        """
        fluid = (1 - s) * kappa_liq + s * kappa_hal
        return porosity * fluid + (1 - porosity) * kappa_r

    def energy_accumulation(self, porosity: sp.Expr) -> sp.Expr:
        """Energy accumulation ``phi (rho h - p) + (1 - phi) rho_r cp_r (T - T0)``.

        Parameters:
            porosity: Porosity of the subdomain.

        Returns:
            The energy accumulation term.

        """
        return porosity * (self.rho * h - p) + (1 - porosity) * rho_r * cp_r * (
            self.temperature - T0
        )


def _build_symbolic() -> SymbolicClosure:
    """Assemble the symbolic closure.

    Returns:
        The closure with every relation in symbolic form.

    """
    temperature = T0 + h / cp
    chi_star = chi0 + beta_T / cp * h + beta_p * (p - p0)
    chi_NaCl_liq = chi_star
    chi_H2O_liq = 1.0 - chi_star
    chi_NaCl_hal = sp.Integer(1)
    chi_H2O_hal = sp.Integer(0)

    s_liq = 1 - s
    rho_mix = s * rho_hal + s_liq * rho_liq
    z_NaCl = (s_liq * rho_liq * chi_NaCl_liq + s * rho_hal * chi_NaCl_hal) / rho_mix
    z_H2O = (s_liq * rho_liq * chi_H2O_liq + s * rho_hal * chi_H2O_hal) / rho_mix

    s_hal = rho_liq * (z - chi_star) / (rho_hal * (1.0 - z) + rho_liq * (z - chi_star))

    k_r_liq = s_liq**2.0
    k_r_hal = sp.Integer(0)

    lambda_total = rho_liq * k_r_liq / mu_liq
    lambda_NaCl = rho_liq * chi_NaCl_liq * k_r_liq / mu_liq
    lambda_H2O = rho_liq * chi_H2O_liq * k_r_liq / mu_liq

    f_NaCl = chi_star
    f_H2O = 1.0 - chi_star

    aperture = a0 * s_liq**theta
    fracture_permeability = aperture**2 / 12.0

    matrix_porosity = phi0 * s_liq
    matrix_permeability = K0 * s_liq**2.0

    return SymbolicClosure(
        temperature=temperature,
        chi_star=chi_star,
        chi_NaCl_liq=chi_NaCl_liq,
        chi_H2O_liq=chi_H2O_liq,
        chi_NaCl_hal=chi_NaCl_hal,
        chi_H2O_hal=chi_H2O_hal,
        z_NaCl=z_NaCl,
        z_H2O=z_H2O,
        s_hal=s_hal,
        rho=rho_mix,
        s_liq=s_liq,
        k_r_liq=k_r_liq,
        k_r_hal=k_r_hal,
        lambda_NaCl=lambda_NaCl,
        lambda_H2O=lambda_H2O,
        lambda_total=lambda_total,
        f_NaCl=f_NaCl,
        f_H2O=f_H2O,
        aperture=aperture,
        fracture_permeability=fracture_permeability,
        matrix_porosity=matrix_porosity,
        matrix_permeability=matrix_permeability,
    )


SYM: SymbolicClosure = _build_symbolic()
"""SymbolicClosure: The closure expressions, in terms of ``(p, h, s)``."""


def _broadcast(value: np.ndarray | float, n: int) -> np.ndarray:
    """Broadcast a possibly scalar ``lambdify`` result to shape ``(n,)``.

    Parameters:
        value: Raw result of a lambdified expression.
        n: Number of evaluation points.

    Returns:
        A float array of shape ``(n,)``.

    """
    return np.broadcast_to(np.asarray(value, dtype=float), (n,)).astype(
        float, copy=True
    )


class NumericClosure:
    """Numeric closure with exact derivatives, generated from :data:`SYM`.

    Callables take ``(p, h, z)`` arrays and return ``(values, derivatives)``, with
    derivatives of shape ``(3, n)`` ordered ``[d/dp, d/dh, d/dz]``.
    """

    def __init__(self, constants: HaliteMMSConstants = MMS_HALITE) -> None:
        self.constants = constants
        self._const_vals = constants.as_sympy_dict(SYMBOLS)
        self._cache: dict[str, Callable] = {}

    # ---- expression access --------------------------------------------------
    def expression(self, name: str, in_phz: bool = True) -> sp.Expr:
        """Closure expression with the constants substituted.

        Parameters:
            name: Attribute name on :class:`SymbolicClosure`.
            in_phz: Rewrite the expression in ``(p, h, z)``; False keeps the saturation form.

        Returns:
            The expression with numeric constants.

        """
        expr = getattr(SYM, name)
        if in_phz:
            expr = SYM.in_phz(expr)
        return sp.sympify(expr).xreplace(self._const_vals)

    # ---- generic (value, jacobian) factory ----------------------------------
    def make_func(self, expr: sp.Expr) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
        """Build a ``(values, derivatives)`` callable for an expression in ``(p, h, z)``.

        Parameters:
            expr: Expression in ``(p, h, z)`` with the constants substituted.

        Returns:
            A callable taking ``(p, h, z)`` arrays and returning ``(values, derivatives)``.

        """
        f_val = sp.lambdify(DEPENDENCIES, expr, "numpy")
        f_der = [
            sp.lambdify(DEPENDENCIES, sp.diff(expr, var), "numpy")
            for var in DEPENDENCIES
        ]

        def _evaluate(*dependencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            p_v, h_v, z_v = _unpack(dependencies)
            n = len(p_v)
            values = _broadcast(f_val(p_v, h_v, z_v), n)
            derivatives = np.vstack([_broadcast(f(p_v, h_v, z_v), n) for f in f_der])
            return values, derivatives

        return _evaluate

    def func(self, name: str) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
        """Cached ``(values, derivatives)`` callable for a closure entry.

        Parameters:
            name: Attribute name on :class:`SymbolicClosure`.

        Returns:
            The callable of :meth:`make_func`.

        """
        if name not in self._cache:
            self._cache[name] = self.make_func(self.expression(name))
        return self._cache[name]

    # ---- named functions in the signature the framework expects -------------
    def temperature_func(self, *dependencies: np.ndarray):
        """Temperature and its derivatives.

        Parameters:
            dependencies: Primary-variable arrays ``(p, h, z)``, or ``(p, h)``.

        Returns:
            The values and the derivatives, of shapes ``(n,)`` and ``(3, n)``.

        """

        return self.func("temperature")(*dependencies)

    def halite_saturation_func(self, *dependencies: np.ndarray):
        """Halite saturation and its derivatives.

        Parameters:
            dependencies: Primary-variable arrays ``(p, h, z)``, or ``(p, h)``.

        Returns:
            The values and the derivatives, of shapes ``(n,)`` and ``(3, n)``.

        """

        return self.func("s_hal")(*dependencies)

    def NaCl_liq_func(self, *dependencies: np.ndarray):
        """NaCl partial fraction in the liquid, ``chi*``.

        Parameters:
            dependencies: Primary-variable arrays ``(p, h, z)``, or ``(p, h)``.

        Returns:
            The values and the derivatives, of shapes ``(n,)`` and ``(3, n)``.

        """

        return self.func("chi_NaCl_liq")(*dependencies)

    def H2O_liq_func(self, *dependencies: np.ndarray):
        """H2O partial fraction in the liquid, ``1 - chi*``.

        Parameters:
            dependencies: Primary-variable arrays ``(p, h, z)``, or ``(p, h)``.

        Returns:
            The values and the derivatives, of shapes ``(n,)`` and ``(3, n)``.

        """

        return self.func("chi_H2O_liq")(*dependencies)

    def NaCl_halite_func(self, *dependencies: np.ndarray):
        """NaCl partial fraction in halite, one.

        Parameters:
            dependencies: Primary-variable arrays ``(p, h, z)``, or ``(p, h)``.

        Returns:
            The values and the derivatives, of shapes ``(n,)`` and ``(3, n)``.

        """

        return self.func("chi_NaCl_hal")(*dependencies)

    def H2O_halite_func(self, *dependencies: np.ndarray):
        """H2O partial fraction in halite, zero.

        Parameters:
            dependencies: Primary-variable arrays ``(p, h, z)``, or ``(p, h)``.

        Returns:
            The values and the derivatives, of shapes ``(n,)`` and ``(3, n)``.

        """

        return self.func("chi_H2O_hal")(*dependencies)

    # ---- convenience scalar evaluations -------------------------------------
    def value(self, name: str, p_v, h_v, z_v) -> np.ndarray:
        """Evaluate a closure quantity at ``(p, h, z)``.

        Parameters:
            name: Attribute name on :class:`SymbolicClosure`.
            p_v: Pressure values.
            h_v: Enthalpy values.
            z_v: Overall NaCl fraction values.

        Returns:
            The values, as a float array.

        """

        return self.func(name)(np.asarray(p_v), np.asarray(h_v), np.asarray(z_v))[0]

    def z_from_saturation(self, p_v, h_v, s_v) -> np.ndarray:
        """Overall NaCl fraction ``z(p, h, s)``.

        Parameters:
            p_v: Pressure values.
            h_v: Enthalpy values.
            s_v: Halite saturation values.

        Returns:
            The overall NaCl mass fraction.

        """

        if "z_forward" not in self._cache:
            expr = sp.sympify(SYM.z_NaCl).xreplace(self._const_vals)
            self._cache["z_forward"] = sp.lambdify((p, h, s), expr, "numpy")
        f = self._cache["z_forward"]
        n = np.broadcast(np.asarray(p_v), np.asarray(h_v), np.asarray(s_v)).size
        return _broadcast(f(np.asarray(p_v), np.asarray(h_v), np.asarray(s_v)), n)

    def zH2O_from_saturation(self, p_v, h_v, s_v) -> np.ndarray:
        """Overall H2O fraction from ``(p, h, s)``.

        Parameters:
            p_v: Pressure values.
            h_v: Enthalpy values.
            s_v: Halite saturation values.

        Returns:
            The overall H2O mass fraction.

        """

        if "zH2O_forward" not in self._cache:
            expr = sp.sympify(SYM.z_H2O).xreplace(self._const_vals)
            self._cache["zH2O_forward"] = sp.lambdify((p, h, s), expr, "numpy")
        f = self._cache["zH2O_forward"]
        n = np.broadcast(np.asarray(p_v), np.asarray(h_v), np.asarray(s_v)).size
        return _broadcast(f(np.asarray(p_v), np.asarray(h_v), np.asarray(s_v)), n)


def _unpack(dependencies: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """Normalise the dependencies passed by the framework to ``(p, h, z)`` arrays.

    Parameters:
        dependencies: The arrays as passed by the framework.

    Returns:
        The triple ``(p, h, z)`` as float arrays.

    Raises:
        ValueError: If there are neither two nor three dependencies.

    """
    if len(dependencies) == 3:
        p_v, h_v, z_v = (np.asarray(d, dtype=float) for d in dependencies)
    elif len(dependencies) == 2:
        p_v, h_v = (np.asarray(d, dtype=float) for d in dependencies)
        z_v = np.zeros_like(p_v)
    else:
        raise ValueError(
            f"Expected (p, h) or (p, h, z), got {len(dependencies)} dependencies."
        )
    return np.atleast_1d(p_v), np.atleast_1d(h_v), np.atleast_1d(z_v)


DEFAULT_CLOSURE = NumericClosure()
"""NumericClosure: Closure built from the canonical constants."""


__all__ = [
    "DEPENDENCIES",
    "NumericClosure",
    "SYM",
    "SymbolicClosure",
    "h",
    "p",
    "s",
    "z",
]
