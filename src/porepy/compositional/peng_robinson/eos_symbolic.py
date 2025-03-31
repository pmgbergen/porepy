"""Module containing general and symbolic functionality related to the standard
Peng-Robinson EoS.

It concerns itself with *analytic* expressions of thermodynamic properties as functions
depending intensive state variables.

We use ``sympy`` to produce symbols and expressions, which are
subsequently turned into numeric functions using :func:`sympy.lambdify`.

Due to the magnitude of expressions and functions (and different versions),
naming conflicts are hard to avoid.
We introcued the naming convention ``<derivative><name>_<type>``,
where the name represents the quantity, and type how the quantity is represented.

The convention for types of a quantity include:

- ``_s``: A symbol representing either an independent quantity, or an intermediate
  quantity serving as an argument. Created using :class:`sympy.Symbol`.
- ``_e``: A symbolic expression created using some algebraic combination of symbols.
  It depends on usually several symbols.
- ``_f``: A lambdify-generated function, based on an expression. The arguments of the
  function reflect the dependency on symbols.

For example, the compressibility factor has the standard symbol ``Z`` in literature:

Example:
    The compressibility factor has the standard symbol ``Z`` as found in the literature.

    - ``Z_s`` denotes the symbolic representation using ``sympy``. It is used as an
      intermediate dependency for e.g., departure functions.
    - ``Z_e`` denotes a symbolic **expression** dependent on some other symbols.
      In this case it is ``A_s`` and ``B_s``.
    - ``dZ_e`` denotes the complete gradient w.r.t. all its dependencies.
    - ``Z_f`` would be a function with signature ``(float, float) -> float``.
    - ``dZ_f`` would be a function with signature ``(float, float) -> array(2)``
      because it depends on two variables.

The following standard names are used for thermodynamic quantities:

- ``Z`` compressibility factor
- ``A`` non-dimensional cohesion
- ``B`` non-dimensional covolume
- ``a`` cohesion
- ``b`` covolume
- ``T`` temperature
- ``p`` pressure
- ``x`` (extended) fractions of components in a phases
- ``y`` molar phase fractions
- ``z`` overall component fractions / feed fractions
- ``sat`` volumetric phase fractions (saturations)
- ``_i`` index related to a component i
- ``_j`` index related to a phase j
- ``_r`` index related to the reference phase (the first one is assumed to be r)

"""

from __future__ import annotations

from typing import Any, Callable, Sequence, TypeAlias

import numba
import numpy as np
import sympy as sp

from .._core import COMPOSITIONAL_VARIABLE_SYMBOLS as SYMBOLS
from .._core import R_IDEAL_MOL
from ..materials import FluidComponent
from ..utils import safe_sum
from .utils import VanDerWaals_cohesion, VanDerWaals_covolume, thd_function_type

__all__ = [
    "A_CRIT",
    "B_CRIT",
    "Z_CRIT",
    "PengRobinsonSymbolic",
]


A_CRIT: float = (
    1
    / 512
    * (
        -59
        + 3 * np.cbrt(276231 - 192512 * np.sqrt(2))
        + 3 * np.cbrt(276231 + 192512 * np.sqrt(2))
    )
)
"""Critical, non-dimensional cohesion value in the Peng-Robinson EoS,
~ 0.457235529."""


B_CRIT: float = (
    1
    / 32
    * (-1 - 3 * np.cbrt(16 * np.sqrt(2) - 13) + 3 * np.cbrt(16 * np.sqrt(2) + 13))
)
"""Critical, non-dimensional covolume in the Peng-Robinson EoS, ~ 0.077796073."""


Z_CRIT: float = (
    1 / 32 * (11 + np.cbrt(16 * np.sqrt(2) - 13) - np.cbrt(16 * np.sqrt(2) + 13))
)
"""Critical compressibility factor in the Peng-Robinson EoS, ~ 0.307401308."""


@numba.njit(cache=True)
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


class PengRobinsonSymbolic:
    """A class providing functions for thermodynamic properties using the Peng-Robinson
    EoS, based on a symbolic representation using ``sympy``.

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
        """List of phase composition fractions associated with a phase.
        Length is equal to number of components, because every component is assumed
        present in every phase in the unified setting."""

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
    def a(self) -> sp.Expr:
        """Cohesion of the mixture according to the set mixing rule."""
        a_i_correction: list[sp.Expr] = [
            1 + k * (1 - sp.sqrt(self.T_s / T_ic))
            for k, T_ic in zip(self.k_i, self.T_i_crit)
        ]

        a_i: list[sp.Expr] = [
            a * corr**2 for a, corr in zip(self.a_i_crit, a_i_correction)
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
        """Lambdified expression :meth:`B` returning the non-dimensional covolume
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
            PengRobinsonSymbolic._truncate(
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
            phi_i.append(sp.exp(PengRobinsonSymbolic._cap(log_phi_i)))

        return sp.Matrix(phi_i)

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


A = PengRobinsonSymbolic.A_s
B = PengRobinsonSymbolic.B_s


# region Functionality related to cubic polynomials


def coeff_0(A: Any, B: Any) -> Any:
    r"""Coefficient for the zeroth monomial of the characteristic equation

    :math:`Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0`.

    For any input type supporting Python's overload of ``+,-,*,**``.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        :math:`B^3 + B^2 - AB`.

    """
    return B**3 + B**2 - A * B


def coeff_1(A: Any, B: Any) -> Any:
    r"""Coefficient for the first monomial of the characteristic equation

    :math:`Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0`.

    For any input type supporting Python's overload of ``+,-,*,**``.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        :math:`A - 2 B - 3 B^2`.

    """
    return A - 2.0 * B - 3.0 * B**2


def coeff_2(A: Any, B: Any) -> Any:
    r"""Coefficient for the second monomial of the characteristic equation

    :math:`Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0`.

    For any input type supporting Python's overload of ``-``.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        :math:`B - 1`.

    """
    return B - 1


def reduced_coeff_0(A: Any, B: Any) -> Any:
    r"""Zeroth coefficient of the reduced characteristic equation

    :math:`Z^3 + c_{r1} Z + c_{r0} = 0`.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        :math:`c_2^3(A, B)\frac{2}{27} - c_2(A, B) c_1(A, B)\frac{1}{3} + c_0(A, B)`

    """
    c2 = coeff_2(A, B)
    return c2**3 * (2.0 / 27.0) - c2 * coeff_1(A, B) * (1.0 / 3.0) + coeff_0(A, B)


def reduced_coeff_1(A: Any, B: Any) -> Any:
    r"""First coefficient of the reduced characteristic equation

    :math:`Z^3 + c_{r1} Z + c_{r0} = 0`.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        :math:`c_1(A, B) - c_2^2(A, B)\frac{1}{3}`

    """
    return coeff_1(A, B) - coeff_2(A, B) ** 2 * (1.0 / 3.0)


def discriminant(rc0: Any, rc1: Any) -> Any:
    r"""Discriminant of the cubic, characteristic polynomial based on the reduced
    coefficient.

    Parameters:
        rc0: Zeroth reduced coefficient.
        rc1: First reduced coefficient.

    Returns:
        :math:`c_{r0}^2\frac{1}{4} - c_{r1}^3\frac{1}{27}`

    """
    return rc0**2 * (1.0 / 4.0) + rc1**3 * (1.0 / 27.0)


class _cbrt(sp.Function):
    """Custom symbolic cubic root to circumvent sympy using the power expression.

    The power expression is costly and does not always work with negative numbers.
    It returns sometimes not the principle cubic root (which is always real).

    Has a custom implementation of the derivative to always return a positive, real
    number.

    For more information
    `see here <https://docs.sympy.org/latest/guides/custom-functions.html>`_.

    Warning:
        As of now, lambdified expressions using this must use
        ``[{'_cbrt': numpy.cbrt}]`` in there module argument to provide a numerical
        evaluation for this function. TODO

        This Function is temprary and experimental. TODO do tests.

    """

    def fdiff(self, argindex=1):
        a = self.args[0]
        return 1.0 / (_cbrt(a**2) * 3)


def triple_root(A: sp.Symbol, B: sp.Symbol) -> sp.Expr:
    r"""Formula for triple root of characteristic polynomial.

    Only valid it indeed has a triple root.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        :math:`-\frac{c_2(A, B)}{3}`

        `See here <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_
        in section triple-root case.

    """
    c2 = coeff_2(A, B)
    return -c2 / 3


def double_root(A: sp.Symbol, B: sp.Symbol, gaslike: bool) -> sp.Expr:
    """Formulae for double roots.

    Only valid in 2-root case (see :func:`get_root_case_c`).

    Important:
        This returns a piece-wise expression, selecting the bigger root for the gas-like
        case. Lambdification only with module ``'math'``.

        The lambdified expression using module ``'numpy'`` cannot be compiled by
        numba.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.
        gaslike: Bool indicating if the bigger root should be returned, or the smaller

    Returns:
        `See here <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_
        in section 2-root case.

    """
    c2 = coeff_2(A, B)
    q = reduced_coeff_0(A, B)
    r = reduced_coeff_1(A, B)

    u = 3 / 2 * q / r

    z1 = 2 * u - c2 / 3
    z23 = -u - c2 / 3

    if gaslike:
        return sp.Piecewise((z1, z1 > z23), (z23, True))
    else:
        return sp.Piecewise((z23, z1 > z23), (z1, True))


def three_root(A: sp.Symbol, B: sp.Symbol, gaslike: bool) -> sp.Expr:
    """Formulae for the 3-root case using the trigonometric representation
    (Casus Irreducibilis)

    Only valid in 3-root case (see :func:`get_root_case_c`).

    Note:
        The formulae allow for a clear distinction which root is the biggest, which
        the smallest. I.e. no piecewise nature required.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.
        gaslike: Bool indicating if the biggest root should be returned, or the smallest

    Returns:
        `See here <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_
        in section 3-root case, trigonometric approach.

    """
    c2 = coeff_2(A, B)
    q = reduced_coeff_0(A, B)
    r = reduced_coeff_1(A, B)

    # trigonometric formula for Casus Irreducibilis
    t_2 = sp.acos(-q / 2 * sp.sqrt(-27 * r ** (-3))) / 3
    t_1 = sp.sqrt(-4 / 3 * r)

    if gaslike:
        return t_1 * sp.cos(t_2) - c2 / 3
    else:
        return -t_1 * sp.cos(t_2 - np.pi / 3) - c2 / 3


def three_root_intermediate(A: sp.Symbol, B: sp.Symbol) -> sp.Expr:
    """Formula for the intermediate root in the 3-root case using the trigonometric
    representation

    Only valid if it indeed has 3 real roots.

    Note:
        This root has no physical meaning and is only used in the smoothing procedure
        proposed by Ben Gharbia et al. (2021).

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        `See here <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_
        in section 3-root case, trigonometric approach.

    """
    c2 = coeff_2(A, B)
    q = reduced_coeff_0(A, B)
    r = reduced_coeff_1(A, B)

    t_2 = sp.acos(-q / 2 * sp.sqrt(-27 * r ** (-3))) / 3
    t_1 = sp.sqrt(-4 / 3 * r)

    return -t_1 * sp.cos(t_2 + np.pi / 3) - c2 / 3


def one_root(A: sp.Symbol, B: sp.Symbol) -> sp.Expr:
    """Formulae for single, real root.

    Only valid in 1-root case (see :func:`get_root_case_c`).

    Important:
        This returns a piece-wise expression, due to some special choices of root
        computation for numerical reasons.

        Lambdification only with module ``'math'``.
        The lambdified expression using module ``'numpy'`` cannot be compiled by
        numba.

        Furthermore, a custom implementation of the cubic root is used, which
        allways returns the principal cubic root (which is always real).

        This custom implementation must be replaced upon lambdification, using
        the module argument ``[{'_cbrt': numpy.cbrt}, 'math']``

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        `See here <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_
        in section 1-root case.

    """
    c2 = coeff_2(A, B)
    q = reduced_coeff_0(A, B)
    r = reduced_coeff_1(A, B)
    d = discriminant(q, r)

    t1 = sp.sqrt(d) - q * 0.5
    t2 = -1 * (sp.sqrt(d) + q * 0.5)

    t = sp.Piecewise((t2, sp.Abs(t2) > sp.Abs(t1)), (t1, True))

    u = _cbrt(t)

    return u - r / (3 * u) - c2 / 3


# endregion


# Symbolic expressions for all root cases

Z_triple: sp.Expr = triple_root(A, B)
dZ_triple: list[sp.Expr] = [Z_triple.diff(A), Z_triple.diff(B)]

Z_one: sp.Expr = one_root(A, B)
dZ_one: list[sp.Expr] = [Z_one.diff(A), Z_one.diff(B)]

Z_double_g: sp.Expr = double_root(A, B, True)
dZ_double_g: list[sp.Expr] = [Z_double_g.diff(A), Z_double_g.diff(B)]
Z_double_l: sp.Expr = double_root(A, B, False)
dZ_double_l: list[sp.Expr] = [Z_double_l.diff(A), Z_double_l.diff(B)]

Z_three_g: sp.Expr = three_root(A, B, True)
dZ_three_g: list[sp.Expr] = [Z_three_g.diff(A), Z_three_g.diff(B)]
Z_three_l: sp.Expr = three_root(A, B, False)
dZ_three_l: list[sp.Expr] = [Z_three_l.diff(A), Z_three_l.diff(B)]
Z_three_i: sp.Expr = three_root_intermediate(A, B)
dZ_three_i: list[sp.Expr] = [Z_three_i.diff(A), Z_three_i.diff(B)]


# Lambdified functions for roots depending on A and B


Z_TYPE: TypeAlias = Callable[[float, float], float]
"""Type alias for functions representing the compressibility factor."""
dZ_TYPE: TypeAlias = Callable[[float, float], list[float] | np.ndarray]
"""Type alias for functions representing the derivatives of the compressibility factor.
"""

Z_triple_f: Z_TYPE = sp.lambdify([A, B], Z_triple)
dZ_triple_f: dZ_TYPE = sp.lambdify([A, B], dZ_triple)

Z_three_g_f: Z_TYPE = sp.lambdify([A, B], Z_three_g)
dZ_three_g_f: dZ_TYPE = sp.lambdify([A, B], dZ_three_g)
Z_three_l_f: Z_TYPE = sp.lambdify([A, B], Z_three_l)
dZ_three_l_f: dZ_TYPE = sp.lambdify([A, B], dZ_three_l)
Z_three_i_f: Z_TYPE = sp.lambdify([A, B], Z_three_i)
dZ_three_i_f: dZ_TYPE = sp.lambdify([A, B], dZ_three_i)

# because piecewise and to provide numeric evaluation of custom cubic root
_modules_one = [{"_cbrt": np.cbrt, "select": _select}, "numpy"]

Z_one_f: Z_TYPE = sp.lambdify([A, B], Z_one, modules=_modules_one)
dZ_one_f: dZ_TYPE = sp.lambdify([A, B], dZ_one, modules=_modules_one)

# because piece-wise
_modules_double = [{"select": _select}, "numpy"]

Z_double_g_f: Z_TYPE = sp.lambdify([A, B], Z_double_g, modules=_modules_double)
dZ_double_g_f: dZ_TYPE = sp.lambdify([A, B], dZ_double_g, modules=_modules_double)
Z_double_l_f: Z_TYPE = sp.lambdify([A, B], Z_double_l, modules=_modules_double)
dZ_double_l_f: dZ_TYPE = sp.lambdify([A, B], dZ_double_l, modules=_modules_double)
