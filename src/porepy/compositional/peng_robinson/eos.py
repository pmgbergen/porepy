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
from .._core import NUMBA_FAST_MATH, R_IDEAL_MOL
from ..compiled_flash.eos_compiler import EoSCompiler, ScalarFunction, VectorFunction
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


def _compile_fugacities(
    phis: Callable[[float, float, np.ndarray, float, float, float], np.ndarray],
) -> Callable[[float, float, np.ndarray, float, float, float], np.ndarray]:
    """Helper function to compile the vector of fugacity coefficients.

    It needs an additional reduction of shape from ``(num_comp, 1)`` to ``(num_comp,)``
    because of the usage of a symbolic, vector-valued function."""
    f = nb.njit(phis)

    @nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8[:], nb.f8, nb.f8, nb.f8))
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
    df = nb.njit(thd_df, fastmath=NUMBA_FAST_MATH)

    @nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8[:]), fastmath=NUMBA_FAST_MATH)
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
    df = nb.njit(ext_thd_df)

    @nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8[:], nb.f8, nb.f8, nb.f8))
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

    dv_ = nb.njit(fastmath=NUMBA_FAST_MATH)(dv)

    @nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8))
    def inner(p_, T_, Z_):
        return np.array(dv_(p_, T_, Z_), dtype=np.float64)

    return inner


@nb.njit(cache=True)
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


class CompiledPengRobinson(EoSCompiler):
    """Class providing compiled computations of thermodynamic quantities for the
    Peng-Robinson EoS.

    The parameter array for the pre-argument function can have up to 3 entries
    (see also :attr:`params`):

    1. ``'smoothing_multiphase'`` : Portion of 2-phase region used for smoothing roots
       near phase borders
    2. ``'eps'``: Numerical tolerance to determine zero (root case computation).

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

        self._cfuncs: dict[str, Callable] = dict()
        """A collection of internally required, compiled callables"""

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

        self.symbolic = SymbolicPengRobinson(components, ideal_enthalpies, bip_matrix)
        """Symbolic representation of the EoS, providing expressions and derivatives
        for properties, which are turned into functions and compiled."""

    # NOTE: The two _get_cohesion* methods are only abstracted for the Soereide
    # extension because of the varying signature. Abstraction of other compilations of
    # symbolic functions can be done analogously once required.
    def _get_cohesion(self) -> ScalarFunction:
        """Abstraction of compilation of non-dimensional cohesion."""
        return nb.njit(nb.f8(nb.f8, nb.f8, nb.f8[:]))(self.symbolic.A_func)

    def _get_cohesion_derivatives(self) -> VectorFunction:
        """Abstraction of compilation of non-dimensional cohesion derivatives."""
        return _compile_thd_function_derivatives(self.symbolic.grad_pTx_A_func)

    def compile(self) -> None:
        """Child method compiles essential functions from symbolic part before calling
        the parent class compiler"""

        logger.info("Compiling symbolic Peng-Robinson EoS ..")
        start = time.time()

        B_c = nb.njit(
            nb.f8(nb.f8, nb.f8, nb.f8[:]),
            fastmath=NUMBA_FAST_MATH,
        )(self.symbolic.B_func)
        logger.debug("Compiling symbolic functions 1/12")
        dB_c = _compile_thd_function_derivatives(self.symbolic.grad_pTx_B_func)
        logger.debug("Compiling symbolic functions 2/12")

        A_c = self._get_cohesion()
        logger.debug("Compiling symbolic functions 3/12")
        dA_c = self._get_cohesion_derivatives()
        logger.debug("Compiling symbolic functions 4/12")

        phi_c = _compile_fugacities(self.symbolic.phis_func)
        logger.debug("Compiling symbolic functions 5/12")
        dphi_c = nb.njit(nb.f8[:, :](nb.f8, nb.f8, nb.f8[:], nb.f8, nb.f8, nb.f8))(
            self.symbolic.jac_phis_func
        )
        logger.debug("Compiling symbolic functions 6/12")

        h_dep_c = nb.njit(nb.f8(nb.f8, nb.f8, nb.f8[:], nb.f8, nb.f8, nb.f8))(
            self.symbolic.h_departure_func
        )
        logger.debug("Compiling symbolic functions 7/12")
        h_ideal_c = nb.njit(nb.f8(nb.f8, nb.f8, nb.f8[:]))(self.symbolic.h_ideal_func)
        logger.debug("Compiling symbolic functions 8/12")
        dh_dep_c = _compile_extended_thd_function_derivatives(
            self.symbolic.grad_pTxABZ_h_departure_func
        )
        logger.debug("Compiling symbolic functions 9/12")
        dh_ideal_c = _compile_thd_function_derivatives(
            self.symbolic.grad_pTx_h_ideal_func
        )
        logger.debug("Compiling symbolic functions 10/12")

        rho_c = nb.njit(
            nb.f8(nb.f8, nb.f8, nb.f8),
            fastmath=NUMBA_FAST_MATH,
        )(self.symbolic.rho_func)
        logger.debug("Compiling symbolic functions 11/12")
        drho_c = _compile_density_derivative(self.symbolic.grad_pTZ_rho_func)
        logger.debug("Compiling symbolic functions 12/12")

        self._cfuncs.update(
            {
                "A": A_c,
                "B": B_c,
                "dA": dA_c,
                "dB": dB_c,
                "phi": phi_c,
                "dphi": dphi_c,
                "h_dep": h_dep_c,
                "h_ideal": h_ideal_c,
                "dh_dep": dh_dep_c,
                "dh_ideal": dh_ideal_c,
                "rho": rho_c,
                "drho": drho_c,
            }
        )

        super().compile()

        logger.info(
            f"{self._nc}-component Peng-Robinson EoS compiled"
            + " (elapsed time: %.5f (s))." % (time.time() - start)
        )

    def get_prearg_for_values(self) -> VectorFunction:
        A_c = self._cfuncs["A"]
        B_c = self._cfuncs["B"]

        eps = self.params["eps"]
        s_m = self.params["smoothing_multiphase"]

        @nb.njit(nb.f8[:](nb.i1, nb.f8, nb.f8, nb.f8[:], nb.f8[:]))
        def prearg_val_c(
            phasetype: int, p: float, T: float, xn: np.ndarray, params: np.ndarray
        ) -> np.ndarray:
            prearg = np.empty((4,), dtype=np.float64)
            A = A_c(p, T, xn)
            B = B_c(p, T, xn)

            # Choose default parameters, and then parse given parameters.
            # Can only be done this way because params are a sub-array of the generic
            # argument.
            s_m_ = s_m
            eps_ = eps
            if params.size >= 1:
                s_m_ = params[0]
            if params.size >= 2:
                eps_ = params[1]

            if phasetype == 1:
                gaslike = True
            elif phasetype == 0:
                gaslike = False
            else:
                raise NotImplementedError(f"Unsupported phase type: {phasetype}")

            prearg[0] = A_c(p, T, xn)
            prearg[1] = B_c(p, T, xn)
            prearg[2] = get_compressibility_factor(A, B, gaslike, eps_, s_m_)
            prearg[3] = float(phasetype)

            return prearg

        return prearg_val_c

    def get_prearg_for_derivatives(self) -> VectorFunction:
        A_c = self._cfuncs["A"]
        B_c = self._cfuncs["B"]
        dA_c = self._cfuncs["dA"]
        dB_c = self._cfuncs["dB"]
        # number of derivatives for A, B, Z (p, T, and per component fraction)
        d = 2 + self._nc

        eps = self.params["eps"]
        s_m = self.params["smoothing_multiphase"]

        @nb.njit(nb.f8[:](nb.i1, nb.f8, nb.f8, nb.f8[:], nb.f8[:]))
        def prearg_jac_c(
            phasetype: int, p: float, T: float, xn: np.ndarray, params: np.ndarray
        ) -> np.ndarray:
            # the pre-arg for the jacobian contains the derivatives of A, B, Z
            # w.r.t. p, T, and fractions.
            prearg = np.empty((3 * d,), dtype=np.float64)

            s_m_ = s_m
            eps_ = eps
            if params.size >= 1:
                s_m_ = params[0]
            if params.size >= 2:
                eps_ = params[1]

            if phasetype == 1:
                gaslike = True
            elif phasetype == 0:
                gaslike = False
            else:
                raise NotImplementedError(f"Unsupported phase type: {phasetype}")

            A = A_c(p, T, xn)
            B = B_c(p, T, xn)

            dA = dA_c(p, T, xn)
            dB = dB_c(p, T, xn)
            dZ_ = get_compressibility_factor_derivatives(A, B, gaslike, eps_, s_m_)
            dZ = dZ_[0] * dA + dZ_[1] * dB

            prearg[0:d] = dA
            prearg[d : 2 * d] = dB
            prearg[2 * d : 3 * d] = dZ

            return prearg

        return prearg_jac_c

    def get_fugacity_function(self) -> VectorFunction:
        phi_c = self._cfuncs["phi"]

        @nb.njit(nb.f8[:](nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def phi_mix_c(
            prearg: np.ndarray, p: float, T: float, xn: np.ndarray
        ) -> np.ndarray:
            return phi_c(p, T, xn, prearg[0], prearg[1], prearg[2])

        return phi_mix_c

    def get_fugacity_derivative_function(self) -> VectorFunction:
        dphi_c = self._cfuncs["dphi"]
        # number of derivatives
        d = 2 + self._nc

        @nb.njit(nb.f8[:, :](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def dphi_mix_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            # computation of phis dependent on A_j, B_j, Z_j
            d_phis = dphi_c(p, T, xn, prearg_val[0], prearg_val[1], prearg_val[2])
            # derivatives of A_j, B_j, Z_j w.r.t. p, T, and X_j
            dA = prearg_jac[0:d]
            dB = prearg_jac[d : 2 * d]
            dZ = prearg_jac[2 * d : 3 * d]
            # expansion of derivatives (chain rule)
            return (
                d_phis[:, :-3]
                + np.outer(d_phis[:, -3], dA)
                + np.outer(d_phis[:, -2], dB)
                + np.outer(d_phis[:, -1], dZ)
            )

        return dphi_mix_c

    def get_enthalpy_function(self) -> ScalarFunction:
        h_dep_c = self._cfuncs["h_dep"]
        h_ideal_c = self._cfuncs["h_ideal"]

        @nb.njit(nb.f8(nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def h_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> np.ndarray:
            return h_ideal_c(p, T, xn) + h_dep_c(
                p, T, xn, prearg[0], prearg[1], prearg[2]
            )

        return h_c

    def get_enthalpy_derivative_function(self) -> VectorFunction:
        d = 2 + self._nc
        dh_dep_c = self._cfuncs["dh_dep"]
        dh_ideal_c = self._cfuncs["dh_ideal"]

        @nb.njit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
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
        rho_c_ = self._cfuncs["rho"]

        @nb.njit(nb.f8(nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def rho_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> np.ndarray:
            return rho_c_(p, T, prearg[2])

        return rho_c

    def get_density_derivative_function(self) -> VectorFunction:
        d = 2 + self._nc
        drho_c_ = self._cfuncs["drho"]

        @nb.njit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def drho_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            d_rho_ = drho_c_(p, T, prearg_val[2])
            # derivatives of Z_j w.r.t. p, T, and X_j
            dZ = prearg_jac[2 * d : 3 * d]
            # expansion of derivatives (chain rule)
            d_rho = d_rho_[-1] * dZ
            d_rho[:2] += d_rho_[:2]  # contribution of p, T derivatives
            return d_rho

        return drho_c
