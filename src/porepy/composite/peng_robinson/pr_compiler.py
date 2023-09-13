"""Experimentel code for efficient unified flash calculations using numba and sympy."""
from __future__ import annotations

import inspect
import os
import time

# os.environ['NUMBA_DISABLE_INTEL_SVML']  = '1'
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '30'

import numba

import numpy as np
import sympy as sm

from typing import Callable, Literal

from .._core import COMPOSITIONAL_VARIABLE_SYMBOLS as SYMBOLS
from ..component import Component, Compound
from ..composite_utils import normalize_fractions, safe_sum
from ..phase import Phase, PhaseProperties
from ..mixture import NonReactiveMixture, BasicMixture

from .eos import PengRobinsonEoS, A_CRIT, B_CRIT
from .mixing import VanDerWaals

__all__ = ['MixtureSymbols', 'PR_Compiler']


_COEFF_COMPILTER_ARGS = {
    'fastmath': True,
    'cache' : True,
}


# Need abstraction for cubic root, sympy will use **(1/3) otherwise,
# which is bad.
# NOTE lambdified expressions need an substitution of for {'cbrt': numpy.cbrt}
# in their module argument
_cbrt = sm.Function('cbrt')


def coef0(A, B):
    """Coefficient for the zeroth monomial."""
    return B**3 + B**2 - A * B


coef0_c = numba.njit(**_COEFF_COMPILTER_ARGS)(coef0)


def coef1(A, B):
    """Coefficient for the first monomial."""
    return A - 2. * B - 3. * B**2


coef1_c = numba.njit(**_COEFF_COMPILTER_ARGS)(coef1)


def coef2(A, B):
    """Coefficient for the second monomial."""
    return B - 1


coef2_c = numba.njit(**_COEFF_COMPILTER_ARGS)(coef2)


def red_coef0(A, B):
    """Zeroth coefficient of the reduced polynomial."""
    c2 = coef2(A, B)
    return c2**3 * (2. / 27.) - c2 * coef1(A, B) * (1. / 3.) + coef0(A, B)


@numba.njit(**_COEFF_COMPILTER_ARGS)
def red_coef0_c(A, B):
    c2 = coef2_c(A, B)
    return c2**3 * (2. / 27.)  - c2 * coef1_c(A, B) * (1. / 3.) + coef0_c(A, B)


def red_coef1(A, B):
    """First coefficient of the reduced polynomial."""
    return coef1(A, B) - coef2(A, B)**2 * (1. / 3.)


@numba.njit(**_COEFF_COMPILTER_ARGS)
def red_coef1_c(A, B):
    return coef1_c(A, B) - coef2_c(A, B)**2 * (1. / 3.)


def discr(rc0, rc1):
    """Discriminant of the polynomial based on the zeroth and first reduced coefficient.
    """
    return rc0**2 * (1. / 4.) + rc1**3 * (1. / 27.)


discr_c = numba.njit(**_COEFF_COMPILTER_ARGS)(discr)


@numba.njit(**_COEFF_COMPILTER_ARGS)
def get_root_case_c(A, B, eps=1e-14):
    """"An piece-wise cosntant function dependent on
    non-dimensional cohesion and covolume, characterizing the root case:

    - 0 : triple root
    - 1 : 1 real root, 2 complex-conjugated roots
    - 2 : 2 real roots, one with multiplicity 2
    - 3 : 3 distinct real roots

    ``eps`` is for defining the numerical zero (degenerate polynomial).

    """
    q = red_coef0_c(A, B)
    r = red_coef1_c(A, B)
    d = discr_c(q, r)

    # c = np.zeros(d.shape)

    # if discriminant is positive, the polynomial has one real root
    if d > eps:
        return 1
    # if discriminant is negative, the polynomial has three distinct real roots
    if d < -eps:
        return 3
    # if discrimant is zero, we are in the degenerate case
    else:
        # if first reduced coefficient is zero, the polynomial has a triple root
        # the critical point is a known triple root
        if np.abs(r) < eps or (np.abs(A - A_CRIT) < eps and np.abs(B - B_CRIT) < eps):
            return 0
        # if first reduced coefficient is not zero, the polynomial has 2 real roots
        # one with multiplicity 2
        # the zero point (A=B=0) is one such case.
        else:
            return 2


get_root_case_cv = numba.vectorize(
    [numba.int8(numba.float64, numba.float64, numba.float64)],
    nopython=True,
    **_COEFF_COMPILTER_ARGS,
)(get_root_case_c)


@numba.njit(**_COEFF_COMPILTER_ARGS)
def critical_line_c(A: float) -> float:
    """Returns the critical line parametrized as ``B(A)``."""
    return B_CRIT / A_CRIT * A


critical_line_cv = numba.vectorize(
    [numba.float64(numba.float64)],
    nopython=True,
    **_COEFF_COMPILTER_ARGS,
)(critical_line_c)


@numba.njit(**_COEFF_COMPILTER_ARGS)
def widom_line_c(A: float) -> float:
    """Returns the Widom-line ``B(A)``"""
    return B_CRIT + 0.8 * 0.3381965009398633 * (A - A_CRIT)


widom_line_cv = numba.vectorize(
    [numba.float64(numba.float64)],
    nopython=True,
    **_COEFF_COMPILTER_ARGS,
)(widom_line_c)


def njit_diffs_v(f, dtype = np.float64, **njit_kwargs):
    """For wrapping symbolic, multivariate derivatives which return a sequence,
    instead of a numpy array.
    
    Intended for functions which take the multivariate, scalar input in array form.
    
    """
    f = numba.njit(f, **njit_kwargs)

    @numba.njit(**njit_kwargs)
    def inner(x):
        return np.array(f(x), dtype=dtype)

    return inner


def njit_diffs(f, dtype = np.float64, **njit_kwargs):
    """For wrapping symbolic, multivariate derivatives which return a sequence,
    instead of a numpy array.
    
    Intended for functions which take the multivariate, scalar input as individual
    arguments.

    """
    f = numba.njit(f, **njit_kwargs)

    @numba.njit(**njit_kwargs)
    def inner(*x):
        return np.array(f(*x), dtype=dtype)

    return inner


@numba.njit(cache=True)
def point_to_line_distance_c(p: np.ndarray, lp1: np.ndarray, lp2: np.ndarray) -> float:
    """Computes the distance between a 2-D point ``p`` and a line spanned by two points
    ``lp1`` and ``lp2``."""

    d = np.sqrt((lp2[0] - lp1[0]) ** 2 + (lp2[1] - lp1[1]) ** 2)
    n = np.abs(
        (lp2[0] - lp1[0]) * (lp1[1] - p[1])
        - (lp1[0] - p[0]) * (lp2[1] - lp1[1])
    )
    return n / d

B_CRIT_LINE_POINTS = (np.array([0., B_CRIT]), np.array([A_CRIT, B_CRIT]))

S_CRIT_LINE_POINTS = (np.zeros(2), np.array([A_CRIT, B_CRIT]))

W_LINE_POINTS = (np.array([0., widom_line_c(0)]), np.array([A_CRIT, widom_line_c(A_CRIT)]))


def triple_root(A: sm.Expr, B: sm.Expr) -> sm.Expr:
    """Formula for tripple root. Only valid if triple root case."""
    c2 = coef2(A, B)
    return - c2 / 3


def double_root(A: sm.Expr, B: sm.Expr, gaslike: bool) -> sm.Expr:
    """Returns the largest root in the 2-root case if ``gaslike==True``.
    Else it returns the smallest one."""
    c2 = coef2(A, B)
    q = red_coef0(A, B)
    r = red_coef1(A, B)

    u = 3 / 2 * q / r

    z1 = 2 * u - c2 / 3
    z23 = -u - c2 / 3

    if gaslike:
        return sm.Piecewise((z1, z1 > z23), (z23, True))
    else:
        return sm.Piecewise((z23, z1 > z23), (z1, True))


def three_root(A: sm.Expr, B: sm.Expr, gaslike: bool) -> sm.Expr:
    """Returns the largest root in the three-root case if ``gaslike==True``.
    Else it returns the smallest one.

    Applies toe trigonometric formula for Casus Irreducibilis."""
    c2 = coef2(A, B)
    q = red_coef0(A, B)
    r = red_coef1(A, B)

    # trigonometric formula for Casus Irreducibilis
    t_2 = sm.acos(-q / 2 * sm.sqrt(-27 * r **(-3))) / 3
    t_1 = sm.sqrt(-4 / 3 * r)

    if gaslike:
        return t_1 * sm.cos(t_2) - c2 / 3
    else:
        return -t_1 * sm.cos(t_2 - np.pi / 3) - c2 / 3


def three_root_intermediate(A: sm.Expr, B: sm.Expr) -> sm.Expr:
    """Returns the middle root in the 3-root-case.
    
    Applies toe trigonometric formula for Casus Irreducibilis."""
    c2 = coef2(A, B)
    q = red_coef0(A, B)
    r = red_coef1(A, B)

    t_2 = sm.acos(-q / 2 * sm.sqrt(-27 * r **(-3))) / 3
    t_1 = sm.sqrt(-4 / 3 * r)

    return -t_1 * sm.cos(t_2 + np.pi / 3) - c2 / 3


def one_root(A: sm.Expr, B: sm.Expr) -> sm.Expr:
    """Returns the single real root in the 1-root case."""
    c2 = coef2(A, B)
    q = red_coef0(A, B)
    r = red_coef1(A, B)
    d = discr(q, r)

    t1 = sm.sqrt(d) - q * .5
    t2 = -1 * (sm.sqrt(d) + q * .5)

    t = sm.Piecewise((t2, sm.Abs(t2) > sm.Abs(t1)), (t1, True))

    u = _cbrt(t)

    return u - r / (3 * u) - c2 / 3


def ext_root_gharbia(Z: sm.Expr, B: sm.Expr) -> sm.Expr:
    """Returns the extended compressibility factor following Ben Gharbia et al."""
    return (1 - B - Z) / 2


def ext_root_scg(Z: sm.Expr, B: sm.Expr) -> sm.Expr:
    """Returns the extended compressibility factor for absend gas phase in the
    supercritical area."""
    return (1 - B - Z) / 2 + B 


def ext_root_scl(Z: sm.Expr, B: sm.Expr) -> sm.Expr:
    """Returns the extended compressibility factor for absend liquid phase above the
    supercritical line."""
    return (B - Z) / 2 + Z


@numba.njit(**_COEFF_COMPILTER_ARGS)
def check_if_root_c(Z, A, B):
    """Checks if given Z is a root of the compressibility polynomial by evaluating
    the polynomial. If Z is a root, the value is (numerically) zero."""
    c2 = coef2_c(A, B)
    c1 = coef1_c(A, B)
    c0 = coef0_c(A, B)

    return Z**3 + c2 * Z**2 + c1 * Z + c0


check_if_root_cv = numba.vectorize(
    [numba.int8(numba.float64, numba.float64, numba.float64)],
    nopython=True,
    **_COEFF_COMPILTER_ARGS,
)(check_if_root_c)


class MixtureSymbols:
    """A class containing basic symbols (thermodynamic properties and variables) for a
    mixture represented using ``sympy``.
    
    It is meant for symbols which are considered primary or independent e.g., pressure
    or molar fractions, not for higher expressions.
    
    """

    def __init__(self, mixture: BasicMixture) -> None:
        
        self._all: dict[str, dict] = dict()
        """Shortcut map between symbol names and the public dictionary they are stored
        in."""

        self.mixture_properties: dict[str, sm.Symbol] = {
            "pressure": sm.Symbol(f"{SYMBOLS['pressure']}"),
            "temperature": sm.Symbol(f"{SYMBOLS['temperature']}"),
            "enthalpy": sm.Symbol(f"{SYMBOLS['enthalpy']}"),
            "volume": sm.Symbol(f"{SYMBOLS['volume']}"),
        }
        """A map containing the symbols for mixture properties considered uniform across
        all phases."""

        self.feed_fractions: dict[str, sm.Symbol] = {}
        """A map containing the symbols for feed fractions (overall molar fraction) per
        component.
        
        The feed fraction of the reference component is eliminated by unity.
        
        """

        self.phase_fractions: dict[str, sm.Symbol] = {}
        """A map containing the symbols for phase molar fractions per phase.

        The molar fraction of the reference phase is eliminated by unity.

        """

        self.phase_saturations: dict[str, sm.Symbol] = {}
        """A map containing the symbols for phase volumetric fractions per phase.

        The saturation of the reference phase is eliminate by unity.

        """

        self.phase_compositions: dict[str, sm.Symbol] = dict()
        """A map containing the symbols for molar component fractions in phases."""

        self.composition_j: dict[str, sm.Symbol] = dict()
        """A map containing symbols for molar component fractions in a phase j,
        independent of the phase index.
        
        They are independent since the expressions comming from an EoS are applied
        to all phases the same way."""

        self.composition_i: dict[str, sm.Symbol] = dict()
        """A map containing generic symbols for molar fractions of a component in all
        all phases, associated with a component i.
        
        They serve for expressions which depend only on the fractions associated with
        a component i, like mass conservation.
        """

        # expression for reference component
        # we do this inconvenient way to preserve the order of components
        s_ = list()
        name_r = ''
        for comp in mixture.components:
            name = f"{SYMBOLS['component_fraction']}_{comp.name}"
            if comp == mixture.reference_component:
                name_r = name
                self.feed_fractions.update({name: 0})
            else:
                s = sm.Symbol(name)
                self.feed_fractions.update({name: s})
                s_.append(s)
        name = f"{SYMBOLS['component_fraction']}_{mixture.reference_component.name}"
        self.feed_fractions.update(
            {name_r: 1 - safe_sum(s_)}
        )

        # expressions for reference phase
        y_ = list()
        s_ = list()
        name_y_r = ''
        name_s_r = ''
        for phase in mixture.phases:
            # symbols for fractions of components in phases
            for comp in mixture.components:
                name = f"{SYMBOLS['phase_composition']}_{comp.name}_{phase.name}"
                self.phase_compositions.update({name: sm.Symbol(name)})

            name_y = f"{SYMBOLS['phase_fraction']}_{phase.name}"
            name_s = f"{SYMBOLS['phase_saturation']}_{phase.name}"
            # skip reference phase due to dependency
            if phase == mixture.reference_phase:
                name_y_r = name_y
                name_s_r = name_s
                self.phase_fractions.update({name_y: 0})
                self.phase_saturations.update({name_s: 0})
            else:
            # symbols for fractions related to phases
                s = sm.Symbol(name_y)
                self.phase_fractions.update({name_y: s})
                y_.append(s)

                s = sm.Symbol(name_s)
                self.phase_saturations.update({name: s})
                s_.append(s)
        self.phase_fractions.update({name_y_r: 1 - safe_sum(y_)})
        self.phase_saturations.update({name_s_r: 1 - safe_sum(s_)})

        # generic phase composition, for a phase j
        for comp in mixture.components:
            name = f"{SYMBOLS['phase_composition']}_{comp.name}_j"
            self.composition_j.update({name: sm.Symbol(name)})

        # generic phase composition, for a component i
        np = 1
        name_r = f"{SYMBOLS['phase_composition']}_i_R"
        for phase in mixture.phases:
            if phase == mixture.reference_phase:
                name = f"{SYMBOLS['phase_composition']}_i_R"
                # add only name key to preserve order from mixture
                self.composition_i.update({name: 0})
            else:
                if phase.gaslike:  # NOTE assumption only 1 gas-like phase
                    name = f"{SYMBOLS['phase_composition']}_i_G"
                else:
                    name = f"{SYMBOLS['phase_composition']}_i_{np}"
                    np += 1
                self.composition_i.update({name: sm.Symbol(name)})
        self.composition_i.update({name_r: sm.Symbol(name_r)})

        # construct shortcut mapping for general access functionality
        for k in self.mixture_properties.keys():
            self._all.update({k: self.mixture_properties})
        for k in self.feed_fractions.keys():
            self._all.update({k: self.feed_fractions})
        for k in self.phase_fractions.keys():
            self._all.update({k: self.phase_fractions})
        for k in self.phase_saturations.keys():
            self._all.update({k: self.phase_saturations})
        for k in self.phase_compositions.keys():
            self._all.update({k: self.phase_compositions})
        for k in self.composition_j.keys():
            self._all.update({k: self.composition_j})
        for k in self.composition_i.keys():
            self._all.update({k: self.composition_i})

    def __call__(self, symbol_name: str) -> sm.Symbol:
        """
        Parameters:
            symbol_name: A name of any symbol contained in this class.

        Raises:
            KeyError: If ``symbol_name`` is unknown.

        Returns:
            The symbol with the passed name.

        """
        d = self._all.get(symbol_name, None)

        if d is None:
            raise KeyError(f"No symbol with name '{symbol_name}' available.")
        else:
            return d[symbol_name]


class PR_Compiler:
    """Class implementing JIT-compiled representation of the equilibrium equations
    using numba and sympy, based on the Peng-Robinson EoS.

    It uses the no-python mode of numba to produce compiled code with near-C-efficiency.
    
    """

    def __init__(self, mixture: NonReactiveMixture) -> None:
        
        self._n_p = mixture.num_phases
        self._n_c = mixture.num_components

        self.symbols: MixtureSymbols = MixtureSymbols(mixture)
        """A container class for symbols representing (potentially) independent
        variables in the equilibrium problem."""

        self.unknowns: dict[str, list[sm.Symbol]] = dict()
        """A map between flash-types (p-T, p-h,...) and the respective list of
        independent variables as symbols."""

        self.arguments: dict[str, list[sm.Symbol]] = dict()
        """A map between flash-types (p-T, p-h,...) and the respective list of
        arguments required for the evaluation of the equations and Jacobians.
        
        The arguments include the fixed values and the unknowns.
        The fixed values for each flash type are the respective specifications e.g.,
        for the p-T flash it includes the symbol for p and T, followed by the symbolic
        unknowns for the p-T flash."""

        self.equations: dict[Literal['p-T', 'p-h', 'v-h'], Callable] = dict()
        """A map between flash-types (p-T, p-h,...) and the equilibrium equations
        represented by multivariate, vector-valued callables."""

        self.jacobians: dict[Literal['p-T', 'p-h', 'v-h'], Callable] = dict()
        """A map between flash-types (p-T, p-h,...) and the Jacobian of respective
        equilibrium equations."""


        self._define_unknowns_and_arguments(mixture)
        self._define_equations_and_jacobians(mixture)

    def _define_unknowns_and_arguments(self, mixture: NonReactiveMixture) -> None:
        """Set the list of symbolic unknowns per flash type."""

        X_pT: list[sm.Symbol] = list()
        X_ph: list[sm.Symbol] = list()
        X_hv: list[sm.Symbol] = list()

        # unknowns for p-T flash
        for phase, y in zip(mixture.phases, self.symbols.phase_fractions.values()):
            if phase != mixture.reference_phase:
                X_pT.append(y)

        X_pT += [x_ij for x_ij in self.symbols.phase_compositions.values()]

        # unknowns for p-h flash
        X_ph += [self.symbols('temperature')] + X_pT

        # unknowns for h-v flash
        X_hv += [self.symbols('pressure'), self.symbols('temperature')]
        for phase, s in zip(mixture.phases, self.symbols.phase_saturations.values()):
            if phase != mixture.reference_phase:
                X_hv.append(s)
        X_hv += X_pT

        self.unknowns.update(
            {
                'p-T': X_pT,
                'p-h': X_ph,
                'v-h': X_hv,
            }
        )

        args_z = [
            z
            for z, comp in zip(self.symbols.feed_fractions.values(), mixture.components)
            if comp != mixture.reference_component
        ]

        # arguments for the p-T system
        args_pT = [self.symbols('pressure'), self.symbols('temperature')] + X_pT

        # arguments for the p-h system
        args_ph = [self.symbols('enthalpy'), self.symbols('pressure')] + X_ph

        # arguments for the h-v system
        args_hv = [self.symbols('volume'), self.symbols('enthalpy')] + X_hv

        self.arguments.update(
            {
                'p-T': args_z + args_pT,
                'p-h': args_z + args_ph,
                'v-h': args_z + args_hv,
            }
        )

    def _define_equations_and_jacobians(self, mixture: NonReactiveMixture):
        """Constructs the equilibrium equatins for each flash specification."""

        ncomp = self._n_c
        nphase = self._n_p

        # region Symbol definition
        # for thermodynamic properties
        p = self.symbols('pressure')
        T = self.symbols('temperature')
        # symbol for compressibility factor, for partial evaluation and derivative
        Z = sm.Symbol('Z')

        # list of fractional unknowns in symbolic form

        # X[j][i] is fraction of component i in phase j
        X_nested = list(self.symbols.phase_compositions.values())
        # molar phase fractions. NOTE first fraction is dependent expression
        Y = list(self.symbols.phase_fractions.values())
        # feed fraction per component. NOTE first fraction is dependent expression
        feed = list(self.symbols.feed_fractions.values())

        # complete array of compositions, ordered per phase, per component
        X: list[list[sm.Symbol]] = list()
        for j in range(self._n_p):
            X_j = X_nested[j * self._n_c : (j + 1) * self._n_c]
            X.append(X_j)

        # generic compositions of a phase
        X_per_phase = list(self.symbols.composition_j.values())
        # generic composition associated with a component i
        X_per_comp = list(self.symbols.composition_i.values())

        # assert number of symbols is consistent
        assert len(Y) == len(X_per_comp), 'Mismatch in generic compositional symbols for a component.'
        assert len(X_per_phase) == len(feed), 'Mismatch in generic comp. symbols per phase.'
        # endregion

        # generic argument for thermodynamic properties
        gen_arg = [p, T] + X_per_phase

        # region Cohesion and Covolume
        # critical covolume per component
        b_i_crit = [
            PengRobinsonEoS.b_crit(comp.p_crit, comp.T_crit)
            for comp in mixture.components
        ]

        # mixed covolume per phase
        b = VanDerWaals.covolume(X_per_phase, b_i_crit)
        B = PengRobinsonEoS.B(b, p, T)
        B_c = numba.njit(sm.lambdify(gen_arg, B))

        # cohesion per phase

        # TODO this is not safe. Special BIP implementations are not sympy compatible
        # and it needs an instantiation of a PR EOS
        bips, dT_bips = mixture.reference_phase.eos._compute_bips(T)

        a_i_crit = [
            PengRobinsonEoS.a_crit(comp.p_crit, comp.T_crit)
            for comp in mixture.components
        ]

        k_i = [
            PengRobinsonEoS.a_correction_weight(comp.omega)
            for comp in mixture.components
        ]

        a_i_correction = [
            1 + k * (1 - sm.sqrt(T / comp.T_crit)) for k, comp in zip(k_i, mixture.components)
        ]

        # cohesion term per component
        a_i = [a * corr**2 for a, corr in zip(a_i_crit, a_i_correction)]

        # mixed cohesion terms per phase
        a = VanDerWaals.cohesion_s(X_per_phase, a_i, bips)
        A = PengRobinsonEoS.A(a, p, T)
        A_c = numba.njit(sm.lambdify(gen_arg, A))

        dT_A = A.diff(T)
        dXi_A = [A.diff(x_) for x_ in X_per_phase]
        # endregion

        # region Compressibility factor computation
        # constructing expressiong for compressibility factors dependent on p-T-X

        Z_triple = triple_root(A, B)  # triple root case
        Z_triple_c = numba.njit(sm.lambdify(gen_arg, Z_triple))

        # need math module, not numpy, because of piecewise nature
        # NOTE np.select not supported by numba
        # NOTE need special substitution of sqrt and cbrt for numerical stability
        Z_one = one_root(A, B)  # one real root case
        Z_one_c = numba.njit(
            sm.lambdify(gen_arg, Z_one, [{"cbrt": np.cbrt, "sqrt": np.sqrt}, "math"]),
        )
        
        Z_ext_sub = ext_root_gharbia(Z_one, B)  # extended subcritical root
        Z_ext_sub_c = numba.njit(
            sm.lambdify(gen_arg, Z_ext_sub, [{"cbrt": np.cbrt, "sqrt": np.sqrt}, "math"]),
        )
        Z_ext_scg = ext_root_scg(Z_one, B)  # extended supercritical gas root
        Z_ext_scg_c = numba.njit(
            sm.lambdify(gen_arg, Z_ext_scg, [{"cbrt": np.cbrt, "sqrt": np.sqrt}, "math"]),
        )
        Z_ext_scl = ext_root_scl(Z_one, B)  # extended supercritical liquid root
        Z_ext_scl_c = numba.njit(
            sm.lambdify(gen_arg, Z_ext_scl, [{"cbrt": np.cbrt, "sqrt": np.sqrt}, "math"]),
        )

        # need math module again because of piecewise operation
        Z_double_g = double_root(A, B, True)  # gas-like root in double root case
        Z_double_g_c = numba.njit(
            sm.lambdify(gen_arg, Z_double_g, 'math'),
        )
        Z_double_l = double_root(A, B, False)  # liquid-like root in double root case
        Z_double_l_c = numba.njit(
            sm.lambdify(gen_arg, Z_double_l, 'math'),
        )

        Z_three_g = three_root(A, B, True)  # gas-like root in 3-root case
        Z_three_g_c = numba.njit(sm.lambdify(gen_arg, Z_three_g))
        Z_three_l = three_root(A, B, False)  # liquid-like root in 3-root case
        Z_three_l_c = numba.njit(sm.lambdify(gen_arg, Z_three_l))
        Z_three_i = three_root_intermediate(A, B)  # intermediate root in 3-root case
        Z_three_i_c = numba.njit(sm.lambdify(gen_arg, Z_three_i))

        # list containing gas-like flags per phase to decide which to use in the
        # 2- or 3-root case
        gaslike: list[bool] = [phase.gaslike for phase in mixture.phases]

        @numba.njit
        def Z_c(
            gaslike: bool,
            p: float,
            T: float,
            X: np.ndarray,
            eps: float=1e-14,
            smooth_e: float = 1e-2,
            smooth_3: float = 1e-3,
        )-> float:
            """Wrapper function computing the compressibility factor for given
            thermodynamic state.

            This function provides the labeling and smoothing, on top of the wrapped
            computations of Z.

            Smoothing can be disabled by setting respective argument to zero.

            ``eps`` is used to determine the root case :func:`get_root_case_c`.

            """
            # X = (_ for _ in X)
            X = X.tolist()  # TODO
            A_ = A_c(p, T, *X)
            B_ = B_c(p, T, *X)

            # super critical check
            is_sc = B_ >= critical_line_c(A_)

            below_widom = B_ <= widom_line_c(A_)

            nroot = get_root_case_c(A_, B_, eps)

            if nroot == 1:
                Z_1_real = Z_one_c(p, T, *X)
                # Extension procedure according Ben Gharbia et al.
                if not is_sc and B_ < B_CRIT:
                    W = Z_ext_sub_c(p, T, *X)
                    if below_widom:
                        return W if gaslike else Z_1_real
                    else:
                        return Z_1_real if gaslike else W
                # Extension procedure with asymmetric extension of gas
                elif below_widom and B_ >= B_CRIT:
                    if gaslike:
                        W = Z_ext_scg_c(p, T, *X)

                        # computing distance to border to subcritical extension
                        # smooth if close
                        d = point_to_line_distance_c(
                            np.array([A_, B_]), *B_CRIT_LINE_POINTS
                        )
                        if smooth_e > 0. and d < smooth_e:
                            d_n = d / smooth_e
                            W = Z_ext_sub_c(p, T, *X) * (1 - d_n) + W * d_n

                        return W
                    else:
                        return Z_1_real
                # Extension procedure with asymmetric extension of liquid
                else:
                    if gaslike:
                        return Z_1_real
                    else:
                        W = Z_ext_scl_c(p, T, *X)
                        ab_ = np.array([A_, B_])

                        # computing distance to Widom-line,
                        # which separates gas and liquid in supercrit area
                        d = point_to_line_distance_c(ab_, *W_LINE_POINTS)
                        if smooth_e > 0. and d < smooth_e and B_ >= B_CRIT:
                            d_n = d / smooth_e
                            W = Z_ext_scg_c(p, T, *X) * (1 - d_n) + W * d_n

                        # Computing distance to supercritical line,
                        # which separates sub- and supercritical liquid extension
                        d = point_to_line_distance_c(ab_, *S_CRIT_LINE_POINTS)
                        if smooth_e > 0. and d < smooth_e and B_ < B_CRIT:
                            d_n = d / smooth_e
                            W = Z_ext_sub_c(p, T, *X) * (1 - d_n) + W * d_n

                        return W
            elif nroot == 2:
                if gaslike:
                    return Z_double_g_c(p, T, *X)
                else:
                    return Z_double_l_c(p, T, *X)
            elif nroot == 3:
                # triple root area above the critical line is substituted with the
                # extended supercritical liquid-like root
                if is_sc:
                    if gaslike:
                        return Z_three_g_c(p, T, *X)
                    else:
                        W = Z_ext_scl_c(p, T, *X)
                        ab_ = np.array([A_, B_])

                        # computing distance to Widom-line,
                        # which separates gas and liquid in supercrit area
                        d = point_to_line_distance_c(ab_, *W_LINE_POINTS)
                        if smooth_e > 0. and d < smooth_e and B_ >= B_CRIT:
                            d_n = d / smooth_e
                            W = Z_ext_scg_c(p, T, *X) * (1 - d_n) + W * d_n

                        # Computing distance to supercritical line,
                        # which separates sub- and supercritical liquid extension
                        d = point_to_line_distance_c(ab_, *S_CRIT_LINE_POINTS)
                        if smooth_e > 0. and d < smooth_e and B_ < B_CRIT:
                            d_n = d / smooth_e
                            W = Z_ext_sub_c(p, T, *X) * (1 - d_n) + W * d_n

                        return W
                else:
                    # smoothing according Ben Gharbia et al., in physical 2-phase region
                    if smooth_3 > 0.:
                        Z_l = Z_three_l_c(p, T, *X)
                        Z_i = Z_three_i_c(p, T, *X)
                        Z_g = Z_three_g_c(p, T, *X)
                        
                        d = (Z_i - Z_l) / (Z_g - Z_l)

                        # gas root smoothing
                        if gaslike:
                            # gas root smoothing weight
                            v_g = (d - (1 - 2 * smooth_3)) / smooth_3
                            v_g = v_g ** 2 * (3 - 2 * v_g)
                            if d >= 1 - smooth_3:
                                v_g = 1.
                            elif d <= 1 - 2 * smooth_3:
                                v_g = 0.

                            return Z_g * (1 - v_g) + (Z_i + Z_g) * .5 * v_g
                        # liquid root smoothing
                        else:

                            v_l = (d - smooth_3) / smooth_3
                            v_l = -v_l**2 * (3 - 2 * v_l) + 1.
                            if d <= smooth_3:
                                v_l = 1.
                            elif d >= 2 * smooth_3:
                                v_l = 0.

                            return Z_l * (1 - v_l) + (Z_i + Z_l) * .5 * v_l
                    else:
                        return Z_three_g_c(p, T, *X) if gaslike else Z_three_l_c(p, T, *X)

            else:
                return Z_triple_c(p, T, *X)
            
        # Z_cv = numba.vectorize(
        #     [
        #         numba.float64(
        #             numba.bool_,
        #             numba.float64,
        #             numba.float64,
        #             *(numba.float64 for _ in range(len(X_per_phase))),
        #             numba.float64,
        #             numba.float64,
        #             numba.float64,
        #         )
        #     ],
        #     nopython=True,
        # )(Z_c)
        # endregion

        p_ = 1e6
        T_ = 300
        X__ = np.array([0.3, 0.7])
        se = 1e-2
        s3 = 1e-3
        eps = 1e-14

        z_ = Z_c(True, p_, T_, X__, eps=eps, smooth_e=se, smooth_3=s3)

        # region Symbolic equations
        NJIT_KWARGS_equ = {
            'fastmath': True,
        }

        # symbolic expression for mass conservation without the feed fraction
        mass = safe_sum([-y * x for y, x in zip(Y, X_per_comp)])
        d_mass = [mass.diff(_) for _ in Y[1:] + X_per_comp]

        # callables taking fractional unknowns as individual arguments
        mass = sm.lambdify(Y[1:] + X_per_comp, mass)
        d_mass = sm.lambdify(Y[1:] + X_per_comp, d_mass)

        mass_c = numba.njit(mass, **NJIT_KWARGS_equ)
        d_mass_c = njit_diffs(d_mass, **NJIT_KWARGS_equ)

        # symbolic expression for complementary condition y_j * (1 - sum_i x_ij)
        cc = [y_ * (1 - safe_sum(X_per_phase)) for y_ in Y]
        d_cc = [[cc_.diff(_) for _ in Y[1:] + X_per_phase] for cc_ in cc]

        cc = [sm.lambdify(Y[1:] + X_per_phase, cc_) for cc_ in cc]
        d_cc = [sm.lambdify(Y[1:] + X_per_phase, d_cc_) for d_cc_ in d_cc]

        cc_c = [numba.njit(cc_, **NJIT_KWARGS_equ) for cc_ in cc]
        d_cc_c = [njit_diffs(d_cc_, **NJIT_KWARGS_equ) for d_cc_ in d_cc]
        # endregion

        @numba.njit
        def _parse_xyz(X_gen: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Helper function to parse the fractions from a generic argument."""
            # feed fraction per component, except reference component
            Z = X_gen[:ncomp - 1]
            # phase compositions
            X = X_gen[- ncomp * nphase:]
            # matrix:
            # rows have compositions per phase,
            # columns have compositions related to a component
            X = X.reshape((ncomp, nphase))
            # phase fractions, -1 because fraction of ref phase is eliminated
            Y = X_gen[- (ncomp * nphase + nphase - 1): - ncomp * nphase]

            return X, Y, Z

        @numba.njit
        def _parse_pT(X_gen: np.ndarray) -> np.ndarray:
            """Helper function extracint pressure and temperature from a generic
            argument."""
            return X_gen[-(ncomp*nphase + nphase -1) - 2: -(ncomp*nphase + nphase -1)]

        # region p-T flash
        def F_pT(X_gen: np.ndarray) -> np.ndarray:
            """Callable representing the p-T flash system"""

            X, Y, Z = _parse_xyz(X_gen)
            p, T = _parse_pT(X_gen)

            # maxx conservation excluding first component
            # NOTE this assumes the first set of compositions in component order belongs
            # to the reference component
            mass = np.array(
                [z + mass_c(*Y, *xi) for z, xi in zip(Z, X.T[1:])]
            )

            cc = np.array([cc_c_(*Y, *xj) for cc_c_, xj in zip(cc_c, X)])

            return np.concatenate((mass, cc))
        
        def DF_pT(X_gen: np.ndarray) -> np.ndarray:

            # degrees of freedom include compositions and independent phase fractions
            dofs = ncomp * nphase + nphase - 1
            # empty dense matrix. NOTE numba can deal only with dense np arrays
            DF = np.zeros((dofs, dofs))

            X, Y, Z = _parse_xyz(X_gen)

            d_mass = np.array(
                [d_mass_c(*Y, *xi) for xi in X.T[1:]]
            )

            d_cc = np.array([d_cc_c_(*Y, *xj) for d_cc_c_, xj in zip(d_cc_c, X)])

            for i in range(ncomp):
                # insert derivatives of mass conservations for component i != ref comp
                if i < ncomp - 1:
                    # derivatives w.r.t. phase fractions
                    DF[i, : nphase - 1] = d_mass[i, : nphase - 1]
                    # derivatives w.r.t compositional fractions of that component
                    DF[i, nphase + i::nphase] = d_mass[i, nphase-1:]
            for j in range(nphase):

                # inserting derivatives of complementary conditions
                # derivatives w.r.t phase fractions
                DF[ncomp*nphase - 1 + j, :nphase - 1] = d_cc[j, :nphase - 1]
                # derivatives w.r.t compositions of that phase
                DF[ncomp*nphase - 1 + j, nphase - 1 + j ::ncomp] = d_cc[j, nphase-1:]

            return DF
        # endregion

        t = np.array([0.01, 1e6, 300, 0.9, 0.1, 0.2, 0.3, 0.4])

        print(F_pT(t))
        print(DF_pT(t))

        self.equations.update(
            {
                'p-T': F_pT,
            }
        )

        self.jacobians.update(
            {
                'p-T': DF_pT
            }
        )