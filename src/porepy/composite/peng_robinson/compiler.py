"""Experimentel code for efficient unified flash calculations using numba and sympy."""
from __future__ import annotations

import os

# os.environ['NUMBA_DISABLE_INTEL_SVML']  = '1'
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "30"

from typing import Callable, Literal, Sequence

import numba
import numpy as np
import sympy as sm

from .._core import COMPOSITIONAL_VARIABLE_SYMBOLS as SYMBOLS
from ..composite_utils import normalize_fractions, safe_sum
from ..mixture import BasicMixture, NonReactiveMixture
from .eos import A_CRIT, B_CRIT, Z_CRIT, PengRobinsonEoS
from .mixing import VanDerWaals

__all__ = ["MixtureSymbols", "PR_Compiler"]


# TODO this needs more work s.t. I must not replace a call to this in lambdified expr.
class cbrt(sm.Function):
    """Custom symbolic cubic root to circumvent sympy using the power expression.

    The power expression is costly and does not always work with negative numbers.
    It returns sometimes not the principle cubic root (which is always real).

    TODO more tests are required here.
    TODO make sure it is upon usage of sympy.lambdify replaced by the module functions
    math.cbrt and numpy.cbrt (which always return the principle root).
    """

    def fdiff(self, argindex=1):
        """Custom implementation of derivative of cubic root such that it always
        returns a positive, real number."""
        a = self.args[0]
        return 1.0 / (cbrt(a**2) * 3)


_COEFF_COMPILTER_ARGS = {
    "fastmath": True,
    "cache": True,
}


def coef0(A, B):
    """Coefficient for the zeroth monomial."""
    return B**3 + B**2 - A * B


coef0_c = numba.njit(**_COEFF_COMPILTER_ARGS)(coef0)


def coef1(A, B):
    """Coefficient for the first monomial."""
    return A - 2.0 * B - 3.0 * B**2


coef1_c = numba.njit(**_COEFF_COMPILTER_ARGS)(coef1)


def coef2(A, B):
    """Coefficient for the second monomial."""
    return B - 1


coef2_c = numba.njit(**_COEFF_COMPILTER_ARGS)(coef2)


def red_coef0(A, B):
    """Zeroth coefficient of the reduced polynomial."""
    c2 = coef2(A, B)
    return c2**3 * (2.0 / 27.0) - c2 * coef1(A, B) * (1.0 / 3.0) + coef0(A, B)


@numba.njit(**_COEFF_COMPILTER_ARGS)
def red_coef0_c(A, B):
    c2 = coef2_c(A, B)
    return c2**3 * (2.0 / 27.0) - c2 * coef1_c(A, B) * (1.0 / 3.0) + coef0_c(A, B)


def red_coef1(A, B):
    """First coefficient of the reduced polynomial."""
    return coef1(A, B) - coef2(A, B) ** 2 * (1.0 / 3.0)


@numba.njit(**_COEFF_COMPILTER_ARGS)
def red_coef1_c(A, B):
    return coef1_c(A, B) - coef2_c(A, B) ** 2 * (1.0 / 3.0)


def discr(rc0, rc1):
    """Discriminant of the polynomial based on the zeroth and first reduced coefficient."""
    return rc0**2 * (1.0 / 4.0) + rc1**3 * (1.0 / 27.0)


discr_c = numba.njit(**_COEFF_COMPILTER_ARGS)(discr)


@numba.njit(**_COEFF_COMPILTER_ARGS)
def get_root_case_c(A, B, eps=1e-14):
    """ "An piece-wise cosntant function dependent on
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


def triple_root(A: sm.Expr, B: sm.Expr) -> sm.Expr:
    """Formula for tripple root. Only valid if triple root case."""
    c2 = coef2(A, B)
    return -c2 / 3


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
    t_2 = sm.acos(-q / 2 * sm.sqrt(-27 * r ** (-3))) / 3
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

    t_2 = sm.acos(-q / 2 * sm.sqrt(-27 * r ** (-3))) / 3
    t_1 = sm.sqrt(-4 / 3 * r)

    return -t_1 * sm.cos(t_2 + np.pi / 3) - c2 / 3


def one_root(A: sm.Expr, B: sm.Expr) -> sm.Expr:
    """Returns the single real root in the 1-root case."""
    c2 = coef2(A, B)
    q = red_coef0(A, B)
    r = red_coef1(A, B)
    d = discr(q, r)

    t1 = sm.sqrt(d) - q * 0.5
    t2 = -1 * (sm.sqrt(d) + q * 0.5)

    t = sm.Piecewise((t2, sm.Abs(t2) > sm.Abs(t1)), (t1, True))

    u = cbrt(t)  # TODO potential source of error

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


def njit_diffs_v(f, dtype=np.float64, **njit_kwargs):
    """For wrapping symbolic, multivariate derivatives which return a sequence,
    instead of a numpy array.

    Intended for functions which take the multivariate, scalar input in array form.

    """
    f = numba.njit(f, **njit_kwargs)

    @numba.njit(**njit_kwargs)
    def inner(x):
        return np.array(f(x), dtype=dtype)

    return inner


def njit_diffs(f, dtype=np.float64, **njit_kwargs):
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
        (lp2[0] - lp1[0]) * (lp1[1] - p[1]) - (lp1[0] - p[0]) * (lp2[1] - lp1[1])
    )
    return n / d


B_CRIT_LINE_POINTS = (np.array([0.0, B_CRIT]), np.array([A_CRIT, B_CRIT]))


S_CRIT_LINE_POINTS = (np.zeros(2), np.array([A_CRIT, B_CRIT]))


W_LINE_POINTS = (
    np.array([0.0, widom_line_c(0)]),
    np.array([A_CRIT, widom_line_c(A_CRIT)]),
)


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

        self.composition_of_phase_j: dict[str, sm.Symbol] = dict()
        """A map containing symbols for molar component fractions in a phase j,
        independent of the phase index.

        They are independent since the expressions comming from an EoS are applied
        to all phases the same way."""

        self.composition_per_component_i: dict[str, sm.Symbol] = dict()
        """A map containing generic symbols for molar fractions of a component in all
        all phases, associated with a component i.

        They serve for expressions which depend only on the fractions associated with
        a component i, like mass conservation.
        """

        # expression for reference component
        # we do this inconvenient way to preserve the order of components
        s_ = list()
        name_r = ""
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
        self.feed_fractions.update({name_r: 1 - safe_sum(s_)})

        # expressions for reference phase
        y_ = list()
        s_ = list()
        name_y_r = ""
        name_s_r = ""
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
            self.composition_of_phase_j.update({name: sm.Symbol(name)})

        # generic phase composition, for a component i
        np = 1
        name_r = f"{SYMBOLS['phase_composition']}_i_R"
        for phase in mixture.phases:
            if phase == mixture.reference_phase:
                name = f"{SYMBOLS['phase_composition']}_i_R"
                # add only name key to preserve order from mixture
                self.composition_per_component_i.update({name: 0})
            else:
                if phase.gaslike:  # NOTE assumption only 1 gas-like phase
                    name = f"{SYMBOLS['phase_composition']}_i_G"
                else:
                    name = f"{SYMBOLS['phase_composition']}_i_{np}"
                    np += 1
                self.composition_per_component_i.update({name: sm.Symbol(name)})
        self.composition_per_component_i.update({name_r: sm.Symbol(name_r)})

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
        for k in self.composition_of_phase_j.keys():
            self._all.update({k: self.composition_of_phase_j})
        for k in self.composition_per_component_i.keys():
            self._all.update({k: self.composition_per_component_i})

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

        self.equations: dict[Literal["p-T", "p-h", "v-h"], Callable] = dict()
        """A map between flash-types (p-T, p-h,...) and the equilibrium equations
        represented by multivariate, vector-valued callables."""

        self.jacobians: dict[Literal["p-T", "p-h", "v-h"], Callable] = dict()
        """A map between flash-types (p-T, p-h,...) and the Jacobian of respective
        equilibrium equations."""

        self.cfuncs: dict[str, Callable] = dict()
        """Contains a collection of numba-compiled callables representing thermodynamic
        properties.

        Keys are names of the properties. Standard symbols from ltierature are used.

        Several naming conventions apply:

        - ``_cv``: compiled and vectorized. Arguments can be numpy arrays and the return
          value is an array
          (see `here <https://numba.readthedocs.io/en/stable/user/vectorize.html>`_).
          If ``_cv`` is not indicated, the function takes only scalar input.
        - ``dx_y``: The partial derivative of ``y`` w.r.t. ``x``.
        - ``d_y``: The complete derivative of ``y``. If ``y`` is multivariate,
          the return value is an array of length equal the number of dependencies
          (in respective order).

        Important:
            The functions are assembled and compiled at runtime.
            For general multiphase, multicomponent mixtures, this implies that the
            number of input args varies. The user must be aware of the dependencies
            explained in the class documentation.

        """

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
        X_ph += [self.symbols("temperature")] + X_pT

        # unknowns for h-v flash
        X_hv += [self.symbols("pressure"), self.symbols("temperature")]
        for phase, s in zip(mixture.phases, self.symbols.phase_saturations.values()):
            if phase != mixture.reference_phase:
                X_hv.append(s)
        X_hv += X_pT

        self.unknowns.update(
            {
                "p-T": X_pT,
                "p-h": X_ph,
                "v-h": X_hv,
            }
        )

        args_z = [
            z
            for z, comp in zip(self.symbols.feed_fractions.values(), mixture.components)
            if comp != mixture.reference_component
        ]

        # arguments for the p-T system
        args_pT = [self.symbols("pressure"), self.symbols("temperature")] + X_pT

        # arguments for the p-h system
        args_ph = [self.symbols("enthalpy"), self.symbols("pressure")] + X_ph

        # arguments for the h-v system
        args_hv = [self.symbols("volume"), self.symbols("enthalpy")] + X_hv

        self.arguments.update(
            {
                "p-T": args_z + args_pT,
                "p-h": args_z + args_ph,
                "v-h": args_z + args_hv,
            }
        )

    def _define_equations_and_jacobians(self, mixture: NonReactiveMixture):
        """Constructs the equilibrium equatins for each flash specification."""

        ncomp = self._n_c
        nphase = self._n_p

        # NOTE This is a rather lengthy piece of code with many quantities.
        # In order to keep an overview, following conventions are made
        # Naming convention:
        # 1. _s is the symbolic representation of an independent quantity (sympy.Symbol)
        # 2. _e is the symbolic representation of a dependent quantity (sympy.Expr)
        # 3. _c is the numba compiled, callable of respective _e

        # region Symbol definition
        # for thermodynamic properties
        p_s = self.symbols("pressure")
        T_s = self.symbols("temperature")
        # symbol for compressibility factor, for partial evaluation and derivative
        Z_s = sm.Symbol("Z")
        # symbols for non-dimensional cohesion and covolume
        A_s = sm.Symbol("A")
        B_s = sm.Symbol("B")

        # list of fractional unknowns in symbolic form
        # molar phase fractions. NOTE first fraction is dependent expression
        Y_s = list(self.symbols.phase_fractions.values())
        # feed fraction per component. NOTE first fraction is dependent expression
        feed = list(self.symbols.feed_fractions.values())
        # generic compositions of a phase
        X_in_j_s = list(self.symbols.composition_of_phase_j.values())
        # generic composition associated with a component i
        X_per_i_s = list(self.symbols.composition_per_component_i.values())

        # assert number of symbols is consistent
        assert len(Y_s) == len(
            X_per_i_s
        ), "Mismatch in generic compositional symbols for a component."
        assert len(X_in_j_s) == len(
            feed
        ), "Mismatch in generic comp. symbols per phase."
        # endregion

        # generic argument for thermodynamic properties of a phase
        thd_arg_j = [p_s, T_s, X_in_j_s]

        def _diff(expr_, thd_arg_):
            """Helper function to define the gradient of a multivariate function,
            where the argument has the special structure of ``thd_arg_j``."""
            p_, T_, X_j_ = thd_arg_
            return [expr_.diff(p_), expr_.diff(T_)] + [expr_.diff(_) for _ in X_j_]

        # region Cohesion and Covolume
        # critical covolume per component
        b_i_crit = [
            PengRobinsonEoS.b_crit(comp.p_crit, comp.T_crit)
            for comp in mixture.components
        ]

        # mixed covolume per phase
        b_e = VanDerWaals.covolume(X_in_j_s, b_i_crit)
        B_e = PengRobinsonEoS.B(b_e, p_s, T_s)
        B_c = numba.njit(sm.lambdify(thd_arg_j, B_e), fastmath=True)

        # d_B_e = [B_e.diff(_) for _ in thd_arg_j]
        d_B_e = _diff(B_e, thd_arg_j)
        d_B_c = njit_diffs(sm.lambdify(thd_arg_j, d_B_e), fastmath=True)

        # cohesion per phase

        # TODO this is not safe. Special BIP implementations are not sympy compatible
        # and it needs an instantiation of a PR EOS
        bips, dT_bips = mixture.reference_phase.eos.compute_bips(T_s)

        ai_crit: list[float] = [
            PengRobinsonEoS.a_crit(comp.p_crit, comp.T_crit)
            for comp in mixture.components
        ]

        ki: list[float] = [
            PengRobinsonEoS.a_correction_weight(comp.omega)
            for comp in mixture.components
        ]

        ai_correction_e = [
            1 + k * (1 - sm.sqrt(T_s / comp.T_crit))
            for k, comp in zip(ki, mixture.components)
        ]

        # cohesion term per component
        ai_e = [a * corr**2 for a, corr in zip(ai_crit, ai_correction_e)]

        # mixed cohesion terms per phase
        a_e = VanDerWaals.cohesion_s(X_in_j_s, ai_e, bips)
        A_e = PengRobinsonEoS.A(a_e, p_s, T_s)
        A_c = numba.njit(sm.lambdify(thd_arg_j, A_e))

        d_A_e = _diff(A_e, thd_arg_j)
        d_A_c = njit_diffs(sm.lambdify(thd_arg_j, d_A_e))

        dT_A_e = A_e.diff(T_s)
        dXi_A_e = [A_e.diff(x_) for x_ in X_in_j_s]

        # endregion

        # region Compressibility factor computation
        # constructing expressiong for compressibility factors dependent on p-T-X
        # NOTE: need math module, not numpy,
        # because of piecewise nature of some root expressions
        # np.select not supported by numba

        AB_arg = [A_s, B_s]

        Z_triple_e = triple_root(A_s, B_s)  # triple root case
        Z_triple_c = numba.njit(sm.lambdify(AB_arg, Z_triple_e))

        d_Z_triple_e = [Z_triple_e.diff(_) for _ in AB_arg]
        d_Z_triple_c = njit_diffs(sm.lambdify(AB_arg, d_Z_triple_e))

        # Need to substitute the custom function with numpy.cbrt
        # Need 'math' built-in module because of piecewise nature
        # But numba has no support for math.cbrt .... hence replace it explicitely.
        _lam = [{"cbrt": np.cbrt}, "math"]
        # _lam = 'math'

        Z_one_e = one_root(A_s, B_s)  # one real root case
        Z_one_c = numba.njit(sm.lambdify(AB_arg, Z_one_e, _lam))

        d_Z_one_e = [Z_one_e.diff(_) for _ in AB_arg]
        d_Z_one_c = njit_diffs(sm.lambdify(AB_arg, d_Z_one_e, _lam))

        Z_ext_sub_e = ext_root_gharbia(Z_one_e, B_s)  # extended subcritical root
        Z_ext_sub_c = numba.njit(sm.lambdify(AB_arg, Z_ext_sub_e, _lam))
        Z_ext_scg_e = ext_root_scg(Z_one_e, B_s)  # extended supercritical gas root
        Z_ext_scg_c = numba.njit(sm.lambdify(AB_arg, Z_ext_scg_e, _lam))
        Z_ext_scl_e = ext_root_scl(Z_one_e, B_s)  # extended supercritical liquid root
        Z_ext_scl_c = numba.njit(sm.lambdify(AB_arg, Z_ext_scl_e, _lam))

        d_Z_ext_sub_e = [Z_ext_sub_e.diff(_) for _ in AB_arg]
        d_Z_ext_sub_c = njit_diffs(sm.lambdify(AB_arg, d_Z_ext_sub_e, _lam))
        d_Z_ext_scg_e = [Z_ext_scg_e.diff(_) for _ in AB_arg]
        d_Z_ext_scg_c = njit_diffs(sm.lambdify(AB_arg, d_Z_ext_scg_e, _lam))
        d_Z_ext_scl_e = [Z_ext_scl_e.diff(_) for _ in AB_arg]
        d_Z_ext_scl_c = njit_diffs(sm.lambdify(AB_arg, d_Z_ext_scl_e, _lam))

        # need math module again because of piecewise operation
        Z_double_g_e = double_root(A_s, B_s, True)  # gas-like root in double root case
        Z_double_g_c = numba.njit(sm.lambdify(AB_arg, Z_double_g_e, "math"))
        Z_double_l_e = double_root(
            A_s, B_s, False
        )  # liquid-like root in double root case
        Z_double_l_c = numba.njit(sm.lambdify(AB_arg, Z_double_l_e, "math"))

        d_Z_double_g_e = [Z_double_g_e.diff(_) for _ in AB_arg]
        d_Z_double_g_c = njit_diffs(sm.lambdify(AB_arg, d_Z_double_g_e, "math"))
        d_Z_double_l_e = [Z_double_l_e.diff(_) for _ in AB_arg]
        d_Z_double_l_c = njit_diffs(sm.lambdify(AB_arg, d_Z_double_l_e, "math"))

        Z_three_g_e = three_root(A_s, B_s, True)  # gas-like root in 3-root case
        Z_three_g_c = numba.njit(sm.lambdify(AB_arg, Z_three_g_e))
        Z_three_l_e = three_root(A_s, B_s, False)  # liquid-like root in 3-root case
        Z_three_l_c = numba.njit(sm.lambdify(AB_arg, Z_three_l_e))
        Z_three_i_e = three_root_intermediate(A_s, B_s)  # interm. root in 3-root case
        Z_three_i_c = numba.njit(sm.lambdify(AB_arg, Z_three_i_e))

        d_Z_three_g_e = [Z_three_g_e.diff(_) for _ in AB_arg]
        d_Z_three_g_c = njit_diffs(sm.lambdify(AB_arg, d_Z_three_g_e))
        d_Z_three_l_e = [Z_three_l_e.diff(_) for _ in AB_arg]
        d_Z_three_l_c = njit_diffs(sm.lambdify(AB_arg, d_Z_three_l_e))
        d_Z_three_i_e = [Z_three_i_e.diff(_) for _ in AB_arg]
        d_Z_three_i_c = njit_diffs(sm.lambdify(AB_arg, d_Z_three_i_e))

        @numba.njit
        def Z_c(
            gaslike: int,
            p: float,
            T: float,
            X: Sequence[float],
            eps: float = 1e-14,
            smooth_e: float = 1e-2,
            smooth_3: float = 1e-3,
        ) -> float:
            """Wrapper function computing the compressibility factor for given
            thermodynamic state.

            This function provides the labeling and smoothing, on top of the wrapped
            computations of Z.

            It expresses Z in terms of p, T and composition, instead of A and B.

            Smoothing can be disabled by setting respective argument to zero.

            ``eps`` is used to determine the root case :func:`get_root_case_c`.

            """

            A_val = A_c(p, T, X)
            B_val = B_c(p, T, X)

            # super critical check
            is_sc = B_val >= critical_line_c(A_val)

            below_widom = B_val <= widom_line_c(A_val)

            nroot = get_root_case_c(A_val, B_val, eps)

            if nroot == 1:
                Z_1_real = Z_one_c(A_val, B_val)
                # Extension procedure according Ben Gharbia et al.
                if not is_sc and B_val < B_CRIT:
                    W = Z_ext_sub_c(A_val, B_val)
                    if below_widom:
                        return W if gaslike else Z_1_real
                    else:
                        return Z_1_real if gaslike else W
                # Extension procedure with asymmetric extension of gas
                elif below_widom and B_val >= B_CRIT:
                    if gaslike:
                        W = Z_ext_scg_c(A_val, B_val)

                        # computing distance to border to subcritical extension
                        # smooth if close
                        d = point_to_line_distance_c(
                            np.array([A_val, B_val]), *B_CRIT_LINE_POINTS
                        )
                        if smooth_e > 0.0 and d < smooth_e:
                            d_n = d / smooth_e
                            W = Z_ext_sub_c(A_val, B_val) * (1 - d_n) + W * d_n

                        return W
                    else:
                        return Z_1_real
                # Extension procedure with asymmetric extension of liquid
                else:
                    if gaslike:
                        return Z_1_real
                    else:
                        W = Z_ext_scl_c(A_val, B_val)
                        ab_ = np.array([A_val, B_val])

                        # computing distance to Widom-line,
                        # which separates gas and liquid in supercrit area
                        d = point_to_line_distance_c(ab_, *W_LINE_POINTS)
                        if smooth_e > 0.0 and d < smooth_e and B_val >= B_CRIT:
                            d_n = d / smooth_e
                            W = Z_ext_scg_c(A_val, B_val) * (1 - d_n) + W * d_n

                        # Computing distance to supercritical line,
                        # which separates sub- and supercritical liquid extension
                        d = point_to_line_distance_c(ab_, *S_CRIT_LINE_POINTS)
                        if smooth_e > 0.0 and d < smooth_e and B_val < B_CRIT:
                            d_n = d / smooth_e
                            W = Z_ext_sub_c(A_val, B_val) * (1 - d_n) + W * d_n

                        return W
            elif nroot == 2:
                if gaslike:
                    return Z_double_g_c(A_val, B_val)
                else:
                    return Z_double_l_c(A_val, B_val)
            elif nroot == 3:
                # triple root area above the critical line is substituted with the
                # extended supercritical liquid-like root
                if is_sc:
                    if gaslike:
                        return Z_three_g_c(A_val, B_val)
                    else:
                        W = Z_ext_scl_c(A_val, B_val)
                        ab_ = np.array([A_val, B_val])

                        # computing distance to Widom-line,
                        # which separates gas and liquid in supercrit area
                        d = point_to_line_distance_c(ab_, *W_LINE_POINTS)
                        if smooth_e > 0.0 and d < smooth_e and B_val >= B_CRIT:
                            d_n = d / smooth_e
                            W = Z_ext_scg_c(A_val, B_val) * (1 - d_n) + W * d_n

                        # Computing distance to supercritical line,
                        # which separates sub- and supercritical liquid extension
                        d = point_to_line_distance_c(ab_, *S_CRIT_LINE_POINTS)
                        if smooth_e > 0.0 and d < smooth_e and B_val < B_CRIT:
                            d_n = d / smooth_e
                            W = Z_ext_sub_c(A_val, B_val) * (1 - d_n) + W * d_n

                        return W
                else:
                    # smoothing according Ben Gharbia et al., in physical 2-phase region
                    if smooth_3 > 0.0:
                        Z_l = Z_three_l_c(A_val, B_val)
                        Z_i = Z_three_i_c(A_val, B_val)
                        Z_g = Z_three_g_c(A_val, B_val)

                        d = (Z_i - Z_l) / (Z_g - Z_l)

                        # gas root smoothing
                        if gaslike:
                            # gas root smoothing weight
                            v_g = (d - (1 - 2 * smooth_3)) / smooth_3
                            v_g = v_g**2 * (3 - 2 * v_g)
                            if d >= 1 - smooth_3:
                                v_g = 1.0
                            elif d <= 1 - 2 * smooth_3:
                                v_g = 0.0

                            return Z_g * (1 - v_g) + (Z_i + Z_g) * 0.5 * v_g
                        # liquid root smoothing
                        else:
                            v_l = (d - smooth_3) / smooth_3
                            v_l = -(v_l**2) * (3 - 2 * v_l) + 1.0
                            if d <= smooth_3:
                                v_l = 1.0
                            elif d >= 2 * smooth_3:
                                v_l = 0.0

                            return Z_l * (1 - v_l) + (Z_i + Z_l) * 0.5 * v_l
                    else:
                        return (
                            Z_three_g_c(A_val, B_val)
                            if gaslike
                            else Z_three_l_c(A_val, B_val)
                        )

            else:
                return Z_triple_c(A_val, B_val)

        @numba.guvectorize(
            [
                "void(int8, float64, float64, float64[:], float64, float64, float64, float64[:])"
            ],
            "(),(),(),(n),(),(),()->()",
            target="parallel",
            nopython=True,
            cache=True,
        )
        def _Z_cv(
            gaslike,
            p,
            T,
            X,
            eps,
            smooth_e,
            smooth_3,
            out,
        ):
            out[0] = Z_c(gaslike, p, T, X, eps, smooth_e, smooth_3)

        def Z_cv(
            gaslike: bool,
            p: float | np.ndarray,
            T: float | np.ndarray,
            X: Sequence[float | np.ndarray],
            eps: float = 1e-14,
            smooth_e: float = 1e-2,
            smooth_3: float = 1e-3,
        ):
            # get correct shape independent of format of input
            out = np.zeros_like(p + T + sum(X.T if isinstance(X, np.ndarray) else X))
            _Z_cv(int(gaslike), p, T, X, eps, smooth_e, smooth_3, out)
            return out

        @numba.njit
        def d_Z_c(
            gaslike: int,
            p: float,
            T: float,
            X: Sequence[float],
            eps: float = 1e-14,
            smooth_e: float = 1e-2,
            smooth_3: float = 1e-3,
        ) -> np.ndarray:
            """Wrapper function computing the derivatives of the
            compressibility factor for given
            thermodynamic state.

            This function provides the labeling and smoothing, on top of the wrapped
            computations of ``d_Z``.

            Smoothing can be disabled by setting respective argument to zero.

            ``eps`` is used to determine the root case :func:`get_root_case_c`.

            """

            A_val = A_c(p, T, X)
            B_val = B_c(p, T, X)

            # super critical check
            is_sc = B_val >= critical_line_c(A_val)

            below_widom = B_val <= widom_line_c(A_val)

            nroot = get_root_case_c(A_val, B_val, eps)

            dz = np.zeros(2)

            if nroot == 1:
                d_Z_1_real = d_Z_one_c(A_val, B_val)
                # Extension procedure according Ben Gharbia et al.
                if not is_sc and B_val < B_CRIT:
                    W = d_Z_ext_sub_c(A_val, B_val)
                    if below_widom:
                        dz = W if gaslike else d_Z_1_real
                    else:
                        dz = d_Z_1_real if gaslike else W
                # Extension procedure with asymmetric extension of gas
                elif below_widom and B_val >= B_CRIT:
                    if gaslike:
                        W = d_Z_ext_scg_c(A_val, B_val)

                        # computing distance to border to subcritical extension
                        # smooth if close
                        d = point_to_line_distance_c(
                            np.array([A_val, B_val]), *B_CRIT_LINE_POINTS
                        )
                        if smooth_e > 0.0 and d < smooth_e:
                            d_n = d / smooth_e
                            W = d_Z_ext_sub_c(A_val, B_val) * (1 - d_n) + W * d_n

                        dz = W
                    else:
                        dz = d_Z_1_real
                # Extension procedure with asymmetric extension of liquid
                else:
                    if gaslike:
                        dz = d_Z_1_real
                    else:
                        W = d_Z_ext_scl_c(A_val, B_val)
                        ab_ = np.array([A_val, B_val])

                        # computing distance to Widom-line,
                        # which separates gas and liquid in supercrit area
                        d = point_to_line_distance_c(ab_, *W_LINE_POINTS)
                        if smooth_e > 0.0 and d < smooth_e and B_val >= B_CRIT:
                            d_n = d / smooth_e
                            W = d_Z_ext_scg_c(A_val, B_val) * (1 - d_n) + W * d_n

                        # Computing distance to supercritical line,
                        # which separates sub- and supercritical liquid extension
                        d = point_to_line_distance_c(ab_, *S_CRIT_LINE_POINTS)
                        if smooth_e > 0.0 and d < smooth_e and B_val < B_CRIT:
                            d_n = d / smooth_e
                            W = d_Z_ext_sub_c(A_val, B_val) * (1 - d_n) + W * d_n

                        dz = W
            elif nroot == 2:
                if gaslike:
                    dz = d_Z_double_g_c(A_val, B_val)
                else:
                    dz = d_Z_double_l_c(A_val, B_val)
            elif nroot == 3:
                # triple root area above the critical line is substituted with the
                # extended supercritical liquid-like root
                if is_sc:
                    if gaslike:
                        dz = d_Z_three_g_c(A_val, B_val)
                    else:
                        W = d_Z_ext_scl_c(A_val, B_val)
                        ab_ = np.array([A_val, B_val])

                        # computing distance to Widom-line,
                        # which separates gas and liquid in supercrit area
                        d = point_to_line_distance_c(ab_, *W_LINE_POINTS)
                        if smooth_e > 0.0 and d < smooth_e and B_val >= B_CRIT:
                            d_n = d / smooth_e
                            W = d_Z_ext_scg_c(A_val, B_val) * (1 - d_n) + W * d_n

                        # Computing distance to supercritical line,
                        # which separates sub- and supercritical liquid extension
                        d = point_to_line_distance_c(ab_, *S_CRIT_LINE_POINTS)
                        if smooth_e > 0.0 and d < smooth_e and B_val < B_CRIT:
                            d_n = d / smooth_e
                            W = d_Z_ext_sub_c(A_val, B_val) * (1 - d_n) + W * d_n

                        dz = W
                else:
                    # smoothing according Ben Gharbia et al., in physical 2-phase region
                    if smooth_3 > 0.0:
                        Z_l = Z_three_l_c(A_val, B_val)
                        Z_i = Z_three_i_c(A_val, B_val)
                        Z_g = Z_three_g_c(A_val, B_val)

                        d_Z_l = d_Z_three_l_c(A_val, B_val)
                        d_Z_i = d_Z_three_i_c(A_val, B_val)
                        d_Z_g = d_Z_three_g_c(A_val, B_val)

                        d = (Z_i - Z_l) / (Z_g - Z_l)

                        # gas root smoothing
                        if gaslike:
                            # gas root smoothing weight
                            v_g = (d - (1 - 2 * smooth_3)) / smooth_3
                            v_g = v_g**2 * (3 - 2 * v_g)
                            if d >= 1 - smooth_3:
                                v_g = 1.0
                            elif d <= 1 - 2 * smooth_3:
                                v_g = 0.0

                            dz = d_Z_g * (1 - v_g) + (d_Z_i + d_Z_g) * 0.5 * v_g
                        # liquid root smoothing
                        else:
                            v_l = (d - smooth_3) / smooth_3
                            v_l = -(v_l**2) * (3 - 2 * v_l) + 1.0
                            if d <= smooth_3:
                                v_l = 1.0
                            elif d >= 2 * smooth_3:
                                v_l = 0.0

                            dz = d_Z_l * (1 - v_l) + (d_Z_i + d_Z_l) * 0.5 * v_l
                    else:
                        dz = (
                            d_Z_three_g_c(A_val, B_val)
                            if gaslike
                            else d_Z_three_l_c(A_val, B_val)
                        )
            else:
                dz = d_Z_triple_c(A_val, B_val)

            # substitute derivatives

            dA = d_A_c(p, T, X)
            dB = d_B_c(p, T, X)

            return dz[0] * dA + dz[1] * dB

        @numba.guvectorize(
            [
                "void(int8, float64, float64, float64[:], float64, float64, float64, float64[:], float64[:])"
            ],
            "(),(),(),(n),(),(),(),(m)->(m)",
            target="parallel",
            nopython=True,
            cache=True,
        )
        def _d_Z_cv(
            gaslike,
            p,
            T,
            X,
            eps,
            smooth_e,
            smooth_3,
            out,
            dummy,
        ):
            # dummy is required to get the dimension of of the derivative
            # per row in vectorized computations (m in layout arg to guvectorize)
            # https://stackoverflow.com/questions/66052723/bad-token-in-signature-with-numba-guvectorize
            out[:] = d_Z_c(gaslike, p, T, X, eps, smooth_e, smooth_3)

        def d_Z_cv(
            gaslike: bool,
            p: float | np.ndarray,
            T: float | np.ndarray,
            X: Sequence[float | np.ndarray],
            eps: float = 1e-14,
            smooth_e: float = 1e-2,
            smooth_3: float = 1e-3,
        ):
            # get correct shape independent of format of input
            out_ = np.zeros_like(p + T + sum(X.T if isinstance(X, np.ndarray) else X))
            if out_.shape:
                out = np.empty((out_.shape[0], 2 + ncomp))
            else:
                out = np.empty((1, 2 + ncomp), dtype=np.float64)
            _d_Z_cv(int(gaslike), p, T, X, eps, smooth_e, smooth_3, out)
            # TODO in the case of scalar input (not vectorized)
            # decide if return arg has shape (1,n) or (n,)
            return out

        # endregion

        NJIT_KWARGS_equ = {
            "fastmath": True,
        }
        # region Symbolic equations

        # symbolic expression for mass conservation without the feed fraction
        mass = safe_sum([-y * x for y, x in zip(Y_s, X_per_i_s)])
        d_mass = [mass.diff(_) for _ in Y_s[1:] + X_per_i_s]

        mass_c = numba.njit(sm.lambdify(Y_s[1:] + X_per_i_s, mass), **NJIT_KWARGS_equ)
        d_mass_c = njit_diffs(
            sm.lambdify(Y_s[1:] + X_per_i_s, d_mass), **NJIT_KWARGS_equ
        )

        # symbolic expression for complementary condition y_j * (1 - sum_i x_ij)
        cc = [y_ * (1 - safe_sum(X_in_j_s)) for y_ in Y_s]
        d_cc = [[cc_.diff(_) for _ in Y_s[1:] + X_in_j_s] for cc_ in cc]

        cc = [sm.lambdify(Y_s[1:] + X_in_j_s, cc_) for cc_ in cc]
        d_cc = [sm.lambdify(Y_s[1:] + X_in_j_s, d_cc_) for d_cc_ in d_cc]

        cc_c = [numba.njit(cc_, **NJIT_KWARGS_equ) for cc_ in cc]
        d_cc_c = [njit_diffs(d_cc_, **NJIT_KWARGS_equ) for d_cc_ in d_cc]
        # endregion

        # list containing gas-like flags per phase to decide which to use in the
        # 2- or 3-root case
        gaslike: list[bool] = [phase.gaslike for phase in mixture.phases]

        @numba.njit
        def _parse_xyz(X_gen: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Helper function to parse the fractions from a generic argument."""
            # feed fraction per component, except reference component
            Z = X_gen[: ncomp - 1]
            # phase compositions
            X = X_gen[-ncomp * nphase :]
            # matrix:
            # rows have compositions per phase,
            # columns have compositions related to a component
            X = X.reshape((ncomp, nphase))
            # phase fractions, -1 because fraction of ref phase is eliminated
            Y = X_gen[-(ncomp * nphase + nphase - 1) : -ncomp * nphase]

            return X, Y, Z

        @numba.njit
        def _parse_pT(X_gen: np.ndarray) -> np.ndarray:
            """Helper function extracint pressure and temperature from a generic
            argument."""
            return X_gen[
                -(ncomp * nphase + nphase - 1) - 2 : -(ncomp * nphase + nphase - 1)
            ]

        # region p-T flash
        def F_pT(X_gen: np.ndarray) -> np.ndarray:
            """Callable representing the p-T flash system"""

            X, Y, Z = _parse_xyz(X_gen)
            p, T = _parse_pT(X_gen)

            # maxx conservation excluding first component
            # NOTE this assumes the first set of compositions in component order belongs
            # to the reference component
            mass = np.array([z + mass_c(*Y, *xi) for z, xi in zip(Z, X.T[1:])])

            cc = np.array([cc_c_(*Y, *xj) for cc_c_, xj in zip(cc_c, X)])

            return np.concatenate((mass, cc))

        def DF_pT(X_gen: np.ndarray) -> np.ndarray:
            # degrees of freedom include compositions and independent phase fractions
            dofs = ncomp * nphase + nphase - 1
            # empty dense matrix. NOTE numba can deal only with dense np arrays
            DF = np.zeros((dofs, dofs))

            X, Y, Z = _parse_xyz(X_gen)
            p, T = _parse_pT(X_gen)

            d_mass = np.array([d_mass_c(*Y, *xi) for xi in X.T[1:]])

            d_cc = np.array([d_cc_c_(*Y, *xj) for d_cc_c_, xj in zip(d_cc_c, X)])

            for i in range(ncomp):
                # insert derivatives of mass conservations for component i != ref comp
                if i < ncomp - 1:
                    # derivatives w.r.t. phase fractions
                    DF[i, : nphase - 1] = d_mass[i, : nphase - 1]
                    # derivatives w.r.t compositional fractions of that component
                    DF[i, nphase + i :: nphase] = d_mass[i, nphase - 1 :]
            for j in range(nphase):
                # inserting derivatives of complementary conditions
                # derivatives w.r.t phase fractions
                DF[ncomp * nphase - 1 + j, : nphase - 1] = d_cc[j, : nphase - 1]
                # derivatives w.r.t compositions of that phase
                DF[ncomp * nphase - 1 + j, nphase - 1 + j :: ncomp] = d_cc[
                    j, nphase - 1 :
                ]

            return DF

        # endregion

        # region Pre-compilation and some sanity checks

        tol = 1e-12  # tolerance for test
        p_test = 1.0
        T_test = 1.0
        X_test = tuple((0.0 for _ in range(ncomp)))

        # if compositions are zero, A and B are zero
        assert (
            B_c(p_test, T_test, *X_test) < tol
        ), "Value-test of compiled call to non-dimensional covolume failed."
        assert (
            A_c(p_test, T_test, *X_test) < tol
        ), "Value-test of compiled call to non-dimensional cohesion failed."

        # if A,B are zero, this should give the double-root case
        z_test_g = Z_c(
            True, p_test, T_test, X_test, eps=1e-14, smooth_e=0.0, smooth_3=0.0
        )
        z_test_l = Z_c(
            False, p_test, T_test, X_test, eps=1e-14, smooth_e=0.0, smooth_3=0.0
        )
        assert (
            np.abs(z_test_g - Z_double_g_c(0.0, 0.0)) < tol
        ), "Value-test for compiled, gas-like compressibility factor failed."
        assert (
            np.abs(z_test_l - Z_double_l_c(0.0, 0.0)) < tol
        ), "Value-test for compiled, liquid-like compressibility factor failed."

        d_z_test_g = d_Z_c(
            True, p_test, T_test, X_test, eps=1e-14, smooth_e=0.0, smooth_3=0.0
        )
        d_z_test_l = d_Z_c(
            False, p_test, T_test, X_test, eps=1e-14, smooth_e=0.0, smooth_3=0.0
        )
        da_ = d_A_c(p_test, T_test, *X_test)
        db_ = d_B_c(p_test, T_test, *X_test)
        dzg_ = d_Z_double_g_c(0.0, 0.0)
        dzl_ = d_Z_double_l_c(0.0, 0.0)
        dzg_ = dzg_[0] * da_ + dzg_[1] * db_
        dzl_ = dzl_[0] * da_ + dzl_[1] * db_
        assert (
            np.linalg.norm(d_z_test_g - dzg_) < tol
        ), "Derivative-test for compiled, gas-like compressibility factor failed."
        assert (
            np.linalg.norm(d_z_test_l - dzl_) < tol
        ), "Derivative-test for compiled, liquid-like compressibility factor failed."

        n = 10000
        p = np.random.rand(n) * 1e6 + 1
        T = np.random.rand(n) * 1e2 + 1
        X = np.random.rand(n, 2)
        s = np.sum(X, axis=1)
        X[:, 0] = X[:, 0] / s
        X[:, 1] = X[:, 1] / s
        x0 = np.array([0.0, 0.0])
        # endregion

        # region Storage of compiled functions and systems
        self.cfuncs.update(
            {
                "A": A_c,
                "d_A": d_A_c,
                "B": B_c,
                "d_B": d_B_c,
                "Z": Z_c,
                "d_Z": d_Z_c,
                "Z_cv": Z_cv,
                "d_Z_cv": d_Z_cv,
            }
        )

        self.equations.update(
            {
                "p-T": F_pT,
            }
        )

        self.jacobians.update(
            {
                "p-T": DF_pT,
            }
        )
        # endregion

        t = np.array([0.01, 1e6, 300, 0.9, 0.1, 0.2, 0.3, 0.4])

        print(F_pT(t))
        print(DF_pT(t))
