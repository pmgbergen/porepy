"""Experimentel code for efficient unified flash calculations using numba and sympy."""
from __future__ import annotations

import os

# os.environ['NUMBA_DISABLE_INTEL_SVML']  = '1'
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "30"

from typing import Any, Callable, Literal, Sequence

import numba
import numpy as np
import sympy as sm

from .._core import COMPOSITIONAL_VARIABLE_SYMBOLS as SYMBOLS
from ..composite_utils import safe_sum
from ..mixture import BasicMixture, NonReactiveMixture
from .eos import A_CRIT, B_CRIT, PengRobinsonEoS
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


def coef0(A: Any, B: Any) -> Any:
    """Coefficient for the zeroth monomial of the characteristic equation

    :math:`Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0`.

    For any input type supporting Python's overload of ``+,-,*,**``.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The result of :math:`B^3 + B^2 - AB`.

    """
    return B**3 + B**2 - A * B


coef0_c = numba.njit("float64(float64, float64)", **_COEFF_COMPILTER_ARGS)(coef0)
"""NJIT-ed version of :func:`coef0`.

Signature: ``(float64, float64) -> float64``

"""


def coef1(A: Any, B: Any) -> Any:
    """Coefficient for the first monomial of the characteristic equation

    :math:`Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0`.

    For any input type supporting Python's overload of ``+,-,*,**``.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The result of :math:`A - 2 B - 3 B^2`.

    """
    return A - 2.0 * B - 3.0 * B**2


coef1_c = numba.njit("float64(float64, float64)", **_COEFF_COMPILTER_ARGS)(coef1)
"""NJIT-ed version of :func:`coef1`.

Signature: ``(float64, float64) -> float64``

"""


def coef2(A: Any, B: Any) -> Any:
    """Coefficient for the second monomial of the characteristic equation

    :math:`Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0`.

    For any input type supporting Python's overload of ``-``.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The result of :math:`B - 1`.

    """
    return B - 1


coef2_c = numba.njit("float64(float64, float64)", **_COEFF_COMPILTER_ARGS)(coef2)
"""NJIT-ed version of :func:`coef2`.

Signature: ``(float64, float64) -> float64``

"""


def red_coef0(A: Any, B: Any) -> Any:
    """Zeroth coefficient of the reduced characteristic equation

    :math:`Z^3 + c_{r1} Z + c_{r0} = 0`.

    Uses :func:`coef0` - :func:`coef2` to compute the expressions in terms of
    ``A`` and ``B``

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The result of

        .. math::

            c_2^3(A, B)\\frac{2}{27} - c_2(A, B) c_1(A, B)\\frac{1}{3} + c_0(A, B)

    """
    c2 = coef2(A, B)
    return c2**3 * (2.0 / 27.0) - c2 * coef1(A, B) * (1.0 / 3.0) + coef0(A, B)


@numba.njit("float64(float64, float64)", **_COEFF_COMPILTER_ARGS)
def red_coef0_c(A: float, B: float) -> float:
    """NJIT-ed version of :func:`red_coef0`.

    Signature: ``(float64, float64) -> float64``

    """
    c2 = coef2_c(A, B)
    return c2**3 * (2.0 / 27.0) - c2 * coef1_c(A, B) * (1.0 / 3.0) + coef0_c(A, B)


def red_coef1(A: Any, B: Any) -> Any:
    """First coefficient of the reduced characteristic equation

    :math:`Z^3 + c_{r1} Z + c_{r0} = 0`.

    Uses :func:`coef0` - :func:`coef2` to compute the expressions in terms of
    ``A`` and ``B``

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The result of

        .. math::

            c_1(A, B) - c_2^2(A, B)\\frac{1}{3}

    """
    return coef1(A, B) - coef2(A, B) ** 2 * (1.0 / 3.0)


@numba.njit("float64(float64, float64)", **_COEFF_COMPILTER_ARGS)
def red_coef1_c(A: float, B: float) -> float:
    """NJIT-ed version of :func:`red_coef1`.

    Signature: ``(float64, float64) -> float64``

    """
    return coef1_c(A, B) - coef2_c(A, B) ** 2 * (1.0 / 3.0)


def discr(rc0: Any, rc1: Any) -> Any:
    """Discriminant of the characeteristic polynomial based on the reduced coefficient.

    Parameters:
        rc0: Zeroth reduced coefficient (see :func:`red_coef0`)
        rc1: First reduced coefficient (see :func:`red_coef1`)

    Returns:
        The result of

        .. math::

            c_{r0}^2\\frac{1}{4} - c_{r1}^3\\frac{1}{27}

    """
    return rc0**2 * (1.0 / 4.0) + rc1**3 * (1.0 / 27.0)


discr_c = numba.njit("float64(float64, float64)", **_COEFF_COMPILTER_ARGS)(discr)
"""NJIT-ed version of :func:`discr`.

Signature: ``(float64, float64) -> float64``

"""


@numba.njit("int8(float64, float64, float64)", **_COEFF_COMPILTER_ARGS)
def get_root_case_c(A, B, eps=1e-14):
    """ "An piece-wise cosntant function dependent on
    non-dimensional cohesion and covolume, representing the number of roots
    of the characteristic polynomial in terms of cohesion and covolume.

    NJIT-ed function with signature ``(float64, float64, float64) -> int8``.

    :data:`red_coef0_c`, :data:`red_coef1_c` and :data:`discr_c` are used to compute
    and determine the root case.

    For more information,
    `see here <https://de.wikipedia.org/wiki/Cardanische_Formeln>`_ .


    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.
        eps: ``default=1e-14``

            Numerical zero to detect degenerate polynomials (zero discriminant).

    Returns:
        An integer indicating the root case

        - 0 : triple root
        - 1 : 1 real root, 2 complex-conjugated roots
        - 2 : 2 real roots, one with multiplicity 2
        - 3 : 3 distinct real roots

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
"""Numpy-universial version of :func:`get_root_case_c`.

Important:
    ``eps`` is not optional eny more. Can be made so with a simple wrapper.

"""


@numba.njit("float64(float64,float64,float64)", **_COEFF_COMPILTER_ARGS)
def characteristic_residual_c(Z, A, B):
    """NJIT-ed function with signature ``(float64,float64,float64) -> float64``.

    Parameters:
        Z: A supposed root.
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The residual of the characteristic polynomial
        :math:`Z^3 + c_2(A, B) Z^2 + c_1(A, B) Z + c_0(A, B)`.

        IF ``Z`` is an actual root, the residual is 0.

    """
    c2 = coef2_c(A, B)
    c1 = coef1_c(A, B)
    c0 = coef0_c(A, B)

    return Z**3 + c2 * Z**2 + c1 * Z + c0


check_if_root_cv = numba.vectorize(
    [numba.float64(numba.float64, numba.float64, numba.float64)],
    nopython=True,
    **_COEFF_COMPILTER_ARGS,
)(characteristic_residual_c)
"""Numpy-universial version of :func:`characteristic_residual_c`."""


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


@numba.njit(
    [
        numba.float64(
            numba.types.Array(numba.float64, 1, "C", readonly=False),
            numba.types.Array(numba.float64, 1, "C", readonly=True),
            numba.types.Array(numba.float64, 1, "C", readonly=True),
        ),
        numba.float64(
            numba.types.Array(numba.float64, 1, "C", readonly=False),
            numba.types.Array(numba.float64, 1, "C", readonly=False),
            numba.types.Array(numba.float64, 1, "C", readonly=False),
        ),
    ],
    cache=True,
)
def point_to_line_distance_c(p: np.ndarray, lp1: np.ndarray, lp2: np.ndarray) -> float:
    """Computes the distance between a 2-D point and a line spanned by two points.

    NJIT-ed function with signature ``(float64[:], float64[:], float64[:]) -> float64``.

    Parameters:
        p: ``shape=(2,)``

            Point in 2D space.
        lp1: ``shape=(2,)``

            First point spanning the line.
        lp2: ``shape=(2,)``

            Second point spanning the line.

    Returns:
        Normal distance between ``p`` and the spanned line.

    """

    d = np.sqrt((lp2[0] - lp1[0]) ** 2 + (lp2[1] - lp1[1]) ** 2)
    n = np.abs(
        (lp2[0] - lp1[0]) * (lp1[1] - p[1]) - (lp1[0] - p[0]) * (lp2[1] - lp1[1])
    )
    return n / d


@numba.njit("float64(float64)", fastmath=True, cache=True)
def critical_line_c(A: float) -> float:
    """Returns the critical line parametrized as ``B(A)``.

    .. math::

        \\frac{B_{crit}}{A_{crit}} A

    NJIT-ed function with signature ``float64(float64)``.

    """
    return (B_CRIT / A_CRIT) * A


critical_line_cv = numba.vectorize(
    [numba.float64(numba.float64)],
    nopython=True,
    fastmath=True,
    cache=True,
)(critical_line_c)
"""Numpy-universial version of :func:`critical_line_cv`."""


@numba.njit("float64(float64)", fastmath=True, cache=True)
def widom_line_c(A: float) -> float:
    """Returns the Widom-line parametrized as ``B(A)`` in the A-B space:

    .. math::

        B_{crit} + 0.8 \\cdot 0.3381965009398633 \\cdot \\left(A - A_{crit}\\right)

    NJIT-ed function with signature ``float64(float64)``.

    """
    return B_CRIT + 0.8 * 0.3381965009398633 * (A - A_CRIT)


widom_line_cv = numba.vectorize(
    [numba.float64(numba.float64)],
    nopython=True,
    fastmath=True,
    cache=True,
)(widom_line_c)
"""Numpy-universial version of :func:`widom_line_c`."""


B_CRIT_LINE_POINTS: tuple[np.ndarray, np.ndarray] = (
    np.array([0.0, B_CRIT]),
    np.array([A_CRIT, B_CRIT]),
)
"""Two 2D points characterizing the line ``B=B_CRIT`` in the A-B space, namely

.. math::

    (0, B_{crit}),~(A_{crit},B_{crit})

See :data:`~porepy.composite.peng_robinson.eos.B_CRIT`.
See :data:`~porepy.composite.peng_robinson.eos.A_CRIT`.

"""


S_CRIT_LINE_POINTS: tuple[np.ndarray, np.ndarray] = (
    np.zeros(2),
    np.array([A_CRIT, B_CRIT]),
)
"""Two 2D points characterizing the super-critical line in the A-B space, namely

.. math::

    (0,0),~(A_{crit},B_{crit})

See :data:`~porepy.composite.peng_robinson.eos.B_CRIT`.
See :data:`~porepy.composite.peng_robinson.eos.A_CRIT`.

"""


W_LINE_POINTS: tuple[np.ndarray, np.ndarray] = (
    np.array([0.0, widom_line_c(0)]),
    np.array([A_CRIT, widom_line_c(A_CRIT)]),
)
"""Two 2D points characterizing the Widom-line for water.

The points are created by using :func:`widom_line_c` for :math:`A\\in\\{0, A_{crit}\\}`.

See :data:`~porepy.composite.peng_robinson.eos.A_CRIT`.

"""


@numba.njit("float64[:,:](float64[:,:])", fastmath=True, cache=True)
def normalize_fractions(X: np.ndarray) -> np.ndarray:
    """Takes a matrix of phase compositions (rows - phase, columns - component)
    and normalizes the fractions.

    Meaning it divides each matrix element by the sum of respective row.

    NJIT-ed function with signature ``(float64[:,:]) -> float64[:,:]``.

    Parameters:
        X: ``shape=(num_phases, num_components)``

            A matrix of phase compositions, containing per row the (extended)
            fractions per component.

    Returns:
        A normalized version of ``X``, with the normalization performed row-wise
        (phase-wise).

    """
    return (X.T / X.sum(axis=1)).T


# TODO check if this is only required for derivatives lambdified with 'math'
# if lambdified with numpy, maybe it returns already an array.
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

        self._sysfuncs: dict[str, Callable | list[Callable]] = dict()
        """A collection of relevant functions, which must not be dereferenced."""

        self._gaslike: list[int] = []
        """List containing gaslike flags per phase as integers."""

        self._Z_cfuncs: dict[str, Callable] = dict()
        """A map containing compiled functions for the compressibility factor,
        dependeng on A and B.

        The functions represent computations for each root-case of the characteristic
        polynomial, as well as the gradient w.r.t. A and B."""

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
        properties. (nopython, just-in-time compilation)

        Keys are names of the properties. Standard symbols from literature are used.

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

        self.ufuncs: dict[str, Callable] = dict()
        """Generalized numpy-ufunc represenatation of some thermodynamic properties.

        See :attr:`cfuncs` for more information.

        Callable contained here describe generalized numpy functions, meaning they
        operatore on scalar and vectorized input (nummpy arrays).
        They exploit numba.guvectorize (among others) for efficient computation.

        """

        self._define_unknowns_and_arguments(mixture)
        self._compile_expressions_equations_jacobians(mixture)
        # self.compile_gufuncs()

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

    def _compile_expressions_equations_jacobians(self, mixture: NonReactiveMixture):
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

        def _diff_thd(expr_, thd_arg_):
            """Helper function to define the gradient of a multivariate function,
            where the argument has the special structure of ``thd_arg_j``."""
            p_, T_, X_j_ = thd_arg_
            return [expr_.diff(p_), expr_.diff(T_)] + [expr_.diff(_) for _ in X_j_]

        def _njit_diff_2(f, **njit_kwargs):
            """Helper function for special wrapping of some compiled derivatives.

            Enforces a special signature."""

            f = numba.njit(f, **njit_kwargs)

            @numba.njit("float64[:](float64[:], float64[:])", **njit_kwargs)
            def inner(x, y):
                return np.array(f(x, y))

            return inner

        def _njit_phi_diffs(f, **njit_kwargs):
            """Helper function to compile the derivative functions for fugacity
            coefficiets. Enforces a special signature.

            """
            f = numba.njit(f, **njit_kwargs)

            @numba.njit(
                "float64[:](float64, float64, float64[:], float64, float64, float64)",
                **njit_kwargs,
            )
            def inner(p_, T_, X_, A_, B_, Z_):
                return np.array(f(p_, T_, X_, A_, B_, Z_))

            return inner

        def _njit_phi(f, **njit_kwargs):
            """Helper function to compile the function for fugacities in phase.

            It needs an intermediate collapse, since sympy.Matrix lambdified returns
            a (n, 1) array not (n,)

            Enforces a special signature.

            """
            f = numba.njit(f, **njit_kwargs)

            @numba.njit(
                "float64[:](float64, float64, float64[:], float64, float64, float64)",
                **njit_kwargs,
            )
            def inner(p_, T_, X_, A_, B_, Z_):
                phi_ = f(p_, T_, X_, A_, B_, Z_)
                return phi_[:, 0]

            return inner

        # region Equation for feed fraction in terms of phase fractions and compositions

        # symbolic expression for mass conservation without the feed fraction
        feed_from_xy = safe_sum([y * x for y, x in zip(Y_s, X_per_i_s)])
        d_feed_from_xy = [feed_from_xy.diff(_) for _ in Y_s[1:] + X_per_i_s]

        # mass equation takes two vectors:
        # vector of independent y and vector of x per component i
        mass_arg = [Y_s[1:], X_per_i_s]

        feed_from_xy_c = numba.njit("float64(float64[:], float64[:])", fastmath=True)(
            sm.lambdify(mass_arg, feed_from_xy)
        )
        d_feed_from_xy_c = _njit_diff_2(
            sm.lambdify(mass_arg, d_feed_from_xy),
            fastmath=True,
        )
        # endregion

        # region Equation for complementary condition

        # NOTE: we have take a special approach here, because we define a list of
        # symbolic equations and compile them. Numba cannot parallelize an iteration
        # over custom compiled function
        # ("first-class function types are an experimental feature")

        # TODO this function is general, along many others
        # pull it out of here and make ncomp and nphase an argument.
        # pre-compile it with fixed signature and fitting numba flags (including cache)

        @numba.njit("float64(float64, float64[:])", fastmath=True, cache=True)
        def cc_c(y: float, x: np.ndarray) -> float:
            """Takes an independent phase fraction, and composition of that phase,
            and computes complementary condition y_j * (1 - sum x_j)"""
            return y * (1.0 - np.sum(x))

        @numba.njit(
            "float64[:](float64[:], float64[:], int32)", fastmath=True, cache=True
        )
        def d_cc_c(y: np.ndarray, x: np.ndarray, j: int) -> np.ndarray:
            """Constructs the derivative of the complementary conditions,
            including the derivative w.r.t. to all phase fractions.

            Hence the phase index j must be given.

            y is assumed to be the vector of independent phase fractions.
            x must be the composition of phase j

            """
            d = np.zeros(nphase - 1 + ncomp)
            unity = 1 - np.sum(x)

            if j == 0:
                d[: nphase - 1] = (-1) * unity
                d[nphase - 1 :] = (-1) * (1 - np.sum(y))
            else:
                # y has independent phase fraction y_0 is eliminted, hence j - 1
                d[j - 1] = unity
                d[nphase - 1 :] = (-1) * y[j - 1]  # y has independent phase fractions

            return d

        # endregion

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
        d_B_e = _diff_thd(B_e, thd_arg_j)
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

        d_A_e = _diff_thd(A_e, thd_arg_j)
        d_A_c = njit_diffs(sm.lambdify(thd_arg_j, d_A_e))

        # endregion

        # region Fugacity coefficients
        # expressions of fugacity coeffs per component,
        # where Z, A, B are independent symbols
        phi_i_e: list[sm.Expr] = list()
        # derivatives w.r.t. p, T, X in phase j, A, B, Z
        d_phi_i_e: list[sm.Expr] = list()
        # dependencies as a list of arguments, X_in_j will be vectorized
        phi_arg = [p_s, T_s, X_in_j_s, A_s, B_s, Z_s]

        # compiled callables
        phi_i_c = list()
        d_phi_i_c = list()

        for i in range(ncomp):
            B_i_e = PengRobinsonEoS.B(b_i_crit[i], p_s, T_s)
            dXi_A_e = A_e.diff(X_in_j_s[i])
            log_phi_i = (
                B_i_e / B_s * (Z_s - 1)
                - sm.ln(Z_s - B_s)
                - A_s
                / (B_s * np.sqrt(8))
                * sm.ln((Z_s + (1 + np.sqrt(2)) * B_s) / (Z_s + (1 - np.sqrt(2)) * B_s))
                * (dXi_A_e / A_s - B_i_e / B_s)
            )

            phi_i_ = sm.exp(log_phi_i)

            d_phi_i_ = (
                [phi_i_.diff(p_s), phi_i_.diff(T_s)]
                + [phi_i_.diff(_) for _ in X_in_j_s]
                + [phi_i_.diff(A_s), phi_i_.diff(B_s), phi_i_.diff(Z_s)]
            )

            # TODO this is numerically disadvantages
            # no truncation and usage of exp
            phi_i_e.append(phi_i_)
            d_phi_i_e.append(d_phi_i_)

            # TODO, replace low and exp by trunclog and truncexp here if necessary
            phi_i_c.append(
                numba.njit(
                    "float64(float64, float64, float64[:], float64, float64, float64)"
                )(sm.lambdify(phi_arg, phi_i_))
            )
            d_phi_i_c.append(_njit_phi_diffs(sm.lambdify(phi_arg, d_phi_i_)))

        phi_i_c: tuple[Callable, ...] = tuple(phi_i_c)
        d_phi_i_c: tuple[Callable, ...] = tuple(d_phi_i_c)

        # matrix approach
        phi_e = sm.Matrix(phi_i_e)
        d_phi_e = phi_e.jacobian([p_s, T_s] + X_in_j_s + [A_s, B_s, Z_s])

        # needs special wrapping to collaps (n,1) to (n,)
        phi_c = _njit_phi(sm.lambdify(phi_arg, phi_e))
        d_phi_c = numba.njit(
            "float64[:,:](float64, float64, float64[:], float64, float64, float64)"
        )(sm.lambdify(phi_arg, d_phi_e))

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

        self._Z_cfuncs.update(
            {
                "one-root": Z_one_c,
                "double-root-gas": Z_double_g_c,
                "double-root-liq": Z_double_l_c,
                "three-root-gas": Z_three_g_c,
                "three-root-inter": Z_three_i_c,
                "three-root-liq": Z_three_l_c,
                "ext-root-sub": Z_ext_sub_c,
                "ext-root-super-gas": Z_ext_scg_c,
                "ext-root-super-liq": Z_ext_scl_c,
                "d-one-root": d_Z_one_c,
                "d-double-root-gas": d_Z_double_g_c,
                "d-double-root-liq": d_Z_double_l_c,
                "d-three-root-gas": d_Z_three_g_c,
                "d-three-root-inter": d_Z_three_i_c,
                "d-three-root-liq": d_Z_three_l_c,
                "d-ext-root-sub": d_Z_ext_sub_c,
                "d-ext-root-super-gas": d_Z_ext_scg_c,
                "d-ext-root-super-liq": d_Z_ext_scl_c,
            }
        )

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
            AB_point = np.array([A_val, B_val])

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
                            AB_point,
                            B_CRIT_LINE_POINTS[0],
                            B_CRIT_LINE_POINTS[1],
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

                        # computing distance to Widom-line,
                        # which separates gas and liquid in supercrit area
                        d = point_to_line_distance_c(
                            AB_point,
                            W_LINE_POINTS[0],
                            W_LINE_POINTS[1],
                        )
                        if smooth_e > 0.0 and d < smooth_e and B_val >= B_CRIT:
                            d_n = d / smooth_e
                            W = Z_ext_scg_c(A_val, B_val) * (1 - d_n) + W * d_n

                        # Computing distance to supercritical line,
                        # which separates sub- and supercritical liquid extension
                        d = point_to_line_distance_c(
                            AB_point,
                            S_CRIT_LINE_POINTS[0],
                            S_CRIT_LINE_POINTS[1],
                        )
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

                        # computing distance to Widom-line,
                        # which separates gas and liquid in supercrit area
                        d = point_to_line_distance_c(
                            AB_point,
                            W_LINE_POINTS[0],
                            W_LINE_POINTS[1],
                        )
                        if smooth_e > 0.0 and d < smooth_e and B_val >= B_CRIT:
                            d_n = d / smooth_e
                            W = Z_ext_scg_c(A_val, B_val) * (1 - d_n) + W * d_n

                        # Computing distance to supercritical line,
                        # which separates sub- and supercritical liquid extension
                        d = point_to_line_distance_c(
                            AB_point,
                            S_CRIT_LINE_POINTS[0],
                            S_CRIT_LINE_POINTS[1],
                        )
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

            AB_point = np.array([A_val, B_val])

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
                            AB_point,
                            B_CRIT_LINE_POINTS[0],
                            B_CRIT_LINE_POINTS[1],
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

                        # computing distance to Widom-line,
                        # which separates gas and liquid in supercrit area
                        d = point_to_line_distance_c(
                            AB_point,
                            W_LINE_POINTS[0],
                            W_LINE_POINTS[1],
                        )
                        if smooth_e > 0.0 and d < smooth_e and B_val >= B_CRIT:
                            d_n = d / smooth_e
                            W = d_Z_ext_scg_c(A_val, B_val) * (1 - d_n) + W * d_n

                        # Computing distance to supercritical line,
                        # which separates sub- and supercritical liquid extension
                        d = point_to_line_distance_c(
                            AB_point,
                            S_CRIT_LINE_POINTS[0],
                            S_CRIT_LINE_POINTS[1],
                        )
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

                        # computing distance to Widom-line,
                        # which separates gas and liquid in supercrit area
                        d = point_to_line_distance_c(
                            AB_point,
                            W_LINE_POINTS[0],
                            W_LINE_POINTS[1],
                        )
                        if smooth_e > 0.0 and d < smooth_e and B_val >= B_CRIT:
                            d_n = d / smooth_e
                            W = d_Z_ext_scg_c(A_val, B_val) * (1 - d_n) + W * d_n

                        # Computing distance to supercritical line,
                        # which separates sub- and supercritical liquid extension
                        d = point_to_line_distance_c(
                            AB_point,
                            S_CRIT_LINE_POINTS[0],
                            S_CRIT_LINE_POINTS[1],
                        )
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

        # endregion

        # list containing gas-like flags per phase to decide which to use in the
        # 2- or 3-root case
        gaslike: tuple[int, ...] = tuple(
            [int(phase.gaslike) for phase in mixture.phases]
        )

        @numba.njit
        def _parse_xyz(X_gen: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Helper function to parse the fractions from a generic argument."""
            # feed fraction per component, except reference component
            Z = X_gen[: ncomp - 1]
            # phase compositions
            X = X_gen[-ncomp * nphase :].copy()
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

        @numba.njit("float64[:](float64[:], float64[:,:])", fastmath=True)
        def complementary_cond_c(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
            """Helper function to evaluate the complementary conditions"""

            ccs = np.empty(nphase, dtype=np.float64)
            ccs[0] = cc_c(1 - np.sum(Y), X[0])
            for j in range(1, nphase):
                ccs[j] = cc_c(Y[j - 1], X[j])

            return ccs

        @numba.njit("float64[:,:](float64[:], float64[:,:])", fastmath=True)
        def d_complementary_cond_c(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
            """Hellper function to construct the derivative of the complementary
            conditions per phase.

            Note however that the derivatives w.r.t. to X_ij are displaced.
            For convenience reasons, the derivatives w.r.t to x_ij is for all phases
            in one column.
            Must be inserted properly in the flash system.
            """

            d_ccs = np.empty((nphase, nphase - 1 + ncomp), dtype=np.float64)

            d_ccs[0] = d_cc_c(Y, X[0], 0)
            for j in range(1, nphase):
                d_ccs[j] = d_cc_c(Y, X[j], j)

            return d_ccs

        @numba.njit
        def isofug_constr_c(
            p: float,
            T: float,
            Xn: np.ndarray,
            A_j: np.ndarray,
            B_j: np.ndarray,
            Z_j: np.ndarray,
        ):
            """Helper function to assemble the isofugacity constraint.

            Formulation is always w.r.t. the reference phase r, assumed to be r=0.

            """
            isofug = np.empty(ncomp * (nphase - 1), dtype=np.float64)

            phi_r = phi_c(p, T, Xn[0], A_j[0], B_j[0], Z_j[0])

            for j in range(1, nphase):
                phi_j = phi_c(p, T, Xn[j], A_j[j], B_j[j], Z_j[j])

                # isofugacity constraint between phase j and phase r
                isofug[(j - 1) * ncomp : j * ncomp] = (
                    Xn[j] * phi_c(p, T, Xn[j], A_j[j], B_j[j], Z_j[j]) - Xn[0] * phi_r
                )

            return isofug

        @numba.njit(
            "float64[:,:](float64, float64, float64[:], float64, float64, float64, int32)"
        )
        def d_isofug_block_j(
            p: float,
            T: float,
            Xn: np.ndarray,
            A_j: float,
            B_j: float,
            Z_j: float,
            gaslike_j: int,
        ):
            """Helper function to construct a block representing the derivative
            of x_ij * phi_ij for all i as a matrix, with i row index.
            This is constructed for a given phase j.
            """
            # derivatives w.r.t. p, T, all compositions, A, B, Z
            dx_phi_j = np.zeros((ncomp, 2 + ncomp + 3))

            phi_j = phi_c(p, T, Xn, A_j, B_j, Z_j)
            d_phi_j = d_phi_c(p, T, Xn, A_j, B_j, Z_j)

            # product rule d(x * phi) = dx * phi + x * dphi
            # dx is is identity
            dx_phi_j[:, 2 : 2 + ncomp] = np.diag(phi_j)
            d_xphi_j = dx_phi_j + (d_phi_j.T * Xn).T

            # expanding derivatives w.r.t. to A, B, Z
            # get derivatives of A, B, Z, w.r.t. p, T and X
            dAj = d_A_c(p, T, Xn)
            dBj = d_B_c(p, T, Xn)
            dZj = d_Z_c(gaslike_j, p, T, Xn)

            d_xphi_j = (
                d_xphi_j[:, :-3]
                + np.outer(d_xphi_j[:, -3], dAj)
                + np.outer(d_xphi_j[:, -2], dBj)
                + np.outer(d_xphi_j[:, -1], dZj)
            )

            return d_xphi_j

        @numba.njit(
            "float64[:,:](float64, float64, float64[:,:], float64[:], float64[:], float64[:])"
        )
        def d_isofug_constr_c(
            p: float,
            T: float,
            Xn: np.ndarray,
            A_j: np.ndarray,
            B_j: np.ndarray,
            Z_j: np.ndarray,
        ):
            """Helper function to assemble the derivative of the isofugacity constraints

            Formulation is always w.r.t. the reference phase r, assumed to be zero 0.

            Important:
                The derivative is taken w.r.t. to A, B, Z (among others).
                An forward expansion must be done after a call to this function.

            """
            d_iso = np.zeros((ncomp * (nphase - 1), 2 + ncomp * nphase))

            # creating derivative parts involving the reference phase
            d_xphi_r = d_isofug_block_j(p, T, Xn[0], A_j[0], B_j[0], Z_j[0], gaslike[0])

            for j in range(1, nphase):
                # construct the same as above for other phases
                d_xphi_j = d_isofug_block_j(
                    p, T, Xn[j], A_j[j], B_j[j], Z_j[j], gaslike[j]
                )

                # filling in the relevant blocks
                # remember: d(x_ij * phi_ij - x_ir * phi_ir)
                # hence every row-block contains (-1)* d_xphi_r
                # p, T derivative
                d_iso[(j - 1) * ncomp : j * ncomp, :2] = (
                    d_xphi_j[:, :2] - d_xphi_r[:, :2]
                )
                # derivative w.r.t. fractions in reference phase
                d_iso[(j - 1) * ncomp : j * ncomp, 2 : 2 + ncomp] = (-1) * d_xphi_r[
                    :, 2:
                ]
                # derivatives w.r.t. fractions in independent phase j
                d_iso[
                    (j - 1) * ncomp : j * ncomp, 2 + j * ncomp : 2 + (j + 1) * ncomp
                ] = d_xphi_j[:, 2:]

            return d_iso

        # region p-T flash
        @numba.njit("float64[:](float64[:])")
        def F_pT(X_gen: np.ndarray) -> np.ndarray:
            """Callable representing the p-T flash system"""

            X, Y, Z = _parse_xyz(X_gen)
            p, T = _parse_pT(X_gen)

            # mass conservation excluding first component
            mass = np.zeros(ncomp - 1)
            for i in range(ncomp - 1):
                # NOTE assuming first column of compositions belongs to ref component
                mass[i] = Z[i] - feed_from_xy_c(Y, X[:, i + 1])

            # complementary conditions
            cc = complementary_cond_c(Y, X)

            # EoS specific calculations
            Xn = normalize_fractions(X)
            Z_j = np.array([Z_c(gaslike[j], p, T, Xn[j]) for j in range(nphase)])
            A_j = np.array([A_c(p, T, Xn[j]) for j in range(nphase)])
            B_j = np.array([B_c(p, T, Xn[j]) for j in range(nphase)])

            isofug = isofug_constr_c(p, T, Xn, A_j, B_j, Z_j)

            F_val = np.zeros(ncomp - 1 + ncomp * (nphase - 1) + nphase)
            F_val[: ncomp - 1] = mass
            F_val[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1)] = isofug
            F_val[-nphase:] = cc  # Complementary conditions always last

            return F_val

        @numba.njit("float64[:,:](float64[:])")
        def DF_pT(X_gen: np.ndarray) -> np.ndarray:
            # degrees of freedom include compositions and independent phase fractions
            dofs = ncomp * nphase + nphase - 1
            # empty dense matrix. NOTE numba can deal only with dense np arrays
            DF = np.zeros((dofs, dofs))

            X, Y, _ = _parse_xyz(X_gen)
            p, T = _parse_pT(X_gen)

            # * (-1) because of z - z(x,y) = 0
            d_mass = np.zeros((ncomp - 1, nphase - 1 + nphase))
            for i in range(1, ncomp):
                d_mass[i - 1, :] = (-1) * d_feed_from_xy_c(Y, X[:, i])

            d_cc = d_complementary_cond_c(Y, X)

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
                DF[
                    ncomp * nphase - 1 + j,
                    nphase - 1 + j * ncomp : nphase - 1 + (j + 1) * ncomp,
                ] = d_cc[j, nphase - 1 :]

            # EoS specific calculations
            Xn = normalize_fractions(X)
            Z_j = np.array([Z_c(gaslike[j], p, T, Xn[j]) for j in range(nphase)])
            A_j = np.array([A_c(p, T, Xn[j]) for j in range(nphase)])
            B_j = np.array([B_c(p, T, Xn[j]) for j in range(nphase)])

            d_iso = d_isofug_constr_c(p, T, Xn, A_j, B_j, Z_j)

            # isofugacity constraints are inserted directly after mass conservation
            # no derivatives w.r.t. p, T or Y
            DF[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), nphase - 1 :] = d_iso[
                :, 2:
            ]

            return DF

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

        self._sysfuncs.update(
            {
                "complementary-conditions": cc_c,
                "d-complementary-conditions": d_cc_c,
                "feed-from-x-y": feed_from_xy_c,
                "d-feed-from-x-y": d_feed_from_xy_c,
                "fugacity-coefficients": phi_i_c,
                "d-fugacity-coefficients": d_phi_i_c,
                "parser-xyz": _parse_xyz,
                "parser-pT": _parse_pT,
            }
        )

        self._gaslike = gaslike
        # endregion

    def test_compiled_functions(self, tol: float = 1e-12, n: int = 100000):
        """Performs some tests on assembled functions.

        Warning:
            This triggers numba's just-in-time compilation!

            I.e., the execution of this function takes a considerable amount of time.

        Warning:
            This method raises AssertionErrors if any test failes.

        Parameters:
            tol: ``default=1e-12``

                Tolerance for numerical zero.
            n: ``default=100000``

                Number for testing of vectorized computations.

        """

        ncomp = self._n_c
        nphase = self._n_p

        p_1 = 1.0
        T_1 = 1.0
        X0 = np.array([0.0] * ncomp)

        A_c = self.cfuncs["A"]
        d_A_c = self.cfuncs["d_A"]
        B_c = self.cfuncs["B"]
        d_B_c = self.cfuncs["d_B"]
        Z_c = self.cfuncs["Z"]
        d_Z_c = self.cfuncs["d_Z"]

        Z_double_g_c = self._Z_cfuncs["double-root-gas"]
        d_Z_double_g_c = self._Z_cfuncs["d-double-root-gas"]
        Z_double_l_c = self._Z_cfuncs["double-root-liq"]
        d_Z_double_l_c = self._Z_cfuncs["d-double-root-liq"]

        # if compositions are zero, A and B are zero
        assert (
            B_c(p_1, T_1, X0) < tol
        ), "Value-test of compiled call to non-dimensional covolume failed."
        assert (
            A_c(p_1, T_1, X0) < tol
        ), "Value-test of compiled call to non-dimensional cohesion failed."

        # if A,B are zero, this should give the double-root case
        z_test_g = Z_c(True, p_1, T_1, X0, eps=1e-14, smooth_e=0.0, smooth_3=0.0)
        z_test_l = Z_c(False, p_1, T_1, X0, eps=1e-14, smooth_e=0.0, smooth_3=0.0)
        assert (
            np.abs(z_test_g - Z_double_g_c(0.0, 0.0)) < tol
        ), "Value-test for compiled, gas-like compressibility factor failed."
        assert (
            np.abs(z_test_l - Z_double_l_c(0.0, 0.0)) < tol
        ), "Value-test for compiled, liquid-like compressibility factor failed."

        d_z_test_g = d_Z_c(True, p_1, T_1, X0, eps=1e-14, smooth_e=0.0, smooth_3=0.0)
        d_z_test_l = d_Z_c(False, p_1, T_1, X0, eps=1e-14, smooth_e=0.0, smooth_3=0.0)
        da_ = d_A_c(p_1, T_1, X0)
        db_ = d_B_c(p_1, T_1, X0)
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

        p = np.random.rand(n) * 1e6 + 1
        T = np.random.rand(n) * 1e2 + 1
        X = np.random.rand(n, 2)
        X = normalize_fractions(X)

    def compile_gufuncs(self):
        """Creates general numpy-ufuncs for some properties, which are required for
        fast fast computation of e.g. fluid properties.

        Compiled functions are stored in :attr:`gufuncs`.

        Important:
            Be aware that the compilation can take a considarable amount of time.

            The compilation has not been tested using a multithreaded or multiprocessing
            approach.

        """

        ncomp = self._n_c
        nphase = self._n_p

        Z_c = self.cfuncs["Z"]
        d_Z_c = self.cfuncs["d_Z"]

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

        self.ufuncs.update(
            {
                "Z_cv": Z_cv,
                "d_Z_cv": d_Z_cv,
            }
        )
