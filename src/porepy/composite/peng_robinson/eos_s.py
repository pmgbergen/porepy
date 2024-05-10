"""Module containing general and symbolic functionality related to the standard
Peng-Robinson EoS.

It concerns itself with *analytic* expressions of thermodynamic properties as functions
depending intensive state variables.

We use ``sympy`` to produce symbols and expressions, which are
subsequently turned into numeric functions using :func:`sympy.lambdify`.

Due to the magnitude of expressions and functions (and different versions),
naming conflicts are hard to avoid.
We introcued the naming convention ``<derivative>_<name>_<type>``,
where the name represents the quantity, and type how the quantity is represented.

The convention for types of a quantity include:

- ``_s``: A symbol representing either an independent quantity, or an intermediate
  quantity serving as an argument. Created using :class:`sympy.Symbol`.
- ``_e``: A symbolic expression created using some algebraic combination of symbols.
  It depends on usually several symbols.
- ``_f``: A lambdy-generated function, based on an expression. The arguments of the
  function reflect the dependency on symbols.

For example, the compressibility factor has the standard symbol ``Z`` in literature:

Example:
    The compressibility factor has the standard symbol ``Z`` as found in the literature.

    - ``Z_s`` denotes the symbolic representation using ``sympy``. It is used as an
    intermediate dependency for e.g., departure functions.
    - ``Z_e`` denotes a symbolic **expression** dependent on some other symbols.
    In this case it is ``A_s`` and ``B_s``.
    - ``dp_Z_e`` denotes the derivative of the symbolic expression w.r.t. to its
      dependency on ``p`` for example.
      In this case ``p_s`` should be a symbols somewhere defined.
    - ``d_Z_e`` denotes the complete gradient w.r.t. all its dependencies.
    - ``Z_f`` would be a function with signature ``(float, float) -> float``.
    - ``d_Z_f`` would be a function with signature ``(float, float) -> array(2)``
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
- ``z`` overal component fractions / feed fractions
- ``sat`` volumetric phase fractions (saturations)
- ``_i`` index related to a component i
- ``_j`` index related to a phase j
- ``_r`` index related to the reference phase (the first one is assumed to be r)

"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
import sympy as sp

from .._core import COMPOSITIONAL_VARIABLE_SYMBOLS as SYMBOLS
from .._core import R_IDEAL
from ..composite_utils import safe_sum
from .pr_bip import load_bip
from .pr_components import ComponentPR

__all__ = [
    "A_CRIT",
    "B_CRIT",
    "Z_CRIT",
    "VanDerWaals_cohesion",
    "VanDerWaals_covolume",
    "PengRobinsonSymbolic",
]

logger = logging.getLogger(__name__)


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


Z_s: sp.Symbol = sp.Symbol("Z")
"""Symbol for the compressibility factor.

Intended use is as an intermediate, independent quantity to evaluate
complex symbolic expressions.

"""


A_s: sp.Symbol = sp.Symbol("A")
"""Symbol for non-dimensional cohesion.

Intended use is as an intermediate, independent quantity to evaluate complex
symbolic expressions.

"""


B_s: sp.Symbol = sp.Symbol("B")
"""Symbol for non-dimensional covolume.

Intended use is as an intermediate, independent quantity to evaluate complex
symbolic expressions.

"""


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


def red_coeff_0(A: Any, B: Any) -> Any:
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


def red_coeff_1(A: Any, B: Any) -> Any:
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
    r"""Formula for tripple root of characteristic polynomial.

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

        The lambdified expression usind module ``'numpy'`` cannot be compiled by
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
    q = red_coeff_0(A, B)
    r = red_coeff_1(A, B)

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
    q = red_coeff_0(A, B)
    r = red_coeff_1(A, B)

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
    q = red_coeff_0(A, B)
    r = red_coeff_1(A, B)

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
        The lambdified expression usind module ``'numpy'`` cannot be compiled by
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
    q = red_coeff_0(A, B)
    r = red_coeff_1(A, B)
    d = discriminant(q, r)

    t1 = sp.sqrt(d) - q * 0.5
    t2 = -1 * (sp.sqrt(d) + q * 0.5)

    t = sp.Piecewise((t2, sp.Abs(t2) > sp.Abs(t1)), (t1, True))

    u = _cbrt(t)  # TODO potential source of error

    return u - r / (3 * u) - c2 / 3


# endregion


# region Symbolic expressions for all root cases


Z_triple_e: sp.Expr = triple_root(A_s, B_s)
d_Z_triple_e: list[sp.Expr] = [Z_triple_e.diff(A_s), Z_triple_e.diff(B_s)]

Z_one_e: sp.Expr = one_root(A_s, B_s)
d_Z_one_e: list[sp.Expr] = [Z_one_e.diff(A_s), Z_one_e.diff(B_s)]

Z_double_g_e: sp.Expr = double_root(A_s, B_s, True)
d_Z_double_g_e: list[sp.Expr] = [Z_double_g_e.diff(A_s), Z_double_g_e.diff(B_s)]

Z_double_l_e: sp.Expr = double_root(A_s, B_s, False)
d_Z_double_l_e: list[sp.Expr] = [Z_double_l_e.diff(A_s), Z_double_l_e.diff(B_s)]

Z_three_g_e: sp.Expr = three_root(A_s, B_s, True)
d_Z_three_g_e: list[sp.Expr] = [Z_three_g_e.diff(A_s), Z_three_g_e.diff(B_s)]

Z_three_l_e: sp.Expr = three_root(A_s, B_s, False)
d_Z_three_l_e: list[sp.Expr] = [Z_three_l_e.diff(A_s), Z_three_l_e.diff(B_s)]

Z_three_i_e: sp.Expr = three_root_intermediate(A_s, B_s)
d_Z_three_i_e: list[sp.Expr] = [Z_three_i_e.diff(A_s), Z_three_i_e.diff(B_s)]


# endregion


# region Lambdified functions for roots depending on A and B


Z_triple_f: Callable[[float, float], float] = sp.lambdify([A_s, B_s], Z_triple_e)
d_Z_triple_f: Callable[[float, float], list[float]] = sp.lambdify(
    [A_s, B_s], d_Z_triple_e
)

Z_three_g_f: Callable[[float, float], float] = sp.lambdify([A_s, B_s], Z_three_g_e)
d_Z_three_g_f: Callable[[float, float], list[float]] = sp.lambdify(
    [A_s, B_s], d_Z_three_g_e
)
Z_three_l_f: Callable[[float, float], float] = sp.lambdify([A_s, B_s], Z_three_l_e)
d_Z_three_l_f: Callable[[float, float], list[float]] = sp.lambdify(
    [A_s, B_s], d_Z_three_l_e
)
Z_three_i_f: Callable[[float, float], float] = sp.lambdify([A_s, B_s], Z_three_i_e)
d_Z_three_i_f: Callable[[float, float], list[float]] = sp.lambdify(
    [A_s, B_s], d_Z_three_i_e
)

# because piecewise and to provide numeric evaluation of custom cubic root
_module_one_root = [{"_cbrt": np.cbrt}, "math"]

Z_one_f: Callable[[float, float], float] = sp.lambdify(
    [A_s, B_s], Z_one_e, modules=_module_one_root
)
d_Z_one_f: Callable[[float, float], list[float]] = sp.lambdify(
    [A_s, B_s], d_Z_one_e, modules=_module_one_root
)

# because piece-wise
_module_double_root = "math"

Z_double_g_f: Callable[[float, float], float] = sp.lambdify(
    [A_s, B_s], Z_double_g_e, modules=_module_double_root
)
d_Z_double_g_f: Callable[[float, float], list[float]] = sp.lambdify(
    [A_s, B_s], d_Z_double_g_e, modules=_module_double_root
)
Z_double_l_f: Callable[[float, float], float] = sp.lambdify(
    [A_s, B_s], Z_double_l_e, modules=_module_double_root
)
d_Z_double_l_f: Callable[[float, float], list[float]] = sp.lambdify(
    [A_s, B_s], d_Z_double_l_e, modules=_module_double_root
)

# endregion


def VanDerWaals_covolume(X: Sequence[Any], b: Sequence[Any]) -> Any:
    r"""
    Parameters:
        x: A sequence of fractions.
        b: A sequence of component covolume values, with the same length and order as
            ``X``.

    Returns:
        :math:`\sum_i x_i b_i`.

    """
    return safe_sum([x_i * b_i for x_i, b_i in zip(X, b)])


def VanDerWaals_cohesion(
    x: Sequence[Any],
    a: Sequence[Any],
    bip: Sequence[Sequence[Any]],
    sqrt_of_any: Optional[Callable[[Any], Any]] = sp.sqrt,
) -> Any:
    r"""
    Parameters:
        x: A sequence of fractions.
        a: A sequence of component cohesion values, with the same length and order as
            ``X``.
        bip: A nested sequence of binary interaction parameters where ``bip[i][j]`` is
            the parameter between components ``i`` and ``j``.
            Symmetric, but does upper triangle of this 2D matrix is sufficient.
        sqrt_func: ``default=``:func:`sympy.sqrt`

            A function representing the square root applicable to the input type.

    Returns:
        :math:`\sum_i\sum_k x_i x_k \sqrt{a_i a_k} (1 - \delta_{ik})`,
        where :math:`\delta` denotes the binary interaction parameter.

    """

    nc = len(x)  # number of components

    a_parts = []

    # mixture matrix is symmetric, sum over all entries in upper triangle
    # multiply off-diagonal elements with 2
    for i in range(nc):
        a_parts.append(x[i] ** 2 * a[i])
        for j in range(i + 1, nc):
            x_ij = x[i] * x[j]
            a_ij_ = sqrt_of_any(a[i] * a[j])
            delta_ij = 1 - bip[i][j]

            a_ij = a_ij_ * delta_ij

            # off-diagonal elements appear always twice due to symmetry
            a_parts.append(2.0 * x_ij * a_ij)

    return safe_sum(a_parts)


class PengRobinsonSymbolic:
    """A class providing functions for thermodynamic properties using the Peng-Robinson
    EoS, based on a symbolic representation using ``sympy``.

    Note:
        The functions are generated using :func:`sympy.lambdify` and are *sourceless*.

    Parameters:
        components: A sequence of ``num_comp`` components which are compatible
            with the Peng-Robinson EoS.

    """

    def __init__(self, components: Sequence[ComponentPR]) -> None:
        self.p_s: sp.Symbol = sp.Symbol(str(SYMBOLS["pressure"]))
        """Symbolic representation fo pressure."""

        self.T_s: sp.Symbol = sp.Symbol(str(SYMBOLS["temperature"]))
        """Symbolic representation fo temperature."""

        self.x_in_j: list[sp.Symbol] = [
            sp.Symbol(f"{SYMBOLS['phase_composition']}_{comp.name}_j")
            for comp in components
        ]
        """List of phase composition fractions associated with a phase.
        Length is equal to number of components, because every component is asssumed
        present in every phase in the unified setting."""

        self.thd_arg: tuple[sp.Symbol, sp.Symbol, list[sp.Symbol]] = (
            self.p_s,
            self.T_s,
            self.x_in_j,
        )
        """General representation of the thermodynamic argument:

        1. a pressure value,
        2. a temperature value,
        3. an array of fractions per component.

        """

        self.ext_thd_arg = [self.p_s, self.T_s, self.x_in_j, A_s, B_s, Z_s]
        """Extended thermodynamic argument (see :attr:`thd_arg`).

        The extended arguments includes:

        4. mixed non-dimensional cohesion,
        5. mixed non-dimensional covolume,
        6. compressibility factor.

        The computation and dependencies have to be split by introducing additional
        dependencies due to their complexity (compilability and efficiency).

        """

        self.A_f: Callable[[float, float, np.ndarray], float]
        """Function evaluating the non-dimensional cohesion depending on
        :attr:`thd_arg`."""

        self.d_A_f: Callable[[float, float, np.ndarray], list[float]]
        """Gradient of :attr:`A_f` returning a list of floats of length
        ``2 + num_comp``."""

        self.B_f: Callable[[float, float, np.ndarray], float]
        """Function evaluating the non-dimensional covolume depending on
        :attr:`thd_arg`t."""

        self.d_B_f: Callable[[float, float, np.ndarray], list[float]]
        """Gradient of :attr:`B_f` returning a list of floats of length
        ``2 + num_comp``."""

        self.phi_f: Callable[
            [float, float, np.ndarray, float, float, float], np.ndarray
        ]
        """Vector-valued function computing fugacity coefficients per component.

        It depends on :attr:`ext_thd_arg` because of its complexity."""

        self.d_phi_f: Callable[
            [float, float, np.ndarray, float, float, float], np.ndarray
        ]
        """Jacobian of :attr:`phi_f`, w.r.t. all dependencies.

        The Jacobian is of shape ``(num_comp, 2 + num_comp + 3)``."""

        self.h_dep_f: Callable[[float, float, np.ndarray, float, float, float], float]
        """Function evaluating the departure enthalpy depending on
        :attr:`ext_thd_arg`."""

        self.d_h_dep_f: Callable[
            [float, float, np.ndarray, float, float, float], list[float]
        ]
        """Gradient of :attr:`h_dep_f` returning a list of floats of length
        ``2 + num_comp + 3``."""

        self.h_ideal_f: Callable[[float, float, np.ndarray], float]
        """Function evaluating the ideal enthalpy depending on :attr:`thd_arg`."""

        self.d_h_ideal_f: Callable[[float, float, np.ndarray], list[float]]
        """Gradient of attr:`h_ideal_f` returning a list of floats of length
        ``2 + num_comp``."""

        self.v_f: Callable[[float, float, float], float]
        """Function evaluating the specific molar volume dependent on

        1. pressure,
        2. temperature,
        3. compressibility factor.

        """

        self.d_v_f: Callable[[float, float, float], list[float]]
        """Gradient of :attr:`v_f` returning a list of floats of length ``3``."""

        # region coterms

        b_i_crit: list[float] = [
            B_CRIT * (R_IDEAL * comp.T_crit) / comp.p_crit for comp in components
        ]
        """List of critical covolumes per component"""

        # mixed covolume
        b_e: sp.Expr = VanDerWaals_covolume(self.x_in_j, b_i_crit)
        """Mixed covolume according to the Van der Waals mixing rule."""
        B_e: sp.Expr = b_e * self.p_s / (R_IDEAL * self.T_s)
        """Non-dimensional, mixed covolume created using :attr:`b_e`"""
        d_B_e: list[sp.Expr] = [
            B_e.diff(self.thd_arg[0]),
            B_e.diff(self.thd_arg[1]),
        ] + [B_e.diff(x) for x in self.x_in_j]
        """Derivatives of :attr:`B_e` w.r.t. pressure, temperature and phase
        compositions."""

        self.B_f = sp.lambdify(self.thd_arg, B_e)
        self.d_B_f = sp.lambdify(self.thd_arg, d_B_e)

        a_i_crit: list[float] = [
            A_CRIT * (R_IDEAL**2 * comp.T_crit**2) / comp.p_crit for comp in components
        ]
        """List of critical cohesion values per component."""

        ki: list[float] = [self.a_correction_weight(comp.omega) for comp in components]
        """List of corrective weights per cohesion of components."""
        a_i_correction_e: list[sp.Expr] = [
            1 + k * (1 - sp.sqrt(self.T_s / comp.T_crit))
            for k, comp in zip(ki, components)
        ]
        """Corrective term in component cohesions (per component)."""

        ai_e: list[sp.Expr] = [
            a * corr**2 for a, corr in zip(a_i_crit, a_i_correction_e)
        ]
        """List of cohesion values per component, including a correction involving
        the critical temperature and acentric factor."""

        a_e: sp.Expr = VanDerWaals_cohesion(
            self.x_in_j, ai_e, self._compute_bips(components)
        )
        """Mixed cohesion according to the Van der Waals mixing rule."""
        A_e: sp.Expr = a_e * self.p_s / (R_IDEAL**2 * self.T_s**2)
        """Non-dimensional, mixed cohesion created using :attr:`b_e`"""
        d_A_e: list[sp.Expr] = [
            A_e.diff(self.thd_arg[0]),
            A_e.diff(self.thd_arg[1]),
        ] + [A_e.diff(x) for x in self.x_in_j]
        """Derivatives of :attr:`B_e` w.r.t. pressure, temperature and phase
        compositions."""

        self.A_f = sp.lambdify(self.thd_arg, A_e)
        self.d_A_f = sp.lambdify(self.thd_arg, d_A_e)
        # endregion

        # region Fugacity coefficients
        phi_i_e: list[sp.Expr] = []

        for i in range(len(components)):
            B_i_e = b_i_crit[i] * self.p_s / (R_IDEAL * self.T_s)
            dXi_A_e = A_e.diff(self.x_in_j[i])
            log_phi_i = (
                B_i_e / B_s * (Z_s - 1)
                - sp.ln(Z_s - B_s)
                - A_s
                / (B_s * np.sqrt(8))
                * sp.ln((Z_s + (1 + np.sqrt(2)) * B_s) / (Z_s + (1 - np.sqrt(2)) * B_s))
                * (dXi_A_e / A_s - B_i_e / B_s)
            )
            # TODO this is numerically disadvantages
            # no truncation and usage of exp
            phi_i_e.append(sp.exp(log_phi_i))

        phi_e: sp.Matrix = sp.Matrix(phi_i_e)
        """A vector-valued symbolic expression containing fugacity coefficients per
        component.

        Note:
            We need to make it vector-valued to avoid looping over individual functions.
            This is not parallelizable with numba.

        """
        d_phi_e: sp.Matrix = phi_e.jacobian(
            [self.p_s, self.T_s] + self.x_in_j + [A_s, B_s, Z_s]
        )
        """The symbolic Jacobian of ``phi_e`` w.r.t. to thermodynamic arguments and
        :data:`A_s`, :data:`B_s`, :data:`Z_s`"""

        self.phi_f = sp.lambdify(self.ext_thd_arg, phi_e)
        self.d_phi_f = sp.lambdify(self.ext_thd_arg, d_phi_e)
        # endregion

        # region Enthalpy

        dT_A_e: sp.Expr = A_e.diff(self.T_s)

        h_dep_e: sp.Expr = (R_IDEAL / np.sqrt(8)) * (
            dT_A_e * self.T_s**2 + A_s * self.T_s
        ) / B_s * sp.ln(
            (Z_s + (1 + np.sqrt(2)) * B_s) / (Z_s + (1 - np.sqrt(2)) * B_s)
        ) + R_IDEAL * self.T_s * (
            Z_s - 1
        )
        """Symbolic expression for departure enthalpy."""

        d_h_dep_e: list[sp.Expr] = [
            h_dep_e.diff(_)
            for _ in [self.p_s, self.T_s] + self.x_in_j + [A_s, B_s, Z_s]
        ]
        """Symbolic gradient of :attr:`h_dep_e` w.r.t. to thermodynamic arguments and
        :data:`A_s`, :data:`B_s`, :data:`Z_s`"""

        h_ideal_e: sp.Expr = safe_sum(
            [
                x * comp.h_ideal(self.p_s, self.T_s)
                for x, comp in zip(self.x_in_j, components)
            ]
        )
        """Symbolic expression for the ideal enthalpy."""

        d_h_ideal_e: list[sp.Expr] = [
            h_ideal_e.diff(_) for _ in [self.p_s, self.T_s] + self.x_in_j
        ]
        """Symbolic gradient of :attr:`h_ideal_e` w.r.t. to thermodynamic arguments."""

        self.h_dep_f = sp.lambdify(self.ext_thd_arg, h_dep_e)
        self.d_h_dep_f = sp.lambdify(self.ext_thd_arg, d_h_dep_e)
        self.h_ideal_f = sp.lambdify(self.thd_arg, h_ideal_e)
        self.d_h_ideal_f = sp.lambdify(self.thd_arg, d_h_ideal_e)
        # endregion

        # region Volume

        v_e: sp.Expr = Z_s * self.T_s * R_IDEAL / self.p_s
        """Symbolic expression for specific volume, depending on pressure, temperature
        and compressibility factor."""

        d_v_e: list[sp.Expr] = [v_e.diff(_) for _ in [self.p_s, self.T_s, Z_s]]
        """Symbolic gradient of specific volume, depending on pressure, temperature
        and compressibility factor."""

        self.v_f = sp.lambdify([self.p_s, self.T_s, Z_s], v_e)
        self.d_v_f = sp.lambdify([self.p_s, self.T_s, Z_s], d_v_e)
        # endregion

    def _compute_bips(
        self, components: Sequence[ComponentPR]
    ) -> Sequence[Sequence[Any]]:
        """Helper method to load the binary interaction parameters.

        TODO: Call to custom bips in PR package needs to be generalized to be able to
        handle a symbolic temperature as input.

        Parameters:
            components: A sequence of Peng-Robinson compatible components.

        Returns:
            A list of lists (2D-array-like), where per components ``i`` and ``j`` the
            interaction parameter is stored as ``result[i][j]``.

            ``result`` is strictly upper triangular, with zeros on main-diagonal and
            below.

        """

        ncomp = len(components)

        # BIP matrix, must be of shape = (ncomp, ncomp) at the end
        bips: list[list[Any]] = []

        for i in range(ncomp):
            comp_i = components[i]
            # start row by filling lower triangle part with zeros
            # (including main diagonal)
            bips_i = [0] * (i + 1)

            # loop over upper triangle part
            for j in range(i + 1, ncomp):
                comp_j = components[j]
                bip_ij = None
                # check if comp i has custom implementations
                if hasattr(comp_i, "bip_map"):
                    bip_ij_f = comp_i.bip_map.get(comp_j.CASr_number, None)
                    if bip_ij_f is not None:
                        bip_ij = bip_ij_f(self.T_s)[0]

                # check for other custom implementations
                if hasattr(comp_j, "bip_map"):
                    bip_ij_f = comp_j.bip_map.get(comp_i.CASr_number, None)
                    if bip_ij_f is not None:
                        # if the bip is still None so far, we chose the custome one
                        # from comp j
                        if bip_ij is None:
                            bip_ij = bip_ij_f(self.T_s)[0]
                        else:  # warn the user if a double implementation is detected
                            logger.warn(
                                "Detected double custom implementation of BIPs for"
                                + f"{comp_i.name} and {comp_j.name}."
                                + f"Chosing model from {comp_i.name}."
                                + "\nA fix to the model components if recommended."
                            )

                # if no custom implementation found
                if bip_ij is None:
                    bip_ij = load_bip(comp_i.CASr_number, comp_j.CASr_number)
                    # warn the user if a zero bip was loaded (likely missing data)
                    if bip_ij == 0.0:
                        logger.warn(
                            "Loaded a BIP with zero value for"
                            + f" components {comp_i.name} and {comp_j.name}."
                        )

                # If error in logic or loading of BIPs gives something unexpected
                assert (
                    bip_ij is not None
                ), f"Failed to load/find BIP for components {comp_i.name} and {comp_j.name}"
                bips_i.append(bip_ij)

            # should never happen
            assert len(bips_i) == ncomp
            bips.append(bips_i)

        # should never happen, but nevertheless asserting 2D-array-like structure
        assert len(bips) == ncomp
        return bips

    @staticmethod
    def a_correction_weight(omega: float) -> float:
        """
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
