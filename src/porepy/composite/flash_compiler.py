"""This module contains a compiled versions of flash-system assemblers.

The functions assembled and NJIT-compiled here, for a given mixture,
provide an efficient evaluation of the residual of flash equations and an efficient
assembly of the Jacobian.

The functions are such that they can be used to solve multiple flash problems in
parallel.

The expressions and compiled functions are generated using a combination of ``sympy``
and ``numba``.

"""
from __future__ import annotations

from typing import Callable, Literal

import numba
import numpy as np
import sympy as sp

import porepy as pp

from ._core import COMPOSITIONAL_VARIABLE_SYMBOLS as SYMBOLS
from .composite_utils import safe_sum
from .mixture import BasicMixture


@numba.njit(
    "Tuple(float64[:,:], float64[:], float64[:])(float64[:], UniTuple(int32, 2))",
    fastmath=True,
    cache=True,
)
def parse_xyz(
    X_gen: np.ndarray, mpmc: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to parse the fractions from a generic argument.

    NJIT-ed function with signature
    ``(float64[:], UniTuple(int32, 2)) -> Tuple(float64[:,:], float64[:], float64[:])``.

    The feed fractions are always the first ``num_comp - 1`` entries
    (feed per component except reference component).

    The phase compositions are always the last ``num_phase * num_comp`` entries,
    orderered per phase per component (phase-major order),
    with phase and component order as given by the mixture model.

    the ``num_phase - 1`` entries before the phase compositions, are always
    the independent molar phase fractions.

    Important:
        This method also computes the feed fraction of the reference component and the
        molar fraction of the reference phase.
        Hence it returns 2 values not found in ``X_gen``.

        The computed values are always the first ones in the respective vector.

    Parameters:
        X_gen: Generic argument for a flash system.
        mpmc: 2-tuple containing information about number of phases and components
            (``num_phase`` and ``num_comp``).
            This information is required for pre-compilation of a mixture-independent
            function.

    Returns:
        A 3-tuple containing

        1. Phase compositions as a matrix with shape ``(num_phase, num_comp)``
        2. Molar phase fractions as an array with shape ``(num_phase,)``
        3. Feed fractions as an array with shape ``(num_comp,)``

    """
    nphase, ncomp = mpmc
    # feed fraction per component, except reference component
    Z = np.empty(ncomp, dtype=np.float64)
    Z[1:] = X_gen[: ncomp - 1]
    Z[0] = 1.0 - np.sum(Z[1:])
    # phase compositions
    X = X_gen[-ncomp * nphase :].copy()  # must copy to be able to reshape
    # matrix:
    # rows have compositions per phase,
    # columns have compositions related to a component
    X = X.reshape((ncomp, nphase))
    # phase fractions, -1 because fraction of ref phase is eliminated and not part of
    # the generic argument
    Y = np.empty(nphase, dtype=np.float64)
    Y[1:] = X_gen[-(ncomp * nphase + nphase - 1) : -ncomp * nphase]
    Y[0] = 1.0 - np.sum(Y[1:])

    return X, Y, Z


@numba.njit(
    "float64[:](float64[:], UniTuple(int, 2))",
    fastmath=True,
    cache=True,
)
def parse_pT(X_gen: np.ndarray, mpmc: tuple[int, int]) -> np.ndarray:
    """Helper function extracing pressure and temperature from a generic
    argument.

    NJIT-ed function with signature
    ``(float64[:], UniTuple(int32, 2)) -> float64[:]``.

    Pressure and temperature are the last two values before the independent molar phase
    fractions (``num_phase - 1``) and the phase compositions (``num_phase * num_comp``).

    Parameters:
        X_gen: Generic argument for a flash system.
        mpmc: 2-tuple containing information about number of phases and components
            (``num_phase`` and ``num_comp``).
            This information is required for pre-compilation of a mixture-independent
            function.

    Returns:
        An array with shape ``(2,)``.

    """
    nphase, ncomp = mpmc
    return X_gen[-(ncomp * nphase + nphase - 1) - 2 : -(ncomp * nphase + nphase - 1)]


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


@numba.njit(
    "float64[:](float64[:,:], float64[:], float64[:])",
    fastmath=True,
    cache=True,
)
def mass_conservation_res(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the mass conservation equations.

    For each component ``i``, except reference component, it holds

    ... math::

        z\left[i\right] - \sum_j y\left[j\right] x\left[j, i\right] = 0

    Number of phases and components is determined from the chape of ``x``.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:], float64[:]) -> float64[:]``.

    Important:
        See :func:`parse_xyz` for obtaining the properly formatted arguments ``x,y,z``.

    Parameters:
        x: ``shape=(num_phase, num_comp)``

            Phase compositions
        y: ``shape=(num_phase,)``

            Molar phase fractions.
        z: ``shape=(num_comp,)``

            Overall fractions per component.

    Returns:
        An array with ``shape=(num_comp - 1,)`` containg the residual of the mass
        conservation equation (left-hand side of above equation) for each component,
        except the first one (in ``z``).

    """
    res = np.empty(z.shape[0] - 1, dtype=np.float64)

    for i in range(1, z.shape[0]):
        res[i - 1] = z[i] - np.sum(y * x[:, i])

    return res


@numba.njit(
    "float64[:,:](float64[:,:], float64[:])",
    fastmath=True,
    cache=True,
)
def mass_conservation_jac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the Jacobian of the residual described in
    :func:`mass_conservation_res`

    The Jacobian is of shape ``(num_comp - 1, num_phase - 1 + num_phase * num_comp)``.
    The derivatives (columns) are taken w.r.t. to each independent molar fraction,
    except feed fractions.

    The order of derivatives w.r.t. phase compositions is in phase-major order.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:,:]``.

    Note:
        The Jacobian does not depend on the overall fractions ``z``, since they are
        assumed given and constant, hence only relevant for residual.

    """
    nphase, ncomp = x.shape

    # must fill with zeros, since slightly sparse and below fill-up does not cover
    # elements which are zero
    jac = np.zeros((ncomp - 1, nphase - 1 + nphase * ncomp), dtype=np.float64)

    for i in range(ncomp - 1):
        # (1 - sum_j y_j) x_ir + y_j x_ij is there, per phase
        # hence d mass_i / d y_j = x_ij - x_ir
        jac[i, : nphase - 1] = x[1:, i + 1] - x[0, i + 1]  # i + 1 to skip ref component

        # d.r.t. w.r.t x_ij is always y_j for all j per mass conv.
        jac[i, nphase + i :: nphase] = y  # nphase -1 + i + 1 to skip ref component

    # -1 because of z - z(x,y) = 0
    # and above is d z(x,y) / d[y, x]
    return (-1) * jac


@numba.njit("float64[:](float64[:], float64[:,:])", fastmath=True, cache=True)
def complementary_conditions_res(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the complementary conditions.

    For each phase ``j`` it holds

    ... math::

        y\left[j\right] \cdot \left(1 - \sum_i x\left[j, i\right]\right) = 0

    Number of phases and components is determined from the chape of ``x``.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:]``.

    Important:
        See :func:`parse_xyz` for obtaining the properly formatted arguments ``x,y``.

    Parameters:
        x: ``shape=(num_phase, num_comp)``

            Phase compositions
        y: ``shape=(num_phase,)``

            Molar phase fractions.

    Returns:
        An array with ``shape=(num_phase,)`` containg the residual of the complementary
        condition per phase.

    """
    nphase = y.shape[0]
    ccs = np.empty(nphase, dtype=np.float64)
    for j in range(nphase):
        ccs[j] = y[j] * (1.0 - np.sum(x[j]))

    return ccs


@numba.njit("float64[:,:](float64[:], float64[:,:])", fastmath=True, cache=True)
def complementary_conditions_jac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the Jacobian of the residual described in
    :func:`complementary_conditions_res`

    The Jacobian is of shape ``(num_phase, num_phase - 1 + num_phase * num_comp)``.
    The derivatives (columns) are taken w.r.t. to each independent molar fraction,
    except feed fractions.

    The order of derivatives w.r.t. phase compositions is in phase-major order.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:,:]``.

    """
    nphase, ncomp = x.shape
    # must fill with zeros, since matrix sparsely populated.
    d_ccs = np.zeros((nphase, nphase - 1 + nphase * ncomp), dtype=np.float64)

    unities = 1 - np.sum(x, axis=1)

    # first complementary condition is w.r.t. to reference phase
    # (1 - sum_j y_j) * (1 - sum_i x_i0)
    d_ccs[0, : nphase - 1] = (-1) * unities[0]
    d_ccs[0, nphase - 1 : nphase - 1 + ncomp] = y[0] * (-1)
    for j in range(1, nphase):
        # for the other phases, its slight easier since y_j * (1 - sum_i x_ij)
        d_ccs[j, j - 1] = unities[j]
        d_ccs[j, nphase - 1 + j * ncomp : nphase - 1 + (j + 1) * ncomp] = y[j] * (-1)

    return d_ccs


class MixtureSymbols:
    """A class containing basic symbols (thermodynamic properties and variables) for a
    mixture represented using ``sympy``.

    It is meant for symbols which are considered primary or independent or given,
    e.g., pressure or molar fractions, not for higher expressions.

    """

    def __init__(self, mixture: BasicMixture) -> None:
        self.p: sp.Symbol = sp.Symbol(str(SYMBOLS["pressure"]))
        """Symbolic representation fo pressure."""
        self.T: sp.Symbol = sp.Symbol(str(SYMBOLS["temperature"]))
        """Symbolic representation fo temperature."""
        self.h: sp.Symbol = sp.Symbol(str(SYMBOLS["enthalpy"]))
        """Symbolic representation fo specific molar enthalpy."""
        self.v: sp.Symbol = sp.Symbol(str(SYMBOLS["volume"]))
        """Symbolic representation fo specific molar volume."""

        self.z: list[sp.Symbol | sp.Expr] = [
            sp.Symbol(f"{SYMBOLS['component_fraction']}_{comp.name}")
            for comp in mixture.components
        ]
        """List of symbolic representations of overal molar fractions per component.

        The fraction of the reference component is eliminated by unity.

        Ordered as in the mixture class.

        """

        self.y: list[sp.Symbol | sp.Expr] = [
            sp.Symbol(f"{SYMBOLS['phase_fraction']}_{phase.name}")
            for phase in mixture.phases
        ]
        """List of symbolic representations of molar phase fractions per phase.

        The fraction of the reference phase is eliminated by unity.

        Ordered as in the mixture class.

        """

        self.s: list[sp.Symbol | sp.Expr] = [
            sp.Symbol(f"{SYMBOLS['phase_saturation']}_{phase.name}")
            for phase in mixture.phases
        ]
        """List of symbolic representations of volumetric phase fractions per phase.

        Analogous to :attr:`y`.

        """

        self.x_per_i: list[sp.Symbol] = []
        """List of phase composition fractions associated with a component.
        Length is equal to number of phases, because every phase is modelled in the
        unified setting.

        The phase indices are given by
        - ``_R`` denoting the reference phase
        - ``_G`` denoting the one assumed gas phase
        - ``_<number>`` some number for additional liquid-like phases

        """

        self.x_in_j: list[sp.Symbol] = [
            sp.Symbol(f"{SYMBOLS['phase_composition']}_{comp.name}_j")
            for comp in mixture.components
        ]
        """List of phase composition fractions associated with a phase.
        Length is equal to number of components, because every component is asssumed
        present in every phase in the unified setting."""

        # representations for symbols eliminated by unity
        z_r = 1.0 - safe_sum(self.z)
        self.z = [
            z if comp != mixture.reference_component else z_r
            for z, comp in zip(self.z, mixture.components)
        ]

        y_r = 1.0 - safe_sum(self.y)
        self.y = [
            y if phase != mixture.reference_phase else y_r
            for y, phase in zip(self.y, mixture.phases)
        ]

        s_r = 1.0 - safe_sum(self.s)
        self.s = [
            s if phase != mixture.reference_phase else s_r
            for s, phase in zip(self.s, mixture.phases)
        ]

        # layout of X per i, with more information on the phase.
        _np = 1
        for phase in mixture.phases:
            if phase == mixture.reference_phase:
                self.x_per_i.append(sp.Symbol(f"{SYMBOLS['phase_composition']}_i_R"))
            else:
                # NOTE assuming only 1 gas phase
                if phase.gaslike:
                    self.x_per_i.append(
                        sp.Symbol(f"{SYMBOLS['phase_composition']}_i_G")
                    )
                else:
                    self.x_per_i.append(
                        sp.Symbol(f"{SYMBOLS['phase_composition']}_i_{_np}")
                    )
                    _np += 1


class FlashCompiler:
    """Class implementing NJIT-compiled representation of the equilibrium equations
    using numba and sympy.

    It uses the no-python mode of numba to produce highly efficient, compiled code.

    """

    def __init__(self, mixture: pp.composite.NonReactiveMixture) -> None:
        self._mpmc: tuple[int, int] = (mixture.num_phases, mixture.num_components)
        """2-tuple containing the number of phases and components.

        This is an import argument for many function compilations
        """

        self._gaslike: tuple[int, ...] = tuple(
            [int(phase.gaslike) for phase in mixture.phases]
        )
        """Tuple of length ``num_phase`` containing flags if modelled phases are
        gas-like. Only one gaslike phase should be allowed"""
        assert sum(self._gaslike) == 1, "Mixture must have exactly 1 gas-like phase."

        self.symbols: MixtureSymbols = MixtureSymbols(mixture)

        self.residuals: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the respective residuals as a callable."""

        self.jacobians: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the respective Jacobian as a callable."""

        self._compile_flash_systems(mixture)

    def _compile_flash_systems(self, mixture: pp.composite.NonReactiveMixture) -> None:
        """Creates compiled callables representing various flash systems.

        The callables represent residuals and Jacobians of the flash equations, and
        their size depends on the number of present phases and components.

        """
        nphase, ncomp = self._mpmc

        # number of equations for the pT system
        # ncomp - 1 mass constraints
        # (nphase - 1) * ncomp fugacity constraints (w.r.t. ref phase formulated)
        # nphase complementary conditions
        pT_dim = ncomp - 1 + (nphase - 1) * ncomp + nphase

        @numba.njit("float64[:](float64[:])")
        def F_pT(X_gen: np.ndarray) -> np.ndarray:
            x, y, z = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))

            # declare residual array of proper dimension
            res = np.empty(pT_dim, dtype=np.float64)

            res[: ncomp - 1] = mass_conservation_res(x, y, z)
            # last nphase equations are always complementary conditions
            res[-nphase:] = complementary_conditions_res(x, y)

            return res

        @numba.njit("float64[:,:](float64[:])")
        def DF_pT(X_gen: np.ndarray) -> np.ndarray:
            x, y, z = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))

            # declare Jacobian or proper dimension
            jac = np.empty((pT_dim, pT_dim), dtype=np.float64)

            jac[: ncomp - 1] = mass_conservation_jac(x, y)
            # last nphase equations are always complementary conditions
            jac[-nphase:] = complementary_conditions_jac(x, y)

            return jac

        self.residuals.update(
            {
                "p-T": F_pT,
            }
        )

        self.jacobians.update(
            {
                "p-T": DF_pT,
            }
        )
