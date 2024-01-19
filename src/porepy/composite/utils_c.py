"""Module containing utility functions for numba compiled flash computations.

The utility functions depend mostly on the order of unknowns stored in the solution
vector ``X_gen``.
The convention in the compiled flash is as follows:

1. ``num_comp - 1`` values for independent feed fractions.
2. 2 values for target equilibrium state (p-T, h-p, v-h) defining the flash procedure.
3. (optional) ``num_phase - 1`` saturation values.
4. (optional) 1 pressure value if pressure is unknown.
5. (optional) 1 temperature value if temperature is unknown.
6. ``num_phase - 1`` molar phase fraction values.
7. ``num_phase * num_comp`` molar phase compositions values, starting with the reference
   phase.

"""
from __future__ import annotations

import time

import numba
import numpy as np

from ._core import NUMBA_CACHE
from .composite_utils import COMPOSITE_LOGGER as logger

_import_start = time.time()


logger.debug(f"(import composite/utils_c.py) Compiling parsers ..\n")
# region Parsers for generic flash arguments


@numba.njit(
    "Tuple((float64[:,:],float64[:],float64[:]))(float64[:],UniTuple(int32,2))",
    fastmath=True,
    cache=True,
)
def parse_xyz(
    X_gen: np.ndarray, npnc: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to parse the fractions from a generic argument.

    NJIT-ed function with signature
    ``(float64[:], UniTuple(int32, 2)) -> Tuple(float64[:,:], float64[:], float64[:])``.

    The feed fractions are always the first ``num_comp - 1`` entries
    (feed per component except reference component).

    The phase compositions are always the last ``num_phase * num_comp`` entries,
    orderered per phase per component (phase-major order),
    with phase and component order as given by the mixture model.

    The ``num_phase - 1`` entries before the phase compositions, are always
    the independent molar phase fractions.

    Important:
        This method also computes the feed fraction of the reference component and the
        molar fraction of the reference phase.
        Hence it returns 2 values not found in ``X_gen``.

        The computed values are always the first ones in the respective vector.

    Parameters:
        X_gen: Generic argument for a flash system.
        npnc: 2-tuple containing information about number of phases and components
            (``num_phase`` and ``num_comp``).
            This information is required for pre-compilation of a mixture-independent
            function.

    Returns:
        A 3-tuple containing

        1. Phase compositions as a matrix with shape ``(num_phase, num_comp)``
        2. Molar phase fractions as an array with shape ``(num_phase,)``
        3. Feed fractions as an array with shape ``(num_comp,)``

    """
    nphase, ncomp = npnc
    # feed fraction per component, except reference component
    Z = np.empty(ncomp, dtype=np.float64)
    Z[1:] = X_gen[: ncomp - 1]
    Z[0] = 1.0 - np.sum(Z[1:])
    # phase compositions
    X = X_gen[-ncomp * nphase :].copy()  # must copy to be able to reshape
    # matrix:
    # rows have compositions per phase,
    # columns have compositions related to a component
    X = X.reshape((nphase, ncomp))
    # phase fractions, -1 because fraction of ref phase is eliminated and not part of
    # the generic argument
    Y = np.empty(nphase, dtype=np.float64)
    Y[1:] = X_gen[-(ncomp * nphase + nphase - 1) : -ncomp * nphase]
    Y[0] = 1.0 - np.sum(Y[1:])

    return X, Y, Z


@numba.njit(
    "float64[:](float64[:],float64[:,:],float64[:],UniTuple(int32,2))",
    fastmath=True,
    cache=True,
)
def insert_xy(
    X_gen: np.ndarray, x: np.ndarray, y: np.ndarray, npnc: tuple[int, int]
) -> np.ndarray:
    """Helper function to insert phase compositions and molar fractions into a generic
    argument.

    Essentially a reverse operation for :func:`parse_xyz`, with the exception
    that ``z`` is assumed to never be modified.

    """
    nphase, ncomp = npnc

    # insert independent phase fractions
    X_gen[-(ncomp * nphase + nphase - 1) : -ncomp * nphase] = y[1:]
    # ravel phase compositions
    X_gen[-ncomp * nphase :] = x.ravel()
    return X_gen


@numba.njit("float64[:](float64[:],UniTuple(int32,2))", cache=True)
def parse_pT(X_gen: np.ndarray, npnc: tuple[int, int]) -> np.ndarray:
    """Helper function extracing pressure and temperature from a generic
    argument for the p-T flash.

    NJIT-ed function with signature
    ``(float64[:], UniTuple(int32, 2)) -> float64[:]``.

    Pressure and temperature are the last two values before the independent molar phase
    fractions (``num_phase - 1``) and the phase compositions (``num_phase * num_comp``).

    Parameters:
        X_gen: Generic argument for a flash system.
        npnc: 2-tuple containing information about number of phases and components
            (``num_phase`` and ``num_comp``).
            This information is required for pre-compilation of a mixture-independent
            function.

    Returns:
        An array with shape ``(2,)``.

    """
    nphase, ncomp = npnc
    return X_gen[-(ncomp * nphase + nphase - 1) - 2 : -(ncomp * nphase + nphase - 1)]


@numba.njit("float64[:](float64[:],float64,float64,UniTuple(int32,2))", cache=True)
def insert_pT(
    X_gen: np.ndarray, p: float, T: float, npnc: tuple[int, int]
) -> np.ndarray:
    """Helper function to insert pressure and temperature into a generic argument.

    Essentially a reverse operation for :func:`parse_pT`.

    """
    nphase, ncomp = npnc
    X_gen[-(ncomp * nphase + nphase - 1 + 2)] = p
    X_gen[-(ncomp * nphase + nphase - 1 + 1)] = T
    return X_gen


@numba.njit("float64[:](float64[:],UniTuple(int32,2))", fastmath=True, cache=True)
def parse_target_state(X_gen: np.ndarray, npnc: tuple[int, int]) -> np.ndarray:
    """Helper function extracing the values for the equilibrium state definition.

    NJIT-ed function with signature
    ``(float64[:], UniTuple(int32, 2)) -> float64[:]``.

    The two values required to define the equilibrium state are always the first 2
    values after the ``num_comp - 1`` feed fractions.

    Parameters:
        X_gen: Generic argument for a flash system.
        npnc: 2-tuple containing information about number of phases and components
            (``num_phase`` and ``num_comp``).
            This information is required for pre-compilation of a mixture-independent
            function.

    Returns:
        An array with shape ``(2,)``.

    """
    _, ncomp = npnc
    return X_gen[ncomp - 1 : ncomp + 1]


@numba.njit("float64[:](float64[:],UniTuple(int32,2))", fastmath=True, cache=True)
def parse_sat(X_gen: np.ndarray, npnc: tuple[int, int]) -> np.ndarray:
    """Helper function extracing the saturation values.

    NJIT-ed function with signature
    ``(float64[:], UniTuple(int32, 2)) -> float64[:]``.

    Saturation values are always after the ``num_comp - 1`` feed fraction and two other
    equilibrium values.

    Note:
        This might change when adding the v-T flash. TODO

    Important:
        This method assumes that only saturation values for independent phases are
        present (i.e. ``num_phase - 1`` values in ``X_gen``).

        But it returns a vector shape ``(num_phase,)`` where the first value represents
        the saturation value for the reference phase expressed by unity.

    Parameters:
        X_gen: Generic argument for a flash system.
        npnc: 2-tuple containing information about number of phases and components
            (``num_phase`` and ``num_comp``).
            This information is required for pre-compilation of a mixture-independent
            function.

    Returns:
        An array with shape ``(num_phase,)``.

    """
    nphase, ncomp = npnc
    sat = np.empty(nphase, dtype=np.float64)
    sat[1:] = X_gen[ncomp + 1 : ncomp + 1 + nphase - 1]
    sat[0] = 1.0 - np.sum(sat[1:])
    return sat


@numba.njit("float64[:](float64[:],float64[:],UniTuple(int32,2))", cache=True)
def insert_sat(X_gen: np.ndarray, sat: np.ndarray, npnc: tuple[int, int]) -> np.ndarray:
    """Helper function to insert phase saturations into a generic argument.

    Essentially a reverse operation for :func:`parse_sat`.

    """
    _, ncomp = npnc
    X_gen[ncomp + 2 : 2 * ncomp] = sat
    return sat


# endregion


logger.debug(f"(import composite/utils_c.py) Compiling miscellaneous functions ..\n")


@numba.njit("float64[:,:](float64[:,:])", fastmath=True, cache=True)
def normalize_fractions(x: np.ndarray) -> np.ndarray:
    """Takes a matrix of phase compositions (rows - phase, columns - component)
    and normalizes the fractions.

    Meaning it divides each matrix element by the sum of respective row.

    NJIT-ed function with signature ``(float64[:,:]) -> float64[:,:]``.

    Parameters:
        x: ``shape=(num_phases, num_components)``

            A matrix of phase compositions, containing per row the (extended)
            fractions per component.

    Returns:
        A normalized version of ``X``, with the normalization performed row-wise
        (phase-wise).

    """
    return (x.T / x.sum(axis=1)).T


@numba.njit("float64[:](float64[:],float64[:])", fastmath=True, cache=True)
def extended_compositional_derivatives(df_dxn: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Expands the derivatives of a scalar function :math:`f(p, T, x_n)`, assuming
    the its derivatives are given w.r.t. to the normalized fractions
    (see :func:`normalize_fractions`).

    Expansion is conducted by simply applying the chain rule to :math:`f(x_n(x))`.

    Intended use is for thermodynamic properties given by :class:`EoSCompiler`, which
    are given as functions with above signature.

    Parameters:
        df_dxn: ``shape=(2 + num_components,)``

            The gradient of a scalar function w.r.t. to pressure, temperature and
            normalized fractions in a phase.
        x: ``shape=(num_components,)``

            The extended fractions for a phase.

    Returns:
        An array with the same shape as ``df_dxn`` where the chain rule was applied.

    """
    assert len(df_dxn) >= len(x), "Dimension mismatch (number of derivatives)."
    df_dx = df_dxn.copy()  # deep copy to avoid messing with values
    ncomp = x.shape[0]
    # constructing the derivatives of xn_ij = x_ij / (sum_k x_kj)
    x_sum = np.sum(x)
    dxn = np.eye(ncomp) / x_sum - np.outer(x, np.ones(ncomp)) / (x_sum**2)
    # dxn = np.eye(ncomp) / x_sum - np.column_stack([x] * ncomp) / (x_sum ** 2)
    # assuming derivatives w.r.t. normalized fractions are in the last num_comp elements
    df_dx[-ncomp:] = df_dx[-ncomp:].dot(dxn)

    return df_dx


# TODO this can be turned into a numpy universal function to collapse to single function
# TODO cache dependency on base function
@numba.njit(
    "float64[:,:](float64[:,:],float64[:,:])",
    parallel=True,
    fastmath=True,
    cache=NUMBA_CACHE,
)
def extended_compositional_derivatives_v(
    df_dxn: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Same as :func:`extended_compositional_derivatives`, only efficiently vectorized.

    Parameters:
        df_dxn: ``shape=(2 + num_components, M)``

            The gradient of a scalar function w.r.t. to pressure, temperature and
            normalized fractions in a phase.

            The derivatives are expected to be given row-wise.

        x: ``shape=(num_components, M)``

            The extended fractions for a phase (row-wise).

    Returns:
        An array with the same shape as ``df_dxn`` where the chain rule was applied.

    """
    assert df_dxn.shape[1] == x.shape[1], "Dimension mismatch (values)."
    assert df_dxn.shape[0] >= x.shape[0], "Dimension mismatch (number of derivatives)."
    df_dx = np.empty_like(df_dxn)
    _, N = x.shape
    for i in numba.prange(N):
        df_dx[:, i] = extended_compositional_derivatives(df_dxn[:, i], x[:, i])
    return df_dx


@numba.njit("float64[:](float64[:],float64[:],float64)", fastmath=True, cache=True)
def compute_saturations(y: np.ndarray, rho: np.ndarray, eps: float) -> np.ndarray:
    r"""Computes the saturation values by solving the phase fraction relations of form

    .. math::

        \left(\sum_k s_k \rho_k\right) y_j - \rho_j s_j = 0~,~ \sum_k s_k - 1 = 0~.

    Parameters:
        y: ``shape=(num_phase,)``

            A vector of molar phase fractions, assuming the first value belongs to the
            reference phase.
        rho: ``shape=(num_phase,)``

            A vector of phase densities, corresponding to ``y``. Must be of same length
            as ``y``.
        eps: A small number to determin saturated phases (``y_j > 1- eps``).

    Raises:
        AssertionError: If ``y`` and ``rho`` are of unequal shape.
        AssertionError: If more than one phase is saturated.
        AssertionError: If the computed saturations violate the unity constraint.

    Returns:
        A vector of analogous shape (and phase order) containing saturation values.

    """
    assert y.shape == rho.shape, "Mismatch in given numbers of fractions and densities."
    nphase = y.shape[0]

    s = np.zeros_like(y)

    if nphase == 1:
        s = np.ones(1)
    else:
        # if any phase is saturated
        saturated = y >= 1.0 - eps
        # sanity check that only one phase is saturated
        assert saturated.sum() == 1, "More than 1 phase saturated."

        # 2-phase saturation evaluation can be done analytically
        if nphase == 2:
            if np.any(saturated):
                s[saturated] = 1.0
            # if no phase is saturated, solve equations.
            else:
                # s_1 = 1. / (1. - (y_1 + 1) / y_1 * rho_1 / rho_2)
                # same with additional y_1 + y_2 - 1 = 0
                s[0] = 1.0 / (1.0 + y[1] / (1 - y[1]) * rho[0] / rho[1])
                s[1] = 1.0 - s[0]
        # More than 2 phases requires the inversion of the matrix given by
        # phase fraction relations
        else:
            if np.any(saturated):
                s[saturated] = 1.0
            else:
                not_vanished = y > eps
                # per logic, there can't be a saturated phase, i.e. at least 2 present
                y_ = y[not_vanished]
                rho_ = rho[not_vanished]

                n = y_.shape[0]
                # solve j=1..n equations (sum_k s_k rho_k) y_j - s_j rho_j = 0
                # where in each equation, s_j is replaced by 1 - sum_k!=j s_k
                rhs = np.ones(n, dtype=np.float64)
                mat = np.array(
                    [1.0 - rho_ / rho_[j] * y_[j] / (1 - y_[j]) for j in range(n)],
                    dtype=np.float64,
                )
                np.fill_diagonal(mat, 0.0)

                s_ = np.linalg.solve(mat, rhs)
                s[not_vanished] = s_

    # sanity check of results, should not happen.
    assert s.sum() <= 1.0 + eps, "Computed saturations violate unity constraint."
    return s


@numba.njit(
    "float64[:,:](float64[:,:],float64[:,:],float64)",
    parallel=True,
    fastmath=True,
    cache=NUMBA_CACHE,
)
def compute_saturations_v(y: np.ndarray, rho: np.ndarray, eps: float) -> np.ndarray:
    """Parallelized form of :func:`compute_saturations` for vectorized input.

    Parameters:
        y: ``shape=(num_phase, N)``

            Matrix containing row-wise molar phase fractions per phase.
        rho: ``shape=(num_phase, N)``

            Same for densities.
        eps: Same as for non-parallelized version.

    Returns:
        An array with ``shape=(num_phase, N)`` containing row-wise saturation values per
        phase.

    """
    assert y.shape == rho.shape, "Mismatch in given numbers of fractions and densities."
    s = np.empty_like(y)
    for n in numba.prange(y.shape[1]):
        s[:, n] = compute_saturations(y[:, n], rho[:, n], eps)
    return s


_import_end = time.time()

logger.debug(
    "(import composite/utils_c.py)"
    + f" Done (elapsed time: {_import_end - _import_start} (s)).\n\n"
)

del _import_start, _import_end
