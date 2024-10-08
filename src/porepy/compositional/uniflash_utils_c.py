"""Module containing utility functions for numba compiled flash computations.

It introduces the indexing convention for the *generic flash argumen*.

Assuming any flash is given by a residual and Jacobian function ``F, DF``, the generic
argument is a specially sorted array, containing unknowns, but also parameters.
Parameters include foremostly the thermodynamic target state (pressure values in the pT
flash for example.)

The order in the generic argument is as follows:

1. ``num_comp - 1`` values for independent feed fractions.
2. 2 values for target equilibrium state (p-T, h-p, v-h) defining the flash procedure.
3. (optional) ``num_phase - 1`` saturation values for flashes with given volume.
4. (optional) 1 pressure value if pressure is unknown.
5. (optional) 1 temperature value if temperature is unknown.
6. ``num_phase - 1`` phase fraction values of independent phases.
7. ``num_phase * num_comp`` extended phase compositions values.

The order of fractions and saturations corresponds to the order of phases and components
in a fluid mixture.

"""

from __future__ import annotations

import numba
import numpy as np

from ._core import NUMBA_FAST_MATH


@numba.njit(
    "Tuple((float64[:,:],float64[:],float64[:]))(float64[:],UniTuple(int32,2))",
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_FAST_MATH,
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
    fastmath=NUMBA_FAST_MATH,
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


@numba.njit(
    "float64[:](float64[:],UniTuple(int32,2))", fastmath=NUMBA_FAST_MATH, cache=True
)
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


@numba.njit(
    "float64[:](float64[:],UniTuple(int32,2))", fastmath=NUMBA_FAST_MATH, cache=True
)
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
    return X_gen
