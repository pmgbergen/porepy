"""Contains utility functions for the compositional subpackage, as well as a custom
exception class :class:`CompositionalModellingError`."""

from __future__ import annotations

from typing import Sequence, TypeVar, cast

import numba
import numpy as np

from ._core import NUMBA_CACHE, NUMBA_FAST_MATH, NUMBA_PARALLEL

__all__ = [
    "safe_sum",
    "normalize_rows",
    "chainrule_fractional_derivatives",
    "compute_saturations",
    "CompositionalModellingError",
]


_Addable = TypeVar("_Addable")
"""A type variable representing any type supporting the + overload.

Note:
    Used in :func:`safe_sum` to state that the return value type is the same as the
    argument type.

"""


def safe_sum(x: Sequence[_Addable]) -> _Addable:
    """Safely sum the elements, without creating a first addition with 0.

    Important for AD operators to avoid overhead.

    Parameters:
        x: A sequence of any objects which support the ``+`` operation.

    Returns:
        The sum of ``x``.

    """
    if len(x) >= 1:
        # TODO do we need a copy here? Test extensively
        sum_ = x[0]
        for i in range(1, len(x)):
            # Using TypeVar to indicate that return type is same as argument type
            # MyPy says that the TypeVar has no __add__, hence not adable...
            sum_ = sum_ + x[i]  # type: ignore[operator]
        return sum_
    else:
        return cast(_Addable, 0)


@numba.njit("float64[:,:](float64[:,:])", fastmath=NUMBA_FAST_MATH, cache=True)
def normalize_rows(x: np.ndarray) -> np.ndarray:
    """Takes a 2D array and normalizes it row-wise.

    Each row vector is divided by the sum of row elements.

    Inteded use is for families of fractional variables, which ought to be normalized
    such that they fulfill the unity constraint.

    NJIT-ed function with signature ``(float64[:,:]) -> float64[:,:]``.

    Parameters:
        x: ``shape=(N, M)``

            Rectangular 2D array.

    Returns:
        A normalized version of ``X``, with the normalization performed row-wise.

    """
    return (x.T / x.sum(axis=1)).T


@numba.njit("float64[:](float64[:],float64[:])", fastmath=NUMBA_FAST_MATH, cache=True)
def _chainrule_fractional_derivatives(df_dxn: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Internal ``numba.njit``-decorated function for
    :meth:`chainrule_fractional_derivatives` for non-vectorized input."""

    df_dx = df_dxn.copy()  # deep copy to avoid messing with values
    ncomp = x.shape[0]
    # constructing the derivatives of xn_ij = x_ij / (sum_k x_kj)
    x_sum = np.sum(x)
    dxn = np.eye(ncomp) / x_sum - np.outer(x, np.ones(ncomp)) / (x_sum**2)
    # assuming derivatives w.r.t. normalized fractions are in the last num_comp elements
    df_dx[-ncomp:] = df_dx[-ncomp:].dot(dxn)

    return df_dx


# NOTE Use guvectorize, not vectorize, because the return value is a 2D array
# vectorize cannot cope with everything, must instruct numba about the output shape
@numba.guvectorize(
    ["void(float64[:],float64[:],float64[:],float64[:])"],
    "(m),(n),(m)->(m)",
    target="parallel" if NUMBA_PARALLEL else "cpu",
    nopython=True,
    cache=NUMBA_CACHE,  # NOTE cache depends on internal function
)
def _chainrule_fractional_derivatives_gu(
    df_dxn: np.ndarray, x: np.ndarray, out: np.ndarray, dummy: np.ndarray
) -> None:
    """Internal ``numba.guvectorize``-decorated function for
    :meth:`chainrule_fractional_derivatives`."""
    out[:] = _chainrule_fractional_derivatives(df_dxn, x)


def chainrule_fractional_derivatives(df_dxn: np.ndarray, x: np.ndarray) -> np.ndarray:
    r"""Applies the chain rule to the derivatives of a scalar function
    :math:`f(y, \tilde{x})`, assuming its derivatives are given w.r.t. to the normalized
    fractions :math:`\tilde{x}_i = \frac{x_i} / \frac{\sum_j x_j}`.

    Intended use is for thermodynamic properties in the unified formulation, which
    are given as functions with above signature.

    Utilizes numba for parallelized, efficient computations.

    Parameters:
        df_dxn: ``shape=(N + num_components, M)``

            The gradient of a scalar function w.r.t. to ``y`` and ``num_components``
            normalized fractions in a phase.

            The derivatives are expected to be given row-wise.

        x: ``shape=(num_components, M)``

            The extended fractions for a phase (row-wise).

    Raises:
        ValueError: If vectorized input is of mismatching dimensions (M).
        ValueError: If ``df_dxn`` has fewer rows than than ``x``
            (number of derivatives must be at least length of ``x``)

    Returns:
        An array with the same shape as ``df_dxn`` where the chain rule was applied.

    """
    if df_dxn.shape[0] < x.shape[0]:
        raise ValueError(
            "Axis 0 of Argument 1 must be at least the size of Axis 0 of Argument 2."
        )

    # allowing 1D arrays
    if len(df_dxn.shape) > 1:
        if df_dxn.shape[1] != x.shape[1]:
            raise ValueError("Dimensions in Axis 1 mismatch.")

        # NOTE Transpose to parallelize over values, not derivatives
        df_dxn_T = df_dxn.T
        df_dx = np.empty_like(df_dxn_T)
        _chainrule_fractional_derivatives_gu(df_dxn_T, x.T, df_dx)

        df_dx = df_dx.T
    else:
        if len(x.shape) != 1:
            raise ValueError("Dimensions in Axis 1 mismatch.")
        df_dx = _chainrule_fractional_derivatives(df_dxn, x)

    return df_dx


@numba.njit(
    "float64[:](float64[:],float64[:],float64)", fastmath=NUMBA_FAST_MATH, cache=True
)
def _compute_saturations(y: np.ndarray, rho: np.ndarray, eps: float) -> np.ndarray:
    """Internal ``numba.njit``-decorated function for :meth:`compute_saturations` for
    non-vectorized input."""

    assert y.shape == rho.shape, "Mismatch in given numbers of fractions and densities."
    nphase = y.shape[0]

    s = np.zeros_like(y)

    if nphase == 1:
        s = np.ones(1)
    else:
        # if any phase is saturated
        saturated = y >= 1.0 - eps
        # sanity check that only one phase is saturated
        if np.any(saturated):
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
                rhs = rho_ * (y_ - 1.0)
                mat = np.empty((n, n), dtype=np.float64)
                for j in range(n):
                    mat[j] = rho_[j] * (y_[j] - 1) - rho_ * y_[j]
                np.fill_diagonal(mat, 0.0)

                s_ = np.linalg.solve(mat, rhs)
                s[not_vanished] = s_

    return s


@numba.guvectorize(
    ["void(float64[:],float64[:],float64,float64[:],float64[:])"],
    "(n),(n),(),(n)->(n)",
    target="parallel" if NUMBA_PARALLEL else "cpu",
    nopython=True,
    cache=NUMBA_CACHE,  # NOTE cache depends on internal function
)
def _compute_saturations_gu(
    y: np.ndarray, rho: np.ndarray, eps: float, out: np.ndarray, dummy: np.ndarray
) -> None:
    """Internal ``numba.guvectorize``-decorated function for
    :meth:`compute_saturations`."""
    out[:] = _compute_saturations(y, rho, eps)


def compute_saturations(
    y: np.ndarray, rho: np.ndarray, eps: float = 1e-10
) -> np.ndarray:
    r"""Computes the saturation values by solving the phase mass conservation

    .. math::

        \left(\sum_k s_k \rho_k\right) y_j - \rho_j s_j = 0~,~ \sum_k s_k - 1 = 0~.

    Utilizes numba for parallelized, efficient computations.

    Parameters:
        y: ``shape=(num_phase, N)``

            A vector of molar phase fractions, assuming the first row belongs to the
            reference phase.
        rho: ``shape=(num_phase, N)``

            A vector of phase densities, corresponding to ``y``. Must be of same shape
            as ``y``.
        eps: ``default=1e-8``

            A small number to determin saturated phases (``y_j > 1- eps``).

    Raises:
        ValueError: If ``y`` and ``rho`` are of unequal shape.
        ValueError: If more than one phase is saturated.
        AssertionError: If the computed saturations violate the unity constraint.

    Returns:
        A vector of analogous shape (and phase order) containing saturation values.

    """
    if y.shape != rho.shape:
        raise ValueError("Arguments 1 and 2 must be of same shape.")

    saturated = y > 1 - eps
    multi_saturated = saturated.sum(axis=0)
    if np.any(multi_saturated > 1):
        raise ValueError("More than 1 phase saturated in terms of y.")

    if len(y.shape) > 1:
        # NOTE transpose to parallelize over values, not phases
        y_T = y.T
        s = np.empty_like(y_T)
        _compute_saturations_gu(y_T, rho.T, eps, s)
        s = s.T
    else:
        s = _compute_saturations(y, rho, eps)

    # checking feasibility of results, should never assert though
    saturated = s > 1 - eps
    multi_saturated = saturated.sum(axis=0)
    assert not np.any(multi_saturated > 1), "More than 1 phase saturated in terms of s."

    return s


class CompositionalModellingError(Exception):
    """Custom exception class to alert the user when the compositional framework is
    inconsistently used.

    Such usage includes for example:

    - passing phases without components into a fluid mixture,
    - creating EoS or mixtures without any components at all,
    - requesting a time-dependent boundary equilibrium problem, without providing
      equilibrium equations,
    - violations of unified assumptions in equilibrium problems.

    """
