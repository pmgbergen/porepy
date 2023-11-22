"""Module containing implementation of the unified flash using (parallel) compiled
functions created with numba.

The flash system, including a non-parametric interior point method, is assembled and
compiled using :func:`numba.njit`, to enable an efficient solution of the equilibrium
problem.

The compiled functions are such that they can be used to solve multiple flash problems
in parallel.

Parallelization is achieved by applying Newton in parallel for multiple input.
The intended use is for larg compositional flow problems, where an efficient solution
to the local equilibrium problem is required.

References:
    [1]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [2]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_


"""
from __future__ import annotations

import abc
import logging
import time
from typing import Callable, Literal, Optional, Sequence

import numba
import numpy as np

from ._core import NUMBA_CACHE
from .composite_utils import safe_sum
from .flash import del_log, logger
from .mixture import BasicMixture, ThermodynamicState

__all__ = [
    "parse_xyz",
    "parse_pT",
    "normalize_fractions",
    "EoSCompiler",
    "Flash_c",
]


# region Helper methods


@numba.njit(
    "Tuple((float64[:,:], float64[:], float64[:]))(float64[:], UniTuple(int32, 2))",
    fastmath=True,
    cache=NUMBA_CACHE,
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

    the ``num_phase - 1`` entries before the phase compositions, are always
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
    "float64[:](float64[:],float64[:,:],float64[:],UniTuple(int32, 2))",
    fastmath=True,
    cache=NUMBA_CACHE,
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


@numba.njit(
    "float64[:](float64[:], UniTuple(int32, 2))",
    fastmath=True,
    cache=NUMBA_CACHE,
)
def parse_pT(X_gen: np.ndarray, npnc: tuple[int, int]) -> np.ndarray:
    """Helper function extracing pressure and temperature from a generic
    argument.

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


@numba.njit("float64[:,:](float64[:,:])", fastmath=True, cache=NUMBA_CACHE)
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
    "float64[:](float64[:], float64[:])",
    fastmath=True,
    cache=NUMBA_CACHE,
)
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
    df_dx = df_dxn.copy()  # deep copy to avoid messing with values
    ncomp = x.shape[0]
    # constructing the derivatives of xn_ij = x_ij / (sum_k x_kj)
    x_sum = np.sum(x)
    dxn = np.eye(ncomp) / x_sum - np.outer(x, np.ones(ncomp)) / (x_sum**2)
    # dxn = np.eye(ncomp) / x_sum - np.column_stack([x] * ncomp) / (x_sum ** 2)
    # assuming derivatives w.r.t. normalized fractions are in the last num_comp elements
    df_dx[-ncomp:] = df_dx[-ncomp:].dot(dxn)

    return df_dx


@numba.njit("float64[:](float64[:], float64[:,:])", fastmath=True, cache=NUMBA_CACHE)
def _rr_poles(y: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Parameters:
        y: Phase fractions, assuming the first one belongs to the reference phase.
        K: Matrix of K-values per independent phase (row) per component (column)

    Returns:
        A vector of length ``num_comp`` containing the denominators in the RR-equation
        related to K-values per component.
        Each demoninator is given by :math:`1 + \\sum_{j\\neq r} y_j (K_{ji} - 1)`.

    """
    return 1 + (K.T - 1) @ y[1:]
    # return 1 + np.dot(y[1:], K[:, i] - 1)


@numba.njit(
    "float64(float64[:], float64[:])",
    fastmath=True,
    cache=NUMBA_CACHE,
)
def _rr_binary_vle_inversion(z: np.ndarray, K: np.ndarray) -> float:
    """Inverts the Rachford-Rice equation for the binary 2-phase case.

    Parameters:
        z: ``shape=(num_comp,)``

            Vector of feed fractions.
        K: ``shape=(num_comp,)``

            Matrix of K-values per per component between vapor and liquid phase.

    Returns:
        The corresponding value of the vapor fraction.

    """
    ncomp = z.shape[0]
    n = np.sum((1 - K) * z)
    d = np.empty(ncomp)
    for i in range(ncomp):
        d[i] = (K[i] - 1) * np.sum(np.delete(K, i) - 1) * z[i]

    return n / np.sum(d)


@numba.njit(
    "float64(float64[:], float64[:], float64[:,:])",
    cache=NUMBA_CACHE,
)
def _rr_potential(z: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    """Calculates the potential according to [1] for the j-th Rachford-Rice equation.

    With :math:`n_c` components, :math:`n_p` phases and :math:`R` the reference phase,
    the potential is given by

    .. math::

        F = \\sum\\limits_{i} -(z_i ln(1 - (\\sum\\limits_{j\\neq R}(1 - K_{ij})y_j)))

    References:
        [1] `Okuno and Sepehrnoori (2010) <https://doi.org/10.2118/117752-PA>`_

    Parameters:
        z: ``len=n_c``

            Vector of feed fractions.
        y: ``len=n_p``

            Vector of molar phase fractions.
        K: ``shape=(n_p, n_c)``

            Matrix of K-values per independent phase (row) per component (column).

    Returns:
        The value of the potential based on above formula.

    """
    return np.sum(-z * np.log(np.abs(_rr_poles(y, K))))
    # F = [-np.log(np.abs(_rr_pole(i, y, K))) * z[i] for i in range(len(z))]
    # return np.sum(F)


# endregion
# region General flash equation independent of flash type and EoS


@numba.njit(
    "float64[:](float64[:,:], float64[:], float64[:])",
    fastmath=True,
    cache=NUMBA_CACHE,
)
def mass_conservation_res(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the mass conservation equations.

    For each component ``i``, except reference component, it holds

    ... math::

        z\left[i\right] - \sum_j y\left[j\right] x\left[j, i\right] = 0

    Number of phases and components is determined from the chape of ``x``.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:], float64[:]) -> float64[:]``.

    Note:
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
    # excluding mass consercation for 1st component
    return (z - np.dot(y, x))[1:]


@numba.njit(
    "float64[:,:](float64[:,:], float64[:])",
    fastmath=True,
    cache=NUMBA_CACHE,
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
    # and above is dz(x,y) / dyx
    return (-1) * jac


@numba.njit("float64[:](float64[:,:], float64[:])", fastmath=True, cache=NUMBA_CACHE)
def complementary_conditions_res(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the complementary conditions.

    For each phase ``j`` it holds

    ... math::

        y\left[j\right] \cdot \left(1 - \sum_i x\left[j, i\right]\right) = 0

    Number of phases and components is determined from the chape of ``x``.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:]``.

    Note:
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
    return y * (1 - np.sum(x, axis=1))


@numba.njit("float64[:,:](float64[:,:], float64[:])", fastmath=True, cache=NUMBA_CACHE)
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


# endregion
# region NPIPM related functions


@numba.njit(
    "float64(float64[:], float64[:], float64, float64, float64, float64)",
    fastmath=True,
    cache=NUMBA_CACHE,
)
def slack_equation_res(
    v: np.ndarray, w: np.ndarray, nu: float, u1: float, u2: float, eta: float
) -> float:
    r"""Implementation of the residual of the slack equation for the non-parametric
    interior point method.

    .. math::

        \frac{1}{2}\left( \lVert v^{-}\rVert^2 + \lVert w^{-}\rVert^2 +
        \frac{u}{n_p}\left(\langle v, w \rangle^{+}\right)^2 \right) +
        \eta\nu + \nu^2 = 0

    Parameters:
        v: ``shape=(num_phase,)``

            Vector containing phase fractions.
        w: ``shape=(num_phase,)``

            Vector containing the unity of phase compositions per phase.
        nu: Value of slack variable.
        u1: Parameter to tune the penalty for violation of complementarity.
        u2: Parameter to tune the penalty for violation of negativity.
        eta: Parameter for steepness of decline of slack variable.

    Returns:
        The evaluation of above equation.

    """

    nphase = v.shape[0]
    # dot = np.dot(v, w)  # numba performance warning
    dot = np.sum(v * w)

    # copy because of modifications for negative and positive
    v = v.copy()
    w = w.copy()
    v[v > 0.0] = 0.0
    w[w > 0.0] = 0.0

    # penalization of negativity
    res = 0.5 * u2 * (np.sum(v**2) + np.sum(w**2))

    # penalization of violation of complementarity
    dot = 0.0 if dot < 0.0 else dot
    res += 0.5 * dot**2 * u1 / nphase

    # decline of slack variable
    res += eta * nu + nu**2

    return res


@numba.njit(
    "float64[:](float64[:], float64[:], float64, float64, float64, float64)",
    fastmath=True,
    cache=NUMBA_CACHE,
)
def slack_equation_jac(
    v: np.ndarray, w: np.ndarray, nu: float, u1: float, u2: float, eta: float
) -> float:
    """Implementation of the gradient of the slack equation for the non-parametric
    interior point method (see :func:`slack_equation_res`).

    Parameters:
        v: ``shape=(num_phase,)``

            Vector containing phase fractions.
        w: ``shape=(num_phase,)``

            Vector containing the unity of phase compositions per phase.
        nu: Value of slack variable.
        u1: Parameter to tune the penalty for violation of complementarity.
        u2: Parameter to tune the penalty for violation of negativity.
        eta: Parameter for steepness of decline of slack variable.

    Returns:
        The gradient of the slcak equation with derivatives w.r.t. all elements in
        ``v``, ``w`` and ``nu``, with ``shape=(2 * num_phase + 1,)``.

    """

    nphase = v.shape[0]

    jac = np.zeros(2 * nphase + 1, dtype=np.float64)

    dot = np.sum(v * w)

    # derivatives of pos() and neg()
    dirac_dot = 1.0 if dot > 0.0 else 0.0  # dirac for positivity of dotproduct
    dirac_v = (v < 0.0).astype(np.float64)  # dirac for negativity in v, elementwise
    dirac_w = (w < 0.0).astype(np.float64)  # same for w

    d_dot_outer = 2 * u1 / nphase**2 * dot * dirac_dot

    # derivatives w.r.t. to elements in v
    jac[:nphase] = u2 * dirac_v * v + d_dot_outer * w
    jac[nphase : 2 * nphase] = u2 * dirac_w * w + d_dot_outer * v

    # derivative w.r.t. nu
    jac[-1] = eta + 2 * nu

    return jac


@numba.njit(
    "Tuple((float64[:,:], float64[:]))"
    + "(float64[:], float64[:,:], float64[:], float64, UniTuple(int32, 2))",
    fastmath=True,
    cache=NUMBA_CACHE,
)
def npipm_regularization(
    X: np.ndarray, A: np.ndarray, b: np.ndarray, u1: float, npnc: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Regularization of a linearized flash system assembled in the unified setting
    using the non-parametric interior point method.

    Parameters:
        X: Generic argument for the flash.
        A: Linearized flash system (including NPIPM).
        b: Residual corresponding to ``A``.
        u1: see :func:`slack_equation_res`
        npnc: ``len=2``

            2-tuple containing the number of phases and number of components

    Returns:
        A regularization according to Vu et al. 2021.

    """
    nphase, ncomp = npnc
    x, y, _ = parse_xyz(X[:-1], npnc)

    # summation of complementarity conditions
    reg = np.sum(y * (1 - np.sum(x, axis=1)))
    # positive part with penalty factor
    reg = 0.0 if reg < 0 else reg
    reg *= u1 / ncomp**2

    # subtract all relaxed complementary conditions multiplied with reg from the slack equation
    b[-1] = b[-1] - reg * np.sum(b[-(nphase + 1) : -1])
    # do the same for respective rows in the Jacobian
    A[-1] = A[-1] - reg * np.sum(A[-(nphase + 1) : -1], axis=0)

    return A, b


@numba.njit(
    "float64[:,:](float64[:,:], UniTuple(int32, 2))",
    fastmath=True,
    cache=NUMBA_CACHE,
)
def initialize_npipm_nu(X_gen: np.ndarray, npnc: tuple[int, int]) -> np.ndarray:
    """Computes an initial guess for the slack variable :math:`\\nu` in the NPIPM.

    Parameters:
        X: Generic argument for the flash.
        npnc: ``len=2``

            2-tuple containing the number of phases and number of components

    Returns:
        ``X_gen`` with the last column containing initial values of :math:`\\nu` based
        on the fractional values found in ``X_gen``.

    """
    nphase, ncomp = npnc
    nu = np.zeros(X_gen.shape[0])

    # contribution from dependent phase
    nu = (
        1 - np.sum(X_gen[:, -(ncomp * nphase + nphase) : -(ncomp * nphase + 1)], axis=1)
    ) * (
        1
        - np.sum(X_gen[:, -(ncomp * nphase + 1) : -(ncomp * (nphase - 1) + 1)], axis=1)
    )

    # contribution from independent phases
    for j in range(nphase - 1):
        y = X_gen[:, -(ncomp * nphase + nphase) + j]
        x_j = X_gen[
            :,
            -(ncomp * (nphase - 1) + 1) : -(ncomp * (nphase - 1) + 1) + (j + 1) * ncomp,
        ]
        nu += y * (1 - np.sum(x_j, axis=1))

    X_gen[:, -1] = nu / nphase
    return X_gen


# endregion
# region Methods related to the numerical solution strategy


@numba.njit("float64(float64[:])", fastmath=True, cache=NUMBA_CACHE)
def l2_potential(vec: np.ndarray) -> float:
    return np.sum(vec * vec) / 2.0


@numba.njit(  # TODO type signature once typing for functions is available
    fastmath=True,
    cache=NUMBA_CACHE,
)
def Armijo_line_search(
    Xk: np.ndarray,
    dXk: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    rho_0: float,
    kappa: float,
    j_max: int,
) -> float:
    r"""Armijo line search to be used inside Newton.

    Uses the L2-potential to find a minimizing step size.

    Parameters:
        Xk: Current Newton iterate.
        dXk: New update obtained from Newton.
        F: Callable to evaluate the residual.
            Must be compatible with ``Xk + dXk`` as argument.
        rho_0: ``(0, 1)``

            First step-size.
        kappa: ``(0, 0.5)``

            Slope for the line search.
        j_max: Maximal number of line search iterations.


    Returns:
        A step size :math:`\rho` minimizing
        :math:`\frac{1}{2}\lVert F(X_k + \rho dX_k)\rVert^2`.

    """

    # evaluation of the potential as is
    fk = F(Xk)
    potk = l2_potential(fk)
    rho_j = rho_0
    # kappa = 0.4
    # j_max = 150

    for j in range(1, j_max + 1):
        rho_j = rho_j**j

        try:
            fk_j = F(Xk + rho_j * dXk)
        except:
            continue

        potk_j = l2_potential(fk_j)

        if potk_j <= (1 - 2 * kappa * rho_j) * potk:
            return rho_j

    # return max
    return rho_j


@numba.njit(  # TODO same as for Armijo signature
    cache=NUMBA_CACHE,
)
def newton(
    X_0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    tol: float,
    max_iter: int,
    npnc: tuple[int, int],
    u1: float,
    rho_0: float,
    kappa: float,
    j_max: int,
) -> tuple[np.ndarray, int, int]:
    """Compiled Newton with Armijo line search and NPIPM regularization.

    Intended use is for the unified flash problem.

    Parameters:
        X_0: Initial guess.
        F: Callable representing the residual. Must be callable with ``X0``.
        DF: Callable representing the Jacobian. Must be callable with ``X0``.
        tol: Residual tolerance as stopping criterion.
        max_iter: Maximal number of iterations.
        npnc: ``len=2``

            2-tuple containing the number of phases and number of flashes in the flash
            problem.
        u1: See :func:`slack_equation_res`. Required for regularization.
        rho_0: See :func:`Armijo_line_search`.
        kappa: See :func:`Armijo_line_search`.
        j_max: See :func:`Armijo_line_search`.

    Returns:
        A 3-tuple containing

        1. the result of the Newton algorithm,
        2. a success flag
            - 0: success
            - 1: max iter reached
            - 2: failure in the evaluation of the residual
            - 3: failure in the evaluation of the Jacobian
            - 4: NAN or infty detected in update (aborted)
        3. final number of perfored iterations

        If the success flag indicates failure, the last iterate state of the unknown
        is returned.

    """
    # default return values
    num_iter = 0
    success = 1

    X = X_0.copy()
    DX = np.zeros_like(X_0)

    try:
        f_i = F(X)
    except:
        return X, 2, num_iter

    if np.linalg.norm(f_i) <= tol:
        success = 0  # root already found
    else:
        for _ in range(max_iter):
            num_iter += 1

            try:
                df_i = DF(X)
            except:
                success = 3
                break

            A, b = npipm_regularization(X, df_i, -f_i, u1, npnc)

            dx = np.linalg.solve(A, b)

            if np.any(np.isnan(dx)) or np.any(np.isinf(dx)):
                success = 4
                break

            # X contains also parameters (p, T, z_i, ...)
            # exactly ncomp - 1 feed fractions and 2 state definitions (p-T, p-h, ...)
            # for broadcasting insert solution into new vector
            DX[npnc[1] - 1 + 2 :] = dx

            s = Armijo_line_search(X, DX, F, rho_0, kappa, j_max)

            X = X + s * DX

            try:
                f_i = F(X)
            except:
                success = 2
                break

            if np.linalg.norm(f_i) <= tol:
                success = 0
                break

    return X, success, num_iter


@numba.njit(
    parallel=True,
    cache=NUMBA_CACHE,
)
def parallel_newton(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    F_dim: int,
    tol: float,
    max_iter: int,
    npnc: tuple[int, int],
    u1: float,
    rho_0: float,
    kappa: float,
    j_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel Newton, assuming each row in ``X0`` is a starting point to find a root
    of ``F``.

    Numba-parallelized loop over all rows in ``X0``, calling
    :func:`newton` for each row.

    For an explanation of all parameters, see :func:`newton`.

    Note:
        ``X0`` can contain parameters for the evaluation of ``F``.
        Therefore the dimension of the image of ``F`` must be defined by passing
        ``F_dim``.
        I.e., ``len(F(X0[i])) == F_dim`` and ``DF(X0[i]).shape == (F_dim, F_dim)``.

    Returns:
        The same as :func:`newton` in vectorized form, containing the results per
        row in ``X0``.

        Note however, that the returned results contain only the actual results,
        not the whole, generic flash argument given in ``X0``.
        More precisely, the first ``num_comp - 1 + 2`` elements per row are assumed to
        contain flash specifications in terms of feed fractions and thermodynamic state.
        Hence they are not duplicated and returned to safe memory.

    """

    N = X0.shape[0]

    result = np.empty((N, F_dim))
    num_iter = np.empty(N, dtype=np.int32)
    converged = np.empty(N, dtype=np.int32)

    for n in numba.prange(N):
        res_i, conv_i, n_i = newton(
            X0[n], F, DF, tol, max_iter, npnc, u1, rho_0, kappa, j_max
        )
        converged[n] = conv_i
        num_iter[n] = n_i
        result[n] = res_i[-F_dim:]

    return result, converged, num_iter


@numba.njit(
    cache=NUMBA_CACHE,
)
def linear_newton(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    F_dim: int,
    tol: float,
    max_iter: int,
    npnc: tuple[int, int],
    u1: float,
    rho_0: float,
    kappa: float,
    j_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Does the same as :func:`parallel_newton`, only the loop over the rows of ``X0``
    is not parallelized, but executed in a classical loop.

    Intended use is for smaller amount of flash problems, where the parallelization
    would produce a certain overhead in the initialization.

    """

    N = X0.shape[0]

    result = np.empty((N, F_dim))
    num_iter = np.empty(N, dtype=np.int32)
    converged = np.empty(N, dtype=np.int32)

    for n in range(N):
        res_i, conv_i, n_i = newton(
            X0[n], F, DF, tol, max_iter, npnc, u1, rho_0, kappa, j_max
        )
        converged[n] = conv_i
        num_iter[n] = n_i
        result[n] = res_i[-F_dim:]

    return result, converged, num_iter


# endregion


class EoSCompiler(abc.ABC):
    """Abstract base class for EoS specific compilation using numba.

    The :class:`FlashCompiler` needs functions computing

    - fugacity coefficients
    - enthalpies
    - densities
    - the derivatives of above w.r.t. pressure, temperature and phase compositions

    Respective functions must be assembled and compiled by a child class with a specific
    EoS.

    The compiled functions are expected to a specific signature (see below).

    ``(prearg: np.ndarray, phase_index: int, p: float, T: float, xn: numpy.ndarray)``

    1. ``prearg``: A 2-dimensional array containing the results of the pre-computation
       using the function returned by :meth:`get_pre_arg_computation`.
    2. ``phase_index``, the index of the phase, for which the quantities should be
       computed (``j = 0 ... num_phase - 1``), assuming 0 is the reference phase
    3. ``p``: The pressure value.
    4. ``T``: The temperature value.
    5 ``xn``: An array with ``shape=(num_comp,)`` containing the normalized fractions
      per component in phase ``phase_index``.

    The purpose of the ``prearg`` is efficiency.
    Many EoS have computions of some coterms or compressibility factors f.e.,
    which must only be computed once for all remaining thermodynamic quantities.

    The function for the ``prearg`` computation must have the signature:

    ``(p: float, T: float, XN: np.ndarray)``

    where ``XN`` contains **all** normalized compositions,
    stored row-wose per phase as a matrix.

    There are two ``prearg`` computations: One for the residual, one for the Jacobian
    of the flash system.

    The ``prearg`` for the Jacobian will be fed to the functions representing
    derivatives of thermodynamic quantities
    (e.g. derivative fugacity coefficients w.r.t. p, T, X),
    **additionally** to the ``prearc`` for residuals.

    I.e., the signature of functions representing derivatives is expected to be

    ``(prearg_res: np.ndarray, prearg_jac: np.ndarray,
    phase_index: int, p: float, T: float, xn: numpy.ndarray)``,

    whereas the signature of functions representing values only is expected to be

    ``(prearg: np.ndarray, phase_index: int, p: float, T: float, xn: numpy.ndarray)``

    """

    # TODO what is more efficient, just one pre-arg having everything?
    # Or splitting for computations for residuals, since it does not need derivatives?
    # 1. Armijo line search evaluated often, need only residual
    # 2. On the other hand, residual pre-arg is evaluated twice, for residual and jac
    @abc.abstractmethod
    def get_pre_arg_computation_res(
        self,
    ) -> Callable[[float, float, np.ndarray], np.ndarray]:
        """Abstract function for obtaining the compiled computation of the pre-argument
        for the residual.

        Returns:
            A NJIT-ed function taking values for pressure, temperature and normalized
            phase compositions for all phases as a matrix, and returning any matrix,
            which will be passed to other functions computing thermodynamic quantities.

        """
        pass

    @abc.abstractmethod
    def get_pre_arg_computation_jac(
        self,
    ) -> Callable[[float, float, np.ndarray], np.ndarray]:
        """Abstract function for obtaining the compiled computation of the pre-argument
        for the Jacobian.

        Returns:
            A NJIT-ed function taking values for pressure, temperature and normalized
            phase compositions for all phases as a matrix, and returning any matrix,
            which will be passed to other functions computing derivatives of
            thermodynamic quantities.

        """
        pass

    @abc.abstractmethod
    def get_fugacity_computation(
        self,
    ) -> Callable[[np.ndarray, int, float, float, np.ndarray], np.ndarray]:
        """Abstract assembler for compiled computations of the fugacity coefficients.

        Returns:
            A NJIT-ed function taking

            - the pre-argument for the residual,
            - the phase index,
            - pressure value,
            - temperature value,
            - an array of normalized fractions of componentn in phase ``phase_index``,

            and returning an array of fugacity coefficients with ``shape=(num_comp,)``.

        """
        pass

    @abc.abstractmethod
    def get_dpTX_fugacity_computation(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, int, float, float, np.ndarray], np.ndarray]:
        """Abstract assembler for compiled computations of the derivative of fugacity
        coefficients.

        The functions should return the derivative fugacities for each component
        row-wise in a matrix.
        It must contain the derivatives w.r.t. pressure, temperature and each fraction
        in a specified phase.
        I.e. the return value must be an array with ``shape=(num_comp, 2 + num_comp)``.

        Returns:
            A NJIT-ed function taking

            - the pre-argument for the residual,
            - the pre-argument for the Jacobian,
            - the phase index,
            - pressure value,
            - temperature value,
            - an array of normalized fractions of componentn in phase ``phase_index``,

            and returning an array of derivatives offugacity coefficients with
            ``shape=(num_comp, 2 + num_comp)``., where the columns indicate the
            derivatives w.r.t. to pressure, temperature and fractions.

        """
        pass


class Flash_c:
    """A class providing efficient unified flash calculations using numba-compiled
    functions.

    It uses the no-python mode of numba to produce highly efficient, compiled code.

    Flash equations are represented by callable residuals and Jacobians. Various
    flash types are assembled in a modular way by combining required, compiled equations
    into a solvable system.

    Since each system depends on the modelled phases and components, significant
    parts of the equilibrium problem must be compiled on the fly.

    This is a one-time action once the modelling process is completed.

    The supported flash types are than available until destruction.

    Supported flash types/specifications:

    1. ``'p-T'``: state definition in terms of pressure and temperature
    2. ``'p-h'``: state definition in terms of pressure and specific mixture enthalpy
    3. ``'v-h'``: state definition in terms of specific volume and enthalpy of the
       mixture

    Multiple flash problems can be solved in parallel by passing vectorized state
    definitions.

    The NPIPM approach is parametrizable. Each re-confugration requires a re-compilation
    since the system of equations must be presented as a vector-valued function
    taking a single vector (thermodynmaic input state).

    Parameters:
        mixture: A mixture model containing modelled components and phases.
        eos_compiler: An EoS compiler instance required to create a
            :class:`~porepy.composite.flash_compiler.FlashCompiler`.

    Raises:
        AssertionError: If not at least 2 components are present.
        AssertionError: If not 2 phases are modelled.

    """

    def __init__(
        self,
        mixture: BasicMixture,
        eos_compiler: EoSCompiler,
    ) -> None:
        nc = mixture.num_components
        np = mixture.num_phases

        assert np == 2, "Supports only 2-phase mixtures."
        assert nc >= 2, "Must have at least two components."

        # data used in initializers
        self._pcrits: list[float] = [comp.p_crit for comp in mixture.components]
        self._Tcrits: list[float] = [comp.T_crit for comp in mixture.components]
        self._vcrits: list[float] = [comp.V_crit for comp in mixture.components]
        self._omegas: list[float] = [comp.omega for comp in mixture.components]

        self.npnc: tuple[int, int] = (np, nc)
        """Number of phases and components, passed at instantiation."""

        self.eos_compiler: EoSCompiler = eos_compiler
        """Assembler and compiler of EoS-related expressions equation.
        passed at instantiation."""

        self.residuals: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the respective residuals as a callable."""

        self.jacobians: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the respective Jacobian as a callable."""

        self.initializers: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the initialization procedure."""

        self.npipm_res: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the residual of extended flash system.

        The extended flash system included the NPIPM slack equation, and hence an
        additional dependency on the slack variable :math:`\\nu`.

        The evaluation calls internally respective functions from :attr:`residuals`.
        The resulting array has one element more.

        """

        self.npipm_jac: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the Jacobian of extended flash system.

        The extended flash system included the NPIPM slack equation, and hence an
        additional dependency on the slack variable :math:`\\nu`.

        The evaluation calls internally respective functions from :attr:`jacobians`.
        The resulting matrix hase one row and one column more.

        """

        self.npipm_parameters: dict[str, float] = {
            "eta": 0.5,
            "u1": 1.0,
            "u2": 1.0,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the NPIPM:

        - ``'eta': 0.5`` linear decline in slack variable
        - ``'u1': 1.`` penalty for violating complementarity
        - ``'u2': 1.`` penalty for violating negativitiy of fractions

        Values can be set directly by modifying the values of this dictionary.

        """

        self.armijo_parameters: dict[str, float] = {
            "kappa": 0.4,
            "rho": 0.99,
            "j_max": 150,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the Armijo line-search:

        - ``'kappa': 0.4``
        - ``'rho_0': 0.99``
        - ``'j_max': 150`` (maximal number of Armijo iterations)

        Values can be set directly by modifying the values of this dictionary.

        """

        self.initialization_parameters: dict[str, float | int] = {
            "N1": 3,
            "N2": 2,
            "N3": 5,
        }
        """Numbers of iterations for initialization procedures.

        - ``'N1'``: 3. Iterations for fractions guess.
        - ``'N2'``: 2. Iterations for state constraint (p/T update).
        - ``'N3'``: 5. Alterations between fractions guess and  p/T update.

        """

        self.tolerance: float = 1e-7
        """Convergence criterion for the flash algorithm. Defaults to ``1e-7``."""

        self.max_iter: int = 100
        """Maximal number of iterations for the flash algorithms. Defaults to 100."""

    def compile(self, verbosity: int = 1) -> None:
        """Triggers the assembly and compilation of equilibrium equations, including
        the NPIPM approach.

        Important:
            This takes a considerable amount of time.
            The compilation is therefore separated from the instantiation of this class.

        Parameters:
            verbosity: ``default=1``

                Enable progress logs. Set to zero to disable.

        """
        # setting logging verbosity
        if verbosity == 1:
            logger.setLevel(logging.INFO)
        elif verbosity >= 2:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        nphase, ncomp = self.npnc

        ## dimension of flash systems, excluding NPIPM
        # number of equations for the pT system
        # ncomp - 1 mass constraints
        # (nphase - 1) * ncomp fugacity constraints (w.r.t. ref phase formulated)
        # nphase complementary conditions
        pT_dim = ncomp - 1 + (nphase - 1) * ncomp + nphase
        # p-h flash: additional var T, additional equ enthalpy constraint
        ph_dim = pT_dim + 1
        # v-h flash: additional vars p, s_j j!= ref
        # additional equations volume constraint and density constraints
        vh_dim = ph_dim + 1 + (nphase - 1)

        ## Compilation start
        logger.info(f"{del_log}Compiling residual pre-argument ..")
        prearg_res_c = self.eos_compiler.get_pre_arg_computation_res()
        logger.info(f"{del_log}Compiling Jacobian pre-argument ..")
        prearg_jac_c = self.eos_compiler.get_pre_arg_computation_jac()
        logger.info(f"{del_log}Compiling fugacity coefficient function ..")
        phi_c = self.eos_compiler.get_fugacity_computation()
        logger.info(f"{del_log}Compiling derivatives of fugacity coefficients ..")
        d_phi_c = self.eos_compiler.get_dpTX_fugacity_computation()

        logger.info(f"{del_log}Compiling residual of isogucacity constraints ..")

        @numba.njit(
            "float64[:](float64[:,:], float64, float64, float64[:,:], float64[:,:])"
        )
        def isofug_constr_c(
            prearg: np.ndarray,
            p: float,
            T: float,
            X: np.ndarray,
            Xn: np.ndarray,
        ):
            """Helper function to assemble the isofugacity constraint.

            Formulation is always w.r.t. the reference phase r, assumed to be r=0.

            """
            isofug = np.empty(ncomp * (nphase - 1), dtype=np.float64)

            phi_r = phi_c(prearg, 0, p, T, Xn[0])

            for j in range(1, nphase):
                phi_j = phi_c(prearg, j, p, T, Xn[j])
                # isofugacity constraint between phase j and phase r
                # NOTE fugacities are evaluated with normalized fractions
                isofug[(j - 1) * ncomp : j * ncomp] = X[j] * phi_j - X[0] * phi_r

            return isofug

        logger.info(f"{del_log}Compiling Jacobian of isogucacity constraints ..")

        @numba.njit(
            "float64[:,:]"
            + "(float64[:,:], float64[:,:],"
            + "int32, float64, float64, float64[:], float64[:])",
        )
        def d_isofug_block_j(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            j: int,
            p: float,
            T: float,
            X: np.ndarray,
            Xn: np.ndarray,
        ):
            """Helper function to construct a block representing the derivative
            of x_ij * phi_ij for all i as a matrix, with i row index.
            This is constructed for a given phase j.
            """

            phi_j = phi_c(prearg_res, j, p, T, Xn)
            d_phi_j = d_phi_c(prearg_res, prearg_jac, j, p, T, Xn)
            # NOTE phi depends on normalized fractions
            # extending derivatives from normalized fractions to extended ones
            for i in range(ncomp):
                d_phi_j[i] = extended_compositional_derivatives(d_phi_j[i], X)

            # product rule: x * dphi
            d_xphi_j = (d_phi_j.T * X).T
            # + phi * dx  (minding the first two columns which contain the p-T derivs)
            d_xphi_j[:, 2:] += np.diag(phi_j)

            return d_xphi_j

        @numba.njit(
            "float64[:,:](float64[:,:], float64[:, :],"
            + "float64, float64, float64[:,:], float64[:,:])"
        )
        def d_isofug_constr_c(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            X: np.ndarray,
            Xn: np.ndarray,
        ):
            """Helper function to assemble the derivative of the isofugacity constraints

            Formulation is always w.r.t. the reference phase r, assumed to be zero 0.

            Important:
                The derivative is taken w.r.t. to A, B, Z (among others).
                An forward expansion must be done after a call to this function.

            """
            d_iso = np.zeros((ncomp * (nphase - 1), 2 + ncomp * nphase))

            # creating derivative parts involving the reference phase
            d_xphi_r = d_isofug_block_j(prearg_res, prearg_jac, 0, p, T, X[0], Xn[0])

            for j in range(1, nphase):
                # construct the same as above for other phases
                d_xphi_j = d_isofug_block_j(
                    prearg_res, prearg_jac, 1, p, T, X[j], Xn[j]
                )

                # p, T derivative
                d_iso[(j - 1) * ncomp : j * ncomp, :2] = (
                    d_xphi_j[:, :2] - d_xphi_r[:, :2]
                )
                # remember: d(x_ij * phi_ij - x_ir * phi_ir)
                # hence every row-block contains (-1)* d_xphi_r
                # derivative w.r.t. fractions in reference phase
                d_iso[(j - 1) * ncomp : j * ncomp, 2 : 2 + ncomp] = -d_xphi_r[:, 2:]
                # derivatives w.r.t. fractions in independent phase j
                d_iso[
                    (j - 1) * ncomp : j * ncomp, 2 + j * ncomp : 2 + (j + 1) * ncomp
                ] = d_xphi_j[:, 2:]

            return d_iso

        logger.info(f"{del_log}Compiling p-T flash equations ..")

        @numba.njit("float64[:](float64[:])")
        def F_pT(X_gen: np.ndarray) -> np.ndarray:
            x, y, z = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))

            # declare residual array of proper dimension
            res = np.empty(pT_dim, dtype=np.float64)

            res[: ncomp - 1] = mass_conservation_res(x, y, z)
            # last nphase equations are always complementary conditions
            res[-nphase:] = complementary_conditions_res(x, y)

            # EoS specific computations
            xn = normalize_fractions(x)
            prearg = prearg_res_c(p, T, xn)

            res[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1)] = isofug_constr_c(
                prearg, p, T, x, xn
            )

            return res

        @numba.njit("float64[:,:](float64[:])")
        def DF_pT(X_gen: np.ndarray) -> np.ndarray:
            x, y, _ = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))

            # declare Jacobian or proper dimension
            jac = np.zeros((pT_dim, pT_dim), dtype=np.float64)

            jac[: ncomp - 1] = mass_conservation_jac(x, y)
            # last nphase equations are always complementary conditions
            jac[-nphase:] = complementary_conditions_jac(x, y)

            # EoS specific computations
            xn = normalize_fractions(x)
            prearg_res = prearg_res_c(p, T, xn)
            prearg_jac = prearg_jac_c(p, T, xn)

            jac[
                ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), nphase - 1 :
            ] = d_isofug_constr_c(prearg_res, prearg_jac, p, T, x, xn)[:, 2:]

            return jac

        logger.info(f"{del_log}Storing compiled flash equations ..")
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

        p_crits = np.array(self._pcrits)
        T_crits = np.array(self._Tcrits)
        v_crits = np.array(self._vcrits)
        omegas = np.array(self._omegas)
        logger.info(f"{del_log}Compiling p-T flash initialization ..")

        @numba.njit("float64[:](float64[:],int32, int32)")
        def guess_fractions(
            X_gen: np.ndarray, N1: int, guess_K_vals: int
        ) -> np.ndarray:
            """Guessing fractions for a single flash configuration"""
            x, y, z = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))

            # pseudo-critical quantities
            T_pc = np.sum(z * T_crits)
            p_pc = np.sum(z * p_crits)

            # storage of K-values (first phase assumed reference phase)
            K = np.zeros((nphase - 1, ncomp))
            K_tol = 1e-10  # tolerance to bind K-values away from 0

            if guess_K_vals > 0:
                for j in range(nphase - 1):
                    K[j, :] = (
                        np.exp(5.37 * (1 + omegas) * (1 - T_crits / T)) * p_crits / p
                        + K_tol
                    )
            else:
                xn = normalize_fractions(x)
                prearg = prearg_res_c(p, T, xn)
                # fugacity coefficients reference phase
                phi_r = phi_c(prearg, 0, p, T, xn[0])
                for j in range(1, nphase):
                    phi_j = phi_c(prearg, j, p, T, xn[j])
                    K_jr = phi_r / phi_j + K_tol
                    K[j - 1, :] = K_jr

            # starting iterations using Rachford Rice
            for n in range(N1):
                # solving RR for molar phase fractions
                if nphase == 2:
                    # only one independent phase assumed
                    K_ = K[0]
                    if ncomp == 2:
                        y_ = _rr_binary_vle_inversion(z, K_)
                    else:
                        raise NotImplementedError(
                            "Multicomponent RR solution not implemented."
                        )

                    # copy the original value s.t. different corrections
                    # do not interfer with eachother
                    # _y = float(y_)
                    negative = y_ < 0.0
                    exceeds = y_ > 1.0
                    invalid = exceeds | negative

                    # correction of invalid gas phase values
                    if invalid:
                        # assuming gas saturated for correction using RR potential
                        y_test = np.array([0.0, 1.0], dtype=np.float64)
                        rr_pot = _rr_potential(z, y_test, K)
                        # checking if y is feasible
                        # for more information see Equation 10 in
                        # `Okuno et al. (2010) <https://doi.org/10.2118/117752-PA>`_
                        t_i = _rr_poles(y_test, K)
                        cond_1 = t_i - z >= 0.0
                        # tests holds for arbitrary number of phases
                        # reflected by implementation, despite nph == 2
                        cond_2 = K * z - t_i <= 0.0
                        gas_feasible = np.all(cond_1) & np.all(cond_2)

                        if rr_pot > 0.0:
                            y_ = 0.0
                        elif (rr_pot < 0.0) & gas_feasible:
                            y_ = 1.0

                        # clearly liquid
                        if (T < T_pc) & (p > p_pc):
                            y_ = 0.0
                        # clearly gas
                        elif (T > T_pc) & (p < p_pc):
                            y_ = 1.0

                        # Correction based on negative flash
                        # value of y_ must be between innermost poles
                        # K_min = np.min(K_)
                        # K_max = np.max(K_)
                        # y_1 = 1 / (1 - K_max)
                        # y_2 = 1 / (1 - K_min)
                        # if y_1 <= y_2:
                        #     y_feasible = y_1 < _y < y_2
                        # else:
                        #     y_feasible = y_2 < _y < y_1

                        # if y_feasible & negative:
                        #     y_ = 0.0
                        # elif y_feasible & exceeds:
                        #     y_ = 1.0

                        # If all K-values are smaller than 1 and gas fraction is negative,
                        # the liquid phase is clearly saturated
                        # Vice versa, if fraction above 1 and K-values greater than 1.
                        # the gas phase is clearly saturated
                        if negative & np.all(K_ < 1.0):
                            y_ = 0.0
                        elif exceeds & np.all(K_ > 1.0):
                            y_ = 1.0

                        # assert corrections did what they have to do
                        assert (
                            0.0 <= y_ <= 1.0
                        ), "y fraction estimate outside bound [0, 1]."
                    y[1] = y_
                    y[0] = 1.0 - y_
                else:
                    raise NotImplementedError(
                        "Fractions guess for more than 2 phases not implemented."
                    )

                # resolve compositions
                t = _rr_poles(y, K)
                x[0] = z / t  # fraction in reference phase
                x[1:] = K * x[0]  # fraction in indp. phases

                # update K-values if another iteration comes
                if n < N1 - 1:
                    xn = normalize_fractions(x)
                    prearg = prearg_res_c(p, T, xn)
                    # fugacity coefficients reference phase
                    phi_r = phi_c(prearg, 0, p, T, xn[0])
                    for j in range(1, nphase):
                        phi_j = phi_c(prearg, j, p, T, xn[j])
                        K_jr = phi_r / phi_j + K_tol
                        K[j - 1, :] = K_jr

            return insert_xy(X_gen, x, y, (nphase, ncomp))

        @numba.njit("float64[:,:](float64[:,:],int32, int32)", parallel=True)
        def pT_initializer(X_gen: np.ndarray, N1: int, guess_K_vals: int) -> np.ndarray:
            """p-T initializer as a parallelized loop over all flash configurations."""
            nf = X_gen.shape[0]
            for f in numba.prange(nf):
                # for f in range(nf):
                X_gen[f] = guess_fractions(X_gen[f], N1, guess_K_vals)
            return X_gen

        logger.info(f"{del_log}Compiling p-h flash initialization ..")

        # @numba.njit("float64[:,:](float64[:,:], int32, int32, int32)", parallel=True)
        def ph_initializer(X_gen: np.ndarray, N1: int, N2: int, N3: int) -> np.ndarray:
            """p-h initializer as a parallelized loop over all configurations"""
            nf = X_gen.shape[0]
            for f in numba.prange(nf):
                xf = X_gen[f]
                _, _, z = parse_xyz(xf, (nphase, ncomp))
                T_pc = np.sum(z * T_crits)  # pseudo-critical T approximation as start
                xf[-(ncomp * nphase + nphase - 1) - 1] = T_pc
                xf = guess_fractions(xf, N1, 1)

                for _ in range(N3):
                    # xf = ... # T update
                    xf = guess_fractions(xf, N1, 0)

                X_gen[f] = xf
            return X_gen

        logger.info(f"{del_log}Compiling h-v flash initialization ..")
        logger.info(f"{del_log}Storing compiled initializers ..")

        self.initializers.update(
            {
                "p-T": pT_initializer,
                "p-h": ph_initializer,
            }
        )
        logger.info(f"{del_log}Compilation completed.\n")

        self.reconfigure_npipm(
            self.npipm_parameters["u1"],
            self.npipm_parameters["u2"],
            self.npipm_parameters["eta"],
            verbosity,
        )

    def reconfigure_npipm(
        self, u1: float = 1.0, u2: float = 1.0, eta: float = 0.5, verbosity: int = 1
    ) -> None:
        """Re-compiles the NPIPM slack equation using updated parameters ``u, eta``.

        For more information on ``u`` and ``eta``, see :func:`slack_equation_res`.

        Default values ``u=1, eta=0.5`` are used in the first compilation
        while calling :meth:`compile`.

        Note:
            The user must recompile the system with ``u,eta`` being quasi-static in
            order to be able to call the system with a single argument
            (slack variable appended thermodynamic state) for the numerical
            methods to work.
            Passing ``u`` and ``eta`` as separate arguments is quite cumbersome
            with the current state of numba and the approach of having a single-argument
            callables which represents residuals and Jacobian.

        """
        # setting logging verbosity
        if verbosity == 1:
            logger.setLevel(logging.INFO)
        elif verbosity >= 2:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        npnc = self.npnc
        nphase, ncomp = npnc
        u1 = float(u1)
        u2 = float(u2)
        eta = float(eta)

        logger.info(f"{del_log}Compiling NPIPM p-T flash ..")

        F_pT = self.residuals["p-T"]
        DF_pT = self.jacobians["p-T"]

        @numba.njit("float64[:](float64[:])")
        def F_npipm_pT(X: np.ndarray) -> np.ndarray:
            X_thd = X[:-1]
            nu = X[-1]
            x, y, _ = parse_xyz(X_thd, npnc)

            f_flash = F_pT(X_thd)

            # couple complementary conditions with nu
            f_flash[-nphase:] -= nu

            # NPIPM equation
            unity_j = np.zeros(nphase)
            for j in range(nphase):
                unity_j[j] = 1.0 - np.sum(x[j])

            # complete vector of fractions
            slack = slack_equation_res(y, unity_j, nu, u1, u2, eta)

            # NPIPM system has one equation more at end
            f_npipm = np.zeros(f_flash.shape[0] + 1)
            f_npipm[:-1] = f_flash
            f_npipm[-1] = slack

            return f_npipm

        @numba.njit("float64[:,:](float64[:])")
        def DF_npipm_pT(X: np.ndarray) -> np.ndarray:
            X_thd = X[:-1]
            nu = X[-1]
            x, y, _ = parse_xyz(X_thd, npnc)

            df_flash = DF_pT(X_thd)

            # NPIPM matrix has one row and one column more
            df_npipm = np.zeros((df_flash.shape[0] + 1, df_flash.shape[1] + 1))
            df_npipm[:-1, :-1] = df_flash
            # relaxed complementary conditions read as y * (1 - sum x) - nu
            # add the -1 for the derivative w.r.t. nu
            df_npipm[-(nphase + 1) : -1, -1] = np.ones(nphase) * (-1)

            # derivative NPIPM equation
            unity_j = np.zeros(nphase)
            for j in range(nphase):
                unity_j[j] = 1.0 - np.sum(x[j])

            # complete vector of fractions
            d_slack = slack_equation_jac(y, unity_j, nu, u1, u2, eta)
            # d slack has derivatives w.r.t. y_j and w_j
            # d w_j must be expanded since w_j = 1 - sum x_j
            # d y_0 must be expanded since reference phase is eliminated by unity
            expand_yr = np.ones(nphase - 1) * (-1)
            expand_x_in_j = np.ones(ncomp) * (-1)

            # expansion of y_0 and cut of redundant value
            d_slack[1 : nphase - 1] += d_slack[0] * expand_yr
            d_slack = d_slack[1:]

            # expand it also to include possibly other derivatives
            d_slack_expanded = np.zeros(df_npipm.shape[1])
            # last derivative is w.r.t. nu
            d_slack_expanded[-1] = d_slack[-1]

            for j in range(nphase):
                # derivatives w.r.t. x_ij, +2 because nu must be skipped and j starts with 0
                vec = expand_x_in_j * d_slack[-(j + 2)]
                d_slack_expanded[-(1 + (j + 1) * ncomp) : -(1 + j * ncomp)] = vec

            # derivatives w.r.t y_j. j != r
            d_slack_expanded[
                -(1 + nphase * ncomp + nphase - 1) : -(1 + nphase * ncomp)
            ] = d_slack[: nphase - 1]

            df_npipm[-1] = d_slack_expanded

            return df_npipm

        logger.info(f"{del_log}Storing compiled flash equations ..")

        self.npipm_res.update(
            {
                "p-T": F_npipm_pT,
            }
        )

        self.npipm_jac.update(
            {
                "p-T": DF_npipm_pT,
            }
        )

        logger.info(f"{del_log}NPIPM compilation completed.\n")

    def flash(
        self,
        z: Sequence[np.ndarray],
        p: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        initial_state: Optional[ThermodynamicState] = None,
        mode: Literal["linear", "parallel"] = "linear",
        verbosity: int = 0,
    ) -> tuple[ThermodynamicState, np.ndarray, np.ndarray]:
        """Performes the flash for given feed fractions and state definition.

        Exactly 2 thermodynamic state must be defined in terms of ``p, T, h`` or ``v``
        for an equilibrium problem to be well-defined.

        One state must relate to pressure or volume.
        The other to temperature or energy.

        Supported combination:

        - p-T
        - p-h
        - v-h

        Parameters:
            z: ``len=num_comp - 1``

                A squence of feed fractions per component, except reference component.
            p: Pressure at equilibrium.
            T: Temperature at equilibrium.
            h: Specific enthalpy of the mixture at equilibrium,
            v: Specific volume of the mixture at equilibrium,
            initial_state: ``default=None``

                If not given, an initial guess is computed for the unknowns of the flash
                type.

                If given, it must have at least values for phase fractions and
                compositions.
                Molar phase fraction for reference phase **must not** be provided.

                It must have additionally values for temperature, for
                a state definition where temperature is not known at equilibrium.

                It must have additionally values for pressure and saturations, for
                state definitions where pressure is not known at equilibrium.
                Saturation for reference phase **must not** be provided.
            mode: ``default='linear'``

                Mode of solving the equilibrium problems for multiple state definitions
                given by arrays.

                - ``'linear'``: A classical loop over state defintions (row-wise).
                - ``'parallel'``: A parallelized loop, intended for larger amounts of
                  problems.

            verbosity: ``default=0``

                For logging information about progress. Note that as of now, there is
                no support for logs during solution procedures in the loop since
                compiled code is exectuded.

        Raises:
            ValueError: If an insufficient amount of feed fractions is passed or they
                violate the unity constraint.
            NotImplementedError: If an unsupported combination of insufficient number of
                of thermodynamic states is passed.

        Returns:
            A 3-tuple containing the results, success flags and number of iterations as
            returned by :func:`newton`.
            The results are stored in a thermodynamic state structure.

        """
        # setting logging verbosity
        if verbosity == 1:
            logger.setLevel(logging.INFO)
        elif verbosity >= 2:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        nphase, ncomp = self.npnc

        for i, z_ in enumerate(z):
            if np.any(z_ <= 0) or np.any(z_ >= 1):
                raise ValueError(
                    f"Violation of strict bound (0,1) for feed fraction {i} detected."
                )

        z_sum = safe_sum(z)
        if len(z) == ncomp - 1:
            if not np.all(z_sum < 1.0):
                raise ValueError(
                    f"{ncomp - 1} ({ncomp}) feed fractions violate unity constraint."
                )
        elif len(z) == ncomp:
            if not np.all(z_sum == 1.0):
                raise ValueError(
                    f"{ncomp} ({ncomp}) feed fractions violate unity constraint."
                )
            z = z[1:]
        else:
            raise ValueError(f"Expecting at least {ncomp - 1} feed fractions.")

        flash_type: Literal["p-T", "p-h", "v-h"]
        F_dim: int
        NF: int  # number of vectorized target states
        X0: np.ndarray  # vectorized, generic flash argument
        gen_arg_dim: int  # number of required values for a flash
        result_state = ThermodynamicState(z=[1 - z_sum] + list(z))
        init_args: tuple

        if p is not None and T is not None and (h is None and v is None):
            flash_type = "p-T"
            F_dim = nphase - 1 + nphase * ncomp + 1
            NF = (z_sum + p + T).shape[0]
            gen_arg_dim = ncomp - 1 + 2 + F_dim
            state_1 = p
            state_2 = T
            result_state.p = p
            result_state.T = T
            init_args = (self.initialization_parameters["N1"], 1)
        elif p is not None and h is not None and (T is None and v is None):
            flash_type = "p-h"
            F_dim = nphase - 1 + nphase * ncomp + 1 + 1
            NF = (z_sum + p + h).shape[0]
            gen_arg_dim = ncomp - 1 + 2 + 1 + F_dim
            state_1 = p
            state_2 = h
            result_state.p = p
            result_state.h = h
            init_args = (
                self.initialization_parameters["N1"],
                self.initialization_parameters["N2"],
                self.initialization_parameters["N3"],
            )
        elif v is not None and h is not None and (T is None and v is None):
            flash_type = "v-h"
            F_dim = nphase - 1 + nphase * ncomp + 2 + nphase - 1 + 1
            NF = (z_sum + p + h).shape[0]
            gen_arg_dim = ncomp - 1 + 2 + nphase - 1 + 2 + F_dim
            state_1 = v
            state_2 = h
            result_state.v = v
            result_state.h = h
            init_args = (
                self.initialization_parameters["N1"],
                self.initialization_parameters["N2"],
                self.initialization_parameters["N3"],
            )
        else:
            raise NotImplementedError(
                f"Unsupported flash with state definitions {p, T, h, v}"
            )

        logger.info(f"{del_log}Determined flash type: {flash_type}\n")

        logger.debug(f"{del_log}Assembling generic flash arguments ..")
        X0 = np.zeros((NF, gen_arg_dim))
        for i, z_i in enumerate(z):
            X0[:, i] = z_i
        X0[:, ncomp - 1] = state_1
        X0[:, ncomp] = state_2

        if initial_state is None:
            logger.info(f"{del_log}Computing initial state ..")
            start = time.time()
            # exclude NPIPM variable (last column) from initialization
            X0[:, :-1] = self.initializers[flash_type](X0[:, :-1], *init_args)
            end = time.time()
            logger.info(f"{del_log}Initial state computed.\n")
            t = end - start
            logger.debug(
                f"Elapsed time (min): {t / 60.}\n"
                if t > 60.0
                else f"Elapsed time (s): {t}\n"
            )
        else:
            logger.debug(f"{del_log}Parsing initial state ..")
            # parsing phase compositions and molar fractions
            for j in range(nphase):
                # values for molar phase fractions except for reference phase
                if j < nphase - 1:
                    X0[:, -(1 + nphase * ncomp + nphase - 1 + j)] = initial_state.y[j]
                # composition of phase j
                for i in range(ncomp):
                    X0[:, -(1 + (nphase - j) * ncomp + i)] = initial_state.X[j][i]

            # If T is unknown, get provided guess for T
            if "T" not in flash_type:
                X0[:, -(1 + ncomp * nphase + nphase - 1 + 1)] = initial_state.T
            # If p is unknown, get provided guess for p and saturations
            if "p" not in flash_type:
                X0[:, -(1 + ncomp * nphase + nphase - 1 + 2)] = initial_state.p
                for j in range(nphase - 1):
                    X0[
                        :, -(1 + ncomp * nphase + nphase - 1 + 2 + nphase - 1 + j)
                    ] = initial_state.s[j]

            # parsing molar phsae fractions

        logger.info(f"{del_log}Computing initial guess for slack variable ..")
        X0 = initialize_npipm_nu(X0, (nphase, ncomp))

        F = self.npipm_res[flash_type]
        DF = self.npipm_jac[flash_type]
        solver_args = (
            F_dim,
            self.tolerance,
            self.max_iter,
            self.npnc,
            self.npipm_parameters["u1"],
            self.armijo_parameters["rho"],
            self.armijo_parameters["kappa"],
            self.armijo_parameters["j_max"],
        )

        logger.info(f"{del_log}Solving ..")
        start = time.time()
        if mode == "linear":
            results, success, num_iter = linear_newton(X0, F, DF, *solver_args)
        elif mode == "parallel":
            results, success, num_iter = parallel_newton(X0, F, DF, *solver_args)
        else:
            raise ValueError(f"Unknown mode of compuation {mode}")
        end = time.time()
        logger.info(f"{del_log}Flash computations done.\n")
        t = end - start
        logger.debug(
            f"Elapsed time (min): {t / 60.}\n"
            if t > 60.0
            else f"Elapsed time (s): {t}\n"
        )

        logger.debug(f"{del_log}Parsing and returning results.\n")
        return (
            self._parse_and_complete_results(results, result_state, flash_type),
            success,
            num_iter,
        )

    def _parse_and_complete_results(
        self, results: np.ndarray, result_state: ThermodynamicState, flash_type: str
    ) -> ThermodynamicState:
        """Helper function to fill a result state with the results from the flash.

        Modifies and returns the passed result state structur containing flash
        specifications.

        Also, fills up secondary expressions for respective flash type.

        """
        nphase, ncomp = self.npnc

        # Parsing phase compositions and molar phsae fractions
        y = [0.0] * nphase
        X = [[0.0] * ncomp for _ in range(nphase)]
        for j in range(nphase):
            # values for molar phase fractions except for reference phase
            if j < nphase - 1:
                y[j + 1] = results[:, -(1 + nphase * ncomp + nphase - 1) + j]
            # composition of phase j
            for i in range(ncomp):
                X[j][i] = results[:, -(1 + (nphase - j) * ncomp) + i]
        # reference phase
        y[0] = 1 - sum(y)

        result_state.y = y
        result_state.X = X
        # If T is unknown, get provided guess for T
        if "T" not in flash_type:
            result_state.T = results[:, -(1 + ncomp * nphase + nphase - 1 + 1)]
        # If p is unknown, get provided guess for p and saturations
        if "p" not in flash_type:
            result_state.p = results[:, -(1 + ncomp * nphase + nphase - 1 + 2)]
            s = [0.0] * nphase
            for j in range(nphase - 1):
                s[j + 1] = results[
                    :, -(1 + ncomp * nphase + nphase - 1 + 2 + nphase - 1 + j)
                ]
            s[0] = 1 - sum(s)
            result_state.s = s

        # TODO fill up missing quantities in result state if any
        return result_state
