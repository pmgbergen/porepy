"""Module containing numba-compiled implementations of the NPIPM using the Newton
algorithm and Armijo line search.

To be used by the compiled flash."""

from __future__ import annotations

from typing import Callable, Literal

import numba
import numpy as np
from numba.core import types as nbtypes
from numba.typed import Dict as nbdict

from ._core import NUMBA_CACHE
from .utils_c import parse_xyz

SOLVER_PARAMS = dict[
    Literal[
        "f_dim",
        "num_phase",
        "num_comp",
        "tol",
        "max_iter",
        "rho",
        "kappa",
        "j_max",
        "u1",
        "u2",
        "eta",
    ],
    float,
]
"""Type alias describing a parameter dictionary containing parameters which are required
for the solution strategy in this module.

- ``'f_dim'``: Int. Dimension of the flash system, including the NPIPM slack equation.
- ``'num_phases'``: Int. Number of phases in fluid mixture.
- ``'num_comps'``: Int. Number of components in fluid mixture.
- ``'tol'``: Float. Tolerance for 2-norm of residual to be used as convergence criterion.
- ``'max_iter'``: Int. Maximal number of Newton iterations.
- ``'rho'``: ``(0, 1)``. First step size in Armijo line search.
- ``'kappa'``: ``(0, 0.5)``. Slope of line for line search.
- ``'j_max'``: Int. Maximal number of line search iterations.
- ``'u1'``: Float. Penalty parameter for violation of complementarity in NPIPM.
- ``'u2'``: Float. Penalty parameter for violation of non-negativity in NPIPM.
- ``'eta'``: Float. Linear decline of NPIPM slack variable.

"""


def convert_param_dict(params: SOLVER_PARAMS) -> nbdict:
    """Helper function to convert the parameter dictionary into a typed dict
    recognizable by numba.

    Key type: unicode type (string).
    Value type: float64.

    Raises:
        KeyError: If any of the expected parameters is missing.

    """

    out = nbdict.empty(key_type=nbtypes.unicode_type, value_type=nbtypes.float64)

    # NOTE Numba must have precise types, so integers are converted temporary to floats
    # The solver then converts what is expected to be an integer to int format.
    out["f_dim"] = float(params["f_dim"])
    out["num_phase"] = float(params["num_phase"])
    out["num_comp"] = float(params["num_comp"])
    out["tol"] = float(params["tol"])
    out["max_iter"] = float(params["max_iter"])
    out["rho"] = float(params["rho"])
    out["kappa"] = float(params["kappa"])
    out["j_max"] = float(params["j_max"])
    out["u1"] = float(params["u1"])
    out["u2"] = float(params["u2"])
    out["eta"] = float(params["eta"])

    return out


# region NPIPM related functions


@numba.njit(
    "float64(float64[:],float64[:],float64,float64,float64,float64)",
    fastmath=True,
    cache=True,
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
    "float64[:](float64[:],float64[:],float64,float64,float64,float64)",
    fastmath=True,
    cache=True,
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
    "float64[:,:](float64[:,:],UniTuple(int32, 2))",
    fastmath=True,
    cache=True,
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
        y_j = X_gen[:, -(ncomp * nphase + nphase) + j]
        x_j = X_gen[
            :,
            -(ncomp * (nphase - 1) + 1) : -(ncomp * (nphase - 1) + 1) + (j + 1) * ncomp,
        ]
        nu += y_j * (1 - np.sum(x_j, axis=1))

    X_gen[:, -1] = nu / nphase
    return X_gen


@numba.njit(
    "float64[:](float64[:],float64[:],UniTuple(int32, 2),float64,float64,float64)",
    fastmath=True,
    cache=NUMBA_CACHE,  # NOTE The cache is dependent on another function
)
def _npipm_extend_and_regularize_res(
    f_res: np.ndarray,
    X: np.ndarray,
    npnc: tuple[int, int],
    u1: float,
    u2: float,
    eta: float,
) -> np.ndarray:
    """Helper function to append the residual of the slack equation to an
    already computed flash residual.

    Important:
        This function assumes that the last ``num_phases`` entries correspond to the
        residual values of the complementarity conditions.

    """

    x, y, _ = parse_xyz(X[:-1], npnc)
    nu = X[-1]

    nphase = x.shape[0]

    # couple complementary conditions with nu
    f_res[-nphase:] -= nu

    # NPIPM equation
    unity_j = 1.0 - np.sum(x, axis=1)
    slack = slack_equation_res(y, unity_j, nu, u1, u2, eta)

    # NPIPM system has one equation more at end
    f_npipm = np.zeros(f_res.shape[0] + 1)
    f_npipm[:-1] = f_res
    f_npipm[-1] = slack

    # regularization
    # summation of complementarity conditions
    reg = np.sum(y * (1 - np.sum(x, axis=1)))
    # positive part with penalty factor
    reg = 0.0 if reg < 0 else reg
    reg *= u1 / nphase**2
    # subtract complementarity conditions multiplied with regularization factor from
    # slack equation residual
    f_npipm[-1] = f_npipm[-1] - reg * np.sum(f_npipm[-(nphase + 1) : -1])

    return f_npipm


@numba.njit(
    "float64[:,:]"
    + "(float64[:,:],float64[:],UniTuple(int32, 2),float64,float64,float64)",
    fastmath=True,
    cache=NUMBA_CACHE,  # NOTE The cache is dependent on another function
)
def _npipm_extend_and_regularize_jac(
    f_jac: np.ndarray,
    X: np.ndarray,
    npnc: tuple[int, int],
    u1: float,
    u2: float,
    eta: float,
) -> np.ndarray:
    """Helper function to append the gradient of the slack equation to an already
    computed flash system Jacobian as its last row.

    Analogous to :func:`_npipm_extend_res`.

    """

    x, y, _ = parse_xyz(X[:-1], npnc)
    nu = X[-1]
    nphase, ncomp = x.shape

    # NPIPM matrix has one row and one column more
    df_npipm = np.zeros((f_jac.shape[0] + 1, f_jac.shape[1] + 1))
    df_npipm[:-1, :-1] = f_jac
    # relaxed complementary conditions read as y * (1 - sum x) - nu
    # add the -1 for the derivative w.r.t. nu
    df_npipm[-(nphase + 1) : -1, -1] = np.ones(nphase) * (-1)

    unity_j = 1.0 - np.sum(x, axis=1)
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
    d_slack_expanded[-(1 + nphase * ncomp + nphase - 1) : -(1 + nphase * ncomp)] = (
        d_slack[: nphase - 1]
    )

    df_npipm[-1] = d_slack_expanded

    # regularization
    # summation of complementarity conditions
    reg = np.sum(y * (1 - np.sum(x, axis=1)))
    # positive part with penalty factor
    reg = 0.0 if reg < 0 else reg
    reg *= u1 / nphase**2
    # subtract complementarity conditions multiplied with regularization factor from
    # slack equation
    df_npipm[-1] = df_npipm[-1] - reg * np.sum(df_npipm[-(nphase + 1) : -1], axis=0)

    return df_npipm


# endregion

# TODO below solver methods need a static signature, once typing for functions as
# arguments is available in numba
# region Methods related to the numerical solution strategy

# @numba.njit("float64[:](float64[:])")
# def _dummy_res(x: np.ndarray):
#     return x.copy()


# @numba.njit("float64[:,:](float64[:])")
# def _dummy_jac(x: np.ndarray):
#     return np.diag(x)


# _dummy_dict = nbdict.empty(key_type=nbtypes.unicode_type, value_type=nbtypes.float64)
# _dummy_dict["f_dim"] = 5.0
# _dummy_dict["num_phase"] = 2.0
# _dummy_dict["num_comp"] = 2.0
# _dummy_dict["tol"] = 1e-8
# _dummy_dict["max_iter"] = 150.0
# _dummy_dict["rho"] = 0.99
# _dummy_dict["kappa"] = 0.4
# _dummy_dict["j_max"] = 50.0
# _dummy_dict["u1"] = 1.0
# _dummy_dict["u2"] = 10.0
# _dummy_dict["eta"] = 0.5


# @numba.njit(
#     # numba.types.Tuple((numba.float64[:],numba.int32,numba.int32))(
#     #     numba.float64[:],
#     #     numba.typeof(_dummy_res),
#     #     numba.typeof(_dummy_jac),
#     #     numba.int32,
#     #     numba.types.UniTuple(numba.int32,2),
#     #     numba.float64,
#     #     numba.int32,
#     #     numba.float64,
#     #     numba.float64,
#     #     numba.int32,
#     #     numba.float64,
#     #     numba.float64,
#     #     numba.float64,
#     # ),
#     # cache=True,
# )
@numba.njit
def _solver(
    X_0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    f_dim: int,
    npnc: tuple[int, int],
    tol: float,
    max_iter: int,
    rho: float,
    kappa: float,
    j_max: int,
    u1: float,
    u2: float,
    eta: float,
) -> tuple[np.ndarray, int, int]:
    """Compiled Newton with Armijo line search and NPIPM regularization.

    Intended use is for the unified flash problem.

    Parameters:
        X_0: Initial guess.
        F: Callable representing the residual. Must be callable with ``X0``.
        DF: Callable representing the Jacobian. Must be callable with ``X0``.
        F_dim: Dimension of system (number of equations and unknowns). Not necessarily
            equal to length of ``X_0``, since flash system arguments contain parameters
            such as state definitions.
            The last ``F_dim`` elements of ``X_0`` are treated as variables, where as
            the rest is left untouched.
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
        3. final number of performed iterations

        If the success flag indicates failure, the last iterate state of the unknown
        is returned.

    """
    # default return values
    num_iter = 0
    success = 1

    X = X_0.copy()
    DX = np.zeros_like(X_0)
    DX_prev = DX.copy()

    try:
        f_i = _npipm_extend_and_regularize_res(F(X[:-1]), X, npnc, u1, u2, eta)
    except:
        return X, 2, num_iter

    res_k = np.linalg.norm(f_i)

    if res_k <= tol:
        success = 0  # root already found
    else:
        for _ in range(max_iter):
            num_iter += 1

            try:
                df_i = _npipm_extend_and_regularize_jac(
                    DF(X[:-1]), X, npnc, u1, u2, eta
                )
            except:
                success = 3
                break

            # Need this test otherwise np.linalg.solve raises an error.
            if (
                np.any(np.isnan(f_i))
                or np.any(np.isinf(f_i))
                # or np.any(np.isnan(df_i))
                # or np.any(np.isinf(df_i))
            ):
                success = 4
                break

            dx = np.linalg.solve(df_i, -f_i)

            if np.any(np.isnan(dx)) or np.any(np.isinf(dx)):
                success = 4
                break

            # X contains also parameters (p, T, z_i, ...)
            # exactly ncomp - 1 feed fractions and 2 state definitions (p-T, p-h, ...)
            # for broadcasting insert solution into new vector
            DX[-f_dim:] = dx

            # Armijo line search
            pot_i = np.sum(f_i * f_i) / 2.0
            rho_j = rho

            for j in range(1, j_max + 1):
                rho_j = rho**j

                try:
                    X_i_j = X + rho_j * DX
                    f_i_j = _npipm_extend_and_regularize_res(
                        F(X_i_j[:-1]), X_i_j, npnc, u1, u2, eta
                    )
                except:
                    continue

                pot_i_j = np.sum(f_i_j * f_i_j) / 2.0

                if pot_i_j <= (1 - 2 * kappa * rho_j) * pot_i:
                    break

            # heavy ball momentum descend (for cases where Armijo is small)
            # weight -> 1, DX -> 0 as solution is approached
            if rho_j < rho ** (j_max / 2):
                # scale with previous update to avoid large over-shooting
                delta_heavy = 1 / (1 + np.linalg.norm(DX_prev))
            else:
                delta_heavy = 0.0
            X = X + rho_j * DX + delta_heavy * DX_prev
            DX_prev = DX

            try:
                f_i = _npipm_extend_and_regularize_res(F(X[:-1]), X, npnc, u1, u2, eta)
                res_k = np.linalg.norm(f_i)
            except:
                success = 2
                break

            if res_k <= tol:
                # if np.linalg.norm(f_i) / res_0 <= tol:
                success = 0
                break

    return X, success, num_iter


@numba.njit(
    # numba.types.Tuple((numba.float64[:,:],numba.int32[:],numba.int32[:]))(
    #     numba.float64[:,:],
    #     numba.typeof(_dummy_res),
    #     numba.typeof(_dummy_jac),
    #     numba.typeof(_dummy_dict),
    # ),
    # cache=True,
    parallel=True,
)
def parallel_solver(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    solver_params: SOLVER_PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel application of the NPIPM solver to vectorized input, assuming each row
    in ``X0`` is a starting point to find a root of ``F``.

    For an explanation of all parameters, see :func:`newton`.

    Note:
        ``X0`` can contain parameters for the evaluation of ``F``.
        Therefore the dimension of the image of ``F`` must be defined by passing
        ``'f_dim'`` in ``solver_params``.
        I.e., ``len(F(X0[i])) == F_dim`` and ``DF(X0[i]).shape == (F_dim, F_dim)``.

    Parameters:
        X_0: Initial guess, including an initial value for the NPIPM slack variable in
            the last column.
        F: Callable representing the residual. Must be callable with ``X0[:-1]``.
        DF: Callable representing the Jacobian. Must be callable with ``X0[:-1]``.
        solver_paramers: All parameters required by this solver. For more information,
            see the respective type.

    Returns:
        The results of the algorithm per row in ``X0``

        Note however, that the returned results contain only the actual results,
        not the whole, generic flash argument given in ``X0``.
        More precisely, the first ``num_comp - 1 + 2`` elements per row are assumed to
        contain flash specifications in terms of feed fractions and thermodynamic state.
        Hence they are not duplicated and returned to safe memory.

    """

    # extracting solver parameters
    f_dim = int(solver_params["f_dim"])
    npnc = (int(solver_params["num_phase"]), int(solver_params["num_comp"]))
    tol = float(solver_params["tol"])
    max_iter = int(solver_params["max_iter"])
    rho = float(solver_params["rho"])
    kappa = float(solver_params["kappa"])
    j_max = int(solver_params["j_max"])
    u1 = float(solver_params["u1"])
    u2 = float(solver_params["u2"])
    eta = float(solver_params["eta"])

    # alocating return values
    N = X0.shape[0]
    result = np.empty((N, f_dim))
    num_iter = np.empty(N, dtype=np.int32)
    converged = np.empty(N, dtype=np.int32)

    for n in numba.prange(N):
        res_i, conv_i, n_i = _solver(
            X0[n], F, DF, f_dim, npnc, tol, max_iter, rho, kappa, j_max, u1, u2, eta
        )
        converged[n] = conv_i
        num_iter[n] = n_i
        result[n] = res_i[-f_dim:]

    return result, converged, num_iter


# @numba.njit(
#     # numba.types.Tuple((numba.float64[:,:],numba.int32[:],numba.int32[:]))(
#     #     numba.float64[:,:],
#     #     numba.typeof(_dummy_res),
#     #     numba.typeof(_dummy_jac),
#     #     numba.typeof(_dummy_dict),
#     # ),
#     cache=True,
# )
@numba.njit
def linear_solver(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    solver_params: SOLVER_PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Does the same as :func:`parallel_solver`, only the loop over the rows of ``X0``
    is not parallelized, but executed in a classical loop.

    Intended use is for smaller amount of flash problems, where the parallelization
    would produce a certain overhead in the initialization.

    """

    # extracting solver parameters
    f_dim = int(solver_params["f_dim"])
    npnc = (int(solver_params["num_phase"]), int(solver_params["num_comp"]))
    tol = float(solver_params["tol"])
    max_iter = int(solver_params["max_iter"])
    rho = float(solver_params["rho"])
    kappa = float(solver_params["kappa"])
    j_max = int(solver_params["j_max"])
    u1 = float(solver_params["u1"])
    u2 = float(solver_params["u2"])
    eta = float(solver_params["eta"])

    # alocating return values
    N = X0.shape[0]
    result = np.empty((N, f_dim))
    num_iter = np.empty(N, dtype=np.int32)
    converged = np.empty(N, dtype=np.int32)

    for n in range(N):
        res_i, conv_i, n_i = _solver(
            X0[n], F, DF, f_dim, npnc, tol, max_iter, rho, kappa, j_max, u1, u2, eta
        )
        converged[n] = conv_i
        num_iter[n] = n_i
        result[n] = res_i[-f_dim:]

    return result, converged, num_iter


# endregion
