"""Module containing numba-compiled implementations of the NPIPM using the Newton
algorithm and Armijo line search.

To be used by the compiled flash for parallelized computations."""

from __future__ import annotations

from typing import Callable, Literal, TypeAlias

import numba as nb
import numpy as np

from ..._core import NUMBA_CACHE, NUMBA_FAST_MATH
from ..uniflash_equations import parse_generic_arg
from ._armijo_line_search import (  # armijo_line_search,
    _ARMIJO_LINE_SEARCH_PARAMS_KEYS,
    DEFAULT_ARMIJO_LINE_SEARCH_PARAMS,
)
from ._core import SOLVER_FUNCTION_SIGNATURE

__all__ = [
    "DEFAULT_NPIPM_SOLVER_PARAMS",
    "npipm",
]


_NPIPM_SOLVER_PARAMS_KEYS: TypeAlias = Literal[
    "npipm_u1", "npipm_u2", "npipm_eta", "heavy_ball_momentum"
]
"""Keys (names) for NPIPM solver parameters."""


DEFAULT_NPIPM_SOLVER_PARAMS: dict[
    Literal[_NPIPM_SOLVER_PARAMS_KEYS, _ARMIJO_LINE_SEARCH_PARAMS_KEYS],
    float,
] = dict(
    **{
        "npipm_u1": 1.0,
        "npipm_u2": 1.0,
        "npipm_eta": 0.5,
        "heavy_ball_momentum": 0.0,
    },
    **DEFAULT_ARMIJO_LINE_SEARCH_PARAMS,  # type:ignore[arg-type,dict-item]
)
"""Default solver parameters required by the :func:`npipm_solver`.

- ``'npipm_u1': 1.`` penalty for violating complementarity
- ``'npipm_u2': 1.`` penalty for violating negativity of fractions
- ``'npipm_eta': 0.5`` linear decline in slack variable
- ``'heavy_ball_momentum': 0.`` if True (non-zero), a heavy-ball momentum technique is
  applied to the line-search, adding the update from the previous iteration with some
  down-scaling to the current update.

This solver uses also the :func:`armijo_line_search`, and respective
:data:`DEFAULT_ARMIJO_LINE_SEARCH_PARAMS`.

"""


@nb.njit(
    nb.f8(nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _slack_equation_res(
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


@nb.njit(
    nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _slack_equation_jac(
    v: np.ndarray, w: np.ndarray, nu: float, u1: float, u2: float, eta: float
) -> np.ndarray:
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


@nb.njit(
    nb.f8[:](nb.f8[:], nb.f8[:], nb.types.UniTuple(nb.i4, 2), nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def _extend_and_regularize_res(
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

    gen = parse_generic_arg(X[:-1], npnc, "p-T")
    x = gen[1]
    y = gen[2]
    nu = X[-1]

    nphase = x.shape[0]

    # couple complementary conditions with nu
    f_res[-nphase:] -= nu

    # NPIPM equation
    unity_j = 1.0 - np.sum(x, axis=1)
    slack = _slack_equation_res(y, unity_j, nu, u1, u2, eta)

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


@nb.njit(
    nb.f8[:, :](
        nb.f8[:, :], nb.f8[:], nb.types.UniTuple(nb.i4, 2), nb.f8, nb.f8, nb.f8
    ),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def _extend_and_regularize_jac(
    f_jac: np.ndarray,
    X: np.ndarray,
    npnc: tuple[int, int],
    u1: float,
    u2: float,
    eta: float,
) -> np.ndarray:
    """Helper function to append the gradient of the slack equation to an already
    computed flash system Jacobian as its last row.

    Analogous to :func:`_extend_and_regularize_res`.

    """

    gen = parse_generic_arg(X[:-1], npnc, "p-T")
    x = gen[1]
    y = gen[2]
    nu = X[-1]
    nphase, ncomp = x.shape

    # NPIPM matrix has one row and one column more
    df_npipm = np.zeros((f_jac.shape[0] + 1, f_jac.shape[1] + 1))
    df_npipm[:-1, :-1] = f_jac
    # relaxed complementary conditions read as y * (1 - sum x) - nu
    # add the -1 for the derivative w.r.t. nu
    df_npipm[-(nphase + 1) : -1, -1] = np.ones(nphase) * (-1)

    unity_j = 1.0 - np.sum(x, axis=1)
    d_slack = _slack_equation_jac(y, unity_j, nu, u1, u2, eta)
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


@nb.njit(SOLVER_FUNCTION_SIGNATURE, cache=NUMBA_CACHE)
def npipm(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    params: dict,
) -> tuple[np.ndarray, int, int]:
    """Compiled Newton with Armijo line search and NPIPM regularization.

    For more information on the signature, see
    :data:`~porepy.compositional.flash.solvers._core.SOLVER_FUNCTION_SIGNATURE` and
    :data:`~porepy.compositional.flash.solvers._core.SOLVER_FUNCTION_TYPE`.

    For a list of required solver parameters, see
    :data:`DEFAULT_NPIPM_SOLVER_PARAMS` and
    :data:`~porepy.compositional.flash.solvers._core.GENERAL_SOLVER_PARAMS`.

    """
    # default return values
    num_iter = 0
    success = 1

    # extracting solver parameters
    f_dim = int(params["f_dim"])
    npnc = (int(params["num_phases"]), int(params["num_components"]))
    tol = float(params["tolerance"])
    max_iter = int(params["max_iterations"])
    rho = float(params["armijo_rho"])
    kappa = float(params["armijo_kappa"])
    max_iter_armijo = int(params["armijo_max_iterations"])
    u1 = float(params["npipm_u1"])
    u2 = float(params["npipm_u2"])
    eta = float(params["npipm_eta"])
    heavy_ball = int(params["heavy_ball_momentum"])

    # Computing initial value for slack variable.
    gen = parse_generic_arg(X0, npnc, "p-T")
    x = gen[1]
    y = gen[2]
    nu = np.sum(y * (1 - np.sum(x, axis=1))) / npnc[0]

    # numba does not support stacking with inhomogenous sequence of array and float
    X = np.zeros(X0.shape[0] + 1)
    X[:-1] = X0
    X[-1] = nu
    DX = np.zeros_like(X)
    DX_prev = DX.copy()

    # complete system size including slack equation
    matrix_rank = f_dim + 1

    try:
        f_i = _extend_and_regularize_res(F(X[:-1]), X, npnc, u1, u2, eta)
    except Exception:  # whatever happens, residual evaluation is faulty
        return X, 3, num_iter

    res_i = np.linalg.norm(f_i)

    if res_i <= tol:
        success = 0  # root already found
    else:
        for _ in range(max_iter):
            num_iter += 1

            try:
                df_i = _extend_and_regularize_jac(DF(X[:-1]), X, npnc, u1, u2, eta)
            except Exception:  # whatever happens, Jacobian assembly is faulty
                success = 4
                break

            # Need this test otherwise np.linalg.solve raises an error.
            if (
                np.any(np.isnan(f_i)) or np.any(np.isinf(f_i))
                # or np.any(np.isnan(df_i))
                # or np.any(np.isinf(df_i))
            ):
                success = 2
                break

            if np.linalg.matrix_rank(df_i) == matrix_rank:
                DX[-matrix_rank:] = np.linalg.solve(df_i, -f_i)
            else:
                # NOTE rcond is the limit to cutting of the smallest singular values.
                # This has quite large effects on the robustness of the flash in the
                # v-h case for example, which is not yet fully understood.
                # NOTE also, the default value in numba is machine precision, while
                # with no-jit (pure numpy) it is as below.
                DX[-matrix_rank:] = np.linalg.lstsq(
                    df_i, -f_i, rcond=np.finfo(np.float64).eps * df_i.shape[0]
                )[0]

            if np.any(np.isnan(DX)) or np.any(np.isinf(DX)):
                success = 2
                break

            # Armijo line search
            # rho_i = armijo_line_search(X[:-1], DX[:-1], F, params)
            pot_i = np.sum(f_i * f_i) / 2.0
            rho_i = rho

            for j in range(1, max_iter_armijo + 1):
                rho_i = rho**j

                try:
                    X_i_j = X + rho_i * DX
                    f_i_j = _extend_and_regularize_res(
                        F(X_i_j[:-1]), X_i_j, npnc, u1, u2, eta
                    )
                except Exception:
                    # NOTE Here we allow the residual evaluation to fail and skip the
                    # line search step, as this might happen when dealing with
                    # non-smooth F.
                    # By continuing the step size comes closer to the old iterate
                    # making the line search more robust, but slowing the overall
                    # progress
                    continue

                pot_i_j = np.sum(f_i_j * f_i_j) / 2.0

                if pot_i_j <= (1 - 2 * kappa * rho_i) * pot_i:
                    break

            X = X + rho_i * DX

            if heavy_ball > 0:
                # heavy ball momentum descend (for cases where Armijo is small)
                # weight -> 1, DX -> 0 as solution is approached
                if rho_i < rho ** (max_iter_armijo / 2):
                    # scale with previous update to avoid large over-shooting
                    delta_heavy = 1 / (1 + np.linalg.norm(DX_prev))
                else:
                    delta_heavy = 0.0  # type:ignore[assignment]
                X = X + delta_heavy * DX_prev
                DX_prev = DX.copy()

            try:
                f_i = _extend_and_regularize_res(F(X[:-1]), X, npnc, u1, u2, eta)
                res_i = np.linalg.norm(f_i)
            except Exception:
                success = 3
                break

            if res_i <= tol:
                success = 0
                break

    return X[:-1], success, num_iter
