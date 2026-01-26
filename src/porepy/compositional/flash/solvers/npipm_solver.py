"""Module containing numba-compiled implementations of the NPIPM using the Newton
algorithm and Armijo line search.

To be used by the compiled flash for parallelized computations."""

from __future__ import annotations

from typing import Callable, Literal, TypeAlias

import numba as nb
import numpy as np

from ..._numba_interface import NUMBA_CACHE, NUMBA_FAST_MATH, njit
from ...utils import FlashSpec
from ..flash_equations import parse_xy
from ._armijo_line_search import (  # armijo_line_search,
    _ARMIJO_LINE_SEARCH_PARAMS_KEYS,
    DEFAULT_ARMIJO_LINE_SEARCH_PARAMS,
)
from ._core import SOLVER_FUNCTION_SIGNATURE

__all__ = [
    "DEFAULT_NPIPM_SOLVER_PARAMS",
    "npipm",
]


_COMPILER = njit
"""Compiler to be used for functions in this module."""


_NPIPM_SOLVER_PARAMS_KEYS: TypeAlias = Literal[
    "npipm_u1",
    "npipm_u2",
    "npipm_eta",
    "rpc_T",
    "rpc_T_damp",
    "rpc_p",
    "rpc_p_damp",
    "trustregion_delta",
    "trustregion_fraction_to_boundary",
    "anderson_acceleration",
    "anderson_acceleration_regularization",
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
        "rpc_T": 1.0,
        "rpc_T_damp": 1.0,
        "rpc_p": 1.0,
        "rpc_p_damp": 1.0,
        "trustregion_delta": 0.5,
        "trustregion_fraction_to_boundary": 0.95,
        "anderson_acceleration": 0,
        "anderson_acceleration_regularization": 1e-7,
    },
    **DEFAULT_ARMIJO_LINE_SEARCH_PARAMS,  # type:ignore[arg-type,dict-item]
)
"""Default solver parameters required by the :func:`npipm_solver`.

- ``'npipm_u1': 1.`` penalty for violating complementarity
- ``'npipm_u2': 1.`` penalty for violating negativity of fractions
- ``'npipm_eta': 0.5`` linear decline in slack variable


This solver uses also the :func:`armijo_line_search`, and respective
:data:`DEFAULT_ARMIJO_LINE_SEARCH_PARAMS`.

"""


@_COMPILER(
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

    n_P = v.shape[0]

    # penalization of negativity
    res = 0.5 * u2 * (np.sum(v[v < 0] ** 2) + np.sum(w[w < 0] ** 2))

    # penalization of violation of complementarity
    dot = max(0.0, np.sum(v * w))
    res += 0.5 * u1 / n_P**2 * dot**2

    # decline of slack variable
    res += eta * nu + nu**2

    return res


@_COMPILER(
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

    n_P = v.shape[0]

    jac = np.zeros(2 * n_P + 1, dtype=np.float64)

    dot = np.sum(v * w)

    # derivatives of pos() and neg()
    dirac_dot = 1.0 if dot > 0.0 else 0.0  # dirac for positivity of dotproduct
    dirac_v = (v < 0.0).astype(np.float64)  # dirac for negativity in v, elementwise
    dirac_w = (w < 0.0).astype(np.float64)  # same for w

    d_dot_outer = u1 / n_P**2 * dot * dirac_dot

    # derivatives w.r.t. to elements in v
    jac[:n_P] = u2 * dirac_v * v + d_dot_outer * w
    jac[n_P : 2 * n_P] = u2 * dirac_w * w + d_dot_outer * v

    # derivative w.r.t. nu
    jac[-1] = eta + 2 * nu

    return jac


@_COMPILER(
    nb.f8[:](nb.f8[:], nb.f8[:, :], nb.f8[:], nb.f8, nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def extend_and_regularize_res(
    f_res: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    nu: float,
    u1: float,
    u2: float,
    eta: float,
) -> np.ndarray:
    """Helper function to append the residual of the slack equation to an
    already computed flash residual.

    Important:
        This function assumes that the last ``num_phases`` entries in the residual
        correspond to the residual values of the complementarity conditions.

    Parameters:
        f_res: Residual of flash equations, shape ``(f_dim,)``.
        x: Phase compositions, shape ``(num_phases, num_components)``.
        y: Phase fractions, shape ``(num_phases,)``.
        nu: Value of slack variable.
        u1: Parameter to tune the penalty for violation of complementarity.
        u2: Parameter to tune the penalty for violation of negativity.
        eta: Parameter for steepness of decline of slack variable.

    Returns:
        The modified residual with the slack equation residual appended.

    """

    n_P = x.shape[0]

    # couple complementary conditions with nu
    f_res[-n_P:] -= nu

    # NPIPM equation
    unity_j = 1.0 - np.sum(x, axis=1)
    slack = _slack_equation_res(y, unity_j, nu, u1, u2, eta)

    # NPIPM system has one equation more at end
    f_npipm = np.zeros(f_res.shape[0] + 1)
    f_npipm[:-1] = f_res
    f_npipm[-1] = slack

    # regularization
    reg = max(0.0, np.sum(y * (1 - np.sum(x, axis=1)))) * u1 / n_P**2
    # subtract complementarity conditions multiplied with regularization factor from
    # slack equation residual
    f_npipm[-1] -= reg * np.sum(f_res[-n_P:])

    return f_npipm


@_COMPILER(
    nb.f8[:, :](nb.f8[:, :], nb.f8[:, :], nb.f8[:], nb.f8, nb.f8, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def extend_and_regularize_jac(
    f_jac: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    nu: float,
    u1: float,
    u2: float,
    eta: float,
) -> np.ndarray:
    """Helper function to append the gradient of the slack equation to an already
    computed flash system Jacobian as its last row.

    Analogous to :func:`_extend_and_regularize_res`.

    """

    n_P, n_C = x.shape
    n_PC = n_P * n_C

    # NPIPM matrix has one row and one column more
    df_npipm = np.zeros((f_jac.shape[0] + 1, f_jac.shape[1] + 1))
    df_npipm[:-1, :-1] = f_jac
    # relaxed complementary conditions read as y * (1 - sum x) - nu
    df_npipm[-(n_P + 1) : -1, -1] = -1.0

    unity_j = 1.0 - np.sum(x, axis=1)
    d_slack = _slack_equation_jac(y, unity_j, nu, u1, u2, eta)

    # expand it also to include possibly other derivatives
    d_slack_expanded = np.zeros(df_npipm.shape[1])
    # last derivative is w.r.t. nu
    d_slack_expanded[-1] = d_slack[-1]
    # derivatives w.r.t y_j, y_0 = 1 - sum_{j>0} y_j
    d_slack_expanded[-(n_PC + n_P) : -(1 + n_PC)] = d_slack[1:n_P] - d_slack[0]

    for j in range(n_P):
        # derivatives w.r.t. x_ij,
        # w_j = 1 - sum_i x_ij, so derivative is -d_slack w.r.t. w_j
        # + 2 to skip nu, since j starts with 0.
        d_slack_expanded[-(1 + (j + 1) * n_C) : -(1 + j * n_C)] = -d_slack[-(j + 2)]

    df_npipm[-1] = d_slack_expanded

    # regularization
    reg = max(0.0, np.sum(y * (1 - np.sum(x, axis=1)))) * u1 / n_P**2
    # subtract complementarity conditions multiplied with regularization factor from
    # slack equation
    df_npipm[-1] -= reg * np.sum(df_npipm[-(n_P + 1) : -1], axis=0)

    return df_npipm


@_COMPILER(nb.f8(nb.f8[:], nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def _trust_region_scale(d: np.ndarray, delta: float) -> float:
    """Trust-region scaling for update vector, if norm of update surpasses given
    delta-value.

    Parameters:
        d: Update vector.
        delta: Trust-region radius.

    Returns:
        A float to scale the update to stay within the trusted region.

    """
    d_norm = np.linalg.norm(d)
    if d_norm > delta:
        return delta / d_norm
    else:
        return 1.0


@_COMPILER(
    nb.f8(nb.f8[:], nb.f8[:], nb.int_, nb.int_, nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _feasible_fractions_scale(
    v: np.ndarray, d: np.ndarray, n_C: int, n_P: int, d_min: float, eps: float
) -> float:
    """Scaling for update to fractions so that they remain bounded in [0, 1] and the
    update does not violate the unity constraints (each family of fractions summed up
    is smaller or equal to 1).

    Parameters:
        v: Current state vector.
        d: Update vector.
        n_P: Number of phases.
        n_C: Number of components.
        d_min: Value considered numerically significant (e.g. 1e-8)
        eps: Detecting near-absent and near-saturated phases with small updates to
            exclude for numerical stability (e.g. 1e-10 to 1e-12)

    Returns:
        A scaling factor for the update which ensures that the fractions remain
        feasible.

    """
    alpha = 1.0  # Default scale.
    rel_tol = 1e-2  # Default relative scale for fractions denoting significance.

    emv = 1.0 - v
    v0 = np.abs(v) < eps
    v1 = np.abs(emv) < eps
    # Numerically relevant update in relative terms.
    d_rel = (np.abs(d) / max(np.abs(v).max(), np.abs(emv).max(), rel_tol)) >= d_min

    # Ensuring lower-bound feasibility:
    # Negative update, fractions not numerically 0 and update numerically significant.
    lbidx = (d < 0) & (~v0) & d_rel
    if np.any(lbidx):
        alpha = min(alpha, np.min(-v[lbidx] / d[lbidx]))

    # Ensuring upper-bound feasibility:
    # Positive update, fractions not numericall 1 and update numerically significant.
    upidx = (d > 0) & (~v1) & d_rel
    if np.any(upidx):
        alpha = min(alpha, np.min(emv[upidx] / d[upidx]))

    # Simplex feasibility for updates:
    # The sum of each family of fractions must remain smaller or equal to 1.
    dx, dy = parse_xy(d, n_C, n_P)
    vx, vy = parse_xy(v, n_C, n_P)

    sdy = dy.sum()
    sy = vy.sum()
    asdy = np.abs(sdy)
    if (sdy > 0) & (asdy >= eps) & ((asdy / max(sy, 1 - sy, rel_tol)) >= d_min):
        alpha = min(alpha, (1.0 - vy.sum()) / sdy)

    for j in range(n_P):
        sdx = dx[j].sum()
        svx = vx[j].sum()
        asdx = np.abs(sdx)
        if (sdx > 0) & (asdx >= eps) & ((asdx / max(svx, 1 - svx, rel_tol)) >= d_min):
            alpha = min(alpha, (1.0 - svx) / sdx)

    return alpha


@_COMPILER(
    nb.f8[:](nb.f8[:], nb.f8[:], nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _trust_region_cap(X: np.ndarray, DX: np.ndarray, tau: float) -> np.ndarray:
    """Helper function to apply the fraction-to-boundary rule to the fraction update.

    Parameters:
        X: Current vector of fractions.
        DX: Update vector.
        tau: Trust-region parameter in ``(0, 1)``.

    Returns:
        The scaled update ensuring fractions do not violate bounds [0,1].

    """
    # For ignoring small updates or fractions close to boundaries.
    EPS = 1e-7
    small = np.abs(DX) <= EPS
    emX = 1.0 - X

    X_0 = np.abs(X) <= EPS
    X_1 = np.abs(emX) <= EPS

    neg_d = DX < 0.0
    pos_d = DX > 0.0

    # Violations of lower bound 0
    aidx = neg_d & (~(X_0 | small))
    a = min(1.0, tau * np.min(X[aidx] / -DX[aidx])) if np.any(aidx) else 1.0

    # Violations of upper bound 1, same logic.
    bidx = pos_d & (~(X_1 | small))
    b = min(1.0, tau * np.min(emX[bidx] / DX[bidx])) if np.any(bidx) else 1.0

    DX_s = DX * min(a, b)

    # Cancel the update where fractions close to 0 and 1
    DX_s[X_0 & neg_d] = 0.0
    DX_s[X_1 & pos_d] = 0.0

    return DX_s


@_COMPILER(SOLVER_FUNCTION_SIGNATURE, cache=NUMBA_CACHE)
def npipm(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    params: dict,
    spec: FlashSpec,
) -> tuple[np.ndarray, int, int]:
    """Compiled Newton with Armijo line search and NPIPM regularization.

    For more information on the signature, see
    :data:`~porepy.compositional.flash.solvers._core.SOLVER_FUNCTION_SIGNATURE` and
    :data:`~porepy.compositional.flash.solvers._core.SOLVER_FUNCTION_TYPE`.

    For a list of required solver parameters, see
    :data:`DEFAULT_NPIPM_SOLVER_PARAMS` and
    :data:`~porepy.compositional.flash.solvers._core.GENERAL_SOLVER_PARAMS`.

    """
    # Default return values.
    num_iter = 0
    exitcode = 1
    EPS = np.finfo(np.float64).eps

    # region Extracting solver and user-given parameters.
    f_dim = int(params["f_dim"])
    n_C = int(params["num_components"])
    n_P = int(params["num_phases"])
    tol = float(params["tolerance"])
    max_iter = int(params["max_iterations"])
    rho = float(params["armijo_step_size"])
    kappa = float(params["armijo_incline"])
    max_iter_armijo = int(params["armijo_max_iterations"])
    u1 = float(params["npipm_u1"])
    u2 = float(params["npipm_u2"])
    eta = float(params["npipm_eta"])
    tr_frac_to_bound = float(params["trustregion_fraction_to_boundary"])
    tr_delta = float(params["trustregion_delta"])
    anderson = int(params["anderson_acceleration"])
    anderson_reg = float(params["anderson_acceleration_regularization"])
    # endregion

    # L-scheme for diagonal stabilization.
    L_scheme = 0.0

    # region System set-up
    arank = f_dim + 1
    # NOTE rcond is the limit to cutting off singular values.
    # This has quite large effects on the robustness of the flash in the
    # vh case for example, which is not yet fully understood.
    # NOTE also, the default value in numba is machine precision, while
    # with no-jit (pure numpy) is shape[0] * eps.
    # The latter is chosen and set to avoid differences between jit and
    # no-jit computations.
    rcond = float(arank * EPS)

    # Generic part of the flash argument vector.
    X_gen = X0[: -arank + 1].copy()

    # DOF part of flash argument vector, including slack variable.
    X_i = np.zeros(arank)
    X_i[:-1] = X0[-arank + 1 :].copy()
    dX_i = np.zeros_like(X_i)

    # Computing initial value for slack variable, as sum of violations of
    # complementarity.
    xf, yf = parse_xy(X_i[:-1], n_C, n_P)
    X_i[-1] = np.abs(np.sum(yf * (1 - np.sum(xf, axis=1))))

    def eval_F(X_loc: np.ndarray) -> np.ndarray:
        nu_loc = X_loc[-1]
        _X_loc = X_loc[:-1]
        _x, _y = parse_xy(_X_loc, n_C, n_P)
        f_loc = F(np.hstack((X_gen, _X_loc)))
        f_npipm = extend_and_regularize_res(f_loc, _x, _y, nu_loc, u1, u2, eta)
        return f_npipm

    def eval_DF(X_loc: np.ndarray) -> np.ndarray:
        nu_loc = X_loc[-1]
        _X_Loc = X_loc[:-1]
        _x, _y = parse_xy(_X_Loc, n_C, n_P)
        df_loc = DF(np.hstack((X_gen, _X_Loc)))
        df_npipm = extend_and_regularize_jac(df_loc, _x, _y, nu_loc, u1, u2, eta)
        return df_npipm

    # endregion

    # region Cycling detection parameters. We expect cycles of 2 to 5.
    res_history = np.zeros(10)
    cycling_tol = 1e-2  # relative tolerance for detecting cycling.
    was_cycling = False
    is_cycling = False
    iter_cycle_detected = 0
    detected_cycle = 0
    res_lb = 0.0  # lower-bound for residual when cycling detected.
    do_perturb = False
    LM_mode = False  # Levenberg-Marquardt normalization
    steepest_descent = False  # Last resort method
    # endregion

    # Adding default value to keep track of residuals for dynamic acceleration switch.
    default_anderson = 3
    if anderson > 0:
        Fk = np.zeros((arank, anderson))
        Gk = np.zeros((arank, anderson))
    else:
        Fk = np.zeros((arank, default_anderson))
        Gk = np.zeros((arank, default_anderson))
    fk1m = np.zeros(arank)
    gk1m = np.zeros(arank)

    # region Right-preconditioning for non-isothermal or isochoric flashes.
    do_rpc_T = False
    rpc_T_idx = -1  # Default value, not used.
    T_rpc = 1.0
    rpc_T_damp = float(params["rpc_T_damp"])
    do_rpc_p = False
    rpc_p_idx = -1
    p_rpc = 1.0
    rpc_p_damp = float(params["rpc_p_damp"])

    if spec not in (FlashSpec.pT, FlashSpec.vT):
        T_rpc = float(params["rpc_T"])
        rpc_T_idx = 0
        do_rpc_T = True

    if spec >= FlashSpec.vT:
        p_rpc = float(params["rpc_p"])
        rpc_p_idx = 0
        # Shift index because T-derivatives come after p-derivatives.
        rpc_T_idx += 1
        do_rpc_p = True
    # endregion

    try:
        f_i = eval_F(X_i)
    except:  # whatever happens, residual evaluation is faulty.
        return X0, 3, num_iter

    res_history[-1] = np.linalg.norm(f_i)
    if res_history[-1] <= tol:
        exitcode = 0  # root already found
    else:
        for i in range(max_iter):
            num_iter = i + 1

            # Need this test otherwise np.linalg.solve raises an error.
            if np.any(np.isnan(f_i)) or np.any(np.isinf(f_i)):
                exitcode = 2
                break

            try:
                df_i = eval_DF(X_i)
            except:  # whatever happens, Jacobian assembly is faulty
                exitcode = 4
                break

            A = df_i
            b = -f_i

            if do_rpc_T:
                A[:, rpc_T_idx] *= T_rpc
            if do_rpc_p:
                A[:, rpc_p_idx] *= p_rpc

            if LM_mode:
                b = A.T @ b
                A = A.T @ A

            if L_scheme > 0.0:
                A += L_scheme * np.eye(A.shape[0])

            if steepest_descent:
                dX_i = b
            else:
                try:
                    dX_i = np.linalg.solve(A, b)
                except:
                    try:
                        dX_i = np.linalg.lstsq(A, b, rcond=rcond)[0]
                    except:
                        # This means the linear solver failed.
                        exitcode = 5
                        break

            if np.any(np.isnan(dX_i)) or np.any(np.isinf(dX_i)):
                exitcode = 2
                break

            # region Trust-region scaling
            dX_i *= _trust_region_scale(dX_i[:-1], tr_delta)
            f2b = _feasible_fractions_scale(X_i[:-1], dX_i[:-1], n_C, n_P, 1e-8, 1e-12)
            if f2b < 1.0:
                f2b *= tr_frac_to_bound
            dX_i *= f2b
            # endregion

            if do_rpc_T:
                dX_i[rpc_T_idx] = np.sign(dX_i[rpc_T_idx]) * min(
                    np.abs(dX_i[rpc_T_idx]), rpc_T_damp
                )
                dX_i[rpc_T_idx] *= T_rpc
            if do_rpc_p:
                dX_i[rpc_p_idx] = np.sign(dX_i[rpc_p_idx]) * min(
                    np.abs(dX_i[rpc_p_idx]), rpc_p_damp
                )
                dX_i[rpc_p_idx] *= p_rpc

            # region Armijo line search.
            pot_i = np.sum(f_i * f_i) * 0.5
            rho_i = rho

            for j in range(0, max_iter_armijo + 1):
                rho_i = rho**j
                X_i_j = X_i + rho_i * dX_i

                try:
                    f_i_j = eval_F(X_i_j)
                except:
                    # NOTE Here we allow the residual evaluation to fail and skip the
                    # line search step, as this might happen when dealing with
                    # non-smooth F.
                    # By continuing the step size comes closer to the old iterate
                    # making the line search more robust, but slowing the overall
                    # progress
                    continue

                pot_i_j = np.sum(f_i_j * f_i_j) * 0.5

                if pot_i_j <= (1 - 2 * kappa * rho_i) * pot_i:
                    break
                # If update becomes too small, risking progress, break.
                if rho_i * np.linalg.norm(dX_i) < 1e-5:
                    break

            dX_i *= rho_i
            # endregion

            # region Anderson acceleration
            if i > 0:
                col = (i - 1) % Fk.shape[1]
                fk = dX_i
                gk = X_i + dX_i
                Fk[:, col] = fk - fk1m
                Gk[:, col] = gk - gk1m
                fk1m = fk.copy()
                gk1m = gk.copy()

            mk = min(i, anderson)
            if mk > 0:
                A = Fk[:, :mk]
                b = fk

                if anderson_reg > 0:
                    b = A.T @ b
                    A = A.T @ A + anderson_reg * np.eye(A.shape[1])

                g = np.linalg.lstsq(A, b, rcond=rcond)[0]

                dX_i = gk - np.dot(Gk[:, :mk], g) - X_i

            # endregion

            # Apply update, with another trust-region cap to be safe.
            X_i += dX_i

            try:
                f_i = eval_F(X_i)
            except:
                exitcode = 3
                break

            res_history = np.roll(res_history, -1)
            res_history[-1] = np.linalg.norm(f_i)
            # print(res_history[-4:])
            if res_history[-1] <= tol:
                exitcode = 0
                break

            # region Cycling detection.
            if num_iter > res_history.size:
                was_cycling = is_cycling
                # Looky only for cycles with period 2 to 5.
                for c in range(2, 6):
                    check_len = 2 * c

                    recent = res_history[-check_len:]
                    scaled_tol = np.max(np.abs(recent)) * cycling_tol

                    is_cycling = True
                    for i in range(c, check_len):
                        if np.abs(recent[i] - recent[i - c]) >= scaled_tol:
                            is_cycling = False
                            break

                    if is_cycling:
                        detected_cycle = c
                        res_lb = np.min(recent) - scaled_tol
                        break

                # Detected new cycling, make line search more aggressive.
                if is_cycling and not was_cycling:
                    rho = max(0.5, 0.5 * rho)
                    max_iter_armijo = max(80, int(params["armijo_max_iterations"]) * 3)
                    kappa = min(1.5 * kappa, 0.5)
                    iter_cycle_detected = num_iter

                    # Special case when cycling detected is often around the isofugacity
                    # constraints, in case there are absent phases. To shortcut the
                    # routine, go immediately to perturbation, which maximizes 1-sum(x)
                    idx = np.zeros(arank, dtype=np.bool)
                    idx[-(n_P + 1 + n_C * (n_P - 1)) : -(n_P + 1)] = True
                    if np.linalg.norm(f_i[~idx]) < tol:
                        do_perturb = True

                # No longer cycling.
                elif not is_cycling and was_cycling:
                    # Check that really not cycling anymore for some iterations.
                    if (
                        np.all(np.diff(res_history[-2 * detected_cycle :]) < 0)
                        and res_history[-1] < res_lb
                    ):
                        # Gradually reset line search parameters.
                        rho = (rho + float(params["armijo_step_size"])) * 0.5
                        kappa = (kappa + float(params["armijo_incline"])) * 0.5
                        max_iter_armijo = int(params["armijo_max_iterations"])
                        anderson = int(params["anderson_acceleration"])
                    # If residual decreases but unsure about cycling, activate
                    # stabilization and acceleration techniques.
                    elif not LM_mode:
                        # else:
                        if res_history[-1] < res_lb:
                            L_scheme = 0.7
                        else:
                            is_cycling = True

                if LM_mode:
                    if res_history[-1] < res_history[-2]:
                        L_scheme = max(1e-5, L_scheme * 0.5)
                    else:
                        L_scheme = min(1e12, 10.0 * L_scheme)

                    if L_scheme > 1e3:
                        max_iter_armijo = 10
                        kappa = 0.1
                        # Last resort, if everything fails, likely due to a numerically
                        # flat minimized functional, we switch to steepest descent to
                        # stabilize the results until iterations run out.
                        if res_history[-1] < 1e-2:
                            steepest_descent = True

                # If cycling continued for some time without being broken, we
                # initiate perturbation.
                do_perturb = (
                    is_cycling
                    and was_cycling
                    and num_iter - iter_cycle_detected > detected_cycle
                ) | do_perturb

                # Add random perturbation to escape cycle if still cycling.
                if do_perturb:
                    do_perturb = False
                    is_perturbed = False
                    # Add perturbation only to extended fractions where phase absent.
                    xf, yf = parse_xy(X_i[:-1], n_C, n_P)
                    for j in range(n_P):
                        xj = xf[j, :]
                        # Perturb extended fractions to be closer to uniform.
                        if xj.sum() <= 1.0 - 1e-3:
                            xj = (xj + 1.0 / n_C) / 2.0
                            # yf[j] = 0.0
                            xf[j, :] = xj
                            is_perturbed = True

                    cond = np.linalg.cond(df_i)

                    # NOTE Perturbing the phase fractions too seems to have negative
                    # effects on convergence.
                    # We set it to zero where the extended fractions are obviously
                    # far from summing to one and re-normalize.
                    if is_perturbed:
                        # yf /= np.sum(yf)
                        # X_i[-(n_P - 1 + n_C * n_P + 1) : -(n_C * n_P + 1)] = yf[1:]

                        X_i[-n_C * n_P - 1 : -1] = xf.flatten()

                        # Reset changes to parameters introduced by cycle-breaking.
                        is_cycling = False
                        rho = float(params["armijo_step_size"])
                        kappa = float(params["armijo_incline"])
                        # L_scheme = 0.0
                        # LM_mode = False
                        # anderson = int(params["anderson_acceleration"])
                    elif cond > 1e5 and not LM_mode:
                        # elif cond > 1e5:
                        L_scheme = 10.0 if cond > 1e6 else 1.0
                        LM_mode = True
                        if anderson == 0:
                            anderson = default_anderson
            # endregion

    if np.any(np.isnan(X_i)) or np.any(np.isinf(X_i)):
        # Return initial guess back to not break subsequent code.
        X_i[:-1] = X0[-arank + 1 :].copy()
        assert exitcode > 1, "Expecting exitcode > 1 in case of divergence."

    return np.hstack((X_gen, X_i[:-1])), exitcode, num_iter
