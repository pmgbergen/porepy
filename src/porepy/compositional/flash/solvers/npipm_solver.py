"""Module containing numba-compiled implementations of the NPIPM using the Newton
algorithm and Armijo line search.

To be used by the compiled flash for parallelized computations."""

from __future__ import annotations

from typing import Callable, Literal

import numba as nb
import numpy as np

from ..._numba_interface import NUMBA_CACHE, NUMBA_FAST_MATH, njit
from ...utils import FlashSpec, FlashSpec_NUMBA_TYPE
from ..flash_equations import parse_generic_arg, parse_xy
from ._core import (
    FLASH_JACOBIAN_FUNCTION_TYPE,
    FLASH_RESIDUAL_FUNCTION_TYPE,
    SOLVER_FUNCTION_SIGNATURE,
    SOLVER_PARAMETERS_TYPE,
)

__all__ = [
    "DEFAULT_NPIPM_SOLVER_PARAMS",
    "npipm",
]


_COMPILER = njit
"""Compiler to be used for functions in this module."""


DEFAULT_NPIPM_SOLVER_PARAMS: dict[
    Literal[
        "npipm_penalty_cc",
        "npipm_penalty_neg",
        "npipm_slack_decline",
        "armijo_step_size",
        "armijo_decline",
        "rpc_T",
        "rpc_T_chop",
        "rpc_p",
        "rpc_p_chop",
        "trustregion_delta",
        "trustregion_fraction_to_boundary",
        "anderson_acceleration",
        "anderson_acceleration_regularization",
        "pT_npc_iterations",
    ],
    float,
] = {
    "npipm_penalty_cc": 1.0,
    "npipm_penalty_neg": 1.0,
    "npipm_slack_decline": 0.5,
    "armijo_step_size": 0.99,
    "armijo_decline": 0.45,
    "rpc_T": 1.0,
    "rpc_T_chop": np.inf,
    "rpc_p": 1.0,
    "rpc_p_chop": np.inf,
    "trustregion_delta": 0.0,
    "trustregion_fraction_to_boundary": 0.95,
    "anderson_acceleration": 0,
    "anderson_acceleration_regularization": 1e-7,
    "pT_npc_iterations": 0,
}
"""Default solver parameters required by the :func:`npipm_solver`.

- ``'npipm_penalty_cc': 1.`` penalty for violating complementarity.
- ``'npipm_penalty_neg': 1.`` penalty for violating negativity of fractions.
- ``'npipm_slack_decline': 0.45`` linear decline in slack variable.


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


@_COMPILER(
    nb.f8(nb.f8[:], nb.f8[:], nb.f8, nb.f8),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _feasible_fractions_scale(
    v: np.ndarray, d: np.ndarray, d_min: float, eps: float
) -> float:
    """Scaling for update to fractions so that they remain bounded in [0, 1] and the
    update does not violate the unity constraints (each family of fractions summed up
    is smaller or equal to 1).

    Parameters:
        v: Vector of fractions.
        d: Update vector for fractions.
        d_min: Value considered numerically significant (e.g. 1e-8)
        eps: Detecting near-absent and near-saturated phases with small updates to
            exclude for numerical stability (e.g. 1e-10 to 1e-12)

    Returns:
        A scaling factor for the update which ensures that the fractions remain
        feasible.

    """
    alpha = 1.0  # Default scale.

    # NOTE: In order to avoid a negative scaling, we must exclude fractions outside
    # [0, 1]. We might require a scaling somewhere else, or postprocessing if that
    # happens.
    feasible = (v >= 0) & (v <= 1)

    emv = 1.0 - v
    v0 = np.abs(v) < eps
    # Numerically relevant update in relative terms.
    d_rel = np.abs(d) >= d_min * max(np.abs(v).max(), np.abs(emv).max())

    # Ensuring lower-bound feasibility:
    # Negative update, fractions not numerically 0 and update numerically significant.
    lbidx = (d < 0) & (~v0) & d_rel & feasible
    if np.any(lbidx):
        alpha = min(alpha, np.min(-v[lbidx] / d[lbidx]))

    # Ensuring upper-bound feasibility:
    # Positive update, fractions not numericall 1 and update numerically significant.
    v1 = np.abs(emv) < eps
    upidx = (d > 0) & (~v1) & d_rel & feasible
    if np.any(upidx):
        alpha = min(alpha, np.min(emv[upidx] / d[upidx]))

    return alpha


@_COMPILER(
    nb.types.Tuple((nb.f8[:], nb.f8, nb.f8, nb.f8))(
        nb.f8[:, :], nb.f8[:], nb.bool, nb.f8, nb.f8
    ),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def get_descent(
    A: np.ndarray,
    b: np.ndarray,
    do_LM: bool,
    tau: float,
    rtol: float,
) -> tuple[np.ndarray, float, float, float]:
    """Go through the Newton - Levenberg-Marquardt - Steepest-descent ladder until
    a descenting direction is found.

    Parameters:
        A: Square matrix (Hessian of minimized function)
        b: Righ-hand side (-Grad of minimized function)
        do_LM: Flag to skip Newton and go straight to LM.
        tau: Global value for regularization.
        rtol: Relative tolerance for descent criterion.

    Returns:
        A descent direction which solves the system ``Ax=b``, and updated values
        for ``tau_min, tau, tau_max``.

    """
    # Updated bounds for regularization parameter.
    # Squared Frobenius norm. Singular values of A.T @ A are smaller than this value.
    # Beyond this regularization parameter, we are in SD regime.
    Af = A.flatten()
    tau_max = min(float(Af.dot(Af)), 1e6)
    # This value is returned if Newton succeeds in finding a descent direction.
    # I.e., the Matrix is not close to singular, so regularization is not needed.
    tau_min = 0.0

    d = b.copy()  # Default return value is steepest-descent direction.
    g = A.T @ b
    # Relative tolerance for descent criterion.
    descent_tol = rtol * np.linalg.norm(g)

    # Do Newton, if not requested to go directly to LM.
    if not do_LM:
        try:
            d = np.linalg.solve(A, b)
        except:
            do_LM = True
        else:
            # Newton found a descend direction.
            if d.dot(g) > descent_tol * np.linalg.norm(d):
                return d, tau_min, tau, tau_max
            else:
                do_LM = True

    if do_LM:
        B = A.T @ A
        I = np.eye(B.shape[0])

        # Refine lower bound since B is computed.
        tau_min = 1e-6 * np.abs(np.diag(B)).max()
        tau = max(tau_min, tau)

        while tau < tau_max:
            d_LM = np.linalg.solve(B + tau * I, g)
            # LM found descend direction. Assign to d and break.
            if d_LM.dot(g) > descent_tol * np.linalg.norm(d_LM):
                d = d_LM
                break
            # Else increase regularization by 1 order.
            else:
                tau *= 10

    return d, tau_min, tau, tau_max


@_COMPILER(
    nb.types.Tuple((nb.f8[:], nb.int_, nb.int_))(
        nb.f8[:],
        FLASH_RESIDUAL_FUNCTION_TYPE,
        FLASH_JACOBIAN_FUNCTION_TYPE,
        SOLVER_PARAMETERS_TYPE,
        FlashSpec_NUMBA_TYPE,
        nb.int_,
    ),
    cache=NUMBA_CACHE,
)
def npipm_inner(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    params: dict,
    spec: FlashSpec,
    pT_npc_cycle: int,
) -> tuple[np.ndarray, int, int]:
    """Inner function for the NPIPM-solver, suitable for recursion when using the
    pT-flash as a non-linear preconditioning."""
    # Default exit code is maximum iterations reached.
    exitcode = 1

    # region Extracting system and user-given parameters.
    f_dim = int(params["f_dim"]) + 1  # + Slack variable
    n_C = int(params["num_components"])
    n_P = int(params["num_phases"])
    tol = np.float64(params["atol_res"])
    max_iter = int(params["max_iterations"])

    npipm_cc = float(params["npipm_penalty_cc"])
    npipm_neg = float(params["npipm_penalty_neg"])
    npipm_dec = float(params["npipm_slack_decline"])

    ls_ss = float(params["armijo_step_size"])
    ls_dec = float(params["armijo_decline"])

    T_rpc = float(params["rpc_T"])
    rpc_T_chop = float(params["rpc_T_chop"])
    p_rpc = float(params["rpc_p"])
    rpc_p_chop = float(params["rpc_p_chop"])

    tr_delta = float(params["trustregion_delta"])
    tr_f2b = float(params["trustregion_fraction_to_boundary"])

    aa_depth = int(params["anderson_acceleration"])
    aa_reg = float(params["anderson_acceleration_regularization"])
    # endregion

    # region System set-up
    # Get generic argument for easy access to constant parts.
    gen_arg = parse_generic_arg(X0, n_C, n_P, spec)

    n_P1m = n_P - 1  # Number of independent phases.
    n_C1m = n_C - 1  # Number of independent components.
    n_CP = n_C * n_P  # Number of independent partial fractions
    n_F = n_P1m + n_CP  # Number of phase and partial fractions.

    # Declare pT nonlinear preconditioning cycle:
    if (pT_npc_cycle > 0) and (spec != FlashSpec.pT):
        is_pT_npc_cycle = True
        f_dim = n_F + 1
    else:
        is_pT_npc_cycle = False

    # Scaling for right-preconditioning pressure and temperature values for higher
    # flashes. Making pressure and temperature non-dimensional.
    # NOTE we must fix the eps for rcond used in least-squares (Anderson) because it is
    # different depending on JIT mode.
    EPS = float(f_dim * np.finfo(np.float64).eps)
    do_rpc_T = False
    Tidx = -1  # Default value, not used.
    do_rpc_p = False
    pidx = -1

    if spec not in (FlashSpec.pT, FlashSpec.vT):
        Tidx = 0
        do_rpc_T = True

    if spec >= FlashSpec.vT:
        pidx = 0
        # Shift index because T-derivatives come after p-derivatives.
        Tidx += 1
        do_rpc_p = True

    # Generic part of the flash argument vector.
    X_gen = X0[: -f_dim + 1].copy()

    # DOF part of flash argument vector, including slack variable.
    X_i = np.zeros(f_dim)
    X_i[:-1] = X0[-f_dim + 1 :].copy()
    dX_i = np.zeros_like(X_i)

    # Computing initial value for slack variable, as sum of violations of
    # complementarity.
    x, y = parse_xy(X_i[:-1], n_C, n_P)
    X_i[-1] = np.abs(np.sum(y * (1 - np.sum(x, axis=1))))

    def eval_F(X_loc: np.ndarray) -> np.ndarray:
        nu_loc = X_loc[-1]
        _X_loc = X_loc[:-1]
        _x, _y = parse_xy(_X_loc, n_C, n_P)
        try:
            f_loc = F(np.hstack((X_gen, _X_loc)))
        except:
            return np.full((f_dim,), np.nan)
        else:
            if is_pT_npc_cycle:
                f_loc = f_loc[-n_F:]
            return extend_and_regularize_res(
                f_loc, _x, _y, nu_loc, npipm_cc, npipm_neg, npipm_dec
            )

    def eval_DF(X_loc: np.ndarray) -> np.ndarray:
        nu_loc = X_loc[-1]
        _X_Loc = X_loc[:-1]
        _x, _y = parse_xy(_X_Loc, n_C, n_P)
        try:
            J_loc = DF(np.hstack((X_gen, _X_Loc)))
        except:
            return np.full((f_dim, f_dim), np.nan)
        else:
            if is_pT_npc_cycle:
                J_loc = J_loc[-n_F:, -n_F:]
            if do_rpc_T:
                J_loc[:, Tidx] *= T_rpc
            if do_rpc_p:
                J_loc[:, pidx] *= p_rpc
            return extend_and_regularize_jac(
                J_loc, _x, _y, nu_loc, npipm_cc, npipm_neg, npipm_dec
            )

    # endregion

    f_i = eval_F(X_i)

    res_0 = np.linalg.norm(f_i)  # First residual.
    if res_0 <= tol:  # Initial guess is already solution.
        return X0, 0, 0
    if np.any(np.isnan(f_i)):  # Failure in evaluation.
        return X0, 4, 0
    if np.any(np.isinf(f_i)):  # Divergence.
        return X0, 3, 0

    # region Adaptive solver parameters

    atol_frac = 1e-8  # abs. tol for considering fractions to be zero.
    atol_num = 1e-8  # abs. tol for update norm considered numerically relevant.
    rtol_desc = 1e-7  # rel. tol for descent criterion.
    rtol_pert = 1e-6  # rel. tol for perturbing solutions.
    rtol_stag = 1e-3  # rel. tol for detecting stagnation in residuals.
    rtol_cyc = 1e-3  # rel. tol for detecting cycling in residuals.
    y_emg = 1e-3  # Trial phase fraction for emerging phases.

    tau = 0.0  # Starting value for diagonal regularization
    tau_min = 1e-8  # Minimal value.
    tau_max = 1e6  # Maximal value.
    do_LM = False  # Levenberg-Marquardt normalization.
    alpha_max = 1.0  # Maximal step size.
    alpha_min_s = 1e-3  # Scale of alpha max to obtain alpha min

    tr_delta_max = f_dim  # Largest trust-region radius, assuming non-dim variables.
    tr_delta = tr_delta if tr_delta > 0 else tr_delta_max
    tr_delta_min = 1e-4 * tr_delta  # Smallest trust-region radius
    # Descent criteria in line search.
    # NOTE Increase lc_dec_min if too many small updates.
    ls_dec_min = 1e-4  # Smallest decrease required by linesearch
    ls_dec_max = 0.5  # Largest decrease required by linesearch
    ls_ss_min = 0.5  # Smallest line search step size.
    ls_ss_max = 0.99  # Largest line search step size.
    ls_fail_count = 0

    # Relative tolerances for detecting cycling/stagnation. NOTE Tighten for subtle
    # changes when near-critical or near-phase border.
    stag_window = 4  # Last m residuals to detect stagnation.
    is_stagnating = False
    lb_stag = 0.99  # Lower bound for residual slope in case of stagnation.

    max_cycle = 6  # Expecting cycles of size 2 .. max_cycle - 1.
    was_cycling = False
    is_cycling = False
    detected_cycle = 0
    lb_res = 0.0  # Lower bound for residual when cycling or stagnation detected.

    i_cyst_detected = np.inf  # Iteration when cycling ot stagnation detected.
    res_history = np.zeros(max(2 * (max_cycle - 1), stag_window))

    aa_depth_default = 3
    if aa_depth > 0:
        Fk = np.zeros((f_dim, aa_depth))
        Gk = np.zeros((f_dim, aa_depth))
    else:
        Fk = np.zeros((f_dim, aa_depth_default))
        Gk = np.zeros((f_dim, aa_depth_default))
    fk = np.zeros(f_dim)
    fk1m = np.zeros(f_dim)
    gk = np.zeros(f_dim)
    gk1m = np.zeros(f_dim)

    # endregion

    i = 0
    for i in range(pT_npc_cycle if is_pT_npc_cycle else max_iter):
        if X_i[-1] < 0:  # Safety net for slack variable, cannot be negative.
            x, y = parse_xy(X_i[:-1], n_C, n_P)
            X_i[-1] = np.abs(np.sum(y * (1 - np.sum(x, axis=1))))

        f_i = eval_F(X_i)
        res_history = np.roll(res_history, -1)
        res_history[-1] = np.linalg.norm(f_i)
        if res_history[-1] <= tol:
            exitcode = 0
            break

        J_i = eval_DF(X_i)

        if np.any(np.isinf(f_i)):
            exitcode = 3
            break
        if np.any(np.isnan(J_i)) or np.any(np.isnan(f_i)):
            exitcode = 4
            break

        dX_i, _, tau, _ = get_descent(J_i, -f_i, do_LM, tau, rtol_desc)

        if np.any(np.isnan(dX_i)) or np.any(np.isinf(dX_i)):  # Divergence.
            exitcode = 3
            break

        # region Anderson acceleration
        if i > 0:
            col = (i - 1) % Fk.shape[1]
            fk = dX_i
            gk = X_i + dX_i
            Fk[:, col] = fk - fk1m
            Gk[:, col] = gk - gk1m
            fk1m = fk.copy()
            gk1m = gk.copy()

        mk = min(i, aa_depth)
        if mk > 0:
            B = Fk[:, :mk]
            b = fk

            if aa_reg > 0:
                b = B.T @ b
                B = B.T @ B + aa_reg * np.eye(B.shape[1])

            h = np.linalg.lstsq(B, b, rcond=EPS)[0]
            dX_i = gk - np.dot(Gk[:, :mk], h) - X_i
        # endregion

        # Apply p-T chopping before trust region.
        if do_rpc_T:
            dT = dX_i[Tidx]
            dX_i[Tidx] = np.sign(dT) * min(np.abs(dT), rpc_T_chop)
        if do_rpc_p:
            dp = dX_i[pidx]
            dX_i[pidx] = np.sign(dp) * min(np.abs(dp), rpc_p_chop)

        # region Trust-region and feasibility scaling. Initializing step size.
        d_norm = np.linalg.norm(dX_i)
        if d_norm > tr_delta:
            # NOTE we scale dX not alpha. Treating trust-region as starting point
            # for further step size exploration. Otherwise the step size gets very
            # soon very small.
            dX_i *= tr_delta / d_norm
        f2b = _feasible_fractions_scale(
            X_i[-(n_F + 1) : -1], dX_i[-(n_F + 1) : -1], atol_num, atol_frac
        )
        if f2b < 1.0:
            alpha = alpha_max * tr_f2b * f2b
        else:
            alpha = alpha_max
        # endregion

        # Rescale pT to physical units.
        if do_rpc_T:
            dX_i[Tidx] *= T_rpc
        if do_rpc_p:
            dX_i[pidx] *= p_rpc

        # region Armijo line search.
        pot_i = f_i.dot(f_i) * 0.5
        alpha_j = alpha
        f_i_j = f_i
        pot_i_j = pot_i
        ls_i = 0
        ls_failed = True
        alpha_min = alpha_min_s * alpha_max

        while alpha_j >= alpha_min:
            alpha_j = ls_ss**ls_i * alpha
            ls_i += 1
            X_i_j = X_i + alpha_j * dX_i

            # NOTE we allow eval to fail in line search to not break the main
            # algorithm.
            f_i_j = eval_F(X_i_j)
            if np.any(np.isnan(f_i_j) | np.isinf(f_i_j)):
                continue

            pot_i_j = f_i_j.dot(f_i_j) * 0.5

            if pot_i_j <= (1 - 2 * ls_dec * alpha_j) * pot_i:
                ls_failed = False
                break

        # endregion

        # Keep track of numerically relevant updates.
        if (
            alpha_j < alpha_min or ls_failed
        ):  # or np.linalg.norm(alpha * dX_i) < atol_num:
            ls_failed = True
            ls_fail_count += 1
            reduction = 0.0
        else:
            ls_fail_count = 0
            # Actual reduction divided by predicted linear reduction
            reduction = (pot_i - pot_i_j) / (2 * ls_dec * alpha_j * pot_i)

        # Due to MPCC nature, we accept even bad steps and allow the algorithm
        # to climb out of bad basins.
        alpha = max(alpha_j, alpha_min)
        X_i += alpha * dX_i

        # region Adapt solver parameters

        # If actual reduction is large enough, decrease tau.
        if reduction >= 1.8:
            # tau = max(tau_min, tau * 0.1)
            tr_delta = min(1.5 * tr_delta, tr_delta_max)
        # If actual reduction is too small, increase tau.
        elif reduction <= 1.0 or ls_failed:
            # tau = min(tau_max, tau * 2.0)
            tr_delta = max(0.5 * tr_delta, tr_delta_min)

        if alpha < 1e-3 * alpha_max:  # or tau > 0.5 * tau_max:
            ls_dec = max(ls_dec * 0.5, ls_dec_min)
        elif alpha > 0.9 * alpha_max:
            ls_dec = min(ls_dec * 1.1, ls_dec_max)

        # endregion

        # region Cycling and stagnation detection.
        if i > res_history.size:
            # Stagnation check
            r_stag = res_history[-stag_window:]
            rtol = rtol_stag * r_stag.max()
            slope_stag = (r_stag[-1] / r_stag[0]) ** (1 / (stag_window - 1))

            if (r_stag.max() - r_stag.min()) < rtol and slope_stag > lb_stag:
                # Keep original iteration in case earlier detected.
                i_cyst_detected = i if not is_stagnating else i_cyst_detected
                is_stagnating = True
                lb_res = r_stag.min() * 0.99
            else:
                is_stagnating = False

            # First, adapt solver params to stagnation.
            if is_stagnating:
                r = np.linalg.norm(dX_i) / np.linalg.norm(X_i)
                # Stagnation and low progress -> mobilize
                if (r < 1e-1) and res_history[-1] < res_0:
                    if i_cyst_detected == i:
                        alpha_max *= 2.0
                    elif i == i_cyst_detected + 3:
                        alpha_max *= 4.0

                    tr_f2b = min(tr_f2b * 1.01, 0.999)
                    ls_dec = max(ls_dec * 0.5, ls_dec_min)
                    aa_depth = max(aa_depth_default, aa_depth)
                    do_LM = False
                    # do_LM = False if not ls_failed else True
                # elif res_history[-1] < res_0:
                # Stagnations far from convergence, likely do divergence which is
                # kept at bay by trust-region or caps -> stabilize
                elif res_history[-1] >= res_0:
                    do_LM = True
                    aa_depth = int(params["anderson_acceleration"])
            # If not stagnating, check for cycling and see if stagnation mode can
            # be reverted.
            else:
                was_cycling = is_cycling
                # Look for cycles with period 2 to max cycles - 1s.
                is_cycling = False
                for c in range(2, max_cycle):
                    res_2c = res_history[-2 * c :]
                    rtol = np.max(np.abs(res_2c)) * rtol_cyc

                    if np.all(np.abs(res_2c[c:] - res_2c[:c]) <= rtol):
                        is_cycling = True
                        # Keep original values from first cycle detection.
                        if not was_cycling:
                            detected_cycle = c
                            i_cyst_detected = i
                        # But make new lower bound in any case.
                        lb_res = res_2c.min() * 0.99
                        break

                # If residual did not change due to failure in line search,
                # reject false cycling detection.
                if ls_fail_count >= 2:
                    is_cycling = False

                # If also not cycling, revert parameters.
                if not (is_cycling or was_cycling or ls_failed):
                    tr_f2b = float(params["trustregion_fraction_to_boundary"])
                    ls_ss = float(params["armijo_step_size"])
                    # ls_dec = float(params["armijo_decline"])
                    aa_depth = int(params["anderson_acceleration"])
                    do_LM = False
                    rtol_desc = 1e-7
                # If cycling detected, make line search more aggressive.
                elif is_cycling and not was_cycling:
                    ls_ss = max(ls_ss_min, 0.5 * ls_ss)
                    ls_dec = min(ls_dec * 1.5, ls_dec_max)
                # If no cycle anymore detected.
                elif not is_cycling and was_cycling:
                    # Check if cycling really broken. Restore parameters.
                    if np.all(np.diff(res_history[-detected_cycle - 1 :]) < 0) and (
                        res_history[-1] < lb_res
                    ):
                        ls_ss = min((ls_ss + ls_ss_max) * 0.5, ls_ss_max)
                        ls_dec = (ls_dec + ls_dec_max) * 0.5
                        aa_depth = int(params["anderson_acceleration"])
                        do_LM = False
                    else:
                        is_cycling = True
        # endregion

        i_diff = i - i_cyst_detected

        # region Special measures for persistent failures.
        if is_cycling or is_stagnating or ls_fail_count > 0:
            # First line of action: check special cases.

            # Issues are often observed due to extended fractions being stuck at too low
            # values when a phase vanished. Try to break free by perturbing using feed
            # fractions. Indicator: Only isofugacity constraints have large residual.
            idx = np.zeros(f_dim, dtype=np.bool)
            idx[-(n_F + 1) : -(n_P + n_C1m + 1)] = True
            res_part = np.linalg.norm(f_i_j[~idx])
            x, y = parse_xy(X_i[:-1], n_C, n_P)

            # First, make sure fractions are feasible.
            y[y > 1] = 1.0
            y[y < 0] = 0.0
            for j in range(n_P):
                xj = x[j, :]
                xj[xj > 1] = 1.0
                xj[xj < 0] = 0.0
                if y[j] > 0:
                    xj /= xj.sum()
                x[j, :] = xj

            # Perturb every n-th iteration, with n being maximal cycle detected.
            if (res_part < tol) and (i_diff % (max_cycle - 1) == 0):
                z = gen_arg[3]
                for j in range(n_P):
                    xj = x[j, :]
                    # Perturb extended fractions where phase absent.
                    # NOTE Perturbing phase fractions is tricky as it often has
                    # a deteriorating effect. Requires more thinking.
                    # if xj.sum() <= 1.0 - 1e-3:
                    if y[j] < atol_frac:
                        xj = (xj + z + 1 / n_C) / 3.0
                        # Keep fractions feasible.
                        sxj = xj.sum()
                        if sxj > 1.0:
                            xj /= sxj
                    x[j, :] = xj

            # If was_cycling and one of the y is small, likely stuck at border.
            # Relaxe fraction-to-boundary-rule
            if was_cycling and np.any(y < atol_frac):
                tr_f2b = 0.999

            # If line search failed twice in a row, try to hop out of an unfavorable
            # area.
            if ls_fail_count == 2:
                alpha_max *= 2.0
                ls_dec = max(ls_dec * 0.5, ls_dec_min)

            # Check if phase is emergent: y is 0  and sum x approaching 1.
            # Perturb system by adding small mass to emerging phase.
            for j in range(n_P):
                sxj = np.sum(x[j])
                yj = y[j]
                if np.abs(yj) <= atol_frac and sxj > 0.95:
                    y_c = y_emg / (n_P - 1)  # Mass taken from other phases
                    y = np.maximum(np.zeros(n_P), y - y_c)
                    y[j] = y_emg
                    x[j] /= sxj
                    # NOTE consider for stability to adapt other partial fractions to
                    # not violate mass constraints.
                    # z_t = (gen_arg[3] - y_eps * x[j]) / (1.0 - y_eps)

            X_i[-(n_F + 1) : -1] = np.hstack((y[1:], x.flatten()))

            # Second line of action: Analyze system (expensive - if failures persist).
            if (
                (ls_fail_count >= 3)
                or (is_cycling and was_cycling and i_diff > detected_cycle)
                or (is_stagnating and i_diff >= stag_window)
            ):
                if ls_fail_count > 5:  # Abort algorithm as stationary point.
                    exitcode = 2
                    break

                nFrob_J = float(np.linalg.norm(J_i.flatten()))  # Frobenius-norm
                g = J_i.T @ f_i_j  # Gradient of least-squares model.
                # NOTE: 1-norm is slightly faster, but often 1 order of magnitude larger
                # if ill-conditioned. Use 1e5 for other norms
                ill_cond = np.linalg.cond(J_i, p=1) >= 1e6
                stationary = g.dot(g) < tol * max(1.0, nFrob_J) ** 2

                if ill_cond:  # If ill-conditioned -> stabilize using LM
                    # NOTE: The rational behind activating LM is to get faster to
                    # steepest descent and a more robust descent direction.
                    tr_delta = max(tr_delta * 0.5, tr_delta_min)
                    if is_stagnating and res_history[-1] < res_0:
                        do_LM = False
                    else:
                        do_LM = True
                    # do_LM = True if not is_stagnating else False
                    if ls_fail_count >= 4:
                        alpha_min_s = 1e-7
                        rtol_desc = 1e-10
                else:  # Else mobilize
                    aa_depth = max(aa_depth_default, aa_depth)
                    if ls_fail_count == 2:
                        alpha_max *= 2.0
                        ls_dec = max(ls_dec * 0.5, ls_dec_min)

                # We check first, if we are at a stationary point where we lost
                # all sensitivy due to small gradient. If yes, perturb slightly.
                if stationary or ls_fail_count == 3:
                    # NOTE tighten to 1e-7 if perturbation leads to worsening.
                    X_p = (
                        rtol_pert
                        * np.maximum(np.abs(X_i), 1.0)
                        * ((1.0 - atol_num) * np.random.rand(f_dim) + atol_num)
                    )
                    X_p[-1] = 0.0
                    X_i += X_p
                    x, y = parse_xy(X_i[:-1], n_C, n_P)
                    y[y < 0] = 0.0
                    y[y > 1] = 1.0
                    xf = x.flatten()  # Flatten for numba.
                    xf[xf < 0] = 0.0
                    xf[xf > 1] = 1.0
                    x = xf.reshape((n_P, n_C))
                    xs = x.sum(axis=1)
                    for j, s in enumerate(xs):
                        if s > 1:
                            x[j, :] /= s + atol_num
                    X_i[-(n_F + 1) : -1] = np.hstack((y[1:], x.flatten()))
        else:
            alpha_min_s = 1e-3
            alpha_max = 1.0
        # endregion

    if np.any(np.isnan(X_i)) or np.any(np.isinf(X_i)):
        # Return initial guess back to not break subsequent code.
        X_i[:-1] = X0[-f_dim + 1 :].copy()
        assert exitcode > 1, "Expecting exitcode > 1 in case of failure."

    return np.hstack((X_gen, X_i[:-1])), exitcode, i


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
    # return npipm_inner(X0, F, DF, params, spec, 0)
    # Use full-algorithm on pT flash.
    if spec in (FlashSpec.pT, FlashSpec.vT):
        return npipm_inner(X0, F, DF, params, spec, 0)

    pT_npc_iter = int(params["pT_npc_iterations"])
    # If no non-linear preconditioning is requested, go with the full algorithm
    if pT_npc_iter == 0:
        return npipm_inner(X0, F, DF, params, spec, 0)

    exitcode = 1  # Default return value.
    f_dim = int(params["f_dim"])
    n_C = int(params["num_components"])
    n_P = int(params["num_phases"])
    tol = np.float64(params["atol_res"])
    max_iter = int(params["max_iterations"])

    T_rpc = float(params["rpc_T"])
    rpc_T_chop = float(params["rpc_T_chop"])
    p_rpc = float(params["rpc_p"])
    rpc_p_chop = float(params["rpc_p_chop"])

    n_P1m = n_P - 1  # Number of independent phases.
    n_C1m = n_C - 1  # Number of independent components.
    n_CP = n_C * n_P  # Number of independent partial fractions
    n_F = n_P1m + n_CP  # Number of phase and partial fractions.

    atol_frac = 1e-10  # abs. tol for considering fractions to be zero.

    Tidx = 0

    if spec > FlashSpec.vT:
        pidx = 0
        do_rpc_p = True
        # Shift index because T-derivatives come after p-derivatives.
        Tidx += 1
    else:
        do_rpc_p = False
        pidx = -1

    gen_arg = parse_generic_arg(X0, n_C, n_P, spec)
    X_gen = X0[:-f_dim].copy()
    X_i = X0[-f_dim:].copy()
    dX_i = np.zeros_like(X_i)

    def eval_F(X_loc: np.ndarray) -> np.ndarray:
        try:
            return F(np.hstack((X_gen, X_loc)))
        except:
            return np.full((f_dim,), np.nan)

    def eval_DF(X_loc: np.ndarray) -> np.ndarray:
        try:
            J_loc = DF(np.hstack((X_gen, X_loc)))
        except:
            J_loc = np.full((f_dim, f_dim), np.nan)

        J_loc[:, Tidx] *= T_rpc
        if do_rpc_p:
            J_loc[:, pidx] *= p_rpc
        return J_loc

    f_i = eval_F(X_i)

    res_0 = np.linalg.norm(f_i)  # First residual.
    if res_0 <= tol:  # Initial guess is already solution.
        return X0, 0, 0
    if np.any(np.isnan(f_i)):  # Failure in evaluation.
        return X0, 4, 0
    if np.any(np.isinf(f_i)):  # Divergence.
        return X0, 3, 0

    res_history = np.zeros(10)
    i = 0
    for i in range(1, max_iter + 1):
        # Nonlinear-preconditioning using pT flash.
        # pT-flash updates only phase and partial fractions.
        X_npc, e_npc, _ = npipm_inner(
            np.hstack((X_gen, X_i)), F, DF, params, spec, pT_npc_iter
        )
        if e_npc == 0:  # Accept pT update if converged.
            X_i[-n_F:] = X_npc[-n_F:]
        else:
            # Check if global residual is decreased and accept.
            if np.linalg.norm(eval_F(X_npc[-f_dim:])) <= res_0:
                X_i[-n_F:] = X_npc[-n_F:]
            else:
                # Check if phase is incumbent: y close to 0  and sum x close to 1
                # Give it a push in that direction and trigger full algorithm.
                phase_incumbent = False
                x, y = parse_xy(X_i, n_P, n_C)

                y_eps = 1e-3  # trial fraction.
                y_c = y_eps / (n_P - 1)  # complementary mass reduced from other phases.
                for j in range(n_P):
                    sxj = np.sum(x[j])
                    yj = y[j]
                    if np.abs(yj) <= atol_frac and sxj > 0.9:
                        phase_incumbent = True
                        y -= y_c
                        y = np.maximum(np.zeros(n_P), y - y_c)
                        y[j] = y_eps
                        x[j] /= sxj
                        # Mass distributed accross other phases
                        # z_t = (gen_arg[3] - y_eps * x[j]) / (1.0 - y_eps)

                if phase_incumbent:
                    X_i[-n_F:] = np.hstack((y[1:], x.flatten()))
                    X_t, e_t, n_t = npipm_inner(
                        np.hstack((X_gen, X_i)), F, DF, params, spec, 0
                    )
                    if e_t == 0:
                        X_i = X_t[-f_dim:]
                        exitcode = 0
                        i += n_t
                        break

        f_i = eval_F(X_i)
        res_history = np.roll(res_history, -1)
        res_history[-1] = np.linalg.norm(f_i)
        if res_history[-1] <= tol:
            exitcode = 0
            break

        J_i = eval_DF(X_i)

        if np.any(np.isinf(f_i)):
            exitcode = 3
            break
        if np.any(np.isnan(J_i)) or np.any(np.isnan(f_i)):
            exitcode = 4
            break

        # Only non-isothermal:Update only T.
        if spec < FlashSpec.vT:
            # NOTE: Least squares steepest descent update.
            dX_i[Tidx] = -f_i[0] * J_i[0, Tidx]
        # Additional isochoric specifications require a pressure update as well.
        else:
            assert False, "WIP"

        # Scale update back to physical dimensions.
        dT = dX_i[Tidx]
        dX_i[Tidx] = np.sign(dT) * min(np.abs(dT), rpc_T_chop) * T_rpc
        if do_rpc_p:
            dp = dX_i[pidx]
            dX_i[pidx] = np.sign(dp) * min(np.abs(dp), rpc_p_chop) * p_rpc

        X_i += dX_i

        # If for isobaric flashes the temperature change is below 1 Kelvin, and
        # the pT-npc-cycle succeeded, we are likely very close to the solution.
        # Trigger Newton. TODO loosen criteria.
        if spec < FlashSpec.vT and np.abs(dX_i[Tidx]) < 1.0 and e_npc == 0:
            X_t, e_t, n_t = npipm_inner(np.hstack((X_gen, X_i)), F, DF, params, spec, 0)
            if e_t == 0:
                X_i = X_t[-f_dim:]
                exitcode = 0
                i += n_t
                break

    if np.any(np.isnan(X_i)) or np.any(np.isinf(X_i)):
        # Return initial guess back to not break subsequent code.
        X_i = X0[-f_dim:].copy()
        assert exitcode > 1, "Expecting exitcode > 1 in case of failure."

    return np.hstack((X_gen, X_i)), exitcode, i
