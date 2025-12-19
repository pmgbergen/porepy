"""Module containing numba-compiled implementations of the NPIPM using the Newton
algorithm and Armijo line search.

To be used by the compiled flash for parallelized computations."""

from __future__ import annotations

from typing import Callable, Literal, TypeAlias

import numba as nb
import numpy as np

from ..._numba_interface import NUMBA_CACHE, NUMBA_FAST_MATH, njit
from ..abstract_flash import FlashSpec
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
    "heavy_ball",
    "appleyard_chop",
    "trustregion_tau",
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
        "heavy_ball": 0.0,
        "appleyard_chop": 0.0,
        "trustregion_tau": 0.995,
        "anderson_acceleration": 0,
        "anderson_acceleration_regularization": 1e-7,
    },
    **DEFAULT_ARMIJO_LINE_SEARCH_PARAMS,  # type:ignore[arg-type,dict-item]
)
"""Default solver parameters required by the :func:`npipm_solver`.

- ``'npipm_u1': 1.`` penalty for violating complementarity
- ``'npipm_u2': 1.`` penalty for violating negativity of fractions
- ``'npipm_eta': 0.5`` linear decline in slack variable
- ``'heavy_ball': 0.`` If True (non-zero), a heavy-ball momentum technique is
  applied, adding the update from the previous iteration multiplied with the given
  factor to the current update. This can help convergence when iterations stall.
- ``'appleyard_chop': 0.0`` if non-zero, chopping the update for phase fractions
  and saturations to allow maximally this value.
- ``''trustregion_tau': 0.995`` scaling factor for the fraction-to-boundary
  rule.

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
    nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.int_, nb.int_, nb.int_),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _trust_region_cap(
    X: np.ndarray, DX: np.ndarray, tau: float, n_F: int, n_C: int, n_P: int
) -> np.ndarray:
    """Helper function to apply the fraction-to-boundary rule to the update
    direction.

    Parameters:
        X: Current solution vector, shape ``(f_dim + 1,)``.
        DX: Update direction, shape ``(f_dim + 1,)``.
        tau: Trust-region parameter in ``(0, 1)``.
        n_F: Number of fractional unknowns
        n_C: Number of components.
        n_P: Number of phases.

    Returns:
        The scaled increment ensuring no update violates trusted region [0, 1].

    """
    EPS = 1e-14
    n_PC = n_P * n_C
    # Where update is negative, i.e. at risk of violating lower bound 0.
    # Ignore small updates to already zero fractions to avoid division by zero
    # and a step size of zero.
    dfracs = DX[-(n_F + 1) : -1]
    fracs = X[-(n_F + 1) : -1]
    small = np.abs(dfracs) <= EPS
    emfracs = 1.0 - fracs

    fracs_0 = np.abs(fracs) <= EPS
    fracs_1 = np.abs(emfracs) <= EPS

    neg_d = dfracs < 0.0
    pos_d = dfracs > 0.0

    # Violations of lower bound 0
    aidx = neg_d & (~(fracs_0 | small))
    a = min(1.0, tau * np.min(fracs[aidx] / -dfracs[aidx])) if np.any(aidx) else 1.0

    # Violations of upper bound 1, same logic.
    bidx = pos_d & (~(fracs_1 | small))
    b = min(1.0, tau * np.min(emfracs[bidx] / dfracs[bidx])) if np.any(bidx) else 1.0

    c = min(a, b)

    dfracs_new = dfracs * c

    # Cancel the update where fractions close to 0 and 1
    dfracs_new[fracs_0 & neg_d] = 0.0
    dfracs_new[fracs_1 * pos_d] = 1.0

    # Scale update further down if violations of unity constraints would occur.
    fracs_new = fracs + dfracs_new
    xf = fracs_new[-n_PC:]

    # In case DX contains also updates to non-fractional variables.
    DX *= c
    DX[-(n_F + 1) : -1] = dfracs_new

    return DX


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
    EPS = 1e-14

    # Extracting solver parameters.
    f_dim = int(params["f_dim"])
    n_C = int(params["num_components"])
    n_P = int(params["num_phases"])
    tol = float(params["tolerance"])
    max_iter = int(params["max_iterations"])
    rho = float(params["armijo_rho"])
    kappa = float(params["armijo_kappa"])
    max_iter_armijo = int(params["armijo_max_iterations"])
    u1 = float(params["npipm_u1"])
    u2 = float(params["npipm_u2"])
    eta = float(params["npipm_eta"])
    appleyard = float(params["appleyard_chop"])
    heavy_ball = float(params["heavy_ball"])
    tau = float(params["trustregion_tau"])
    anderson = int(params["anderson_acceleration"])
    anderson_reg = float(params["anderson_acceleration_regularization"])

    # L-scheme for stabilization when oscillating
    L_scheme = 0.0

    # Number of fractional unknowns
    n_F = n_P * n_C + n_P - 1
    if spec >= FlashSpec.vh:
        n_F += n_P - 1  # saturations

    X_i = np.zeros(X0.shape[0] + 1)
    X_i[:-1] = X0.copy()

    # Computing initial value for slack variable, as sum of violations of
    # complementarity.
    xf, yf = parse_xy(X_i[:-1], n_C, n_P)
    X_i[-1] = np.abs(np.sum(yf * (1 - np.sum(xf, axis=1))))

    DX_i = np.zeros_like(X_i)
    DX_i1m = np.zeros_like(X_i)

    # Complete system size including slack equation
    matrix_rank = f_dim + 1
    # NOTE rcond is the limit to cutting off singular values.
    # This has quite large effects on the robustness of the flash in the
    # vh case for example, which is not yet fully understood.
    # NOTE also, the default value in numba is machine precision, while
    # with no-jit (pure numpy) is shape[0] * eps.
    # The latter is chosen and set to avoid differences between jit and
    # no-jit computations.
    rcond = matrix_rank * EPS

    def eval_F(X_l: np.ndarray) -> np.ndarray:
        f_l = F(X_l[:-1])
        xf_l, yf_l = parse_xy(X_l[:-1], n_C, n_P)
        f_npipm_l = extend_and_regularize_res(f_l, xf_l, yf_l, X_l[-1], u1, u2, eta)
        return f_npipm_l

    def eval_DF(X_l: np.ndarray) -> np.ndarray:
        df_l = DF(X_l[:-1])
        xf_l, yf_l = parse_xy(X_l[:-1], n_C, n_P)
        df_npipm_l = extend_and_regularize_jac(df_l, xf_l, yf_l, X_l[-1], u1, u2, eta)
        return df_npipm_l

    # Tracking residual history for detecting cycles. We expect cycles of 2 to 5.
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

    # Adding default value to keep track of residuals for dynamic acceleration switch.
    default_anderson = 3
    if anderson > 0:
        Fk = np.zeros((matrix_rank, anderson))
        Gk = np.zeros((matrix_rank, anderson))
    else:
        Fk = np.zeros((matrix_rank, default_anderson))
        Gk = np.zeros((matrix_rank, default_anderson))
    fk1m = np.zeros(matrix_rank)
    gk1m = np.zeros(matrix_rank)

    try:
        f_i = eval_F(X_i)
    except:  # whatever happens, residual evaluation is faulty
        return X_i, 3, num_iter

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

            if LM_mode:
                b = A.T @ b
                A = A.T @ A

            if L_scheme > 0.0:
                A += L_scheme * np.eye(A.shape[0])

            if steepest_descent:
                DX_i[-matrix_rank:] += b
            else:
                try:
                    DX_i[-matrix_rank:] = np.linalg.solve(A, b)
                except:
                    try:
                        DX_i[-matrix_rank:] = np.linalg.lstsq(A, b, rcond=rcond)[0]
                    except:
                        # This means the linear solver failed.
                        exitcode = 5
                        break

            if np.any(np.isnan(DX_i)) or np.any(np.isinf(DX_i)):
                exitcode = 2
                break

            # Trust region capping to avoid shooting out of boundaries for fractions.
            # strictly necessary before going to Armijo (other residual evaluations).
            DX_i = _trust_region_cap(X_i, DX_i, tau, n_F, n_C, n_P)

            # Appleyard chop to update.
            if appleyard > 0.0:
                dys = DX_i[-(1 + n_F) : -(1 + n_C * n_P)]
                dys[dys > appleyard] = appleyard
                DX_i[-(1 + n_F) : -(1 + n_C * n_P)] = dys

            # region Armijo line search.
            pot_i = np.sum(f_i * f_i) * 0.5
            rho_i = rho

            for j in range(0, max_iter_armijo + 1):
                rho_i = rho**j
                X_i_j = X_i + rho_i * DX_i

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
                if rho_i * np.linalg.norm(DX_i) < 1e-5:
                    break

            # endregion

            DX_i *= rho_i

            # region Anderson acceleration
            if i > 0:
                col = (i - 1) % Fk.shape[1]
                fk = DX_i[-matrix_rank:]
                gk = (X_i + DX_i)[-matrix_rank:]
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

                DX_i[-matrix_rank:] = gk - np.dot(Gk[:, :mk], g) - X_i[-matrix_rank:]

            # endregion

            if heavy_ball > 0:
                # Cap update to avoid instability.
                delta_heavy = min(heavy_ball, 1.0 / (1.0 + np.linalg.norm(DX_i1m)))
                DX_i_ = DX_i.copy()
                DX_i += delta_heavy * DX_i1m
                DX_i1m = DX_i_

            # Apply update
            X_i += _trust_region_cap(X_i, DX_i, tau, n_F, n_C, n_P)

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

            # Detect cycling.
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
                    idx = np.zeros(matrix_rank, dtype=np.bool)
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
                        rho = (rho + float(params["armijo_rho"])) * 0.5
                        kappa = (kappa + float(params["armijo_kappa"])) * 0.5
                        max_iter_armijo = int(params["armijo_max_iterations"])
                        anderson = int(params["anderson_acceleration"])
                    # If residual decreases but unsure about cycling, activate
                    # stabilization and acceleration techniques.
                    elif not LM_mode:
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
                            yf[j] = 0.0
                            xf[j, :] = xj
                            is_perturbed = True

                    cond = np.linalg.cond(df_i)

                    # NOTE Perturbing the phase fractions too seems to have negative
                    # effects on convergence.
                    # We set it to zero where the extended fractions are obviously
                    # far from summing to one and re-normalize.
                    if is_perturbed:
                        yf /= np.sum(yf)
                        X_i[-(n_P - 1 + n_C * n_P + 1) : -(n_C * n_P + 1)] = yf[1:]

                        X_i[-n_C * n_P - 1 : -1] = xf.flatten()

                        # Reset changes to parameters introduced by cycle-breaking.
                        is_cycling = False
                        L_scheme = 0.0
                        rho = float(params["armijo_rho"])
                        kappa = float(params["armijo_kappa"])
                        LM_mode = False
                        # anderson = int(params["anderson_acceleration"])
                    elif cond > 1e5 and not LM_mode:
                        L_scheme = 10.0 if cond > 1e6 else 1.0
                        LM_mode = True
                        if anderson == 0:
                            anderson = default_anderson

    return X_i[:-1], exitcode, num_iter
