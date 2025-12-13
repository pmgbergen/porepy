"""Module containing numba-compiled implementations of the NPIPM using the Newton
algorithm and Armijo line search.

To be used by the compiled flash for parallelized computations."""

from __future__ import annotations

from typing import Callable, Literal, TypeAlias

import numba as nb
import numpy as np

from ..._numba_interface import NUMBA_CACHE, NUMBA_FAST_MATH, njit
from ..abstract_flash import FlashSpec
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
    "npipm_heavy_ball",
    "npipm_appleyard_chop",
    "npipm_trustregion_tau",
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
        "npipm_heavy_ball": 0.0,
        "npipm_appleyard_chop": 0.0,
        "npipm_trustregion_tau": 0.995,
    },
    **DEFAULT_ARMIJO_LINE_SEARCH_PARAMS,  # type:ignore[arg-type,dict-item]
)
"""Default solver parameters required by the :func:`npipm_solver`.

- ``'npipm_u1': 1.`` penalty for violating complementarity
- ``'npipm_u2': 1.`` penalty for violating negativity of fractions
- ``'npipm_eta': 0.5`` linear decline in slack variable
- ``'npipm_heavy_ball': 0.`` if True (non-zero), a heavy-ball momentum technique is
  applied to the line-search, adding the update from the previous iteration with some
  down-scaling to the current update.
- ``'npipm_appleyard_chop': 0.0`` if non-zero, chopping the update for phase fractions
  and saturations to allow maximally this value.
- ``''npipm_trustregion_tau': 0.995`` scaling factor for the fraction-to-boundary
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
def _extend_and_regularize_res(
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
def _extend_and_regularize_jac(
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
    npnc = n_P * n_C

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
    d_slack_expanded[-(npnc + n_P) : -(1 + npnc)] = d_slack[1:n_P] - d_slack[0]

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
    nb.types.Tuple((nb.f8[:, :], nb.f8[:]))(nb.f8[:], nb.int_, nb.int_),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def _parse_xy(
    X_gen: np.ndarray, ncomp: int, nphase: int
) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to extract phase compositions and fractions from generic
    argument.

    Parameters:
        Xgen: Generic argument, shape ``(nphase * ncomp + nphase,)``.
        ncomp: Number of components.
        nphase: Number of phases.

    Returns:
        Tuple containing:

        - Phase compositions, shape ``(nphase, ncomp)``.
        - Phase fractions, shape ``(nphase,)``.

    """
    npnc = nphase * ncomp
    x = X_gen[-npnc:].copy().reshape((nphase, ncomp))
    # Phase fractions
    y = np.zeros(nphase)
    y[1:] = X_gen[-(npnc + nphase - 1) : -npnc]
    y[0] = 1.0 - y.sum()

    return x, y


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
    appleyard = float(params["npipm_appleyard_chop"])
    heavy_ball = int(params["npipm_heavy_ball"])
    tau = float(params["npipm_trustregion_tau"])

    # Number of fractional unknowns
    n_F = n_P * n_C + n_P - 1
    if spec >= FlashSpec.vh:
        n_F += n_P - 1  # saturations

    X_i = np.zeros(X0.shape[0] + 1)
    X_i[:-1] = X0.copy()

    # Computing initial value for slack variable, as sum of violations of
    # complementarity.
    gen = _parse_xy(X_i[:-1], n_C, n_P)
    xf = gen[0]
    yf = gen[1]
    nu = np.abs(np.sum(yf * (1 - np.sum(xf, axis=1))))

    # numba does not support stacking with inhomogeneous sequence of array and float.

    X_i[-1] = nu
    DX_i = np.zeros_like(X_i)
    DX_im1 = DX_i.copy()

    # complete system size including slack equation
    matrix_rank = f_dim + 1

    def eval_F(X_local: np.ndarray) -> np.ndarray:
        xf_local, yf_local = _parse_xy(X_local[:-1], n_C, n_P)
        f_local = F(X_local[:-1])
        f_npipm_local = _extend_and_regularize_res(
            f_local, xf_local, yf_local, X_local[-1], u1, u2, eta
        )
        return f_npipm_local

    def eval_DF(X_local: np.ndarray) -> np.ndarray:
        xf_local, yf_local = _parse_xy(X_local[:-1], n_C, n_P)
        df_local = DF(X_local[:-1])
        df_npipm_local = _extend_and_regularize_jac(
            df_local, xf_local, yf_local, X_local[-1], u1, u2, eta
        )
        return df_npipm_local

    # Tracking residual history for detecting cycles.
    res_history = np.zeros(10)
    was_cycling = False
    is_cycling = False
    iter_cycle_detected = 0
    detected_cycle = 0

    # The strategy when detecting cycling is to make the line search more aggressive.
    rho_0 = rho
    N_armijo = max_iter_armijo
    kappa_0 = kappa

    try:
        f_i = eval_F(X_i)
    except Exception:  # whatever happens, residual evaluation is faulty
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

            # try:
            df_i = eval_DF(X_i)
            DX_im1 = DX_i.copy()
            # except Exception:  # whatever happens, Jacobian assembly is faulty
            #     exitcode = 4
            #     break

            # try:
            if np.linalg.matrix_rank(df_i) == matrix_rank:
                DX_i[-matrix_rank:] = np.linalg.solve(df_i, -f_i)
            else:
                # NOTE rcond is the limit to cutting off singular values.
                # This has quite large effects on the robustness of the flash in the
                # vh case for example, which is not yet fully understood.
                # NOTE also, the default value in numba is machine precision, while
                # with no-jit (pure numpy) is shape[0] * eps.
                # The latter is chosen and set to avoid differences between jit and
                # no-jit computations.
                rcond = df_i.shape[0] * EPS
                DX_i[-matrix_rank:] = np.linalg.lstsq(df_i, -f_i, rcond=rcond)[0]
            # except Exception:
            #     # This means the linear solver failed.
            #     exitcode = 5
            #     break

            if np.any(np.isnan(DX_i)) or np.any(np.isinf(DX_i)):
                exitcode = 2
                break

            # Ensure that fractions remain in [0,1] after update via
            # fraction-to-boundary rule.
            dfracs = DX_i[-(n_F + 1) : -1]
            fracs = X_i[-(n_F + 1) : -1]
            # Where update is negative, i.e. at risk of violating lower bound 0.
            # Ignore small updates to already zero fractions to avoid division by zero
            # and a step size of zero.
            aidx = (dfracs < 0) & (~((np.abs(fracs) <= EPS) & (np.abs(dfracs) <= EPS)))
            # Violations of lower bound 0
            a = (
                min(1, tau * np.min(fracs[aidx] / -dfracs[aidx]))
                if np.any(aidx)
                else 1.0
            )
            # Violations of upper bound 1, same logic.
            bidx = (dfracs > 0) & (~((fracs >= 1 - EPS) & (np.abs(dfracs) <= EPS)))
            b = (
                min(1, tau * np.min((1 - fracs)[bidx] / dfracs[bidx]))
                if np.any(bidx)
                else 1.0
            )

            DX_i *= min(a, b)

            # Appleyard chop to update.
            if appleyard > 0.0:
                dys = DX_i[-(1 + n_F) : -(1 + n_C * n_P)]
                dys[dys > appleyard] = appleyard
                DX_i[-(1 + n_F) : -(1 + n_C * n_P)] = dys

            # Armijo line search.
            pot_i = np.sum(f_i * f_i) * 0.5
            rho_i = rho_0

            for j in range(0, N_armijo + 1):
                rho_i = rho_0**j
                X_i_j = X_i + rho_i * DX_i

                # try:
                f_i_j = eval_F(X_i_j)
                # except Exception:
                #     # NOTE Here we allow the residual evaluation to fail and skip the
                #     # line search step, as this might happen when dealing with
                #     # non-smooth F.
                #     # By continuing the step size comes closer to the old iterate
                #     # making the line search more robust, but slowing the overall
                #     # progress
                #     continue

                pot_i_j = np.sum(f_i_j * f_i_j) * 0.5

                if pot_i_j <= (1 - 2 * kappa_0 * rho_i) * pot_i:
                    break

            DX_i *= rho_i

            if heavy_ball > 0:
                # heavy ball momentum descend (for cases where Armijo is small)
                # weight -> 1, DX -> 0 as solution is approached
                if rho_i < rho_0 ** (N_armijo / 2):
                    # scale with previous update to avoid large over-shooting
                    delta_heavy = 1 / (1 + np.linalg.norm(DX_im1))
                else:
                    delta_heavy = 0.0  # type:ignore[assignment]
                X_i = X_i + delta_heavy * DX_im1
                DX_i += delta_heavy * DX_im1

            # Apply update.
            X_i += DX_i

            # try:
            f_i = eval_F(X_i)
            # except Exception:
            #     exitcode = 3
            #     break

            # print(np.linalg.norm(DX_i), np.linalg.norm(f_i))
            res_history = np.roll(res_history, -1)
            res_history[-1] = np.linalg.norm(f_i)
            if res_history[-1] <= tol:
                exitcode = 0
                break

            # Detect cycling.
            if num_iter > res_history.size:
                was_cycling = is_cycling
                # Looky only for cycles with period 2 to 5.
                for p in range(2, 6):
                    check_len = 2 * p

                    recent = res_history[-check_len:]
                    scale = np.max(np.abs(recent)) if np.max(np.abs(recent)) > 0 else 1

                    is_cycling = True
                    for i in range(p, check_len):
                        # Choosing a higher relative tolerance 1e-4 because KKT systems
                        # are often ill-conditioned.
                        if np.abs(recent[i] - recent[i - p]) >= 1e-3 * scale:
                            is_cycling = False
                            break

                    if is_cycling:
                        detected_cycle = p
                        break

                # Detected new cycling, make line search more aggressive.
                if is_cycling and not was_cycling:
                    rho_0 = max(rho_0 * rho_0, 0.5 * rho_0)
                    N_armijo = max(80, max_iter_armijo * 3)
                    kappa_0 = min(1.5 * kappa_0, 0.4)
                    iter_cycle_detected = num_iter

                # No longer cycling, reset line search parameters.
                elif not is_cycling and was_cycling:
                    # Check that really not cycling anymore for some iterations.
                    if np.all(np.diff(res_history[-2 * detected_cycle :]) < 0):
                        rho_0 = rho
                        N_armijo = max_iter_armijo
                        kappa_0 = kappa
                    else:
                        is_cycling = True
                # Add random perturbation to escape cycle if still cycling
                elif (
                    is_cycling
                    and was_cycling
                    and num_iter - iter_cycle_detected > 2 * detected_cycle
                ):
                    # Add perturbation only to extended fractions where phase absent.
                    xf, yf = _parse_xy(X_i[:-1], n_C, n_P)
                    for j in range(n_P):
                        if yf[j] < EPS:
                            xj = xf[j, :]
                            xj += np.random.rand(n_C) * 1e-4
                            xj[xj < 0.0] = EPS
                            xj[xj > 1.0] = 1.0 - EPS
                            xf[j, :] = xj

                    # yf += np.random.rand(n_P) * 1e-5
                    # yf[yf < 0.0] = EPS
                    # yf[yf > 1.0] = 1.0 - EPS
                    # yf /= np.sum(yf)

                    # X_i[-n_F - 1 : -n_C * n_P - 1] = yf[1:]
                    X_i[-n_C * n_P - 1 : -1] = xf.flatten()
                    # reset cycling detection
                    is_cycling = False
                    was_cycling = False
                    rho_0 = rho
                    N_armijo = max_iter_armijo
                    kappa_0 = kappa

    return X_i[:-1], exitcode, num_iter
