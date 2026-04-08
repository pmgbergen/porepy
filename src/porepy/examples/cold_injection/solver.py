"""Contains the global solver used in this example."""

from __future__ import annotations

import logging
from typing import Optional, TypedDict, cast, Any

import numpy as np
import scipy.sparse as sps
from scipy.linalg import lstsq

import porepy as pp
import porepy.models.compositional_flow as cf

logger = logging.getLogger(__name__)


class CFSolverParams(TypedDict):
    """Parameters for the compositional flow solver. They are used additionally to the
    parameters of the default Newton solver."""

    logp_cap: float
    """Capping logarithmic pressure update, if 
    ``model.params["use_logp_nonlinear_rpc"] == True``."""
    newton_chop: float
    """Global chop for raw Newton update."""
    appleyard_chop: float | None
    """Chop for saturation and phase fraction variables. If None, chop is deactivated.
    Otherwise it must be a number between 0 and 1."""

    atol_objective: float
    """Absolute tolerance for merit function."""

    do_armijo_line_search: bool
    """Activate Armijo line search."""
    armijo_line_search_weight: float
    """Initial Armijo step size."""
    armijo_line_search_incline: float
    """Incline parameter for Armijo line search (kappa in the Armijo condition)."""
    armijo_line_search_max_iterations: int
    """Maximal number of line search iterations."""
    armijo_stop_after_residual_reaches: float
    """Stop line search if residual 2-norm reaches this value."""
    armijo_start_after_residual_reaches: float
    """Start line search if residual 2-norm reaches this value."""
    armijo_least_squares_form: bool
    """Whether to use the least squares form of the Armijo condition, i.e.,
    ``||F(x+rho*dx)||^2 <= (1 - 2*kappa*rho)*||F(x)||^2``, instead of the more common
    form ``F(x+rho*dx) <= F(x) + kappa*rho*F'(x)*dx``.
    """

    do_anderson_acceleration: bool
    """Activate Anderson acceleration."""
    anderson_acceleration_depth: int
    """Depth of Anderson acceleration."""
    anderson_acceleration_constrained: bool
    """Adds the constraint that the coefficients of the Anderson acceleration must
    sum up to 1 explicitly in the least squares problem."""
    anderson_acceleration_regularization_parameter: float
    """Regularization parameter for Anderson acceleration least squares problem."""
    anderson_acceleration_relaxation_parameter: float
    """Relaxation parameter for Anderson acceleration. If larger than 0, the Anderson
    acceleration step is relaxed towards the previous iterate."""
    anderson_start_after_residual_reaches: float
    """Start Anderson acceleration only after the residual 2-norm reaches this value."""
    anderson_stop_after_residual_reaches: float
    """Stop Anderson acceleration after the residual 2-norm reaches this value."""

    do_ntrdc: bool
    """Activate Newton Trust-Region Dogleg-Cauchy."""
    ntrdc_delta_0: float
    """Initial trust region radius relative to norm of initial iterate."""
    ntrdc_delta_tol: float
    """Criterion for trust-region radius to exit iteration and return nans
    (time step failed to find direction of sufficient decrease)."""
    ntrdc_eta_1: float
    """Criterion for approximate objective improvement to break the iterations."""
    ntrdc_eta_2: float
    """Criterion for approximate objective improvement to decrease trust-region
    radius."""
    ntrdc_eta_3: float
    """Criterion for approximate objective improvement to increase trust-region
    radius."""
    ntrdc_t_1: float
    """Scaling factor for decreasing trust-region radius."""
    ntrdc_t_2: float
    """Scaling factor for increasing trust-region radius."""


class AndersonAcceleration:
    """Anderson acceleration as described by Walker and Ni in doi:10.2307/23074353."""

    def __init__(
        self,
        params: Optional[dict] = None,
    ) -> None:
        if params is None:
            params = {}
        self._depth = int(params.get("anderson_acceleration_depth", 3))
        self._constrain_acceleration: bool = bool(
            params.get("anderson_acceleration_constrained", False)
        )
        self._reg_param: float = float(
            params.get("anderson_acceleration_regularization_parameter", 0.0)
        )
        self._beta: float = float(
            params.get("anderson_acceleration_relaxation_parameter", 0.0)
        )

        assert 0 <= self._reg_param < 1
        assert 0 <= self._beta < 1

    def apply(self, gk: np.ndarray, fk: np.ndarray, iteration: int) -> np.ndarray:
        """Apply Anderson acceleration.

        Parameters:
            gk: application of some fixed point iteration onto approximation xk, i.e.,
                g(xk).
            fk: residual g(xk) - xk; in general some increment.
            iteration: current iteration count.

        Returns:
            Modified application of fixed point approximation after acceleration, i.e.,
            the new iterate xk+1.

        """

        if iteration == 0:
            dimension = gk.size
            assert dimension == fk.size
            self._Fk: np.ndarray = np.zeros((dimension, self._depth))
            self._Gk: np.ndarray = np.zeros((dimension, self._depth))
            self._xk: np.ndarray = np.zeros((dimension, self._depth))
            self._fkm1: np.ndarray = np.zeros(dimension)
            self._gkm1: np.ndarray = np.zeros(dimension)

        mk = min(iteration, self._depth)

        # Apply actual acceleration (not in the first iteration).
        if mk > 0:
            # Build matrices of changes.
            col = (iteration - 1) % self._depth
            self._Fk[:, col] = fk  # - self._fkm1
            self._Gk[:, col] = gk  # - self._gkm1

            # Solve least squares problem.
            A = self._Fk[:, 0:mk]
            b = fk
            if self._constrain_acceleration:
                A = np.vstack((A, np.ones((1, mk))))
                b = np.concatenate((b, np.ones(1)))

            direct_solve = False

            if self._reg_param > 0:
                b = A.T @ b
                A = A.T @ A + self._reg_param * np.eye(mk)
                direct_solve = np.linalg.matrix_rank(A) >= mk

            if direct_solve:
                gamma_k = np.linalg.solve(A, b)
            else:
                gamma_k = lstsq(A, b)[0]

            # Do the mixing
            # x_k_plus_1 = gk - np.dot(self._Gk[:, 0:mk], gamma_k)
            x_k_plus_1 = np.dot(self._Gk[:, 0:mk], gamma_k)
            if self._beta > 0:
                x_k_plus_1 *= self._beta
                x_k_plus_1 += (1 - self._beta) * np.dot(self._xk[:, 0:mk], gamma_k)
        else:
            x_k_plus_1 = gk

        self._xk[:, :-1] = self._xk[:, 1:]
        self._xk[:, -1] = x_k_plus_1
        # Store values for next iteration.
        # self._fkm1 = fk.copy()
        # self._gkm1 = gk.copy()

        return x_k_plus_1


class CFLESolver(pp.NewtonSolver, AndersonAcceleration):
    """Numerical methods on top of the raw Newton solver for compositional flow
    problems.

    See :class:`CFSolverParams` for the parameters that can be used to control the
    behavior of the solver.
    """

    def __init__(self, params: dict | None = None):
        if params is None:
            params = {}
        assert isinstance(params, dict)
        default_params: dict[Any, Any] = {}
        default_params.update(self.default_params())
        default_params.update(params)
        pp.NewtonSolver.__init__(self, params)
        AndersonAcceleration.__init__(self, params)

        self._J: sps.csr_matrix
        """Current Jacobian matrix."""
        self._F: np.ndarray
        """Current residual vector."""
        self._grad_pot: np.ndarray
        """Current gradient of the potential (i.e., Jac.T times residual)."""
        self._F_norm: float | np.floating
        """Current residual 2-norm."""
        self._xk: np.ndarray
        """Current iterate."""
        self._xk_norm: float | np.floating
        """Current iterate 2-norm."""

        self._delta0: float
        """Initial trust-region radius set in the first iteration."""

        self.params = cast(CFSolverParams, default_params)

    @staticmethod
    def default_params() -> CFSolverParams:
        return CFSolverParams(
            logp_cap=np.log(2.0),
            newton_chop=1.0,
            appleyard_chop=None,
            atol_objective=1e-5,
            do_armijo_line_search=True,
            armijo_line_search_weight=0.9,
            armijo_line_search_incline=1e-4,
            armijo_line_search_max_iterations=20,
            armijo_stop_after_residual_reaches=0.0,
            armijo_start_after_residual_reaches=np.inf,
            armijo_least_squares_form=False,
            do_anderson_acceleration=False,
            anderson_acceleration_depth=3,
            anderson_acceleration_constrained=False,
            anderson_acceleration_regularization_parameter=0.0,
            anderson_acceleration_relaxation_parameter=0.0,
            anderson_start_after_residual_reaches=np.inf,
            anderson_stop_after_residual_reaches=0.0,
            do_ntrdc=False,
            ntrdc_delta_0=0.2,
            ntrdc_delta_tol=1e-4,
            ntrdc_eta_1=1e-3,
            ntrdc_eta_2=0.25,
            ntrdc_eta_3=0.75,
            ntrdc_t_1=0.25,
            ntrdc_t_2=2.0,
        )

    @staticmethod
    def model_uses_logp(model: pp.PorePyModel) -> bool:
        return bool(model.params.get("use_logp_nonlinear_rpc", False))

    def _state(
        self, model: pp.PorePyModel, x: np.ndarray, dx: np.ndarray
    ) -> np.ndarray:
        """Assembles new state considering model parameters."""
        x_new = x + dx
        if self.model_uses_logp(model):
            dofs = model.equation_system.dofs_of(["pressure"])
            p_k = model.equation_system.get_variable_values(
                ["pressure"], iterate_index=0
            )
            p_k1p = p_k * np.exp(dx[dofs])
            x_new[dofs] = p_k1p

        return x_new

    def _increment(
        self, model: pp.PorePyModel, x1: np.ndarray, x0: np.ndarray
    ) -> np.ndarray:
        """Calculate increment from x0 to x1 considering model parameters."""
        dx = x1 - x0
        if self.model_uses_logp(model):
            dofs = model.equation_system.dofs_of(["pressure"])
            dx[dofs] = np.log(x1[dofs] / x0[dofs])
        return dx

    def iteration(self, model: pp.PorePyModel) -> np.ndarray:
        """An iteration consists of performing the Newton step and obtaining the step
        size from the line search."""

        # Raw Newton update.
        dx = super().iteration(model)  # type:ignore[arg-type]
        dx_norm_raw = np.linalg.norm(dx)

        # Catch initial bad iterates.
        if np.any(np.isnan(dx)) or np.any(np.isinf(dx)):
            return np.full_like(dx, np.nan)  # Trigger NanDivergence criterion.

        dx *= self.params["newton_chop"]
        if isinstance(self.params["appleyard_chop"], float):
            dx = self.appleyard_chop(model, dx)

        do_armijo = self.params["do_armijo_line_search"]
        do_anderson = self.params["do_anderson_acceleration"]
        do_ntrdc = self.params["do_ntrdc"]
        least_squares = self.params["armijo_least_squares_form"]

        if do_armijo or do_anderson or do_ntrdc:
            self._xk = model.equation_system.get_variable_values(iterate_index=0)
            self._xk_norm = np.linalg.norm(self._xk)

        if do_ntrdc or (do_armijo and not least_squares):
            A, b = model.equation_system.assemble(evaluate_jacobian=True)
            if self.model_uses_logp(model):
                logp_t = np.ones(model.equation_system.num_dofs())
                logp_t[model.equation_system.dofs_of(["pressure"])] = (
                    model.equation_system.get_variable_values(
                        ["pressure"], iterate_index=0
                    )
                )
                assert A.shape[1] == logp_t.size
                A = A @ sps.diags_array(
                    [logp_t],
                    offsets=[0],
                    shape=(A.shape[1], A.shape[1]),
                    format="csr",
                )
            self._J = A
            self._F = -b
            self._F_norm = np.linalg.norm(self._F)
            self._grad_pot = self._J.transpose() @ self._F
        elif do_armijo:
            self._F = -model.equation_system.assemble(evaluate_jacobian=False)
            self._F_norm = np.linalg.norm(self._F)

        if self.model_uses_logp(model):
            dofs = model.equation_system.dofs_of(["pressure"])
            cap = self.params["logp_cap"]
            dx[dofs] = np.clip(dx[dofs], -cap, cap)

        if do_ntrdc:
            dx = self.ntrdc(model, dx)

        if do_armijo:
            dx *= self.armijo_line_search(model, dx)

        if do_anderson:
            dx = self.anderson_acceleration(model, dx)

        logger.debug(
            f"Change in update norm: {dx_norm_raw:.4e} -> ({np.linalg.norm(dx):.4e})"
        )

        return dx

    def armijo_line_search(self, model: pp.PorePyModel, dx: np.ndarray) -> float:
        """Performs the Armijo line search."""
        F_upper = self.params["armijo_start_after_residual_reaches"]
        F_lower = self.params["armijo_stop_after_residual_reaches"]
        if not (F_lower <= self._F_norm <= F_upper):
            return 1.0

        rho_0 = self.params["armijo_line_search_weight"]
        kappa = self.params["armijo_line_search_incline"]
        N = self.params["armijo_line_search_max_iterations"]
        least_squares = self.params["armijo_least_squares_form"]
        atol = self.params["atol_objective"]

        xk = self._xk

        lin_decrease = 0.0
        if not least_squares:
            lin_decrease = float(np.dot(self._grad_pot, dx))

        pot_0 = self._F_norm**2 * 0.5

        if pot_0 <= atol:
            logger.info(f"Armijo line search potential below cap. Returning 1.")
            return 1.0
        rho = rho_0

        for i in range(N):
            rho = rho_0**i

            try:
                pot_i = self.objective_function(model, self._state(model, xk, rho * dx))
            except:
                # In case this was the last evaluation and it failed, return nan to flag
                # divergence. Avoid errors downstream.
                rho = np.nan
                continue

            if least_squares:
                break_condition = pot_i <= (1 - 2 * kappa * rho) * pot_0
            else:
                break_condition = pot_i <= pot_0 + kappa * rho * lin_decrease

            if break_condition or pot_i <= atol:
                break

        model.nonlinear_solver_statistics.log_custom_data(
            append=True, armijo_iterations=i
        )

        if np.isnan(rho):
            logger.warning("Armijo line search failed. Returning nan.")
        else:
            logger.info(f"Armijo line search determined weight: {rho:.4f} ({i})")

        return rho

    def objective_function(self, model: pp.PorePyModel, state: np.ndarray) -> float:
        """Objective function for the residual depending on a state vector."""
        if isinstance(model, cf.SolutionStrategyPhaseProperties):
            model.update_thermodynamic_properties_of_phases(state=state)
        residual = model.equation_system.assemble(state=state, evaluate_jacobian=False)
        return float(np.dot(residual, residual)) * 0.5

    def appleyard_chop(self, model: pp.PorePyModel, dx: np.ndarray) -> np.ndarray:
        """ "Simple chopping of updates for saturatons such that their absolute values
        is not larger than a defined value ``params['appleyard_chop']``.

        By default, no chop is applied.

        """
        if hasattr(model, "saturation_variables"):
            chop = cast(float, self.params["appleyard_chop"])
            dofs = model.equation_system.dofs_of(model.saturation_variables)
            ds = dx[dofs]

            idx = np.abs(ds) > chop
            if np.any(idx):
                logger.info(f"Appleyard chop on saturations in {int(idx.sum())} cells.")
                ds[idx] = chop * np.sign(ds[idx])
                dx[dofs] = ds

            if model.phase_fraction_variables:  # type:ignore[attr-defined]
                dofs = model.equation_system.dofs_of(model.phase_fraction_variables)  # type:ignore[attr-defined]
                dy = dx[dofs]
                idx = np.abs(dy) > chop
                if np.any(idx):
                    logger.info(
                        f"Appleyard chop on phase fractions in {int(idx.sum())} cells."
                    )
                    dy[idx] = chop * np.sign(dy[idx])
                    dx[dofs] = dy

        return dx

    def ntrdc(self, model: pp.PorePyModel, dx_n: np.ndarray) -> np.ndarray:
        """Newton Trust-Region Dogleg-Cauchy method."""
        if not isinstance(
            model.nonlinear_solver_statistics, pp.NonlinearSolverStatistics
        ):
            logger.warning(
                "Skipping NTRDC. Model does not track nonlinear solver statistics."
            )
            return dx_n

        # Calculate delta0 in first iteration based on initial iterate for Newton.
        # NOTE: Iteration counter is increased after Newton iteration.
        if model.nonlinear_solver_statistics.num_iterations == 0:
            delta0 = self.params["ntrdc_delta_0"] * max(1.0, self._xk_norm)
            self._delta0 = delta0
        else:
            delta0 = self._delta0

        # Extract parameters.
        delta_tol = self.params["ntrdc_delta_tol"]
        eta_1 = self.params["ntrdc_eta_1"]
        eta_2 = self.params["ntrdc_eta_2"]
        eta_3 = self.params["ntrdc_eta_3"]
        t_1 = self.params["ntrdc_t_1"]
        t_2 = self.params["ntrdc_t_2"]
        atol = self.params["atol_objective"]

        xk = self._xk
        xk1p = self._state(model, xk, dx_n)
        grad_pot = self._grad_pot
        delta = delta0
        B = self._J.transpose() @ self._J
        pot_0 = self._F_norm**2 * 0.5
        gBg = np.dot(grad_pot, B @ grad_pot)
        grad_pot_norm = np.linalg.norm(grad_pot)
        g_gBg = grad_pot_norm**2 / gBg

        while True:
            if np.linalg.norm(dx_n) <= delta:
                dx_k = dx_n
            else:
                alpha = min(delta / grad_pot_norm, g_gBg)
                dx_c = -alpha * grad_pot
                if np.linalg.norm(dx_c) >= delta:
                    dx_k = dx_c
                else:
                    dx_ = dx_n - dx_c
                    a = np.dot(dx_, dx_)
                    b = 2.0 * np.dot(dx_c, dx_)
                    c = np.dot(dx_c, dx_c) - delta**2
                    d = np.sqrt(b**2 - 4 * a * c)
                    n = 1.0 / (2 * a)
                    tau = max((-b + d) * n, (-b - d) * n)
                    dx_k = dx_c + tau * dx_

            xk1p = self._state(model, xk, dx_k)

            pot_k = self.objective_function(model, xk1p)
            m_k = pot_0 + np.dot(grad_pot, dx_k) + 0.5 * np.dot(dx_k, B @ dx_k)

            # Approximate improvement.
            rho = (pot_0 - pot_k) / (pot_0 - m_k)

            # NOTE: If objective function (norm of residual squared) is small enough,
            # m_k indicates that the quadratic model is a good approximation and we
            # accept the step. If m_k is very small, rho can be numerically misleading.
            if rho > eta_1 or abs(m_k) <= atol:  # Success condition.
                logger.info(
                    f"NTRDC accepted step with radius {delta:.4e} and improvement "
                    f"{rho:.3e}."
                )
                break

            # Adaption of trust-region radius.
            if rho < eta_2:
                delta *= t_1
            elif rho > eta_3:
                delta *= t_2

            if delta < delta_tol:  # Failure condition.
                logger.warning(
                    "NTRDC reached minimal trust-region radius. Returning nans."
                )
                xk1p = np.full_like(xk, np.nan)
                break

        dx_new = self._increment(model, xk1p, xk)
        return dx_new

    def anderson_acceleration(
        self, model: pp.PorePyModel, dx: np.ndarray
    ) -> np.ndarray:
        """Apply the anderson acceleration."""
        assert not self.model_uses_logp(model), (
            "Anderson acceleration is not currently implemented for use with logp "
            "nonlinear RPC."
        )
        if not isinstance(
            model.nonlinear_solver_statistics, pp.NonlinearSolverStatistics
        ):
            logger.warning(
                "Skipping Anderson acceleration. Model does not track nonlinear "
                "solver statistics."
            )
            return dx

        xk = self._xk
        dx_new = dx
        F_upper = self.params["anderson_start_after_residual_reaches"]
        F_lower = self.params["anderson_stop_after_residual_reaches"]
        if F_lower <= self._F_norm <= F_upper:
            logger.debug("Applying Anderson acceleration.")
            xk1p = self.apply(
                xk + dx, dx, model.nonlinear_solver_statistics.num_iterations
            )
            dx_new = self._increment(model, xk1p, xk)
        return dx_new
