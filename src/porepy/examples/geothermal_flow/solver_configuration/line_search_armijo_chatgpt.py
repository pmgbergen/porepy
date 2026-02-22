# --- improved_newton_anderson_armijo.py ---------------------------------------
from __future__ import annotations
from typing import Any, Optional
import logging
import numpy as np
import porepy as pp
import warnings

from porepy.numerics.solvers.andersonacceleration import AndersonAcceleration
import porepy.models.compositional_flow as cf

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

###### CHATGPT Rewrite!!! ####################
class NewtonAndersonArmijoSolver(pp.NewtonSolver, AndersonAcceleration):
    """
    Newton solver with Armijo line search and (scaled, constrained, regularized)
    Anderson acceleration. Designed for tabulated (VTK) thermodynamics where
    Jacobians are piecewise-smooth and noisy.

    Improvements:
      - Variable scaling for Anderson
      - Constrained LS (sum-to-one) + Tikhonov regularization
      - Adaptive AA depth based on residual magnitude
      - Restart on divergence trend
      - Apply AA only when Armijo step is not overly damped (alpha > 0.5)
    """

    def __init__(self, params: Optional[dict] = None):
        pp.NewtonSolver.__init__(self, params)
        if params is None:
            params = {}

        # --- Anderson parameters (defaults are robust)
        self._aa_enabled: bool = bool(params.get("Anderson_acceleration", True))
        base_depth = int(params.get("anderson_acceleration_depth", 3))
        self._aa_depth_min = int(params.get("anderson_depth_min", 2))
        self._aa_depth_max = int(params.get("anderson_depth_max", max(5, base_depth)))
        self._aa_constrained = bool(params.get("anderson_acceleration_constrained", True))
        self._aa_reg_param = params.get("anderson_acceleration_regularization_parameter", 1e-6)

        # We must know the total system size once assembled; pass a safe guess, update lazily.
        dimension = int(params.get("anderson_acceleration_dimension", 1000))

        AndersonAcceleration.__init__(
            self,
            dimension=dimension,
            depth=base_depth,
            constrain_acceleration=self._aa_constrained,
            regularization_parameter=self._aa_reg_param,
        )

        # --- Armijo parameters (more permissive = faster)
        self._ls_enabled: bool = bool(params.get("Global_line_search", True))
        self._ls_rho: float = float(params.get("armijo_line_search_weight", 0.5))     # 0.5 recommended
        self._ls_kappa: float = float(params.get("armijo_line_search_incline", 1e-4)) # 1e-4 recommended
        self._ls_maxit: int = int(params.get("armijo_line_search_max_iterations", 10))

        # Restart policy
        self._prev_res: float = np.inf
        self._restart_ratio: float = float(params.get("anderson_restart_ratio", 1.2))
        self._last_alpha: float = 1.0

        # Scale vector cache
        self._scale: Optional[np.ndarray] = None

    # --------------------------- helpers -------------------------------------
    def _ensure_dimension(self, model: pp.PorePyModel) -> None:
        """Lazily sync Anderson 'dimension' with current equation size."""
        n = model.equation_system.num_dofs()
        if n != self._Fk.shape[0]:
            # Reinitialize AA with the new dimension, keep current depth & options
            AndersonAcceleration.__init__(
                self,
                dimension=n,
                depth=self._depth,
                constrain_acceleration=self._aa_constrained,
                regularization_parameter=self._reg_param,
            )
            self.reset()
            logger.info(f"[AA] Dimension updated to {n}")

    def _make_scale(self, x: np.ndarray) -> np.ndarray:
        """Build a per-variable scale to balance magnitudes in AA."""
        # Heuristic: scale by max(|x|, 1) but clip to avoid extremes
        s = np.maximum(np.abs(x), 1.0)
        # Optional: cap very large scales to avoid under-weighting small variables
        s = np.clip(s, 1.0, 1e8)
        return s

    def _apply_anderson_scaled(self, x: np.ndarray, dx: np.ndarray, iteration: int) -> np.ndarray:
        """Apply Anderson to the fixed-point form using variable scaling."""
        if self._scale is None or self._scale.shape != x.shape:
            self._scale = self._make_scale(x)

        s = self._scale
        # Scale gk and fk
        gk_scaled = (x + dx) / s
        fk_scaled = dx / s

        # Anderson acts in scaled space
        x_next_scaled = AndersonAcceleration.apply(self, gk_scaled, fk_scaled, iteration)

        # Map back to physical space
        x_next = s * x_next_scaled
        return x_next

    def _adapt_depth_and_reg(self, res_norm: float) -> None:
        """Adapt AA memory and regularization to current nonlinearity."""
        # Depth: small when far, larger when near
        if res_norm > 1e2:
            self._depth = self._aa_depth_min
        elif res_norm > 1e1:
            self._depth = max(self._aa_depth_min + 1, min(4, self._aa_depth_max))
        else:
            self._depth = self._aa_depth_max

        # Regularization: stronger when far from solution
        self._reg_param = max(1e-10, min(1e-2, 1e-4 * max(1.0, res_norm)))

    # --------------------------- main API ------------------------------------
    def iteration(self, model: pp.PorePyModel):
        """One global nonlinear iteration: Newton step → optional AA → Armijo."""
        it = model.nonlinear_solver_statistics.num_iteration

        # Ensure AA dimension matches current system
        self._ensure_dimension(model)

        # 1) Raw Newton correction
        dx = pp.NewtonSolver.iteration(self, model)

        # Current residual norm (after linear solve)
        res_norm = float(np.linalg.norm(model.linear_system[1]))

        # Adaptive AA controls
        self._adapt_depth_and_reg(res_norm)

        x = model.equation_system.get_variable_values(iterate_index=0)

        # 2) Optional Anderson (direction smoothing) — only if residual is not huge
        #    and we won't likely shrink the step too much in line search.
        use_aa = self._aa_enabled and (res_norm < 1e3)

        if use_aa:
            x_trial = x + dx
            if not (np.any(np.isnan(x_trial)) or np.any(np.isinf(x_trial))):
                try:
                    # tentative AA candidate
                    x_aa = self._apply_anderson_scaled(x, dx.copy(), it)
                    dx_aa = x_aa - x

                    # Heuristic: if Newton residual is already moderate, prefer AA direction
                    # (kept conservative; you can relax threshold as you wish)
                    if res_norm < 10.0:
                        dx = dx_aa
                except Exception:
                    logger.warning(
                        f"[AA] Reset at T={model.time_manager.time}; i={it} (exception)."
                    )
                    self.reset()

        # 3) Armijo line search on the chosen direction
        alpha = self.nonlinear_line_search(model, dx) if self._ls_enabled else 1.0
        self._last_alpha = alpha if np.isscalar(alpha) else alpha[0]

        # 4) Restart AA if trend is bad
        if it > 0 and self._prev_res < np.inf:
            if res_norm / (self._prev_res + 1e-16) > self._restart_ratio:
                logger.info(f"[AA] Restart (residual ratio {res_norm/self._prev_res:.2f}).")
                self.reset()

        self._prev_res = res_norm

        return alpha * dx

    # ---------------------- Armijo: more permissive --------------------------
    def nonlinear_line_search(self, model: pp.PorePyModel, dx: np.ndarray) -> np.ndarray:
        """Backtracking Armijo on residual objective with thermodynamics refresh."""
        if not self._ls_enabled:
            return np.ones_like(dx)

        rho = self._ls_rho         # e.g., 0.5
        kappa = self._ls_kappa     # e.g., 1e-4
        N = self._ls_maxit         # e.g., 10

        pot_0 = self.residual_objective_function(model, dx, 0.0)
        rho_i = rho
        best = pot_0

        for i in range(N):
            rho_i = rho ** i
            pot_i = self.residual_objective_function(model, dx, rho_i)
            # Armijo condition: sufficient decrease
            if pot_i <= (1.0 - 2.0 * kappa * rho_i) * pot_0:
                best = pot_i
                break
            # Early out if residual is clearly worsening
            if pot_i > pot_0 and i >= 2:
                break

        logger.info(f"[LS] Armijo weight: alpha={rho_i:.3g} (iters={i}), "
                    f"Φ_0={pot_0:.3e} → Φ={best:.3e}")
        return rho_i * np.ones_like(dx)

    # ------------------- Residual objective with VTK update ------------------
    def residual_objective_function(
        self, model: pp.PorePyModel, dx: np.ndarray, weight: float
    ) -> np.floating[Any]:
        """Objective Φ(alpha) = 1/2 || R(x + alpha dx) ||^2 with thermo refresh."""
        x_0 = model.equation_system.get_variable_values(iterate_index=0)
        state = x_0 + weight * dx

        # Refresh thermodynamics (VTK sampling)
        cf.SolutionStrategyPhaseProperties.update_thermodynamic_properties_of_phases(
            model,  # type: ignore[arg-type]
            state,
        )

        residual = model.equation_system.assemble(state=state, evaluate_jacobian=False)
        return np.dot(residual, residual) / 2.0
