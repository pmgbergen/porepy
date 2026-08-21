from __future__ import annotations

from typing import Any, Union, Optional
import logging
import warnings

import numpy as np
# os.environ["NUMBA_DISABLE_JIT"] = "1"

import porepy as pp
from porepy.numerics.solvers.anderson_acceleration import AndersonAcceleration

import porepy.models.compositional_flow as cf

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class NewtonAndersonArmijoSolver(pp.solvers.NewtonSolver, AndersonAcceleration):
    """Newton solver with Armijo line search, Anderson acceleration,
    and Appleyard variable chopping for compositional stability."""
    
    # Variables subject to Appleyard chooping and their settings
    # APPLEYARD_DEFAULTS: dict[str, dict] = {
    #     "z_NaCl":   {"max_relative_change": 0.2, "abs_floor": 1e-3},
    #     "s_halite": {"max_relative_change": 0.2, "abs_floor": 1e-3},
    #     "s_gas":    {"max_relative_change": 0.2, "abs_floor": 1e-3},
    # }
     
    def __init__(self, params: dict | None = None):
        pp.solvers.NewtonSolver.__init__(self, params)
        if params is None:
            params = {}
        depth = int(params.get("anderson_acceleration_depth", 3))
        dimension = int(params["anderson_acceleration_dimension"])
        constrain = params.get("anderson_acceleration_constrained", False)
        reg_param = params.get("anderson_acceleration_regularization_parameter", 0.0)
        AndersonAcceleration.__init__(
            self,
            dimension,
            depth,
            constrain_acceleration=constrain,
            regularization_parameter=reg_param,
        )
    
    def _apply_appleyard_chop(self, model: pp.PorePyModel, dx: np.ndarray) -> np.ndarray:
        """Chop updates for saturations, phase fractions, and overall
        compositions so that no absolute change exceeds the limit
        ``params['appleyard_chop']``.
        """
        m = self.params.get("appleyard_chop_value", None)
        if not isinstance(m, float):
            return dx

        assert 0 < m < 1, "Chopping limit must be strictly in (0, 1)."

        # Collect all variable groups that need chopping
        var_groups = []
        if hasattr(model, "saturation_variables"):
            var_groups.append(model.saturation_variables)
        if hasattr(model, "phase_fraction_variables") and model.phase_fraction_variables:
            var_groups.append(model.phase_fraction_variables)
        if hasattr(model, "overall_fraction_variables"):
            var_groups.append(model.overall_fraction_variables)

        for var_names in var_groups:
            dofs = model.equation_system.dofs_of(var_names)
            d = dx[dofs]
            chop = np.abs(d) > m
            if np.any(chop):
                logger.info(
                    f"Appleyard chop on {var_names}: "
                    f"{int(chop.sum())} DOFs clamped to ±{m}"
                )
                d[chop] = m * np.sign(d[chop])
                dx[dofs] = d

        return dx
    

    def _apply_appleyard_chop_clip_z(self, model, dx):
        m = self.params.get("appleyard_chop_value", None)
        if not isinstance(m, float):
            return dx
        assert 0 < m < 1, "Chopping limit must be strictly in (0, 1)."

        # current iterate (so we can clip the resulting VALUE, not just the step)
        x = model.equation_system.get_variable_values(iterate_index=0)

        # variables that are physically bounded in [0, 1]
        bounded_groups = []
        if hasattr(model, "saturation_variables"):
            bounded_groups.append(model.saturation_variables)
        if hasattr(model, "phase_fraction_variables") and model.phase_fraction_variables:
            bounded_groups.append(model.phase_fraction_variables)
        if hasattr(model, "overall_fraction_variables"):
            bounded_groups.append(model.overall_fraction_variables)

        for var_names in bounded_groups:
            dofs = model.equation_system.dofs_of(var_names)
            d = dx[dofs]

            # 1) step limiting (your existing behaviour)
            chop = np.abs(d) > m
            d[chop] = m * np.sign(d[chop])

            # 2) VALUE clipping: ensure x + d stays in [0, 1]
            x_old = x[dofs]
            x_new = x_old + d
            x_new_clipped = np.clip(x_new, 1.0e-17, 1.0)
            d = x_new_clipped - x_old          # corrected increment

            dx[dofs] = d

        return dx
    
    def iteration(self, model: pp.PorePyModel) -> tuple[np.ndarray, pp.solvers.LinearSolverStatus]:
        """An iteration consists of performing the Newton step, obtaining the step size
        from the line search, and then performing the Anderson acceleration based on
        the iterates which are obtained using the step size."""

        dx, linear_solver_status = pp.solvers.NewtonSolver.iteration(self, model)

        # If the linear solve failed, the failure is handled upwards.
        if linear_solver_status.is_failure():
            return dx, linear_solver_status
        
        # Appleyard chop (before Anderson and line search)
        if self.params.get("appleyard_chop", False):
            dx = self._apply_appleyard_chop(model, dx)

        if self.params.get("Anderson_acceleration", False):
            res_norm = float(
                np.linalg.norm(
                    model.equation_system.assemble(evaluate_jacobian=False)
                )
            )
            x = model.equation_system.get_variable_values(iterate_index=0)
            x_temp = x + dx
            if not (np.any(np.isnan(x_temp)) or np.any(np.isinf(x_temp))):
                try:
                    xp1 = self.apply(x_temp, dx.copy(), self.iteration_index)
                    if res_norm < 10.0:
                        dx = xp1 - x
                except Exception:
                    logger.warning(
                        f"Resetting Anderson acceleration at"
                        f" T={model.time_manager.time}; i={self.iteration_index} due to failure."
                    )
                    self.reset()
        alpha = self.nonlinear_line_search(model, dx)
        sol = alpha * dx

        model._current_update = sol  # type: ignore[attr-defined]

        return sol, linear_solver_status

    def nonlinear_line_search(
        self, model: pp.PorePyModel, dx: np.ndarray
    ) -> np.ndarray:
        """Performs the Armijo line search."""

        if not self.params.get("Global_line_search", False):
            return np.ones_like(dx)

        rho = float(self.params.get("armijo_line_search_weight", 0.9))
        kappa = float(self.params.get("armijo_line_search_incline", 0.4))
        N = int(self.params.get("armijo_line_search_max_iterations", 50))

        pot_0 = self.residual_objective_function(model, dx, 0.0)
        rho_i = rho
        n = 0

        for i in range(N):
            n = i
            rho_i = rho**i

            pot_i = self.residual_objective_function(model, dx, rho_i)
            if pot_i <= (1 - 2 * kappa * rho_i) * pot_0:
                break

        logger.info(f"Armijo line search determined weight: {rho_i} ({n})")
        return rho_i * np.ones_like(dx)

    def residual_objective_function(
        self, 
        model: pp.PorePyModel, dx: np.ndarray, weight: float
    ) -> np.floating[Any]:
        """The objective function to be minimized is the norm of the residual squared
        and divided by 2."""
        
        x_0 = model.equation_system.get_variable_values(iterate_index=0)
        state = x_0 + weight * dx
        # model.update_thermodynamic_properties_of_phases(state)
        cf.SolutionStrategyPhaseProperties.update_thermodynamic_properties_of_phases(
            model,  # type:ignore[arg-type]
            state,
        )
        # model.update_thermodynamic_properties_of_phases(state)
        residual = model.equation_system.assemble(state=state, evaluate_jacobian=False)
        return np.dot(residual, residual) / 2.0


class AdaptiveTimeManager(pp.TimeManager):
    """Time manager that forces dt recovery after prolonged small dt periods."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._consecutive_small_dt_steps = 0
        self._small_dt_threshold = 10.0  # seconds
        self._recovery_threshold = 50    # steps

    def compute_time_step(
        self,
        iterations: Optional[int] = None,
        recompute_solution: bool = False
    ) -> Union[float, None]:
        """Override to force dt growth after many small dt steps."""
        
        result = super().compute_time_step(iterations, recompute_solution)
        
        if result is None:
            return None
        
        # Track consecutive small dt steps
        if self.dt < self._small_dt_threshold:
            self._consecutive_small_dt_steps += 1
        else:
            self._consecutive_small_dt_steps = 0
        
        # Force dt growth if stuck at small dt for too long
        if (self._consecutive_small_dt_steps > self._recovery_threshold and not recompute_solution):
            # Force dt to grow
            self.dt = min(self.dt * 2.0, self.dt_min_max[1])
            self._consecutive_small_dt_steps = 0
            print(f">>> FORCED dt recovery: new dt = {self.dt:.2f}s")

        return self.dt
