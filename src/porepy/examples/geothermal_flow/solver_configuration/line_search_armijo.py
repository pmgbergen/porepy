from __future__ import annotations

from typing import Any
import logging
import warnings

import numpy as np

import porepy as pp
from porepy.numerics.nonlinear.anderson_acceleration import AndersonAcceleration

import porepy.models.compositional_flow as cf

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class NewtonAndersonArmijoSolver(pp.NewtonSolver, AndersonAcceleration):
    """Newton solver with Armijo line search, Anderson acceleration,
    and Appleyard variable chopping for compositional stability."""

    def __init__(self, params: dict | None = None):
        pp.NewtonSolver.__init__(self, params)
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

    def _apply_appleyard_chop(
        self, model: pp.PorePyModel, dx: np.ndarray
    ) -> np.ndarray:
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
        if (
            hasattr(model, "phase_fraction_variables")
            and model.phase_fraction_variables
        ):
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

    def iteration(self, model: pp.PorePyModel):
        """An iteration consists of performing the Newton step, obtaining the step size
        from the line search, and then performing the Anderson acceleration based on
        the iterates which are obtained using the step size."""

        iteration = model.nonlinear_solver_statistics.num_iteration

        dx = pp.NewtonSolver.iteration(self, model)

        # Appleyard chop (before Anderson and line search)
        if self.params.get("appleyard_chop", False):
            dx = self._apply_appleyard_chop(model, dx)

        if model.params["apply_schur_complement_reduction"]:
            res_norm = np.linalg.norm(model.equation_system.assemble()[1])
        else:
            res_norm = np.linalg.norm(model.linear_system[1])

        if self.params.get("Anderson_acceleration", False):
            x = model.equation_system.get_variable_values(iterate_index=0)
            x_temp = x + dx
            if not (np.any(np.isnan(x_temp)) or np.any(np.isinf(x_temp))):
                try:
                    xp1 = self.apply(x_temp, dx.copy(), iteration)
                    if res_norm < 10.0:
                        dx = xp1 - x
                except Exception:
                    logger.warning(
                        f"Resetting Anderson acceleration at"
                        f" T={model.time_manager.time}; i={iteration} due to failure."
                    )
                    self.reset()
        alpha = self.nonlinear_line_search(model, dx)
        return alpha * dx

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
        self, model: pp.PorePyModel, dx: np.ndarray, weight: float
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
