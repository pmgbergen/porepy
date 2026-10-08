from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import logging
from pathlib import Path

import pp_solvers
import porepy as pp
from thm_fractured_reservoir import ThmFracturedReservoir, set_model_params

logger = logging.getLogger(__name__)


@dataclass
class SteadyStateModelRunnerSuccess(pp.ModelRunnerStatusSuccess):
    pass


class SteadyStateEarlyStopCriterion(pp.EarlyStopCriterion):
    """Stops the initialization run once both the rate of change of the variables and
    the residual of the equations are below their tolerances.

    After an accepted time step, the previous time step equals the new state, so the
    residual is that of the steady problem.

    Parameters:
        rate_tolerances: Tolerances of the rate of change of variables, see
            :func:`make_tolerances`.
        residual_tolerance: Tolerance of the residual of all equations.
        earliest_stop_time: The simulation is not stopped before this time.

    """

    def __init__(
        self,
        rate_tolerances: dict[str, float],
        residual_tolerances: dict[str, float],
        earliest_stop_time: float = 0,
    ) -> None:
        self.rate_tolerances = rate_tolerances
        self.residual_tolerances = residual_tolerances
        self.earliest_stop_time = earliest_stop_time
        self.previous: tuple[np.ndarray, float] | None = None
        """State and time at the previous call."""

    def simulation_should_stop(
        self, model: pp.PorePyModel
    ) -> pp.ModelRunnerStatus | None:
        state = model.equation_system.get_variable_values(iterate_index=0)
        time = model.time_manager.time
        previous, self.previous = self.previous, (state, time)
        if previous is None:
            return None
        rate = (state - previous[0]) / (time - previous[1])
        failed_rate = check_and_log_tolerances(
            pp.VariableBasedEuclideanMetric(model),
            rate,
            self.rate_tolerances,
            "rate",
        )
        failed_residual = check_and_log_tolerances(
            pp.EquationBasedEuclideanMetric(model),
            model.equation_system.assemble(evaluate_jacobian=False),
            self.residual_tolerances,
            "residual",
        )
        if failed_rate or failed_residual or time < self.earliest_stop_time:
            return None
        return SteadyStateModelRunnerSuccess()


def check_and_log_tolerances(
    metric: pp.Metric,
    values: np.ndarray,
    tolerances: dict[str, float],
    name: str,
) -> dict[str, float]:
    """Log and check the norm of each variable or equation against its tolerance.

    Parameters:
        metric: Computes the norm of each variable or equation block of ``values``.
        values: Rate of change of all variables, or residual of all equations.
        tolerances: Absolute tolerance by name. Others use ``default_tolerance``.
        name: What the values are, for logging.

    Returns:
        The norms that exceed the tolerance, by name.

    """
    norms = metric(values)
    failed = {key: norm for key, norm in norms.items() if norm >= tolerances[key]}
    max_symbols_offset = max(len(key) for key in norms)
    for key, norm in norms.items():
        converged = key not in failed
        logger.info(
            f"{key}:{' ' * (1 + max_symbols_offset - len(key))}"
            f"{name} = {norm:.1e} {'<' if converged else '≥'} "
            f"{tolerances[key]:.1e}{', converged' if converged else ''}"
        )
    return failed
