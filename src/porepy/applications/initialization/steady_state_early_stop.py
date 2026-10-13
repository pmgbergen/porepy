"""Criterion to stop the time stepping to a steady state.

The criterion is evaluated by the model runner after each accepted time step, see
:class:`~porepy.models.model_runner.EarlyStopCriterion`. It is used by
:func:`~porepy.applications.initialization.initialization.initialization_pipeline`.

"""

import logging
from dataclasses import dataclass

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)


@dataclass
class SteadyStateModelRunnerSuccess(pp.ModelRunnerStatusSuccess):
    """Status of a model runner that stopped because a steady state was reached.

    It distinguishes this from reaching the final time, which is a success of the
    runner but not an equilibrium.

    """


class SteadyStateEarlyStopCriterion(pp.EarlyStopCriterion):
    """Stops the initialization run once both the rate of change of the variables and
    the residual of the equations are below their tolerances.

    The rate of change of the variables is the change since the previous call divided by
    the time between the calls, measured per variable with the variable indexer of the
    model. The residual is that of the equations of the model at the new state, measured
    per equation with the equation indexer. After an accepted time step the previous
    time step equals the new state, so the accumulation terms vanish and the residual is
    that of the steady problem. Each norm is the root mean square of the block.

    Parameters:
        rate_tolerances: Tolerance of the norm of the rate of change per variable name,
            in model units per second. Must return a value for all variable names, e.g.
            a ``defaultdict``.
        residual_tolerances: Tolerance of the norm of the residual per equation name, in
            model units. Must return a value for all equation names.
        earliest_stop_time: The simulation is not stopped before this time, although the
            tolerances are fulfilled.

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
        """Check the rate of change and the residual after an accepted time step.

        Returns:
            :class:`SteadyStateModelRunnerSuccess` if the model is in a steady state,
            otherwise None. The first call only stores the state, since the rate needs
            two states.

        """
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
        tolerances: Absolute tolerance by name. Must return a value for all names.
        name: What the values are, for logging, e.g. "rate".

    Returns:
        The norms that are not below the tolerance, by name.

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
