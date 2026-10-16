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
    """Stops the run once the variables and equations are in a steady state.

    The model is in a steady state if both the rate of change of the variables and the
    residual of the equations are below their tolerances.

    The rate of change of the variables is the change since the previous call divided by
    the time between the calls. The residual is that of the equations of the model at
    the new state. After an accepted time step the previous time step equals the new
    state, so the accumulation terms vanish and the residual is that of the steady
    problem.

    The norms are computed per variable and per equation by the metrics
    :attr:`variable_rate_metric` and :attr:`residual_metric`. By default, these are the
    normalized Euclidean norm (root mean square of the block) for the rate and the
    Lebesgue L2 norm for the residual.

    Parameters:
        rate_tolerances: Tolerance of the norm of the rate of change per variable name,
            in model units per second. Must contain all variable names.
        residual_tolerances: Tolerance of the norm of the residual per equation name, in
            model units. Must contain all equation names.
        earliest_stop_time: The simulation is not stopped before this time, although the
            tolerances are fulfilled.
        variable_rate_metric: Metric computing the norm of the rate of change per
            variable name. If None (default), a
            :class:`~porepy.VariableBasedEuclideanMetric` is created at the first call.
        residual_metric: Metric computing the norm of the residual per equation name.
            If None (default), an :class:`~porepy.EquationBasedLebesgueMetric` is
            created at the first call.

    """

    def __init__(
        self,
        rate_tolerances: dict[str, float],
        residual_tolerances: dict[str, float],
        earliest_stop_time: float = 0,
        variable_rate_metric: pp.Metric | None = None,
        residual_metric: pp.Metric | None = None,
    ) -> None:
        self.rate_tolerances = rate_tolerances
        """Tolerance of the norm of the rate of change per variable name, in model units
        per second. Must contain all variable names.

        """
        self.residual_tolerances = residual_tolerances
        """Tolerance of the norm of the residual per equation name, in model units.
        Must contain all equation names.

        """
        self.earliest_stop_time = earliest_stop_time
        """The simulation is not stopped before this time, although the tolerances are
        fulfilled.

        """
        self.previous: tuple[np.ndarray, float] | None = None
        """(State, time) at the previous call."""

        self.variable_rate_metric: pp.Metric | None = variable_rate_metric
        """Metric computing the norm of the rate of change per variable name.

        If None, it is set to :class:`~porepy.VariableBasedEuclideanMetric` of the model
        at the first call of :meth:`simulation_should_stop`.

        """
        self.residual_metric: pp.Metric | None = residual_metric
        """Metric computing the norm of the residual per equation name.

        If None, it is set to :class:`~porepy.EquationBasedLebesgueMetric` of the model
        at the first call of :meth:`simulation_should_stop`.

        """

    def simulation_should_stop(
        self, model: pp.PorePyModel
    ) -> pp.ModelRunnerStatus | pp.EarlyStopNotRequested:
        """Check the rate of change and the residual after an accepted time step.

        Parameters:
            model: The model in the state after the time step.

        Returns:
            :class:`SteadyStateModelRunnerSuccess` if the model is in a steady state,
            otherwise :class:`~porepy.EarlyStopNotRequested`. The first call only stores
            the state, since the rate needs two states.

        """
        # Lazy initialization of metrics with the model.
        if self.variable_rate_metric is None:
            self.variable_rate_metric = pp.VariableBasedEuclideanMetric(model)
        if self.residual_metric is None:
            self.residual_metric = pp.EquationBasedLebesgueMetric(model)

        state = model.equation_system.get_variable_values(iterate_index=0)
        time = model.time_manager.time
        previous, self.previous = self.previous, (state, time)
        if previous is None:
            return pp.EarlyStopNotRequested()
        rate = (state - previous[0]) / (time - previous[1])
        failed_rate = check_and_log_tolerances(
            self.variable_rate_metric,
            rate,
            self.rate_tolerances,
            "rate",
        )
        failed_residual = check_and_log_tolerances(
            self.residual_metric,
            model.equation_system.assemble(evaluate_jacobian=False),
            self.residual_tolerances,
            "residual",
        )
        if failed_rate or failed_residual or time < self.earliest_stop_time:
            return pp.EarlyStopNotRequested()
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
        tolerances: Absolute tolerance by name. Must contain all names.
        name: What the values are, for logging, e.g. "rate".

    Returns:
        The norms that are not below the tolerance, by name-value mappings.

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
