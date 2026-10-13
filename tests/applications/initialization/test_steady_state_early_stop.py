"""Unit tests of the criterion to stop the time stepping at a steady state.

The model is replaced by a mock which only provides what the criterion reads: variable
values, an assembled residual, their indexers and the time.

"""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

import porepy as pp
from porepy.applications.initialization.steady_state_early_stop import (
    SteadyStateEarlyStopCriterion,
    SteadyStateModelRunnerSuccess,
    check_and_log_tolerances,
)


def mock_model(state: np.ndarray, residual: np.ndarray, time: float) -> MagicMock:
    """Model with two variables "a" and "b" of one dof each and one equation "eq"."""
    model = MagicMock()
    model.equation_system.variable_indexer.group_by_name.return_value = {
        "a": {"grid": np.array([0])},
        "b": {"grid": np.array([1])},
    }
    model.equation_system.equation_indexer.group_by_name.return_value = {
        "eq": {"grid": np.array([0, 1])}
    }
    model.equation_system.get_variable_values.return_value = state
    model.equation_system.assemble.return_value = residual
    model.time_manager.time = time
    return model


def tolerances(rate: float, residual: float, **kwargs):
    return SteadyStateEarlyStopCriterion(
        rate_tolerances={"a": rate, "b": rate},
        residual_tolerances={"eq": residual},
        **kwargs,
    )


@pytest.mark.parametrize(
    "rate_tol, residual_tol, earliest_stop_time, stops",
    [
        (1.0, 2.0, 0, True),
        # Rate of "b" is 1 / 2 per second, which is not below the tolerance.
        (0.5, 2.0, 0, False),
        # Residual norm is 1, which is not below a tolerance of 1.
        (1.0, 1.0, 0, False),
        # Tolerances are fulfilled but it is too early.
        (1.0, 2.0, 3, False),
        (1.0, 2.0, 2, True),
    ],
)
def test_simulation_should_stop(rate_tol, residual_tol, earliest_stop_time, stops):
    """Stops if rate and residual are below tolerances after the earliest stop time.
    The first call only stores the state, since the rate needs two states."""
    criterion = tolerances(
        rate_tol, residual_tol, earliest_stop_time=earliest_stop_time
    )

    assert (
        criterion.simulation_should_stop(mock_model(np.zeros(2), np.ones(2), 0)) is None
    )

    # Over two seconds, "a" does not change and "b" changes by 1.
    model = mock_model(np.array([0.0, 1.0]), np.ones(2), 2)
    status = criterion.simulation_should_stop(model)
    assert isinstance(status, SteadyStateModelRunnerSuccess) == stops


def test_rate_uses_previous_call():
    """The rate is measured against the state of the previous call, not the first."""
    criterion = tolerances(0.1, np.inf)
    criterion.simulation_should_stop(mock_model(np.zeros(2), np.zeros(2), 0))
    assert (
        criterion.simulation_should_stop(mock_model(np.ones(2), np.zeros(2), 1)) is None
    )
    # No change since the previous call.
    status = criterion.simulation_should_stop(mock_model(np.ones(2), np.zeros(2), 2))
    assert isinstance(status, SteadyStateModelRunnerSuccess)


def test_check_and_log_tolerances(caplog):
    """Returns the norms which are not below the tolerance, and logs all."""
    values = np.array([3.0, 4.0])  # Norm is 5 / sqrt(2).
    with caplog.at_level(logging.INFO):
        failed = check_and_log_tolerances(
            pp.EuclideanMetric(), values, {"norm": 1.0}, "rate"
        )
    assert failed == {"norm": pytest.approx(5 / np.sqrt(2))}
    assert "rate" in caplog.text

    assert (
        check_and_log_tolerances(pp.EuclideanMetric(), values, {"norm": 10}, "") == {}
    )
