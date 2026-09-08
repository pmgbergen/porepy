"""Unit tests for time-step constraints."""

import numpy as np
import pytest

import porepy as pp
from porepy.time_stepper.time_step_constraint import (
    CannotRecomputeTimeStep,
    CourantTimeStepConstraint,
)
from tests.time_stepper.test_scheduler import (
    get_context_failure,
    get_context_success,
    make_default_scheduler,
)


class MockSubdomain:
    def cell_diameters(self, cell_wise: bool, func):
        assert not cell_wise
        assert func is np.max
        return np.array([0.5, 1.0])


class MockMixedDimensionalGrid:
    def subdomains(self):
        return [subdomain]


class MockEquationSystem:
    def evaluate(self, operator):
        assert operator == "darcy_flux"
        return np.array([2.0, 1.0])


class MockModel(pp.SolutionStrategy):
    def darcy_flux(self, domains):
        assert domains == [subdomain]
        return "darcy_flux"


subdomain = MockSubdomain()


def test_courant_time_step_constraint():
    model = MockModel()
    model.mdg = MockMixedDimensionalGrid()
    model.equation_system = MockEquationSystem()

    constraint = CourantTimeStepConstraint(target_cfl=0.8)

    # min(cell diameter) / max(abs(Darcy flux)) * target CFL = 0.5 / 2 * 0.8.
    adjusted_dt = constraint.suggest_dt(dt=10.0, context={"model": model})

    assert adjusted_dt == 0.2


@pytest.mark.parametrize(
    "case",
    [
        {
            "context": get_context_success(num_linear_iterations=1),
            "expected_dt": 0.5 * 1.3,
        },
        {
            "context": get_context_success(num_linear_iterations=4),
            "expected_dt": 0.5 * 1.3,
        },
        {
            "context": get_context_success(num_linear_iterations=5),
            "expected_dt": 0.5,
        },
        {
            "context": get_context_success(num_linear_iterations=6),
            "expected_dt": 0.5,
        },
        {
            "context": get_context_success(num_linear_iterations=7),
            "expected_dt": 0.5 * 0.7,
        },
        {
            "context": get_context_success(num_linear_iterations=9),
            "expected_dt": 0.5 * 0.7,
        },
        {
            "context": get_context_failure(num_linear_iterations=0),
            "expected_dt": 0.5 * 0.4,
            "success": False,
        },
        {
            "context": get_context_success(num_linear_iterations=1),
            "expected_dt": 0.6,
            "dt_max": 0.6,
        },
        {
            "context": get_context_success(num_linear_iterations=8),
            "expected_dt": 0.49,
            "dt_min": 0.49,
        },
        {
            "context": get_context_success(num_linear_iterations=8),
            "expected_dt": 0.5,
            "dt_min": 0.5,
        },
        {
            "context": get_context_failure(num_linear_iterations=8),
            "expected_dt": "unreachable",
            "dt_min": 0.5,
            "success": False,
            "should_raise": True,
        },
    ],
)
def test_target_nonlinear_iterations(case: dict):
    """Test how dt is adjusted based on the nonlinear solver status. This tests both
    the TargetNonlinearIterations and the integration between the scheduler and
    constaints.

    """
    context = case["context"]
    expected_dt = case["expected_dt"]
    success = case.get("success", True)
    dt_max = case.get("dt_max", None)
    dt_min = case.get("dt_min", None)
    should_raise = case.get("should_raise", False)

    scheduler, time_manager = make_default_scheduler(
        schedule=[0, 2],
        dt_init=0.5,
        constant_dt=False,
        nonlinear_iter_optimal_range=(4, 7),
        nonlinear_iter_relax_factors=(0.7, 1.3),
        nonlinear_iter_retry_factor=0.4,
        dt_max=dt_max,
        dt_min=dt_min,
    )
    if not should_raise:
        dt = scheduler.compute_next_time_step(
            time_manager=time_manager, success=success, context=context
        )
        assert dt == expected_dt
    else:
        with pytest.raises(CannotRecomputeTimeStep):
            _ = scheduler.compute_next_time_step(
                time_manager=time_manager, success=success, context=context
            )
