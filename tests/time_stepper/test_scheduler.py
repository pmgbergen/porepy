import logging
from itertools import repeat
from typing import Iterable, Optional

import numpy as np
import pytest

import porepy as pp
from porepy.time_stepper.scheduler import (
    CannotRecomputeTimeStep,
    TargetNonlinearIterations,
    TimeInterval,
    TimeScheduler,
    TimeSchedulerConstantDt,
    assemble_default_time_scheduler,
)


def run_scheduler_collect_data(
    scheduler: pp.time_stepper.TimeSchedulerBase,
    num_nonlinear_iterations: Optional[Iterable[int]] = None,
    time_step_converged: Optional[Iterable[bool]] = None,
) -> tuple[list[float], list[float]]:
    if num_nonlinear_iterations is None:
        num_nonlinear_iterations = repeat(0)
    if time_step_converged is None:
        time_step_converged = repeat(True)

    times = []
    checkpoints_hit = []

    # Append initial time step before the loop.
    times.append(scheduler.get_time())
    if scheduler.is_hitting_schedule():
        checkpoints_hit.append(scheduler.get_time())

    # Simulate the simulation main loop.
    for ts_converged, num_iters in zip(time_step_converged, num_nonlinear_iterations):
        if scheduler.is_finished():
            break
        context = (
            get_context_success(num_iters)
            if ts_converged
            else get_context_failure(num_iters)
        )
        scheduler.compute_next_time_step(success=ts_converged, context=context)
        times.append(scheduler.get_time())
        if scheduler.is_hitting_schedule():
            checkpoints_hit.append(scheduler.get_time())

    assert scheduler.is_finished()
    return times, checkpoints_hit


def get_context_success(num_linear_iterations: int = 0):
    # Default successful time step context for compute_next_time_step.
    return {
        "nonlinear_solver_status": pp.solvers.NewtonSolverConverged(
            linear_solver_statuses=(
                [pp.solvers.LinearSolverStatusSuccess(solve_time=1.0)]
                * num_linear_iterations
            ),
            convergence_statuses=pp.solvers.ConvergenceStatusCollection(),
            divergence_statuses=pp.solvers.ConvergenceStatusCollection(),
        )
    }


def get_context_failure(num_linear_iterations: int = 0):
    # Default failure time step context for compute_next_time_step.
    return {
        "nonlinear_solver_status": pp.solvers.NewtonSolverFailed(
            linear_solver_statuses=(
                [pp.solvers.LinearSolverStatusSuccess(solve_time=1.0)]
                * num_linear_iterations
            ),
            convergence_statuses=pp.solvers.ConvergenceStatusCollection(),
            divergence_statuses=pp.solvers.ConvergenceStatusCollection(),
        )
    }


@pytest.mark.parametrize("constant_dt", [True, False])
@pytest.mark.parametrize("dt_snap", [1e-8, 1e-50])
@pytest.mark.parametrize(
    "dt",
    [
        2.5 - 1e-8,
        2.5 + 1e-8,
        2.5e8 - 1,
        2.5e8 + 1,
        2.5e-8 + 1e-16,
        2.5e-8 - 1e-16,
    ],
)
def test_scheduler_floating_point_inaccuracy(
    constant_dt: bool, dt: float, dt_snap: float
):
    """Test the accumulaton of floating point error when hitting schedule points.

    Purposefully taking weird dt to foster error accumulation. With the default dt_snap
    (1e-8), the schedule points are registered correctly and well within the margin of
    error. Problems would start in this example if dt_snap < 1e-15.

    The dt_snap = 1e-50 case tests the behavior when the schedule points stop
    registering correctly. Its goal is to ensure that, while they are not registering,
    the simulation does not abort and continue until completion.

    """

    scheduler = assemble_default_time_scheduler(
        schedule=[0, 3 * dt, 6 * dt, 9 * dt, 10 * dt],
        dt_init=dt,
        constant_dt=constant_dt,
        atol=dt_snap,
        dt_min=dt,
        dt_max=dt,
    )

    times, checkpoints_hit = run_scheduler_collect_data(scheduler)

    # Check the results. Treat special cases corresponding to incorrect schedule points
    # registration with dt_snap = 1e-50. Importantly, with a reasonable dt_snap = 1e-8,
    # it is always the general case. If the implementation chages, it is not necessery
    # to preserve these special cases, it is just a "known misbehavior".
    if not constant_dt and dt_snap == 1e-50 and dt == (2.5 - 1e-8):
        # Special case: scheduler adjusts to dt ~ 1e-15 to match the schedule point.
        # Therefore, we make one very small time step (still above dt_snap = 1e-50).
        assert len(times) == 12
    elif constant_dt and dt_snap == 1e-50 and dt == (2.5 - 1e-8):
        # Special case: Constant dt scheduler does not acknowledge that we reached t_end
        # due to accumulated floating point error. It makes an additional time step.
        assert len(times) == 12
    else:
        # General case: 11 time steps as expected.
        assert len(times) == 11

    expected_schedule = scheduler.get_schedule()

    if constant_dt and dt_snap == 1e-50 and dt in [(2.5 - 1e-8), (2.5 + 1e-8)]:
        # Special case: Constant dt scheduler does not acknowledge that we hit schedule
        # points. We still complete the simulation successfully.
        assert not len(checkpoints_hit) == len(expected_schedule)
        assert scheduler.get_time() >= scheduler.get_time_end()
    else:
        # General case: All schedule points are handled correctly.
        np.testing.assert_allclose(
            checkpoints_hit, expected_schedule, atol=dt_snap, rtol=0
        )


@pytest.mark.parametrize("constant_dt", [True, False])
def test_inconsistent_schedule(constant_dt: bool):
    """Inconsistent schedule with constant time step.

    TimeSchedulerConstantDt should fail during initialization. TimeScheduler (configured
    with dt_init == dt_min == dt_max) should decrease the time step to meet the schedule
    and recover the original time step after it.

    """
    dt = 1.0
    schedule = [0, 1.5, 3]
    if constant_dt:
        with pytest.raises(ValueError):
            scheduler = assemble_default_time_scheduler(
                schedule=schedule, dt_init=dt, constant_dt=constant_dt
            )
        return
    else:
        scheduler = assemble_default_time_scheduler(
            schedule=schedule,
            dt_init=dt,
            constant_dt=constant_dt,
            dt_min=dt,
            dt_max=dt,
        )

    times, checkpoints_hit = run_scheduler_collect_data(scheduler)

    np.testing.assert_allclose(times, [0, 1, 1.5, 2.5, 3])
    np.testing.assert_allclose(checkpoints_hit, scheduler.get_schedule())


def test_schedule_length_greater_than_2():
    """An error should be raised if len(schedule) < 2.

    This test is not parametrized because the options are not a Cartesian product of
    parameters.

    """
    for schedule in ([], [1.0]):
        for constant_dt in [True, False]:
            # Construct with the default factory.
            with pytest.raises(ValueError):
                _ = assemble_default_time_scheduler(
                    schedule=schedule, dt_init=0.5, constant_dt=constant_dt
                )

        # Construct TimeSchedulerConstantDt manually.
        with pytest.raises(ValueError):
            _ = TimeSchedulerConstantDt(schedule=schedule, dt=0.5)

    # Construct TimeScheduler manually.
    with pytest.raises(ValueError):
        _ = TimeScheduler(intervals=[], t_end=1.0)


@pytest.mark.parametrize(
    "schedule", [[1, 0], [2, 30, 15, 16], [2, 3, 15, 14], [1, 2, 2, 3], [1, 2, 3, 3]]
)
@pytest.mark.parametrize("constant_dt", [True, False])
def test_increasing_time_in_schedule(schedule: list[int], constant_dt: bool):
    """An error should be raised if a the schedule is not strictly increasing."""
    with pytest.raises(ValueError):
        _ = assemble_default_time_scheduler(
            schedule=schedule, dt_init=0.5, constant_dt=constant_dt
        )


@pytest.mark.parametrize("bad_dt", [-0.5, 0])
def test_positive_initial_time_step(bad_dt: float):
    """An error should be raised if the time step is non-positive."""

    for constant_dt in [True, False]:
        with pytest.raises(ValueError):
            _ = assemble_default_time_scheduler(
                schedule=[0, 1], dt_init=bad_dt, constant_dt=constant_dt
            )

    # Construct TimeScheduler manually with bad_dt in the second interval.
    with pytest.raises(ValueError):
        _ = TimeScheduler(
            intervals=[
                TimeInterval.create(t_start=0, dt_start=0.5),
                TimeInterval.create(t_start=0.5, dt_start=bad_dt),
            ],
            t_end=1.0,
        )


@pytest.mark.parametrize("schedule", [[0, 1, 2], [0, 2, 3], [0, 1]])
def test_initial_time_step_overshoots_schedule_point(schedule: list[int]):
    with pytest.raises(ValueError):
        _ = assemble_default_time_scheduler(
            schedule=schedule, dt_init=2.0, constant_dt=True
        )

    scheduler = assemble_default_time_scheduler(
        schedule=schedule, dt_init=2.0, constant_dt=False
    )

    times, checkpoints_hit = run_scheduler_collect_data(scheduler)

    np.testing.assert_allclose(checkpoints_hit, scheduler.get_schedule())
    np.testing.assert_allclose(times, checkpoints_hit)


@pytest.mark.parametrize("dt_init", [1.0, 5.0])
@pytest.mark.parametrize("bad_interval_index", [0, 1, 2])
def test_dt_not_within_min_max_range(dt_init: float, bad_interval_index: int):
    dt_min = 2.0
    dt_max = 4.0
    intervals = []
    for i in range(3):
        if i != bad_interval_index:
            intervals.append(TimeInterval.create(t_start=i * 100, dt_start=dt_init))
        else:
            intervals.append(
                TimeInterval.create(
                    t_start=i * 100, dt_start=dt_init, dt_min=dt_min, dt_max=dt_max
                )
            )

    with pytest.raises(ValueError):
        _ = TimeScheduler(intervals=intervals, t_end=300)


def test_target_nonlinear_iterations_init():
    with pytest.raises(ValueError):
        _ = TargetNonlinearIterations(iter_min=5, iter_max=4, dt_min=0.1)

    with pytest.raises(ValueError):
        _ = TargetNonlinearIterations(increase_factor=0.5, dt_min=0.1)

    with pytest.raises(ValueError):
        _ = TargetNonlinearIterations(decrease_factor=1.2, dt_min=0.1)

    with pytest.raises(ValueError):
        _ = TargetNonlinearIterations(retry_factor=1.2, dt_min=0.1)


@pytest.mark.parametrize("constant_dt", [True, False])
def test_compute_time_step_after_final_time(constant_dt: bool):
    scheduler = assemble_default_time_scheduler(
        schedule=[0, 1], dt_init=0.5, constant_dt=constant_dt
    )
    # Reach simulation end.
    _ = run_scheduler_collect_data(scheduler)

    assert scheduler.is_finished()

    dt_final = scheduler.get_dt()
    dt_new = scheduler.compute_next_time_step(
        success=True, context=get_context_success()
    )
    assert dt_final == dt_new


@pytest.mark.parametrize("schedule", [[0, 10], [0, 20]])
@pytest.mark.parametrize("dt", [0.1, 0.5])
@pytest.mark.parametrize("time", [0, 1, 1.5])
@pytest.mark.parametrize("is_success", [True, False])
@pytest.mark.parametrize("context", [get_context_success(), get_context_failure()])
def test_constant_time_step(schedule, dt, time, is_success, context):
    """Test if a constant dt is returned, independent of any configuration or
    input."""
    scheduler = TimeSchedulerConstantDt(schedule=schedule, dt=dt)
    scheduler.time = time
    if is_success:
        _ = scheduler.compute_next_time_step(success=is_success, context=context)
        assert scheduler.dt == dt
    else:
        with pytest.raises(CannotRecomputeTimeStep):
            _ = scheduler.compute_next_time_step(success=is_success, context=context)


@pytest.mark.parametrize(
    "case",
    [
        {
            "context": get_context_success(num_linear_iterations=1),
            "expected_dt": 0.5 * 1.3,
        },
        {
            "context": get_context_success(num_linear_iterations=4),
            "expected_dt": 0.5,
        },
        {
            "context": get_context_success(num_linear_iterations=5),
            "expected_dt": 0.5,
        },
        {
            "context": get_context_success(num_linear_iterations=7),
            "expected_dt": 0.5,
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
    """Test behaviour of the algorithm when the solution should be recomputed. Note
    that this should be independent of the number of iterations that the user passes
    """
    context = case["context"]
    expected_dt = case["expected_dt"]
    success = case.get("success", True)
    dt_max = case.get("dt_max", None)
    dt_min = case.get("dt_min", None)
    should_raise = case.get("should_raise", False)

    scheduler = assemble_default_time_scheduler(
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
        dt = scheduler.compute_next_time_step(success=success, context=context)
        assert dt == expected_dt
    else:
        with pytest.raises(CannotRecomputeTimeStep):
            _ = scheduler.compute_next_time_step(success=success, context=context)


@pytest.mark.parametrize(
    "schedule, dt_init",
    [
        ([0, 1], 0.1),
        ([0, 10, 20, 30], 1),
        ([10, 11, 15, 16, 19, 20], 1),
        (
            [0, 0.01, 1 * pp.HOUR, 2 * pp.HOUR, 100 * pp.HOUR, 101 * pp.HOUR],
            2 * pp.HOUR,
        ),
    ],
)
def test_hitting_schedule_times(schedule, dt_init):
    """Test if algorithm respects the passed target times from the schedule,"""
    t_snap = 1e-6
    scheduler = assemble_default_time_scheduler(
        schedule=schedule,
        dt_init=dt_init,
        atol=t_snap,
    )

    _, checkpoint_hits = run_scheduler_collect_data(scheduler)
    np.testing.assert_allclose(checkpoint_hits, schedule, atol=t_snap, rtol=0)
    assert scheduler.is_finished()


@pytest.mark.parametrize("constant_dt", [True, False])
def test_time_step_match_schedule_exactly(constant_dt: bool):
    """Checks the edge case when the dynamic time stepping is on, but the next time
    step matches the schedule exactly.

    See: https://github.com/pmgbergen/porepy/issues/1152

    """
    scheduler = assemble_default_time_scheduler(
        schedule=[0, 1, 2], dt_init=1, dt_min=0.1, dt_max=1, constant_dt=constant_dt
    )

    times, checkpoint_hits = run_scheduler_collect_data(scheduler)
    np.testing.assert_array_equal(checkpoint_hits, scheduler.get_schedule())
    np.testing.assert_array_equal(times, checkpoint_hits)


def test_time_manager_deprecation():
    time_manager = pp.TimeManager(schedule=[0, 1], dt_init=0.5)

    with pytest.raises(ValueError):
        time_manager.time = 1

    with pytest.raises(ValueError):
        time_manager.dt = 1
