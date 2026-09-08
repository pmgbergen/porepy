"""Unit tests for schedule validation and constant or adaptive time-step control.

The tests cover schedule-point handling, time-step bounds and adjustment, floating-point
inaccuracy, and targeting an optimal range of nonlinear iterations.

"""

from itertools import repeat
from typing import Iterable, Optional

import numpy as np
import pytest

import porepy as pp
from porepy.time_stepper.scheduler import (
    TargetNonlinearIterations,
    TimeScheduler,
    TimeSchedulerConstantDt,
    assemble_default_time_scheduler,
)
from porepy.time_stepper.time_step_constraint import CannotRecomputeTimeStep
from porepy.time_stepper.time_step_control import Schedule, TimeInterval


def make_default_scheduler(
    schedule: list[float],
    dt_init: float,
    constant_dt: bool = False,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
    nonlinear_iter_optimal_range: tuple[int, int] = (4, 7),
    nonlinear_iter_relax_factors: tuple[float, float] = (0.7, 1.3),
    nonlinear_iter_retry_factor: float = 0.5,
    atol: float = 1e-16,
):
    """Construct the default scheduler through its TimeManager interface."""
    if dt_min is None and dt_max is None:
        dt_min_max = None
    else:
        dt_min_max = (
            dt_init * 1e-3 if dt_min is None else dt_min,
            dt_init * 1e3 if dt_max is None else dt_max,
        )
    time_manager = pp.TimeManager(
        schedule=schedule,
        dt_init=dt_init,
        constant_dt=constant_dt,
        dt_min_max=dt_min_max,
        iter_optimal_range=nonlinear_iter_optimal_range,
        iter_relax_factors=nonlinear_iter_relax_factors,
        recomp_factor=nonlinear_iter_retry_factor,
        atol=atol,
    )
    return assemble_default_time_scheduler(time_manager), time_manager


def run_scheduler_collect_data(
    scheduler: pp.time_stepper.TimeSchedulerBase,
    time_manager: pp.TimeManager,
    num_nonlinear_iterations: Optional[Iterable[int]] = None,
    time_step_converged: Optional[Iterable[bool]] = None,
) -> tuple[list[float], list[float]]:
    """Routine for the tests that imitates the simulation's time stepping process by
    calling `scheduler.compute_next_time_step` while the final time is not reached.

    Parameters:
        scheduler: The scheduler.
        time_manager: Time data structure.
        num_nonlinear_iterations: If given, the requested numbers of nonlinear
            iterations that the mock simulation made at each time step. This data is
            passed to the scheduler. If not given (default), it is always 0 iterations.
        time_step_converged: If given, the requested status of each time step (converged
            or not). This data is passed to the scheduler. If not given (default), it is
            always True.

    Returns:
        Two lists: (i) all simulation times and (ii) simulation time that were
            registered as the schedule points.

    """
    if num_nonlinear_iterations is None:
        num_nonlinear_iterations = repeat(0)
    if time_step_converged is None:
        time_step_converged = repeat(True)

    times = []
    checkpoints_hit = []

    # Append initial time step before the loop.
    times.append(time_manager.time)
    if time_manager.is_at_schedule_point():
        checkpoints_hit.append(time_manager.time)

    # Simulate the simulation main loop.
    for ts_converged, num_iters in zip(time_step_converged, num_nonlinear_iterations):
        if time_manager.final_time_reached():
            break
        context = (
            get_context_success(num_iters)
            if ts_converged
            else get_context_failure(num_iters)
        )

        # Mock the simulation time step.
        if ts_converged:
            time_manager.time += time_manager.dt
            time_manager.time_index += 1
        time_manager.dt = scheduler.compute_next_time_step(
            time_manager=time_manager, success=ts_converged, context=context
        )

        # Append results.
        times.append(time_manager.time)
        if time_manager.is_at_schedule_point():
            checkpoints_hit.append(time_manager.time)

    assert time_manager.final_time_reached()
    return times, checkpoints_hit


def get_context_success(num_linear_iterations: int = 0):
    """Default successful time step context for compute_next_time_step."""
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
    """Default failure time step context for compute_next_time_step."""
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
def test_scheduler_floating_point_inaccuracy(constant_dt: bool, dt: float):
    """Test the accumulation of floating-point error when hitting schedule points.

    Purposefully taking unusual time steps to foster error accumulation. With the
    standard snapping tolerance, schedule points are registered within the margin of
    error.

    """
    scheduler, time_manager = make_default_scheduler(
        schedule=[0, 3 * dt, 6 * dt, 9 * dt, 10 * dt],
        dt_init=dt,
        constant_dt=constant_dt,
        atol=1e-8,
        dt_min=dt,
        dt_max=dt,
    )

    times, checkpoints_hit = run_scheduler_collect_data(scheduler, time_manager)

    assert len(times) == 11
    np.testing.assert_allclose(
        checkpoints_hit, time_manager.schedule, atol=1e-8, rtol=0
    )


# Three tests below check the known failures of the time scheduler if an unreasonably
# small snapping time is taken. They ensure that the time stepping breaks in known,
# harmless ways and does not stop the whole simulation. If the time-stepping algorithm
# changes, it is not necessery to preserve this behavior.


def test_scheduler_floating_point_inaccuracy_adaptive_adjusts_to_tiny_step():
    """With strict snapping, the adaptive scheduler inserts a ~1e-15 step to reach a
    schedule point after accumulated floating-point error.

    This is a known behavior for ``dt = 2.5 - 1e-8`` and ``atol = 1e-50``. The tiny
    step remains above the tolerance and the schedule points are still registered.

    """
    dt = 2.5 - 1e-8
    scheduler, time_manager = make_default_scheduler(
        schedule=[0, 3 * dt, 6 * dt, 9 * dt, 10 * dt],
        dt_init=dt,
        atol=1e-50,
        dt_min=dt,
        dt_max=dt,
    )

    times, checkpoints_hit = run_scheduler_collect_data(scheduler, time_manager)

    assert len(times) == 12
    np.testing.assert_allclose(
        checkpoints_hit, time_manager.schedule, atol=1e-50, rtol=0
    )


def test_scheduler_floating_point_inaccuracy_constant_dt_extra_step():
    """With strict snapping, the constant-step scheduler fails to recognize the final
    time after accumulated floating-point error and takes one extra step.

    This is a known behavior for ``dt = 2.5 - 1e-8`` and ``atol = 1e-50``. The run
    nonetheless completes, but not every schedule point is registered.

    """
    dt = 2.5 - 1e-8
    scheduler, time_manager = make_default_scheduler(
        schedule=[0, 3 * dt, 6 * dt, 9 * dt, 10 * dt],
        dt_init=dt,
        constant_dt=True,
        atol=1e-50,
        dt_min=dt,
        dt_max=dt,
    )

    times, checkpoints_hit = run_scheduler_collect_data(scheduler, time_manager)

    assert len(times) == 12
    assert len(checkpoints_hit) != len(time_manager.schedule)
    assert time_manager.time >= time_manager.schedule[-1]


def test_scheduler_floating_point_inaccuracy_constant_dt_missed_checkpoints():
    """With strict snapping, the constant-step scheduler misses schedule points after
    accumulated floating-point error, although it still completes the simulation.

    This is a known behavior for ``dt = 2.5 + 1e-8`` and ``atol = 1e-50``. Unlike the
    lower nearby time step, it does not take an additional final step.

    """
    dt = 2.5 + 1e-8
    scheduler, time_manager = make_default_scheduler(
        schedule=[0, 3 * dt, 6 * dt, 9 * dt, 10 * dt],
        dt_init=dt,
        constant_dt=True,
        atol=1e-50,
        dt_min=dt,
        dt_max=dt,
    )

    times, checkpoints_hit = run_scheduler_collect_data(scheduler, time_manager)

    assert len(times) == 11
    assert len(checkpoints_hit) != len(time_manager.schedule)
    assert time_manager.time >= time_manager.schedule[-1]


def test_inconsistent_schedule_constant_dt():
    """TimeSchedulerConstantDt should fail during initialization."""
    dt = 1.0
    schedule = [0, 1.5, 3]
    with pytest.raises(ValueError):
        _ = make_default_scheduler(schedule=schedule, dt_init=dt, constant_dt=True)


def test_inconsistent_schedule_nonconstant_dt():
    """TimeScheduler (configured with dt_init == dt_min == dt_max) should decrease the
    time step to meet the schedule and recover the original time step after it.

    """
    dt = 1.0
    schedule = [0, 1.5, 3]

    scheduler, time_manager = make_default_scheduler(
        schedule=schedule,
        dt_init=dt,
        constant_dt=False,
        dt_min=dt,
        dt_max=dt,
    )

    times, checkpoints_hit = run_scheduler_collect_data(scheduler, time_manager)

    np.testing.assert_allclose(times, [0, 1, 1.5, 2.5, 3])
    np.testing.assert_allclose(checkpoints_hit, time_manager.schedule)


def test_schedule_length_greater_than_2():
    """An error should be raised if len(schedule) < 2.

    This test is not parametrized because the options are not a Cartesian product of
    parameters.

    """
    for schedule in ([], [1.0]):
        for constant_dt in [True, False]:
            # Construct with the default factory.
            with pytest.raises(ValueError):
                _ = make_default_scheduler(
                    schedule=schedule, dt_init=0.5, constant_dt=constant_dt
                )

        # Construct TimeSchedulerConstantDt manually.
        time_manager = pp.TimeManager(schedule=[0, 1], dt_init=0.5, constant_dt=True)
        with pytest.raises(ValueError):
            _ = TimeSchedulerConstantDt(
                time_manager=time_manager, schedule=schedule, dt=0.5
            )

    # Construct TimeScheduler manually.
    time_manager = pp.TimeManager(schedule=[0, 1], dt_init=0.5)
    with pytest.raises(ValueError):
        _ = TimeScheduler(
            time_manager=time_manager, schedule=Schedule(intervals=[], t_end=1.0)
        )


@pytest.mark.parametrize(
    "schedule", [[1, 0], [2, 30, 15, 16], [2, 3, 15, 14], [1, 2, 2, 3], [1, 2, 3, 3]]
)
@pytest.mark.parametrize("constant_dt", [True, False])
def test_increasing_time_in_schedule(schedule: list[int], constant_dt: bool):
    """An error should be raised if the schedule is not strictly increasing."""
    with pytest.raises(ValueError):
        _ = make_default_scheduler(
            schedule=schedule, dt_init=0.5, constant_dt=constant_dt
        )


@pytest.mark.parametrize("bad_dt", [-0.5, 0])
def test_positive_initial_time_step(bad_dt: float):
    """An error should be raised if the time step is non-positive."""

    for constant_dt in [True, False]:
        with pytest.raises(ValueError):
            _ = make_default_scheduler(
                schedule=[0, 1], dt_init=bad_dt, constant_dt=constant_dt
            )

    # Construct TimeScheduler manually with bad_dt in the second interval.
    time_manager = pp.TimeManager(schedule=[0, 1], dt_init=0.5)
    with pytest.raises(ValueError):
        _ = TimeScheduler(
            time_manager=time_manager,
            schedule=Schedule(
                intervals=[
                    TimeInterval.create(t_start=0, dt_start=0.5),
                    TimeInterval.create(t_start=0.5, dt_start=bad_dt),
                ],
                t_end=1.0,
            ),
        )


@pytest.mark.parametrize("schedule", [[0, 1, 2], [0, 2, 3], [0, 1]])
def test_initial_time_step_overshoots_schedule_point(schedule: list[int]):
    """Test that time scheduler with constant_dt fails to initialize with a
    non-conforming schedule, and the non-constant dt scheduler initializes correctly and
    corrects dt to respect the schedule.

    """
    with pytest.raises(ValueError):
        _ = make_default_scheduler(schedule=schedule, dt_init=2.0, constant_dt=True)

    scheduler, time_manager = make_default_scheduler(
        schedule=schedule, dt_init=2.0, constant_dt=False
    )

    times, checkpoints_hit = run_scheduler_collect_data(scheduler, time_manager)

    np.testing.assert_allclose(checkpoints_hit, time_manager.schedule)
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

    time_manager = pp.TimeManager(schedule=[0, 300], dt_init=dt_init)
    with pytest.raises(ValueError):
        _ = TimeScheduler(
            time_manager=time_manager,
            schedule=Schedule(intervals=intervals, t_end=300),
        )


def test_target_nonlinear_iterations_init():
    with pytest.raises(ValueError):
        _ = TargetNonlinearIterations(iter_min=5, iter_max=4)

    with pytest.raises(ValueError):
        _ = TargetNonlinearIterations(increase_factor=0.5)

    with pytest.raises(ValueError):
        _ = TargetNonlinearIterations(decrease_factor=1.2)

    with pytest.raises(ValueError):
        _ = TargetNonlinearIterations(retry_factor=1.2)


@pytest.mark.parametrize("constant_dt", [True, False])
def test_compute_time_step_after_final_time(constant_dt: bool):
    scheduler, time_manager = make_default_scheduler(
        schedule=[0, 1], dt_init=0.5, constant_dt=constant_dt
    )
    # Reach simulation end.
    _ = run_scheduler_collect_data(scheduler, time_manager)

    assert time_manager.final_time_reached()

    dt_final = time_manager.dt
    dt_new = scheduler.compute_next_time_step(
        time_manager=time_manager, success=True, context=get_context_success()
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
    time_manager = pp.TimeManager(schedule=schedule, dt_init=dt, constant_dt=True)
    scheduler = TimeSchedulerConstantDt(
        time_manager=time_manager, schedule=schedule, dt=dt
    )
    time_manager.time = time
    if is_success:
        new_dt = scheduler.compute_next_time_step(
            time_manager=time_manager, success=is_success, context=context
        )
        assert new_dt == dt
    else:
        with pytest.raises(CannotRecomputeTimeStep):
            _ = scheduler.compute_next_time_step(
                time_manager=time_manager, success=is_success, context=context
            )


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
    """Test if scheduler respects the schedule points."""
    t_snap = 1e-6
    scheduler, time_manager = make_default_scheduler(
        schedule=schedule,
        dt_init=dt_init,
        atol=t_snap,
    )

    _, checkpoint_hits = run_scheduler_collect_data(scheduler, time_manager)
    np.testing.assert_allclose(checkpoint_hits, schedule, atol=t_snap, rtol=0)
    assert time_manager.final_time_reached()


@pytest.mark.parametrize("constant_dt", [True, False])
def test_time_step_match_schedule_exactly(constant_dt: bool):
    """Checks the edge case when the next time step matches the schedule exactly.

    See: https://github.com/pmgbergen/porepy/issues/1152

    """
    scheduler, time_manager = make_default_scheduler(
        schedule=[0, 1, 2], dt_init=1, dt_min=0.1, dt_max=1, constant_dt=constant_dt
    )

    times, checkpoint_hits = run_scheduler_collect_data(scheduler, time_manager)
    np.testing.assert_array_equal(checkpoint_hits, time_manager.schedule)
    np.testing.assert_array_equal(times, checkpoint_hits)
