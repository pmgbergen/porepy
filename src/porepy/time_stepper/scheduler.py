"""Module defines a `TimeSchedulerBase` protocol responsible for adjusting the
simulation time step to conform to the schedule and provided constraints. Two
implementations are available:
- TimeSchedulerConstantDt: a naive implementation with a fixed time step.
- TimeScheduler: an implementation that supports multiple intervals with different
    constraints.

Also provides a convenience factory function `assemble_default_time_scheduler`.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from logging import getLogger
from typing import Iterable, Optional

import numpy as np

import porepy as pp
from porepy.time_stepper.time_step_constraint import (
    CannotRecomputeTimeStep,
    TargetNonlinearIterations,
    TimeStepConstraint,
)
from porepy.time_stepper.time_step_control import Schedule, TimeInterval

__all__ = [
    "assemble_default_time_scheduler",
    "TimeSchedulerBase",
    "TimeSchedulerConstantDt",
    "TimeScheduler",
]

logger = getLogger(__name__)


class TimeSchedulerBase(ABC):
    """Interface for time schedulers."""

    @abstractmethod
    def compute_next_time_step(
        self, time_manager: pp.TimeManager, success: bool, context: dict
    ) -> float:
        """Given the current dt (in `time_manager`), decide what the next time step
        should be.

        Parameters:
            time_manager: Simulation's time data structure.
            success: Whether the current time step was successful.
            context: Data used by `TimeStepConstraint`s' to adjust the time step.

        Raises:
            CannotRecomputeTimeStep: If the time step cannot be adjusted and the
                simulation should be stopped.

        Returns:
            The next time step magnitude.

        """


class TimeSchedulerConstantDt(TimeSchedulerBase):
    """Constant time step scheduler.

    Parameters:
        time_manager: The simulation time data structure. Used in constructor for
            validation.
        schedule: Array of time points, which the simulation time must match exactly
            within tolerance defined by `t_snap`. Must contain at least two points:
            start and end time.
        dt: Constant time step.
        t_snap: Snapping time. Time difference below it is treated as zero.

    """

    def __init__(
        self,
        time_manager: pp.TimeManager,
        schedule: Iterable[pp.number],
        dt: float,
        t_snap: float = 1e-8,
    ) -> None:
        self.schedule = np.array(schedule, dtype=float)
        if len(self.schedule) < 2:
            raise ValueError("Schedule must have at least two points: start and end.")
        assert t_snap > 0
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("Time step must be positive.")
        if self.dt != time_manager.dt:
            raise ValueError(
                "Mismatch between requested time step and what the time_manager has."
            )
        self.t_snap: float = t_snap
        """Snapping time. Time differences below it are treated as zero."""
        _validate_schedule_constant_dt(
            schedule=self.schedule, dt=self.dt, atol=self.t_snap
        )

    def compute_next_time_step(
        self, time_manager: pp.TimeManager, success: bool, context: dict
    ) -> float:
        if not success:
            raise CannotRecomputeTimeStep(
                "Constant time scheduler cannot decrease time step size "
                f"({self.dt:.1e})."
            )
        return self.dt


class TimeScheduler(TimeSchedulerBase):
    """Non-constant time step scheduler.

    Each schedule interval may contain its own set of time step constraints, that
    adjust dt based on simulation context. Each constraint suggests the new dt value,
    which may be smaller or larger than the current dt. The minimum of the suggestions
    is applied. If the minimum is below the interval's `dt_min`, the simulation is
    stopped.

    The algorithm is inspired by:
    [1] Simunek, J., Van Genuchten, M. T., & Sejna, M. (2005). The HYDRUS-1D software
        package for simulating the one-dimensional movement of water, heat, and multiple
        solutes in variably-saturated media. University of California-Riverside Research
        Reports, 3, 1-240.

    [2] Varela, J., Gasda, S. E., Keilegavlen, E., & Nordbotten, J. M. (2021). A
        Finite-Volume-Based Module for Unsaturated Poroelasticity. Advanced Modeling
        with the MATLAB Reservoir Simulation Toolbox.

    Parameters:
        time_manager: The simulation time data structure. Used in constructor for
            validation.
        schedule: Data structure defining schedule points, which the simulation time
            must match exactly within tolerance defined by `t_snap`. Must contain at
            least one time interval.
        t_snap: Snapping time. Time difference below it is treated as zero.

    """

    def __init__(
        self,
        time_manager: pp.TimeManager,
        schedule: Schedule,
        t_snap: float = 1e-6,
    ) -> None:
        if len(schedule.intervals) < 1:
            raise ValueError(
                "At least one interval must be passed, starting at t_start."
            )
        assert t_snap > 0
        self.t_snap: float = t_snap
        """Snapping time. Time differences below it are treated as zero."""
        self.schedule = schedule
        """Simulation schedule."""
        _validate_schedule_non_constant_dt(
            intervals=schedule.intervals, t_end=schedule.t_end, atol=self.t_snap
        )
        self.interval_map = _IntervalMap(schedule.intervals, atol=self.t_snap)
        """A data structure that returns the current and next intervals for any
        simulation time.
    
        """
        current_interval, next_interval = self.interval_map.get(time=time_manager.time)
        time_manager.dt = self._adjust_dt_min_max_schedule(
            time=time_manager.time,
            dt=time_manager.dt,
            current_interval=current_interval,
            next_interval=next_interval,
        )

    def compute_next_time_step(
        self, time_manager: pp.TimeManager, success: bool, context: dict
    ) -> float:
        # Early exit if the simulation is complete.
        if time_manager.final_time_reached():
            return time_manager.dt

        current_interval, next_interval = self.interval_map.get(time=time_manager.time)
        dt = time_manager.dt
        t_end = self.schedule.t_end
        next_checkpoint = t_end if next_interval is None else next_interval.t_start

        # If this is a start of a new interval, set dt to interval.dt_start and log.
        if success and self._is_hitting_interval_start(
            interval=current_interval, time=time_manager.time
        ):
            # We are at the start of a new interval.
            dt = current_interval.dt_start
            _log_schedule_interval_start(
                t_start=current_interval.t_start,
                t_end=next_checkpoint,
                name=current_interval.name,
            )

        # Apply constraints. Each constraint suggests the new dt value, which may be
        # smaller or larger than the current dt. The minimum of the suggestions
        # is applied.
        # Note: If this is a start of a new interval, dt_start can be adjusted as well.
        if len(current_interval.constraints) > 0:
            suggested_dt = [
                constraint.suggest_dt(dt=dt, context=context)
                for constraint in current_interval.constraints
            ]
            dt = min(suggested_dt)

        return self._adjust_dt_min_max_schedule(
            time=time_manager.time,
            dt=dt,
            current_interval=current_interval,
            next_interval=next_interval,
        )

    def _adjust_dt_min_max_schedule(
        self,
        time: float,
        dt: float,
        current_interval: TimeInterval,
        next_interval: TimeInterval | None,
    ) -> float:
        """Adjust dt based on schedule requirements."""

        # If next_interval is None, we are in the last interval. Next checkpoint is
        # t_end.
        next_checkpoint = (
            self.schedule.t_end if next_interval is None else next_interval.t_start
        )

        # Dt should be not larger than the interval's dt_max.
        dt = min(dt, current_interval.dt_max)
        # If dt is smaller than the interval's dt_min, warn about it.
        if dt < current_interval.dt_min:
            logger.warning(
                f"Adjusted time step size ({dt:.1e}) is lower than the minimum "
                f"requested value in this interval ({current_interval.dt_min:.1e})."
            )

        # Prevent overshooting the interval's end.
        if time + dt > next_checkpoint:
            dt = max(next_checkpoint - time, self.t_snap)

        return dt

    def _is_hitting_interval_start(self, interval: TimeInterval, time: float) -> bool:
        """Whether the given time matches the given interval's start time."""
        return (
            (interval.t_start - self.t_snap) <= time <= (interval.t_start + self.t_snap)
        )


def assemble_default_time_scheduler(
    time_manager: pp.TimeManager, constraints: Optional[list[TimeStepConstraint]] = None
) -> TimeSchedulerBase:
    """Convenience factory function that constructs the time scheduler based on the
    parameters specified by the `time_manager`. Additional `constraints` can be passed.

    If `dt_min` and `dt_max` are not specified by the `time_manager`, defaults to 3
    orders of magnitude difference from `dt_init`.

    """
    if constraints is None:
        constraints = []

    # Unpacking time_manager.
    dt_init = time_manager.dt_init
    iter_min, iter_max = time_manager.iter_optimal_range
    decrease_factor, increase_factor = time_manager.iter_relax_factors
    nonlinear_iter_retry_factor = time_manager.recomp_factor

    if len(time_manager.schedule) < 2:
        raise ValueError("Schedule must have at least two points (t_start and t_end).")
    schedule = np.array(time_manager.schedule, dtype=float)

    # Initialize dt_min and dt_max if not given.
    if time_manager.dt_min_max is None:
        dt_min = dt_max = None
    else:
        dt_min, dt_max = time_manager.dt_min_max
    if dt_min is None:
        dt_min = float(dt_init) * 1e-3
    if dt_max is None:
        dt_max = float(dt_init) * 1e3

    if not time_manager.is_constant:
        # Avoid repeating constraint of target nonlinear iterations.
        assert not any(isinstance(c, TargetNonlinearIterations) for c in constraints)
        constraints.append(
            TargetNonlinearIterations(
                dt_min=dt_min,
                iter_min=iter_min,
                iter_max=iter_max,
                increase_factor=increase_factor,
                decrease_factor=decrease_factor,
                retry_factor=nonlinear_iter_retry_factor,
            )
        )
        return TimeScheduler(
            time_manager=time_manager,
            schedule=Schedule(
                intervals=[
                    TimeInterval.create(
                        t_start=t_start,
                        dt_start=dt_init,
                        constraints=constraints,
                        dt_min=dt_min,
                        dt_max=dt_max,
                    )
                    for t_start in schedule[:-1]
                ],
                t_end=schedule[-1],
            ),
            t_snap=time_manager.atol,
        )
    else:
        dt_min = dt_max = dt_init
        return TimeSchedulerConstantDt(
            time_manager=time_manager,
            schedule=schedule,
            dt=dt_init,
            t_snap=time_manager.atol,
        )


class _IntervalMap:
    """An auxiliary data structure used by TimeScheduler. For any simulation time,
    returns the time interval it belongs to, and the next interval.

    Implementation note: does binary search over a sorted array of interval starts with
    O(log n) time complexity for n intervals.

    Parameters:
        intervals: List of intervals. The last interval is `[t_last, ∞)`.
        atol: Snapping time. Time differences below it are treated as zero.

    """

    def __init__(self, intervals: list[TimeInterval], atol: float) -> None:
        self.intervals = intervals
        self._interval_starts = [interval.t_start - atol for interval in intervals]
        # Sanity check: they must be sorted.
        assert self._interval_starts == sorted(self._interval_starts)

    def get(self, time: float) -> tuple[TimeInterval, TimeInterval | None]:
        """Get the interval that corresponds to the requested time.

        Raises:
            ValueError: If the requested time is below the first interval's start time.

        Returns:
            Tuple of two intervals: Current (the one `time` belongs to) and next. If the
                current interval is the last one, next is `None`.

        """
        # Do the binary search. Off by one to match the array index (bisect_right
        # returns 0 if we are below minimum).
        i = bisect_right(self._interval_starts, time) - 1
        if i < 0:
            raise ValueError(
                "The requested time is below the first interval's start time."
            )
        elif i >= len(self.intervals):
            raise ValueError("This should never happen.")
        current_interval = self.intervals[i]
        next_interval = self.intervals[i + 1] if i < (len(self.intervals) - 1) else None
        return current_interval, next_interval


def _log_schedule_interval_start(t_start: float, t_end: float, name: str = "") -> None:
    log_message = "Reached new schedule interval"
    if name != "":
        log_message += f' "{name}"'
    delta = t_end - t_start
    log_message += f" [{t_start:.1e}, {t_start:.1e} + {delta:.1e})."
    logger.info(log_message)


def _validate_schedule_constant_dt(
    schedule: np.ndarray, dt: float, atol: float
) -> None:
    time_from_start = schedule - schedule[0]
    nearest_num_steps = np.rint(time_from_start / dt)
    nearest_constant_dt_times = schedule[0] + nearest_num_steps * dt

    if np.any(abs(schedule - nearest_constant_dt_times) > atol):
        raise ValueError(
            "Mismatch between the time step and scheduled time. Make sure the two are "
            "compatible, or consider adjusting the tolerance."
        )

    _validate_schedule_common(schedule=schedule, atol=atol)


def _validate_schedule_non_constant_dt(
    intervals: list[TimeInterval], t_end: float, atol: float
) -> None:
    for interval in intervals:
        if interval.dt_start < atol:
            raise ValueError(
                f"The new interval's initial step size ({interval.dt_start:.1e}) is "
                f"below the scheduler's minimum resolution ({atol:.1e})."
            )
        if not (interval.dt_min <= interval.dt_start <= interval.dt_max):
            raise ValueError(
                f"The interval's initial time step size ({interval.dt_start:.1e}) is "
                f"not within the allowed range: [{interval.dt_min:.1e}, "
                f"{interval.dt_max:.1e}]."
            )

    schedule = np.array([interval.t_start for interval in intervals] + [t_end])
    _validate_schedule_common(schedule=schedule, atol=atol)


def _validate_schedule_common(schedule: np.ndarray, atol: float) -> None:
    increments = np.ediff1d(schedule)
    if not np.all(increments > atol):
        raise ValueError(
            f"Time schedule points must be strictly increasing, {schedule}."
        )
