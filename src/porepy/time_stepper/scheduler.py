from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import Optional
import porepy as pp
import numpy as np
from dataclasses import dataclass

__all__ = [
    "TimeStepConstraint",
    "TimeInterval",
    "TimeScheduler",
]

"""
- decrease: too many newton iteration
- increase: too few newton iterations
- decrease: newton failed
- decrease: physics (CFL)

- not increase (max)
- abort simulation: can't decrease (min)

- decrease: about to hit schedule
- set new: after hit schedule

"""

from logging import getLogger

__all__ = [
    "TimeStepConstraint",
    "TargetNonlinearIterations",
    "TimeInterval",
    "CannotRecomputeTimeStep",
    "TimeScheduler",
]

logger = getLogger(__name__)


class TimeStepConstraint(ABC):
    @abstractmethod
    def suggest_dt(self, dt: float, context: dict) -> float:
        pass


class TargetNonlinearIterations(TimeStepConstraint):
    def __init__(
        self,
        iter_min: int = 4,
        iter_max: int = 7,
        increase_factor: float = 1.3,
        decrease_factor: float = 0.7,
        retry_factor: float = 0.5,
    ) -> None:
        self.iter_min = iter_min
        self.iter_max = iter_max
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.retry_factor = retry_factor

    def suggest_dt(self, dt: float, context: dict) -> float:
        status: pp.solvers.NonlinearSolverStatus | None = context.get(
            "nonlinear_solver_status", None
        )
        if status is None:
            # TODO YZ: Should warn
            assert False

        if status.is_converged():
            num_iter = status.number_of_iterations()
            if num_iter < self.iter_min:
                return dt * self.increase_factor
            elif num_iter > self.iter_max:
                return dt * self.decrease_factor
            else:
                return dt
        else:
            return dt * self.retry_factor


@dataclass
class TimeInterval:
    t_start: float
    dt_start: float
    constraints: list[TimeStepConstraint]
    dt_min: float
    dt_max: float
    name: str

    @classmethod
    def create(
        cls,
        t_start: float,
        dt_start: float,
        constraints: Optional[list[TimeStepConstraint]] = None,
        dt_min: Optional[float] = None,
        dt_max: Optional[float] = None,
        name: str = "",
    ):
        if constraints is None:
            constraints = []
        if dt_min is None:
            dt_min = dt_start * 1e-3
        if dt_max is None:
            dt_max = dt_start * 1e3
        return cls(
            t_start=t_start,
            dt_start=dt_start,
            constraints=constraints,
            dt_min=dt_min,
            dt_max=dt_max,
            name=name,
        )


class IntervalDict:
    def __init__(self, intervals: list[TimeInterval]) -> None:
        self.intervals = intervals
        self.interval_starts = [interval.t_start for interval in intervals]
        assert self.interval_starts == sorted(self.interval_starts)

    def get(self, time: float) -> tuple[TimeInterval, TimeInterval | None]:
        # [start, end)
        i = bisect_right(self.interval_starts, time) - 1
        if i < 0:
            raise ValueError
        if i >= len(self.intervals):
            raise ValueError
        current_interval = self.intervals[i]
        next_interval = self.intervals[i + 1] if i < (len(self.intervals) - 1) else None
        return current_interval, next_interval


class CannotRecomputeTimeStep(Exception):
    pass


class TimeScheduler:
    def __init__(
        self, intervals: list[TimeInterval], t_end: float, tol: float = 1e-8
    ) -> None:
        assert len(intervals) > 0
        self.intervals = intervals
        self.interval_dict = IntervalDict(intervals)
        self.time: float = intervals[0].t_start
        self.t_end: float = t_end
        self.dt: float = intervals[0].dt_start
        self.tol: float = tol

        increments = np.ediff1d(self.interval_dict.interval_starts + [self.t_end])
        assert np.all(increments > self.tol)

    def compute_next_time_step(self, success: bool, context: dict) -> float:
        current_interval, next_interval = self.interval_dict.get(time=self.time)
        dt = self.dt
        t = self.time
        t_end = self.t_end
        assert t <= t_end

        if success and self.is_hitting_schedule(current_interval):
            dt = current_interval.dt_start
            if dt < self.tol:
                logger.warning(
                    f"The new interval's initial step size ({dt:.1e}) is below the "
                    f"scheduler's minimum resolution ({self.tol:.1e}); quantizing it."
                )
                dt = self.tol

        next_checkpoint = t_end if next_interval is None else next_interval.t_start

        suggested_dt = [
            constraint.suggest_dt(dt=dt, context=context)
            for constraint in current_interval.constraints
        ]
        dt = min(suggested_dt)

        if t + dt > next_checkpoint:
            dt = next_checkpoint - t

        dt = min(dt, current_interval.dt_max)
        if dt < current_interval.dt_min:
            raise CannotRecomputeTimeStep(
                f"Adjusted time step size ({dt:.1e}) is lower than the minimum "
                f"admissible value ({current_interval.dt_min:.1e})."
            )

        self.dt = dt
        return dt

    def get_current_interval(self) -> TimeInterval:
        current_interval, _ = self.interval_dict.get(time=self.time)
        return current_interval

    def is_hitting_schedule(self, interval: TimeInterval) -> bool:
        t = self.time
        return interval.t_start < t < (interval.t_start + self.tol)

    def is_finished(self) -> bool:
        return self.time < self.t_end


# def assemble_default_time_scheduler()
