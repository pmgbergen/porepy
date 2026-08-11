from abc import ABC, abstractmethod
from bisect import bisect_right
from dataclasses import dataclass
from typing import Optional

import numpy as np

import porepy as pp

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
    "CourantTimeStepConstraint",
    "TimeScheduler",
    "SimulationTimeData",
    "assemble_default_time_scheduler",
]

logger = getLogger(__name__)


@dataclass(frozen=True)
class SimulationTimeData:
    time: float
    dt: float


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


class CourantTimeStepConstraint(TimeStepConstraint):
    def __init__(self, target_cfl: float = 1.0, tol: float = 1e-10) -> None:
        self.target_cfl = target_cfl
        self.tol = tol

    def suggest_dt(self, dt: float, context: dict) -> float:
        model: pp.PorePyModel | None = context.get("model", None)
        if model is None:
            # TODO YZ: Should warn
            assert False
        dt = float("inf")
        for subdomain in model.mdg.subdomains():
            v = model.equation_system.evaluate(model.darcy_flux([subdomain])).max()
            if v < self.tol:
                continue
            x = subdomain.cell_diameters().min()
            dt = min(dt, self.target_cfl * x / v)
        return dt


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
        self.time_step_index: int = 0
        """Not used for anything, just for logging."""

        increments = np.ediff1d(self.interval_dict.interval_starts + [self.t_end])
        assert np.all(increments > self.tol)

    def compute_next_time_step(self, success: bool, context: dict) -> float:
        if self.is_finished():
            return self.dt

        current_interval, next_interval = self.interval_dict.get(time=self.time)
        dt = self.dt
        t = self.time
        t_end = self.t_end
        assert t <= t_end

        next_checkpoint = t_end if next_interval is None else next_interval.t_start
        if success and self.is_hitting_schedule(interval=current_interval):
            dt = current_interval.dt_start
            if dt < self.tol:
                logger.warning(
                    f"The new interval's initial step size ({dt:.1e}) is below the "
                    f"scheduler's minimum resolution ({self.tol:.1e}); quantizing it."
                )
                dt = self.tol
            log_message = "Reached new schedule interval"
            if current_interval.name != "":
                log_message += f' "{current_interval.name}"'
            delta = next_checkpoint - current_interval.t_start
            log_message += (
                f" [{current_interval.t_start:.1e}, {current_interval.t_start:.1e} + "
                f"{delta:.1e})."
            )

            logger.info(log_message)

        if len(current_interval.constraints) > 0:
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
        return interval.t_start <= self.time < (interval.t_start + self.tol)

    def is_finished(self) -> bool:
        return self.time >= self.t_end


def assemble_default_time_scheduler(
    schedule: np.ndarray,
    dt_init: float,
    constant_dt: bool,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
    nonlinear_iter_optimal_range: tuple[int, int] = (4, 7),
    nonlinear_iter_relax_factors: tuple[float, float] = (0.7, 1.3),
    nonlinear_iter_retry_factor: float = 0.5,
) -> TimeScheduler:
    iter_min, iter_max = nonlinear_iter_optimal_range
    increase_factor, decrease_factor = nonlinear_iter_relax_factors
    constraints: list[TimeStepConstraint] = []
    if not constant_dt:
        constraints.append(
            TargetNonlinearIterations(
                iter_min=iter_min,
                iter_max=iter_max,
                increase_factor=increase_factor,
                decrease_factor=decrease_factor,
                retry_factor=nonlinear_iter_retry_factor,
            )
        )
    return TimeScheduler(
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
    )
