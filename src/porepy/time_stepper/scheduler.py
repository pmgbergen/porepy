from abc import ABC, abstractmethod
from bisect import bisect_right
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, cast

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
    "TimeSchedulerBase",
    "TimeScheduler",
    "SimulationTimeData",
    "assemble_default_time_scheduler",
]

logger = getLogger(__name__)



# TODO: I/O (Methods below are copied from TimeManager and should be removed).
class TimeIO:
    def __init__(self):
        self.exported_dt: list[float] = list()
        self.exported_times: list[float] = list()

    def write_time_information(self, time, dt, path: Path) -> None:
        """Keep track of history of time and time step size and store as json file
        storing lists the evolution of both as lists.

        NOTE: The history only contains time and dt for all occasions when this routine
        is called. This routine does neither guarantee completeness, nor duplicated.

        Parameters:
            path: Specified path for storing time and dt.

        """

        # Bookkeeping
        self.exported_times.append(
            int(time) if isinstance(time, np.integer) else float(time)
        )
        self.exported_dt.append(int(dt) if isinstance(dt, np.integer) else float(dt))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as out_file:
            json.dump({"time": self.exported_times, "dt": self.exported_dt}, out_file)

    def load_time_information(self, path: Path) -> None:
        """Keep track of history of time and time step size and store.

        Mirrors :meth:`write_time_information`.

        Parameters:
            path: Specified path for retrieving time and dt.

        """
        with path.open("r") as in_file:
            data = json.load(in_file)
            self.exported_times = data["time"]
            self.exported_dt = data["dt"]

    def set_time_and_dt_from_exported_steps(
        self, time_index: int = -1
    ) -> tuple[float, float]:
        """Load time and dt (time step) and cut off all later times and time steps.

        NOTE: This method by itself does NOT update the simulation state arrays.

        NOTE: It is implicitly assumed that the first entry of the history corresponds
        to the initial solution.

        Parameters:
            time_index: reference index addressing the currently stored history. By
                default, the latest accessible time and dt is retrieved.

        Raises:
            ValueError

        """
        if not hasattr(self, "exported_times") or not hasattr(self, "exported_dt"):
            raise ValueError(
                """The time manager does not hold information on previously used time
                and dt."""
            )

        time = self.exported_times[time_index]
        dt = self.exported_dt[time_index]

        self.exported_times = self.exported_times[:time_index]
        self.exported_dt = self.exported_dt[:time_index]
        return time, dt


@dataclass
class SimulationTimeData:
    time: float
    """At the end of the time step."""
    dt: float
    time_index_successful: int
    schedule: np.ndarray
    constant_dt: bool
    """Needed for backward compatability. Should be removed if not needed anymore. Why
    should a physics provider care if dt is constant?

    """
    io: TimeIO = TimeIO()
    """TODO: I/O"""

    def is_at_initial_time(self) -> bool:
        # TODO YZ: Should respect scheduler's t_snap.
        return self.time == self.schedule[0]

    def final_time_reached(self) -> bool:
        # TODO YZ: Should respect scheduler's t_snap.
        return self.time == self.schedule[-1]

    def write_time_information(self, path):
        assert self.io is not None
        self.io.write_time_information(time=self.time, dt=self.dt, path=path)

    def load_time_information(self, path):
        assert self.io is not None
        self.io.load_time_information(path)

    def set_time_and_dt_from_exported_steps(self, time_index):
        assert self.io is not None
        self.time, self.dt = self.io.set_time_and_dt_from_exported_steps(time_index)

    @property
    def exported_times(self):
        assert self.io is not None
        return self.io.exported_times

    @property
    def exported_dt(self):
        assert self.io is not None
        return self.io.exported_dt


class TimeStepConstraint(ABC):
    @abstractmethod
    def suggest_dt(self, dt: float, context: dict) -> float:
        pass


class TargetNonlinearIterations(TimeStepConstraint):
    def __init__(
        self,
        dt_min: float,
        iter_min: int = 4,
        iter_max: int = 7,
        increase_factor: float = 1.3,
        decrease_factor: float = 0.7,
        retry_factor: float = 0.5,
        t_snap: float = 1e-6,
    ) -> None:
        if iter_min > iter_max:
            raise ValueError(
                f"Incorrect optimal iteration range: [{iter_min, {iter_max}}]."
            )
        self.dt_min: float = dt_min
        self.iter_min = iter_min
        self.iter_max = iter_max
        if (
            not (increase_factor >= 1)
            or not (0 < decrease_factor <= 1)
            or not (0 < retry_factor < 1)
        ):
            raise ValueError(
                f"Incorrect adjustment factors: {increase_factor = }, "
                f"{decrease_factor = }, {retry_factor = }."
            )
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.retry_factor = retry_factor
        self.t_snap: float = t_snap

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
                # Decrease dt, but not below dt_min.
                return max(dt * self.decrease_factor, self.dt_min)
            else:
                return dt
        else:
            if abs(dt - self.dt_min) < self.t_snap:
                return dt * self.retry_factor
            return max(dt * self.retry_factor, self.dt_min)


class CourantTimeStepConstraint(TimeStepConstraint):
    def __init__(self, target_cfl: float = 1.0, tol: float = 1e-10) -> None:
        self.target_cfl = target_cfl
        self.tol = tol

    def suggest_dt(self, dt: float, context: dict) -> float:
        model = cast(pp.PorePyModel | None, context.get("model", None))
        if model is None:
            # TODO YZ: Should warn
            assert False
        dt = float("inf")
        for subdomain in model.mdg.subdomains():
            v = np.max(model.equation_system.evaluate(model.darcy_flux([subdomain])))
            if v < self.tol:
                continue
            x = subdomain.cell_diameters(cell_wise=False, func=np.max).min()
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
    def __init__(self, intervals: list[TimeInterval], atol: float) -> None:
        self.intervals = intervals
        self._interval_starts = [interval.t_start - atol for interval in intervals]
        assert self._interval_starts == sorted(self._interval_starts)

    def get(self, time: float) -> tuple[TimeInterval, TimeInterval | None]:
        # [start, end)
        i = bisect_right(self._interval_starts, time) - 1
        if i < 0:
            raise ValueError
        if i >= len(self.intervals):
            raise ValueError
        current_interval = self.intervals[i]
        next_interval = self.intervals[i + 1] if i < (len(self.intervals) - 1) else None
        return current_interval, next_interval


class CannotRecomputeTimeStep(Exception):
    pass


class TimeSchedulerBase(ABC):
    io: TimeIO
    """I/O bookkeeping for exported times, set by subclasses on construction."""

    @abstractmethod
    def get_schedule(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_time(self) -> float:
        """At the beginning of this time step."""

    @abstractmethod
    def get_time_end(self) -> float:
        pass

    @abstractmethod
    def get_dt(self) -> float:
        pass

    @abstractmethod
    def get_time_index_successful(self) -> int:
        pass

    @abstractmethod
    def compute_next_time_step(self, success: bool, context: dict) -> float:
        pass

    @abstractmethod
    def is_hitting_schedule(self) -> bool:
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        pass


class TimeSchedulerConstantDt(TimeSchedulerBase):
    def __init__(
        self, schedule: np.ndarray | list[float | int], dt: float, atol: float = 1e-8
    ) -> None:
        self.schedule = np.array(schedule, dtype=float)
        if len(self.schedule) < 2:
            raise ValueError("Schedule must have at least two points: start and end.")
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("Time step must be positive.")
        self.atol: float = atol
        self.time = float(self.schedule[0])
        self.time_index_successful: int = 0
        _validate_schedule_constant_dt(
            schedule=self.schedule, dt=self.dt, atol=self.atol
        )
        self.io = TimeIO()

    def get_schedule(self) -> np.ndarray:
        return self.schedule

    def get_time(self) -> float:
        return self.time

    def get_dt(self) -> float:
        return self.dt

    def get_time_end(self) -> float:
        return self.schedule[-1]

    def get_time_index_successful(self) -> int:
        return self.time_index_successful

    def compute_next_time_step(self, success: bool, context: dict) -> float:
        if not success:
            raise CannotRecomputeTimeStep(
                "Constant time scheduler cannot decrease time step size "
                f"({self.dt:.1e})."
            )
        self.time += self.dt
        self.time_index_successful += 1
        return self.dt

    def is_hitting_schedule(self) -> bool:
        return bool(np.any(abs(self.schedule - self.time) < self.atol))

    def is_finished(self) -> bool:
        return self.time >= (self.schedule[-1] - self.atol)


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
        raise ValueError


class TimeScheduler(TimeSchedulerBase):
    def __init__(
        self, intervals: list[TimeInterval], t_end: float, dt_snap: float = 1e-6
    ) -> None:
        if len(intervals) < 1:
            raise ValueError(
                "At least one interval must be passed, starting at t_start."
            )
        assert dt_snap > 0
        self.dt_snap: float = dt_snap
        self.intervals = intervals
        _validate_schedule_non_constant_dt(
            intervals=self.intervals, t_end=t_end, atol=self.dt_snap
        )
        self.time: float = intervals[0].t_start
        self.t_end: float = t_end
        self.dt: float = intervals[0].dt_start
        self.time_index_successful: int = 0
        self.interval_dict = IntervalDict(intervals, atol=self.dt_snap)

        current_interval, next_interval = self.get_current_next_intervals()
        self._adjust_dt_min_max_schedule(self.dt, current_interval, next_interval)

        self.io = TimeIO()

    def get_schedule(self) -> np.ndarray:
        return np.array(
            [interval.t_start for interval in self.intervals] + [self.t_end]
        )

    def get_time_index_successful(self) -> int:
        return self.time_index_successful

    def get_time(self) -> float:
        return self.time

    def get_dt(self) -> float:
        return self.dt

    def get_time_end(self) -> float:
        return self.t_end

    def compute_next_time_step(self, success: bool, context: dict) -> float:
        if success:
            self.time += self.dt
            self.time_index_successful += 1

        if self.is_finished():
            return self.dt

        current_interval, next_interval = self.interval_dict.get(time=self.time)
        dt = self.dt
        t_end = self.t_end
        next_checkpoint = t_end if next_interval is None else next_interval.t_start

        # If this is a start of a new interval, set dt to interval.dt_start and log.
        if success and self._is_hitting_interval_start(interval=current_interval):
            # We are at the start of a new interval.
            dt = current_interval.dt_start
            _log_schedule_interval_start(
                t_start=current_interval.t_start,
                t_end=next_checkpoint,
                name=current_interval.name,
            )

        # Apply constraints. If this is a start of a new interval, dt_start can be
        # adjusted by constraints as well.
        if len(current_interval.constraints) > 0:
            suggested_dt = [
                constraint.suggest_dt(dt=dt, context=context)
                for constraint in current_interval.constraints
            ]
            dt = min(suggested_dt)

        return self._adjust_dt_min_max_schedule(dt, current_interval, next_interval)

    def _adjust_dt_min_max_schedule(
        self,
        dt: float,
        current_interval: TimeInterval,
        next_interval: TimeInterval | None,
    ) -> float:
        next_checkpoint = self.t_end if next_interval is None else next_interval.t_start

        # Constraints should not make dt larger than the interval's dt_max.
        dt = min(dt, current_interval.dt_max)
        # If constraints made dt smaller than the interval's dt_min, abort simulation.
        if dt < current_interval.dt_min:
            raise CannotRecomputeTimeStep(
                f"Adjusted time step size ({dt:.1e}) is lower than the minimum "
                f"admissible value ({current_interval.dt_min:.1e})."
            )

        # Prevent overshooting the interval's end.
        if self.time + dt > next_checkpoint:
            dt = max(next_checkpoint - self.time, self.dt_snap)

        self.dt = dt
        return dt

    def is_finished(self) -> bool:
        return self.time >= (self.t_end - self.dt_snap)

    def get_current_next_intervals(self) -> tuple[TimeInterval, TimeInterval | None]:
        return self.interval_dict.get(time=self.time)

    def is_hitting_schedule(self) -> bool:
        if (self.t_end - self.dt_snap) <= self.time <= (self.t_end + self.dt_snap):
            return True
        current_interval, _ = self.interval_dict.get(time=self.time)
        return self._is_hitting_interval_start(current_interval)

    def _is_hitting_interval_start(self, interval: TimeInterval) -> bool:
        return (
            (interval.t_start - self.dt_snap)
            <= self.time
            <= (interval.t_start + self.dt_snap)
        )


def assemble_default_time_scheduler(
    schedule: np.ndarray | list,
    dt_init: float,
    constant_dt: bool = False,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
    nonlinear_iter_optimal_range: tuple[int, int] = (4, 7),
    nonlinear_iter_relax_factors: tuple[float, float] = (0.7, 1.3),
    nonlinear_iter_retry_factor: float = 0.5,
    constraints: Optional[list[TimeStepConstraint]] = None,
    atol: float = 1e-8,
) -> TimeSchedulerBase:
    if len(schedule) < 2:
        raise ValueError("Schedule must have at least two points (t_start and t_end).")

    schedule = np.array(schedule, dtype=float)
    if dt_min is None:
        dt_min = float(dt_init) * 1e-3
    if dt_max is None:
        dt_max = float(dt_init) * 1e3
    iter_min, iter_max = nonlinear_iter_optimal_range
    decrease_factor, increase_factor = nonlinear_iter_relax_factors
    if constraints is None:
        constraints = []
    if not constant_dt:
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
            dt_snap=atol,
        )
    else:
        dt_min = dt_max = dt_init
        return TimeSchedulerConstantDt(
            schedule=schedule,
            dt=dt_init,
            atol=atol,
        )


def _log_schedule_interval_start(t_start: float, t_end: float, name: str = "") -> None:
    log_message = "Reached new schedule interval"
    if name != "":
        log_message += f' "{name}"'
    delta = t_end - t_start
    log_message += f" [{t_start:.1e}, {t_start:.1e} + {delta:.1e})."
    logger.info(log_message)
