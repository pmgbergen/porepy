"""Module provides data structures related to simulation time stepping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

import porepy as pp
from porepy.time_stepper.time_step_constraint import (
    TargetNonlinearIterations,
    TimeStepConstraint,
)

__all__ = [
    "TimeManager",
    "TimeInterval",
    "Schedule",
]


class TimeManager:
    """The source of truth about the simulation time and time step.

    This class should be seen as a data structure with simulation time-related
    information. By convention, the only place where these data (time, dt, etc.) can be
    modified is the :class:`pp.time_stepper.TimeStepper`. It is responsible for
    advancing the simulation time and retracting it in case of a failed time step.

    Others, including PorePy models, should treat this class as a read-only data
    structure with a few read-only convenience methods. The name "TimeManager" remains
    for historical reasons, it in fact does not manage anything.

    One more responsibility on this class is to read and write time information on disk.
    Corresponding methods (in I/O section) are the known exception from the convention
    above due to historical reasons.

    Most of the __init__ parameters correspond to the time step control, which is
    out of scope of this class. They are kept for a legacy reason, and used to
    initialize :class:`pp.time_stepper.TimeScheduler`.

    Parameters:
        schedule: Array of time points which the simulation must pass exactly within
            tolerance. The first and the last entries correspond to the start and the
            end simulation times, respectively. Alternatively, the schedule data
            structure, used to define a more refined schedule.
        dt_init: Initial time step. Must be passed if the `schedule` is an array.
            Ignored otherwise.
        constant_dt: If True, constant time stepping is requested. Otherwise, the
            scheduler can adjust dt.
        dt_min_max: Smallest and largest allowed time step.
        iter_max: Deprecated, does nothing. Control the nonlinear iteration limit via
            nonlinear solver parameters. See, e.g., :class:`pp.solvers.NewtonSolver`.
        iter_optimal_range: Optimal range of nonlinear solver iterations. Passed to
            :class:`pp.time_stepper.TimeScheduler`.
        iter_relax_factors: Factors of how to decrease / increase dt if the nonlinear
            solver iterations are higher / lower than the optimal range. Passed to
            :class:`pp.time_stepper.TimeScheduler`.
        recomp_factor: Factor of how to decrease dt if the nonlinear solver failed and
            the time step is to be recomputed. Passed to
            :class:`pp.time_stepper.TimeScheduler`.
        recomp_max: Deprecated, does nothing. Control the number of attempts to make a
            time step in :class:`pp.time_stepper.TimeStepper`.
        rtol: Deprecated, does nothing.
        atol: Snapping time. If the time difference is below it, treats two time points
            as equal.

    """

    def __init__(
        self,
        schedule: ArrayLike | Schedule,
        dt_init: Optional[pp.number] = None,
        constant_dt: bool = False,
        dt_min_max: Optional[tuple[pp.number, pp.number]] = None,
        iter_max: Optional[int] = None,
        iter_optimal_range: tuple[int, int] = (4, 7),
        iter_relax_factors: tuple[float, float] = (0.7, 1.3),
        recomp_factor: float = 0.5,
        recomp_max: Optional[int] = None,
        rtol: Optional[float] = None,
        atol: float = 1e-16,
    ) -> None:
        if iter_max is not None:
            warn(
                message=(
                    "TimeManager.iter_max is deprecated and does nothing. Control the "
                    "nonlinear iteration limit via nonlinear solver parameters. See "
                    "pp.solvers.NewtonSolver."
                ),
                category=FutureWarning,
                stacklevel=2,
            )
        if recomp_max is not None:
            warn(
                message=(
                    "TimeManager.recomp_max is deprecated and does nothing. Control "
                    "the number of attempts to make a time step in "
                    "pp.time_stepper.TimeStepper."
                ),
                category=FutureWarning,
                stacklevel=2,
            )
        if rtol is not None:
            warn(
                message=("TimeManager.rtol is deprecated and does nothing. Use atol."),
                category=FutureWarning,
                stacklevel=2,
            )

        self.atol = atol
        """Snapping time. If the time difference is below it, treats two time points
        as equal.

        """

        if isinstance(schedule, Schedule):
            if dt_init is not None:
                warn(
                    "dt_init argument is ignored if Schedule object is passed.",
                    stacklevel=2,
                )
        elif isinstance(schedule, (list, tuple, np.ndarray)):
            if dt_init is None:
                raise ValueError(
                    "Passing the schedule as an array requires to pass dt_init."
                )

            if dt_min_max is None:
                dt_min = dt_max = None
            else:
                dt_min, dt_max = dt_min_max
            schedule = Schedule.assemble_default(
                schedule=schedule,
                dt_init=dt_init,
                constant_dt=constant_dt,
                dt_min=dt_min,
                dt_max=dt_max,
                iter_optimal_range=iter_optimal_range,
                iter_relax_factors=iter_relax_factors,
                recomp_factor=recomp_factor,
            )
        else:
            raise ValueError(f"Unsupported schedule format: {type(schedule)}")
        self.schedule: Schedule = schedule
        """A simulation schedule defined by its intervals and the end time. See
        :class:`pp.time_stepper.TimeScheduler` for details on how it is used.

        Migration notice: Previously, this attribute was a numpy array. To fix errors in
        the runscripts and preserve the old behavior, replace time_manager.schedule with
        time_manager.schedule.get_array().

        """

        if len(self.schedule.intervals) < 1:
            raise ValueError("Schedule must have at least one interval.")

        # Legacy properties. Accessed through read-only getters and should not be
        # modified, since it does not affect anything.
        self._iter_optimal_range = iter_optimal_range
        self._iter_relax_factors = iter_relax_factors
        self._recomp_factor = recomp_factor
        self._is_constant = constant_dt

        self.time = float(self.time_init)
        """Current simulation time, seconds. If accessed from within the PorePy model
        simulation loop, corresponds to the implicit trial time, where the unknown
        solution is defined. E.g., in the very first time step with dt = 0.5, time =
        0.5.
        
        """
        self.dt = float(self.dt_init)
        """Current time step size, seconds."""

        self.time_index: int = 0
        """Counter of successful time steps. E.g., the simulation attempted to make the
        very first time step 3 times: failed, failed and succeeded. Then,
        time_index = 1.
        
        """

        # Bookkeeping of saved time steps for restarting purposes.
        self.exported_dt: list[pp.number] = []
        """A list of time steps for the simulation states that were saved on disk with
        `write_time_information` for restarting purposes. Completeness and lack of
        duplication are NOT guaranteed.

        NOTE: This property cannot be inferred from `exported_times`, consider the case
        when not every time step is saved.

        """
        self.exported_times: list[pp.number] = []
        """A list of time points for the simulation states that were saved on disk with
        `write_time_information` for restarting purposes. Completeness and lack of
        duplication are NOT guaranteed.

        """

    @property
    def time_init(self) -> float:
        """Initial simulation time."""
        return self.schedule.intervals[0].t_start

    @property
    def time_final(self) -> float:
        """Simulation end time."""
        return self.schedule.t_end

    @property
    def dt_init(self) -> float:
        """Initial time step."""
        return self.schedule.intervals[0].dt_start

    @property
    def dt_min_max(self) -> tuple[float, float]:
        """Smallest and largest allowed time step."""
        return (
            min(interval.dt_min for interval in self.schedule.intervals),
            max(interval.dt_max for interval in self.schedule.intervals),
        )

    @property
    def iter_optimal_range(self) -> tuple[int, int]:
        """Optimal range of nonlinear solver iterations. Passed to
        :class:`pp.time_stepper.TimeScheduler`.

        """
        return self._iter_optimal_range

    @property
    def iter_relax_factors(self) -> tuple[float, float]:
        """Factors of how to decrease / increase dt if the nonlinear solver iterations
        are higher / lower than the optimal range. Passed to
        :class:`pp.time_stepper.TimeScheduler`.

        """
        return self._iter_relax_factors

    @property
    def recomp_factor(self) -> float:
        """Factor used to reduce the time step after a failed nonlinear solve. Passed
        to :class:`pp.time_stepper.TimeScheduler`. Passed to
        :class:`pp.time_stepper.TimeScheduler`.

        """
        return self._recomp_factor

    @property
    def is_constant(self) -> bool:
        """Whether constant time stepping is requested."""
        return self._is_constant

    def __repr__(self) -> str:
        s = "Time-stepping control object with attributes:\n"
        s += "Initial and final simulation time = "
        s += f"({self.time_init}, {self.time_final})\n"
        s += f"Initial time step = {self.dt_init}\n"
        s += f"Current time step and time are {self.dt} and {self.time}."

        return s

    def elapsed_time(self) -> float:
        """Return the elapsed simulation time."""
        return self.time - self.time_init

    def is_at_initial_time(self) -> bool:
        """Check whether the time manager is at the initial time."""
        return self.time < (self.time_init + self.atol)

    def is_at_schedule_point(self) -> bool:
        """Check whether the time manager is hitting any schedule point."""
        return bool(np.any(abs(self.schedule.get_array() - self.time) < self.atol))

    def final_time_reached(self) -> bool:
        """Check whether the time manager has reached the end of the schedule.

        Returns:
            Whether the final time has reached or been overstepped.

        """
        return self.time >= (self.time_final - self.atol)

    # I/O
    def write_time_information(self, path: Path) -> None:
        """Keep track of history of time and time step size and store as json file
        storing lists the evolution of both as lists.

        NOTE: The history only contains time and dt for all occasions when this routine
        is called. This routine does neither guarantee completeness, nor duplicated.

        Parameters:
            path: Specified path for storing time and dt.

        """

        # Bookkeeping
        self.exported_times.append(
            int(self.time) if isinstance(self.time, np.integer) else float(self.time)
        )
        self.exported_dt.append(
            int(self.dt) if isinstance(self.dt, np.integer) else float(self.dt)
        )
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

    def set_time_and_dt_from_exported_steps(self, time_index: int = -1) -> None:
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

        self.time = self.exported_times[time_index]
        self.dt = self.exported_dt[time_index]

        self.exported_times = self.exported_times[:time_index]
        self.exported_dt = self.exported_dt[:time_index]


@dataclass
class TimeInterval:
    """A data structure defining a time interval in the schedule.

    The interval is defined by its `t_start`, `t_end` is not explicitly specified, so
    the interval ends when the next interval starts.

    """

    t_start: float
    """Start time of the interval, seconds."""
    dt_start: float
    """Desired time step applied at the interval start, seconds."""
    constraints: list[pp.time_stepper.TimeStepConstraint]
    """List of constraints that control dt based on simulation behavior."""
    dt_min: float
    """Minimum time step allowed for this interval, seconds."""
    dt_max: float
    """Maximum time step allowed for this interval, seconds."""
    name: str
    """Interval name. Used for debugging. No logic is bound to it."""

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
        """Convenience constructor with defaults. If `dt_min` or `dt_max` are not
        specified, sets them to 3 magnitudes smaller / larger than `dt_start`,
        respectively.

        """
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


@dataclass
class Schedule:
    """A data structure that combines the list of intervals, and the whole simulation's
    end time.

    """

    intervals: list[TimeInterval]
    """Simulation's time intervals."""
    t_end: float
    """Simulation end time, seconds."""

    def get_array(self) -> np.ndarray:
        """Returns schedule points as a numpy array, including the simulation start and
        end time."""
        return np.array([i.t_start for i in self.intervals] + [self.t_end], dtype=float)

    @staticmethod
    def assemble_default(
        schedule: ArrayLike,
        dt_init: pp.number,
        constant_dt: bool = False,
        dt_min: Optional[pp.number] = None,
        dt_max: Optional[pp.number] = None,
        iter_optimal_range: tuple[int, int] = (4, 7),
        iter_relax_factors: tuple[float, float] = (0.7, 1.3),
        recomp_factor: float = 0.5,
    ) -> Schedule:
        """Convenience factory that constructs the schedule based on the parameters.

        Parameters:
            schedule: Array of schedule points. Must include at least two points: start
                and end.
            dt_init: Initial time step.
            constant_dt: If False, initializes TargetNonlinearIterations constraint for
                all intervals. If True, does not initialize any constraints.
            dt_min: Minimal dt for all the intervals.
            dt_max: Maximum dt for all the intervals.
            iter_optimal_range: Target range of nonlinear iterations. Ignored if
                `constant_dt == True`.
            iter_relax_factors: Decrease and increase factors for
                TargetNonlinearIterations. Ignored if `constant_dt == True`.
            recomp_factor: Decrease factor for failed solve attempts in
                TargetNonlinearIterations. Ignored if `constant_dt == True`.


        """
        schedule = np.array(schedule, dtype=float)
        if len(schedule) < 2:
            raise ValueError(
                "Schedule must have at least two points (t_start and t_end)."
            )

        constraints: list[TimeStepConstraint] = []
        if not constant_dt:
            constraints.append(
                TargetNonlinearIterations(
                    iter_min=iter_optimal_range[0],
                    iter_max=iter_optimal_range[1],
                    decrease_factor=iter_relax_factors[0],
                    increase_factor=iter_relax_factors[1],
                    retry_factor=recomp_factor,
                )
            )
        return Schedule(
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
