"""Module provides data structures related to simulation time stepping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

import porepy as pp
from porepy.time_stepper.time_step_constraint import TimeStepConstraint

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
            end simulation times, respectively.
        dt_init: Initial time step.
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
        advanced_schedule: A more advanced way to define the simulation schedule. See
            :class:`pp.time_stepper.TimeScheduler` for details. If None (default), the
            old way of defining the schedule with array is used. Otherwise, it is
            prioritized over the old schedule.

    """

    @classmethod
    def with_advanced_schedule(cls, schedule: Schedule) -> Self:
        """A factory method to consistently initialize the TimeManager with the advanced
        schedule.

        """
        if len(schedule.intervals) == 0:
            raise ValueError("Need at least a single time interval.")
        schedule_array = [interval.t_start for interval in schedule.intervals]
        schedule_array.append(schedule.t_end)
        return cls(
            schedule=schedule_array,
            dt_init=schedule.intervals[0].dt_start,
            constant_dt=False,
            advanced_schedule=schedule,
        )

    def __init__(
        self,
        schedule: ArrayLike,
        dt_init: pp.number,
        constant_dt: bool = False,
        dt_min_max: Optional[tuple[pp.number, pp.number]] = None,
        iter_max: Optional[int] = None,
        iter_optimal_range: tuple[int, int] = (4, 7),
        iter_relax_factors: tuple[float, float] = (0.7, 1.3),
        recomp_factor: float = 0.5,
        recomp_max: Optional[int] = None,
        rtol: Optional[float] = None,
        atol: float = 1e-16,
        advanced_schedule: Optional[Schedule] = None,
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
        self.advanced_schedule: Schedule | None = advanced_schedule
        """A more advanced way to define the simulation schedule. See
        :class:`pp.time_stepper.TimeScheduler` for details. If None (default), the old
        way of defining the schedule with array is used. Otherwise, is prioritized over
        the old schedule.

        """
        self.schedule = np.array(schedule, dtype=float)
        """Array of time points which the simulation must pass exactly within atol. The
        first and the last entries correspond to the start and the end simulation times,
        respectively.

        """
        if len(self.schedule) < 2:
            raise ValueError("Schedule must have at least two points: start and end.")
        self.time_init = float(self.schedule[0])
        """Initial simulation time."""
        self.time_final = float(self.schedule[-1])
        """Simulation end time."""
        self.dt_init = float(dt_init)
        """Initial time step."""
        self.dt_min_max = dt_min_max
        """Smallest and largest allowed time step."""
        self.iter_optimal_range = iter_optimal_range
        """Optimal range of nonlinear solver iterations. Passed to
        :class:`pp.time_stepper.TimeScheduler`.

        """
        self.iter_relax_factors = iter_relax_factors
        """Factors of how to decrease / increase dt if the nonlinear
        solver iterations are higher / lower than the optimal range. Passed to
        :class:`pp.time_stepper.TimeScheduler`.

        """
        self.recomp_factor = float(recomp_factor)
        """Factor used to reduce the time step after a failed nonlinear solve. Passed
        to :class:`pp.time_stepper.TimeScheduler`.

        """
        self.is_constant = constant_dt
        """Whether constant time stepping is requested."""

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
        return bool(np.any(abs(self.schedule - self.time) < self.atol))

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
