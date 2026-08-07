"""Collection of objects related to time-step acceptance checking.

Note: Statuses (TimeStepperStatus, NonlinearSolverStatus etc.) serve two purposes: (i)
they are return codes that control the caller behavior, e.g., "the time step diverged";
(ii) they are used for statistics logging.

The latter requires a "in-progress" state, which will never be returned from a function.
After a discussion, EK, IS and YZ agreed this is not optimal and to be reconsidered. If
this happens, the `TimeStepperStatusContinueIterating` class and the `serialize` methods
are immediate candidates for removal.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from porepy.numerics import solvers

__all__ = [
    "TimeStepperStatus",
    "TimeStepperStatusSuccess",
    "TimeStepperStatusFailure",
]


@dataclass
class TimeStepperStatus(ABC):
    """A status object used to indicate the TimeStepper state. This is an enum of three
    allowed states: success / failure / continue_iterating. Each state can have data
    associated with it. `TimeStepperStatusSuccess`
    and `TimeStepperStatusFailure` can be subclassed to (i) introduce specific cases of
    these states and (ii) associate additional data with them. The base class
    `TimeStepperStatus` should NOT be subclassed.

    """

    @abstractmethod
    def serialize(self) -> str:
        """Return the stable string representation used in stored statistics.

        Note: Read the module docstring for a related discussion.

        """

    def is_success(self) -> bool:
        """Whether the time step is made successfully."""
        # Developer note: This breaks the OOP principle that the base class should not
        # know of its children, but we agreed on having these methods (is_success and
        # is_failure) for convenience. One can think of TimeStepperStatus as a closed
        # enum of 3 cases, which in this case justifies this binding with child classes.
        return isinstance(self, TimeStepperStatusSuccess)

    def is_failure(self) -> bool:
        """Whether the time step is failed and we gave up."""
        return isinstance(self, TimeStepperStatusFailure)


@dataclass
class TimeStepperAttemptData:
    dt: float
    nonlinear_solve_status: solvers.NonlinearSolverStatus


@dataclass
class TimeStepperStatusSuccess(TimeStepperStatus):
    """The TimeStepper made a time step successfully."""

    time: float
    """Simulation time at the start of the time step (t0)."""

    attempts: list[TimeStepperAttemptData]

    def serialize(self) -> str:
        return "successful"

    @property
    def dt(self) -> float:
        """Simulation time step magnitude."""
        if len(self.attempts) == 0:
            raise ValueError
        return self.attempts[-1].dt


@dataclass
class TimeStepperStatusFailure(TimeStepperStatus):
    """The TimeStepper attempted to make a time step, but failed and gave up."""

    reason: str
    """Reason of failure."""

    time: float
    """Simulation time at the start of the time step (t0)."""

    attempts: list[TimeStepperAttemptData]

    def serialize(self) -> str:
        return "failed"
