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

from porepy.numerics.nonlinear.nonlinear_solver_status import (
    NonlinearSolverStatus,
    NonlinearSolverStatusConverged,
)


@dataclass
class TimeStepperStatus(ABC):
    """A status object used to indicate the TimeStepper state."""

    @abstractmethod
    def serialize(self) -> str:
        """Return the stable string representation used in stored statistics.

        Note: Read the module docstring for a related discussion.

        """

    def is_success(self) -> bool:
        """Whether the time step is made successfully."""
        return isinstance(self, TimeStepperStatusSuccess)

    def is_failure(self) -> bool:
        """Whether the time step is failed and we gave up."""
        return isinstance(self, TimeStepperStatusFailure)


@dataclass
class TimeStepperStatusContinueIterating(TimeStepperStatus):
    """The TimeStepper attempted to make a time step, failed, but continue trying.

    Note: Read the module docstring for a related discussion.

    """

    attempt: int
    """Retry attempt number."""
    nonlinear_solver_status: NonlinearSolverStatus
    """Nonlinear solver status that caused the time step retry."""

    def serialize(self) -> str:
        return "in_progress"


@dataclass
class TimeStepperStatusSuccess(TimeStepperStatus):
    """The TimeStepper made a time step successfully."""

    nonlinear_solver_status: NonlinearSolverStatusConverged
    """Nonlinear solver status that caused the time step success."""

    def serialize(self) -> str:
        return "successful"


@dataclass
class TimeStepperStatusFailure(TimeStepperStatus):
    """The TimeStepper attempted to make a time step, but failed and gave up."""

    nonlinear_solver_status: NonlinearSolverStatus
    """Nonlinear solver status that caused the time step failure."""
    reason: str
    """Reason of failure."""

    def serialize(self) -> str:
        return "failed"
