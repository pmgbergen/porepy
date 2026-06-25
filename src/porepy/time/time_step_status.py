"""Collection of objects related to time-step acceptance checking."""

from abc import ABC
from dataclasses import dataclass

from porepy.numerics.nonlinear.nonlinear_solver_status import (
    NonlinearSolverStatus,
    NonlinearSolverStatusConverged,
)


@dataclass
class TimeStepperStatus(ABC):
    """A status object used to indicate the TimeStepper state."""

    def is_success(self) -> bool:
        """Whether the time step is made successfully."""
        return isinstance(self, TimeStepperStatusSuccess)

    def is_failure(self) -> bool:
        """Whether the time step is failed and we gave up."""
        return isinstance(self, TimeStepperStatusFailure)


@dataclass
class TimeStepperStatusContinueIterating(TimeStepperStatus):
    """The TimeStepper attempted to make a time step, failed, but continue trying."""

    attempt: int
    """Retry attempt number."""
    nonlinear_solver_status: NonlinearSolverStatus
    """Nonlinear solver status that caused the time step retry."""


@dataclass
class TimeStepperStatusSuccess(TimeStepperStatus):
    """The TimeStepper made a time step successfully."""

    nonlinear_solver_status: NonlinearSolverStatusConverged
    """Nonlinear solver status that caused the time step success."""


@dataclass
class TimeStepperStatusFailure(TimeStepperStatus):
    """The TimeStepper attempted to make a time step, but failed and gave up."""

    nonlinear_solver_status: NonlinearSolverStatus
    """Nonlinear solver status that caused the time step failure."""
    reason: str
    """Reason of failure."""
