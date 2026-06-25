"""Collection of objects related to time-step acceptance checking.

"""

from abc import ABC
from dataclasses import dataclass


@dataclass
class TimeStepperStatus(ABC):
    """A status object used to indicate the TimeStepper state."""


@dataclass
class TimeStepperStatusContinueIterating(TimeStepperStatus):
    """The TimeStepper attempted to make a time step, failed, but continue trying."""


@dataclass
class TimeStepperStatusSuccess(TimeStepperStatus):
    """The TimeStepper made a time step successfully."""


@dataclass
class TimeStepperStatusFailure(TimeStepperStatus):
    """The TimeStepper attempted to make a time step, but failed and gave up."""
    reason: str
    """Reason of failure."""
