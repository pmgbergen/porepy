"""Collection of objects and functions related to time-step acceptance checking.

This includes:
- Status classes for time-step acceptance.
- Information classes for time-step checks.
- Base time-step criterion classes.
- Concrete acceptance and rejection criteria.

The design mirrors pp.numerics.nonlinear.convergence_check.

"""

from dataclasses import dataclass
import logging
from enum import StrEnum

from porepy.numerics.nonlinear.convergence_check import _recursive_append

logger = logging.getLogger(__name__)

# # ============================================================================
# # Status and Info Classes
# # ============================================================================


@dataclass
class TimeStepperStatus:
    pass


@dataclass
class TimeStepperStatusContinueIterating(TimeStepperStatus):
    pass


@dataclass
class TimeStepperStatusSuccess(TimeStepperStatus):
    pass


@dataclass
class TimeStepperStatusFailure(TimeStepperStatus):
    reason: str
