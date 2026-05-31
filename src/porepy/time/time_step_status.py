"""Collection of objects and functions related to time-step acceptance checking.

This includes:
- Status classes for time-step acceptance.
- Information classes for time-step checks.
- Base time-step criterion classes.
- Concrete acceptance and rejection criteria.

The design mirrors pp.numerics.nonlinear.convergence_check.

"""

import logging
from enum import StrEnum

from porepy.numerics.nonlinear.convergence_check import _recursive_append

logger = logging.getLogger(__name__)

# ============================================================================
# Status and Info Classes
# ============================================================================


class TimeStepStatus(StrEnum):
    """Enumeration of potential time-step acceptance statuses."""

    ACCEPTED = "accepted"
    """Time step is accepted; time advances."""
    REJECTED = "rejected"
    """Time step is rejected; dt is reduced and trial repeats."""
    STOPPED = "stopped"
    """Time stepping was stopped due to an error or max retries exceeded."""

    def __str__(self):
        return self.value

    def is_accepted(self) -> bool:
        """Check if the status indicates acceptance."""
        return self == TimeStepStatus.ACCEPTED

    def is_rejected(self) -> bool:
        """Check if the status indicates rejection."""
        return self == TimeStepStatus.REJECTED

    def is_stopped(self) -> bool:
        """Check if the status indicates stopping."""
        return self == TimeStepStatus.STOPPED


TimeStepInfo = str
"""Expected type for time-step check information."""

TimeStepInfoCollection = dict[str, TimeStepInfo]
"""Collection of time-step information for a collection of criteria."""


# ============================================================================
# Status and Info Collection Classes
# ============================================================================


class TimeStepStatusCollection(dict[str, TimeStepStatus]):
    """Collection of time-step statuses for a collection of criteria.

    Keys are criterion names; values are TimeStepStatus objects.
    """

    def is_accepted(self) -> bool:
        """Check if all statuses indicate acceptance."""
        return all(status.is_accepted() for status in self.values())

    def is_rejected(self) -> bool:
        """Check if any status indicates rejection."""
        return any(status.is_rejected() for status in self.values())

    def is_stopped(self) -> bool:
        """Check if any status indicates stopping."""
        return any(status.is_stopped() for status in self.values())

    def union(self, other: "TimeStepStatusCollection") -> "TimeStepStatusCollection":
        """Union of two TimeStepStatusCollection needing to be disjunct."""
        result = TimeStepStatusCollection()
        assert len(set(self.keys()).intersection(other.keys())) == 0
        result.update(self)
        result.update(other)
        return result


class TimeStepStatusHistory(dict[str, list[TimeStepStatus]]):
    """Collection of time-step statuses in the form of nested dictionaries.

    The keys are the names of the criteria, and the values are lists of
    time-step statuses recorded over time steps.
    """

    def to_str(self) -> dict:
        """Convert the time-step statuses to strings.

        Returns:
            dict[str, list[str]]: Time-step statuses as strings.
        """
        return {k: [str(s) for s in v] for k, v in self.items()}

    def append(self, status: TimeStepStatusCollection) -> None:
        """Append another TimeStepStatusCollection to this one.

        Parameters:
            status: Time-step statuses to append.
        """
        _recursive_append(self, status)


class TimeStepInfoHistory(dict[str, list[str] | dict[str, list[str]]]):
    """Collection of time-step information with lists at the leafs.

    Used to track diagnostic information from time-step acceptance checks
    over multiple time steps.
    """

    def append(self, info: TimeStepInfoCollection) -> None:
        """Append another TimeStepInfoCollection to this one.

        Parameters:
            info: Time-step information to append.
        """
        _recursive_append(self, info)
