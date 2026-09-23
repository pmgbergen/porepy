"""Collection of objects and functions related to convergence checking.

This includes:
- Status classes for simulation and convergence.
- Information classes for convergence.
- Base convergence criterion classes.
- Absolute and relative convergence criteria for nonlinear problems.
- Divergence criteria for detecting divergence.

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import copy
from enum import StrEnum
from typing import Callable, Optional, cast

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)

__all__ = [
    "ConvergenceStatus",
    "ConvergenceStatusCollection",
    "ConvergenceStatusHistory",
    "ConvergenceInfo",
    "ConvergenceInfoCollection",
    "ConvergenceMetricType",
    "ConvergenceInfoHistory",
    "ConvergenceCriterion",
    "DivergenceCriterion",
    "ConvergenceCriteria",
    "DivergenceCriteria",
    "NanDivergenceCriterion",
    "AbsoluteCriterion",
    "AbsoluteConvergenceCriterion",
    "AbsoluteDivergenceCriterion",
    "RelativeCriterion",
    "RelativeConvergenceCriterion",
    "RelativeDivergenceCriterion",
    "CombinedCriterion",
    "CombinedConvergenceCriterion",
    "CombinedDivergenceCriterion",
    "IncrementBasedNanCriterion",
    "ResidualBasedNanCriterion",
    "IncrementBasedAbsoluteDivergenceCriterion",
    "ResidualBasedAbsoluteDivergenceCriterion",
    "IncrementBasedRelativeDivergenceCriterion",
    "ResidualBasedRelativeDivergenceCriterion",
    "IncrementBasedAbsoluteCriterion",
    "IncrementBasedRelativeCriterion",
    "ResidualBasedAbsoluteCriterion",
    "ResidualBasedRelativeCriterion",
    "IncrementBasedCombinedCriterion",
    "ResidualBasedCombinedCriterion",
    "IncrementBasedCombinedDivergenceCriterion",
    "ResidualBasedCombinedDivergenceCriterion",
    "MaxIterationsCriterion",
    "assemble_default_convergence_criteria",
    "has_active_criterion",
    "assemble_default_divergence_criteria",
]

# Status and info classes


class ConvergenceStatus(StrEnum):
    """Enumeration of potential convergence statuses."""

    CONVERGED = "converged"
    """Convergence criterion is satisfied."""
    CONTINUE_ITERATING = "continue_iterating"
    """We neither succeeded nor failed: should continue iterating."""
    FAILED = "failed"
    """Divergence criterion flags a failure: divergence, exception, iteration limit,
    etc.

    """

    def __str__(self):
        return self.value

    def is_converged(self) -> bool:
        """Check if the status indicates convergence."""
        return self == ConvergenceStatus.CONVERGED

    def is_iterating(self) -> bool:
        """Check if the status indicates neither converged nor failed."""
        return self == ConvergenceStatus.CONTINUE_ITERATING

    def is_failed(self) -> bool:
        """Check if the status indicates divergence."""
        return self == ConvergenceStatus.FAILED


class ConvergenceStatusCollection(dict[str, ConvergenceStatus]):
    """Collection of convergence statuses for a collection of criteria."""

    def is_converged(self) -> bool:
        """Check if all statuses indicate convergence."""
        return all(status.is_converged() for status in self.values())

    def is_iterating(self) -> bool:
        """Check if any status indicates not converged."""
        return any(status.is_iterating() for status in self.values())

    def is_failed(self) -> bool:
        """Check if any status indicates divergence."""
        return any(status.is_failed() for status in self.values())

    def union(
        self, other: "ConvergenceStatusCollection"
    ) -> "ConvergenceStatusCollection":
        """Union of two ConvergenceStatusCollection needing to be disjunct."""
        result = ConvergenceStatusCollection()
        assert len(set(self.keys()).intersection(other.keys())) == 0
        result.update(self)
        result.update(other)
        return result


def _make_leafs_to_list(d: dict) -> dict:
    """Auxiliary function to convert leafs of a dictionary to lists."""
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = _make_leafs_to_list(value)
        else:
            d[key] = [value]
    return d


def _recursive_append(d: dict, v: dict) -> dict:
    """Auxiliary function to recursively append dictionaries."""
    for key_v, value_v in v.items():
        if key_v not in d:
            if isinstance(value_v, dict):
                d[key_v] = _make_leafs_to_list(copy(value_v))
            else:
                d[key_v] = [copy(value_v)]
        else:
            if isinstance(d[key_v], dict):
                assert isinstance(value_v, dict)
                d[key_v] = _recursive_append(d[key_v], value_v)
            elif isinstance(d[key_v], list):
                d[key_v].append(value_v)
            else:
                assert type(d[key_v]) == type(value_v)
                d[key_v] = [d[key_v], value_v]

    return d


class ConvergenceStatusHistory(dict[str, list[ConvergenceStatus]]):
    """Collection of convergence statuses in form of nested dictionaries.

    The keys are the names of the criteria, and the values are lists of convergence
    statuses, e.g., recorded over iterations, as used in the
    :class:`pp.SolverStatistics`.

    """

    def to_str(self) -> dict:
        """Convert the convergence statuses to strings.

        Returns:
            dict[str, list[str]]: Convergence statuses as strings.

        """
        return {k: [str(s) for s in v] for k, v in self.items()}

    def append(self, status: ConvergenceStatusCollection) -> None:
        """Append another ConvergenceStatusCollection to this one.

        Parameters:
            status: Convergence statuses to append.

        """
        # Since this class inherits from dict, we should always be a dict
        # The recursive append modifies self in-place, so no reassignment needed
        _recursive_append(self, status)


ConvergenceInfo = float | dict[str, float]
"""Expected type for convergence information."""

ConvergenceInfoCollection = dict[str, ConvergenceInfo]
"""Collection of convergence information for a collection of criteria."""

ConvergenceMetricType = Callable[[np.ndarray], float | dict]
"""Type annotation for a convergence metric function. It takes an array and returns
either a single float or a dictionary of keys (e.g. equation / variable names) to float
values.

"""


class ConvergenceInfoHistory(dict[str, list[float] | dict[str, list[float]]]):
    """Collection of convergence information with list at the leafs."""

    def append(self, convergence_info: ConvergenceInfoCollection) -> None:
        """Append another ConvergenceInfoCollection to this one.

        Parameters:
            convergence_info: Convergence information to append.

        """
        _recursive_append(self, convergence_info)


# Base convergence criterion classes.


class ConvergenceCriterion(ABC):
    """Base class for convergence criteria."""

    @abstractmethod
    def check(self, *args, **kwargs) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Check convergence.

        Parameters:
            kwargs: Quantities to check for convergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.
            ConvergenceInfo: Information about the convergence check.

        """
        pass

    def reset(self) -> None:
        """Reset any internal state of the convergence criterion."""
        pass


class DivergenceCriterion(ABC):
    """Divergence criterion."""

    @abstractmethod
    def check(self, *args, **kwargs) -> ConvergenceStatus:
        """Check convergence.

        Parameters:
            kwargs: Quantities to check for convergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.
            float: Information about the convergence check.

        """
        pass

    def reset(self) -> None:
        """Reset any internal state of the divergence criterion."""
        pass

    def divergence_msg(self) -> str:
        """Return a message describing the divergence criterion."""
        return f"\033[91m{self.__class__.__name__} triggered divergence.\033[0m"


class ConvergenceCriteria(dict[str, ConvergenceCriterion]):
    """Collection of convergence criteria."""

    def check(
        self, *args, **kwargs
    ) -> tuple[ConvergenceStatusCollection, ConvergenceInfoCollection]:
        """Check convergence using all criteria in the collection.

        Parameters:
            args: Positional arguments for the convergence checks.
            kwargs: Keyword arguments for the convergence checks.

        Returns:
            tuple[ConvergenceStatusCollection, dict]: Convergence statuses with the
                names of the criteria as keys, and information about the
                convergence checks (format of the values depends on used metrics).

        """
        status = ConvergenceStatusCollection()
        info = ConvergenceInfoCollection()
        for name, criterion in self.items():
            stat, inf = criterion.check(*args, **kwargs)
            status[name] = stat
            info[name] = inf
        return status, info

    def reset(self) -> None:
        """Reset all convergence criteria in the collection."""
        for criterion in self.values():
            criterion.reset()


def has_active_criterion(criteria: dict[str, ConvergenceCriterion]) -> bool:
    """Check whether any criterion in a collection actually tests something.

    A criterion whose tolerances are all infinite reports convergence unconditionally,
    so a collection of only such criteria would declare the very first iteration
    converged. A criterion exposing no tolerance at all is assumed to test something.

    Parameters:
        criteria: The convergence criteria to inspect.

    Returns:
        True if at least one criterion has a finite tolerance.

    """

    def is_active(criterion: ConvergenceCriterion) -> bool:
        tolerances = [
            getattr(criterion, name)
            for name in ("tol", "atol", "rtol")
            if hasattr(criterion, name)
        ]
        return not tolerances or not all(np.isinf(tol) for tol in tolerances)

    return any(is_active(criterion) for criterion in criteria.values())


class DivergenceCriteria(dict[str, DivergenceCriterion]):
    """Collection of divergence criteria."""

    def check(self, *args, **kwargs) -> ConvergenceStatusCollection:
        """Check convergence using all criteria in the collection.

        Parameters:
            args: Positional arguments for the divergence checks.
            kwargs: Keyword arguments for the divergence checks.

        Returns:
            ConvergenceStatusCollection: Divergence statuses of the non-linear iteration
                with the names of the criteria as keys.

        """
        status = ConvergenceStatusCollection()
        for name, criterion in self.items():
            status[name] = criterion.check(*args, **kwargs)
        return status

    def reset(self) -> None:
        """Reset all divergence criteria in the collection."""
        for criterion in self.values():
            criterion.reset()


class NanDivergenceCriterion(DivergenceCriterion):
    """Divergence criterion, that checks for NaN values."""

    def check(self, *args, **kwargs) -> ConvergenceStatus:
        """Check for NaN values in the nonlinear increment and residual.

        Parameters:
            args: Positional arguments for the convergence check.
            kwargs: Quantities to check for NaN values.
                - value: The value to check for NaN values.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        if np.isnan(kwargs["value"]).any():
            logger.info(self.divergence_msg())
            return ConvergenceStatus.FAILED
        return ConvergenceStatus.CONVERGED


class AbsoluteCriterion:
    def __init__(
        self,
        tol: float,
        metric: ConvergenceMetricType,
    ) -> None:
        self.tol = tol
        """Tolerance for convergence - criterion in active if set to `np.inf`."""
        self.metric = metric
        """Metric to compute the convergence measure."""

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(tol={self.tol}, "
        s += f"metric={self.metric.__class__.__name__})"
        return s


class AbsoluteConvergenceCriterion(AbsoluteCriterion, ConvergenceCriterion):
    """Absolute convergence criterion."""

    def check(self, *args, **kwargs) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Check convergence.

        Parameters:
            args: Positional arguments for the convergence check.
            kwargs: Quantities to check for convergence.
                - value: The value to check for convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of
                the non-linear iteration and information about the convergence check.

        """
        metric_value = self.metric(kwargs["value"])
        if isinstance(metric_value, dict):
            status = (
                ConvergenceStatus.CONVERGED
                if all(v < self.tol for v in metric_value.values())
                else ConvergenceStatus.CONTINUE_ITERATING
            )
        else:
            status = (
                ConvergenceStatus.CONVERGED
                if metric_value < self.tol
                else ConvergenceStatus.CONTINUE_ITERATING
            )
        return status, metric_value


class AbsoluteDivergenceCriterion(AbsoluteCriterion, DivergenceCriterion):
    def check(self, *args, **kwargs) -> ConvergenceStatus:
        """Check divergence.

        Parameters:
            args: Positional arguments for the divergence check.
            kwargs: Quantities to check for divergence.
                - value: The value to check for divergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        status, _ = AbsoluteConvergenceCriterion.check(
            cast(AbsoluteConvergenceCriterion, self), *args, **kwargs
        )
        if status.is_iterating():
            status = ConvergenceStatus.FAILED
            logger.info(self.divergence_msg())
        elif status.is_converged():
            status = ConvergenceStatus.CONVERGED
        return status


class RelativeCriterion:
    """Base class for criteria measuring a quantity against a reference value.

    Parameters:
        tol: Tolerance for convergence - criterion is inactive if set to `np.inf`.
        metric: Metric to compute the convergence measure.
        reference_value: Reference value for relative convergence.
        zero_reference_atol: Absolute tolerance applied where the reference value is
            zero, since a relative measure is undefined there. `None`, the default,
            leaves such entries out of the criterion altogether.

    """

    def __init__(
        self,
        tol: float,
        metric: ConvergenceMetricType,
        reference_value: ConvergenceInfo | None = None,
        zero_reference_atol: float | None = None,
    ) -> None:
        self.tol = tol
        """Tolerance for convergence - criterion in active if set to `np.inf`."""
        self.metric = metric
        """Metric to compute the convergence measure."""
        self.reference_value = reference_value
        """Reference value for relative convergence."""
        self.zero_reference_atol = zero_reference_atol
        """Absolute tolerance used where the reference value is zero."""

    def reset(self) -> None:
        """Reset the reference value."""
        self.reference_value = None

    def set_reference_value(self, reference_value: ConvergenceInfo) -> None:
        """Set the reference value for relative convergence.

        Only entries that are not already set are taken from the provided value, so the
        reference is the one from the iteration at which the criterion was first
        checked. Zero references are stored like any other: dropping them would both
        leave a scalar reference unset, and silently move a block's reference to a later
        iteration than every other block's.

        Parameters:
            reference_value: Reference value to set.

        """
        if isinstance(reference_value, dict):
            self.reference_value = self.reference_value or {}
            assert isinstance(self.reference_value, dict)
            for key, val in reference_value.items():
                if self.reference_value.get(key) is None:
                    self.reference_value[key] = val
        else:  # float
            if self.reference_value is None:
                self.reference_value = reference_value

    def bound(self, reference: float) -> float | None:
        """Return the value a quantity must fall below to satisfy this criterion.

        Parameters:
            reference: The reference value for the entry being checked.

        Returns:
            `tol * reference`, or, where the reference is zero,
            :attr:`zero_reference_atol` - which is `None` when the entry is to be left
            out of the criterion.

        """
        if not np.isclose(reference, 0.0):
            return self.tol * reference
        return self.zero_reference_atol

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(tol={self.tol}, "
        s += f"metric={self.metric.__class__.__name__}, "
        if self.reference_value is not None:
            s += f"reference_value={self.reference_value})"
        else:
            s += "reference_value=None)"
        return s


class RelativeConvergenceCriterion(RelativeCriterion, ConvergenceCriterion):
    """Relative convergence criterion."""

    def check(self, *args, **kwargs) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Check convergence.

        If self.reference_value is a dictionary, the criterion is checked for each entry
        in this dictionary separately, and the convergence is declared only if all
        entries satisfy the criterion.

        A reference value of zero makes the relative measure meaningless, so such an
        entry is instead held to :attr:`zero_reference_atol`, or left out of the
        criterion entirely when that is `None`. Convergence is not declared if every
        entry was left out, since nothing was then tested. The reported convergence
        information is the ratio to the reference value, except for zero-reference
        entries, where it is the quantity itself.

        Parameters:
            args: Positional arguments for the convergence check.
            kwargs: Quantities to check for convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of the
                non-linear iteration and information about the convergence check.

        """
        # Check if tol is np.inf - do not check convergence in this case.
        if self.tol == np.inf:
            return ConvergenceStatus.CONVERGED, 0.0

        reference_value = self.metric(kwargs["reference"])
        if reference_value is not None:
            self.set_reference_value(reference_value)

        metric_value = self.metric(kwargs["value"])
        if isinstance(metric_value, dict):
            assert isinstance(self.reference_value, dict)
            bounds = {
                key: self.bound(self.reference_value[key])
                for key in metric_value
                if key in self.reference_value
            }
            judged = {
                key: val
                for key, val in metric_value.items()
                if bounds.get(key) is not None
            }
            status = (
                ConvergenceStatus.CONVERGED
                if judged
                and all(val < cast(float, bounds[key]) for key, val in judged.items())
                else ConvergenceStatus.CONTINUE_ITERATING
            )
            relative_metric_value: ConvergenceInfo = {
                key: val / self.reference_value[key]
                if not np.isclose(self.reference_value[key], 0.0)
                else val
                for key, val in judged.items()
            }
        else:
            assert isinstance(self.reference_value, float)
            bound = self.bound(self.reference_value)
            status = (
                ConvergenceStatus.CONVERGED
                if bound is not None and metric_value < bound
                else ConvergenceStatus.CONTINUE_ITERATING
            )
            relative_metric_value = (
                metric_value / self.reference_value
                if not np.isclose(self.reference_value, 0.0)
                else metric_value
            )
        return status, relative_metric_value


class RelativeDivergenceCriterion(RelativeCriterion, DivergenceCriterion):
    """Relative divergence criterion."""

    def check(self, *args, **kwargs) -> ConvergenceStatus:
        """Check divergence.

        If self.reference_value is a dictionary, the criterion is checked for each entry
        in this dictionary separately, and divergence is declared if any entry satisfy
        the criterion.

        Parameters:
            args: Positional arguments for the divergence check.
            kwargs: Quantities to check for divergence.
                - value: The value to check for divergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        status, _ = RelativeConvergenceCriterion.check(
            cast(RelativeConvergenceCriterion, self), *args, **kwargs
        )
        if status.is_iterating():
            status = ConvergenceStatus.FAILED
            logger.info(self.divergence_msg())
        return status


class CombinedCriterion:
    """Combined convergence criterion using both absolute and relative criteria.

    The criteria is on the form 'metric_value < atol + rtol * reference_value', where
    `atol` and `rtol` are absolute and relative tolerances, respectively.

    """

    def __init__(
        self,
        atol: float,
        rtol: float,
        metric: ConvergenceMetricType,
        reference_value: ConvergenceInfo | None = None,
    ) -> None:
        self.atol = atol
        """Absolute tolerance for convergence."""
        self.rtol = rtol
        """Relative tolerance for convergence."""
        self.metric = metric
        """Metric to compute the convergence measure."""
        self.reference_value = reference_value
        """Reference value for relative convergence."""

    def reset(self) -> None:
        """Reset the reference value."""
        self.reference_value = None

    def set_reference_value(self, reference_value: ConvergenceInfo) -> None:
        """Set the reference value for relative convergence."""
        if self.reference_value is not None:
            return
        self.reference_value = reference_value


class CombinedConvergenceCriterion(CombinedCriterion, ConvergenceCriterion):
    """Combined convergence criterion using both absolute and relative criteria.

    The criteria is on the form 'metric_value < atol + rtol * reference_value', where
    `atol` and `rtol` are absolute and relative tolerances, respectively.

    """

    def check(self, *args, **kwargs) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Check convergence.

        If self.reference_value is a dictionary, the criterion is checked for each entry
        in this dictionary separately, and the convergence is declared only if all
        entries satisfy the criterion.

        Parameters:
            args: Positional arguments for the convergence check.
            kwargs: Quantities to check for convergence.
                - value: The value to check for convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of the
                non-linear iteration and information about the convergence check.

        """
        reference_value = self.metric(kwargs["reference"])
        if reference_value is not None:
            self.set_reference_value(reference_value)

        metric_value = self.metric(kwargs["value"])
        if isinstance(metric_value, dict):
            assert isinstance(self.reference_value, dict)
            status = (
                ConvergenceStatus.CONVERGED
                if all(
                    v < self.atol + self.rtol * self.reference_value[key]
                    for key, v in metric_value.items()
                    if key in self.reference_value
                )
                else ConvergenceStatus.CONTINUE_ITERATING
            )
        else:
            assert isinstance(self.reference_value, float)
            status = (
                ConvergenceStatus.CONVERGED
                if metric_value < self.atol + self.rtol * self.reference_value
                else ConvergenceStatus.CONTINUE_ITERATING
            )
        return status, metric_value


class CombinedDivergenceCriterion(CombinedCriterion, DivergenceCriterion):
    def check(self, *args, **kwargs) -> ConvergenceStatus:
        """Check divergence.

        If self.reference_value is a dictionary, the criterion is checked for each entry
        in this dictionary separately, and divergence is declared if any entry satisfy
        the criterion.

        Parameters:
            args: Positional arguments for the divergence check.
            kwargs: Quantities to check for divergence.
                - value: The value to check for divergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        status, _ = CombinedConvergenceCriterion.check(
            cast(CombinedConvergenceCriterion, self), *args, **kwargs
        )
        if status.is_iterating():
            status = ConvergenceStatus.FAILED
            logger.info(self.divergence_msg())
        return status


# Specific convergence and divergence criterion implementations.


class IncrementBasedNanCriterion(NanDivergenceCriterion):
    """NaN divergence criterion based on the increment."""

    def check(self, increment: np.ndarray, **kwargs) -> ConvergenceStatus:
        """Check for NaN values in the increment.

        Parameters:
            increment: Nonlinear increment to check for NaN values.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        return super().check(value=increment)


class ResidualBasedNanCriterion(NanDivergenceCriterion):
    """NaN divergence criterion based on the residual."""

    def check(self, residual: np.ndarray, **kwargs) -> ConvergenceStatus:
        """Check for NaN values in the residual.

        Parameters:
            residual: Residual to check for NaN values.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        return super().check(value=residual)


class IncrementBasedAbsoluteDivergenceCriterion(AbsoluteDivergenceCriterion):
    """Absolute divergence criterion based on the increment."""

    def check(self, increment: np.ndarray, **kwargs) -> ConvergenceStatus:
        """Check for divergence based on the increment.

        Parameters:
            increment: Nonlinear increment to check for divergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        return super().check(value=increment)


class ResidualBasedAbsoluteDivergenceCriterion(AbsoluteDivergenceCriterion):
    """Absolute divergence criterion based on the residual."""

    def check(self, residual: np.ndarray, **kwargs) -> ConvergenceStatus:
        """Check for divergence based on the residual.

        Parameters:
            residual: Residual to check for divergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """

        return super().check(value=residual)


class IncrementBasedRelativeDivergenceCriterion(RelativeDivergenceCriterion):
    """Relative divergence criterion based on the increment."""

    def check(
        self, increment: np.ndarray, reference_increment: np.ndarray | None, **kwargs
    ) -> ConvergenceStatus:
        """Check divergence based on the increment.

        Parameters:
            increment: Nonlinear increment to check for divergence.
            reference_increment: Reference increment for relative divergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        return super().check(value=increment, reference=reference_increment)


class ResidualBasedRelativeDivergenceCriterion(RelativeDivergenceCriterion):
    """Relative divergence criterion based on the residual."""

    def check(
        self, residual: np.ndarray, reference_residual: np.ndarray | None, **kwargs
    ) -> ConvergenceStatus:
        """Check divergence based on the residual.

        Parameters:
            residual: Residual to check for divergence.
            reference_residual: Reference residual for relative divergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        return super().check(value=residual, reference=reference_residual)


class IncrementBasedAbsoluteCriterion(AbsoluteConvergenceCriterion):
    """Absolute convergence criterion based on the increment."""

    def check(
        self, increment: np.ndarray, **kwargs
    ) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Check convergence based on the increment.

        Parameters:
            increment: Nonlinear increment to check for convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of the
            non-linear iteration and information about the convergence check.

        """
        return super().check(value=increment)


class IncrementBasedRelativeCriterion(RelativeConvergenceCriterion):
    """Relative convergence criterion based on the increment."""

    def check(
        self, increment: np.ndarray, reference_increment: np.ndarray | None, **kwargs
    ) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Check convergence based on the increment.

        Parameters:
            increment: Nonlinear increment to check for convergence.
            reference_increment: Reference increment for relative convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of the
            non-linear iteration and information about the convergence check.

        """
        return super().check(value=increment, reference=reference_increment)


class ResidualBasedAbsoluteCriterion(AbsoluteConvergenceCriterion):
    """Absolute convergence criterion based on the residual."""

    def check(
        self, residual: np.ndarray, **kwargs
    ) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Check convergence based on the residual.

        Parameters:
            residual: Residual to check for convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of the
                non-linear iteration.

        """
        return super().check(value=residual)


class ResidualBasedRelativeCriterion(RelativeConvergenceCriterion):
    """Relative convergence criterion based on the residual."""

    def check(
        self, residual: np.ndarray, reference_residual: np.ndarray | None, **kwargs
    ) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Check convergence based on the residual.

        Parameters:
            residual: Residual to check for convergence.
            reference_residual: Reference residual for relative convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of the
                non-linear iteration and information about the convergence check.

        """
        return super().check(value=residual, reference=reference_residual)


class IncrementBasedCombinedCriterion(CombinedConvergenceCriterion):
    """Combined convergence criterion based on the increment."""

    def check(
        self, increment: np.ndarray, reference_increment: np.ndarray | None, **kwargs
    ) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Check convergence based on the increment.

        Parameters:
            increment: Nonlinear increment to check for convergence.
            reference_increment: Reference increment for relative convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of the
                non-linear iteration and information about the convergence check.

        """
        return super().check(value=increment, reference=reference_increment)


class ResidualBasedCombinedCriterion(CombinedConvergenceCriterion):
    """Combined convergence criterion based on the residual."""

    def check(
        self, residual: np.ndarray, reference_residual: np.ndarray | None, **kwargs
    ) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Check convergence based on the residual.

        Parameters:
            residual: Residual to check for convergence.
            reference_residual: Reference residual for relative convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of the
                iteration and information about the convergence check.

        """
        return super().check(value=residual, reference=reference_residual)


class IncrementBasedCombinedDivergenceCriterion(CombinedDivergenceCriterion):
    """Combined divergence criterion based on the increment."""

    def check(
        self, increment: np.ndarray, reference_increment: np.ndarray | None, **kwargs
    ) -> ConvergenceStatus:
        """Check divergence based on the increment.

        Parameters:
            increment: Nonlinear increment to check for divergence.
            reference_increment: Reference increment for relative divergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        return super().check(value=increment, reference=reference_increment)


class ResidualBasedCombinedDivergenceCriterion(CombinedDivergenceCriterion):
    """Combined divergence criterion based on the residual."""

    def check(
        self, residual: np.ndarray, reference_residual: np.ndarray | None, **kwargs
    ) -> ConvergenceStatus:
        """Check divergence based on the residual.

        Parameters:
            residual: Residual to check for divergence.
            reference_residual: Reference residual for relative divergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        return super().check(value=residual, reference=reference_residual)


class MaxIterationsCriterion(DivergenceCriterion):
    """Divergence criterion based on maximum number of iterations."""

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        """Maximum allowed iterations (where counter starts at 0)."""

    def check(self, num_iterations: int, **kwargs) -> ConvergenceStatus:
        """Check if the maximum number of iterations has been reached.

        NOTE: Assume num_iterations is 1 for the first iteration, i.e.,
        it is increased at the start of the call of the iterative algorithm.

        Parameters:
            num_iterations: Current number of iterations.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        if num_iterations >= self.max_iterations:
            logger.info(self.divergence_msg())
            return ConvergenceStatus.FAILED
        else:
            return ConvergenceStatus.CONVERGED


def check_convergence(
    convergence_criteria: ConvergenceCriteria,
    divergence_criteria: DivergenceCriteria,
    nonlinear_increment: np.ndarray,
    solution: np.ndarray,
    residual: np.ndarray,
    iteration_index: int,
) -> tuple[
    ConvergenceStatusCollection,
    ConvergenceStatusCollection,
    ConvergenceInfoCollection,
]:
    """A helper function that checks convergence and divergence of the model by passing
    all the parameters that convergence and divergence criteria may expect.

    Parameters:
        convergence_criteria: Convergence criteria.
        divergence_criteria: Divergence criteria.
        nonlinear_increment: Array of how much the solution values changed since the
            last convergence check.
        solution: Array of the solution values.
        residual: Array of residual values.
        iteration_index: The current iteration number of a nonlinear solver.

    Returns:
        Tuple containing:
            - ConvergenceStatusCollection: Status and info about convergence.
            - ConvergenceStatusCollection: Status and info about divergence.
            - ConvergenceInfoCollection: Detailed information about the
                convergence process.

    """
    # Check convergence status based on current iteration.
    convergence_status, convergence_info = convergence_criteria.check(
        increment=nonlinear_increment,
        reference_increment=solution,
        residual=residual,
        reference_residual=residual,
    )

    # Check divergence status based on current iteration.
    divergence_status = divergence_criteria.check(
        increment=nonlinear_increment,
        reference_increment=solution,
        residual=residual,
        reference_residual=residual,
        num_iterations=iteration_index,
    )

    return convergence_status, divergence_status, convergence_info


def assemble_default_divergence_criteria(
    is_nonlinear_problem: bool,
    max_iterations: int,
    inc_div_atol: float,
    res_div_atol: float,
    metric: ConvergenceMetricType,
) -> DivergenceCriteria:
    """A convenience factory for the default divergence criteria. Returns different
    criteria based on whether the problem is nonlinear.

    """
    if is_nonlinear_problem:
        return DivergenceCriteria(
            {
                "max_iter": pp.solvers.MaxIterationsCriterion(
                    max_iterations=max_iterations
                ),
                "inc_nan": pp.solvers.IncrementBasedNanCriterion(),
                "res_nan": pp.solvers.ResidualBasedNanCriterion(),
                "inc_max": pp.solvers.IncrementBasedAbsoluteDivergenceCriterion(
                    tol=inc_div_atol, metric=metric
                ),
                "res_max": pp.solvers.ResidualBasedAbsoluteDivergenceCriterion(
                    tol=res_div_atol, metric=metric
                ),
            }
        )
    else:
        return DivergenceCriteria(
            {
                "inc_nan": pp.solvers.IncrementBasedNanCriterion(),
                "res_nan": pp.solvers.ResidualBasedNanCriterion(),
            }
        )


def assemble_default_convergence_criteria(
    is_nonlinear_problem: bool,
    inc_atol: float,
    inc_rtol: float,
    res_atol: float,
    res_rtol: float,
    metric: ConvergenceMetricType,
) -> ConvergenceCriteria:
    """A convenience factory for the default convergence criteria. Returns different
    criteria based on whether the problem is nonlinear.

    """
    if is_nonlinear_problem:
        return ConvergenceCriteria(
            {
                "inc_abs": pp.solvers.IncrementBasedAbsoluteCriterion(
                    tol=inc_atol, metric=metric
                ),
                "inc_rel": pp.solvers.IncrementBasedRelativeCriterion(
                    tol=inc_rtol, metric=metric, zero_reference_atol=inc_atol
                ),
                "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                    tol=res_atol, metric=metric
                ),
                "res_rel": pp.solvers.ResidualBasedRelativeCriterion(
                    tol=res_rtol, metric=metric, zero_reference_atol=res_atol
                ),
            }
        )
    else:
        return ConvergenceCriteria()
