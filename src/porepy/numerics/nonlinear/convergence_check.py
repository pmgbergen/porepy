"""Collection of objects and functions related to convergence checking.

This includes:
- Status classes for simulation and convergence.
- Information classes for convergence.
- Base convergence criterion classes.
- Absolute and relative convergence criteria for nonlinear problems.
- Divergence criteria for detecting divergence.

"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceStatus:
    pass


@dataclass
class ConvergenceStatusSuccess(ConvergenceStatus):
    pass


@dataclass
class ConvergenceStatusSuccessWithMessage(ConvergenceStatus):
    msg: str


@dataclass
class ConvergenceStatusSuccessMultipleCriteria(ConvergenceStatusSuccess):
    statuses: list[ConvergenceStatusSuccess]


@dataclass
class ConvergenceStatusSuccessWithMetricValue(ConvergenceStatus):
    metric_name: str
    metric_value: float


@dataclass
class ConvergenceStatusContinueIterating(ConvergenceStatus):
    pass


@dataclass
class ConvergenceStatusFailure(ConvergenceStatus):
    msg: str
    """A message describing the failure reason."""


class ConvergenceCriterion(ABC):
    """Base class for convergence criteria."""

    @abstractmethod
    def check(
        self, convergence_data: dict
    ) -> ConvergenceStatus:  # TODO Specify signature
        """Check convergence.

        Parameters:
            kwargs: Quantities to check for convergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.
            ConvergenceInfo: Information about the convergence check.

        """

    def reset(self) -> None:
        pass

    def failure_msg(self) -> str:
        """Return a message describing the divergence reason."""
        return f"\033[91m{self.__class__.__name__} triggered failure.\033[0m"


class NanDivergenceCriterion(ConvergenceCriterion):
    def __init__(self, quantity_key: str):
        self.quantity_key: str = quantity_key

    """Divergence criterion, that checks for NaN values."""

    def check(self, convergence_data: dict) -> ConvergenceStatus:
        """Check for NaN values in the nonlinear increment and residual.

        Parameters:
            args: Positional arguments for the convergence check.
            kwargs: Quantities to check for NaN values.
                - value: The value to check for NaN values.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        value = convergence_data[self.quantity_key]
        if np.isnan(value).any():
            return ConvergenceStatusFailure(msg=self.failure_msg())
        return ConvergenceStatusContinueIterating()


class AbsoluteCriterion:
    def __init__(
        self,
        quantity_key: str,
        tol: float,
        metric: Callable[[np.ndarray], float],
    ) -> None:
        self.quantity_key = quantity_key
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

    def check(self, convergence_data: dict) -> ConvergenceStatus:
        """Check convergence.

        Parameters:
            args: Positional arguments for the convergence check.
            kwargs: Quantities to check for convergence.
                - value: The value to check for convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of
                the non-linear iteration and information about the convergence check.

        """
        value = convergence_data[self.quantity_key]
        metric_value = self.metric(value)
        assert np.isscalar(metric_value)
        metric_value = float(metric_value)
        if metric_value < self.tol:
            return ConvergenceStatusSuccessWithMetricValue(
                metric_name=self.metric.__class__.__name__,  # TODO Metric name
                metric_value=metric_value,
            )

        return ConvergenceStatusContinueIterating()


class AbsoluteDivergenceCriterion(AbsoluteCriterion, ConvergenceCriterion):
    def check(self, convergence_data: dict) -> ConvergenceStatus:
        """Check divergence. TODO

        Parameters:
            args: Positional arguments for the divergence check.
            kwargs: Quantities to check for divergence.
                - value: The value to check for divergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        value = convergence_data[self.quantity_key]
        metric_value = self.metric(value)
        assert np.isscalar(metric_value)
        metric_value = float(metric_value)
        if metric_value > self.tol:
            return ConvergenceStatusFailure(msg=self.failure_msg())

        return ConvergenceStatusContinueIterating()


class RelativeCriterion:
    def __init__(
        self,
        tol: float,
        quantity_key: str,
        metric: Callable[[np.ndarray], float],
        reference_value: float | None = None,
    ) -> None:
        self.quantity_key: str = quantity_key
        self.tol = tol
        """Tolerance for convergence - criterion in active if set to `np.inf`."""
        self.metric = metric
        """Metric to compute the convergence measure."""
        self.reference_value = reference_value
        """Reference value for relative convergence."""

    def reset(self) -> None:
        self.reference_value = None

    def set_reference_value(self, reference_value: np.ndarray) -> None:
        """Set the reference value for relative convergence. TODO

        The reference value is only set for entries of self.reference_value that are not
        already set and are non-zero in the provided reference value.

        Parameters:
            reference_value: Reference value to set.

        """
        assert not np.isclose(reference_value, 0.0)
        self.reference_value = float(self.metric(reference_value))

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(tol={self.tol}, "
        s += f"metric={self.metric.__class__.__name__}, "
        s += f"reference_value={self.reference_value})"
        return s


class RelativeConvergenceCriterion(RelativeCriterion, ConvergenceCriterion):
    """Relative convergence criterion."""

    def check(self, convergence_data: dict) -> ConvergenceStatus:
        """Check convergence. TODO

        If self.reference_value is a dictionary, the criterion is checked for each entry
        in this dictionary separately, and the convergence is declared only if all
        entries satisfy the criterion.

        Parameters:
            args: Positional arguments for the convergence check.
            kwargs: Quantities to check for convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of the
                non-linear iteration and information about the convergence check.

        """
        # Check if tol is np.inf - do not check convergence in this case.
        if self.tol == np.inf:
            return ConvergenceStatusSuccessWithMessage(
                msg="RelativeConvergenceCriterion tol is inf."
            )

        if self.reference_value is None:
            reference = convergence_data[f"{self.quantity_key}_reference"]
            self.set_reference_value(reference)

        value = convergence_data[self.quantity_key]

        metric_value = self.metric(value)
        assert isinstance(self.reference_value, float)
        relative_metric_value = metric_value / self.reference_value
        if relative_metric_value < self.tol:
            return ConvergenceStatusSuccessWithMetricValue(
                metric_name=self.metric.__class__.__name__,
                metric_value=relative_metric_value,
            )
        return ConvergenceStatusContinueIterating()


class RelativeDivergenceCriterion(RelativeCriterion, ConvergenceCriterion):
    """Relative divergence criterion."""

    def check(self, convergence_data: dict) -> ConvergenceStatus:
        # Check if tol is np.inf - do not check convergence in this case.
        if self.tol == np.inf:
            return ConvergenceStatusContinueIterating()

        if self.reference_value is None:
            reference = convergence_data[f"{self.quantity_key}_reference"]
            self.set_reference_value(reference)

        value = convergence_data[self.quantity_key]

        metric_value = self.metric(value)
        assert isinstance(self.reference_value, float)
        relative_metric_value = metric_value / self.reference_value
        if relative_metric_value > self.tol:
            return ConvergenceStatusFailure(msg=self.failure_msg())
        return ConvergenceStatusContinueIterating()


class ConvergedAll(ConvergenceCriterion):
    def __init__(self, children: list[ConvergenceCriterion]) -> None:
        self.children: list[ConvergenceCriterion] = children

    def check(self, convergence_data: dict) -> ConvergenceStatus:
        converged_statuses: list[ConvergenceStatusSuccess] = []
        for child in self.children:
            status = child.check(convergence_data)
            if isinstance(status, ConvergenceStatusFailure):
                return status
            if isinstance(status, ConvergenceStatusSuccess):
                converged_statuses.append(status)
        if len(converged_statuses) == len(self.children):
            return ConvergenceStatusSuccessMultipleCriteria(statuses=converged_statuses)
        else:
            


class ConvergedAtLeastOne(ConvergenceCriterion):
    def __init__(self, children: list[ConvergenceCriterion]) -> None:
        self.children: list[ConvergenceCriterion] = children

    def check(self, convergence_data: dict) -> ConvergenceStatus:
        for child in self.children:
            status = child.check(convergence_data)
            if isinstance(status, (ConvergenceStatusFailure, ConvergenceStatusSuccess)):
                return status
        return
