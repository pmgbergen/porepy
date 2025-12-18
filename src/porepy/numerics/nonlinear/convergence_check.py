"""Collection of objects and functions related to convergence checking.

This includes:
- Convergence status enumeration.
- Reference value management for defining reference norms.
- Base convergence criterion classes.
- Absolute and relative convergence criteria for nonlinear problems.
- A NaN convergence criterion for detecting divergence due to NaN values.

"""

from abc import ABC, abstractmethod
from copy import copy
from enum import StrEnum
from typing import Callable

import numpy as np


class SimulationStatus(StrEnum):
    """Enumeration of potential simulation statuses."""

    SUCCESSFUL = "successful"
    FAILED = "failed"
    STOPPED = "stopped"

    def __str__(self):
        return self.value

    def is_successful(self) -> bool:
        """Check if the status indicates a successful simulation."""
        return self == SimulationStatus.SUCCESSFUL

    def is_failed(self) -> bool:
        """Check if the status indicates a failed simulation."""
        return self == SimulationStatus.FAILED

    def is_stopped(self) -> bool:
        """Check if the status indicates a stopped simulation."""
        return self == SimulationStatus.STOPPED


class ConvergenceStatus(StrEnum):
    """Enumeration of potential convergence statuses."""

    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    DIVERGED = "diverged"
    CYCLED = "cycled"
    STAGNATED = "stagnated"
    NAN = "nan"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    STOPPED = "stopped"

    def __str__(self):
        return self.value

    def is_converged(self) -> bool:
        """Check if the status indicates convergence."""
        return self == ConvergenceStatus.CONVERGED

    def is_not_converged(self) -> bool:
        """Check if the status indicates not converged."""
        return self == ConvergenceStatus.NOT_CONVERGED

    def is_diverged(self) -> bool:
        """Check if the status indicates divergence."""
        return self == ConvergenceStatus.DIVERGED

    def is_cycled(self) -> bool:
        """Check if the status indicates cycling."""
        return self == ConvergenceStatus.CYCLED

    def is_stagnated(self) -> bool:
        """Check if the status indicates stagnation."""
        return self == ConvergenceStatus.STAGNATED

    def is_nan(self) -> bool:
        """Check if the status indicates NaN."""
        return self == ConvergenceStatus.NAN

    def is_max_iterations_reached(self) -> bool:
        """Check if the status indicates that the maximum number of iterations
        was reached.

        """
        return self == ConvergenceStatus.MAX_ITERATIONS_REACHED

    def is_stopped(self) -> bool:
        """Check if the status indicates that the process was stopped."""
        return self == ConvergenceStatus.STOPPED

    def is_failed(self) -> bool:
        """Check if the status indicates a failure."""
        return self in {
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.CYCLED,
            ConvergenceStatus.STAGNATED,
            ConvergenceStatus.NAN,
            ConvergenceStatus.MAX_ITERATIONS_REACHED,
        }


class ConvergenceStatusDict(dict[str, ConvergenceStatus]):
    """Dict of convergence statuses."""

    def is_converged(self) -> bool:
        """Check if all statuses indicate convergence."""
        return all(status.is_converged() for status in self.values())

    def is_not_converged(self) -> bool:
        """Check if any status indicates not converged."""
        return any(status.is_not_converged() for status in self.values())

    def is_diverged(self) -> bool:
        """Check if any status indicates divergence."""
        return any(status.is_diverged() for status in self.values())

    def is_cycled(self) -> bool:
        """Check if any status indicates cycling."""
        return any(status.is_cycled() for status in self.values())

    def is_stagnated(self) -> bool:
        """Check if any status indicates stagnation."""
        return any(status.is_stagnated() for status in self.values())

    def is_nan(self) -> bool:
        """Check if any status indicates NaN."""
        return any(status.is_nan() for status in self.values())

    def is_stopped(self) -> bool:
        """Check if any status indicates stopping."""
        return any(status.is_stopped() for status in self.values())

    def is_failed(self) -> bool:
        """Check if any status indicates failure."""
        return any(status.is_failed() for status in self.values())

    def union(self, other: "ConvergenceStatusDict") -> "ConvergenceStatusDict":
        """Union of two ConvergenceStatusDicts needing to be disjunct."""
        result = ConvergenceStatusDict()
        assert len(set(self.keys()).intersection(other.keys())) == 0
        result.update(self)
        result.update(other)
        return result


class ConvergenceStatusCollection(dict[str, list[ConvergenceStatus]]):
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

    def append(self, status: ConvergenceStatusDict) -> None:
        """Append another ConvergenceStatusCollection to this one.

        Parameters:
            status: Convergence statuses to append.

        """
        # Since this class inherits from dict, we should always be a dict
        # The recursive append modifies self in-place, so no reassignment needed
        self._recursive_dict_append(self, status)

    def _recursive_dict_append(
        self, d: "ConvergenceStatusCollection", v: dict
    ) -> "ConvergenceStatusCollection":
        """Auxiliary function to recursively append dictionaries.

        Parameters:
            d: ConvergenceStatusCollection to append to.
            v: Dictionary to append.

        Returns:
            ConvergenceStatusCollection: Updated ConvergenceStatusCollection.

        """
        for key_v, value_v in v.items():
            if key_v not in d:
                d[key_v] = [copy(value_v)]
            else:
                value = d[key_v]
                if isinstance(value, list):
                    value.append(value_v)
                else:
                    assert isinstance(value_v, dict)
                    assert isinstance(d[key_v], ConvergenceStatusCollection)
                    d[key_v] = self._recursive_dict_append(d[key_v], value_v)

        return d


### Base convergence criterion classes.


class ConvergenceCriterion(ABC):
    """Base class for convergence criteria."""

    @abstractmethod
    def check(self, *args, **kwargs) -> tuple[ConvergenceStatus, float | dict]:
        """Check convergence.

        Parameters:
            kwargs: Quantities to check for convergence.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.
            float: Information about the convergence check.

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


class ConvergenceCriteria(dict[str, ConvergenceCriterion]):
    """Collection of convergence criteria."""

    def check(
        self, *args, **kwargs
    ) -> tuple[ConvergenceStatusDict, dict[str, float | dict]]:
        """Check convergence using all criteria in the collection.

        Parameters:
            args: Positional arguments for the convergence checks.
            kwargs: Keyword arguments for the convergence checks.

        Returns:
            tuple[ConvergenceStatusDict, dict]: Convergence statuses with the names of
                the criteria as keys, and information about the convergence checks
                (format of the values depends on used metrics).

        """
        status = ConvergenceStatusDict()
        info = {}
        for name, criterion in self.items():
            stat, inf = criterion.check(*args, **kwargs)
            status[name] = stat
            info[name] = inf
        return status, info

    def reset(self) -> None:
        """Reset all convergence criteria in the collection."""
        for criterion in self.values():
            criterion.reset()


class DivergenceCriteria(dict[str, DivergenceCriterion]):
    """Collection of divergence criteria."""

    def check(self, *args, **kwargs) -> ConvergenceStatusDict:
        """Check convergence using all criteria in the collection.

        Parameters:
            args: Positional arguments for the divergence checks.
            kwargs: Keyword arguments for the divergence checks.

        Returns:
            ConvergenceStatusDict: Convergence statuses of the non-linear iteration
                with the names of the criteria as keys.

        """
        status = ConvergenceStatusDict()
        for name, criterion in self.items():
            status[name] = criterion.check(*args, **kwargs)
        return status

    def reset(self) -> None:
        """Reset all divergence criteria in the collection."""
        for criterion in self.values():
            criterion.reset()


class MaxIterationsCriterion(DivergenceCriterion):
    """Convergence criterion based on maximum number of iterations."""

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        """Maximum allowed iterations."""

    def check(self, num_iterations: int, **kwargs) -> ConvergenceStatus:
        """Check if the maximum number of iterations has been reached.

        Parameters:
            num_iterations: Current number of iterations.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.

        """
        # Assume iteration counting starts at 0
        if num_iterations >= self.max_iterations - 1:
            return ConvergenceStatus.DIVERGED
        else:
            return ConvergenceStatus.CONVERGED


class NanConvergenceCriterion(DivergenceCriterion):
    """Convergence criterion, that checks for NaN values."""

    def check(self, *args, **kwargs) -> ConvergenceStatus:
        """Check for NaN values in the nonlinear increment and residual.

        Parameters:
            value: Quantity to check for NaN values.

        Returns:
            ConvergenceStatus: Convergence status of the non-linear iteration.
            float: Information about the convergence check.

        """
        if np.isnan(kwargs["value"]).any():
            return ConvergenceStatus.DIVERGED
        return ConvergenceStatus.CONVERGED


class AbsoluteConvergenceCriterion(ConvergenceCriterion):
    def __init__(
        self,
        tol: float,
        metric: Callable[[np.ndarray], dict | float],
    ) -> None:
        self.tol = tol
        """Tolerance for convergence."""
        self.metric = metric
        """Metric to compute the convergence measure."""

    def check(self, *args, **kwargs) -> tuple[ConvergenceStatus, dict | float]:
        metric_value = self.metric(kwargs["value"])
        if isinstance(metric_value, dict):
            status = (
                ConvergenceStatus.CONVERGED
                if all(v < self.tol for v in metric_value.values())
                else ConvergenceStatus.NOT_CONVERGED
            )
        else:
            status = (
                ConvergenceStatus.CONVERGED
                if metric_value < self.tol
                else ConvergenceStatus.NOT_CONVERGED
            )
        return status, metric_value


class AbsoluteDivergenceCriterion(DivergenceCriterion):
    def __init__(
        self,
        tol: float,
        metric: Callable[[np.ndarray], dict | float],
    ) -> None:
        self.tol = tol
        """Tolerance for divergence."""
        self.metric = metric
        """Metric to compute the divergence measure."""

    def check(self, *args, **kwargs) -> ConvergenceStatus:
        metric_value = self.metric(kwargs["value"])
        if isinstance(metric_value, dict):
            status = (
                ConvergenceStatus.DIVERGED
                if any(v > self.tol for v in metric_value.values())
                else ConvergenceStatus.CONVERGED
            )
        else:
            status = (
                ConvergenceStatus.DIVERGED
                if metric_value > self.tol
                else ConvergenceStatus.CONVERGED
            )
        return status


class RelativeConvergenceCriterion(ConvergenceCriterion):
    """Relative convergence criterion."""

    def __init__(
        self,
        tol: float,
        metric: Callable[[np.ndarray], dict | float],
        reference_value: dict | float | None = None,
    ) -> None:
        self.tol = tol
        """Tolerance for convergence - criterion in active if set to `np.inf`."""
        self.metric = metric
        """Metric to compute the convergence measure."""
        self.reference_value = reference_value
        """Reference value for relative convergence."""

    def reset(self) -> None:
        """Reset the reference value."""
        self.reference_value = None

    def set_reference_value(self, reference_value: dict | float) -> None:
        """Set the reference value for relative convergence.

        Parameters:
            reference_value: Reference value to set.

        """
        if isinstance(reference_value, dict):
            self.reference_value = self.reference_value or {}
            assert isinstance(self.reference_value, dict)
            non_zero_reference_value = {}
            for key, val in reference_value.items():
                if self.reference_value.get(key) is None and not np.isclose(val, 0.0):
                    non_zero_reference_value[key] = val
            self.reference_value.update(non_zero_reference_value)
        else:
            if self.reference_value is not None:
                return
            self.reference_value = reference_value

    def check(self, *args, **kwargs) -> tuple[ConvergenceStatus, dict | float]:
        """Check convergence.

        Parameters:
            args: Positional arguments for the convergence check.
            kwargs: Quantities to check for convergence.

        Returns:
            tuple[ConvergenceStatus, dict | float]: Convergence status of the non-linear
                iteration and information about the convergence check.

        """
        # Check if tol is np.inf - do not check convergence in this case
        if self.tol == np.inf:
            return ConvergenceStatus.CONVERGED, 0.0

        metric_value = self.metric(kwargs["value"])
        if isinstance(metric_value, dict):
            assert isinstance(self.reference_value, dict)
            status = (
                ConvergenceStatus.CONVERGED
                if all(
                    val < self.tol * (self.reference_value[key])
                    for key, val in metric_value.items()
                    if key in self.reference_value
                )
                else ConvergenceStatus.NOT_CONVERGED
            )
            relative_metric_value: dict | float = {
                key: val / self.reference_value[key]
                for key, val in metric_value.items()
                if key in self.reference_value
            }
        else:
            assert isinstance(self.reference_value, float)
            status = (
                ConvergenceStatus.CONVERGED
                if metric_value < self.tol * self.reference_value
                else ConvergenceStatus.NOT_CONVERGED
            )
            relative_metric_value = metric_value / self.reference_value
        return status, relative_metric_value


class RelativeDivergenceCriterion(DivergenceCriterion):
    def __init__(
        self,
        tol: float,
        metric: Callable[[np.ndarray], dict | float],
        reference_value: dict | float | None = None,
    ) -> None:
        self.tol = tol
        self.metric = metric
        self.reference_value = reference_value

    def reset(self) -> None:
        self.reference_value = None

    def set_reference_value(self, reference_value: dict | float) -> None:
        if isinstance(reference_value, dict):
            self.reference_value = self.reference_value or {}
            assert isinstance(self.reference_value, dict)
            non_zero_reference_value = {}
            for key, val in reference_value.items():
                if self.reference_value.get(key) is None and not np.isclose(val, 0.0):
                    non_zero_reference_value[key] = val
            self.reference_value.update(non_zero_reference_value)
        else:
            if self.reference_value is not None:
                return
            self.reference_value = reference_value

    def check(self, *args, **kwargs) -> ConvergenceStatus:
        metric_value = self.metric(kwargs["value"])
        if isinstance(metric_value, dict):
            assert isinstance(self.reference_value, dict)
            status = (
                ConvergenceStatus.DIVERGED
                if any(
                    v > self.tol * r
                    for v, r in zip(
                        metric_value.values(), self.reference_value.values()
                    )
                )
                else ConvergenceStatus.CONVERGED
            )
        else:
            assert isinstance(self.reference_value, float)
            status = (
                ConvergenceStatus.DIVERGED
                if metric_value > self.tol * self.reference_value
                else ConvergenceStatus.CONVERGED
            )
        return status


class CombinedConvergenceCriterion(ConvergenceCriterion):
    def __init__(
        self,
        atol: float,
        rtol: float,
        metric: Callable[[np.ndarray], dict | float],
        reference_value: dict | float | None = None,
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.metric = metric
        self.reference_value = reference_value

    def reset(self) -> None:
        self.reference_value = None

    def set_reference_value(self, reference_value: dict | float) -> None:
        if self.reference_value is not None:
            return
        self.reference_value = reference_value

    def check(self, *args, **kwargs) -> tuple[ConvergenceStatus, dict | float]:
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
                else ConvergenceStatus.NOT_CONVERGED
            )
        else:
            assert isinstance(self.reference_value, float)
            status = (
                ConvergenceStatus.CONVERGED
                if metric_value < self.atol + self.rtol * self.reference_value
                else ConvergenceStatus.NOT_CONVERGED
            )
        return status, metric_value


class IncrementBasedNanCriterion(NanConvergenceCriterion):
    def check(self, increment: np.ndarray, **kwargs) -> ConvergenceStatus:
        return super().check(value=increment)


class ResidualBasedNanCriterion(NanConvergenceCriterion):
    def check(self, residual: np.ndarray, **kwargs) -> ConvergenceStatus:
        return super().check(value=residual)


class IncrementBasedAbsoluteDivergenceCriterion(AbsoluteDivergenceCriterion):
    def check(self, increment: np.ndarray, **kwargs) -> ConvergenceStatus:
        return super().check(value=increment)


class ResidualBasedAbsoluteDivergenceCriterion(AbsoluteDivergenceCriterion):
    def check(self, residual: np.ndarray, **kwargs) -> ConvergenceStatus:
        return super().check(value=residual)


class IncrementBasedRelativeDivergenceCriterion(RelativeDivergenceCriterion):
    def check(
        self, increment: np.ndarray, reference_increment: np.ndarray | None, **kwargs
    ) -> ConvergenceStatus:
        if reference_increment is not None:
            self.set_reference_value(self.metric(reference_increment))
        return super().check(value=increment)


class ResidualBasedRelativeDivergenceCriterion(RelativeDivergenceCriterion):
    def check(
        self, residual: np.ndarray, reference_residual: np.ndarray | None, **kwargs
    ) -> ConvergenceStatus:
        if reference_residual is not None:
            self.set_reference_value(self.metric(reference_residual))
        return super().check(value=residual)


class IncrementBasedAbsoluteCriterion(AbsoluteConvergenceCriterion):
    def check(
        self, increment: np.ndarray, **kwargs
    ) -> tuple[ConvergenceStatus, dict | float]:
        return super().check(value=increment)


class IncrementBasedRelativeCriterion(RelativeConvergenceCriterion):
    def check(
        self, increment: np.ndarray, reference_increment: np.ndarray | None, **kwargs
    ) -> tuple[ConvergenceStatus, dict | float]:
        if reference_increment is not None:
            self.set_reference_value(self.metric(reference_increment))
        return super().check(value=increment)


class ResidualBasedAbsoluteCriterion(AbsoluteConvergenceCriterion):
    def check(
        self, residual: np.ndarray, **kwargs
    ) -> tuple[ConvergenceStatus, dict | float]:
        return super().check(value=residual)


class ResidualBasedRelativeCriterion(RelativeConvergenceCriterion):
    def check(
        self, residual: np.ndarray, reference_residual: np.ndarray | None, **kwargs
    ) -> tuple[ConvergenceStatus, dict | float]:
        if reference_residual is not None:
            self.set_reference_value(self.metric(reference_residual))
        return super().check(value=residual)


class IncrementBasedCombinedCriterion(CombinedConvergenceCriterion):
    def check(
        self, increment: np.ndarray, reference_increment: np.ndarray | None, **kwargs
    ) -> tuple[ConvergenceStatus, dict | float]:
        if reference_increment is not None:
            self.set_reference_value(self.metric(reference_increment))
        return super().check(value=increment)


class ResidualBasedCombinedCriterion(CombinedConvergenceCriterion):
    def check(
        self, residual: np.ndarray, reference_residual: np.ndarray | None, **kwargs
    ) -> tuple[ConvergenceStatus, dict | float]:
        if reference_residual is not None:
            self.set_reference_value(self.metric(reference_residual))
        return super().check(value=residual)
