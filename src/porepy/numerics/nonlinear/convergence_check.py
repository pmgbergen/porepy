"""Collection of objects and functions related to convergence checking.

This includes:
- Convergence status enumeration.
- Reference value management for defining reference norms.
- Base convergence criterion classes.
- Absolute and relative convergence criteria for nonlinear problems.
- A NaN convergence criterion for detecting divergence due to NaN values.

"""

from abc import abstractmethod
from enum import StrEnum
from typing import Callable, Tuple

import numpy as np


class ConvergenceStatus(StrEnum):
    """Enumeration of potential convergence statuses."""

    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    DIVERGED = "diverged"
    CYCLED = "cycled"
    STAGNATED = "stagnated"
    NAN = "nan"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"

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

    def is_failed(self) -> bool:
        """Check if the status indicates a failure."""
        return self in {
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.CYCLED,
            ConvergenceStatus.STAGNATED,
            ConvergenceStatus.NAN,
            ConvergenceStatus.MAX_ITERATIONS_REACHED,
        }


class ReferenceValue:
    """Reference value manager.

    It allows initializing a reference value only when a certain condition is met, and
    provides a default value if the condition is not met. For updating reference
    values, the object needs to be reset.

    """

    def __init__(
        self,
        condition: Callable[[float], bool],
        default_reference_value: float,
    ) -> None:
        """Define the reference value manager.

        Parameters:
            condition: A callable that takes a value and returns True if it is a
                valid reference value, False otherwise.
            default_reference_value: The default value to return if the reference
                value is not set or is None.

        """
        self.condition: Callable[[float], bool] = condition
        """Condition for updating the reference value."""
        self.default_reference_value: float = default_reference_value
        """Default value to return if the reference value is None."""
        self.reference_value: dict[str, float | None] = {}
        """Dictionary to store reference values for different keys."""

    def __call__(self, values: dict[str, float]) -> dict[str, float]:
        """Update the reference value for a key if it meets the condition.

        Parameters:
            values: A dictionary of values to use for setting the reference value.

        Returns:
            dict[str, float]: A dictionary with the valid reference values, and
                ensuring that if a value is None, it is replaced with the default
                reference value.

        """
        for key, value in values.items():
            if key not in self.reference_value:
                self.reference_value[key] = None
            if self.condition(value) and self.reference_value[key] is None:
                self.reference_value[key] = value

        def default_value_for_none(value: float | None) -> float:
            """Return a default value if the input is None."""
            return value if value is not None else self.default_reference_value

        return {
            key: default_value_for_none(self.reference_value[key])
            for key in values.keys()
        }

    def reset(self) -> None:
        """Reset the reference values to None."""
        self.reference_value = {key: None for key in self.reference_value.keys()}


### Base convergence criterion classes


class ConvergenceCriterion:
    """Base class for convergence criteria.

    Requires the implementation of the `_check` method to define the convergence.

    """

    def check(
        self,
        nonlinear_increment: float | dict[str, float],
        residual: float | dict[str, float],
        params: dict,
    ) -> Tuple[ConvergenceStatus, float, float]:
        """Check convergence.

        Parameters:
            nonlinear_increment: The increment in the solution variables from the
                previous nonlinear iteration.
            residual: The current residual vector of the nonlinear system.
            params: Dictionary of parameters for the convergence check.

        Returns:
            ConvergenceStatus: The convergence status of the non-linear iteration.

        """
        # Convert values to dict format
        nonlinear_increment = self._make_dict(nonlinear_increment)
        residual = self._make_dict(residual)

        # Call the actual comparison
        return self._check(nonlinear_increment, residual, params)

    def _check_dicts(self, dict1: dict[str, float], dict2: dict[str, float]) -> None:
        """Check if two dictionaries have the same keys.

        Parameters:
            dict1: First dictionary to compare.
            dict2: Second dictionary to compare.

        Raises:
            ValueError: If the dictionaries do not have the same keys.

        """
        if set(dict1.keys()) != set(dict2.keys()):
            raise ValueError(
                "Dictionaries do not have the same keys: "
                f"{dict1.keys()} vs {dict2.keys()}"
            )

    def _make_dict(self, value: float | dict[str, float]) -> dict[str, float]:
        """Convert a float or a dict to a unified dict format.

        Parameters:
            value: A float or a dict with string keys and float values.

        Returns:
            Float converted to a dict, or the original dict.

        """
        if isinstance(value, dict):
            return value
        else:
            return {"": value}

    @abstractmethod
    def _check(
        self,
        nonlinear_increment: dict[str, float],
        residual: dict[str, float],
        params: dict,
    ) -> Tuple[ConvergenceStatus, float, float]:
        """Check convergence.

        Parameters:
            nonlinear_increment: The increment in the solution variables from the
                previous nonlinear iteration.
            residual: The current residual vector of the nonlinear system.
            params: Dictionary of parameters for the convergence check.

        Returns:
            ConvergenceStatus: The convergence status of the non-linear iteration.

        """
        pass


class RelativeConvergenceCriterion(ConvergenceCriterion):
    """Relative convergence criterion based on reference values.

    The convergence criterion is met if all relative norms of the nonlinear increment
    and the residual are below the specified thresholds. A combined absolute-relative
    norm is used to ensure that the relative norms are meaningful:

        ||inc|| / (1 + ||ref_inc||) < tol_inc
        ||res|| / (1 + ||ref_res||) < tol_res

    The reference values for the nonlinear increment and residual norms are managed,
    allowing for a flexible convergence check that adapts to the problem. This class
    is abstract and requires the implementation of the `init_reference_value` method,
    defining `reference_value`.

    """

    reference_value: ReferenceValue
    """Reference value for relative convergence checks."""

    def __init__(self) -> None:
        """Initialize the relative convergence criterion."""
        self.init_reference_value()
        self.reference_nonlinear_increment_norm: dict[str, float] = {}
        """Reference value for the nonlinear increment norm."""
        self.reference_residual_norm: dict[str, float] = {}
        """Reference value for the residual norm."""

    ### Manager methods for setting reference values

    @abstractmethod
    def init_reference_value(self) -> None:
        """Expect to instantiate `self.reference_value`."""

    def set_reference_value(
        self,
        reference_nonlinear_increment_norm: float | dict[str, float],
        reference_residual_norm: float | dict[str, float],
    ) -> None:
        """Set the reference values for increments and residuals.

        The actual update is done by the `reference_value` manager, which
        ensures that the reference values are only updated if they meet the
        specified conditions.

        Parameters:
            reference_nonlinear_increment_norm: Reference value for the nonlinear
                increment norm.
            reference_residual_norm: Reference value for the residual norm.

        """
        self.reference_nonlinear_increment_norm = self.reference_value(
            self._make_dict(reference_nonlinear_increment_norm)
        )
        self.reference_residual_norm = self.reference_value(
            self._make_dict(reference_residual_norm)
        )

    def reset_reference_values(self) -> None:
        """Reset the reference values to their initial state."""
        self.reference_value.reset()

    ### Convergence check methods

    def _check(
        self,
        nonlinear_increment_norm: dict[str, float],
        residual_norm: dict[str, float],
        params: dict,
    ) -> Tuple[ConvergenceStatus, float, float]:
        """Check convergence using relative norms.

        Parameters:
            nonlinear_increment_norm: Norm of the nonlinear increment.
            residual_norm: Norm of the residual.
            params: Dictionary of parameters for the convergence check.

        Returns:
            ConvergenceStatus: The convergence status of the non-linear iteration.
            float: Global norm of the nonlinear increment.
            float: Global norm of the residual.

        """
        # Consistency checks
        self._check_dicts(
            nonlinear_increment_norm, self.reference_nonlinear_increment_norm
        )
        self._check_dicts(residual_norm, self.reference_residual_norm)

        # Check divergence.
        is_diverged = any(
            res_norm > params["nl_divergence_tol_res"]
            for res_norm in residual_norm.values()
        )

        # Reduce norms to floats using l-infinity norm over combined
        # absolute-relative values.
        reduced_nonlinear_increment_norm = max(
            inc_norm / (1 + self.reference_nonlinear_increment_norm[key])
            for key, inc_norm in nonlinear_increment_norm.items()
        )
        reduced_residual_norm = max(
            res_norm / (1 + self.reference_residual_norm[key])
            for key, res_norm in residual_norm.items()
        )

        # Check convergence using relative norms.
        converged_inc = (
            reduced_nonlinear_increment_norm < params["nl_convergence_tol_inc"]
        )
        converged_res = reduced_residual_norm < params["nl_convergence_tol_res"]
        is_converged = converged_inc and converged_res

        # Determine convergence status.
        convergence_status = ConvergenceStatus.NOT_CONVERGED
        if is_diverged:
            convergence_status = ConvergenceStatus.DIVERGED
        elif is_converged:
            convergence_status = ConvergenceStatus.CONVERGED

        return (
            convergence_status,
            reduced_nonlinear_increment_norm,
            reduced_residual_norm,
        )


### Concrete convergence criteria


class NanConvergenceCriterion(ConvergenceCriterion):
    """Convergence criterion that checks for NaN values."""

    def _check(
        self,
        nonlinear_increment: dict[str, float],
        residual: dict[str, float],
        params: dict = {},
    ) -> Tuple[ConvergenceStatus, float, float]:
        """Check for NaN values in the nonlinear increment and residual.

        Parameters:
            nonlinear_increment: The increment in the solution variables from the
                previous nonlinear iteration.
            residual: The current residual vector of the nonlinear system.

        Returns:
            ConvergenceStatus: The convergence status of the non-linear iteration.

        """
        has_nan_increment = any(
            np.isnan(value) for value in nonlinear_increment.values()
        )
        has_nan_residual = any(np.isnan(value) for value in residual.values())
        if has_nan_increment or has_nan_residual:
            return ConvergenceStatus.DIVERGED, np.nan, np.nan
        else:
            return ConvergenceStatus.CONVERGED, 0.0, 0.0


class AbsoluteConvergenceCriterion(RelativeConvergenceCriterion):
    """Absolute convergence criterion for nonlinear problems."""

    def init_reference_value(self):
        """Initialize the reference value manager for absolute convergence."""
        self.reference_value = ReferenceValue(
            condition=lambda x: False,
            default_reference_value=0.0,
        )


class DynamicRelativeConvergenceCriterion(RelativeConvergenceCriterion):
    """Relative convergence criterion for nonlinear problems.

    Reference values are set based on the current state but are not allowed to
    be zero or nan.

    """

    def init_reference_value(self):
        """Initialize the reference value manager for relative convergence."""
        self.reference_value = ReferenceValue(
            condition=lambda x: not np.isclose(x, 0.0) and not np.isnan(x),
            default_reference_value=1.0,
        )
