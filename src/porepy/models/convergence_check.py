"""Collection of objects and functions related to convergence checking in models."""

from enum import Enum
import numpy as np
from abc import abstractmethod


class ConvergenceStatus(Enum):
    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    DIVERGED = "diverged"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, status_str: str):
        """Convert a string to a ConvergenceStatus."""
        return cls[status_str.upper()]

    def is_converged(self) -> bool:
        """Check if the status indicates convergence."""
        return self == ConvergenceStatus.CONVERGED

    def is_not_converged(self) -> bool:
        """Check if the status indicates not converged."""
        return self == ConvergenceStatus.NOT_CONVERGED

    def is_diverged(self) -> bool:
        """Check if the status indicates divergence."""
        return self == ConvergenceStatus.DIVERGED


class ConvergenceCriterion:
    @abstractmethod
    def check(
        self, nonlinear_increment: np.ndarray, residual: np.ndarray, params: dict
    ) -> ConvergenceStatus:
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


class NanConvergenceCriterion(ConvergenceCriterion):
    """Convergence criterion that checks for NaN values."""

    def check(
        self, nonlinear_increment: np.ndarray, residual: np.ndarray, params: dict = {}
    ) -> ConvergenceStatus:
        """Check for NaN values in the nonlinear increment and residual.

        Parameters:
            nonlinear_increment: The increment in the solution variables from the
                previous nonlinear iteration.
            residual: The current residual vector of the nonlinear system.

        Returns:
            ConvergenceStatus: The convergence status of the non-linear iteration.

        """
        if bool(np.any(np.isnan(nonlinear_increment))) or bool(
            np.any(np.isnan(residual))
        ):
            return ConvergenceStatus.DIVERGED
        else:
            return ConvergenceStatus.CONVERGED


class SingleObjectiveConvergenceCriterion(ConvergenceCriterion):
    """Base class for single physics convergence criteria."""

    def check(self, nonlinear_increment_norm, residual_norm, params):
        """Check convergence for a single physics model."""
        # Check divergence.
        is_diverged = (
            params["nl_divergence_tol"] is not np.inf
            and residual_norm > params["nl_divergence_tol"]
        )

        # Check convergence requiring both the increment and residual to be small.
        converged_inc = (
            params["nl_convergence_tol"] is np.inf
            or nonlinear_increment_norm < params["nl_convergence_tol"]
        )
        converged_res = (
            params["nl_convergence_tol_res"] is np.inf
            or residual_norm < params["nl_convergence_tol_res"]
        )
        is_converged = converged_inc and converged_res

        convegence_status = ConvergenceStatus.NOT_CONVERGED
        if is_diverged:
            convegence_status = ConvergenceStatus.DIVERGED
        elif is_converged:
            convegence_status = ConvergenceStatus.CONVERGED

        return convegence_status
