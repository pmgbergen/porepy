"""Time-step acceptance/rejection criteria with flexible data context."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import porepy as pp
from porepy.numerics.nonlinear.convergence_check import SimulationStatus
from porepy.time.time_step_status import (
    TimeStepInfo,
    TimeStepInfoCollection,
    TimeStepStatus,
    TimeStepStatusCollection,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Evaluation Context
# ============================================================================


@dataclass
class TimeStepEvaluationContext:
    """Unified context for time-step acceptance/rejection evaluation.

    This is the **single source of truth** for all data available to criteria.
    Add new fields here as new metrics become available (temporal increments,
    error estimates, etc.). Criteria access only what they need.

    Attributes:
        model: The SolutionStrategy model
        statistics: The NonlinearSolverStatistics from the solve
        solver_status: The solver_status of the current solve
        # TODO: clean up use of statistics and solver_status

    """

    model: pp.SolutionStrategy
    statistics: pp.NonlinearSolverStatistics
    solver_status: SimulationStatus
    # Future extensibility: add fields as new metrics become available


# ============================================================================
# Criterion Base Classes
# ============================================================================


class TimeStepAcceptanceCriterion(ABC):
    """Base class for time-step acceptance criteria.

    Subclasses check a specific aspect of the time step and decide whether
    to accept it (allowing time to advance) or reject it (triggering dt reduction).
    """

    @abstractmethod
    def check(
        self, context: TimeStepEvaluationContext
    ) -> tuple[TimeStepStatus, TimeStepInfo]:
        """Check the acceptance criterion.

        Parameters:
            context: TimeStepEvaluationContext with all available data.

        Returns:
            tuple: (status, diagnostic_info)
        """
        pass

    def reset(self) -> None:
        """Reset any internal state (e.g., reference values)."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of this criterion."""
        return self.__class__.__name__


class TimeStepRejectionCriterion(ABC):
    """Base class for time-step rejection criteria.

    Subclasses check for conditions that should trigger rejection and
    dt reduction.
    """

    @abstractmethod
    def check(self, context: TimeStepEvaluationContext) -> TimeStepStatus:
        """Check the rejection criterion.

        Parameters:
            context: TimeStepEvaluationContext with all available data.

        Returns:
            TimeStepStatus: ACCEPTED or REJECTED.
        """
        pass

    def reset(self) -> None:
        """Reset any internal state."""
        pass

    def rejection_msg(self) -> str:
        """Return a colored rejection message."""
        return f"\033[93m{self.__class__.__name__} triggered rejection.\033[0m"

    @property
    def description(self) -> str:
        """Human-readable description of this criterion."""
        return self.__class__.__name__


# ============================================================================
# Concrete Acceptance Criteria
# ============================================================================


class SolverConvergenceCriterion(TimeStepAcceptanceCriterion):
    """Accept if the nonlinear solver converged successfully."""

    def check(
        self, context: TimeStepEvaluationContext
    ) -> tuple[TimeStepStatus, TimeStepInfo]:
        """Check solver convergence status.

        Parameters:
            context: Evaluation context with trial.solver_status.

        Returns:
            tuple: (ACCEPTED/REJECTED, diagnostic_info)
        """
        is_accepted = context.solver_status == SimulationStatus.SUCCESSFUL
        status = TimeStepStatus.ACCEPTED if is_accepted else TimeStepStatus.REJECTED
        info = f"Solver status: {context.solver_status}"

        return status, info


class IterationCountCriterion(TimeStepAcceptanceCriterion):
    """Accept if iteration count is within acceptable range.

    Parameters:
        min_iterations: Minimum iterations (advisory).
        max_iterations: Maximum iterations (rejection threshold).
    """

    def __init__(self, min_iterations: int = 2, max_iterations: int = 15) -> None:
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

    def check(
        self, context: TimeStepEvaluationContext
    ) -> tuple[TimeStepStatus, TimeStepInfo]:
        """Check iteration count.

        Parameters:
            context: Evaluation context with trial.statistics.num_iterations.

        Returns:
            tuple: (ACCEPTED/REJECTED, diagnostic_info)
        """
        iters = context.statistics.num_iterations

        if iters > self.max_iterations:
            status = TimeStepStatus.REJECTED
            info = f"Too many iterations: {iters} > {self.max_iterations}"
        else:
            status = TimeStepStatus.ACCEPTED
            info = (
                f"Iterations: {iters} (range: [{self.min_iterations}, "
                f"{self.max_iterations}])"
            )

        return status, info


# ============================================================================
# Concrete Rejection Criteria
# ============================================================================


class SolverFailureRejectionCriterion(TimeStepRejectionCriterion):
    """Reject trial if solver reported failure (NaN, divergence, etc.)."""

    def check(self, context: TimeStepEvaluationContext) -> TimeStepStatus:
        """Check if solver failed.

        Parameters:
            context: Evaluation context with context.solver_status.

        Returns:
            TimeStepStatus: REJECTED if solver failed, ACCEPTED otherwise.
        """
        if context.solver_status == SimulationStatus.FAILED:
            logger.info(self.rejection_msg())
            return TimeStepStatus.REJECTED

        # TODO: stopped? Fetch solver_status from statistics or trial?

        return TimeStepStatus.ACCEPTED


class MaxIterationsRejectionCriterion(TimeStepRejectionCriterion):
    """Reject if nonlinear iterations exceed hard limit.

    Parameters:
        max_iterations: Hard limit on iterations.
    """

    def __init__(self, max_iterations: int = 20) -> None:
        self.max_iterations = max_iterations

    def check(self, context: TimeStepEvaluationContext) -> TimeStepStatus:
        """Check iteration count.

        Parameters:
            context: Evaluation context with trial.nonlinear_iterations.

        Returns:
            TimeStepStatus: REJECTED if limit exceeded, ACCEPTED otherwise.
        """
        if context.statistics.num_iterations >= self.max_iterations:
            logger.info(
                f"{self.rejection_msg()} "
                f"({context.statistics.num_iterations} >= {self.max_iterations})"
            )
            return TimeStepStatus.REJECTED

        return TimeStepStatus.ACCEPTED


# ============================================================================
# Criterion Collections
# ============================================================================


class TimeStepAcceptanceCriteria(dict[str, TimeStepAcceptanceCriterion]):
    """Collection of time-step acceptance criteria.

    All criteria in the collection are checked. A trial is accepted only if
    all criteria accept (logical AND). If any criterion rejects, evaluation
    stops immediately (short-circuit).

    Similar to ConvergenceCriteria in the nonlinear solver framework.

    Parameters:
        criteria: Dict[str, TimeStepAcceptanceCriterion] of named criteria.
    """

    def check(
        self, context: TimeStepEvaluationContext
    ) -> tuple[TimeStepStatusCollection, TimeStepInfoCollection]:
        """Check all acceptance criteria.

        Parameters:
            context: TimeStepEvaluationContext with all available data.

        Returns:
            tuple[TimeStepStatusCollection, TimeStepInfoCollection]:
                Collections of statuses and diagnostic info.
        """
        status_collection = TimeStepStatusCollection()
        info_collection = TimeStepInfoCollection()

        for name, criterion in self.items():
            status, info = criterion.check(context)
            status_collection[name] = status
            info_collection[name] = info

            # Short-circuit: if any criterion rejects, return immediately
            if status == TimeStepStatus.REJECTED:
                logger.debug(f"Trial rejected by acceptance criterion '{name}': {info}")
                return status_collection, info_collection

        logger.debug("Trial accepted by all acceptance criteria.")
        return status_collection, info_collection

    def reset(self) -> None:
        """Reset all criteria in the collection."""
        for criterion in self.values():
            criterion.reset()


class TimeStepRejectionCriteria(dict[str, TimeStepRejectionCriterion]):
    """Collection of time-step rejection criteria.

    All criteria are checked. A trial is rejected if any criterion detects
    rejection conditions (logical OR). Evaluation stops on first rejection.

    Similar to DivergenceCriteria in the nonlinear solver framework.

    Parameters:
        criteria: Dict[str, TimeStepRejectionCriterion] of named criteria.
    """

    def check(self, context: TimeStepEvaluationContext) -> TimeStepStatusCollection:
        """Check all rejection criteria.

        Parameters:
            context: TimeStepEvaluationContext with all available data.

        Returns:
            TimeStepStatusCollection: Status for each criterion.
        """
        status_collection = TimeStepStatusCollection()

        for name, criterion in self.items():
            status = criterion.check(context)
            status_collection[name] = status

            # Short-circuit: if any criterion rejects, return immediately
            if status == TimeStepStatus.REJECTED:
                logger.debug(f"Trial rejected by rejection criterion '{name}'")
                return status_collection

        logger.debug("Trial accepted by all rejection criteria.")
        return status_collection

    def reset(self) -> None:
        """Reset all criteria in the collection."""
        for criterion in self.values():
            criterion.reset()


# ============================================================================
# Convenience Factory Functions
# ============================================================================


def default_time_step_criteria() -> tuple[
    TimeStepAcceptanceCriteria, TimeStepRejectionCriteria
]:
    """Create default time-step acceptance and rejection criteria.

    Recommended for most applications. Checks solver convergence, iteration
    count, and basic failure modes.

    Returns:
        tuple: (acceptance_criteria, rejection_criteria)
    """
    acceptance_criteria = TimeStepAcceptanceCriteria(
        {
            "solver_convergence": SolverConvergenceCriterion(),
        }
    )

    rejection_criteria = TimeStepRejectionCriteria(
        {
            "solver_failure": SolverFailureRejectionCriterion(),
            # TODO: Remove hardcoded.
            "max_iterations": MaxIterationsRejectionCriterion(max_iterations=15),
        }
    )

    return acceptance_criteria, rejection_criteria


def permissive_time_step_criteria() -> tuple[
    TimeStepAcceptanceCriteria, TimeStepRejectionCriteria
]:
    """Create permissive time-step criteria for robust problems.

    Minimal checks; use for initial testing or very robust problems.

    Returns:
        tuple: (acceptance_criteria, rejection_criteria)
    """
    acceptance_criteria = TimeStepAcceptanceCriteria(
        {
            "solver_convergence": SolverConvergenceCriterion(),
        }
    )

    rejection_criteria = TimeStepRejectionCriteria(
        {
            "solver_failure": SolverFailureRejectionCriterion(),
        }
    )

    return acceptance_criteria, rejection_criteria
