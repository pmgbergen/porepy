"""Common functions, utilized both by the NewtonSolver and the LinearSolver."""

from logging import getLogger

from porepy.models.solution_strategy import SolutionStrategy
from porepy.numerics.nonlinear.convergence_check import ConvergenceStatusCollection
from porepy.numerics.nonlinear.nonlinear_solver_status import (
    NonlinearSolverStatus,
    NonlinearSolverStatusConverged,
    NonlinearSolverStatusFailed,
)

logger = getLogger(__name__)


def update_solver_statistics_after_nonlinear_solve(
    model: SolutionStrategy,
    solver_status: NonlinearSolverStatus,
) -> None:
    """Update the solver statistics in the model.

    Parameters:
        model: The model instance specifying the problem to be solved.
        solver_status: Simulation status of the solver.

    """
    # Basic discretization-related information and overall simulation status.
    model.nonlinear_solver_statistics.log_solver_status(solver_status)
    model.nonlinear_solver_statistics.log_mesh_information(model.mdg.subdomains())


def summarize_solver_status(
    convergence_status: ConvergenceStatusCollection,
    divergence_status: ConvergenceStatusCollection,
) -> NonlinearSolverStatus:
    """Called by the nonlinear solver after the nonlinear iteration is done. Considers a
    collection of convergence and divergence statuses from multiple criteria and makes a
    overall verdict on whether we accept the sollution or not.

    NOTE: Convergence status takes precedence over divergence status.

    Parameters:
        convergence_status: Multiple convergence statuses from different criteria.
        divergence_status: Multiple divergence statuses from variaous criteria.

    Returns:
        NonlinearSolverStatus: Either Converged or Failed.

    """
    is_converged = convergence_status.is_converged()
    is_diverged = divergence_status.is_diverged()
    if is_converged:
        if is_diverged:
            logger.warning(
                "Nonlinear solver convergence criteria indicate convergence and "
                "divergence at the same time. Accepting this solution."
            )
        return NonlinearSolverStatusConverged(
            convergence_statuses=convergence_status,
            divergence_statuses=divergence_status,
        )
    elif is_diverged:
        logger.warning("Failed to solve the nonlinear problem.")
        return NonlinearSolverStatusFailed(
            convergence_statuses=convergence_status,
            divergence_statuses=divergence_status,
        )
    else:
        logger.error(
            "Nonlinear solver did not fail, but the convergence criterion did not "
            "accept the solution. Treating it as a failure."
        )
        return NonlinearSolverStatusFailed(
            convergence_statuses=convergence_status,
            divergence_statuses=divergence_status,
        )
