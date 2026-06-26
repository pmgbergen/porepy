"""Time-stepping strategies.

Design mirrors pp.numerics.nonlinear.nonlinear_solvers.NewtonSolver.
Main method perform_time_step() orchestrates the workflow and is
called from the model runner.

"""

from __future__ import annotations

import logging

import porepy as pp
from porepy.numerics.nonlinear.convergence_check import ConvergenceStatusCollection
from porepy.numerics.nonlinear.nonlinear_solver_status import (
    NonlinearSolverStatus,
    NonlinearSolverStatusConverged,
    NonlinearSolverStatusFailed,
)
from porepy.numerics.time_step_control import TimeManager
from porepy.time.time_step_status import (
    TimeStepperStatus,
    TimeStepperStatusContinueIterating,
    TimeStepperStatusFailure,
    TimeStepperStatusSuccess,
)
from porepy.viz.solver_statistics import NonlinearSolverStatistics

logger = logging.getLogger(__name__)


class TimeStepper:
    """Base class for time-stepping strategies.

    Responsibilities:
    - Orchestrate the single time-step workflow to be called from the model runner.
    - Execute trials (delegating nonlinear solves)
    - Adjust dt (currently delegated to TimeManager)

    Workflow:
    1. For each retry (up to max_retries):
       a. Execute trial with current dt
       b. If success: update time, adapt dt for next step, return
       c. If rejected: reduce dt, loop
    2. If all retries exhausted: stop

    The constant dt case is supported internally by setting max_retries = 1.

    Parameters:
        time_manager: TimeManager instance.

    """

    def __init__(self, time_manager: TimeManager) -> None:
        """Initialize the time stepper."""
        self.time_manager = time_manager
        """TimeManager for tracking time and dt."""

        self.max_attempts = time_manager.recomp_max + 1
        """Maximum number of retry attempts. Set it to 1 for no retries, which is
        equivalent to the constant_dt policy."""
        # Note: +1 because recomp_max=0 means a single attempt.

        if self.time_manager.is_constant:
            logger.warning("Overriding time manager parameters to ensure constant_dt.")
            self.max_attempts = 1
            time_manager.recomp_max = 0

        assert self.max_attempts > 0, "max_attempts must be greater than 0."

    def perform_time_step(
        self,
        model: pp.PorePyModel,
        solver: pp.NewtonSolver | pp.LinearSolver,
    ) -> TimeStepperStatusSuccess | TimeStepperStatusFailure:
        """Perform a time step. If the nonlinear solver fails, cut the time step and
        retry.

        Returns:
            TimeStepperStatus: Success if criteria met, Failure if max retries exhausted
                or dt_min is reached, or something went unexpectedly wrong.

        """
        # Cache previous time for trial.
        previous_time = self.time_manager.time

        # Logging time step start.
        _log_time_step(time_manager=self.time_manager)

        for attempt in range(self.max_attempts):
            # Update time manager for new trial.
            self.time_manager.time = previous_time + self.time_manager.dt
            self.time_manager.time_index += 1

            # Attempt a standard time step.
            nonlinear_solver_status = self._perform_trial_time_step(model, solver)

            if not nonlinear_solver_status.is_converged():
                # Roll back if the time step attempt failed.
                self.time_manager.time = previous_time
                self.time_manager.time_index -= 1

            # New time step size based on trial results.
            time_step_status = self._compute_next_time_step(
                nonlinear_solver_status, model, attempt
            )

            # Saving statistics
            self._update_time_statistics(model, time_step_status)

            # Return on success or error when there is no way to continue trying.
            if isinstance(
                time_step_status, (TimeStepperStatusSuccess, TimeStepperStatusFailure)
            ):
                return time_step_status

        # We should never reach this code, but it is added as a safeguard.
        return TimeStepperStatusFailure(
            reason=f"Max retries ({self.max_attempts}) exhausted; stopping.",
            nonlinear_solver_status=NonlinearSolverStatusFailed(
                convergence_statuses=ConvergenceStatusCollection(),
                divergence_statuses=ConvergenceStatusCollection(),
            ),
        )

    def _compute_next_time_step(
        self,
        nonlinear_solver_status: NonlinearSolverStatus,
        model: pp.PorePyModel,
        attempt: int,
    ) -> TimeStepperStatus:
        """Compute the new dt based on the convergence status and solver performance.
        Decides what to do next (end or continue iterating) by returning the
        TimeStepperStatus.

        Parameters:
            nonlinear_solver_status: Nonlinear solver status (converged/failed).
            model: The PorePy model (for accessing statistics).
            attempt: The number of attempt to make a time step.

        """
        # YZ: This currently uses time_manager, but this logic is to be outsourced and
        # will be more elegant.

        if isinstance(nonlinear_solver_status, NonlinearSolverStatusConverged):
            # For accepted steps, we may want to increase dt for the next step.
            # This logic can be based on solver performance (e.g., #iterations).
            if isinstance(model.nonlinear_solver_statistics, NonlinearSolverStatistics):
                # The problem is nonlinear.
                num_iterations = model.nonlinear_solver_statistics.num_iterations
            else:
                # The problem is time-dependent and linear.
                num_iterations = 1

            self.time_manager.compute_time_step(iterations=num_iterations)
            return TimeStepperStatusSuccess(
                nonlinear_solver_status=nonlinear_solver_status
            )

        elif isinstance(nonlinear_solver_status, NonlinearSolverStatusFailed):
            if attempt >= (self.max_attempts - 1):
                # Limit of attempts was reached, failing.
                return TimeStepperStatusFailure(
                    nonlinear_solver_status=nonlinear_solver_status,
                    reason=f"Max retries ({self.max_attempts}) exhausted; stopping.",
                )
            try:
                # For rejected steps, we want to reduce dt for the next attempt.
                self.time_manager.compute_time_step(recompute_solution=True)
                return TimeStepperStatusContinueIterating(
                    attempt=attempt, nonlinear_solver_status=nonlinear_solver_status
                )
            except ValueError as e:
                # Time manager raises a value error if dt cannot be reduced any further.
                return TimeStepperStatusFailure(
                    nonlinear_solver_status=nonlinear_solver_status, reason=str(e)
                )

        # This should never happen since the contract is that the nonlinear solver
        # should return either CONVERGED or FAILED.
        else:
            error_msg = f"Unknown nonlinear solver status: {nonlinear_solver_status}"
            logger.error(error_msg)
            return TimeStepperStatusFailure(
                reason=error_msg, nonlinear_solver_status=nonlinear_solver_status
            )

    def _perform_trial_time_step(
        self,
        model: pp.PorePyModel,
        solver: pp.LinearSolver | pp.NewtonSolver,
    ) -> NonlinearSolverStatus:
        """Perform a nonlinear solve to make the time step.

        Returns:
            The nonlinear solver status (converged/failed).

        """

        # Execute trial time step.
        model.before_time_step()
        nonlinear_solver_status = solver.solve(model)  # type: ignore

        # Model update based on trial results.
        if nonlinear_solver_status.is_converged():
            model.after_time_step_convergence()
        elif nonlinear_solver_status.is_failed():
            model.after_time_step_failure()

        return nonlinear_solver_status

    def _update_time_statistics(
        self, model: pp.PorePyModel, time_step_status: TimeStepperStatus
    ) -> None:
        """Update statistics from the time step."""
        assert isinstance(model.nonlinear_solver_statistics, pp.TimeStatistics)
        model.nonlinear_solver_statistics.log_simulation_status(
            simulation_status=time_step_status
        )
        model.nonlinear_solver_statistics.log_time_information(
            self.time_manager.time_index,
            self.time_manager.time,
            self.time_manager.dt,
            self.time_manager.final_time_reached(),
        )


def _log_time_step(time_manager: pp.TimeManager) -> None:
    """Log the current state of the time step."""
    logger.info(
        f"Time step #{time_manager.time_index}: "
        f"dt={time_manager.dt:.2e}, "
        f"time={time_manager.time:.2e} of "
        f"{time_manager.time_final:.2e}"
    )
