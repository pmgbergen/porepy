"""Time-stepping strategies.

Design mirrors pp.numerics.nonlinear.nonlinear_solvers.NewtonSolver. Main method
perform_time_step() orchestrates the workflow and is called from the model runner.

"""

from __future__ import annotations

import logging

import porepy as pp
from porepy.numerics import solvers
from porepy.time_stepper.time_step_control import TimeManager
from porepy.time_stepper.time_step_status import (
    TimeStepperStatus,
    TimeStepperStatusContinueIterating,
    TimeStepperStatusFailure,
    TimeStepperStatusSuccess,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TimeStepper",
]


class TimeStepper:
    """A class for defining a time-stepping strategy.

    Responsibilities:
    - Orchestrate the single time-step workflow to be called from the model runner.
    - Execute trials (delegating nonlinear solves)
    - Adjust dt (currently delegated to TimeManager)

    Workflow:
    1. For each retry (up to max_retries):
        a. Update trial time;
        b. Execute trial with current dt;
        c. If success: update solution values, adapt dt for next step, return;
        d. If rejected: reduce dt, loop, revert trial time.
    2. If all retries exhausted: return.

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
        solver: pp.solvers.NonlinearSolverBase,
    ) -> TimeStepperStatusSuccess | TimeStepperStatusFailure:
        """Perform a time step. If the nonlinear solver fails, alter the time step and
        retry.

        Parameters:
            model: The PorePy model to perform a time step on.
            solver: The nonlinear solver to integrate the discretized problem.

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

            # Log time step information for statistics.
            self._update_time_statistics(model)

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

            # Log time step status for statistics.
            self._update_nonlinear_solver_statistics(model, time_step_status)

            # Update model (also saves logged statistics) based on trial results.
            self._update_model_after_trial(model, time_step_status)

            # Return on success or error when there is no way to continue trying.
            if isinstance(
                time_step_status, (TimeStepperStatusSuccess, TimeStepperStatusFailure)
            ):
                return time_step_status

        # We should never reach this code, but it is added as a safeguard.
        return TimeStepperStatusFailure(
            reason=f"Max retries ({self.max_attempts}) exhausted; stopping.",
            nonlinear_solver_status=solvers.NewtonSolverFailed(
                linear_solver_statuses=[],
                convergence_statuses=solvers.ConvergenceStatusCollection(),
                divergence_statuses=solvers.ConvergenceStatusCollection(),
            ),
        )

    def _compute_next_time_step(
        self,
        nonlinear_solver_status: solvers.NonlinearSolverStatus,
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

        if isinstance(nonlinear_solver_status, solvers.NonlinearSolverStatusConverged):
            # For accepted steps, we may want to increase dt for the next step.
            # This logic can be based on solver performance (e.g., #iterations).
            num_iterations = nonlinear_solver_status.number_of_iterations()

            current_dt = self.time_manager.dt
            new_time = self.time_manager.time
            self.time_manager.compute_time_step(iterations=num_iterations)
            return TimeStepperStatusSuccess(
                dt=current_dt,
                time=new_time,
                nonlinear_solver_status=nonlinear_solver_status,
            )

        elif isinstance(nonlinear_solver_status, solvers.NonlinearSolverStatusFailed):
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
        solver: pp.solvers.NonlinearSolverBase,
    ) -> solvers.NonlinearSolverStatus:
        """Perform a nonlinear solve to make the time step.

        Returns:
            The nonlinear solver status (converged/failed).

        """
        # Execute trial time step.
        model.before_time_step()
        nonlinear_solver_status = solver.solve(model)  # type: ignore

        return nonlinear_solver_status

    def _update_model_after_trial(
        self, model: pp.PorePyModel, time_step_status: TimeStepperStatus
    ) -> None:
        """Model update based on trial results."""
        if time_step_status.is_success():
            model.after_time_step_convergence()
        elif time_step_status.is_failure() or isinstance(
            time_step_status, TimeStepperStatusContinueIterating
        ):
            model.after_time_step_failure()

    def _update_time_statistics(self, model: pp.PorePyModel) -> None:
        """Update statistics from the time step."""
        assert isinstance(model.nonlinear_solver_statistics, pp.TimeStatistics)
        model.nonlinear_solver_statistics.log_time_information(
            self.time_manager.time_index,
            self.time_manager.time,
            self.time_manager.dt,
            self.time_manager.final_time_reached(),
        )

    def _update_nonlinear_solver_statistics(
        self, model: pp.PorePyModel, time_step_status: TimeStepperStatus
    ) -> None:
        """Update statistics from the time step."""
        model.nonlinear_solver_statistics.log_simulation_status(
            simulation_status=time_step_status
        )


def _log_time_step(time_manager: pp.time_stepper.TimeManager) -> None:
    """Log the current state of the time step."""
    logger.info(
        f"Time step #{time_manager.time_index}: "
        f"dt={time_manager.dt:.2e}, "
        f"time={time_manager.time:.2e} of "
        f"{time_manager.time_final:.2e}"
    )
