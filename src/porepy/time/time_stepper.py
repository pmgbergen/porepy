"""Time-stepping strategies.

Design mirrors pp.numerics.nonlinear.nonlinear_solvers.NewtonSolver.
Main method perform_time_step() orchestrates the workflow and is
called from the model runner.

"""

import logging
from typing import Optional

import porepy as pp
from porepy.numerics.nonlinear.convergence_check import ConvergenceStatus
from porepy.numerics.time_step_control import TimeManager
from porepy.time.time_step_status import (
    TimeStepperStatus,
    TimeStepperStatusContinueIterating,
    TimeStepperStatusFailure,
    TimeStepperStatusSuccess,
)
from porepy.viz.solver_statistics import NonlinearSolverStatistics
# from porepy.time.time_step_status import TimeStepStatus

logger = logging.getLogger(__name__)


class TimeStepper:
    """Base class for time-stepping strategies.
    TODO: Docstring

    Responsibilities:
    - Orchestrate the single time-step workflow to be called from the model runner.
    - Execute trials (delegating nonlinear solves)
    - Check acceptance and rejection criteria
    - Delegate time-tracking to TimeManager

    Design mirrors NewtonSolver: acceptance criteria are checked first (positive logic),
    then rejection criteria (negative logic), and results are summarized.

    """

    """Adaptive stepper: retries with reduced dt if criteria fail.

    Meant for use of non-constant time step size.

    Workflow:
    1. For each retry (up to max_retries):
       a. Execute trial with current dt
       b. Check acceptance and rejection criteria
       c. If accepted: update time, adapt dt for next step, return
       d. If rejected: reduce dt, loop
    2. If all retries exhausted: stop

    """

    def __init__(
        self,
        time_manager: TimeManager,
        params: Optional[dict] = None,
    ) -> None:
        """Initialize the time stepper.

        Parameters:
            time_manager: TimeManager instance.
            params: Model parameters.

        """
        self.time_manager = time_manager
        """TimeManager for tracking time and dt."""
        self.params = params or {}
        """Parameters for time stepping."""

        self.max_attempts = time_manager.recomp_max
        """Maximum number of retry attempts."""

        if self.time_manager.is_constant:
            logger.warning("Overriding time manager parameters")
            self.max_attempts = 1

        assert self.max_attempts > 0, "max_attempts must be greater than 0."

    def perform_time_step(
        self,
        model: pp.PorePyModel,
        solver: pp.NewtonSolver | pp.LinearSolver,
    ) -> TimeStepperStatusSuccess | TimeStepperStatusFailure:
        """Perform a time step with accept/reject logic and retries.

        Returns:
            TimeStepStatus: ACCEPTED if criteria met, STOPPED if max retries exhausted.

        """
        # Cache previous time for trial.
        previous_time = self.time_manager.time

        # Logging time step start.
        self.log_time_step()

        for attempt in range(self.max_attempts):
            # Update time manager for new trial.
            self.time_manager.time = previous_time + self.time_manager.dt
            self.time_manager.time_index += 1

            # Attempt a standard time step.
            convergence_status = self.perform_trial_time_step(model, solver)

            if not convergence_status.is_converged():
                # Roll back if the time step attempt failed.
                self.time_manager.time = previous_time
                self.time_manager.time_index -= 1

            # New time step size based on trial results.
            time_step_status = self.compute_next_time_step(
                convergence_status, model, attempt
            )

            # Saving statistics
            self.update_time_statistics(model, time_step_status)

            # Abort simulation on success or error.
            if isinstance(
                time_step_status, (TimeStepperStatusSuccess, TimeStepperStatusFailure)
            ):
                return time_step_status

        return TimeStepperStatusFailure(
            f"Max retries ({self.max_attempts}) exhausted; stopping."
        )

    def compute_next_time_step(
        self, convergence_status: ConvergenceStatus, model: pp.PorePyModel, attempt: int
    ) -> TimeStepperStatus:
        """Compute the new time step size based on the trial status.

        Parameters:
            time_step_status: Status of the current trial (accepted/rejected/stopped).
            model: The SolutionStrategy model (for accessing statistics).
            attempt: The number of attempt to make a time step.

        Updates the time manager's dt based on the trial outcome and solver performance.
        """
        # YZ: This currently uses time_manager, but this logic is to be outsourced and
        # will be more elegant.

        if convergence_status.is_converged():
            # For accepted steps, we may want to increase dt for the next step.
            # This logic can be based on solver performance (e.g., #iterations).
            if isinstance(model.nonlinear_solver_statistics, NonlinearSolverStatistics):
                # The problem is nonlinear.
                num_iterations = model.nonlinear_solver_statistics.num_iterations
            else:
                # The problem is time-dependent and linear.
                num_iterations = 1

            self.time_manager.compute_time_step(iterations=num_iterations)
            return TimeStepperStatusSuccess()
        elif convergence_status.is_failed():
            if attempt >= (self.max_attempts - 1):
                # Limit of attempts was reached, failing.
                return TimeStepperStatusFailure(
                    f"Max retries ({self.max_attempts}) exhausted; stopping."
                )
            try:
                # For rejected steps, we want to reduce dt for the next attempt.
                model.time_manager.compute_time_step(recompute_solution=True)
                return TimeStepperStatusContinueIterating()
            except ValueError as e:
                # Time manager raises a value error if dt cannot be reduced any further.
                return TimeStepperStatusFailure(reason=str(e))
        else:
            raise ValueError(f"Unknown convergence status: {convergence_status}")

    def perform_trial_time_step(
        self,
        model: pp.PorePyModel,  #: pp.SolutionStrategy,
        solver: pp.LinearSolver | pp.NewtonSolver,
    ) -> ConvergenceStatus:
        """Perform a single time step and evaluate acceptance/rejection criteria."""

        # Execute trial time step.
        model.before_time_step()
        convergence_status = solver.solve(model)

        # Model update based on trial results.
        if convergence_status.is_converged():
            model.after_time_step_convergence()
        elif convergence_status.is_failed():
            model.after_time_step_failure()

        return convergence_status

    def log_time_step(self) -> None:
        """Log the current state of the time step."""
        logger.info(
            f"Time step #{self.time_manager.time_index}: "
            f"dt={self.time_manager.dt:.2e}, "
            f"time={self.time_manager.time:.2e} of "
            f"{self.time_manager.time_final:.2e}"
        )

    def update_time_statistics(
        self, model: pp.PorePyModel, time_step_status: TimeStepperStatus
    ) -> None:
        """Update statistics from the time step."""
        assert isinstance(model.nonlinear_solver_statistics, pp.TimeStatistics)
        model.nonlinear_solver_statistics.log_time_information(
            self.time_manager.time_index,
            self.time_manager.time,
            self.time_manager.dt,
            self.time_manager.final_time_reached(),
            time_step_status=time_step_status,
        )
