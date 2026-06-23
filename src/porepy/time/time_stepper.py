"""Time-stepping strategies.

Design mirrors pp.numerics.nonlinear.nonlinear_solvers.NewtonSolver.
Main method perform_time_step() orchestrates the workflow and is
called from the model runner.

"""

import logging
from abc import ABC, abstractmethod

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
        params: dict = None,
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
            # assert self.time_manager.dt_min_max[0] == self.time_manager.dt_min_max[1]

        assert self.max_attempts > 0, "max_attempts must be greater than 0."

        # Cache previous time at the start of the trial for use in retries.
        self.previous_time = self.time_manager.time
        """Cached time at the start of the current trial."""

        # self.init_acceptance_criteria()
        # self.init_rejection_criteria()

    # def init_acceptance_criteria(self) -> None:
    #     """Parse and initialize acceptance criteria."""
    #     # TODO: Extend to reading criteria from params like in NonlinearSolver.
    #     self.acceptance_criteria, _ = default_time_step_criteria()
    #     """Acceptance criteria for time-step trials."""

    # def init_rejection_criteria(self) -> None:
    #     """Parse and initialize rejection criteria."""
    #     # TODO: Extend to reading criteria from params like in NonlinearSolver.
    #     _, self.rejection_criteria = default_time_step_criteria()
    #     """Rejection criteria for time-step trials."""

    def perform_time_step(
        self,
        model,  #: pp.SolutionStrategy,
        solver,  #: pp.NewtonSolver,
    ) -> TimeStepperStatusSuccess | TimeStepperStatusFailure:
        """Perform a time step with accept/reject logic and retries.

        Returns:
            TimeStepStatus: ACCEPTED if criteria met, STOPPED if max retries exhausted.

        """
        # Advance time index for entire time step.
        # self.time_manager.increase_time_index()

        # Cache previous time for trial.
        self.previous_time = self.time_manager.time

        for _ in range(self.max_attempts):
            # Update time manager for new trial (if not first attempt).
            # NOTE: No use of self.time_manager.increase_time() here.
            self.time_manager.time = self.previous_time + self.time_manager.dt
            self.time_manager.time_index += 1

            # Attempt a standard time step.
            convergence_status = self.perform_trial_time_step(model, solver)

            if not convergence_status.is_converged():
                self.time_manager.time = self.previous_time
                self.time_manager.time_index -= 1

            # New time step size based on trial results.
            time_step_status = self.compute_next_time_step(convergence_status, model)

            # Abort simulation on success or error.
            if isinstance(
                time_step_status, (TimeStepperStatusSuccess, TimeStepperStatusFailure)
            ):
                return time_step_status

        return TimeStepperStatusFailure(
            f"Max retries ({self.max_attempts}) exhausted; stopping."
        )

    def compute_next_time_step(
        self, convergence_status: ConvergenceStatus, model: pp.PorePyModel
    ) -> TimeStepperStatus:
        """Compute the new time step size based on the trial status.

        Parameters:
            time_step_status: Status of the current trial (accepted/rejected/stopped).
            model: The SolutionStrategy model (for accessing statistics).

        Updates the time manager's dt based on the trial outcome and solver performance.
        """
        # TODO: Update time manager's computation of dt. E.g.
        # dt = self.time_criteria.compute_time_step(context)
        # self.time_manager.set_dt(dt) # clips into range.

        if convergence_status.is_converged():
            # For accepted steps, we may want to increase dt for the next step.
            # This logic can be based on solver performance (e.g., #iterations).
            if isinstance(model.nonlinear_solver_statistics, NonlinearSolverStatistics):
                # The problem is nonlinear.
                num_iterations = model.nonlinear_solver_statistics.num_iterations
            else:
                # The problem is time-dependent and linear.
                num_iterations = 1

            model.time_manager.compute_time_step(iterations=num_iterations)
            return TimeStepperStatusSuccess()
        elif convergence_status.is_failed():
            try:
                # For rejected steps, we want to reduce dt for the next attempt.
                model.time_manager.compute_time_step(recompute_solution=True)
                return TimeStepperStatusContinueIterating()
            except ValueError as e:
                # Time manager raises a value error if dt cannot be reduced any further.
                # TODO: this will be more elegant when this logic is moved out of the
                # time manager.
                return TimeStepperStatusFailure(reason=str(e))
        else:
            raise ValueError(f"Unknown convergence status: {convergence_status}")

    def perform_trial_time_step(
        self,
        model,  #: pp.SolutionStrategy,
        solver: pp.LinearSolver | pp.NewtonSolver,
    ) -> ConvergenceStatus:
        """Perform a single time step and evaluate acceptance/rejection criteria."""

        # Execute trial time step.
        model.before_time_step()
        convergence_status = solver.solve(model)

        # # Build context once for both acceptance and rejection checks.
        # context = self._build_evaluation_context(
        #     model, model.nonlinear_solver_statistics, solver_status
        # )

        # Check criteria using context.
        # acceptance_status, acceptance_info = self.acceptance_criteria.check(context)
        # rejection_status = self.rejection_criteria.check(context)

        # # Summarize trial status.
        # time_step_status = self.summarize_time_step_status(
        #     acceptance_status, rejection_status
        # )

        # Logging.
        self.log_time_step()

        # Update statistics
        # self.update_time_statistics(
        #     model,
        #     time_step_status,
        #     acceptance_status,
        #     rejection_status,
        #     acceptance_info,
        # )

        # Model update based on trial results.
        self.after_time_step(convergence_status, model)

        return convergence_status

    # def _build_evaluation_context(
    #     self,
    #     model: "pp.SolutionStrategy",
    #     statistics: "pp.NonlinearSolverStatistics",  # TODO: clean up use of statistics
    #     solver_status: SimulationStatus,  # TODO: clean up
    # ) -> TimeStepEvaluationContext:
    #     """Build the evaluation context for acceptance/rejection checking.

    #     This centralizes all data needed by criteria into a single container.
    #     Optional fields (temporal_increment) can be computed
    #     on-demand or left None if not available.

    #     Parameters:
    #         model: The SolutionStrategy model
    #         statistics: The NonlinearSolverStatistics from the solve
    #         solver_status: The SimulationStatus from the solve

    #     Returns:
    #         TimeStepEvaluationContext: Unified context for criteria evaluation.
    #     """
    #     context = TimeStepEvaluationContext(
    #         model=model,
    #         statistics=statistics,
    #         solver_status=solver_status,
    #     )
    #     return context

    # def summarize_time_step_status(
    #     self,
    #     acceptance_status: TimeStepStatusCollection,
    #     rejection_status: TimeStepStatusCollection,
    # ) -> TimeStepStatus:
    #     """Conclude on the overall trial status.

    #     NOTE: Acceptance status takes precedence; rejection status is checked only
    #     if acceptance is mixed. Final status determines if trial is
    #     accepted/rejected/stopped.

    #     Parameters:
    #         acceptance_status: Acceptance criterion statuses.
    #         rejection_status: Rejection criterion statuses.

    #     Returns:
    #         TimeStepStatus: ACCEPTED, REJECTED, or STOPPED.
    #     """
    #     if acceptance_status.is_accepted():
    #         # All acceptance criteria passed
    #         time_step_status = TimeStepStatus.ACCEPTED
    #         logger.debug("Trial accepted by all acceptance criteria.")
    #     elif rejection_status.is_rejected():
    #         # Any rejection criterion triggered
    #         time_step_status = TimeStepStatus.REJECTED
    #         logger.debug("Trial rejected by rejection criteria.")
    #     else:
    #         raise ValueError(
    #             f"Invalid time step acceptance status {acceptance_status} and "
    #             f"rejection status {rejection_status}."
    #         )

    #     return time_step_status

    def after_time_step(self, convergence_status: ConvergenceStatus, model) -> None:
        """Update model state after time step based on trial status."""
        if convergence_status.is_converged():
            model.after_time_step_convergence()
        elif convergence_status.is_failed():
            model.after_time_step_failure()

    def log_time_step(self) -> None:
        """Log the current state of the time step."""
        logger.info(
            f"Time step #{self.time_manager.time_index}: "
            f"dt={self.time_manager.dt:.2e}, "
            f"time={self.time_manager.time:.2e} of "
            f"{self.time_manager.time_final:.2e}"
        )

    # def update_time_statistics(
    #     self,
    #     model,  #: pp.SolutionStrategy,
    #     time_step_status: TimeStepStatus,
    #     acceptance_status,
    #     rejection_status,
    #     acceptance_info,
    # ) -> None:
    #     """Update statistics from the time step.

    #     Parameters:
    #         model: The SolutionStrategy model.
    #         time_step_status: Status of the time step (accepted/rejected/stopped).
    #         acceptance_status: Acceptance statuses.
    #         rejection_status: Rejection statuses.
    #         info: Diagnostic information.
    #     """
    #     # TODO time_step_status, acceptance_status/rejection_status, acceptance_info

    #     assert isinstance(model.nonlinear_solver_statistics, pp.TimeStatistics)
    #     model.nonlinear_solver_statistics.log_time_information(
    #         model.time_manager.time_index,
    #         model.time_manager.time,
    #         model.time_manager.dt,
    #         model.time_manager.final_time_reached(),
    #     )


# # ============================================================================
# # Time stepper factory
# # ============================================================================


class TimeStepperFactory:
    """Factory for creating time steppers based on time manager configuration.

    Selects between DirectTimeStepper (constant dt) and AdaptiveTimeStepper
    (adaptive dt) depending on the time manager's configuration.
    """

    @staticmethod
    def create_time_stepper(
        time_manager: TimeManager, params: dict | None
    ) -> TimeStepper:
        """Create an appropriate time stepper based on time manager configuration.

        Parameters:
            time_manager: The TimeManager instance
            params:  Model parameters

        Returns:
            TimeStepper: DirectTimeStepper if dt is constant, AdaptiveTimeStepper
                otherwise.

        """
        return TimeStepper(time_manager, params)
        # Check if time stepping is constant or adaptive
        is_constant_dt = getattr(time_manager, "is_constant", True)
        if is_constant_dt:
            logger.info(
                "Time manager configured for constant dt; using DirectTimeStepper."
            )
            return DirectTimeStepper(time_manager, params)
        else:
            logger.info(
                "Time manager configured for adaptive dt; using AdaptiveTimeStepper."
            )
            return AdaptiveTimeStepper(time_manager, params)
