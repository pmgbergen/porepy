"""Time-stepping strategies.

Design mirrors pp.numerics.nonlinear.nonlinear_solvers.NewtonSolver. Main method
perform_time_step() orchestrates the workflow and is called from the model runner.

"""

from __future__ import annotations

import logging
import numpy as np
import porepy as pp
from porepy.numerics import solvers
from porepy.time_stepper.scheduler import (
    CannotRecomputeTimeStep,
    TimeSchedulerBase,
)
from porepy.time_stepper.time_step_status import (
    TimeStepperAttemptData,
    TimeStepperStatus,
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
        a. Execute trial with current dt;
        b. If success: update solution values, adapt dt for next step, return;
        c. If rejected: reduce dt, loop, revert trial time.
    2. If all retries exhausted: return.

    The constant dt case is supported internally by setting max_retries = 1.

    Parameters:
        time_manager: TimeManager instance.

    """

    def __init__(self, scheduler: TimeSchedulerBase, max_attempts: int = 10) -> None:
        """Initialize the time stepper."""
        self.scheduler = scheduler

        assert max_attempts > 0, "max_attempts must be greater than 0."
        self.max_attempts = max_attempts
        """Maximum number of retry attempts. Set it to 1 for no retries, which is
        equivalent to the constant_dt policy.

        """

    def perform_time_step(
        self,
        model: pp.PorePyModel,
        solver: pp.solvers.NonlinearSolverBase,
    ) -> TimeStepperStatus:
        """Perform a time step. If the nonlinear solver fails, alter the time step and
        retry.

        Parameters:
            model: The PorePy model to perform a time step on.
            solver: The nonlinear solver to integrate the discretized problem.

        Returns:
            TimeStepperStatus: Success if criteria met, Failure if max retries exhausted
                or dt_min is reached, or something went unexpectedly wrong.

        """
        previous_time = self.scheduler.get_time()
        dt = self.scheduler.get_dt()
        attempts_data = []

        for attempt in range(self.max_attempts):
            # Logging time step start.
            log_message = (
                f"Time step #{self.scheduler.get_time_index_successful()}: dt={dt:.2e},"
                f" time={previous_time:.2e} of {self.scheduler.get_time_end():.2e}"
            )
            if attempt > 0:
                log_message += f", retry={attempt + 1} / {self.max_attempts}"
            logger.info(log_message)

            # Update time manager for new trial.
            previous_time_data = model.time_data
            model.time_data = pp.time_stepper.SimulationTimeData(
                time=previous_time + dt,
                dt=dt,
                time_index_successful=(self.scheduler.get_time_index_successful() + 1),
                schedule=self.scheduler.get_schedule(),
                constant_dt=isinstance(
                    self.scheduler, pp.time_stepper.scheduler.TimeSchedulerConstantDt
                ),
            )

            # Log time step information for statistics.
            self._update_time_statistics(model)

            # Attempt a standard time step.
            nonlinear_solver_status = self._perform_trial_time_step(model, solver)
            attempt_data = TimeStepperAttemptData(
                dt=dt, nonlinear_solve_status=nonlinear_solver_status
            )
            attempts_data.append(attempt_data)
            # Save statistics.
            self._update_time_statistics(model, attempt_data)

            if not nonlinear_solver_status.is_converged():
                # Roll back if the time step attempt failed. This is needed in case if
                # the simulation stops and the time_data is not reassigned above in the
                # next loop iteration.
                model.time_data = previous_time_data

            try:
                # New time step size based on trial results.
                dt = self.scheduler.compute_next_time_step(
                    success=nonlinear_solver_status.is_converged(),
                    context={
                        "model": model,
                        "nonlinear_solver_status": nonlinear_solver_status,
                    },
                )
            except CannotRecomputeTimeStep as exc:
                reason = str(exc.args[0])
                return TimeStepperStatusFailure(
                    reason=reason,
                    time=previous_time,
                    attempts=attempts_data,
                )

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
            time=previous_time,
            attempts=attempts_data,
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
