"""Time-stepping strategies.

Design mirrors pp.numerics.nonlinear.nonlinear_solvers.NewtonSolver. Main method
perform_time_step() orchestrates the workflow and is called from the model runner.

"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import porepy as pp
from porepy.numerics import solvers
from porepy.time_stepper.scheduler import (
    CannotRecomputeTimeStep,
    SimulationTimeData,
    TimeSchedulerBase,
)
from porepy.time_stepper.time_step_status import (
    TimeStepperAttemptData,
    TimeStepperStatus,
    TimeStepperStatusFailure,
    TimeStepperStatusInProgress,
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
        previous_time = model.time_data.time
        attempts_data: list[TimeStepperAttemptData] = []

        for attempt in range(self.max_attempts):
            # Update time manager for new trial.
            previous_time_data = model.time_data
            trial_time_data = self.scheduler.generate_time_data(trial=True)
            model.time_data = trial_time_data

            # Logging time step start.
            log_message = (
                f"Time step #{trial_time_data.time_index_successful}: dt="
                f"{trial_time_data.dt:.2e}, time={previous_time:.2e} of "
                f"{trial_time_data.schedule[-1]:.2e}"
            )
            if attempt > 0:
                log_message += f",  attempt={attempt + 1} / {self.max_attempts}"
            logger.info(log_message)

            # Attempt a standard time step.
            nonlinear_solver_status = self._perform_trial_time_step(model, solver)

            if not nonlinear_solver_status.is_converged():
                # Roll back if the time step attempt failed. This is needed in case if
                # the simulation stops and the time_data is not reassigned above in the
                # next loop iteration.
                model.time_data = previous_time_data

            attempts_data.append(
                TimeStepperAttemptData(
                    dt=trial_time_data.dt,
                    nonlinear_solve_status=nonlinear_solver_status,
                )
            )

            success = nonlinear_solver_status.is_converged()
            try:
                # New time step size based on trial results.
                self.scheduler.compute_next_time_step(
                    success=success,
                    context={
                        "model": model,
                        "nonlinear_solver_status": nonlinear_solver_status,
                    },
                )
            except CannotRecomputeTimeStep as exc:
                reason = str(exc.args[0])
                return self._log_and_return_time_step_data(
                    model=model,
                    time_step_data=TimeStepperStatusFailure(
                        reason=reason,
                        time=previous_time,
                        attempts=attempts_data,
                    ),
                    time_data=trial_time_data,
                )

            if success:
                return self._log_and_return_time_step_data(
                    model=model,
                    time_step_data=TimeStepperStatusSuccess(
                        time=model.time_data.time, attempts=attempts_data
                    ),
                    time_data=trial_time_data,
                )
            else:
                _ = self._log_and_return_time_step_data(
                    model=model,
                    time_data=trial_time_data,
                    time_step_data=TimeStepperStatusInProgress(attempts=attempts_data),
                )

        # TODO YZ: Test what if we reach max attempts.
        return self._log_and_return_time_step_data(
            model=model,
            time_step_data=TimeStepperStatusFailure(
                reason=f"Max retries ({self.max_attempts}) exhausted; stopping.",
                time=previous_time,
                attempts=attempts_data,
            ),
            time_data=trial_time_data,
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

    def _log_and_return_time_step_data(
        self,
        model: pp.PorePyModel,
        time_step_data: TimeStepperStatus,
        time_data: SimulationTimeData,
    ) -> TimeStepperStatus:
        # It is important that we update statistics before calling
        # after_time_step_failure, because the latter writes it.
        assert isinstance(model.nonlinear_solver_statistics, pp.TimeStatistics)
        model.nonlinear_solver_statistics.log_time_information(
            time_index=time_data.time_index_successful,
            time=time_data.time,
            dt=time_data.dt,
            final_time_reached=time_data.final_time_reached(),
        )
        # Log time step status for statistics.
        model.nonlinear_solver_statistics.log_simulation_status(time_step_data)
        if time_step_data.is_failure() or isinstance(
            time_step_data, TimeStepperStatusInProgress
        ):
            model.after_time_step_failure()
        elif time_step_data.is_success():
            model.after_time_step_convergence()
        else:
            raise ValueError(time_step_data)
        return time_step_data
