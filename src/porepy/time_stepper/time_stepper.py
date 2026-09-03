"""Module implements the TimeStepper class, responsible for making a single simulation
time step.

"""

from __future__ import annotations

import logging
from typing import Self

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
    TimeStepperStatusContinueIterating,
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
    - Execute trials (delegating nonlinear solves).
    - Adjust dt.

    Workflow:
    1. For each retry (up to max_retries):
        a. Execute trial with current dt;
        b. If success: update solution values, adapt dt for next step, return;
        c. If failure: reduce dt, loop, revert trial time.
    2. If all retries exhausted: return.

    The constant dt case is supported internally by setting max_retries = 1.

    Parameters:
        scheduler: Class that adjust dt to match the schedule and constraints.
        max_attempts: Limit of attempts to make a single time step. Set it to 1 for no
            retries.

    """

    @classmethod
    def with_time_manager(
        cls, time_manager: pp.TimeManager, max_attempts: int = 10
    ) -> Self:
        """Convenience initializer. Initializes scheduler based on the `time_manager`.

        Parameters:
            time_manager: Simulation's time data structure.
            max_attempts: Limit of attempts to make a single time step.

        """
        scheduler: TimeSchedulerBase
        if time_manager.advanced_schedule is not None:
            scheduler = pp.time_stepper.TimeScheduler(
                time_manager=time_manager,
                schedule=time_manager.advanced_schedule,
                t_snap=time_manager.atol,
            )
        else:
            scheduler = pp.time_stepper.assemble_default_time_scheduler(
                time_manager=time_manager
            )
        return cls(scheduler=scheduler, max_attempts=max_attempts)

    def __init__(self, scheduler: TimeSchedulerBase, max_attempts: int = 10) -> None:
        self.scheduler = scheduler
        """Class that adjust dt to match the schedule and constraints."""

        assert max_attempts > 0, "max_attempts must be greater than 0."
        self.max_attempts = max_attempts
        """Maximum number of retry attempts."""

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
        time_manager = model.time_manager
        previous_time = time_manager.time
        attempts_data: list[TimeStepperAttemptData] = []

        for attempt in range(self.max_attempts):
            # Update time manager for new trial.
            time_manager.time += time_manager.dt
            time_manager.time_index += 1

            # Logging time step start.
            log_message = (
                f"Time step #{time_manager.time_index}: dt={time_manager.dt:.2e}, time="
                f"{previous_time:.2e} of {time_manager.schedule[-1]:.2e}"
            )
            if attempt > 0:
                log_message += f", attempt={attempt + 1} / {self.max_attempts}"
            logger.info(log_message)

            # Execute trial time step.
            model.before_time_step()
            nonlinear_solver_status = solver.solve(model)
            success = nonlinear_solver_status.is_converged()

            attempts_data.append(
                TimeStepperAttemptData(
                    dt=time_manager.dt,
                    nonlinear_solve_status=nonlinear_solver_status,
                )
            )

            # It is important that we update statistics before calling
            # after_time_step_failure, because the latter writes it. It is also
            # important that we log it before rolling back dt for unsuccessful attempts,
            # because the expected format is to report time step failure at the "end" of
            # the unsuccessful time step interval.
            # If for some reason model.nonlinear_solver_statistics is not a
            # time-dependent statistics, do nothing (should never happen).
            if isinstance(model.nonlinear_solver_statistics, pp.TimeStatistics):
                model.nonlinear_solver_statistics.log_time_information(
                    time_index=model.time_manager.time_index,
                    time=model.time_manager.time,
                    dt=model.time_manager.dt,
                    final_time_reached=model.time_manager.final_time_reached(),
                )

            if not success:
                # Roll back if the time step attempt failed. This is needed in case if
                # the simulation stops and the time_data is not reassigned above in the
                # next loop iteration.
                time_manager.time = previous_time
                time_manager.time_index -= 1

            try:
                # New time step size based on trial results.
                time_manager.dt = self.scheduler.compute_next_time_step(
                    time_manager=time_manager,
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
                )

            if success:
                return self._log_and_return_time_step_data(
                    model=model,
                    time_step_data=TimeStepperStatusSuccess(
                        time=time_manager.time, attempts=attempts_data
                    ),
                )
            else:
                _ = self._log_and_return_time_step_data(
                    model=model,
                    time_step_data=TimeStepperStatusContinueIterating(
                        attempts=attempts_data
                    ),
                )

        # We reached max_attepts.
        return self._log_and_return_time_step_data(
            model=model,
            time_step_data=TimeStepperStatusFailure(
                reason=f"Max retries ({self.max_attempts}) exhausted; stopping.",
                time=previous_time,
                attempts=attempts_data,
            ),
        )

    def _log_and_return_time_step_data(
        self,
        model: pp.PorePyModel,
        time_step_data: TimeStepperStatus,
    ) -> TimeStepperStatus:
        """Update model's statistics with `time_step_data` and return it."""
        model.nonlinear_solver_statistics.log_simulation_status(time_step_data)
        if time_step_data.is_failure() or isinstance(
            time_step_data, TimeStepperStatusContinueIterating
        ):
            model.after_time_step_failure()
        elif time_step_data.is_success():
            model.after_time_step_convergence()
        else:
            raise ValueError(time_step_data)
        return time_step_data
