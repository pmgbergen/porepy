"""Module implements the TimeStepper class, responsible for making a single simulation
time step.

"""

from __future__ import annotations

import logging
from typing import Self

import porepy as pp
from porepy.numerics import solvers
from porepy.time_stepper.scheduler import TimeSchedulerBase
from porepy.time_stepper.time_step_constraint import CannotRecomputeTimeStep
from porepy.time_stepper.time_step_status import (
    TimeStepperAttemptData,
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
        scheduler: Class that adjusts dt to match the schedule and constraints.
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
        """Class that adjusts dt to match the schedule and constraints."""

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
        accepted_time = time_manager.time
        accepted_index = time_manager.time_index
        attempts_data: list[TimeStepperAttemptData] = []

        def rollback_time() -> None:
            time_manager.time = accepted_time
            time_manager.time_index = accepted_index

        for attempt in range(self.max_attempts):
            attempted_dt = time_manager.dt

            # Enter trial state.
            time_manager.time = accepted_time + attempted_dt
            time_manager.time_index = accepted_index + 1

            # Logging time step start.
            log_message = (
                f"Time step #{time_manager.time_index}: dt={time_manager.dt:.2e}, time="
                f"{accepted_time:.2e} of {time_manager.schedule[-1]:.2e}"
            )
            if attempt > 0:
                log_message += f", attempt={attempt + 1} / {self.max_attempts}"
            logger.info(log_message)

            # Execute trial time step.
            model.before_time_step()
            nonlinear_status = solver.solve(model)
            success = nonlinear_status.is_converged()

            attempts_data.append(
                TimeStepperAttemptData(
                    dt=attempted_dt,
                    nonlinear_solve_status=nonlinear_status,
                )
            )

            # It is important that we update statistics before calling
            # after_time_step_failure, because the latter writes it. It is also
            # important that we log it before rolling back dt for unsuccessful attempts,
            # because the expected format is to report time step failure at the "end" of
            # the unsuccessful time step interval.
            _log_trial_time_information(model)

            if not success:
                # Scheduler must compute the retry from the last accepted time.
                rollback_time()

                if attempt == self.max_attempts - 1:
                    return _log_and_return_time_step_data(
                        model,
                        TimeStepperStatusFailure(
                            reason=(
                                f"Max attempts ({self.max_attempts}) exhausted; "
                                "stopping."
                            ),
                            time=accepted_time,
                            attempts=attempts_data,
                        ),
                    )

            try:
                # New time step size based on trial results.
                next_dt = self.scheduler.compute_next_time_step(
                    time_manager=time_manager,
                    success=success,
                    context={
                        "model": model,
                        "nonlinear_solver_status": nonlinear_status,
                    },
                )
            except CannotRecomputeTimeStep as exc:
                # Necessary after a successful trial, harmless after done twice.
                rollback_time()

                return _log_and_return_time_step_data(
                    model,
                    TimeStepperStatusFailure(
                        reason=str(exc.args[0]),
                        time=accepted_time,
                        attempts=list(attempts_data),
                    ),
                )

            time_manager.dt = next_dt

            if success:
                return _log_and_return_time_step_data(
                    model,
                    TimeStepperStatusSuccess(
                        time=accepted_time,
                        attempts=list(attempts_data),
                    ),
                )
            else:
                _ = _log_and_return_time_step_data(
                    model,
                    TimeStepperStatusContinueIterating(
                        attempts=list(attempts_data),
                    ),
                )

        raise AssertionError("Time-step attempt loop terminated unexpectedly.")


def _log_trial_time_information(model: pp.PorePyModel):
    # If for some reason model.nonlinear_solver_statistics is not a time-dependent
    # statistics, do nothing (should never happen).
    if isinstance(model.nonlinear_solver_statistics, pp.TimeStatistics):
        model.nonlinear_solver_statistics.log_time_information(
            time_index=model.time_manager.time_index,
            time=model.time_manager.time,
            dt=model.time_manager.dt,
            final_time_reached=model.time_manager.final_time_reached(),
        )


def _log_and_return_time_step_data(
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
