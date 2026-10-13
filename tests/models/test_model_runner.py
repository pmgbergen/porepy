from unittest.mock import MagicMock

import numpy as np
import pytest

import porepy as pp


def test_failed_nonlinear_solve_dynamic_time_step():
    """Test that a failed nonlinear solve resets iterates to the last known state.

    When the linear solver returns NaN (simulating a failure), the time step
    manager should retry with a smaller time step. Before each retry, the
    iterate array must be reset to the last accepted time state, and not left at
    the NaN values from the failed attempt.

    The test also checks that the solver makes exactly as many attempts as
    allowed by `nl_max_iterations`, and that NaN values from one iteration
    do not propagate into the residual vector of the next.
    """
    STATE_VALUE = 1.0  # The value for the primary variables.
    num_times_visited_before_nonlinear_iteration = 0
    num_times_visited_solve_linear_system = 0

    class FailingModel(pp.SinglePhaseFlow):
        def initial_condition(self):
            super().initial_condition()
            # Setting some non-trivial initial condition.
            values = np.full(self.equation_system.num_dofs(), STATE_VALUE)
            self.equation_system.set_variable_values(values, iterate_index=0)

        def before_nonlinear_iteration(self) -> None:
            # The iterate array should be equal to the state array, since we never
            # proceed further than the 0-th Newton iteration.
            nonlocal num_times_visited_before_nonlinear_iteration
            num_times_visited_before_nonlinear_iteration += 1

            state = self.equation_system.get_variable_values(time_step_index=0)
            iterate = self.equation_system.get_variable_values(iterate_index=0)
            assert np.all(state == STATE_VALUE)
            assert np.all(iterate == STATE_VALUE)
            return super().before_nonlinear_iteration()

    class MockLinearSolver(pp.solvers.LinearSolverBase):
        def solve_linear_system(
            self, linear_system: pp.solvers.LinearSystem
        ) -> tuple[np.ndarray, pp.solvers.LinearSolverStatus]:
            nonlocal num_times_visited_solve_linear_system
            num_times_visited_solve_linear_system += 1

            # Nans from the previous iteration must not propagate here.
            rhs = linear_system.rhs
            assert not np.any(np.isnan(rhs))
            # The linear solver failed and returned an array of nans.
            return np.full_like(rhs, np.nan), pp.solvers.LinearSolverStatusFailure(
                reason="Mock linear solver failure"
            )

    model_params = {
        "time_manager": pp.TimeManager(
            schedule=[0, 1],
            dt_init=0.1,
            dt_min_max=(0.05, 0.1),
        )
    }
    model = FailingModel(params=model_params)
    runner_params = {
        "nl_max_iterations": 2,  # Only 2 Newton iterations
    }
    model_runner = pp.ModelRunner(
        model,
        params=runner_params,
        nonlinear_solver=pp.solvers.NewtonSolver(
            params=runner_params, linear_solver=MockLinearSolver()
        ),
    )
    with pytest.raises(RuntimeError):
        model_runner.run()

    assert num_times_visited_solve_linear_system == 2, "Should do exactly 2 attempts."
    assert num_times_visited_before_nonlinear_iteration == 2, (
        "Should do exactly 2 attempts."
    )


def test_time_data_seeded_from_time_stepper_before_prepare_simulation():
    """Test that model.time_data reflects the real schedule from the time_stepper
    already during prepare_simulation(), not just after the first time step.

    ModelRunner.__init__ must resolve the passed-in time_stepper and seed
    model.time_data from its scheduler *before* calling prepare_simulation(). Otherwise,
    anything invoked during prepare_simulation() that depends on self.time_data.schedule
    (e.g. time-dependent boundary conditions defined per schedule point) would
    incorrectly see the SolutionStrategy.__init__ placeholder schedule [0.0, 1.0]
    instead of the real one.

    """
    schedule = [0, 1, 2, 3]
    prepare_simulation_called = False

    class RecordingModel(pp.SinglePhaseFlow):
        def prepare_simulation(self) -> None:
            assert np.all(self.time_manager.schedule.get_array() == schedule)
            assert self.time_manager.time == schedule[0]

            nonlocal prepare_simulation_called
            prepare_simulation_called = True

    model = RecordingModel(
        {
            "times_to_export": [],
            "time_manager": pp.TimeManager(schedule=schedule, dt_init=1),
        }
    )
    _ = pp.ModelRunner(model)
    assert prepare_simulation_called


@pytest.mark.parametrize("num_steps", [1, 3])
@pytest.mark.parametrize(
    "stop_status",
    [pp.ModelRunnerStatusSuccess(), pp.ModelRunnerStatusFailure(reason="Mock stop")],
)
def test_early_stop(num_steps, stop_status):
    """The runner stops after the time step at which an early stop criterion returns a
    status, although the final time is not reached. A success status is returned by the
    runner, a failure status is raised.

    """

    class StopAfter(pp.EarlyStopCriterion):
        def __init__(self):
            self.calls = 0

        def simulation_should_stop(self, model):
            self.calls += 1
            return stop_status if self.calls == num_steps else None

    # MagicMock stands in for the model and the time stepper, so that the test does not
    # need a real model. Attributes the runner reads are set explicitly.
    model = MagicMock()
    model.time_manager.final_time_reached.return_value = False
    model.time_manager.dt = 1.0
    time_stepper = MagicMock()
    time_stepper.perform_time_step.return_value = (
        pp.time_stepper.TimeStepperStatusSuccess(time=0, attempts=[])
    )
    runner = pp.ModelRunner(
        model, time_stepper=time_stepper, early_stop_criteria=[StopAfter()]
    )

    if stop_status.is_failure():
        with pytest.raises(RuntimeError):
            runner.run()
    else:
        assert runner.run() is stop_status
    assert time_stepper.perform_time_step.call_count == num_steps
