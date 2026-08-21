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

    TODO: Written based on discovery pattern. Consider less specific test.
    """
    observed_schedule_sizes = []

    class RecordingModel(pp.SinglePhaseFlow):
        def update_all_boundary_conditions(self) -> None:
            observed_schedule_sizes.append(self.time_data.schedule.size)
            super().update_all_boundary_conditions()

    schedule = [0, 1, 2, 3]
    time_stepper = pp.time_stepper.TimeStepper(
        scheduler=pp.time_stepper.assemble_default_time_scheduler(
            schedule=schedule,
            dt_init=1,
            constant_dt=True,
        )
    )
    model = RecordingModel({"times_to_export": []})
    pp.ModelRunner(model, time_stepper=time_stepper)

    assert observed_schedule_sizes, (
        "update_all_boundary_conditions should be invoked during prepare_simulation."
    )
    assert observed_schedule_sizes[0] == len(schedule)
    assert np.array_equal(model.time_data.schedule, schedule)
