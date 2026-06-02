import numpy as np

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
    num_times_visited_assemble_linear_system = 0
    num_times_visited_solve_linear_system = 0

    class FailingModel(pp.SinglePhaseFlow):
        def initial_condition(self):
            super().initial_condition()
            # Setting some non-trivial initial condition.
            values = np.full(self.equation_system.num_dofs(), STATE_VALUE)
            self.equation_system.set_variable_values(values, iterate_index=0)

        def assemble_linear_system(self) -> None:
            # The iterate array should be equal to the state array, since we never
            # proceed further than the 0-th Newton iteration.
            nonlocal num_times_visited_assemble_linear_system
            num_times_visited_assemble_linear_system += 1

            state = self.equation_system.get_variable_values(time_step_index=0)
            iterate = self.equation_system.get_variable_values(iterate_index=0)
            assert np.all(state == STATE_VALUE)
            assert np.all(iterate == STATE_VALUE)
            return super().assemble_linear_system()

        def solve_linear_system(self) -> np.ndarray:
            nonlocal num_times_visited_solve_linear_system
            num_times_visited_solve_linear_system += 1

            _, b = self.linear_system
            # Nans from the previous iteration must not propagate here.
            assert not np.any(np.isnan(b))
            # The linear solver failed and returned an array of nans.
            return np.full_like(b, np.nan)

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
    model_runner = pp.ModelRunner(model, params=runner_params)
    model_runner.run()

    assert num_times_visited_solve_linear_system == 2, "Should do exactly 2 attempts."
    assert num_times_visited_assemble_linear_system == 2, (
        "Should do exactly 2 attempts."
    )
