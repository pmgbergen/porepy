import porepy as pp
import numpy as np
from typing import Any

from porepy.models.fluid_mass_balance import SinglePhaseFlow


class NonlinearSinglePhaseFlow(SinglePhaseFlow):
    """Class setup which forces a set number of nonlinear iterations to be run for
    testing."""

    def _is_nonlinear_problem(self):
        """Method for stating that the problem is nonlinear."""
        return True

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Method for checking convergence.

        This method is only used for testing. It returns not converged if the iteration
        count is smaller than a pre set value.

        """
        diverged = False
        if (
            self.nonlinear_solver_statistics.num_iteration
            < self.expected_number_of_iterations
        ):
            converged = False
            return converged, diverged
        converged = True
        return converged, diverged

    def after_nonlinear_failure(self) -> None:
        """Method which is called if the non-linear solver fails to converge."""
        self.save_data_time_step()
        prev_solution = self.equation_system.get_variable_values(time_step_index=0)
        self.equation_system.set_variable_values(prev_solution, iterate_index=0)


def test_nonlinear_iteration_count():
    """Test for checking if the nonlinear iterations are counted as expected.

    A pre set value of expected iterations is set, and the test checks that the
    iteration count matches the pre set value after convergence is obtained.

    """
    params = {"max_iterations": 96}
    model = NonlinearSinglePhaseFlow()
    model.expected_number_of_iterations = 13
    pp.run_time_dependent_model(model, params)

    assert (
        model.nonlinear_solver_statistics.num_iteration
        == model.expected_number_of_iterations
    )
