from typing import Any

import numpy as np

import porepy as pp
from porepy.models.fluid_mass_balance import SinglePhaseFlow


class NonlinearSinglePhaseFlow(SinglePhaseFlow):
    """Model class which forces a set number of nonlinear iterations to be run for
    testing."""

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


def test_nonlinear_iteration_count():
    """Test for checking if the nonlinear iterations are counted as expected.

    A pre set value of expected iterations is set, and the test checks that the
    iteration count matches the pre set value after convergence is obtained.

    """
    model = NonlinearSinglePhaseFlow({"times_to_export": []})
    model.expected_number_of_iterations = 3
    pp.run_time_dependent_model(model, {})

    assert (
        model.nonlinear_solver_statistics.num_iteration
        == model.expected_number_of_iterations
    )
