import porepy as pp
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.numerics.nonlinear.convergence_check import ConvergenceTolerance


def test_nonlinear_iteration_count():
    """Test for checking if the nonlinear iterations are counted as expected.

    A pre set value of expected iterations is set, and the test checks that the
    iteration count matches the pre set value after convergence is obtained.

    """
    model = SinglePhaseFlow({"times_to_export": []})
    model.expected_number_of_iterations = 3
    pp.run_time_dependent_model(
        model,
        {
            "nl_convergence_tol": ConvergenceTolerance(
                tol_increment=0, tol_residual=0, max_iterations=3
            )
        },
    )

    assert (
        model.nonlinear_solver_statistics.num_iteration
        == model.expected_number_of_iterations
    )
    assert (
        len(model.nonlinear_solver_statistics.nonlinear_increment_norms)
        == model.expected_number_of_iterations
    )
    assert (
        len(model.nonlinear_solver_statistics.residual_norms)
        == model.expected_number_of_iterations
    )
