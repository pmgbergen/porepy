import porepy as pp
from porepy.models.fluid_mass_balance import SinglePhaseFlow


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
            "nl_convergence_inc_atol": 0,
            "nl_convergence_res_atol": 0,
            "nl_max_iterations": 3,
        },
    )

    assert (
        model.nonlinear_solver_statistics.num_iteration
        == model.expected_number_of_iterations
    )
    assert len(model.nonlinear_solver_statistics.convergence_info) > 0
    for key in model.nonlinear_solver_statistics.convergence_info:
        assert (
            len(model.nonlinear_solver_statistics.convergence_info[key])
            == model.expected_number_of_iterations
        )
