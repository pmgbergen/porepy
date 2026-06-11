"""
Test executable examples.

The tests verify that the examples can be well executed with
a successful simulation status.

"""

import porepy.examples.flow_benchmark_2d_case_1 as flow_benchmark_2d_case_1
import porepy as pp


def test_flow_benchmark_2d_case_1_main(monkeypatch) -> None:
    """Test that the flow benchmark 2D case 1 example runs as main.

    Plotting is disabled and the test verifies that the simulation status
    is successful for both the conductive and blocking fracture cases.

    """
    # Disable plotting to avoid creating figures.
    monkeypatch.setattr(
        flow_benchmark_2d_case_1.pp, "plot_grid", lambda *args, **kwargs: None
    )
    models = flow_benchmark_2d_case_1.run_example()

    # Both the conductive and blocking fracture cases should be executed.
    assert len(models) == 2

    # Verify that the simulation status is successful for cases.
    for model in models:
        stats = model.nonlinear_solver_statistics
        assert stats.simulation_status.is_successful()
