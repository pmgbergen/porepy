"""
Test executable examples.

"""

# from porepy.examples.flow_benchmark_2d_case_1 import execution
import porepy.examples.flow_benchmark_2d_case_1 as flow_benchmark_2d_case_1
import porepy as pp


def test_flow_benchmark_2d_case_1_main() -> None:
    """Test that the flow benchmark 2D case 1 example runs as main."""

    models = flow_benchmark_2d_case_1.execution()
    for model in models:
        assert model.solver_info["nl_convergence"] == "converged"



