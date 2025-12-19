"""Tests of functionality of :class:`~porepy.viz.solver_statistics.SolverStatistics`."""

import porepy as pp
from porepy.applications.test_utils.models import Poromechanics
from porepy.viz.solver_statistics import NonlinearSolverStatistics


def test_solver_statistic_attributes():
    """Runs default Poromechanics simulation and tests availability of solver
    statistics."""
    model = Poromechanics()
    pp.run_time_dependent_model(model)

    # Unit tests
    assert hasattr(model, "nonlinear_solver_statistics")
    assert isinstance(model.nonlinear_solver_statistics, NonlinearSolverStatistics)
    # Basic attributes of pp.SolverStatistics
    assert hasattr(model.nonlinear_solver_statistics, "counter")
    assert hasattr(model.nonlinear_solver_statistics, "path")
    assert hasattr(model.nonlinear_solver_statistics, "num_cells")
    assert hasattr(model.nonlinear_solver_statistics, "simulation_status_history")
    assert hasattr(model.nonlinear_solver_statistics, "custom_data")
    # Specific attributes of pp.NonlinearSolverStatistics
    assert hasattr(model.nonlinear_solver_statistics, "num_iteration")
    assert hasattr(model.nonlinear_solver_statistics, "num_iteration_history")
    assert hasattr(model.nonlinear_solver_statistics, "convergence_status")
    assert hasattr(model.nonlinear_solver_statistics, "convergence_info")

    assert model.nonlinear_solver_statistics.path is None


def test_solver_statistics_save():
    """Check whether solver statistics are saved to file."""
    path = "solver_statistics.json"
    params = {"solver_statistics_file_name": path}
    model = Poromechanics(params)
    pp.run_time_dependent_model(model)
    # Check whether file was saved
    assert model.nonlinear_solver_statistics.path.exists()
    # Clean up
    model.nonlinear_solver_statistics.path.unlink()
