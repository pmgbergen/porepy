""" Tests of functionality of :class:`~porepy.viz.solver_statistics.SolverStatistics`."""

import porepy as pp
from porepy.applications.test_utils.models import Poromechanics


def test_solver_statistic_attributes():
    """Runs default Poromechanics simulation and tests availability of solver statistics."""
    params = {}
    model = Poromechanics()
    pp.run_time_dependent_model(model, params)

    # Unit tests
    assert hasattr(model, "nonlinear_solver_statistics")
    assert isinstance(model.nonlinear_solver_statistics, pp.SolverStatistics)
    assert hasattr(model.nonlinear_solver_statistics, "num_iteration")
    assert hasattr(model.nonlinear_solver_statistics, "nonlinear_increment_norms")
    assert hasattr(model.nonlinear_solver_statistics, "residual_norms")
    assert hasattr(model.nonlinear_solver_statistics, "path")
    assert model.nonlinear_solver_statistics.path is None


def test_solver_statistics_save():
    """Check whether solver statistics are saved to file."""
    path = "solver_statistics.json"
    params = {"solver_statistics_file_name": path}
    model = Poromechanics(params)
    pp.run_time_dependent_model(model, params)
    # Check whether file was saved
    assert model.nonlinear_solver_statistics.path.exists()
    # Clean up
    model.nonlinear_solver_statistics.path.unlink()
