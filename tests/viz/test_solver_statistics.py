""" Tests of functionality of :class:`~porepy.viz.solver_statistics.SolverStatistics`."""

import pytest

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
    assert hasattr(model.nonlinear_solver_statistics, "increment_errors")
    assert hasattr(model.nonlinear_solver_statistics, "residual_errors")


@pytest.mark.parametrize("path", [None, "solver_statistics.json"])
def test_solver_statistics_save(path):
    """Check whether solver statistics are saved to file."""
    params = {"solver_statistics_file_name": path}
    model = Poromechanics(params)
    pp.run_time_dependent_model(model, params)
    if path is not None:
        # Check whether file was saved
        assert model.nonlinear_solver_statistics.path.exists()
        # Clean up
        model.nonlinear_solver_statistics.path.unlink()
    else:
        assert model.nonlinear_solver_statistics.path is None
