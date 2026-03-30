import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.nonlinear.convergence_check import SimulationStatus


class MockEquationSystem:
    def __init__(self) -> None:
        self.iteration_index = -1

    def assemble(self, **kwargs) -> np.ndarray:
        """Return a mock residual array."""
        self.iteration_index += 2
        return np.array([10 ** (-self.iteration_index)])

    def get_variable_values(self, **kwargs) -> np.ndarray:
        return np.array([1.0])


class MockMDG:
    def subdomains(self) -> list[pp.Grid]:
        return [pp.CartGrid(np.array([1, 1]))]


class MockModel:
    """Mock model to test the progressbars interface in :mod:`~porepy.models.run_models`
    and :mod:`~porepy.numerics.nonlinear.nonlinear_solvers`.

    Does nothing but expose dummy hooks to
    :func:`~porepy.models.run_models.run_time_dependent_model` and
    :func:`~porepy.numerics.nonlinear.nonlinear_solvers.NewtonSolver`.

    Does nothing but expose :attr:`time_manager` and :attr:`nonlinear_solver_statistics`
    to relevant progressbar methods in
    :func:`~porepy.models.run_models.run_time_dependent_model` and
    :func:`~porepy.numerics.nonlinear.nonlinear_solvers.NewtonSolver` to ensure the
    progressbars are updated correctly as the time and nonlinear loop progress,
    respectively.

    """

    def __init__(self):
        self.mdg = MockMDG()
        self.equation_system = MockEquationSystem()
        self.nonlinear_solver_statistics = pp.NonlinearSolverAndTimeStatistics()
        self.time_manager = pp.TimeManager([0, 2], dt_init=1, constant_dt=True)

    def _is_time_dependent(self) -> bool:
        return True

    def _is_nonlinear_problem(self) -> bool:
        return True

    def prepare_simulation(self) -> None:
        pass

    def after_simulation(self) -> None:
        pass

    def before_nonlinear_loop(self) -> None:
        self.nonlinear_solver_statistics.increase_index()

    def before_nonlinear_iteration(self) -> None:
        pass

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        self.nonlinear_solver_statistics.num_iterations += 1

    def after_nonlinear_convergence(self) -> None:
        pass

    def after_nonlinear_failure(self) -> SimulationStatus:
        if self.time_manager.is_constant:
            return SimulationStatus.STOPPED
        else:
            return SimulationStatus.FAILED

    def assemble_linear_system(self) -> None:
        pass

    def solve_linear_system(self) -> np.ndarray:
        return np.array([1.0])


def test_mock_model():
    model = MockModel()
    pp.run_time_dependent_model(model)
