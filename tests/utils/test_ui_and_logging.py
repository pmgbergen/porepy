import logging

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.nonlinear.convergence_check import SimulationStatus

logger = logging.getLogger()


class MockEquationSystem:
    def assemble(self, **kwargs) -> np.ndarray:
        # Artificially satisfy residual norm convergence criterion.
        return np.array([1e-11])

    def get_variable_values(self, **kwargs) -> np.ndarray:
        return np.array([0.0])


class MockMDG:
    def subdomains(self) -> list[pp.Grid]:
        return [pp.CartGrid(np.array([1, 1]))]


class MockModel:
    """Mock model to test the progressbars interface in :mod:`~porepy.models.run_models`
    and :mod:`~porepy.numerics.nonlinear.nonlinear_solvers`.

    Does nothing but expose dummy hooks to
    :func:`~porepy.models.run_models.run_time_dependent_model` and
    :func:`~porepy.numerics.nonlinear.nonlinear_solvers.NewtonSolver`.

    Nonlinear convergence is fully controlled by the size of the nonlinear increment as
    the mock residual value is always below tolerance, cf.
    :meth:`~MockEquationSystem.assemble`.

    Does nothing but expose :attr:`time_manager` and :attr:`nonlinear_solver_statistics`
    to relevant progressbar methods in
    :func:`~porepy.models.run_models.run_time_dependent_model` and
    :func:`~porepy.numerics.nonlinear.nonlinear_solvers.NewtonSolver` to ensure the
    progressbars are updated correctly as the time and nonlinear loop progress,
    respectively.

    """

    def __init__(self, num_time_steps: int, num_nl_iterations: int):
        self.mdg = MockMDG()
        self.equation_system = MockEquationSystem()
        self.nonlinear_solver_statistics = pp.NonlinearSolverAndTimeStatistics()
        self.time_manager = pp.TimeManager(
            [0, num_time_steps], dt_init=1, constant_dt=True
        )
        self.num_nl_iterations = num_nl_iterations

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
        # Artificially fail/satisfy Newton update norm convergence criterion.
        # Implementation NOTE: nonlinear_solver_statistics.num_iterations lags behind
        # one iteration at this point in the solver step.
        if self.nonlinear_solver_statistics.num_iterations < self.num_nl_iterations - 1:
            return np.array([1.0])
        else:
            return np.array([1e-11])


@pytest.fixture
def num_time_steps() -> int:
    return 2


@pytest.fixture
def num_nl_iterations() -> int:
    return 3


@pytest.fixture
def setup_model(num_time_steps: int, num_nl_iterations: int) -> MockModel:
    return MockModel(num_time_steps, num_nl_iterations)


@pytest.mark.parametrize("progressbars", [True, False])
@pytest.mark.parametrize("logging_level", [logging.DEBUG, logging.CRITICAL])
def test_line_count(
    setup_model: MockModel,
    progressbars: bool,
    logging_level: bool,
    num_time_steps: int,
    num_nl_iterations: int,
    capsys,
) -> None:
    # Fix progressbars and logging params.
    params = {"progressbars": progressbars}
    logging.basicConfig(level=logging_level)

    pp.run_time_dependent_model(setup_model, params)

    captured_stderr = capsys.readouterr().err
    captured_stderr_carr_returns = captured_stderr.split("\r")

    # NOTE: Some explanation on the inner workings of tqdm: Progressbars are updated by
    # moving the cursor to the start of the line with "\r" and overwriting the previous
    # output. For nested time/Newton progressbars, the cursor is moved to the upper
    # level and down again with "\x1bA[" + "\n" before its moved to the beginning of the
    # inner loop.

    # IMPLEMENTATION NOTE: `captured_stderr` is hard to read for humans, as tqdm
    # combines "\r" (carriage return), "\n" (new line), and "\x1bA[" (move cursor up one
    # line) to overwrite (and therefore update) progressbars and navigate between nested
    # bars.
    # Instead of explicitly checking the entire logic, we separate by "\r" to get every
    # occurence where a progressbar was written or updated. Then, we check that the
    # number of lines that start with "Time loop"/"Newton loop" match the expected
    # number of time steps/Newton steps.

    num_time_progressbar_updates: int = 0
    num_newton_progressbar_updates: int = 0
    for line in captured_stderr_carr_returns:
        if line.startswith("Time loop"):
            num_time_progressbar_updates += 1
        elif line.startswith("Newton loop"):
            num_newton_progressbar_updates += 1

    # NOTE: Whenever a tqdm bar is updated multiple times in short succession and
    # without any other writes to stdout inbetween, all updates are combined in a single
    # write to stdout.
    # E.g., self.solver_progressbar.update and self.solver_progressbar.set_postfix_str
    # in NewtonSolver.logging may be simultaneously written to stdout.
    # On the other hand, time_progressbar.set_postfix_str and time_progressbar.update in
    # run_time_dependent_model are separated by a call to time_step and during the first
    # time step, both may be written independently to stdout. Afterwards,
    # time_progressbar.update from the previous time step and
    # time_progressbar.set_postfix_str from the current time step may be written
    # simultaneously.

    # This behavior seems to be not entirely deterministic, so we just check the number
    # of progressbar updates against lower bounds.

    if progressbars:
        # Progressbar updates during time loop: 1 for initialization + 1 for all but the
        # last time step.
        min_expected_time_progressbar_updates = num_time_steps
        # Progressbar updates per Newton loop: 1 for 0th step + 1 for each Newton step.
        min_expected_newton_progressbar_updates = (
            num_nl_iterations + 1
        ) * num_time_steps
    else:
        min_expected_time_progressbar_updates = 0
        min_expected_newton_progressbar_updates = 0

    assert num_time_progressbar_updates >= min_expected_time_progressbar_updates
    assert num_newton_progressbar_updates >= min_expected_newton_progressbar_updates

    # TODO Implement logging check. These are not written to stdout?
