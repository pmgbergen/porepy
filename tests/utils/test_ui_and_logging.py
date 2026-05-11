import logging
from typing import Any, Optional

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import porepy as pp

mock_logger = logging.getLogger(__name__)


class MockEquationSystem:
    equation_indexer = pp.ad.EquationIndexer(indices={})
    variable_indexer = pp.ad.VariableIndexer(indices={})

    def assemble(self, evaluate_jacobian: bool = True, **kwargs):
        if not evaluate_jacobian:
            # Artificially satisfy residual norm convergence criterion.
            return np.array([1e-11])
        return pp.solvers.LinearSystem(
            matrix=csr_matrix(np.array([[1.0]])),
            rhs=np.array([1e-11]),
            equation_indexer=pp.ad.EquationIndexer(indices={}),
            variable_indexer=pp.ad.VariableIndexer(indices={}),
        )

    def get_variable_values(self, **kwargs) -> np.ndarray:
        return np.array([0.0])


class MockMDG:
    def subdomains(self) -> list[pp.Grid]:
        return [pp.CartGrid(np.array([1, 1]))]


class MockModel:
    """Mock model to test the progressbars and logging interface in
    :mod:`~porepy.models.run_models` and
    :mod:`~porepy.numerics.nonlinear.nonlinear_solvers`.


    Exposes mock hooks to
    :func:`~porepy.models.run_models.run_time_dependent_model` and
    :func:`~porepy.numerics.nonlinear.nonlinear_solvers.NewtonSolver` and writes mock
    logging messages of all levels.

    Nonlinear convergence is fully controlled by the size of the nonlinear increment as
    the mock residual value is always below tolerance, cf.
    :meth:`~MockEquationSystem.assemble`.


    """

    def __init__(self, num_time_steps: int):
        self.mdg = MockMDG()
        self.equation_system = MockEquationSystem()
        self.nonlinear_solver_statistics = pp.NonlinearSolverAndTimeStatistics()
        self.time_manager = pp.TimeManager(
            [0, num_time_steps], dt_init=1, constant_dt=True
        )

    def _is_time_dependent(self) -> bool:
        return True

    def _is_nonlinear_problem(self) -> bool:
        return True

    def prepare_simulation(self) -> None:
        pass

    def before_time_step(self) -> None:
        pass

    def after_time_step_convergence(self) -> None:
        pass

    def after_simulation(self) -> None:
        pass

    def before_nonlinear_loop(self) -> None:
        self.nonlinear_solver_statistics.increase_index()

    def before_nonlinear_iteration(self) -> None:
        # IMPLEMENTATION NOTE Write logging messages of all levels. This way we can
        # test in test_logging_and_progressbars that
        # pp.utils.ui_and_logging.logging_redirect_tqdm_with_level correctly redirects
        # messages of all levels through tqdm.
        mock_logger.debug(f"Starting Newton step")
        mock_logger.info(f"Starting Newton step")
        mock_logger.warning(f"Starting Newton step")
        mock_logger.error(f"Starting Newton step")

    def after_nonlinear_iteration(
        self,
        nonlinear_increment: np.ndarray,
        updated_variables: Optional[list[pp.ad.Variable]] = None,
    ) -> None:
        pass

    def after_nonlinear_convergence(self) -> None:
        pass

    def after_nonlinear_failure(self):
        pass


class MockLinearSolver(pp.solvers.LinearSolverBase):
    def __init__(self, num_nl_iterations: int):
        self.num_nl_iterations = num_nl_iterations
        self.iteration_count = 0

    def solve_linear_system(
        self, linear_system: pp.solvers.LinearSystem
    ) -> tuple[np.ndarray, pp.solvers.LinearSolverStatus]:
        self.iteration_count += 1
        if self.iteration_count < self.num_nl_iterations:
            increment = np.array([1.0])
        else:
            self.iteration_count = 0
            increment = np.array([1e-11])
        return increment, pp.solvers.LinearSolverStatusSuccess(solve_time=0.0)


@pytest.fixture(scope="module")
def num_time_steps() -> int:
    return 2


@pytest.fixture(scope="module")
def num_nl_iterations() -> int:
    return 3


@pytest.fixture
def progressbars(request) -> bool:
    return request.param


@pytest.fixture
def logging_level(request) -> bool:
    return request.param


# IMPLEMENTATION NOTE It is tempting to make this fixture module-scoped to avoid
# running the model individually for each test, but because capsys and caplog are
# function-scoped, this is not possible.
@pytest.fixture
def run_model_and_save_output(
    progressbars: bool,
    logging_level: int,
    num_time_steps: int,
    num_nl_iterations: int,
    capsys,
    caplog,
) -> tuple[str, list[logging.LogRecord]]:
    # Initialize logging capture of the correct levels.
    caplog.set_level(logging_level)

    model = MockModel(num_time_steps)
    params = {"progressbars": progressbars}
    nonlinear_solver = pp.solvers.NewtonSolver(
        params=params, linear_solver=MockLinearSolver(num_nl_iterations)
    )

    pp.ModelRunner(model, params, nonlinear_solver=nonlinear_solver).run()

    captured_stderr = capsys.readouterr().err
    captured_logging_records = caplog.records

    return captured_stderr, captured_logging_records


# Run with progressbars on and off.
@pytest.mark.parametrize("progressbars", [True, False], indirect=True)
# To test that logging messages do not interfere with displaying progressbars, one
# logging level is sufficient.
@pytest.mark.parametrize("logging_level", [logging.DEBUG], indirect=True)
def test_progressbars(
    progressbars: bool,
    logging_level: int,
    num_time_steps: int,
    num_nl_iterations: int,
    run_model_and_save_output: tuple[str, list[logging.LogRecord]],
) -> None:
    """Test that nested progressbars are displayed correctly and work with logging.

    :class:`~porepy.models.model_runner.ModelRunner` and
    :class:`~porepy.numerics.nonlinear.nonlinear_solvers.NewtonSolver` can employ `tqdm`
    progressbars to display simulation and solver progress, respectively. This test
    checks that both nested progressbars are displayed and updated after each time step
    and Newton iteration, respectively.

    The test is run with varying logging levels to ensure that logging messages do
    not interfere with the displaying of the progressbars, i.e., it checks that
    :func:`~porepy.utils.ui_and_logging.logging_redirect_tqdm_with_level` works from the
    progressbars side.

    """
    captured_stderr = run_model_and_save_output[0]
    captured_stderr_carr_returns = captured_stderr.split("\r")

    # IMPLEMENTATION NOTE `captured_stderr` is hard to read for humans, as tqdm
    # combines "\r" (carriage return), "\n" (new line), and "\x1bA[" (move cursor up one
    # line) to overwrite (and therefore update) progressbars and navigate between nested
    # bars.
    # Instead of explicitly checking the entire logic, we separate by "\r" to get every
    # occurence where a progressbar was written or updated. Then, we check that the
    # number of lines that start with "Time loop"/"Newton loop" matches the expected
    # number of time steps/Newton steps.

    num_time_progressbar_updates: int = 0
    num_newton_progressbar_updates: int = 0
    for line in captured_stderr_carr_returns:
        if line.startswith("Time loop"):
            num_time_progressbar_updates += 1
        elif line.startswith("Newton loop"):
            num_newton_progressbar_updates += 1

    # NOTE tqdm bars are refreshed at most every tqdm.mininterval (default = 0.1)
    # seconds. If a progressbar is updated multiple times in short succession, the
    # updates are compressed into a single refresh, i.e., into a single write to stdout.
    # E.g., self.solver_progressbar.update and self.solver_progressbar.set_postfix_str
    # in NewtonSolver.logging may be simultaneously written to stdout.
    # On the other hand, time_progressbar.set_postfix_str and time_progressbar.update in
    # run_time_dependent_model are separated by a call to time_step and during the first
    # time step, both may be written independently to stdout. Afterwards,
    # time_progressbar.update from the previous time step and
    # time_progressbar.set_postfix_str from the current time step may be written
    # simultaneously.

    # The amount of writes to stdout during a time/Newton loop is therefore not
    # deterministic. We just check the number of progressbar updates against lower
    # bounds.

    if progressbars and progressbar_class is not DummyProgressBar:
        # Progressbar updates during time loop: 1 for initialization + 1 for all but the
        # last time step.
        min_expected_time_progressbar_updates = num_time_steps
        # Progressbar updates per Newton loop: 1 for 0th step + 1 for each Newton step.
        min_expected_newton_progressbar_updates = (
            num_nl_iterations + 1
        ) * num_time_steps
        assert num_time_progressbar_updates >= min_expected_time_progressbar_updates
        assert num_newton_progressbar_updates >= min_expected_newton_progressbar_updates
    else:
        assert num_time_progressbar_updates == 0
        assert num_newton_progressbar_updates == 0


# Run with progressbars on and off to test that ``logging_redirect_tqdm_with_level``
# works with real and placeholder progressbars.
@pytest.mark.parametrize("progressbars", [True, False], indirect=True)
# Run with different logging levels.
@pytest.mark.parametrize("logging_level", [logging.DEBUG, logging.ERROR], indirect=True)
def test_logging_redirect(
    progressbars: bool,
    logging_level: int,
    num_time_steps: int,
    num_nl_iterations: int,
    run_model_and_save_output: tuple[str, Any],
) -> None:
    """Tests :func:`~porepy.utils.ui_and_logging.logging_redirect_tqdm_with_level`.

    To avoid a visual mess when using both progressbars and logging, all logging
    messages are redirected through a contextmanager. ``tqdm``s own implementation does
    not redirect the logging level correctly, so we use
    :func:`~porepy.utils.ui_and_logging.logging_redirect_tqdm_with_level` in PorePy.
    This test checks that our implementation correctly handles logging messages with
    their respective level.

    """
    captured_logging_records = run_model_and_save_output[1]

    # Count the number of "Starting Newton step" messages originating from
    # MockModel.before_nonlinear_iteration of each level.
    num_before_newton_step_debug_records: int = 0
    num_before_newton_step_info_records: int = 0
    num_before_newton_step_warning_records: int = 0
    num_before_newton_step_error_records: int = 0

    for record in captured_logging_records:
        if record.msg == "Starting Newton step":
            if record.levelno == logging.DEBUG:
                num_before_newton_step_debug_records += 1
            elif record.levelno == logging.INFO:
                num_before_newton_step_info_records += 1
            elif record.levelno == logging.WARNING:
                num_before_newton_step_warning_records += 1
            elif record.levelno == logging.ERROR:
                num_before_newton_step_error_records += 1

    total_num_nl_iterations = num_nl_iterations * num_time_steps

    if logging_level <= logging.DEBUG:
        # If logging_level == logging.DEBUG, debug, info, and warning messages are
        # logged.
        assert num_before_newton_step_debug_records == total_num_nl_iterations
        assert num_before_newton_step_info_records == total_num_nl_iterations
        assert num_before_newton_step_warning_records == total_num_nl_iterations
    else:
        # If logging_level == logging.ERROR, only error messages are logged.
        assert num_before_newton_step_debug_records == 0
        assert num_before_newton_step_info_records == 0
        assert num_before_newton_step_warning_records == 0

    # Error messages are logged for both logging_level == logging.DEBUG and
    # logging_level == logging.ERROR.
    assert num_before_newton_step_error_records == total_num_nl_iterations
