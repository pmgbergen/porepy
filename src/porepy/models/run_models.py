""" This module contains functions to run stationary and time-dependent models."""
from __future__ import annotations

import logging
from typing import Union

# ``tqdm`` is not a dependency. Up to the user to install it.
try:
    # Avoid some mypy trouble.
    from tqdm.autonotebook import trange  # type: ignore

    # Only import this if needed
    from porepy.utils.ui_and_logging import (
        logging_redirect_tqdm_with_level as logging_redirect_tqdm,
    )

except ImportError:
    _IS_TQDM_AVAILABLE: bool = False
else:
    _IS_TQDM_AVAILABLE = True

import porepy as pp

# Module-wide logger
logger = logging.getLogger(__name__)


def run_stationary_model(model, params: dict) -> None:
    """Run a stationary model.

    Note:
        If the ``"progressbars"`` key in ``params`` is set to ``True`` (default is
        ``False``), the progress of nonlinear iterations will be shown on a progressbar.
        This requires the ``tqdm`` package to be installed. The package is not included
        in the dependencies, but can be installed with
        ```
        pip install tqdm
        ```

    Parameters:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate model for documentation.
        params: Parameters related to the solution procedure.

    """
    model.prepare_simulation()

    solver = _choose_solver(model, params)

    solver.solve(model)

    model.after_simulation()


def run_time_dependent_model(model, params: dict) -> None:
    """Run a time dependent model.

    Note:
        If the ``"progressbars"`` key in ``params`` is set to ``True`` (default is
        ``False``), the progress of time steps and nonlinear iterations will be shown on
        a progressbar. This requires the ``tqdm`` package to be installed. The package
        is not included in the dependencies, but can be installed with
        ```
        pip install tqdm
        ```

    Parameters:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate solver for documentation.
        params: Parameters related to the solution procedure. Why not just set these
            as e.g. model.solution_parameters?

    """
    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    # When multiple nested ``tqdm`` bars are used, their position needs to be specified
    # such that they are displayed in the correct order. The orders are increasing, i.e.
    # 0 specifies the lowest level, 1 the next-lowest etc.
    # When the ``NewtonSolver`` is called inside ``run_time_dependent_model``, the
    # ``progress_bar_position`` parameter with the updated position of the progress bar
    # for the ``NewtonSolver`` is passed.
    params.update({"progress_bar_position": 1})

    # Assign a solver
    solver = _choose_solver(model, params)

    # Define a function that does all the work during one time step, except
    # for everything ``tqdm`` related.
    def time_step() -> None:
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()
        logger.info(
            f"\nTime step {model.time_manager.time_index} at time"
            + f" {model.time_manager.time:.1e}"
            + f" of {model.time_manager.time_final:.1e}"
            + f" with time step {model.time_manager.dt:.1e}"
        )
        solver.solve(model)

    # Progressbars turned off or tqdm not installed:
    if not params.get("progressbars", False) or not _IS_TQDM_AVAILABLE:
        while not model.time_manager.final_time_reached():
            time_step()

    # Progressbars turned on:
    else:
        # Redirect the root logger, s.t. no logger interferes with with the
        # progressbars.
        with logging_redirect_tqdm([logging.root]):
            # Time loop
            # Create a time bar. The length is estimated as the timesteps predetermined
            # by the schedule and initial time step size.
            # NOTE: If e.g. adaptive time stepping results in more time steps, the time
            # bar will increase with partial steps corresponding to the ratio of the
            # modified time step size to the initial time step size.
            expected_timesteps: int = int(
                (model.time_manager.schedule[-1] - model.time_manager.schedule[0])
                / model.time_manager.dt
            )
            initial_time_step: float = model.time_manager.dt
            time_progressbar = trange(
                expected_timesteps,
                desc="time loop",
                position=0,
            )

            while model.time_manager.time < model.time_manager.time_final:
                time_progressbar.set_description_str(
                    f"Time step {model.time_manager.time_index} + 1"
                )
                time_step()
                # Update time progressbar length by the time step size divided by the
                # initial time step size.
                time_progressbar.update(n=model.time_manager.dt / initial_time_step)

    model.after_simulation()


def _run_iterative_model(model, params: dict) -> None:
    """Run an iterative model.

    The intended use is for multi-step models with iterative couplings. Only known
    instance so far is the combination of fracture deformation and propagation.

    Note:
        If the ``"progressbars"`` key in ``params`` is set to ``True`` (default is
        ``False``), the progress of time steps and nonlinear iterations will be shown on
        a progressbar. This requires the ``tqdm`` package to be installed. The package
        is not included in the dependencies, but can be installed with
        ```
        pip install tqdm
        ```

    Parameters:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate solver for documentation.
        params: Parameters related to the solution procedure.

    """
    # Assign parameters, variables and discretizations. Discretize time-indepedent
    # terms.
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    # Change the position of the solver progress bar to 1, as position 0 is
    # occupied by the time progress bar.
    # This needs to be adapted, once a progress bar for the iterations is
    # introduced.
    params.update({"progress_bar_position": 1})

    # Assign a solver
    solver = _choose_solver(model, params)

    # Define a function that does all the work during one time step, except
    # for everything ``tqdm`` related.
    def time_step() -> None:
        model.propagation_index = 0
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()
        logger.info(
            f"\nTime step {model.time_manager.time_index} at time"
            + f" {model.time_manager.time:.1e} of"
            + f" {model.time_manager.time_final:.1e}"
            + f" with time step {model.time_manager.dt:.1e}"
        )
        model.before_propagation_loop()
        while model.keep_propagating():
            model.propagation_index += 1
            solver.solve(model)
        model.after_propagation_loop()

    # Progressbars turned off or tqdm not installed:
    if not params.get("progressbars", False) or not _IS_TQDM_AVAILABLE:
        while model.time_manager.time < model.time_manager.time_final:
            time_step()
    # Progressbars turned on:
    else:
        # Redirect the root logger, s.t. no logger interferes with with the
        # progressbars.
        with logging_redirect_tqdm([logging.root]):
            # Time loop
            # Create a time bar. The length is estimated as the number of timesteps
            # predetermined by the schedule and initial time step size.
            # Note: If e.g., some manual time stepping results in more time steps, the
            # time bar will increase with partial steps corresponding to the ratio of
            # the modified time step size to the initial time step size.
            expected_timesteps: int = int(
                (model.time_manager.schedule[-1] - model.time_manager.schedule[0])
                / model.time_manager.dt
            )
            initial_time_step: float = model.time_manager.dt
            # Assert that the initial time step is not zero, to avoid division by zero
            # later on.
            assert initial_time_step != 0
            time_progressbar = trange(
                expected_timesteps,
                desc="time loop",
                position=0,
            )

            while model.time_manager.time < model.time_manager.time_final:
                time_progressbar.set_description_str(
                    f"Time step {model.time_manager.time_index}"
                )
                time_step()
                # Update time progressbar by the time step size divided by the initial
                # time step size.
                time_progressbar.update(n=model._time_step / initial_time_step)

    model.after_simulation()


def _choose_solver(model, params: dict) -> Union[pp.LinearSolver, pp.NewtonSolver]:
    """Choose between linear and non-linear solver.

    Parameters:
        model: Model class containing all information on material parameters, variables,
            discretization and geometry. Various methods such as those relating to solving
            the system, see the appropriate solver for documentation.
        params: Parameters related to the solution procedure.

    """
    if "nonlinear_solver" in params:
        solver = params["nonlinear_solver"](params)
    elif model._is_nonlinear_problem():
        solver = pp.NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)
    return solver
