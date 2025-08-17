"""This module contains functions to run stationary and time-dependent models."""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

import porepy as pp
from porepy.utils.ui_and_logging import DummyProgressBar, progressbar_class
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from porepy.models.convergence_check import ConvergenceStatus

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


def run_time_dependent_model(model, params: Optional[dict] = None) -> None:
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
        params: Parameters related to the solution procedure.

    """
    params = params or {}
    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    if params.get("progressbars", False) and progressbar_class is DummyProgressBar:
        logger.warning(
            "Progress bars are requested, but `tqdm` is not installed. The time loop"
            + " will run without progress bars."
        )
    # When multiple nested ``tqdm`` bars are used, their position needs to be specified
    # such that they are displayed in the correct order. The orders are increasing, i.e.
    # 0 specifies the lowest level, 1 the next-lowest etc.
    # When the ``NewtonSolver`` is called inside ``run_time_dependent_model``, the
    # ``nl_progress_bar_position`` parameter with the updated position of the progress
    # bar for the ``NewtonSolver`` is passed.
    params.update({"nl_progress_bar_position": 1})

    # Assign a solver
    solver = _choose_solver(model, params)

    # Define a function that does all the work during one time step, except
    # for everything ``tqdm`` related.
    def time_step() -> ConvergenceStatus:
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()
        logger.info(
            f"\nTime step {model.time_manager.time_index} at time"
            + f" {model.time_manager.time:.1e}"
            + f" of {model.time_manager.time_final:.1e}"
            + f" with time step {model.time_manager.dt:.1e}"
        )
        # Return convergence status s.t. the time loop can determine whether the time
        # step succeeded or failed.
        return solver.solve(model)

    # Redirect the root logger, s.t. no logger interferes with with the
    # progressbars.
    with logging_redirect_tqdm([logging.root]):
        initial_time_step: float = model.time_manager.dt

        # Check if the user wants a progress bar. Initialize an instance of the
        # progressbar_class, which is either :class:`~tqdm.trange` or
        # :class:`~DummyProgressbar` in case `tqdm` is not installed.
        if params.get("progressbars", False):
            # Create a time bar. The length is estimated as the time_steps predetermined
            # by the schedule and initial time step size.
            # NOTE: If, e.g., adaptive time stepping results in more time steps, the
            # time bar will increase with partial steps corresponding to the ratio of
            # the modified time step size to the initial time step size.
            expected_time_steps: int = int(
                np.round(
                    (model.time_manager.schedule[-1] - model.time_manager.schedule[0])
                    / initial_time_step
                )
            )
            time_progressbar = progressbar_class(
                range(expected_time_steps),
                desc="time loop",
                position=0,
                dynamic_ncols=True,
            )
        # Otherwise, use a dummy progress bar.
        else:
            time_progressbar = DummyProgressBar()

        # Time loop.
        while not model.time_manager.final_time_reached():
            time_progressbar.set_description_str(
                f"Time step {model.time_manager.time_index + 1}"
            )
            status: ConvergenceStatus = time_step()
            # Update progressbar length.
            if status.is_converged():
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

    if params.get("progressbars", False) and progressbar_class is DummyProgressBar:
        logger.warning(
            "Progress bars are requested, but `tqdm` is not installed. The iterative"
            + " loop will run without progress bars."
        )
    # Change the position of the solver progress bar to 1, as position 0 is
    # occupied by the time progress bar.
    # This needs to be adapted, once a progress bar for the iterations is
    # introduced.
    params.update({"nl_progress_bar_position": 1})

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

    # Redirect all loggers to not interfere with the progressbar.
    with logging_redirect_tqdm([logging.root]):
        initial_time_step: float = model.time_manager.dt

        # Check if the user wants a progress bar. Initialize an instance of the
        # progressbar_class, which is either :class:`~tqdm.trange` or
        # :class:`~DummyProgressbar` in case `tqdm` is not installed.
        if params.get("progressbars", False):
            # Create a time bar. The length is estimated as the time_steps predetermined
            # by the schedule and initial time step size.
            # NOTE: If e.g. adaptive time stepping results in more time steps, the time
            # bar will increase with partial steps corresponding to the ratio of the
            # modified time step size to the initial time step size.
            expected_time_steps: int = int(
                np.round(
                    (model.time_manager.schedule[-1] - model.time_manager.schedule[0])
                    / initial_time_step
                )
            )
            time_progressbar = progressbar_class(
                range(expected_time_steps),
                desc="time loop",
                position=0,
                dynamic_ncols=True,
            )
        # Otherwise, use a dummy progress bar.
        else:
            time_progressbar = DummyProgressBar()

        # Time loop.
        while not model.time_manager.final_time_reached():
            time_progressbar.set_description_str(
                f"Time step {model.time_manager.time_index + 1}"
            )
            time_step()
            # Update progressbar length. Currently, there is no convergence check
            # returned by :meth:`time_step`. Failed time steps will cause the progress
            # bar to overestimate progress and exceed the maximal time.
            time_progressbar.update(n=model._time_step / initial_time_step)

    model.after_simulation()


def _choose_solver(model, params: dict) -> Union[pp.LinearSolver, pp.NewtonSolver]:
    """Choose between linear and non-linear solver.

    Parameters:
        model: Model class containing all information on material parameters, variables,
            discretization and geometry. Various methods such as those relating to
            solving the system, see the appropriate solver for documentation.
        params: Parameters related to the solution procedure.

    """
    if "nonlinear_solver" in params:
        solver = params["nonlinear_solver"](params)
    elif model._is_nonlinear_problem():
        solver = pp.NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)
    return solver
