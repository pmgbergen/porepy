""" This module contains functions to run stationary and time-dependent models."""
from __future__ import annotations

import logging
from typing import Union

# Avoid some mpy trouble.
from tqdm.autonotebook import trange  # type: ignore

import porepy as pp
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)

# Module-wide logger
logger = logging.getLogger(__name__)


def run_stationary_model(model, params: dict) -> None:
    """Run a stationary model.

    Parameters:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate model for documentation.
        params: Parameters related to the solution procedure. # Why not just set these
            as e.g. model.solution_parameters.

    """
    model.prepare_simulation()

    solver: Union[pp.LinearSolver, pp.NewtonSolver]
    if model._is_nonlinear_problem():
        solver = pp.NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)

    solver.solve(model)

    model.after_simulation()


def run_time_dependent_model(model, params: dict) -> None:
    """Run a time dependent model.

    Parameters:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate solver for documentation.
        params: Parameters related to the solution procedure. # Why not just set these
            as e.g. model.solution_parameters.

    """
    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    # Change the position of the solver progress bar to 1, as the time progress bar
    # will occupy position 0.
    params.update({"progress_bar_position": 1})

    # Assign a solver
    solver: Union[pp.LinearSolver, pp.NewtonSolver]

    if model._is_nonlinear_problem():
        solver = pp.NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)

    # Progressbars turned on:
    if params.get("progressbars", True):
        # Redirect the root logger, s.t. no logger interferes with with the
        # progressbars.
        with logging_redirect_tqdm([logging.root]):
            # Time loop
            # Create a time bar. The length is estimated as the timesteps predetermined
            # by the schedule and initial time step size.
            # NOTE: If e.g., some manual time stepping results in more time steps, the
            # time bar will increase with partial steps corresponding to the ratio of
            # the modified time step size to the initial time step size.
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
                model.time_manager.increase_time()
                model.time_manager.increase_time_index()
                time_progressbar.set_description_str(
                    f"Time step {model.time_manager.time_index}"
                    + f" at time {model.time_manager.time:.1e}"
                )
                logger.debug(
                    f"\nTime step {model.time_manager.time_index} at time"
                    + f" {model.time_manager.time:.1e}"
                    + f" of {model.time_manager.time_final:.1e}"
                    + f" with time step {model.time_manager.dt:.1e}"
                )
                solver.solve(model)
                model.time_manager.compute_time_step()
                # Update time progressbar by the time step size divided by the initial
                # time step size.
                time_progressbar.update(n=model.time_manager.dt / initial_time_step)

    # Progressbars turned off:
    else:
        while model.time_manager.time < model.time_manager.time_final:
            model.time_manager.increase_time()
            model.time_manager.increase_time_index()
            logger.debug(
                f"\nTime step {model.time_manager.time_index} at time"
                + f" {model.time_manager.time:.1e}"
                + f" of {model.time_manager.time_final:.1e}"
                + f" with time step {model.time_manager.dt:.1e}"
            )
            solver.solve(model)
            model.time_manager.compute_time_step()

    model.after_simulation()


def _run_iterative_model(model, params: dict) -> None:
    """Run an iterative model.

    The intended use is for multi-step models with iterative couplings. Only known
    instance so far is the combination of fracture deformation and propagation.

    Parameters:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate solver for documentation.
        params: Parameters related to the solution procedure. # Why not just set these
            as e.g. model.solution_parameters.

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
    solver: Union[pp.LinearSolver, pp.NewtonSolver]

    if model._is_nonlinear_problem():
        solver = pp.NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)

    # Progressbars turned on:
    if params.get("progressbars", True):
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
                model.propagation_index = 0
                model.time_manager.increase_time()
                model.time_manager.increase_time_index()
                time_progressbar.set_description_str(
                    f"Time step {model.time_manager.time_index}"
                    + f" at time {model.time_manager.time:.1e}"
                )
                logger.debug(
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
                # Update time progressbar by the time step size divided by the initial
                # time step size.
                time_progressbar.update(n=model._time_step / initial_time_step)

    # Progressbars turned off:
    else:
        while model.time_manager.time < model.time_manager.time_final:
            model.propagation_index = 0
            model.time_manager.increase_time()
            model.time_manager.increase_time_index()
            logger.debug(
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

    model.after_simulation()
