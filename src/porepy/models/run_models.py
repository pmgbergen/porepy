""" This module contains functions to run stationary and time-dependent models."""
from __future__ import annotations

import logging
from typing import Union
import numpy as np

import porepy as pp

import pdb

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

    - TODO: run_time_dependent_model is not versatile enough. improve it. but i'm lazy.

    """
    print("\n\n\nrun_time_dependent_model has been mod")

    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    # Assign a solver
    solver: Union[pp.LinearSolver, pp.NewtonSolver]
    if model._is_nonlinear_problem():
        solver = pp.NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)

    # Time loop
    global_cumulative_iteration_counter = 0
    global_cumulative_flips = np.zeros(model.number_upwind_dirs)

    while model.time_manager.time < model.time_manager.time_final:
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()
        logger.info(
            "\nTime step {} at attempt time {:.1e} of {:.1e} with attempt time step {:.1e}".format(
                model.time_manager.time_index,
                model.time_manager.time,
                model.time_manager.time_final,
                model.time_manager.dt,
            )
        )

        print(
            "\n\n\nTime step ",
            model.time_manager.time_index,
            " at attempt time ",
            model.time_manager.time,
            " of ",
            model.time_manager.time_final,
            " with attempt time step ",
            model.time_manager.dt,
        )

        is_converged = False

        time_chops = 0
        cumulative_iteration_counter = 0
        cumulative_flips = np.zeros(model.number_upwind_dirs)

        while not is_converged:
            error_norm, is_converged, iteration_counter, flips = solver.solve(model)

            if not is_converged:
                model.time_manager.decrease_time()
                time_chops += 1

            previous_dt, dt = model.time_manager.compute_time_step(
                is_converged, iteration_counter
            )
            print("model.time_manager.dt = ", model.time_manager.dt)
            print(
                "(computed solution time) model.time_manager.time = ",
                model.time_manager.time,
            )

            if not is_converged:
                model.time_manager.increase_time()

            cumulative_iteration_counter += iteration_counter
            cumulative_flips += flips

        global_cumulative_iteration_counter += cumulative_iteration_counter
        global_cumulative_flips += cumulative_flips

        model.write_newton_info(
            model.time_manager.time,
            previous_dt,
            time_chops,
            cumulative_iteration_counter,
            global_cumulative_iteration_counter,
            iteration_counter,
        )

        model.save_flip_flop(
            model.time_manager.time, cumulative_flips, global_cumulative_flips
        )

        model.eb_after_timestep()

    model.after_simulation()


def _run_iterative_model(model, params: dict) -> None:
    """Run an iterative model.

    The intended use is for multi-step models with iterative couplings. Only known instance
    so far is the combination of fracture deformation and propagation.

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

    # Assign a solver
    solver: Union[pp.LinearSolver, pp.NewtonSolver]
    if model._is_nonlinear_problem():
        solver = pp.NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)

    # Time loop
    while model.time_manager.time < model.time_manager.time_final:
        model.propagation_index = 0
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()
        model.before_propagation_loop()
        logger.info(
            "\nTime step {} at time {:.1e} of {:.1e} with time step {:.1e}".format(
                model.time_manager.time_index,
                model.time_manager.time,
                model.time_manager.time_final,
                model.time_manager.dt,
            )
        )
        while model.keep_propagating():
            model.propagation_index += 1
            solver.solve(model)
        model.after_propagation_loop()

    model.after_simulation()
