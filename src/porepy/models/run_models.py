""" This module contains functions to run stationary and time-dependent models."""

import logging
from typing import Union

import porepy as pp

logger = logging.getLogger(__name__)


def run_stationary_model(model, params: dict) -> None:
    """
    Run a stationary model.

    Args:
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


def run_time_dependent_model(model, params) -> None:
    """
    Run a time dependent model.

    Args:
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
    while model.tsc.time < model.tsc.time_final:
        model.tsc.increase_time()
        model.tsc.increase_time_index()
        logger.info(
            "\nTime step {} at time {:.1e} of {:.1e} with time step {:.1e}".format(
                model.tsc.time_index, model.tsc.time, model.tsc.time_final, model.tsc.dt
            )
        )
        solver.solve(model)
        model.tsc.next_time_step()

    model.after_simulation()


def _run_iterative_model(model, params: dict) -> None:
    """
    Run an iterative model.

    The intended use is for multi-step models with iterative couplings. Only known instance
    so far is the combination of fracture deformation and propagation.

    Args:
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
    while model.tsc.time < model.tsc.time_final:
        model.propagation_index = 0
        model.tsc.increase_time()
        model.tsc.increase_time_index()
        model.before_propagation_loop()
        logger.info(
            "\nTime step {} at time {:.1e} of {:.1e} with time step {:.1e}".format(
                model.tsc.time_index, model.tsc.time, model.tsc.time_final, model.tsc.dt
            )
        )
        while model.keep_propagating():
            model.propagation_index += 1
            solver.solve(model)
        model.after_propagation_loop()

    model.after_simulation()
