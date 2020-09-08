""" This module contains functions to run stationary and time-dependent models.

"""
import logging

import porepy as pp

logger = logging.getLogger(__name__)


def run_stationary_model(setup, params):
    setup.prepare_simulation()

    nl_solver = pp.NewtonSolver(params)

    nl_solver.solve(setup)

    setup.after_simulation()


def run_time_dependent_model(setup, params):
    """
    Time loop for the model classes.

    Args:
        setup: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate solver for documentation.
        params: Parameters related to the solution proceedure. # Why not just set these
            as e.g. setup.solution_parameters.
    """
    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        setup.prepare_simulation()

    # Prepare for the time loop
    t_end = setup.end_time
    k = 0
    if setup._is_nonlinear_problem():
        solver = pp.NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)
    while setup.time < t_end:
        setup.time += setup.time_step
        k += 1
        logger.info(
            "\nTime step {} at time {:.1e} of {:.1e} with time step {:.1e}".format(
                k, setup.time, t_end, setup.time_step
            )
        )
        solver.solve(setup)

    setup.after_simulation()
