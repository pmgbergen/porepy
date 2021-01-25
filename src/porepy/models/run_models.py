""" This module contains functions to run stationary and time-dependent models.

"""
import logging

import porepy as pp

logger = logging.getLogger(__name__)
module_sections = ["models", "numerics"]


@pp.time_logger(sections=module_sections)
def run_stationary_model(model, params):
    model.prepare_simulation()

    nl_solver = pp.NewtonSolver(params)

    nl_solver.solve(model)

    model.after_simulation()


@pp.time_logger(sections=module_sections)
def run_time_dependent_model(model, params):
    """
    Time loop for the model classes.

    Args:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate solver for documentation.
        params: Parameters related to the solution proceedure. # Why not just set these
            as e.g. model.solution_parameters.
    """
    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    # Prepare for the time loop
    t_end = model.end_time
    model.time_index = 0
    if model._is_nonlinear_problem():
        solver = pp.NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)
    while model.time < t_end:
        model.time += model.time_step
        model.time_index += 1
        logger.info(
            "\nTime step {} at time {:.1e} of {:.1e} with time step {:.1e}".format(
                model.time_index, model.time, t_end, model.time_step
            )
        )
        solver.solve(model)

    model.after_simulation()


@pp.time_logger(sections=module_sections)
def _run_iterative_model(model, params):
    """Intended use is for multi-step models with iterative couplings.

    Only known instance so far is the combination of fracture deformation
    and propagation.
    """
    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    # Prepare for the time loop
    t_end = model.end_time
    model.time_index = 0
    if model._is_nonlinear_problem():
        solver = pp.NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)
    while model.time < t_end:
        model.propagation_index = 0
        model.time += model.time_step
        model.time_index += 1
        model.before_propagation_loop()
        logger.info(
            "\nTime step {} at time {:.1e} of {:.1e} with time step {:.1e}".format(
                model.time_index, model.time, t_end, model.time_step
            )
        )
        while model.keep_propagating():
            model.propagation_index += 1

            solver.solve(model)
        model.after_propagation_loop()
    model.after_simulation()
