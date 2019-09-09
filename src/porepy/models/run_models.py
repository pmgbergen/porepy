""" This module contains functions to run stationary and time-dependent models.

"""
import logging

import porepy as pp

logger = logging.getLogger(__name__)

def run_stationary_model(setup, params):
    setup.prepare_simulation(params)

    nl_solver = pp.NewtonSolver(params)
    
    nl_solver.solve(setup)
    
def run_time_dependent_model(setup, params):
    """
    """
    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    setup.prepare_simulation()
    
    # Prepare for the time loop
    t_end = setup.end_time
    k = 0
    
    nl_solver = pp.NewtonSolver(params)
    
    while setup.time < t_end:
        setup.time += setup.time_step
        k += 1
        logger.info(
            "\nTime step {} at time {:.1e} of {:.1e} with time step {:.1e}".format(
                k, setup.time, t_end, setup.time_step
            )
        )
        nl_solver.solve(setup)
        
    
    