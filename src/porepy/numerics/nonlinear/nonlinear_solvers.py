"""
Nonlinear solvers to be used with model classes.
Implemented classes
    NewtonSolver
"""
import logging

import numpy as np

import pdb

# Module-wide logger
logger = logging.getLogger(__name__)


class NewtonSolver:
    def __init__(self, params=None):
        if params is None:
            params = {}

        default_options = {
            "max_iterations": 10,
            "nl_convergence_tol": 1e-10,
            "nl_divergence_tol": 1e5,
        }
        default_options.update(params)
        self.params = default_options

    def solve(self, model):
        model.before_nonlinear_loop()

        iteration_counter = 0

        is_converged = False
        prev_sol = model.equation_system.get_variable_values(time_step_index=0)

        init_sol = prev_sol
        errors = []
        error_norm = 1

        while iteration_counter <= self.params["max_iterations"] and not is_converged:
            # while True:
            print("\n NEWTON iteration_counter = ", iteration_counter)

            # pdb.set_trace()

            logger.info(
                "Newton iteration number {} of {}".format(
                    iteration_counter, self.params["max_iterations"]
                )
            )

            # Re-discretize the nonlinear term
            model.before_nonlinear_iteration()

            # print('\n ------------before iteration')
            sol = self.iteration(model)
            # print('\n ------------right after iteration')

            model.after_nonlinear_iteration(sol)

            error_norm, is_converged, is_diverged = model.check_convergence(
                sol, prev_sol, init_sol, self.params
            )
            prev_sol = sol
            errors.append(error_norm)

            if is_diverged:
                model.after_nonlinear_failure(sol, errors, iteration_counter)
            elif is_converged:
                model.after_nonlinear_convergence(sol, errors, iteration_counter)

            iteration_counter += 1
            print("error_norm = ", error_norm)  ###

        if not is_converged:
            model.after_nonlinear_failure(sol, errors, iteration_counter)

        return error_norm, is_converged, iteration_counter

    def iteration(self, model) -> np.ndarray:
        """A single nonlinear iteration.

        Right now, this is a single line, however, we keep it as a separate function
        to prepare for possible future introduction of more advanced schemes.
        """

        # Assemble and solve
        model.assemble_linear_system()
        sol = model.solve_linear_system()

        return sol
