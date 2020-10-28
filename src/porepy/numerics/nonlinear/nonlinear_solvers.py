#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 11:21:54 2019

@author: eke001
"""
import logging

import numpy as np

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

    def solve(self, setup):
        setup.before_newton_loop()

        iteration_counter = 0

        is_converged = False

        prev_sol = setup.get_state_vector()
        init_sol = prev_sol
        errors = []
        error_norm = 1

        while iteration_counter <= self.params["max_iterations"] and not is_converged:
            logger.info(
                "Newton iteration number {} of {}".format(
                    iteration_counter, self.params["max_iterations"]
                )
            )

            # Re-discretize the nonlinear term
            setup.before_newton_iteration()

            lin_tol = np.minimum(1e-4, error_norm)
            sol = self.iteration(setup, lin_tol)

            setup.after_newton_iteration(sol)

            error_norm, is_converged, is_diverged = setup.check_convergence(
                sol, prev_sol, init_sol, self.params
            )
            prev_sol = sol
            errors.append(error_norm)

            if is_diverged:
                setup.after_newton_failure(sol, errors, iteration_counter)
            elif is_converged:
                setup.after_newton_convergence(sol, errors, iteration_counter)

            iteration_counter += 1

        if not is_converged:
            setup.after_newton_failure(sol, errors, iteration_counter)

        return error_norm, is_converged, iteration_counter

    def iteration(self, setup, lin_tol):
        """A single Newton iteration.

        Right now, this is a single line, however, we keep it as a separate function
        to prepare for possible future introduction of more advanced schemes.
        """

        # Assemble and solve
        sol = setup.assemble_and_solve_linear_system(lin_tol)

        return sol
