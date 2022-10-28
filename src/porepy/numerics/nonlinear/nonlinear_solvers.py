"""
Nonlinear solvers to be used with model classes.
Implemented classes
    NewtonSolver
"""
from __future__ import annotations

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

    def solve(self, model) -> tuple[float, bool, int]:
        """
        Solve non-linear problem using standard Newton.

        Args:
            model: Properly initialized mixed-dimensional model.

        Returns:
            L2-norm of the difference between the current and previous solutions.
            True if the solution converged. False otherwise.
            Number of non-linear iterations.
        """

        def repeat_loop(iters: int, has_converged: bool, has_diverged: bool) -> bool:
            """Whether the Newton loop has to be repeated.

            The loop is repeated if ALL three criteria are met:
                (C1) Number of iterations is equal or less than maximum number of iterations.
                (C2) The solution did not converge, e.g., error still larger than target
                    tolerance.
                (C3) The solution did not diverge, e.g., solution is not Nan.
            Otherwise, the loop is broken. In this case, the linearization step may be
            reattempted or the simulation terminated, subject to the solution algorithm,
            particularly the `TimeManager`.

            Args:
                iters: number of non-linear iterations.
                has_converged: whether the solution has converged.
                has_diverged: whether the solution has diverged.

            Returns:
                True if the while loop has to be repeated. False otherwise.

            """
            repeat = (
                iters <= self.params["max_iterations"]  # (C1)
                and not has_converged  # (C2)
                and not has_diverged  # (C3)
            )
            return repeat

        model.before_newton_loop()

        iteration_counter = 1
        is_converged = False
        is_diverged = False

        prev_sol = model.dof_manager.assemble_variable(from_iterate=False)
        init_sol = prev_sol
        errors = []
        error_norm = 1.0

        while repeat_loop(iteration_counter, is_converged, is_diverged):
            logger.info(
                "Newton iteration number {} of {}".format(
                    iteration_counter, self.params["max_iterations"]
                )
            )

            # Re-discretize the nonlinear term
            model.before_newton_iteration()

            sol = self.nonlinear_iteration(model)

            model.after_newton_iteration(sol)

            # Convergence check
            error_norm, is_converged, is_diverged = model.check_convergence(
                sol, prev_sol, init_sol, self.params
            )
            prev_sol = sol
            errors.append(error_norm)

            iteration_counter += 1

        iteration_counter -= 1  # fix off-set

        # Post-Newton steps
        if is_diverged or not is_converged:
            model.after_newton_failure(sol, errors, iteration_counter)
        else:
            model.after_newton_convergence(sol, errors, iteration_counter)

        return error_norm, is_converged, iteration_counter

    def nonlinear_iteration(self, model) -> np.ndarray:
        """A single nonlinear iteration.

        Right now, this is a single line, however, we keep it as a separate function
        to prepare for possible future introduction of more advanced schemes.

        Returns:
            Solution array for the iteration step.

        """

        # Assemble and solve
        model.assemble_linear_system()
        sol = model.solve_linear_system()

        return sol
