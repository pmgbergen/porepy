"""
Nonlinear solvers to be used with model classes.
Implemented classes
    NewtonSolver
"""
import logging

import numpy as np

# Avoid some mpy trouble.
from tqdm.autonotebook import trange  # type: ignore

from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)

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

        self.progress_bar: bool = params.get("progressbars", True)
        # Allow the position of the progress bar to be flexible, depending on whether
        # this is called inside a time loop, a time loop and an additional propagation
        # loop or inside a stationary problem (default).
        self.progress_bar_position: int = params.get("progress_bar_position", 0)

    def solve(self, model):
        model.before_nonlinear_loop()

        iteration_counter = 0

        is_converged = False
        prev_sol = model.equation_system.get_variable_values(time_step_index=0)

        init_sol = prev_sol
        errors = []
        error_norm = 1

        # Progressbars turned on:
        if self.progress_bar:
            # Redirect the root logger, s.t. no logger interferes with with the
            # progressbars.
            with logging_redirect_tqdm([logging.root]):
                # Initialize a progress bar. Length is the number of maximal Newton
                # iterations.
                solver_progressbar = trange(
                    self.params["max_iterations"],
                    desc="Newton loop",
                    position=self.progress_bar_position,
                    leave=False,
                )

                while (
                    iteration_counter <= self.params["max_iterations"]
                    and not is_converged
                ):
                    # Logging.
                    logger.info(
                        f"Newton iteration number {iteration_counter}"
                        + f" of {self.params['max_iterations']}"
                    )

                    # Re-discretize the nonlinear term
                    model.before_nonlinear_iteration()

                    sol = self.iteration(model)

                    model.after_nonlinear_iteration(sol)

                    error_norm, is_converged, is_diverged = model.check_convergence(
                        sol, prev_sol, init_sol, self.params
                    )
                    prev_sol = sol
                    errors.append(error_norm)

                    # Update progressbar.
                    solver_progressbar.update(n=1)
                    solver_progressbar.set_description_str(
                        f"Newton iteration number {iteration_counter} of \
                            {self.params['max_iterations']}"
                    )
                    solver_progressbar.set_postfix_str(f"Error {error_norm}")

                    if is_diverged:
                        # If the process finishes early, the tqdm bar needs to be
                        # manually closed. See https://stackoverflow.com/a/73175351.
                        solver_progressbar.close()
                        model.after_nonlinear_failure(sol, errors, iteration_counter)
                        break
                    elif is_converged:
                        solver_progressbar.close()
                        model.after_nonlinear_convergence(
                            sol, errors, iteration_counter
                        )
                        break

                    iteration_counter += 1

        # Progressbars turned off:
        else:
            while (
                iteration_counter <= self.params["max_iterations"] and not is_converged
            ):
                # Logging.
                logger.info(
                    f"Newton iteration number {iteration_counter}"
                    + f" of {self.params['max_iterations']}"
                )

                # Re-discretize the nonlinear term
                model.before_nonlinear_iteration()

                sol = self.iteration(model)

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
