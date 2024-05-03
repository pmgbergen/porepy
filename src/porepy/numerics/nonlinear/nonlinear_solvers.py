"""
Nonlinear solvers to be used with model classes.
Implemented classes
    NewtonSolver
"""

import logging

import numpy as np

# ``tqdm`` is not a dependency. Up to the user to install it.
try:
    # Avoid some mypy trouble.
    from tqdm.autonotebook import trange  # type: ignore

    # Only import this if needed
    from porepy.utils.ui_and_logging import (
        logging_redirect_tqdm_with_level as logging_redirect_tqdm,
    )

except ImportError:
    _IS_TQDM_AVAILABLE: bool = False
else:
    _IS_TQDM_AVAILABLE = True


# Module-wide logger
logger = logging.getLogger(__name__)


class NewtonSolver:
    def __init__(self, params=None) -> None:
        if params is None:
            params = {}

        default_options = {
            "max_iterations": 10,
            "nl_convergence_tol": 1e-10,
            "nl_convergence_tol_res": np.inf,
            "nl_divergence_tol": 1e5,
        }
        default_options.update(params)
        self.params = default_options

        self.progress_bar: bool = params.get("progressbars", False)
        # Allow the position of the progress bar to be flexible, depending on whether
        # this is called inside a time loop, a time loop and an additional propagation
        # loop or inside a stationary problem (default).
        self.progress_bar_position: int = params.get("progress_bar_position", 0)

    def solve(self, model) -> tuple[bool, int]:
        """Solve the nonlinear problem.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            A 2-tuple containing:

            bool:

                True if the solution is converged.
            int:

                Number of iterations used.

        """
        model.before_nonlinear_loop()

        iteration_counter = 0

        residual_norm = np.inf
        nonlinear_increment_norm = np.inf
        is_converged = False
        is_diverged = False
        nonlinear_increment = model.equation_system.get_variable_values(
            time_step_index=0
        )

        # Extract residual of initial guess.
        reference_residual = model.equation_system.assemble(evaluate_jacobian=False)

        # Define a function that does all the work during one Newton iteration, except
        # for everything ``tqdm`` related.
        def newton_step() -> None:
            # Bind to variables in the outer function
            nonlocal nonlinear_increment
            nonlocal reference_residual
            nonlocal residual_norm
            nonlocal nonlinear_increment_norm
            nonlocal is_converged
            nonlocal is_diverged

            # Logging.
            logger.info(
                f"Newton iteration number {iteration_counter}"
                + f" of {self.params['max_iterations']}"
            )

            # Re-discretize the nonlinear term
            model.before_nonlinear_iteration()

            nonlinear_increment = self.iteration(model)

            model.after_nonlinear_iteration(nonlinear_increment)
            # Note: The residual is extracted after the solution has been updated by the
            # after_nonlinear_iteration() method.
            residual = model.equation_system.assemble(evaluate_jacobian=False)

            residual_norm, nonlinear_increment_norm, is_converged, is_diverged = (
                model.check_convergence(
                    nonlinear_increment, residual, reference_residual, self.params
                )
            )

        # Progressbars turned off or tqdm not installed:
        if not self.progress_bar or not _IS_TQDM_AVAILABLE:
            while (
                iteration_counter <= self.params["max_iterations"] and not is_converged
            ):
                newton_step()

                if is_diverged:
                    model.after_nonlinear_failure()
                    break
                elif is_converged:
                    model.after_nonlinear_convergence()
                    break

                iteration_counter += 1

        # Progressbars turned on:
        else:
            # Redirect the root logger, s.t. no logger interferes with the progressbars.
            with logging_redirect_tqdm([logging.root]):
                # Initialize a progress bar. Length is the number of maximal Newton
                # iterations.
                solver_progressbar = trange(  # type: ignore
                    self.params["max_iterations"],
                    desc="Newton loop",
                    position=self.progress_bar_position,
                    leave=False,
                )

                while (
                    iteration_counter <= self.params["max_iterations"]
                    and not is_converged
                ):
                    solver_progressbar.set_description_str(
                        f"""Newton iteration number {iteration_counter + 1} of \t"""
                        f"""{self.params['max_iterations']}"""
                    )
                    newton_step()
                    solver_progressbar.update(n=1)
                    solver_progressbar.set_postfix_str(
                        f"Increment {nonlinear_increment_norm:.3E}"
                    )

                    if is_diverged:
                        # If the process finishes early, the tqdm bar needs to be
                        # manually closed. See https://stackoverflow.com/a/73175351.
                        solver_progressbar.close()
                        model.after_nonlinear_failure()
                        break
                    elif is_converged:
                        solver_progressbar.close()
                        model.after_nonlinear_convergence()
                        break

                    iteration_counter += 1

        if not is_converged:
            model.after_nonlinear_failure()

        return is_converged, iteration_counter

    def iteration(self, model) -> np.ndarray:
        """A single nonlinear iteration.

        Right now, this is an almost trivial function. However, we keep it as a separate
        function to prepare for possible future introduction of more advanced schemes.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            np.ndarray: Solution to linearized system, i.e. the update increment.

        """
        model.assemble_linear_system()
        nonlinear_increment = model.solve_linear_system()
        return nonlinear_increment
