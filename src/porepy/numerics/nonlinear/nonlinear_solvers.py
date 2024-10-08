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

    def solve(self, model) -> bool:
        """Solve the nonlinear problem.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            A 2-tuple containing:

            bool:
                True if the solution is converged.

        """
        model.before_nonlinear_loop()

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
            nonlocal is_converged
            nonlocal is_diverged

            # Logging.
            logger.info(
                "Newton iteration number "
                + f"{model.nonlinear_solver_statistics.num_iteration}"
                + f" of {self.params['max_iterations']}"
            )

            # Re-discretize the nonlinear term
            model.before_nonlinear_iteration()

            nonlinear_increment = self.iteration(model)

            model.after_nonlinear_iteration(nonlinear_increment)
            # Note: The residual is extracted after the solution has been updated by the
            # after_nonlinear_iteration() method.
            residual = model.equation_system.assemble(evaluate_jacobian=False)

            is_converged, is_diverged = model.check_convergence(
                nonlinear_increment, residual, reference_residual, self.params
            )

        # Progressbars turned off or tqdm not installed:
        if not self.progress_bar or not _IS_TQDM_AVAILABLE:
            while (
                model.nonlinear_solver_statistics.num_iteration
                <= self.params["max_iterations"]
                and not is_converged
            ):
                newton_step()

                if is_diverged:
                    # The nonlinear solver failure is handled after the loop.
                    break
                elif is_converged:
                    model.after_nonlinear_convergence()
                    break

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
                    model.nonlinear_solver_statistics.num_iteration
                    <= self.params["max_iterations"]
                    and not is_converged
                ):
                    solver_progressbar.set_description_str(
                        "Newton iteration number "
                        + f"{model.nonlinear_solver_statistics.num_iteration + 1} of \t"
                        + f"{self.params['max_iterations']}"
                    )
                    newton_step()

                    solver_progressbar.update(n=1)
                    # Ignore line being too long, because we would need an additional
                    # variable to fix this.
                    solver_progressbar.set_postfix_str(
                        f"Increment {model.nonlinear_solver_statistics.nonlinear_increment_norms[-1]:.2e}"  # noqa: E501
                    )

                    if is_diverged:
                        # If the process finishes early, the tqdm bar needs to be
                        # manually closed. See https://stackoverflow.com/a/73175351.
                        solver_progressbar.close()
                        # The nonlinear solver failure is handled after the loop.
                        break
                    elif is_converged:
                        solver_progressbar.close()
                        model.after_nonlinear_convergence()
                        break

        if not is_converged:
            model.after_nonlinear_failure()

        return is_converged

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
