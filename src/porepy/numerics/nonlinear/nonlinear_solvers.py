"""
Nonlinear solvers to be used with model classes.
Implemented classes
    NewtonSolver
"""

import logging
import traceback

import numpy as np

from porepy.utils.ui_and_logging import DummyProgressBar, progressbar_class
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)

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
            "nl_divergence_tol": np.inf,
            "flag_failure_as_diverged": False,
        }
        default_options.update(params)
        self.params = default_options

        self.progress_bar: bool = params.get("progressbars", False)
        if self.progress_bar and progressbar_class is DummyProgressBar:
            logger.warning(
                "Progress bars are requested, but `tqdm` is not installed. The solver"
                + " will run without progress bars."
            )
        # Allow the position of the progress bar to be flexible, depending on whether
        # this is called inside a time loop, a time loop and an additional propagation
        # loop or inside a stationary problem (default).
        self.progress_bar_position: int = params.get("_nl_progress_bar_position", 0)

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

        # Define a function that runs everything inside one Newton iteration.
        def newton_step() -> None:
            # Bind to variables in the outer function
            nonlocal nonlinear_increment
            nonlocal reference_residual
            nonlocal is_converged
            nonlocal is_diverged

            logger.info(
                "Newton iteration number "
                + f"{model.nonlinear_solver_statistics.num_iteration}"
                + f" of {self.params['max_iterations']}"
            )
            solver_progressbar.set_description_str(
                "Newton iteration number "
                + f"{model.nonlinear_solver_statistics.num_iteration + 1} of"
                + f" {self.params['max_iterations']}"
            )

            try:
                model.before_nonlinear_iteration()
                nonlinear_increment = self.iteration(model)
                model.after_nonlinear_iteration(nonlinear_increment)

                if (
                    self.params["nl_convergence_tol_res"] is not np.inf
                    or self.params["nl_divergence_tol"] is not np.inf
                ):
                    # Note: The residual is extracted after the solution has been
                    # updated by the after_nonlinear_iteration() method. This is
                    # required if the residual is used to check convergence or
                    # divergence, i.e., the tolerance of one of them is not np.inf.
                    residual = model.equation_system.assemble(evaluate_jacobian=False)
                else:
                    residual = None

                is_converged, is_diverged = model.check_convergence(
                    nonlinear_increment, residual, reference_residual, self.params
                )
            except Exception as err:
                if self.params["flag_failure_as_diverged"]:
                    logger.warning(
                        "\nNewton step failure:\n%s\nFlagging as diverged.\n"
                        % (traceback.format_exc())
                    )
                    is_converged = False
                    is_diverged = True
                else:
                    raise err

        # Redirect all loggers to not interfere with the progressbar.
        with logging_redirect_tqdm([logging.root]):
            # Check if the user wants a progress bar. Initialize an instance of the
            # progressbar_class, which is either :class:`~tqdm.trange` or
            # :class:`~DummyProgressbar` in case `tqdm` is not installed.
            if self.progress_bar:
                # Length is the maximal number of Newton iterations.
                solver_progressbar = progressbar_class(  # type: ignore
                    self.params["max_iterations"],
                    desc="Newton loop",
                    position=self.progress_bar_position,
                    leave=False,
                    dynamic_ncols=True,
                )
            # Otherwise, use a dummy progress bar.
            else:
                solver_progressbar = DummyProgressBar()

            while (
                model.nonlinear_solver_statistics.num_iteration
                <= self.params["max_iterations"]
                and not is_converged
            ):
                newton_step()

                # Do not update the progress bar if Newton diverged. If it diverged
                # during the first iteration,
                # :attr:`~model.nonlinear_solver_statistics.nonlinear_increment_norms`
                # will be empty and the following code will raise an error.
                if not is_diverged and len(model.nonlinear_solver_statistics.nonlinear_increment_norms) > 0:
                    solver_progressbar.update(n=1)
                    # Ignore the long line; fixing it would require an extra variable.
                    solver_progressbar.set_postfix_str(
                        f"Increment {model.nonlinear_solver_statistics.nonlinear_increment_norms[-1]:.2e}"  # noqa: E501
                    )

                if is_diverged:
                    # Handle nonlinear divergence outside the loop.
                    break
                elif is_converged:
                    solver_progressbar.close()
                    model.after_nonlinear_convergence()
                    break

        if not is_converged:
            # If Newton fails, the progressbar bar needs to be
            # manually closed. See https://stackoverflow.com/a/73175351.
            solver_progressbar.close()
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
