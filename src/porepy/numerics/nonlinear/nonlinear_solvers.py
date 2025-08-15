"""
Nonlinear solvers to be used with model classes.
Implemented classes
    NewtonSolver
"""

import logging

from typing import Tuple
import numpy as np

from porepy.utils.ui_and_logging import DummyProgressBar, progressbar_class
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from porepy.models.convergence_check import (
    ConvergenceStatus,
    SingleObjectiveConvergenceCriterion,
    NanConvergenceCriterion,
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
            "nl_convergence_criterion": SingleObjectiveConvergenceCriterion,
        }
        default_options.update(params)
        self.params = default_options

        # Set a convergence criterion.
        self.convergence_criterion = self.params.get("nl_convergence_criterion")()

    def set_solver_progressbar(self) -> None:
        use_progress_bar: bool = self.params.get("progressbars", False)
        if use_progress_bar and progressbar_class is DummyProgressBar:
            logger.warning(
                "Progress bars are requested, but `tqdm` is not installed. The solver"
                + " will run without progress bars."
            )

        # Allow the position of the progress bar to be flexible, depending on whether
        # this is called inside a time loop, a time loop and an additional propagation
        # loop or inside a stationary problem (default).
        progress_bar_position: int = self.params.get("nl_progressbar_position", 0)

        # Check if the user wants a progress bar. Initialize an instance of the
        # progressbar_class, which is either :class:`~tqdm.trange` or
        # :class:`~DummyProgressbar` in case `tqdm` is not installed.
        if use_progress_bar:
            # Length is the maximal number of Newton iterations.
            self.solver_progressbar = progressbar_class(  # type: ignore
                range(int(self.params["max_iterations"])),
                desc="Newton loop",
                position=progress_bar_position,
                leave=False,
                dynamic_ncols=True,
            )
        # Otherwise, use a dummy progress bar.
        else:
            self.solver_progressbar = DummyProgressBar()

    def solve(self, model) -> ConvergenceStatus:
        """Solve the nonlinear problem using the Newton-Raphson method.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            ConvergenceStatus: The convergence status of the nonlinear solver.

        """
        # Prepare nonlinear loop.
        model.before_nonlinear_loop()

        # Redirect all loggers to not interfere with the progressbar.
        with logging_redirect_tqdm([logging.root]):
            # Set a solver progress bar.
            self.set_solver_progressbar()

            # Newton loop.
            while True:
                # Perform a single Newton iteration.
                model.before_nonlinear_iteration()
                nonlinear_increment = self.iteration(model)
                model.after_nonlinear_iteration(nonlinear_increment)
                residual = model.equation_system.assemble(evaluate_jacobian=False)

                # Monitor convergence.
                convergence_status, nonlinear_increment_norm, residual_norm = (
                    self.check_convergence(
                        model,
                        nonlinear_increment,
                        residual,
                    )
                )

                # Logging and progress bar update.
                model.nonlinear_solver_statistics.log_error(
                    nonlinear_increment_norm, residual_norm
                )
                logger.info(
                    "Newton iteration number "
                    + f"{model.nonlinear_solver_statistics.num_iteration}"
                    + f" of {self.params['max_iterations']}"
                )
                logger.info(
                    f"Nonlinear increment norm: {nonlinear_increment_norm:.2e}, "
                    f"Nonlinear residual norm: {residual_norm:.2e}"
                )
                self.solver_progressbar.update(n=1)
                self.solver_progressbar.set_postfix_str(
                    f"""Increment {nonlinear_increment_norm:.2e} """
                    f"""Residual {residual_norm:.2e}"""
                )

                # Exit the Newton loop.
                if (
                    convergence_status.is_converged()
                    or convergence_status.is_diverged()
                    or model.nonlinear_solver_statistics.num_iteration
                    >= self.params["max_iterations"]
                ):
                    break

        # Close the progress bar.
        self.solver_progressbar.close()

        # React to convergence status.
        if convergence_status.is_converged():
            model.after_nonlinear_convergence()
        elif convergence_status.is_diverged():
            model.after_nonlinear_failure()
        else:
            raise ValueError

        return convergence_status

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

    def check_convergence(
        self,
        model,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
    ) -> Tuple[ConvergenceStatus, float, float]:
        """Implements a convergence check, to be called by a non-linear solver.

        Parameters:
            model: The model instance specifying the problem to be solved.
            nonlinear_increment: Newly obtained solution increment vector
            residual: Residual vector of non-linear system, evaluated at the newly
                obtained solution vector. Potentially None, if not needed.

        Returns:
            ConvergenceStatus: The convergence status of the non-linear iteration.
            float: Norm of the nonlinear increment.
            float: Norm of the residual.

        """
        # If the model is linear, we do not need to check convergence.
        if not model._is_nonlinear_problem():
            convergence_criterion = NanConvergenceCriterion()
            convergence_status = convergence_criterion.check(
                nonlinear_increment, residual
            )
            norm_value = np.nan if convergence_status.is_diverged() else 0.0
            return convergence_status, norm_value, norm_value

        # Compute norms of the nonlinear increment and residual.
        nonlinear_increment_norm = model.compute_variable_norm(nonlinear_increment)
        residual_norm = model.compute_residual_norm(residual)

        # Check convergence using the convergence criterion.
        convergence_status = self.convergence_criterion.check(
            nonlinear_increment_norm,
            residual_norm,
            self.params,
        )

        return convergence_status, nonlinear_increment_norm, residual_norm
