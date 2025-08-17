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
    AbsoluteConvergenceCriterion,
    NanConvergenceCriterion,
)

# Module-wide logger
logger = logging.getLogger(__name__)


class NewtonSolver:
    def __init__(self, params=None) -> None:
        if params is None:
            params = {}

        default_options = {
            "nl_max_iterations": 10,
            "nl_convergence_tol_inc": 1e-10,
            "nl_convergence_tol_res": np.inf,
            "nl_divergence_tol_res": np.inf,
            "nl_convergence_criterion": AbsoluteConvergenceCriterion,
        }
        default_options.update(params)
        self.params = default_options
        """Dictionary of parameters for the nonlinear solver."""

        self.convergence_criterion = self.params.get("nl_convergence_criterion")()
        """Convergence criterion to be used in the nonlinear solver."""

        self.init_solver_progressbar()

    def init_solver_progressbar(self) -> None:
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
                range(int(self.params["nl_max_iterations"])),
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
                self.logging(model, nonlinear_increment_norm, residual_norm)

                # Exit the Newton loop.
                if (
                    convergence_status.is_converged()
                    or convergence_status.is_diverged()
                    or model.nonlinear_solver_statistics.num_iteration
                    >= self.params["nl_max_iterations"]
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

        The convergence check implicitly assumes relative errors and passes reference
        norms for both increments and residuals to the managing `convergence_criterion`.
        The explicit goal is to build the current state as reference for the increment,
        and the initial residual as reference for the residual. The responsibility for
        managing its reference values is given to the convergence criterion.

        Parameters:
            model: The model instance specifying the problem to be solved, knowing
                of its metrics for measuring states and residuals.
            nonlinear_increment: Newly obtained solution increment vector.
            residual: Residual vector of non-linear system, evaluated at the newly
                obtained solution vector.

        Returns:
            ConvergenceStatus: The convergence status of the non-linear iteration.
            float: Norm of the nonlinear increment.
            float: Norm of the residual.

        """
        # If the model is linear, we do not need to check convergence.
        # TODO: How important is this? Newton is only called for nonlinear problems, by default.
        if not model._is_nonlinear_problem():
            convergence_criterion = NanConvergenceCriterion()
            convergence_status = convergence_criterion.check(
                nonlinear_increment, residual, self.params
            )
            norm_value = np.nan if convergence_status.is_diverged() else 0.0
            return convergence_status, norm_value, norm_value

        # Start with trivial nan-check.
        if np.isnan(nonlinear_increment).any() or np.isnan(residual).any():
            return ConvergenceStatus.DIVERGED, np.nan, np.nan

        # Compute norms of the nonlinear increment and residual. Potentially a scalar,
        # but also dictionaries are possible if equation-based norms are used, e.g.
        nonlinear_increment_norm = model.variable_norm(nonlinear_increment)
        residual_norm = model.residual_norm(residual)

        # Also fetch the norm of the iterate, which is used as a reference value.
        iterate = model.equation_system.get_variable_values(iterate_index=0)
        iterate_norm = model.variable_norm(iterate)

        # Each iteration requires a new reference value for the convergence criterion.
        if model.nonlinear_solver_statistics.num_iteration == 1:
            self.convergence_criterion.reset_reference_values()

        # Cache reference values for convergence checks.
        self.convergence_criterion.set_reference_value(
            reference_nonlinear_increment_norm=iterate_norm,
            reference_residual_norm=residual_norm,
        )

        # Check convergence using the convergence criterion.
        convergence_status, scalar_nonlinear_increment_norm, scalar_residual_norm = (
            self.convergence_criterion.check(
                nonlinear_increment_norm,
                residual_norm,
                self.params,
            )
        )

        return (
            convergence_status,
            scalar_nonlinear_increment_norm,
            scalar_residual_norm,
        )

    def logging(
        self, model, nonlinear_increment_norm: float, residual_norm: float
    ) -> None:
        """Log the current state of the nonlinear solver.

        This includes updating the solver statistics and printing the current
        iteration number, nonlinear increment norm, and residual norm, as well
        as updating the progress bar.

        Parameters:
            model: The model instance specifying the problem to be solved.
            nonlinear_increment_norm: The norm of the nonlinear increment.
            residual_norm: The norm of the residual.

        """
        model.nonlinear_solver_statistics.log_error(
            nonlinear_increment_norm, residual_norm
        )
        logger.info(
            "Newton iteration number "
            + f"{model.nonlinear_solver_statistics.num_iteration}"
            + f" of {self.params['nl_max_iterations']}"
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
