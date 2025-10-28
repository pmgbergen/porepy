"""
Nonlinear solvers to be used with model classes.
Implemented classes
    NewtonSolver
"""

import logging

import numpy as np

from porepy.numerics.nonlinear.convergence_check import (  # NanConvergenceCriterion,
    AbsoluteConvergenceCriterion,
    RelativeConvergenceCriterion,
    ConvergenceInfo,
    ConvergenceStatus,
    ConvergenceTolerance,
)
from porepy.utils.ui_and_logging import DummyProgressBar
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from porepy.utils.ui_and_logging import progressbar_class
from typing import cast

# Module-wide logger
logger = logging.getLogger(__name__)


class NewtonSolver:
    def __init__(self, params=None) -> None:
        if params is None:
            params = {}

        default_options = {
            "nl_convergence_tol": ConvergenceTolerance(
                tol_increment=1e-10, max_iterations=10
            ),
            "nl_convergence_criterion": AbsoluteConvergenceCriterion(),
        }
        default_options.update(params)
        self.params = default_options
        """Dictionary of parameters for the nonlinear solver."""

        self.tol: ConvergenceTolerance = cast(
            ConvergenceTolerance, self.params["nl_convergence_tol"]
        )
        """Convergence tolerance used in the convergence check."""

        self.convergence_criterion: RelativeConvergenceCriterion = cast(
            RelativeConvergenceCriterion, self.params.get("nl_convergence_criterion")
        )
        """Convergence criterion used in the convergence check."""

        self.init_solver_progressbar()

    def init_solver_progressbar(self) -> None:
        use_progress_bar = bool(self.params.get("progressbars", False))
        if use_progress_bar and progressbar_class is DummyProgressBar:
            logger.warning(
                "Progress bars are requested, but `tqdm` is not installed. The solver"
                + " will run without progress bars."
            )

        # Check if the user wants a progress bar. Initialize an instance of the
        # progressbar_class, which is either :class:`~tqdm.trange` or
        # :class:`~DummyProgressbar` in case `tqdm` is not installed.
        if use_progress_bar:
            # Allow the position of the progress bar to be flexible, depending on
            # whether this is called inside a time loop, a time loop and an
            # additional propagation loop or inside a stationary problem (default).
            progress_bar_position = cast(
                int, self.params.get("_nl_progress_bar_position", 0)
            )

            # Length is the maximal number of Newton iterations.
            self.solver_progressbar = progressbar_class(  # type: ignore
                range(self.tol.max_iterations),
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
                # Prepare a nonlinear iteration.
                model.before_nonlinear_iteration()

                # Perform a single Newton iteration.
                nonlinear_increment = self.nonlinear_iteration(model)

                # Monitor convergence.
                status, info = self.check_nonlinear_convergence(
                    model, nonlinear_increment
                )

                # Logging and progress bar update.
                self.logging(model, info)

                # Update model status.
                model.after_nonlinear_iteration(nonlinear_increment)

                # Update solver statistics.
                self.update_solver_statistics(model, status, info)

                # Exit the Newton loop.
                if status.is_converged() or status.is_failed():
                    break

        # Close the progress bar.
        self.solver_progressbar.close()

        # React to convergence status.
        if status.is_converged():
            model.after_nonlinear_convergence()
        elif status.is_failed():
            model.after_nonlinear_failure()
        else:
            raise ValueError
        return status

    def nonlinear_iteration(self, model) -> np.ndarray:
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

    def check_nonlinear_convergence(
        self,
        model,
        nonlinear_increment: np.ndarray,
    ) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Implements a nonlinear convergence check.

        The convergence check implicitly assumes relative errors and passes reference
        norms for both increments and residuals to the managing `convergence_criterion`.
        The explicit goal is to build the current state as reference for the increment,
        and the initial residual as reference for the residual. The responsibility for
        managing its reference values is given to the convergence criterion.

        Parameters:
            model: The model instance specifying the problem to be solved, knowing
                of its metrics for measuring states and residuals.
            nonlinear_increment: Newly obtained solution increment vector.

        Returns:
            ConvergenceStatus: Convergence status of the nonlinear solver.
            ConvergenceInfo: Norms of the error quantities.

        """
        # Fetch the residual and current iterate.
        residual = model.equation_system.assemble(evaluate_jacobian=False)
        iterate = model.equation_system.get_variable_values(iterate_index=0)

        # TODO: How important is this? Newton is only called for nonlinear
        # problems, by default.
        # Suggestion: Remove - kept for now for discussion.
        # If the model is linear, we do not need to check convergence.
        # if not model._is_nonlinear_problem():
        #     convergence_criterion = NanConvergenceCriterion()
        #     status, info = convergence_criterion.check(
        #         nonlinear_increment, residual, self.params
        #     )
        #     return status, info

        # Trivial nan-check
        if np.isnan(nonlinear_increment).any() or np.isnan(residual).any():
            return ConvergenceStatus.NAN, ConvergenceInfo(np.nan, np.nan)

        # Model-specific check. Compute norms of the nonlinear increment and residual.
        # Potentially a scalar, but also dictionaries are possible if equation-based
        # norms are used.
        nonlinear_increment_norm = model.variable_norm(nonlinear_increment)
        residual_norm = model.residual_norm(residual)
        iterate_norm = model.variable_norm(iterate)

        # Each iteration requires a new reference value for the convergence criterion.
        if model.nonlinear_solver_statistics.num_iteration == 0:
            self.convergence_criterion.reset_reference_values()

        # Cache reference values for convergence checks.
        self.convergence_criterion.set_reference_value(
            reference_nonlinear_increment_norm=iterate_norm,
            reference_residual_norm=residual_norm,
        )

        # Check convergence using the convergence criterion.
        status, info = self.convergence_criterion.check(
            nonlinear_increment_norm,
            residual_norm,
            self.tol,
        )

        # Overwrite convergence status if maximum iterations have been reached.
        if model.nonlinear_solver_statistics.num_iteration >= self.tol.max_iterations:
            status = ConvergenceStatus.MAX_ITERATIONS_REACHED

        return status, info

    def logging(
        self,
        model,
        info: ConvergenceInfo,
    ) -> None:
        """Log the current state of the nonlinear solver.

        This includes printing the current iteration number, nonlinear increment norm,
        and residual norm, as well as updating the progress bar.

        Parameters:
            model: The model instance specifying the problem to be solved.
            nonlinear_increment_norm: The norm of the nonlinear increment.
            residual_norm: The norm of the residual.

        """
        logger.info(
            "Newton iteration number "
            + f"{model.nonlinear_solver_statistics.num_iteration}"
            + f" of {self.tol.max_iterations}"
        )
        logger.info(
            f"Nonlinear increment norm: {info.nonlinear_increment_norm:.2e}, "
            f"Nonlinear residual norm: {info.residual_norm:.2e}"
        )
        self.solver_progressbar.update(n=1)
        self.solver_progressbar.set_postfix_str(
            f"""Increment {info.nonlinear_increment_norm:.2e} """
            f"""Residual {info.residual_norm:.2e}"""
        )

    def update_solver_statistics(
        self,
        model,
        status: ConvergenceStatus,
        info: ConvergenceInfo,
    ) -> None:
        """Update the solver statistics in the model.

        Parameters:
            model: The model instance specifying the problem to be solved.
            status (ConvergenceStatus): Convergence status of the solver.
            info (dict): Dictionary containing norms and other information.

        """
        # Administration of solver statistics.
        model.nonlinear_solver_statistics.advance_iteration()

        # Convergence-related information.
        model.nonlinear_solver_statistics.log_convergence_status(status)
        model.nonlinear_solver_statistics.log_error(info)

        # Basic discretization-related information.
        model.nonlinear_solver_statistics.log_mesh_information(model.mdg.subdomains())
        if model._is_time_dependent():
            model.nonlinear_solver_statistics.log_time_information(
                model.time_manager.time_index,
                model.time_manager.time,
                model.time_manager.dt,
                model.time_manager.final_time_reached(),
            )
