"""
Nonlinear solvers to be used with model classes.
Implemented classes
    NewtonSolver
"""

import logging
from typing import cast

import numpy as np

from porepy.models.solution_strategy import SolutionStrategy
from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceStatus,
    ConvergenceStatusDict,
    ConvergenceCriteria,
    DivergenceCriteria,
    IncrementBasedAbsoluteCriterion,
    IncrementBasedRelativeCriterion,
    IncrementBasedNanCriterion,
    IncrementBasedAbsoluteDivergenceCriterion,
    ResidualBasedAbsoluteCriterion,
    ResidualBasedRelativeCriterion,
    ResidualBasedNanCriterion,
    ResidualBasedAbsoluteDivergenceCriterion,
    MaxIterationsCriterion,
)
from porepy.models.metric import EuclideanMetric

# from porepy.models.metric import EuclideanMetric
from porepy.utils.ui_and_logging import DummyProgressBar
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from porepy.utils.ui_and_logging import progressbar_class
from porepy.viz.solver_statistics import NonlinearSolverStatistics, TimeStatistics

# Module-wide logger
logger = logging.getLogger(__name__)


class NewtonSolver:
    def __init__(self, params=None) -> None:
        if params is None:
            params = {}
        self.params = params
        """Dictionary of parameters for the nonlinear solver."""

        # Default parameters for convergence and divergence criteria
        max_iterations = params.get("max_iterations", 10)
        inc_atol = params.get("nl_convergence_inc_atol", 1e-6)
        inc_rtol = params.get("nl_convergence_inc_rtol", 1e-6)
        res_atol = params.get("nl_convergence_res_atol", 1e-4)
        res_rtol = params.get("nl_convergence_res_rtol", 1e-4)
        inc_div_tol = params.get("nl_divergence_inc_tol", np.inf)
        res_div_tol = params.get("nl_divergence_res_tol", np.inf)

        if "nl_convergence_criteria" not in params:
            self.params["nl_convergence_criteria"] = (
                {
                    "inc_abs": IncrementBasedAbsoluteCriterion(
                        tol=inc_atol, metric=EuclideanMetric()
                    ),
                    "inc_rel": IncrementBasedRelativeCriterion(
                        tol=inc_rtol, metric=EuclideanMetric()
                    ),
                    "res_abs": ResidualBasedAbsoluteCriterion(
                        tol=res_atol, metric=EuclideanMetric()
                    ),
                    "res_rel": ResidualBasedRelativeCriterion(
                        tol=res_rtol, metric=EuclideanMetric()
                    ),
                },
            )
        self.convergence_criteria = ConvergenceCriteria(
            self.params.get("nl_convergence_criteria")
        )
        """Convergence criterion used in the convergence check."""

        if "nl_divergence_criteria" not in params:
            self.params["nl_divergence_criteria"] = (
                {
                    "max_iter": MaxIterationsCriterion(max_iterations=max_iterations),
                    "inc_nan": IncrementBasedNanCriterion(metric=EuclideanMetric()),
                    "res_nan": ResidualBasedNanCriterion(metric=EuclideanMetric()),
                    "inc_max": IncrementBasedAbsoluteDivergenceCriterion(
                        tol=inc_div_tol, metric=EuclideanMetric()
                    ),
                    "res_max": ResidualBasedAbsoluteDivergenceCriterion(
                        tol=res_div_tol, metric=EuclideanMetric()
                    ),
                },
            )
        self.divergence_criteria = DivergenceCriteria(
            self.params.get("nl_divergence_criteria")
        )
        """Divergence criterion used in the convergence check."""

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
                range(self.params["max_iteration"]),
                desc="Newton loop",
                position=progress_bar_position,
                leave=False,
                dynamic_ncols=True,
            )
        # Otherwise, use a dummy progress bar.
        else:
            self.solver_progressbar = DummyProgressBar()

    def solve(
        self, model: SolutionStrategy
    ) -> ConvergenceStatus | ConvergenceStatusDict:
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
                status, info = self.check_convergence(model, nonlinear_increment)

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
            status = model.after_nonlinear_failure(status)
        else:
            raise ValueError(f"Unknown convergence status: {status}")
        return status

    def nonlinear_iteration(self, model: SolutionStrategy) -> np.ndarray:
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
        model: SolutionStrategy,
        nonlinear_increment: np.ndarray,
    ) -> tuple[ConvergenceStatusDict, dict] | tuple[ConvergenceStatus | float]:
        """Implements a nonlinear convergence check.

        The convergence check implicitly assumes relative errors and passes reference
        norms for both increments and residuals to the managing `convergence_criteria`.
        The explicit goal is to build the current state as reference for the increment,
        and the initial residual as reference for the residual. The responsibility for
        managing its reference values is given to the convergence criterion.

        Parameters:
            model: The model instance specifying the problem to be solved, knowing
                of its metrics for measuring states and residuals.
            nonlinear_increment: Newly obtained solution increment vector.

        Returns:
            tuple[ConvergenceStatusDict, dict]: Convergence status and norms
                of the error quantities.

        """
        # Fetch the residual and current iterate.
        residual = model.equation_system.assemble(evaluate_jacobian=False)
        iterate = model.equation_system.get_variable_values(iterate_index=0)

        # Each iteration requires a new reference value for the convergence criterion.
        assert isinstance(model.nonlinear_solver_statistics, NonlinearSolverStatistics)
        if model.nonlinear_solver_statistics.num_iteration == 0:
            self.convergence_criteria.reset()

        # Check convergence status based on current iteration.
        convergence_status, convergence_info = self.convergence_criteria.check(
            increment=nonlinear_increment,
            reference_increment=iterate,
            residual=residual,
            reference_residual=residual,
        )

        # Check divergence status based on current iteration.
        divergence_status = self.divergence_criteria.check(
            increment=nonlinear_increment,
            reference_increment=iterate,
            residual=residual,
            reference_residual=residual,
            num_iterations=model.nonlinear_solver_statistics.num_iteration,
        )

        return (
            convergence_status.union(divergence_status),
            convergence_info,
        )

    def logging(
        self,
        model: SolutionStrategy,
        info: dict | float,
    ) -> None:
        """Log the current state of the nonlinear solver.

        This includes printing the current iteration number, nonlinear increment norm,
        and residual norm, as well as updating the progress bar.

        Parameters:
            model: The model instance specifying the problem to be solved.
            info: Convergence information containing norms and other details.

        """
        assert isinstance(model.nonlinear_solver_statistics, NonlinearSolverStatistics)
        max_iterations = self.params.get("max_iterations", 10)
        logger.info(
            "Newton iteration number "
            + f"{model.nonlinear_solver_statistics.num_iteration}"
            + f" of {max_iterations}"
        )
        # TODO: Provide a way of logging.
        # logger.info(
        #    f"Nonlinear increment norm: {info.nonlinear_increment_norm:.2e}, "
        #    f"Nonlinear residual norm: {info.residual_norm:.2e}"
        # )
        self.solver_progressbar.update(n=1)
        # self.solver_progressbar.set_postfix_str(
        #    f"""Increment {info.nonlinear_increment_norm:.2e} """
        #    f"""Residual {info.residual_norm:.2e}"""
        # )

    def update_solver_statistics(
        self,
        model: SolutionStrategy,
        status: ConvergenceStatus,
        info: dict | float,
    ) -> None:
        """Update the solver statistics in the model.

        Parameters:
            model: The model instance specifying the problem to be solved.
            status: Convergence status of the solver.
            info: Dictionary containing norms and other information.

        """
        # Administration of solver statistics.
        assert isinstance(model.nonlinear_solver_statistics, NonlinearSolverStatistics)
        model.nonlinear_solver_statistics.advance_iteration()

        # Convergence-related information.
        model.nonlinear_solver_statistics.log_convergence_status(status)
        model.nonlinear_solver_statistics.log_info(info)

        # Basic discretization-related information.
        model.nonlinear_solver_statistics.log_mesh_information(model.mdg.subdomains())
        if model._is_time_dependent():
            assert isinstance(model.nonlinear_solver_statistics, TimeStatistics)
            model.nonlinear_solver_statistics.log_time_information(
                model.time_manager.time_index,
                model.time_manager.time,
                model.time_manager.dt,
                model.time_manager.final_time_reached(),
            )
