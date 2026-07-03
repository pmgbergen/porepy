"""
Nonlinear solvers to be used with model classes.
Implemented classes
    NewtonSolver
"""

import logging
from typing import Optional, cast

import numpy as np

import porepy as pp
from porepy.models.solution_strategy import SolutionStrategy
from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceCriteria,
    ConvergenceInfoCollection,
    ConvergenceMetricType,
    ConvergenceStatusCollection,
    DivergenceCriteria,
)
from porepy.numerics.nonlinear.nonlinear_solver_status import (
    NonlinearSolverStatus,
    NonlinearSolverStatusConverged,
    NonlinearSolverStatusFailed,
)
from porepy.utils.ui_and_logging import DummyProgressBar
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from porepy.utils.ui_and_logging import progressbar_class

# Module-wide logger
logger = logging.getLogger(__name__)


DEFAULT_NEWTON_MAX_ITERATIONS = 10
"""Default maximum number of Newton iterations."""


class NewtonSolver:
    """Nonlinear solver class implementing the Newton-Raphson method.

    This class is responsible for solving nonlinear equations using the
    Newton-Raphson method. It manages the iteration process, while convergence
    and divergence criteria are checked at each iteration.

    Parameters:
        params: Dictionary of parameters for the nonlinear solver. This can include
            - 'nl_convergence_criteria': Custom convergence criteria.
            - 'nl_divergence_criteria': Custom divergence criteria.
            - 'nl_max_iterations': Maximum number of iterations.
            - 'nl_convergence_inc_atol': Increment-based absolute tolerance.
            - 'nl_convergence_inc_rtol': Increment-based relative tolerance.
            - 'nl_convergence_res_atol': Residual-based absolute tolerance.
            - 'nl_convergence_res_rtol': Residual-based relative tolerance.
            - 'nl_metric': Metric used for convergence checks.
        is_nonlinear_problem: Whether the underlying problem is nonlinear. This
            parameter selects the default convergence and divergence criteria for the
            problem. For nonlinear problems, the criteria require the method to perform
            as many iterations as needed to reduce both the increment and the residual
            within the specified tolerance. For linear problems, only a single
            iteration is performed, and expensive convergence checks are skipped. If
            custom convergence or divergence criteria are provided, this parameter is
            ignored.

    If custom convergence or divergence criteria are provided, individual tolerance
    parameters should not be provided to avoid double specification. If no custom
    criteria are provided, default criteria are used based on the individual tolerance
    parameters and metric.

    """

    def __init__(self, params=None, is_nonlinear_problem: bool = True) -> None:
        if params is None:
            params = {}
        self.params = params
        """Dictionary of parameters for the nonlinear solver."""
        self.iteration_index: int = 0
        """Current iteration index - equivalent with number of iterations."""

        self.init_convergence_criteria(is_nonlinear_problem=is_nonlinear_problem)
        self.init_divergence_criteria(is_nonlinear_problem=is_nonlinear_problem)

    def init_convergence_criteria(self, is_nonlinear_problem: bool) -> None:
        """Parse and initialize convergence criteria.

        Parameters:
            is_nonlinear_problem: Whether the underlying problem is nonlinear.

        Convergence criteria can either be provided as a dictionary in the
        'nl_convergence_criteria' parameter, or default criteria are used
        controlled by individual tolerance parameters based on the following template.

        - Increment-based absolute criterion: 'nl_convergence_inc_atol'
        - Increment-based relative criterion: 'nl_convergence_inc_rtol'
        - Residual-based absolute criterion: 'nl_convergence_res_atol'
        - Residual-based relative criterion: 'nl_convergence_res_rtol'
        - Metric: 'nl_metric'

        Default convergence criteria for a nonlinear problem:
        - Increment atol < 1e-10;
        - Residual atol < 1e-10;
        - Using Euclidian metric.

        Default convergence criteria for a linear problem:
        - None, it accepts any solution after a single iteration if not diverged.

        """

        if "nl_convergence_criteria" in self.params:
            # Use user-provided convergence criteria.
            convergence_criteria = self.params["nl_convergence_criteria"]

            # Perform sanity check to avoid double specification of tolerances.
            assert not any(
                [
                    key in self.params
                    for key in [
                        "nl_convergence_inc_atol",
                        "nl_convergence_inc_rtol",
                        "nl_convergence_res_atol",
                        "nl_convergence_res_rtol",
                        # "nl_metric", # Potentially, used for divergence
                    ]
                ]
            ), (
                "If 'nl_convergence_criteria' is provided, do not provide "
                + "individual convergence tolerances."
            )
        else:
            # If no custom convergence criteria are provided, use default ones.
            convergence_criteria = _default_convergence_criteria(
                is_nonlinear_problem=is_nonlinear_problem,
                inc_atol=self.params.get("nl_convergence_inc_atol", 1e-10),
                inc_rtol=self.params.get("nl_convergence_inc_rtol", np.inf),
                res_atol=self.params.get("nl_convergence_res_atol", 1e-10),
                res_rtol=self.params.get("nl_convergence_res_rtol", np.inf),
                metric=self.params.get("nl_metric", pp.EuclideanMetric()),
            )

        # Initialize convergence criteria.
        self.convergence_criteria = ConvergenceCriteria(convergence_criteria)
        """Convergence criterion used in the convergence check."""

    def init_divergence_criteria(self, is_nonlinear_problem: bool) -> None:
        """Parse and initialize divergence criteria.

        Parameters:
            is_nonlinear_problem: Whether the underlying problem is nonlinear.

        Divergence criteria can either be provided as a dictionary in the
        'nl_divergence_criteria' parameter, or default criteria are used based on
        the following template.

        - Maximum number of iterations: 'nl_max_iterations'
        - Increment-based divergence tolerance: 'nl_divergence_inc_atol'
        - Residual-based divergence tolerance: 'nl_divergence_res_atol'
        - Metric: 'nl_metric'

        Default divergence criteria for a nonlinear problem:
        - Iteration number within the limit.
        - Residual and increment are not nans.

        Default divergence criteria for a linear problem:
        - Residual and increment are not nans.

        """

        if "nl_divergence_criteria" in self.params:
            # Use user-provided divergence criteria.
            divergence_criteria = self.params["nl_divergence_criteria"]

            # Perform sanity check to avoid double specification of tolerances.
            assert not any(
                [
                    key in self.params
                    for key in [
                        "nl_max_iterations",
                        "nl_divergence_inc_atol",
                        "nl_divergence_res_atol",
                        # "nl_metric", # Potentially, used for convergence
                    ]
                ]
            ), (
                "If 'nl_divergence_criteria' is provided, do not provide "
                + "individual divergence tolerances."
            )

            # Fetch max iterations from the provided criteria. Note, it may not
            # be provided. Using the default value if not provided.
            max_iterations = DEFAULT_NEWTON_MAX_ITERATIONS
            for c in self.params["nl_divergence_criteria"].values():
                if isinstance(c, pp.MaxIterationsCriterion):
                    max_iterations = c.max_iterations
        else:
            # Default parameters for divergence criteria.

            # If a user provides max_iteration for a linear problem, this value is
            # ignored, because the default divergence criterion does not include it.
            max_iterations = self.params.get(
                "nl_max_iterations", DEFAULT_NEWTON_MAX_ITERATIONS
            )
            divergence_criteria = _default_divergence_criteria(
                is_nonlinear_problem=is_nonlinear_problem,
                max_iterations=max_iterations,
                inc_div_atol=self.params.get("nl_divergence_inc_atol", np.inf),
                res_div_atol=self.params.get("nl_divergence_res_atol", np.inf),
                metric=self.params.get("nl_metric", pp.EuclideanMetric()),
            )

        # Cache maximum number of iterations for easy access.
        self.max_iterations: int = max_iterations
        """Maximum number of nonlinear iterations."""

        # Initialize divergence criteria.
        self.divergence_criteria = DivergenceCriteria(divergence_criteria)
        """Divergence criterion used in the convergence check."""

    def init_solver_progressbar(self) -> None:
        """Initialize the solver progress bar.

        To enable the progress bar, set the 'progressbars' parameter to True.

        """
        use_progress_bar = bool(self.params.get("progressbars", False))
        if use_progress_bar and progressbar_class is DummyProgressBar:
            logger.warning(
                "Progress bars are requested, but `tqdm` is not installed. The solver"
                " will run without progress bars."
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
                range(self.max_iterations),
                desc="Newton loop",
                position=progress_bar_position,
                leave=False,
                dynamic_ncols=True,
            )
        # Otherwise, use a dummy progress bar.
        else:
            self.solver_progressbar = DummyProgressBar()

    def increase_iteration_index(self) -> None:
        """Advance to the next iteration."""
        self.iteration_index += 1

    def solve(self, model: SolutionStrategy) -> NonlinearSolverStatus:
        """Solve the nonlinear problem using the Newton-Raphson method.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            The status of the nonlinear solver.

        """
        # Prepare for nonlinear loop.
        self.before_nonlinear_loop(model)

        # Actual Newton loop.
        convergence_status, divergence_status = self.nonlinear_loop(model)

        # Summarizing the convergence message from multiple criteria into an overall
        # status.
        solver_status = _summarize_solver_status(
            convergence_status, divergence_status, num_iterations=self.iteration_index
        )

        # Logging basic discretization-related information and overall simulation status
        _update_solver_statistics_after_nonlinear_solve(
            model=model, solver_status=solver_status
        )

        # This must be done after writing statistics, since model.after_*** calls write
        # statistics to json file.
        if solver_status.is_converged():
            model.after_nonlinear_convergence()
        else:
            model.after_nonlinear_failure()

        # Finalize the nonlinear loop.
        self.after_nonlinear_loop()

        return solver_status

    def before_nonlinear_loop(self, model: SolutionStrategy) -> None:
        """Prepare for the nonlinear loop.

        Parameters:
            model: The model instance specifying the problem to be solved.

        """
        # Prepare model for nonlinear loop.
        model.before_nonlinear_loop()

        # Prepare solver for nonlinear loop.
        self.iteration_index = 0
        self.convergence_criteria.reset()

        # Reset solver progressbar.
        self.init_solver_progressbar()

    def nonlinear_loop(
        self, model: SolutionStrategy
    ) -> tuple[ConvergenceStatusCollection, ConvergenceStatusCollection]:
        """Perform the nonlinear loop (Newton iterations).

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            tuple[ConvergenceStatusCollection, ConvergenceStatusCollection]:
                Convergence and divergence status.

        """
        # Redirect all loggers to not interfere with the progressbar.
        with logging_redirect_tqdm([logging.root]):
            # Perform at least one Newton iteration.
            while True:
                # Prepare for nonlinear iteration.
                self.before_nonlinear_iteration(model)

                # Perform nonlinear iteration and obtain increment.
                nonlinear_increment = self.nonlinear_iteration(model)

                # Finalize nonlinear iteration and determine status.
                convergence_status, divergence_status = self.after_nonlinear_iteration(
                    model, nonlinear_increment
                )

                # Exit the Newton loop.
                if convergence_status.is_converged() or divergence_status.is_failed():
                    break

        return convergence_status, divergence_status

    def after_nonlinear_loop(self) -> None:
        """Finalize the nonlinear loop."""
        # Close the progress bar.
        self.solver_progressbar.close()

    def before_nonlinear_iteration(self, model: SolutionStrategy) -> None:
        """Prepare for a nonlinear iteration.

        Parameters:
            model: The model instance specifying the problem to be solved.

        """
        # Start iteration.
        self.increase_iteration_index()

        # Prepare model for a nonlinear iteration.
        model.before_nonlinear_iteration()

    def nonlinear_iteration(self, model: SolutionStrategy) -> np.ndarray:
        """Perform a single nonlinear iteration.

        Right now, this is an almost trivial function. However, we keep it as a separate
        function to prepare for possible future introduction of more advanced schemes.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            np.ndarray: Solution to linearized system, i.e. the update increment.

        """
        nonlinear_increment = self.iteration(model)
        return nonlinear_increment

    def iteration(self, model: SolutionStrategy) -> np.ndarray:
        """A single linearization step.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            np.ndarray: Solution to linearized system, i.e. the update increment.

        """
        model.assemble_linear_system()
        nonlinear_increment = model.solve_linear_system()
        return nonlinear_increment

    def after_nonlinear_iteration(
        self, model: SolutionStrategy, nonlinear_increment: np.ndarray
    ) -> tuple[ConvergenceStatusCollection, ConvergenceStatusCollection]:
        """Finalize a nonlinear iteration.

        Parameters:
            model: The model instance specifying the problem to be solved.
            nonlinear_increment: Newly obtained solution increment vector.

        Returns:
            tuple[ConvergenceStatusCollection, ConvergenceStatusCollection]:
                Convergence and divergence status.

        """
        # Update model status (iterate) before checking convergence, so that the
        # convergence check uses the updated state. Also, after_nonlinear_convergence
        # may expect the converged solution to already be stored as an iterate.
        model.after_nonlinear_iteration(nonlinear_increment)

        # Monitor convergence.
        convergence_status, divergence_status, convergence_info = (
            self.check_convergence(model, nonlinear_increment)
        )

        # Logging and progress bar update.
        self.logging(model, convergence_info, nonlinear_increment)

        # Update (iteration-based) solver statistics.
        self.update_solver_statistics(
            model,
            convergence_status=convergence_status.union(divergence_status),
            convergence_info=convergence_info,
        )

        return convergence_status, divergence_status

    def check_convergence(
        self,
        model: SolutionStrategy,
        nonlinear_increment: np.ndarray,
    ) -> tuple[
        ConvergenceStatusCollection,
        ConvergenceStatusCollection,
        ConvergenceInfoCollection,
    ]:
        """Check convergence and divergence based on passed criteria.

        Parameters:
            model: The model instance specifying the problem to be solved, knowing
                of its metrics for measuring states and residuals.
            nonlinear_increment: Newly obtained solution increment vector.

        Returns:
            Tuple containing:
                - ConvergenceStatusCollection: Status and info about convergence.
                - ConvergenceStatusCollection: Status and info about divergence.
                - ConvergenceInfoCollection: Detailed information about the
                    convergence process.

        """
        # Fetch the residual and current iterate.
        residual = model.equation_system.assemble(evaluate_jacobian=False)
        iterate = model.equation_system.get_variable_values(iterate_index=0)

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
            num_iterations=self.iteration_index,
        )

        return convergence_status, divergence_status, convergence_info

    def logging(
        self,
        model: SolutionStrategy,
        convergence_info: dict[str, dict | float],
        nonlinear_increment: np.ndarray,
    ) -> None:
        """Log the current state of the nonlinear solver.

        This includes printing the current iteration number, nonlinear increment norm,
        and residual norm, as well as updating the progress bar.

        Parameters:
            model: The model instance specifying the problem to be solved.
            convergence_info: Convergence information containing norms.
            nonlinear_increment: Newly obtained solution increment vector.

        """
        # TODO: The logging should be agnostic to the chosen criteria and metric.
        # Use currently simple np.linalg norms for logging instead of convergence_info.
        # To be revisited - remove nonlinear_increment parameter then as well.

        # Log iteration number.
        iteration_msg = f"Newton iteration number {self.iteration_index}"
        iteration_msg += f" of {self.max_iterations}"
        logger.info(iteration_msg)

        # Log norms.
        nonlinear_increment_norm = np.linalg.norm(nonlinear_increment)
        residual = model.equation_system.assemble(evaluate_jacobian=False)
        residual_norm = np.linalg.norm(residual)
        logger.info(
            f"Nonlinear increment norm: {nonlinear_increment_norm:.2e}, "
            f"Nonlinear residual norm: {residual_norm:.2e}"
        )

        # Update progress bar.
        self.solver_progressbar.update(n=1)
        self.solver_progressbar.set_postfix_str(
            f"Increment {nonlinear_increment_norm:.2e} Residual {residual_norm:.2e}"
        )

    def update_solver_statistics(
        self,
        model: SolutionStrategy,
        convergence_status: ConvergenceStatusCollection,
        convergence_info: ConvergenceInfoCollection,
    ) -> None:
        """Update the solver statistics in the model.

        Parameters:
            model: The model instance specifying the problem to be solved.
            convergence_status: Convergence (and divergence) status of the solver.
            convergence_info: Dictionary containing norms and other information.

        """

        if isinstance(model.nonlinear_solver_statistics, pp.NonlinearSolverStatistics):
            # Convergence-related information.
            model.nonlinear_solver_statistics.log_convergence_status(convergence_status)
            model.nonlinear_solver_statistics.log_convergence_info(convergence_info)


def _default_convergence_criteria(
    is_nonlinear_problem: bool,
    inc_atol=1e-10,
    inc_rtol=np.inf,
    res_atol=1e-10,
    res_rtol=np.inf,
    metric: Optional[ConvergenceMetricType] = None,
) -> dict:
    """A convenience factory for the default convergence criteria. Returns different
    criteria based on whether the problem is nonlinear.

    """
    if metric is None:
        metric = pp.EuclideanMetric()
    if is_nonlinear_problem:
        return {
            "inc_abs": pp.IncrementBasedAbsoluteCriterion(tol=inc_atol, metric=metric),
            "inc_rel": pp.IncrementBasedRelativeCriterion(tol=inc_rtol, metric=metric),
            "res_abs": pp.ResidualBasedAbsoluteCriterion(tol=res_atol, metric=metric),
            "res_rel": pp.ResidualBasedRelativeCriterion(tol=res_rtol, metric=metric),
        }
    else:
        return {}


def _default_divergence_criteria(
    is_nonlinear_problem: bool,
    max_iterations=10,
    inc_div_atol=np.inf,
    res_div_atol=np.inf,
    metric: Optional[ConvergenceMetricType] = None,
) -> dict:
    """A convenience factory for the default divergence criteria. Returns different
    criteria based on whether the problem is nonlinear.

    """
    if metric is None:
        metric = pp.EuclideanMetric()
    if is_nonlinear_problem:
        return {
            "max_iter": pp.MaxIterationsCriterion(max_iterations=max_iterations),
            "inc_nan": pp.IncrementBasedNanCriterion(),
            "res_nan": pp.ResidualBasedNanCriterion(),
            "inc_max": pp.IncrementBasedAbsoluteDivergenceCriterion(
                tol=inc_div_atol, metric=metric
            ),
            "res_max": pp.ResidualBasedAbsoluteDivergenceCriterion(
                tol=res_div_atol, metric=metric
            ),
        }
    else:
        return {
            "inc_nan": pp.IncrementBasedNanCriterion(),
            "res_nan": pp.ResidualBasedNanCriterion(),
        }


def _update_solver_statistics_after_nonlinear_solve(
    model: SolutionStrategy,
    solver_status: NonlinearSolverStatus,
) -> None:
    """Update the solver statistics in the model.

    Parameters:
        model: The model instance specifying the problem to be solved.
        solver_status: Simulation status of the solver.

    """
    # Basic discretization-related information and overall simulation status.
    model.nonlinear_solver_statistics.log_solver_status(solver_status)
    model.nonlinear_solver_statistics.log_mesh_information(model.mdg.subdomains())


def _summarize_solver_status(
    convergence_status: ConvergenceStatusCollection,
    divergence_status: ConvergenceStatusCollection,
    num_iterations: int,
) -> NonlinearSolverStatus:
    """Called by the nonlinear solver after the nonlinear iteration is done. Considers a
    collection of convergence and divergence statuses from multiple criteria and makes a
    overall verdict on whether we accept the sollution or not.

    NOTE: Convergence status takes precedence over divergence status. See related issue:
    https://github.com/pmgbergen/porepy/issues/1713

    Parameters:
        convergence_status: Multiple convergence statuses from different criteria.
        divergence_status: Multiple divergence statuses from variaous criteria.

    Returns:
        NonlinearSolverStatus: Either Converged or Failed.

    """
    is_converged = convergence_status.is_converged()
    is_failed = divergence_status.is_failed()
    if is_converged:
        if is_failed:
            logger.warning(
                "Nonlinear solver convergence criteria indicate convergence and "
                "divergence at the same time. Accepting this solution."
            )
        return NonlinearSolverStatusConverged(
            num_nonlinear_iterations=num_iterations,
            convergence_statuses=convergence_status,
            divergence_statuses=divergence_status,
        )
    elif is_failed:
        logger.warning("Failed to solve the nonlinear problem.")
        return NonlinearSolverStatusFailed(
            num_nonlinear_iterations=num_iterations,
            convergence_statuses=convergence_status,
            divergence_statuses=divergence_status,
        )
    else:
        logger.error(
            "Nonlinear solver did not fail, but the convergence criterion did not "
            "accept the solution. Treating it as a failure."
        )
        return NonlinearSolverStatusFailed(
            num_nonlinear_iterations=num_iterations,
            convergence_statuses=convergence_status,
            divergence_statuses=divergence_status,
        )
