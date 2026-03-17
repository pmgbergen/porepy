"""
Nonlinear solvers to be used with model classes.
Implemented classes
    NewtonSolver
"""

import logging
from typing import cast

import numpy as np

import porepy as pp
from porepy.models.solution_strategy import SolutionStrategy

# from porepy.numerics.linear_solvers import LinearSolver
from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceCriteria,
    ConvergenceInfoCollection,
    ConvergenceStatusCollection,
    DivergenceCriteria,
    SimulationStatus,
)
from porepy.utils.ui_and_logging import DummyProgressBar
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from porepy.utils.ui_and_logging import progressbar_class

# Module-wide logger
logger = logging.getLogger(__name__)


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

    If custom convergence or divergence criteria are provided, individual tolerance
    parameters should not be provided to avoid double specification. If no custom
    criteria are provided, default criteria are used based on the individual tolerance
    parameters and metric.

    """

    def __init__(self, params=None) -> None:
        if params is None:
            params = {}
        self.params = params
        """Dictionary of parameters for the nonlinear solver."""
        self.iteration_index: int = 0
        """Current iteration index - equivalent with number of iterations."""

        self.init_convergence_criteria()
        self.init_divergence_criteria()
        self.init_solver_progressbar()

    def init_convergence_criteria(self) -> None:
        """Parse and initialize convergence criteria.

        Convergence criteria can either be provided as a dictionary in the
        'nl_convergence_criteria' parameter, or default criteria are used
        controlled by individual tolerance parameters based on the following template.

        - Increment-based absolute criterion: 'nl_convergence_inc_atol'
        - Increment-based relative criterion: 'nl_convergence_inc_rtol'
        - Residual-based absolute criterion: 'nl_convergence_res_atol'
        - Residual-based relative criterion: 'nl_convergence_res_rtol'
        - Metric: 'nl_metric'

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
            inc_atol = self.params.get("nl_convergence_inc_atol", 1e-10)
            inc_rtol = self.params.get("nl_convergence_inc_rtol", np.inf)
            res_atol = self.params.get("nl_convergence_res_atol", 1e-10)
            res_rtol = self.params.get("nl_convergence_res_rtol", np.inf)
            metric = self.params.get("nl_metric", pp.EuclideanMetric())
            convergence_criteria = {
                "inc_abs": pp.IncrementBasedAbsoluteCriterion(
                    tol=inc_atol, metric=metric
                ),
                "inc_rel": pp.IncrementBasedRelativeCriterion(
                    tol=inc_rtol, metric=metric
                ),
                "res_abs": pp.ResidualBasedAbsoluteCriterion(
                    tol=res_atol, metric=metric
                ),
                "res_rel": pp.ResidualBasedRelativeCriterion(
                    tol=res_rtol, metric=metric
                ),
            }

        # Initialize convergence criteria.
        self.convergence_criteria = ConvergenceCriteria(convergence_criteria)
        """Convergence criterion used in the convergence check."""

    def init_divergence_criteria(self) -> None:
        """Parse and initialize divergence criteria.

        Divergence criteria can either be provided as a dictionary in the
        'nl_divergence_criteria' parameter, or default criteria are used based on
        the following template.

        - Maximum number of iterations: 'nl_max_iterations'
        - Increment-based divergence tolerance: 'nl_divergence_inc_atol'
        - Residual-based divergence tolerance: 'nl_divergence_res_atol'
        - Metric: 'nl_metric'

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
            # be provided.
            max_iterations = None
            for c in self.params["nl_divergence_criteria"].values():
                if isinstance(c, pp.MaxIterationsCriterion):
                    max_iterations = c.max_iterations
        else:
            # Default parameters for divergence criteria.
            max_iterations = self.params.get("nl_max_iterations", 10)
            inc_div_atol = self.params.get("nl_divergence_inc_atol", np.inf)
            res_div_atol = self.params.get("nl_divergence_res_atol", np.inf)
            metric = self.params.get("nl_metric", pp.EuclideanMetric())
            divergence_criteria = {
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

        # Cache maximum number of iterations for easy access.
        self.max_iterations: int | None = max_iterations
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
                range(
                    self.max_iterations or 10
                ),  # Fallback to 10 iterations if not set
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

    def solve(self, model: SolutionStrategy) -> SimulationStatus:
        """Solve the nonlinear problem using the Newton-Raphson method.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            SimulationStatus: The status of the nonlinear solver.

        """
        # Prepare for nonlinear loop.
        self.before_nonlinear_loop(model)

        # Actual Newton loop.
        convergence_status, divergence_status = self.nonlinear_loop(model)

        # Finalize the nonlinear loop.
        simulation_status = self.after_nonlinear_loop(
            model, convergence_status, divergence_status
        )

        return simulation_status

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
                (convergence_status, divergence_status) = (
                    self.after_nonlinear_iteration(model, nonlinear_increment)
                )

                # Exit the Newton loop.
                if convergence_status.is_converged() or divergence_status.is_diverged():
                    break

        return convergence_status, divergence_status

    def after_nonlinear_loop(
        self,
        model: SolutionStrategy,
        convergence_status: ConvergenceStatusCollection,
        divergence_status: ConvergenceStatusCollection,
    ) -> SimulationStatus:
        """Finalize the nonlinear loop.

        Parameters:
            model: The model instance specifying the problem to be solved.
            convergence_status: The convergence status collection.
            divergence_status: The divergence status collection.

        Returns:
            SimulationStatus: The status of the nonlinear solver.

        """
        # React to convergence status. Let convergence trump divergence.
        if convergence_status.is_converged():
            simulation_status = SimulationStatus.SUCCESSFUL
            self.update_solver_statistics(model, simulation_status=simulation_status)
            model.after_nonlinear_convergence()
        elif divergence_status.is_diverged():
            simulation_status = SimulationStatus.FAILED
            self.update_solver_statistics(model, simulation_status=simulation_status)
            # TODO: Get back to this when reimplementing time stepping.
            # NOTE: Currently, if a simulation fully stopps, this is not logged in
            # SolverStatistics. For this, better coordination between solver and time
            # stepping is needed.
            simulation_status = model.after_nonlinear_failure()
        else:
            raise ValueError(f"Unknown convergence status: {convergence_status}")

        # Close the progress bar.
        self.solver_progressbar.close()

        return simulation_status

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
            tuple[
                ConvergenceStatusCollection,
                ConvergenceStatusCollection,
            ]: Convergence and divergence status.

        """
        # Update model status.
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
            tuple[ConvergenceStatusCollection, ConvergenceStatusCollection,
            ConvergenceInfoCollection]: Status and
                info about convergence and divergence.

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
        assert isinstance(
            model.nonlinear_solver_statistics, pp.NonlinearSolverStatistics
        )

        # Log iteration number.
        iteration_msg = f"Newton iteration number {self.iteration_index}"
        if self.max_iterations is not None:
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
            f"""Increment {nonlinear_increment_norm:.2e} """
            f"""Residual {residual_norm:.2e}"""
        )

    def update_solver_statistics(
        self,
        model: SolutionStrategy,
        simulation_status: SimulationStatus | None = None,
        convergence_status: ConvergenceStatusCollection | None = None,
        convergence_info: ConvergenceInfoCollection | None = None,
    ) -> None:
        """Update the solver statistics in the model.

        Parameters:
            model: The model instance specifying the problem to be solved.
            simulation_status: Simulation status of the solver.
            convergence_status: Convergence (and divergence) status of the solver.
            convergence_info: Dictionary containing norms and other information.

        """
        assert isinstance(
            model.nonlinear_solver_statistics, pp.NonlinearSolverStatistics
        )

        # Convergence-related information.
        if convergence_status is not None and convergence_info is not None:
            model.nonlinear_solver_statistics.log_convergence_status(convergence_status)
            model.nonlinear_solver_statistics.log_convergence_info(convergence_info)

        # Basic discretization-related information and overall simulation status.
        if simulation_status is not None:
            pp.LinearSolver.update_solver_statistics(
                cast(pp.LinearSolver, self), model, simulation_status
            )
