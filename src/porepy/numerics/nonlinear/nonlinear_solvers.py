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
    def __init__(self, params=None) -> None:
        if params is None:
            params = {}
        self.params = params
        """Dictionary of parameters for the nonlinear solver."""

        self.init_criteria()
        self.init_solver_progressbar()

    def init_criteria(self) -> None:
        """Parse and initialize convergence and divergence criteria."""

        # Default parameters for convergence criteria.
        inc_atol = self.params.get("nl_convergence_inc_atol", 1e-6)
        inc_rtol = self.params.get("nl_convergence_inc_rtol", np.inf)
        res_atol = self.params.get("nl_convergence_res_atol", 1e-6)
        res_rtol = self.params.get("nl_convergence_res_rtol", np.inf)
        metric = self.params.get("nl_metric", pp.EuclideanMetric())

        if "nl_convergence_criteria" not in self.params:
            self.params["nl_convergence_criteria"] = {
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
        else:
            assert not any(
                [
                    key in self.params
                    for key in [
                        "nl_convergence_inc_atol",
                        "nl_convergence_inc_rtol",
                        "nl_convergence_res_atol",
                        "nl_convergence_res_rtol",
                        "nl_metric",
                    ]
                ]
            ), (
                "If 'nl_convergence_criteria' is provided, do not provide "
                + "individual convergence tolerances."
            )
        self.convergence_criteria = ConvergenceCriteria(
            self.params.get("nl_convergence_criteria")
        )
        """Convergence criterion used in the convergence check."""

        # Default parameters for divergence criteria.
        max_iterations = self.params.get("nl_max_iterations", 10)
        inc_div_tol = self.params.get("nl_divergence_inc_tol", np.inf)
        res_div_tol = self.params.get("nl_divergence_res_tol", np.inf)
        if "nl_divergence_criteria" not in self.params:
            self.params["nl_divergence_criteria"] = {
                "max_iter": pp.MaxIterationsCriterion(max_iterations=max_iterations),
                "inc_nan": pp.IncrementBasedNanCriterion(),
                "res_nan": pp.ResidualBasedNanCriterion(),
                "inc_max": pp.IncrementBasedAbsoluteDivergenceCriterion(
                    tol=inc_div_tol, metric=metric
                ),
                "res_max": pp.ResidualBasedAbsoluteDivergenceCriterion(
                    tol=res_div_tol, metric=metric
                ),
            }
        else:
            assert not any(
                [
                    key in self.params
                    for key in [
                        # "nl_max_iterations", # Currently nl_max_iterations is used also
                        # for controlling the progress bar length, as well as iteration exporting.
                        # Thus, skip this check for now.
                        "nl_divergence_inc_tol",
                        "nl_divergence_res_tol",
                    ]
                ]
            ), (
                "If 'nl_divergence_criteria' is provided, do not provide "
                + "individual divergence tolerances."
            )
            for c in self.params["nl_divergence_criteria"].values():
                if isinstance(c, pp.MaxIterationsCriterion):
                    assert c.max_iterations == max_iterations, (
                        "Inconsistent max iterations across criteria."
                    )
        self.divergence_criteria = DivergenceCriteria(
            self.params.get("nl_divergence_criteria")
        )
        """Divergence criterion used in the convergence check."""

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
                range(self.params["nl_max_iterations"]),
                desc="Newton loop",
                position=progress_bar_position,
                leave=False,
                dynamic_ncols=True,
            )
        # Otherwise, use a dummy progress bar.
        else:
            self.solver_progressbar = DummyProgressBar()

    def solve(self, model: SolutionStrategy) -> SimulationStatus:
        """Solve the nonlinear problem using the Newton-Raphson method.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            SimulationStatus: The status of the nonlinear solver.

        """
        # Prepare nonlinear loop.
        model.before_nonlinear_loop()

        # Redirect all loggers to not interfere with the progressbar.
        with logging_redirect_tqdm([logging.root]):
            # Newton loop.
            while True:
                # Prepare a nonlinear iteration.
                model.before_nonlinear_iteration()

                # Increase the iteration count at the start to ensure natural counting
                # and logging (starting at 1 for a total of one iteration etc.).
                # Keep the control with the nonlinear solver, instead of the model.
                model.nonlinear_solver_statistics.advance_iteration()

                # Perform a single Newton iteration.
                nonlinear_increment = self.nonlinear_iteration(model)

                # Monitor convergence.
                convergence_status, divergence_status, convergence_info = (
                    self.check_convergence(model, nonlinear_increment)
                )

                # Logging and progress bar update.
                self.logging(model, convergence_info, nonlinear_increment)

                # Update model status.
                model.after_nonlinear_iteration(nonlinear_increment)

                # Update (iteration-based) solver statistics.
                self.update_solver_statistics(
                    model,
                    convergence_status=convergence_status,
                    convergence_info=convergence_info,
                    simulation_status=SimulationStatus.IN_PROGRESS,
                )

                # Exit the Newton loop.
                if convergence_status.is_converged() or divergence_status.is_failed():
                    break

        # React to convergence status. Let convergence trump divergence.
        if convergence_status.is_converged():
            simulation_status = SimulationStatus.SUCCESSFUL
            self.update_solver_statistics(model, simulation_status=simulation_status)
            model.after_nonlinear_convergence()
        elif divergence_status.is_failed():
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

        # Each iteration requires a new reference value for the convergence criterion.
        assert isinstance(
            model.nonlinear_solver_statistics, pp.NonlinearSolverStatistics
        )
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
        # Use currently the old norms for logging instead of convergence_info.
        # To be revisited - remove nonlinear_increment parameter then as well.
        assert isinstance(
            model.nonlinear_solver_statistics, pp.NonlinearSolverStatistics
        )
        max_iterations = self.params.get("nl_max_iterations", 10)
        logger.info(
            "Newton iteration number "
            + f"{model.nonlinear_solver_statistics.num_iteration}"
            + f" of {max_iterations}"
        )
        nonlinear_increment_norm = (
            np.linalg.norm(nonlinear_increment) / nonlinear_increment.size
        )
        residual = model.equation_system.assemble(evaluate_jacobian=False)
        residual_norm = np.linalg.norm(residual) / residual.size
        logger.info(
            f"Nonlinear increment norm: {nonlinear_increment_norm:.2e}, "
            f"Nonlinear residual norm: {residual_norm:.2e}"
        )
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
            convergence_status: Convergence status of the solver.
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
