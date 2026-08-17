import logging
from dataclasses import dataclass
from time import time
from typing import Optional, cast
from warnings import warn

import numpy as np

import porepy as pp
from porepy.numerics.solvers.convergence_check import (
    ConvergenceCriteria,
    ConvergenceInfoCollection,
    ConvergenceStatus,
    ConvergenceStatusCollection,
    DivergenceCriteria,
    assemble_default_convergence_criteria,
    assemble_default_divergence_criteria,
    check_convergence,
)
from porepy.numerics.solvers.equation_variable_tags import EquationTag, VariableTag
from porepy.numerics.solvers.linear_solvers.linear_solver import (
    LinearSolverBase,
    LinearSolverDirect,
    LinearSolverStatus,
)
from porepy.numerics.solvers.nonlinear_solvers import (
    NonlinearSolverBase,
    NonlinearSolverStatus,
    NonlinearSolverStatusConverged,
    NonlinearSolverStatusFailed,
)
from porepy.utils.ui_and_logging import DummyProgressBar
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from porepy.utils.ui_and_logging import progressbar_class

__all__ = [
    "NewtonSolverConverged",
    "NewtonSolverFailed",
    "NewtonSolver",
]

# Module-wide logger
logger = logging.getLogger(__name__)

DEFAULT_NEWTON_MAX_ITERATIONS = 10
"""Default maximum number of Newton iterations."""


@dataclass
class NewtonSolverConverged(NonlinearSolverStatusConverged):
    linear_solver_statuses: list[LinearSolverStatus]

    def number_of_iterations(self) -> int:
        return len(self.linear_solver_statuses)


@dataclass
class NewtonSolverFailed(NonlinearSolverStatusFailed):
    linear_solver_statuses: list[LinearSolverStatus]

    def number_of_iterations(self) -> int:
        return len(self.linear_solver_statuses)


class NewtonSolver(NonlinearSolverBase):
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
        linear_solver: The linear solver object. If None (default), initializes a direct
            linear solver.
        equation_tags: List of tags that describes equations and domains this solver
            operates on. If None (default), it operates on all the equations in the
            model.
        variable_tags: List of tags that describes variables and domains this solver
            operates on. If None (default), it operates on all the variables in the
            model.

    If custom convergence or divergence criteria are provided, individual tolerance
    parameters should not be provided to avoid double specification. If no custom
    criteria are provided, default criteria are used based on the individual tolerance
    parameters and metric.

    """

    def __init__(
        self,
        params: Optional[dict] = None,
        is_nonlinear_problem: bool = True,
        linear_solver: Optional[LinearSolverBase] = None,
        equation_tags: Optional[list[EquationTag]] = None,
        variable_tags: Optional[list[VariableTag]] = None,
    ) -> None:
        if params is None:
            params = {}
        self.params = params
        """Dictionary of parameters for the nonlinear solver."""
        self.iteration_index: int = 0
        """Current iteration index - equivalent with number of iterations."""

        if linear_solver is None:
            linear_solver = LinearSolverDirect(backend="pypardiso")
        self.linear_solver: LinearSolverBase = linear_solver
        """Linear solver object to solve the Jacobian linear systems."""
        self._linear_solver_initialized = False
        """Whether model-dependent linear solver state has been initialized.
        Initialization is done once at :meth:`solve`. It is now assumed that this class
        must be used with the same model, and should not be reused for a different
        model.

        """
        self.solver_progressbar = DummyProgressBar()
        """The UI progress bar. By default, is a dummy object that does nothing.
        Reinitialized every Newton loop at :meth:`init_progress_bar`.

        """

        if equation_tags is None:
            equation_tags = []
        if variable_tags is None:
            variable_tags = []
        self.equation_tags: list[EquationTag] = equation_tags
        """List of tags that describes equations and domains this solver operates on.
        Empty list implies that it operates on all the equations in the model.

        """
        self.variable_tags: list[VariableTag] = variable_tags
        """List of tags that describes variables and domains this solver operates on.
        Empty list implies that it operates on all the variables in the model.

        """
        self._active_equations: Optional[list[pp.ad.EquationOnDomain]] = None
        """List of atomic equations this solver operates on. Initialized on first call
        of :meth:`solve`. Use :meth:`get_active_equations` to ensure you use initialized
        equations.

        """
        self._active_variables: Optional[list[pp.ad.Variable]] = None
        """List of atomic variables this solver operates on. Initialized on first call
        of :meth:`solve`. Use :meth:`get_active_variables` to ensure you use initialized
        variables.

        """

        self.init_convergence_criteria(is_nonlinear_problem=is_nonlinear_problem)
        self.init_divergence_criteria(is_nonlinear_problem=is_nonlinear_problem)

    def init_convergence_criteria(self, is_nonlinear_problem: bool) -> None:
        """Parse and initialize convergence criteria.

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

        Parameters:
            is_nonlinear_problem: Whether the underlying problem is nonlinear.

        """

        # Check for old parameter keys in self.params, replace them with new keys and
        # give a deprecation warning.
        for old_key in ["nl_convergence_tol_res", "nl_convergence_tol"]:
            old_to_new = {
                "nl_convergence_tol_res": "nl_convergence_res_atol",
                "nl_convergence_tol": "nl_convergence_inc_atol",
            }
            if old_key in self.params:
                new_key = old_to_new[old_key]
                logger.warning(
                    f"You are using a parameter name that has been changed: '{old_key}'"
                    f". Replace it with '{new_key}'. Currently replacing it "
                    "automatically, but it will not always be the case."
                )
                self.params[new_key] = self.params[old_key]

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
            convergence_criteria = assemble_default_convergence_criteria(
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

        Parameters:
            is_nonlinear_problem: Whether the underlying problem is nonlinear.

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

            # Fetch max iterations from the provided criteria, falling back on
            # the default.
            max_iterations = DEFAULT_NEWTON_MAX_ITERATIONS
            for c in self.params["nl_divergence_criteria"].values():
                if isinstance(c, pp.solvers.MaxIterationsCriterion):
                    max_iterations = c.max_iterations
        else:
            # Default parameters for divergence criteria.

            # If a user provides max_iteration for a linear problem, this value is
            # ignored, because the default divergence criterion does not include it.
            max_iterations = self.params.get(
                "nl_max_iterations", DEFAULT_NEWTON_MAX_ITERATIONS
            )
            divergence_criteria = assemble_default_divergence_criteria(
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

    def get_active_equations(
        self, model: pp.PorePyModel
    ) -> list[pp.ad.EquationOnDomain]:
        """Return active equations. If they are not initialized, initialize them."""
        if self._active_equations is None:
            # Lazy initialization.
            eq_indexer = model.equation_system.equation_indexer
            if len(self.equation_tags) > 0:
                self._active_equations, _ = eq_indexer.filter_by_tags(
                    self.equation_tags, model=model
                )
            else:
                # Empty equation_tags implies we use all equations.
                self._active_equations = list(eq_indexer.indices)
        return self._active_equations

    def get_active_variables(self, model: pp.PorePyModel) -> list[pp.ad.Variable]:
        """Return active variables. If they are not initialized, initialize them."""
        if self._active_variables is None:
            # Lazy initialization.
            var_indexer = model.equation_system.variable_indexer
            if len(self.variable_tags) > 0:
                self._active_variables, _ = var_indexer.filter_by_tags(
                    self.variable_tags, model=model
                )
            else:
                # Empty variable_tags implies we use all variables.
                self._active_variables = list(var_indexer.indices)
        return self._active_variables

    def increase_iteration_index(self) -> None:
        """Advance to the next iteration."""
        self.iteration_index += 1

    def solve(self, model: pp.PorePyModel) -> NonlinearSolverStatus:
        """Solve the nonlinear problem using the Newton-Raphson method.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            The status of the nonlinear solver.

        """
        # Model-dependent setup of a linear solver is done once.
        if not self._linear_solver_initialized:
            self.linear_solver.initialize_with_model(model)
            _deprecation_warning_assemble_linear_system(model)
            self._linear_solver_initialized = True

        # Prepare for nonlinear loop.
        self.before_nonlinear_loop(model)

        # Actual Newton loop.
        convergence_status, divergence_status, linear_solver_statuses = (
            self.nonlinear_loop(model)
        )

        # Summarizing the convergence message from multiple criteria into an overall
        # status.
        solver_status = _summarize_solver_status(
            convergence_status,
            divergence_status,
            linear_solver_statuses=linear_solver_statuses,
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

    def before_nonlinear_loop(self, model: pp.PorePyModel) -> None:
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
        self, model: pp.PorePyModel
    ) -> tuple[
        ConvergenceStatusCollection,
        ConvergenceStatusCollection,
        list[LinearSolverStatus],
    ]:
        """Perform the nonlinear loop (Newton iterations).

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            tuple[ConvergenceStatusCollection, ConvergenceStatusCollection]:
                Convergence and divergence status.

        """
        linear_solver_statuses: list[LinearSolverStatus] = []
        # Redirect all loggers to not interfere with the progressbar.
        with logging_redirect_tqdm([logging.root]):
            # Perform at least one Newton iteration.
            while True:
                # Prepare for nonlinear iteration.
                self.before_nonlinear_iteration(model)

                # Perform nonlinear iteration and obtain increment.
                nonlinear_increment, linear_solver_status = self.nonlinear_iteration(
                    model
                )
                linear_solver_statuses.append(linear_solver_status)

                # Finalize nonlinear iteration and determine status.
                convergence_status, divergence_status = self.after_nonlinear_iteration(
                    model, nonlinear_increment
                )

                # Exit the Newton loop.
                if convergence_status.is_converged() or divergence_status.is_failed():
                    break

        return convergence_status, divergence_status, linear_solver_statuses

    def after_nonlinear_loop(self) -> None:
        """Finalize the nonlinear loop."""
        # Close the progress bar.
        self.solver_progressbar.close()

    def before_nonlinear_iteration(self, model: pp.PorePyModel) -> None:
        """Prepare for a nonlinear iteration.

        Parameters:
            model: The model instance specifying the problem to be solved.

        """
        # Start iteration.
        self.increase_iteration_index()

        # Prepare model for a nonlinear iteration.
        model.before_nonlinear_iteration()

    def nonlinear_iteration(
        self, model: pp.PorePyModel
    ) -> tuple[np.ndarray, LinearSolverStatus]:
        """Perform a single nonlinear iteration.

        Right now, this is an almost trivial function. However, we keep it as a separate
        function to prepare for possible future introduction of more advanced schemes.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            np.ndarray: Solution to linearized system, i.e. the update increment.

        """
        nonlinear_increment, linear_solver_status = self.iteration(model)
        return nonlinear_increment, linear_solver_status

    def iteration(self, model: pp.PorePyModel) -> tuple[np.ndarray, LinearSolverStatus]:
        """A single linearization step.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            np.ndarray: Solution to linearized system, i.e. the update increment.

        """
        t_0 = time()

        active_equations = self.get_active_equations(model)
        active_variables = self.get_active_variables(model)

        linear_system = model.equation_system.assemble(
            equations=active_equations, variables=active_variables
        )
        logger.debug(f"Assembled linear system in {time() - t_0:.2e} seconds.")

        return self.linear_solver.solve_linear_system(linear_system)

    def after_nonlinear_iteration(
        self, model: pp.PorePyModel, nonlinear_increment: np.ndarray
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
        try:
            model.after_nonlinear_iteration(
                nonlinear_increment=nonlinear_increment,
                updated_variables=self.get_active_variables(model),
            )
        except ValueError:
            # model.after_nonlinear_iteration tends to raise this error if
            # discretization encounter nans:
            # "Tensor is not positive definite because of components in x-direction"
            # If not intercepted here, this exception leads to a simulation crash
            # instead of retrying.
            return ConvergenceStatusCollection(), ConvergenceStatusCollection(
                {"failed to rediscretize model": ConvergenceStatus.FAILED}
            )

        # Monitor convergence.
        convergence_status, divergence_status, convergence_info = check_convergence(
            convergence_criteria=self.convergence_criteria,
            divergence_criteria=self.divergence_criteria,
            nonlinear_increment=nonlinear_increment,
            solution=model.equation_system.get_variable_values(
                variables=self.get_active_variables(model), iterate_index=0
            ),
            residual=model.equation_system.assemble(
                evaluate_jacobian=False, equations=self.get_active_equations(model)
            ),
            iteration_index=self.iteration_index,
        )

        # Logging and progress bar update.
        self.logging(convergence_info)

        # Update (iteration-based) solver statistics.
        self.update_solver_statistics(
            model,
            convergence_status=convergence_status.union(divergence_status),
            convergence_info=convergence_info,
        )

        return convergence_status, divergence_status

    def check_convergence(
        self,
        model: pp.PorePyModel,
        nonlinear_increment: np.ndarray,
    ) -> tuple[
        ConvergenceStatusCollection,
        ConvergenceStatusCollection,
        ConvergenceInfoCollection,
    ]:
        """Check convergence and divergence based on passed criteria.

        Parameters:
            model: The model instance specifying the problem to be solved.
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

    def logging(self, convergence_info: dict[str, dict | float]) -> None:
        """Log the current state of the nonlinear solver.

        This includes printing the current iteration number, nonlinear increment norm,
        and residual norm, as well as updating the progress bar. The norms are
        considered only if they are computed by convergence criteria.

        Parameters:
            convergence_info: Convergence information possibly containing norms.

        """
        progressbar_string = ""
        inc_abs = convergence_info.get("inc_abs", None)
        res_abs = convergence_info.get("res_abs", None)
        if inc_abs is not None and isinstance(inc_abs, str):
            progressbar_string = f"{progressbar_string} {inc_abs=:.2e}"
        if res_abs is not None and isinstance(res_abs, str):
            progressbar_string = f"{progressbar_string} {res_abs=:.2e}"

        logger.info(
            f"Iter {self.iteration_index}/{self.max_iterations}. {progressbar_string}"
        )

        # Update progress bar.
        self.solver_progressbar.update(n=1)
        self.solver_progressbar.set_postfix_str(progressbar_string)

    def update_solver_statistics(
        self,
        model: pp.PorePyModel,
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


def _summarize_solver_status(
    convergence_status: ConvergenceStatusCollection,
    divergence_status: ConvergenceStatusCollection,
    linear_solver_statuses: list[LinearSolverStatus],
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
        return NewtonSolverConverged(
            linear_solver_statuses=linear_solver_statuses,
            convergence_statuses=convergence_status,
            divergence_statuses=divergence_status,
        )
    elif is_failed:
        logger.warning("Failed to solve the nonlinear problem.")
        return NewtonSolverFailed(
            linear_solver_statuses=linear_solver_statuses,
            convergence_statuses=convergence_status,
            divergence_statuses=divergence_status,
        )
    else:
        logger.error(
            "Nonlinear solver did not fail, but the convergence criterion did not "
            "accept the solution. Treating it as a failure."
        )
        return NewtonSolverFailed(
            linear_solver_statuses=linear_solver_statuses,
            convergence_statuses=convergence_status,
            divergence_statuses=divergence_status,
        )


def _deprecation_warning_assemble_linear_system(model: pp.PorePyModel):
    assemble_linear_system = getattr(model, "assemble_linear_system", None)
    if assemble_linear_system is not None:
        implementation = getattr(
            assemble_linear_system, "__func__", assemble_linear_system
        )
        if implementation is not pp.SolutionStrategy.assemble_linear_system:
            warn(
                "The model overrides assemble_linear_system method, but NewtonSolver no"
                " longer calls it.",
                category=FutureWarning,
                stacklevel=3,
            )


def _update_solver_statistics_after_nonlinear_solve(
    model: pp.PorePyModel,
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
