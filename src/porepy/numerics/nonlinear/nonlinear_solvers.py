"""
Nonlinear solvers to be used with model classes.
Implemented classes
    NewtonSolver
"""

import logging

import numpy as np

from porepy.models.model_runner import ModelInstance
from porepy.numerics.linear_solvers import LinearSolver
from porepy.utils.ui_and_logging import DummyProgressBar
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from porepy.utils.ui_and_logging import progressbar_class

# Module-wide logger
logger = logging.getLogger(__name__)


class NewtonSolver(LinearSolver):
    """Standard Newton solver for nonlinear problems.

    Performs iterations until convergence or divergence is detected, or the
    maximum number of iterations is reached.

    For more information on parametrization, see :attr:`params`

    """

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)

        default_params = {
            "max_iterations": 10,
            "nl_convergence_tol": 1e-10,
            "nl_convergence_tol_res": np.inf,
            "nl_divergence_tol": np.inf,
        }
        default_params.update(self.params)

        self.params = default_params
        """The Newton solver supports the following parameters:
        
        - ``max_iterations``: Maximum number of iterations before declaring failure.
          Default is 10.
        - ``nl_convergence_tol``: Tolerance for convergence based on the norm of the
          nonlinear increment. Default is ``1e-10``.
        - ``nl_convergence_tol_res``: Tolerance for convergence based on the norm of the
          residual. Default is ``np.inf``, i.e., not used.
        - ``nl_divergence_tol``: Tolerance for divergence based on the norm of the
          nonlinear increment. Default is ``np.inf``, i.e., not used.

        """

        self.progress_bar: bool = bool(self.params.get("progressbars", False))
        if self.progress_bar and progressbar_class is DummyProgressBar:
            logger.warning(
                "Progress bars are requested, but `tqdm` is not installed. The solver"
                " will run without progress bars."
            )
        # Allow the position of the progress bar to be flexible, depending on whether
        # this is called inside a time loop, a time loop and an additional propagation
        # loop or inside a stationary problem (default).
        self.progress_bar_position: int = int(
            self.params.get("_nl_progress_bar_position", 0)
        )

    def solve(self, model: ModelInstance) -> bool:
        """Solve the nonlinear problem.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            A 2-tuple containing:

            bool:
                True if the solution is converged.

        """
        # Empty the log in the model's statistics object.
        model.nonlinear_solver_statistics.reset()
        # Any model bookkeeping that has to happen before a nonlinear loop.
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
            # Bind to variables in the outer function.
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

            model.before_solver_iteration()
            nonlinear_increment = self.iteration(model)
            model.after_solver_iteration(nonlinear_increment)
            model.nonlinear_solver_statistics.num_iteration += 1

            if (
                self.params["nl_convergence_tol_res"] is not np.inf
                or self.params["nl_divergence_tol"] is not np.inf
            ):
                # Note: The residual is extracted after the solution has been updated by
                # the after_solver_iteration() method. This is required if the
                # residual is used to check convergence or divergence, i.e., the
                # tolerance of one of them is not np.inf.
                residual = model.equation_system.assemble(evaluate_jacobian=False)
            else:
                residual = None

            is_converged, is_diverged = model.check_convergence(
                nonlinear_increment, residual, reference_residual, self.params
            )

        # Redirect the root logger, to avoid logger-progressbars interference.
        with logging_redirect_tqdm([logging.root]):
            # Check if the user wants a progress bar. Initialize an instance of the
            # progressbar_class, which is either :class:`~tqdm.trange` or
            # :class:`~DummyProgressbar` in case `tqdm` is not installed.
            if self.progress_bar:
                # Length is the maximal number of Newton iterations.
                solver_progressbar = progressbar_class(  # type: ignore
                    range(int(self.params["max_iterations"])),
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
                if not is_diverged:
                    solver_progressbar.update(n=1)
                    norms = model.nonlinear_solver_statistics.nonlinear_increment_norms
                    # Ignore the long line; fixing it would require an extra variable.
                    if len(norms) != 0:
                        solver_progressbar.set_postfix_str(
                            f"Increment {norms[-1]:.2e}"  # noqa: E501
                        )

                # Sanity check for convergence criteria.
                if is_diverged and is_converged:
                    raise RuntimeError(
                        "The solution cannot be both converged and diverged."
                    )
                if is_diverged or is_converged:
                    break

        solver_progressbar.close()
        if is_converged:
            model.after_solver_convergence()
        if is_diverged or not is_converged:  # Covers max iterations reached.
            model.after_solver_failure()

        return is_converged

    def iteration(self, model: ModelInstance) -> np.ndarray:
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
