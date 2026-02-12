"""This module contains functions to run stationary and time-dependent models."""

from __future__ import annotations

import logging
import warnings
from typing import Optional, TypeVar

import numpy as np

import porepy as pp
from porepy.utils.ui_and_logging import DummyProgressBar, progressbar_class
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)

__all__ = ["ModelRunner", "ModelInstance"]

# Module-wide logger
logger = logging.getLogger(__name__)

ModelInstance = TypeVar("ModelInstance", bound=pp.PorePyModel)
"""Type variable for objects inheriting from the PorePy model protocol."""


def run_stationary_model(model, params: dict) -> None:
    """Run a stationary model.

    Deprecated: This function is deprecated and will be removed in a future version.
        Instead, use
        ```
        runner = pp.ModelRunner(model, params)
        runner.run()
        ```

    Note:
        If the ``"progressbars"`` key in ``params`` is set to ``True`` (default is
        ``False``), the progress of nonlinear iterations will be shown on a progressbar.
        This requires the ``tqdm`` package to be installed. The package is not included
        in the dependencies, but can be installed with
        ```
        pip install tqdm
        ```

    Parameters:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate model for documentation.
        params: Parameters related to the solution procedure.

    """
    warnings.deprecated(
        "run_stationary_model is deprecated in favor of ModelRunner.run and will be"
        + " removed in future versions."
    )
    runner = ModelRunner(model, params)
    runner.run()


def run_time_dependent_model(model, params: Optional[dict] = None) -> None:
    """Run a time dependent model.

    Deprecated: This function is deprecated and will be removed in a future version.
        Instead, use
        ```
        runner = pp.ModelRunner(model, params)
        runner.run()
        ```

    Note:
        If the ``"progressbars"`` key in ``params`` is set to ``True`` (default is
        ``False``), the progress of time steps and nonlinear iterations will be shown on
        a progressbar. This requires the ``tqdm`` package to be installed. The package
        is not included in the dependencies, but can be installed with
        ```
        pip install tqdm
        ```

    Parameters:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate solver for documentation.
        params: Parameters related to the solution procedure.

    """
    warnings.deprecated(
        "run_time_dependent_model is deprecated in favor of ModelRunner.run and will be"
        + " removed in future versions."
    )
    runner = ModelRunner(model, params)
    runner.run()


class ModelRunner:
    """Class for running PorePy models according to their configurations.

    Sets the outer solver, linear or nonlinear, depending on `model.is_nonlinear`. In
    the nonlinear case the solver can be customized by providing a solver type as
    ``params["nonlinear_solver"]``.

    If ``params["prepare_simulation"]`` is ``True`` (default), calls the respective
    method during initialization. Otherwise it assumes it was already called **before**
    instantiating the runner.


    :meth:`~ModelRunner.run` runs the simulation, stationary or time dependent,
    depending on ``model.is_time_dependent.`

    Parameters:
        model: A PorePy model instance.
        params: Parameters related to the solution procedure. Defaults to None.

    """

    def __init__(self, model: ModelInstance, params: dict | None = None) -> None:
        self.params = params if isinstance(params, dict) else {}
        """Parameters passed at instantiation."""

        self.model = model
        """Model instance passed at instantiation."""

        self.solver: pp.NewtonSolver | pp.LinearSolver
        """Solver instance, set in :meth:`set_solver`."""

        if self.params.get("prepare_simulation", True):
            self.model.prepare_simulation()

        self._is_nonlinear = self.model.is_nonlinear_problem()
        """Flag indicating whether the problem is nonlinear, set at initialization."""

        self._is_time_dependent = self.model.is_time_dependent()
        """Flag indicating whether the problem is time-dependent, set at
        initialization."""

        self.set_solver()

        self.init_time_progressbar()

    def set_solver(self) -> None:
        """Choose between linear and non-linear solver and set :attr:`solver`.

        Custom nonlinear solvers can be used by providing a solver type
        as ``params["nonlinear_solver"]``. The default nonlinear solver is
        :class:`~porepy.numerics.nonlinear.nonlinear_solvers.NewtonSolver`.

        If the model is linear, sets :attr:`solver` to an instance of
        :class:`~porepy.numerics.linear_solvers.LinearSolver`.

        """
        if self._is_nonlinear:
            self.solver = self.params.get("nonlinear_solver", pp.NewtonSolver)(
                self.params
            )
        else:
            self.solver = pp.LinearSolver(self.params)

    def init_time_progressbar(self) -> None:
        """Initializes the a progressbar for logging according to
        ``params["progressbars"]``.

        Note:
            If the ``"progressbars"`` key in ``params`` is set to ``True`` (default is
            ``False``), the progress of time steps and nonlinear iterations will be
            shown on a progressbar. This requires the ``tqdm`` package to be installed.
            The package is not included in the dependencies, but can be installed with
            ```
            pip install tqdm
            ```

        """
        # Use time progressbar only when requested and the model is time dependent.
        use_progress_bar = (
            self.params.get("progressbars", False) and self._is_time_dependent
        )
        if use_progress_bar and progressbar_class is DummyProgressBar:
            logger.warning(
                "Progress bars are requested, but `tqdm` is not installed. The time"
                " loop will run without progress bars."
            )
        # To display nested ``tqdm`` bars in the correct order, their positions have to
        # be specified. The orders are increasing, i.e., 0 is the lowest level, then 1,
        # etc.
        # When ``NewtonSolver`` is called inside ``run``, the
        # ``_nl_progress_bar_position`` parameter specifying the position of the
        # ``NewtonSolver`` progressbar is passed.
        self.params.update({"_nl_progress_bar_position": 1})

        # Check if the user wants a progress bar. Initialize an instance of the
        # progressbar_class, which is either :class:`~tqdm.trange` or
        # :class:`~DummyProgressbar` in case `tqdm` is not installed.
        if use_progress_bar:
            # Initial time step size, used for progress bar updates.
            self._dt_0: float = self.model.time_manager.dt

            # Create a time bar. The length is estimated as the time_steps predetermined
            # by the schedule and initial time step size.
            # NOTE: If, e.g., adaptive time stepping results in more time steps, the
            # time bar will increase with partial steps corresponding to the ratio of
            # the modified time step size to the initial time step size.
            expected_time_steps: int = int(
                np.round(
                    (
                        self.model.time_manager.schedule[-1]
                        - self.model.time_manager.schedule[0]
                    )
                    / self._dt_0
                )
            )
            self.time_progressbar = progressbar_class(
                range(expected_time_steps),
                desc="time loop",
                position=0,
                dynamic_ncols=True,
            )
        # Otherwise, use a dummy progress bar.
        else:
            self.time_progressbar = DummyProgressBar()

    def run(self, *args, **kwargs) -> None:
        """Runs the model as specified."""

        if self._is_time_dependent:
            # Redirect the root logger, to avoid logger-progressbars interference.
            with logging_redirect_tqdm([logging.root]):
                # Time loop.
                while not self.model.time_manager.final_time_reached():
                    self.before_time_step()
                    time_step_converged = self.solver.solve(self.model)
                    self.after_time_step(time_step_converged)
        else:
            converged = self.solver.solve(self.model)
            if converged:
                # NOTE: time_step_convergence can be considered a misnomer.
                # But technically this is the only time we solve for. Thus we reuse the
                # method to set the solution and save data.
                self.model.after_time_step_convergence()
            else:
                raise RuntimeError("Stationary model did not converge.")

        self.model.after_simulation()

    def before_time_step(self) -> None:
        """Method to be executed at the beginning of each time step.

        Increases the time and sets the model's AD time step value.
        Executes :meth:`~porepy.models.solution_strategy.ModelSolverInterface.
        before_time_step` and logs the progress.

        """
        # Increase the simulation time.
        self.model.time_manager.increase_time()
        self.model.time_manager.increase_time_index()
        # Update the model's AD time step object.
        self.model.ad_time_step.set_value(self.model.time_manager.dt)
        # Prepare model.
        self.model.before_time_step()

        # Logging and progressbar update.
        logger.info(
            f"\nTime step {self.model.time_manager.time_index} at time"
            + f" {self.model.time_manager.time:.1e}"
            + f" of {self.model.time_manager.time_final:.1e}"
            + f" with time step {self.model.time_manager.dt:.1e}"
        )
        self.time_progressbar.set_description_str(
            f"Time step {self.model.time_manager.time_index + 1}"  # Why +1? Consistent?
        )

    def after_time_step(self, time_step_converged: bool) -> None:
        if time_step_converged:
            # Update the time step magnitude if the dynamic scheme is used.
            if not self.model.time_manager.is_constant:
                self.model.time_manager.compute_time_step(
                    iterations=self.model.nonlinear_solver_statistics.num_iteration
                )
            self.model.after_time_step_convergence()

            # Update progressbar length.
            self.time_progressbar.update(n=self.model.time_manager.dt / self._dt_0)
        else:
            if self.model.time_manager.is_constant:
                raise pp.TimeSteppingError(
                    "Solver failed to converge with constant time step size."
                )
            else:
                # This calls
                # ``time_manager._adaptation_based_on_recomputation``, which substracts
                # the current ``dt`` from the simulation time, computes a shorter
                # ``dt``, and adds the updated ``dt`` to the simulation time again.
                # It will also raise a ValueError if the minimal time step is reached.
                self.model.time_manager.compute_time_step(recompute_solution=True)
                self.model.after_time_step_failure()
