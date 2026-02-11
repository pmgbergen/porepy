"""This module contains functions to run stationary and time-dependent models."""

from __future__ import annotations

import abc
import logging
from typing import Optional, TypeVar

import numpy as np

import porepy as pp
from porepy.utils.ui_and_logging import DummyProgressBar, progressbar_class
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)

# Module-wide logger
logger = logging.getLogger(__name__)

ModelInstance = TypeVar("ModelInstance", bound=pp.PorePyModel)
"""Type variable for objects inheriting from the PorePy model protocol."""


class ModelRunner(abc.ABC):
    """Abstract base class for running a porepy model.

    Sets the model and solver, and defines the abstract method ``run`` to be implemented
    by subclasses.

    Executes :meth:`~porepy.models.solution_strategy.ModelSolverInterface.
    prepare_simulation` if ``params["prepare_simulation"]`` is ``True`` (default).

    Parameters:
        model: A PorePy model instance.
        params: Parameters related to the solution procedure. Defaults to None.

    """

    def __init__(self, model: ModelInstance, params: Optional[dict] = None) -> None:
        self.params = params if isinstance(params, dict) else {}
        """Parameters passed at instantiation."""

        self.model = model
        """Model instance passed at instantiation."""

        self.solver: pp.NewtonSolver | pp.LinearSolver
        """Solver instance, set in :meth:`set_solver`."""

        self.set_solver()

        if self.params.get("prepare_simulation", True):
            self.model.prepare_simulation()

    def set_solver(self) -> None:
        """Choose between linear and non-linear solver and set :attr:`solver`.

        Custom nonlinear solvers can be used by providing a solver type
        as ``params["nonlinear_solver"]``. The default nonlinear solver is
        :class:`~porepy.numerics.nonlinear.nonlinear_solvers.NewtonSolver`.

        If the model is linear, sets :attr:`solver` to an instance of
        :class:`~porepy.numerics.linear_solvers.LinearSolver`.

        """
        if self.model.is_nonlinear_problem():
            self.solver = self.params.get("nonlinear_solver", pp.NewtonSolver)(
                self.params
            )
        else:
            self.solver = pp.LinearSolver(self.params)

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> None:
        """Abstract method to run the model. To be implemented by subclasses."""


class StationaryModelRunner(ModelRunner):
    """Runner for time-independent models."""

    def run(self, *args, **kwargs) -> None:
        """Calls the solver once and executes
        :meth:`~porepy.models.solution_strategy.ModelSolverInterface.after_simulation`.
        """
        self.solver.solve(self.model)
        self.model.after_simulation()


class TimeDependentModelRunner(ModelRunner):
    """Runner for time-dependent models."""

    def __init__(self, model: ModelInstance, params: dict | None = None) -> None:
        super().__init__(model, params)

        self._dt_0: float = model.time_manager.dt
        """Initial time step size, used for progress bar updates."""

        self.time_step_converged: bool = False
        """Flag set after each solver call indicating convergence."""

        self.init_time_progressbar()

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
        use_progress_bar = self.params.get("progressbars", False)
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
        if self.params.get("progressbars", False):
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

    def after_time_step(self) -> None:
        if self.time_step_converged:
            # Update the time step magnitude if the dynamic scheme is used.
            if not self.model.time_manager.is_constant:
                self.model.time_manager.compute_time_step(
                    iterations=self.model.nonlinear_solver_statistics.num_iteration
                )

            # Update progressbar length.
            self.time_progressbar.update(n=self.model.time_manager.dt / self._dt_0)

    def run(self, *args, **kwargs) -> None:
        """Run a time dependent model.

        Executes :meth:`before_time_step`, the solver and :meth:`after_time_step` until
        final time is reached.
        Executes :meth:`~porepy.models.solution_strategy.ModelSolverInterface.
        after_simulation` at the end of the simulation.

        """
        # Redirect the root logger, to avoid logger-progressbars interference.
        with logging_redirect_tqdm([logging.root]):
            # Time loop.
            while not self.model.time_manager.final_time_reached():
                self.before_time_step()
                self.time_step_converged = self.solver.solve(self.model)
                self.after_time_step()

        self.model.after_simulation()
