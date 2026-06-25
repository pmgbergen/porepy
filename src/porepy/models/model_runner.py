"""This module contains functions to run stationary and time-dependent models."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
import logging
import warnings
from typing import Optional

import numpy as np

from porepy.time.time_stepper import TimeStepper
from porepy.utils.ui_and_logging import DummyProgressBar
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from porepy.utils.ui_and_logging import progressbar_class
from porepy.time.time_step_status import TimeStepperStatusFailure
from porepy.models.solution_strategy import SolutionStrategy
import porepy as pp

__all__ = ["ModelRunner"]

# Module-wide logger
logger = logging.getLogger(__name__)


@dataclass
class ModelRunnerStatus(ABC):
    """A status object used to indicate the ModelRunner state."""

    def is_success(self) -> bool:
        """Whether the simulation finished successfully."""
        return isinstance(self, ModelRunnerStatusSuccess)

    def is_failure(self) -> bool:
        """Whether the simulation finished with a failure."""
        return isinstance(self, ModelRunnerStatusFailure)


@dataclass
class ModelRunnerStatusSuccess(ModelRunnerStatus):
    """A status object that indicates that the simulation finished successfully."""


@dataclass
class ModelRunnerStatusFailure(ModelRunnerStatus):
    """A status object that indicates that the simulation finished with a failure."""

    reason: str
    "Reason why the model runner failed."


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
        " removed in future versions."
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
        " removed in future versions."
    )
    runner = ModelRunner(model, params)
    runner.run()


class ModelRunner:
    """Class for running PorePy models according to their configurations.

    Sets the outer solver, linear or nonlinear, depending on `model._is_nonlinear`. In
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

    def __init__(
        self,
        model: SolutionStrategy,
        params: Optional[dict] = None,
        time_stepper: Optional[TimeStepper] = None,
    ) -> None:
        self.params = params if isinstance(params, dict) else {}
        """Parameters passed at instantiation."""

        self.model = model
        """Model instance passed at instantiation."""

        self.solver: pp.NewtonSolver | pp.LinearSolver
        """Solver instance, set in :meth:`set_solver`."""

        # Construct the default if not provided. This time stepper is constructed even
        # for a stationary problem, but used only for time-dependent problems.
        if time_stepper is None:
            time_stepper = TimeStepper(time_manager=model.time_manager)
        self.time_stepper: TimeStepper = time_stepper
        """Responsible for the time stepping logic."""

        if self.params.get("prepare_simulation", True):
            self.model.prepare_simulation()

        self._is_nonlinear = self.model._is_nonlinear_problem()
        """Flag indicating whether the problem is nonlinear, set at initialization."""

        self._is_time_dependent = self.model._is_time_dependent()
        """Flag indicating whether the problem is time-dependent, set at
        initialization."""

        self.set_solver()

        if self._is_time_dependent:
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
        # be specified. The orders are increasing, i.e., 0 is the lowest level, then 1.
        # Position is passed via '_nl_progress_bar_position' when calling 'NewtonSolver'
        # in ``run``.
        self.params.update({"_nl_progress_bar_position": 1})

        if use_progress_bar:
            # Create a time bar of length of expected number of time steps, estimated
            # from the initial time step size.
            # NOTE: Adaptive time stepping updates the bar proportionally if step sizes
            # change from initial time step size.
            expected_time_steps: int = int(
                np.round(
                    (
                        self.model.time_manager.schedule[-1]
                        - self.model.time_manager.schedule[0]
                    )
                    / self.model.time_manager.dt
                )
            )

            # NOTE: If tqdm is not installed, this returns a DummyProgressBar instance.
            self.time_progressbar = progressbar_class(
                range(expected_time_steps),
                desc="Time loop",
                position=0,
                dynamic_ncols=True,
            )
        else:
            self.time_progressbar = DummyProgressBar()

        self.use_progress_bar = use_progress_bar

    def run(self) -> ModelRunnerStatus:
        """Run the model (stationary or time-dependent)."""
        # Run simulation.
        if self._is_time_dependent:
            simulation_status = self._run_time_dependent()
        else:
            simulation_status = self._run_stationary()

        # Clean up model after simulation.
        self.model.after_simulation()

        return simulation_status

    def _run_stationary(self) -> ModelRunnerStatus:
        """Run a stationary model."""
        # Perform stationary solve.
        convergence_status = self.solver.solve(self.model)

        # Conclude the simulation status based on the solver status.
        if convergence_status.is_converged():
            # NOTE: time_step_convergence can be considered a misnomer.
            # But technically this is the only time we solve for. Thus we reuse the
            # method to set the solution and save data.
            self.model.after_time_step_convergence()
            return ModelRunnerStatusSuccess()
        else:
            self.model.after_time_step_failure()
            return ModelRunnerStatusFailure("Solver did not converge.")

    def _run_time_dependent(self) -> ModelRunnerStatus:
        """Run a time-dependent model with trial-based time stepping."""

        with logging_redirect_tqdm([logging.root]):
            while not self.model.time_manager.final_time_reached():
                # Perform the time step.
                time_step_status = self.time_stepper.perform_time_step(
                    self.model, self.solver
                )

                # Progressbar update.
                self.update_time_progressbar()

                # Abort simulation if time step was stopped.
                match time_step_status:
                    case TimeStepperStatusFailure(nonlinear_solver_status, reason):
                        logger.error(f"Time stepping failed: {reason}")
                        return ModelRunnerStatusFailure(reason=reason)

            # Conclude the simulation status.
            if self.model.time_manager.final_time_reached():
                return ModelRunnerStatusSuccess()
            return ModelRunnerStatusFailure("Final time was not reached.")

    def update_time_progressbar(self) -> None:
        """Update the time progressbar with the current time and time step size."""
        self.time_progressbar.set_postfix_str(
            f"Time step size {self.model.time_manager.dt:.2e}"
        )
        self.time_progressbar.update(
            n=np.round(
                self.model.time_manager.time
                / self.model.time_manager.time_final
                * self.time_progressbar.total
            )
        )
