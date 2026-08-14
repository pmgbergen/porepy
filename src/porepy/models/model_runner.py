"""This module contains functions to run stationary and time-dependent models."""

from __future__ import annotations

import logging
import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Optional, cast
import numpy as np

import porepy as pp
from porepy.models.solution_strategy import SolutionStrategy
from porepy.time_stepper.time_step_status import (
    TimeStepperStatusFailure,
    TimeStepperStatusSuccess,
)
from porepy.time_stepper.time_stepper import TimeStepper
from porepy.utils.ui_and_logging import DummyProgressBar
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from porepy.utils.ui_and_logging import progressbar_class

__all__ = [
    "ModelRunnerStatus",
    "ModelRunnerStatusSuccess",
    "ModelRunnerStatusFailure",
    "ModelRunner",
]

# Module-wide logger
logger = logging.getLogger(__name__)


@dataclass
class ModelRunnerStatus(ABC):
    """A status object used to indicate the ModelRunner state. This is an enum of two
    allowed states: either success or failure. Each state can have data associated with
    it. `ModelRunnerStatusSuccess` and `ModelRunnerStatusFailure` can be subclassed to
    (i) introduce specific cases of success or failure and (ii) associate additional
    data with these cases. The base class `ModelRunnerStatus` should NOT be subclassed.

    """

    def is_success(self) -> bool:
        """Whether the simulation finished successfully."""
        # Developer note: This breaks the OOP principle that the base class should not
        # know of its children, but we agreed on having these methods (is_success and
        # is_failure) for convenience. One can think of ModelRunnerStatus as a closed
        # enum of two cases (success and failure), which in this case justifies this
        # binding with child classes.
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

    If ``params["prepare_simulation"]`` is ``True`` (default), calls the respective
    method during initialization. Otherwise it assumes it was already called **before**
    instantiating the runner.

    :meth:`~ModelRunner.run` runs the simulation, stationary or time dependent,
    depending on ``model.is_time_dependent.`

    Parameters:
        model: A PorePy model instance.
        params: Parameters related to the solution procedure. Defaults to None.
        time_stepper: The object corresponding for making a single time step. Passing
            None (default) initializes the default PorePy time stepper.
        nonlinear_solver: The solver for the discretized problem described by the PorePy
            model). Passing None (default) initializes the default nonlinear solver with
            a default set of convergence / divergence criteria. The default solver
            may exploit model's linearity and apply shortcuts for it (e.g. avoid
            expensive convergence checks). The reverse is also true: You can pass a
            customized solver that treats the problem as linear even if it is not.

    """

    def __init__(
        self,
        model: SolutionStrategy,
        params: Optional[dict] = None,
        time_stepper: Optional[TimeStepper] = None,
        nonlinear_solver: Optional[pp.solvers.NonlinearSolverBase] = None,
    ) -> None:
        self.params = params if isinstance(params, dict) else {}
        """Parameters passed at instantiation."""

        self.model = model
        """Model instance passed at instantiation."""

        if self.params.get("prepare_simulation", True):
            self.model.prepare_simulation()

        self._is_nonlinear = self.model._is_nonlinear_problem()
        """Flag indicating whether the problem is nonlinear, set at initialization."""

        self._is_time_dependent = self.model._is_time_dependent()
        """Flag indicating whether the problem is time-dependent, set at
        initialization."""

        self.time_stepper: TimeStepper = _extract_time_stepper_from_params(
            time_stepper=time_stepper,
            model_params=model.params,
            is_time_dependent=self._is_time_dependent,
        )
        """Responsible for the time stepping logic. Used only in time-dependent
        simulations."""

        self.solver: pp.solvers.NonlinearSolverBase = (
            _extract_nonlinear_solver_from_params(
                nonlinear_solver=nonlinear_solver,
                params=self.params,
                is_nonlinear_problem=self._is_nonlinear,
            )
        )
        """Solver instance."""

        if self._is_time_dependent:
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
            scheduler = self.time_stepper.scheduler
            # Create a time bar of length of expected simulation time.

            # Creating a custom format string. Difference from the default: it converts
            # the simulation time (done/total) to the a scientific format with :.1e.
            l_bar = "{desc}: {percentage:3.0f}%|"
            r_bar = "| {n:.1e}/{total:.1e} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            bar_format = l_bar + "{bar}" + r_bar

            # NOTE: If tqdm is not installed, this returns a DummyProgressBar instance.
            self.time_progressbar = progressbar_class(
                total=scheduler.get_time_end(),
                desc="Time loop",
                position=0,
                dynamic_ncols=True,
                bar_format=bar_format,
                postfix=self._progressbar_postfix(),
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

        if simulation_status.is_failure():
            raise RuntimeError(simulation_status)

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
            while not self.time_stepper.scheduler.is_finished():
                # Update the progressbar before the time step.
                self.time_progressbar.set_postfix_str(self._progressbar_postfix())

                # Perform the time step.
                time_step_status = self.time_stepper.perform_time_step(
                    self.model, self.solver
                )

                # Update the progressbar after the time step.
                if isinstance(time_step_status, TimeStepperStatusSuccess):
                    self.time_progressbar.update(n=time_step_status.dt)

                # Abort simulation if time step was stopped.
                if isinstance(time_step_status, TimeStepperStatusFailure):
                    logger.error(f"Time stepping failed: {time_step_status.reason}")
                    return ModelRunnerStatusFailure(reason=time_step_status.reason)

            # Conclude the simulation status.
            if self.time_stepper.scheduler.is_finished():
                return ModelRunnerStatusSuccess()
            return ModelRunnerStatusFailure("Final time was not reached.")

    def _progressbar_postfix(self) -> str:
        """Formats a progressbar postfix string with dt."""
        assert self.time_stepper is not None
        return f"Δt={self.time_stepper.scheduler.get_dt():.1e}"


def _extract_nonlinear_solver_from_params(
    nonlinear_solver: Optional[pp.solvers.NonlinearSolverBase],
    params: dict,
    is_nonlinear_problem: bool,
) -> pp.solvers.NonlinearSolverBase:
    """A nonlinear solver may be passed directly or in the parameters dictionary. This
    function extracts it and ensures it is not passed twice. If nothing is passed, it
    constructs a default solver.

    Parameters:
        nonlinear_solver: A nonlinear solver from user.
        params: A dictionary that may contain a key "nonlinear_solver".
        is_nonlinear_solver: Used to construct a default solver.

    Raises:
        ValueError: If two nonlinear solvers are passed both directly and through
            params.

    Returns:
        An instantiated nonlinear solver object.

    """
    solver_from_params = params.get("nonlinear_solver", None)
    if solver_from_params is not None and nonlinear_solver is None:
        warnings.warn(
            "You should pass the nonlinear solver directly to the ModelRunner: use "
            "ModelRunner(nonlinear_solver=...). Passing it through params will be "
            "deprecated.",
            category=FutureWarning,
            stacklevel=3,
        )
        return cast(type[pp.solvers.NewtonSolver], solver_from_params)(params)
    if solver_from_params is not None and nonlinear_solver is not None:
        raise ValueError(
            "You cannot pass the nonlinear solver both directly to the ModelRunner and "
            "through params."
        )

    if nonlinear_solver is None:
        return pp.solvers.NewtonSolver(
            is_nonlinear_problem=is_nonlinear_problem, params=params
        )
    else:
        return nonlinear_solver


def _extract_time_stepper_from_params(
    time_stepper: pp.time_stepper.TimeStepper | None,
    model_params: dict,
    is_time_dependent: bool,
) -> pp.time_stepper.TimeStepper:
    time_manager = model_params.get("time_manager", None)
    # Return early if time_stepper is provided.
    if time_stepper is not None:
        if time_manager is not None:
            warnings.warn(
                "TimeStepper is passed to ModelRunner, and TimeManager is found in "
                "model.params. The latter is deprecated and will be ignored. Remove it "
                "from model.params.",
                category=FutureWarning,
                stacklevel=3,
            )
        return time_stepper

    # Construct time_stepper based on time_manager.
    if time_manager is not None:
        warnings.warn(
            "model.params['time_manager'] is deprecated and will be removed. Set the "
            "simulation schedule by using "
            "TimeStepper(scheduler=pp.time_stepper.assemble_default_time_scheduler())",
            category=FutureWarning,
            stacklevel=3,
        )
        assert isinstance(time_manager, pp.TimeManager)
        if time_manager.dt_min_max is not None:
            dt_min = float(time_manager.dt_min_max[0])
            dt_max = float(time_manager.dt_min_max[1])
        else:
            dt_min = dt_max = None
        return TimeStepper(
            scheduler=pp.time_stepper.assemble_default_time_scheduler(
                schedule=time_manager.schedule,
                dt_init=time_manager.dt_init,
                constant_dt=time_manager.is_constant,
                dt_min=dt_min,
                dt_max=dt_max,
                nonlinear_iter_optimal_range=time_manager.iter_optimal_range,
                nonlinear_iter_relax_factors=time_manager.iter_relax_factors,
                nonlinear_iter_retry_factor=time_manager.recomp_factor,
            )
        )

    if is_time_dependent:
        # The user has not provided neither time_manager nor time_stepper. We are either
        # (i) in a unit test that does not care about time, or (ii) the user forgot it.
        # Printing a warning and initializing default. In case (i), the warning will be
        # omitted by pytest. In case (ii), it will be useful.
        warnings.warn(
            "Initializing the default simulatuion schedule: [0, 1] seconds, dt=1. Set "
            "the simulation schedule by using "
            "TimeStepper(scheduler=pp.time_stepper.assemble_default_time_scheduler())",
            category=FutureWarning,
            stacklevel=3,
        )

    return TimeStepper(
        scheduler=pp.time_stepper.assemble_default_time_scheduler(
            schedule=np.array([0, 1], dtype=float),
            dt_init=1.0,
            constant_dt=True,
        )
    )
