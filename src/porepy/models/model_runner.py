"""This module contains functions to run stationary and time-dependent models."""

from __future__ import annotations

import logging
import warnings
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, cast

import porepy as pp
from porepy.models.solution_strategy import SolutionStrategy
from porepy.numerics.solvers.convergence_check import (
    ConvergenceCriteria,
    ConvergenceStatusCollection,
    DivergenceCriteria,
)
from porepy.numerics.solvers.nonlinear_solver_status import NonlinearSolverStatus
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
class ModelRunnerStatusInProgress(ModelRunnerStatus):
    """A status object that indicates that the simulation is in progress."""


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
        nonlinear_solver: Optional[pp.solvers.NewtonSolver] = None,
    ) -> None:
        self.params = params if isinstance(params, dict) else {}
        """Parameters passed at instantiation."""

        self.model = model
        """Model instance passed at instantiation."""

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

        self.solver: pp.solvers.NewtonSolver = _extract_nonlinear_solver_from_params(
            nonlinear_solver=nonlinear_solver,
            params=self.params,
            is_nonlinear_problem=self._is_nonlinear,
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
            # Create a time bar of length of expected simulation time.

            # Creating a custom format string. Difference from the default: it converts
            # the simulation time (done/total) to the a scientific format with :.1e.
            l_bar = "{desc}: {percentage:3.0f}%|"
            r_bar = "| {n:.1e}/{total:.1e} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            bar_format = l_bar + "{bar}" + r_bar

            # NOTE: If tqdm is not installed, this returns a DummyProgressBar instance.
            self.time_progressbar = progressbar_class(
                total=self.model.time_manager.schedule[-1],
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
            while not self.model.time_manager.final_time_reached():
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
            if self.model.time_manager.final_time_reached():
                return ModelRunnerStatusSuccess()
            return ModelRunnerStatusFailure("Final time was not reached.")

    def _progressbar_postfix(self) -> str:
        """Formats a progressbar postfix string with dt."""
        return f"Δt={self.model.time_manager.dt:.1e}"

    def initialize(self) -> None:
        """Initializes the model for a steady-state or time-dependent simulation.

        Raises:
            ValueError: If an invalid mode is provided.

        """
        # Sanity check.
        if not self._is_time_dependent:
            raise ValueError("Initialization for steady-state mode is not supported.")

        # Set default initialization parameters.
        self.params.setdefault("initialization", {"mode": "steady-state"})

        # Choose initialization class based on mode.
        mode = self.params["initialization"]["mode"]
        if mode == "steady-state":
            Initialization = SteadyStateInitialization
        elif mode == "steady-reference-state":
            Initialization = ReferenceStateInitialization
        else:
            raise ValueError(f"Invalid initialization mode: {mode}")

        # Run the initialization procedure.
        Initialization(
            self.model,
            self.solver,
            self.params["initialization"],
        ).run()

        # Export the initialized state.
        # TODO: This does not overwrite the exported initial (computational) state
        # for counter 0, but advances the exporting counter. Fix this when revising
        # the exporting logic.
        self.model.save_data_time_step()


def _extract_nonlinear_solver_from_params(
    nonlinear_solver: Optional[pp.solvers.NewtonSolver],
    params: dict,
    is_nonlinear_problem: bool,
) -> pp.solvers.NewtonSolver:
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
        logger.warning(
            "You should pass the nonlinear solver directly to the ModelRunner: use "
            "ModelRunner(nonlinear_solver=...). Passing it through params will be "
            "deprecated."
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


# NOTE: Initialization has the same structure as a time loop.
# The only/main difference is the "convergence check" which does not
# check the end of time, but the steady state convergence.
# Furthermore subtle differences in the time manager, e.g., the time is not
# updated in the same way.
# TODO: Refactor and unify the time loop and initialization loop,
# once upgrade of time stepping is finished.
# NOTE: The check of steady state convergence is based on the time increment.
# Since the time stepping updates the time step after the convergence check,
# the time increment is not anymore accessible from outside after a time
# step is performed. This difference in the structure does not allow for
# direct reuse of the time stepping logic for initialization at this stage.


@dataclass
class InitializationParameters:
    """Dataclass to hold initialization parameters."""

    convergence_criteria: ConvergenceCriteria
    divergence_criteria: DivergenceCriteria
    pseudo_dt_init: float
    pseudo_dt_max: float

    @classmethod
    def from_dict(cls, params: dict) -> InitializationParameters:
        """Create an InitializationParameters instance from a dictionary.

        Parameters:
            params: A dictionary containing initialization parameters.
            - "steady_state_convergence_criteria": ConvergenceCriteria
            - "steady_state_divergence_criteria": DivergenceCriteria
            - "pseudo_dt_init": float
            - "pseudo_dt_max": float

        Returns:
            An instance of InitializationParameters with the specified or
            default values.

        """
        default_config = {
            "steady_state_convergence_criteria": ConvergenceCriteria(
                {
                    "inc": pp.solvers.IncrementBasedAbsoluteCriterion(
                        tol=1e-10, metric=pp.EuclideanMetric()
                    )
                }
            ),
            "steady_state_divergence_criteria": DivergenceCriteria(
                {
                    "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=50),
                    "inc_nan": pp.solvers.IncrementBasedNanCriterion(),
                    "inc_max": pp.solvers.IncrementBasedAbsoluteDivergenceCriterion(
                        tol=1e14, metric=pp.EuclideanMetric()
                    ),
                }
            ),
            "pseudo_dt_init": 1000 * pp.YEAR,
            "pseudo_dt_max": 100000 * pp.YEAR,
        }
        return cls(
            convergence_criteria=params.get(
                "steady_state_convergence_criteria",
                default_config["steady_state_convergence_criteria"],
            ),
            divergence_criteria=params.get(
                "steady_state_divergence_criteria",
                default_config["steady_state_divergence_criteria"],
            ),
            pseudo_dt_init=params.get(
                "pseudo_dt_init", default_config["pseudo_dt_init"]
            ),
            pseudo_dt_max=params.get("pseudo_dt_max", default_config["pseudo_dt_max"]),
        )


class SteadyStateInitialization:
    """Class to perform iterative pseudo time stepping for initialization."""

    def __init__(
        self,
        model: pp.SolutionStrategy,
        solver: pp.NewtonSolver | pp.LinearSolver,
        params: dict,
    ):
        self.model = model
        """Model instance passed at instantiation."""
        self.solver = solver
        """Solver instance, set in :meth:`set_solver`."""
        self.config = InitializationParameters.from_dict(params)
        """Initialization parameters."""
        self.iteration = 0
        """Number of pseudo time stepping iterations performed during initialization."""

    def run(self) -> None:
        """Run the iterative pseudo time stepping for initialization."""
        # Artificial time control for quasi-static initialization.
        # Cache the original time manager to restore it after initialization.
        # NOTE: Be careful with creating new instances of the time manager, as it
        # is used as a reference in various places (model, time stepper, etc.).
        # Instead, we modify the existing instance and restore it afterwards.
        cached_schedule = self.model.time_manager.schedule
        cached_dt = self.model.time_manager.dt
        cached_dt_min_max = self.model.time_manager.dt_min_max
        cached_constant_dt = self.model.time_manager.is_constant
        cached_iters = self.model.time_manager._iters

        # Setup pseudo time manager for initialization.
        self.model.time_manager.schedule = [
            self.model.time_manager.time_init,
            self.model.time_manager.time_final + self.config.pseudo_dt_max,
        ]
        self.model.time_manager.dt = self.config.pseudo_dt_init
        self.model.time_manager.dt_min_max = (
            self.model.time_manager.dt_min_max[0],
            self.config.pseudo_dt_max,
        )
        self.model.time_manager.is_constant = False

        # Perform a pseudo time stepping to initialize the reference state.
        self.iteration = 0
        while True:
            # Advance iter.
            self.iteration += 1
            logger.info("Initialization iteration %d", self.iteration)

            # Communicate dt to the model and update time-dependent arrays and
            # derived quantities.
            self.model.before_time_step()

            # Solve pseudo time step.
            nonlinear_solver_status: NonlinearSolverStatus = self.solver.solve(
                self.model
            )

            # Check initialization status.
            # Based on time increment and thus needs to be checked before the time step
            # is updated.
            initialization_status = self.check_initialization_status(
                nonlinear_solver_status
            )

            # React to successful solver status.
            if nonlinear_solver_status.is_converged():
                self.after_pseudo_time_step_convergence()

            if initialization_status.is_success() or initialization_status.is_failure():
                break

        logger.info(
            "Initialization status: %s; after %d iterations",
            initialization_status,
            self.iteration,
        )

        # Restore time manager.
        self.model.time_manager.schedule = cached_schedule
        self.model.time_manager.dt = cached_dt
        self.model.time_manager.dt_min_max = cached_dt_min_max
        self.model.time_manager.is_constant = cached_constant_dt
        self.model.time_manager._iters = cached_iters

    def check_initialization_status(
        self, solver_status: NonlinearSolverStatus
    ) -> ModelRunnerStatus:
        """Check the initialization status based on the solver status.

        Args:
            solver_status: The status of the solver.

        Returns:
            ModelRunnerStatus: The initialization status.

        """
        # React to solver_status.
        if solver_status.is_converged():
            # Evaluate whether steady state has been reached.
            steady_state_status = self.check_steady_state()

            # Conclude with initialization status.
            if steady_state_status.is_converged():
                initialization_status = ModelRunnerStatusSuccess()
            else:
                initialization_status = ModelRunnerStatusInProgress()

            # Update the time step magnitude if the dynamic scheme is used.
            if not self.model.time_manager.is_constant:
                assert isinstance(
                    self.model.nonlinear_solver_statistics, pp.NonlinearSolverStatistics
                )  # For type checking, to ensure the method is available.
                self.model.time_manager.compute_time_step(
                    iterations=self.model.nonlinear_solver_statistics.num_iterations
                )

        elif solver_status.is_failed():
            if self.model.time_manager.is_constant:
                initialization_status = ModelRunnerStatusFailure(
                    reason="Solver failed and time step is constant."
                )

            else:
                try:
                    initialization_status = ModelRunnerStatusFailure(
                        reason="Solver failed."
                    )
                    self.model.time_manager.compute_time_step(recompute_solution=True)
                except Exception as e:
                    logger.warning(str(e))
                    initialization_status = ModelRunnerStatusFailure(
                        reason="Solver and time step recomputation failed."
                    )
        elif solver_status.is_failed():
            initialization_status = ModelRunnerStatusFailure(
                reason="Simulation stopped."
            )

        else:
            raise ValueError("Unrecognized solver status.")

        return initialization_status

    def check_steady_state(self) -> ConvergenceStatusCollection:
        """Check steady state convergence.

        Returns:
            ConvergenceStatus: Enum indicating whether the current state is a steady
                state.

        """
        # Define the increment in time.
        state = self.model.equation_system.get_variable_values(iterate_index=0)
        prev_state = self.model.equation_system.get_variable_values(time_step_index=0)
        time_increment = state - prev_state

        # Convergence equals steady state.
        convergence_status, convergence_info = self.config.convergence_criteria.check(
            increment=time_increment, reference_increment=state
        )

        return convergence_status

    def after_pseudo_time_step_convergence(self) -> None:
        """Shift solution for next computation."""
        self.model.update_time_step_solution()


class ReferenceStateInitialization(SteadyStateInitialization):
    """Initialization class for steady reference state initialization."""

    def after_pseudo_time_step_convergence(self) -> None:
        """Update reference state after successful step."""
        super().after_pseudo_time_step_convergence()
        self.model.update_reference_solution()
