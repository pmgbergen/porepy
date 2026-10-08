from collections import defaultdict

import numpy as np
import logging
from pathlib import Path

import porepy as pp
from porepy.applications.initialization.model_state_storage import (
    ModelStateStorage,
    SteadyStateMismatch,
)
from porepy.applications.initialization.steady_state_early_stop import (
    SteadyStateEarlyStopCriterion,
    SteadyStateModelRunnerSuccess,
    check_and_log_tolerances,
)

logger = logging.getLogger(__name__)


def default_initialization_time_manager():
    return pp.TimeManager(
        pp.time_stepper.Schedule(
            intervals=[
                pp.time_stepper.TimeInterval.create(
                    t_start=0,
                    dt_start=pp.SECOND,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                    name="2 seconds",
                ),
                pp.time_stepper.TimeInterval.create(
                    t_start=2 * pp.SECOND,
                    dt_start=pp.HOUR,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                    name="2 hours",
                ),
                pp.time_stepper.TimeInterval.create(
                    t_start=2 * pp.HOUR,
                    dt_start=pp.DAY,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                    name="2 days",
                ),
                pp.time_stepper.TimeInterval.create(
                    t_start=2 * pp.DAY,
                    dt_start=pp.WEEK,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                    name="2 weeks",
                ),
                pp.time_stepper.TimeInterval.create(
                    t_start=2 * pp.WEEK,
                    dt_start=4 * pp.WEEK,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                    name="2 months",
                ),
                pp.time_stepper.TimeInterval.create(
                    t_start=8 * pp.WEEK,
                    dt_start=pp.YEAR,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                    name="100 years",
                ),
            ],
            t_end=100 * pp.YEAR,
        )
    )


def default_rate_tolerances(model: pp.PorePyModel) -> dict[str, float]:
    """Tolerances of the rate of change of variables, in the units of the model per
    second. Variables without an entry use :data:`DEFAULT_TOLERANCE`."""
    return defaultdict(
        lambda: 1e-3,
        {
            "pressure": model.units.convert_units(100, "Pa") / pp.YEAR,
            "temperature": model.units.convert_units(1, "K") / pp.YEAR,
            "u": model.units.convert_units(1e-2, "m") / pp.YEAR,
        },
    )


def default_residual_tolerances() -> dict[str, float]:
    return defaultdict(lambda: 1e-9)


class InitializationError(RuntimeError):
    """Raised if the initialization did not end in a verified steady state."""


class _InitializationContext:
    """Context manager that temporarily replaces the model class, time manager and
    output folder with those used for initialization.

    The initialization model sets shear dilation to zero and porosity to its reference
    value, and evaluates time-dependent data (boundary values, sources) at a fixed time.

    """

    def __init__(self, model: pp.PorePyModel, time_manager: pp.TimeManager) -> None:
        self.model = model
        self.time_manager = time_manager
        self.original_class = model.__class__
        self.original_time_manager = model.time_manager
        self.original_folder_name = model.params["folder_name"]
        self.folder_name = Path(self.original_folder_name).with_name(
            Path(self.original_folder_name).name + "_initialization"
        )
        # Time at which boundary conditions and sources are evaluated during
        # initialization. The initialization clock runs far beyond the simulation's
        # start, but the reference state must match the data seen by the first time
        # step of the main simulation. This is the first target time rather than the
        # start time, since e.g. lithostatic boundary conditions are different at the
        # initial time itself.
        data_time = model.params.get(
            "initialization_data_time",
            self.original_time_manager.time_init + self.original_time_manager.dt_init,
        )

        class InitializedModel(self.original_class):  # type: ignore[name-defined]
            def shear_dilation_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
                return pp.ad.Scalar(0)

            def matrix_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
                space = pp.ad.OperatorSpace.from_domains(subdomains)
                ones = pp.ad.DenseArray(
                    np.ones(sum(sd.num_cells for sd in subdomains)),
                    source=space,
                    target=space,
                )
                return self.reference_porosity(subdomains) * ones

            def before_time_step(self) -> None:
                # The initialization clock runs far beyond the simulation start, but
                # time dependent data (boundary values, sources) must be the data of
                # the first time step of the simulation. Therefore, the clock is
                # frozen at data_time from before_time_step, which updates this data,
                # until the nonlinear loop has finished. It cannot be frozen until
                # after_time_step_*: the time stepper reads the clock between the
                # solve and these hooks to log the step and to choose the next time
                # step size, so it needs the actual time.
                self.__clock_time = self.time_manager.time
                self.time_manager.time = data_time
                super().before_time_step()

            def after_nonlinear_convergence(self) -> None:
                super().after_nonlinear_convergence()
                self.time_manager.time = self.__clock_time

            def after_nonlinear_failure(self) -> None:
                super().after_nonlinear_failure()
                self.time_manager.time = self.__clock_time

        self.initialized_class = InitializedModel

    def __enter__(self) -> None:
        self.model.__class__ = self.initialized_class
        self.model.time_manager = self.time_manager
        self.model.params["folder_name"] = self.folder_name

    def __exit__(self, *args: object) -> None:
        self.model.__class__ = self.original_class
        self.model.time_manager = self.original_time_manager
        self.model.params["folder_name"] = self.original_folder_name


def _run_to_steady_state(
    model: pp.PorePyModel,
    storage: ModelStateStorage,
    nonlinear_solver: pp.solvers.NewtonSolver | None,
    rate_tolerances: dict[str, float],
    residual_tolerance: dict[str, float],
    earliest_stop_time: float,
) -> None:
    """Time step the prepared initialization model to a steady state, and save it.

    Raises:
        InitializationError: If a steady state was not reached.

    """
    runner = pp.ModelRunner(
        model,
        params={"prepare_simulation": False},
        nonlinear_solver=nonlinear_solver,
        early_stop_criteria=[
            SteadyStateEarlyStopCriterion(
                rate_tolerances=rate_tolerances,
                residual_tolerances=residual_tolerance,
                earliest_stop_time=earliest_stop_time,
            )
        ],
    )
    try:
        status = runner.run()
    except RuntimeError as e:
        if not (e.args and isinstance(e.args[0], pp.ModelRunnerStatus)):
            raise
        status = e.args[0]

    # Only a steady state is saved: reaching the final time or a failed time step is
    # not an equilibrium.
    if not isinstance(status, SteadyStateModelRunnerSuccess):
        raise InitializationError(
            f"Steady state was not reached: {status}. The model state is "
            "not initialized."
        )
    storage.save_state(model)


def _validate_steady_state(
    model: pp.PorePyModel, residual_tolerances: dict[str, float]
) -> bool:
    """Check that the residual of the equations of the model is below the tolerances
    at its current state."""
    failed = check_and_log_tolerances(
        metric=pp.EquationBasedEuclideanMetric(model),
        values=model.equation_system.assemble(evaluate_jacobian=False),
        tolerances=residual_tolerances,
        name="residual",
    )
    if failed:
        logger.warning(
            "Not a steady state, residual is too large for: "
            + ", ".join(f"{k} ({v:.1e})" for k, v in failed.items())
        )
    return not failed


def initialization_pipeline(
    model: pp.PorePyModel,
    rate_tolerances: dict[str, float],
    residual_tolerances: dict[str, float],
    nonlinear_solver: pp.solvers.NewtonSolver | None = None,
    storage: ModelStateStorage | None = None,
    time_manager: pp.TimeManager | None = None,
    earliest_stop_time: float = 1 * pp.YEAR,
) -> pp.PorePyModel:
    """Initialize a model to a steady state.

    If a steady state was saved before, it is loaded, and used if it is consistent with
    the model and verified to be a steady state. Otherwise it is found by time stepping
    with a simplified model, and saved. The state is then set in the original model and
    verified there.

    Parameters:
        model: The model.
        rate_tolerances: Tolerances of the rate of change of variables, see
            :func:`make_tolerances`.
        residual_tolerance: Tolerance of the residual of all equations.
        nonlinear_solver: Solver for the time steps. Defaults to
            :func:`make_nonlinear_solver`.
        storage: Where to save the steady state. Defaults to a file in the simulation
            folder.
        time_manager: Time manager of the time stepping to the steady state. Defaults
            to :func:`default_initialization_time_manager`.
        earliest_stop_time: The time stepping is not stopped before this time.

    Raises:
        InitializationError: If no verified steady state was found.

    """
    if storage is None:
        storage = ModelStateStorage(
            Path(model.params["folder_name"]) / "steady_state.npz"
        )

    if time_manager is None:
        time_manager = default_initialization_time_manager()
    init_context = _InitializationContext(model, time_manager)

    def load_steady_state() -> bool:
        if not storage.exists():
            return False
        logger.info(f"Trying to load steady state from {storage.path}.")
        try:
            storage.load_state(model)
        except SteadyStateMismatch as e:
            logger.warning(f"{e} It is discarded.")
            storage.discard()
            return False
        if not _validate_steady_state(model, residual_tolerances):
            storage.discard()
            # Reset the model state.
            model.prepare_simulation()
            return False
        return True

    with init_context:
        model.prepare_simulation()
        if not load_steady_state():
            _run_to_steady_state(
                model=model,
                storage=storage,
                nonlinear_solver=nonlinear_solver,
                rate_tolerances=rate_tolerances,
                residual_tolerance=residual_tolerances,
                earliest_stop_time=earliest_stop_time,
            )

    state_arrays = storage.capture_state(model)
    # Check that the full model, with the nonlinearities that were disabled, is in a
    # steady state.
    model.prepare_simulation()
    storage.apply_state(model, arrays=state_arrays)
    if not _validate_steady_state(model, residual_tolerances):
        storage.discard()
        raise InitializationError(
            "The model is not in a steady state at the found state."
        )

    logger.info(
        "Initialization complete. The simulation initial and reference states are set "
        "to the found steady state."
    )
    return model
