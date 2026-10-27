"""Initialization of a model with mechanics to a steady state.

This module finds the steady state of a model, given its boundary conditions and
sources at initial time. The steady state becomes initial *and* the reference state of
the model. In the new initial state, the stress, the porosity and the shear dilation of
the model are the reference values, and the equations of the model are satisfied without
accumulation.

What the pipeline does
======================

:func:`initialization_pipeline` takes a model that is not yet prepared and does the
following:

1. Replaces temporarily the model class, the time manager and the output folder (see
   :class:`_InitializationContext`). The simplified model has zero shear dilation and
   constant porosity, because they are assumed known. Its time dependent data (boundary
   values, sources) is evaluated at the initial time. The results of the time stepping
   are written to ``<folder_name>_initialization``.

2. Prepares the simplified model, and either

   - loads a previously saved state from ``<folder_name>/steady_state.npz``, if it
     exists. If the saved state is incompatible (e.g., the model has changed), it is
     discarded.
   - or time steps the simplified model until both the rate of change of the variables
     and the residual of the equations are below the tolerances
     (:class:`~.steady_state_early_stop.SteadyStateEarlyStopCriterion`). The time
     stepping starts from the initial condition of the model. The state is saved.

   The found state is made the reference state of the simplified model: the variables
   and the boundary values are set as reference values, and the total stress becomes the
   reference stress.

3. Restores the original model and checks that the residual of the equations of the
   original model, with original shear dilation and porosity, is below the tolerances.
   If not, the saved state is discarded and :class:`InitializationError` is raised.

"""

import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional, cast

import numpy as np

import porepy as pp
from porepy.applications.initialization.model_state_storage import (
    LoadedStateMismatch,
    ModelStateStorage,
)
from porepy.applications.initialization.steady_state_early_stop import (
    SteadyStateEarlyStopCriterion,
    SteadyStateModelRunnerSuccess,
    check_and_log_tolerances,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InitializationError",
    "initialization_pipeline",
]


def default_initialization_time_manager() -> pp.TimeManager:
    """Create the default time manager of the time stepping to the steady state.

    The time steps grow from a second to a year, within 100 years. This schedule should
    hopefully be sufficient for most cases.

    Returns:
        The time manager.

    """
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
    """Create the default tolerances of the rate of change of variables.

    The tolerances are in the units of the model per second. They correspond to the
    changes of 100 Pa, 1 K and 1 cm per year. Other variables use 1e-3.

    Parameters:
        model: The model, whose units are used.

    Returns:
        The tolerances per variable name.

    """
    return defaultdict(
        lambda: 1e-3,
        {
            "pressure": model.units.convert_units(100, "Pa") / pp.YEAR,
            "temperature": model.units.convert_units(1, "K") / pp.YEAR,
            "u": model.units.convert_units(1e-2, "m") / pp.YEAR,
        },
    )


def default_residual_tolerances() -> dict[str, float]:
    """Create the default tolerances of the residual of equations.

    Returns:
        The tolerance of 1e-9 in the units of the model, for all equations.

    """
    return defaultdict(lambda: 1e-9)


class InitializationError(RuntimeError):
    """Raised if the initialization did not end in a verified steady state."""


class _InitializationContext:
    """Temporarily replaces the model class, time manager and output folder.

    The context manager sets those used for initialization.

    The class of the model is replaced by a subclass, with the following changes:

    - The shear dilation is zero, and the matrix porosity is the reference porosity.
      This removes the nonlinearities that depend on the plastic displacement and the
      deformation.
    - The clock of the time manager is fixed to t_0 + dt_0 (as in the first time step of
      the simulation). This makes the time-dependent data (boundary conditions and
      source terms) constant. The clock is restored at the end of the nonlinear loop.
    - The model parameters are updated with ``params`` and the original values are
      restored on exit.

    The time manager and the folder are restored on exit, also if an error is raised.
    The exporter and statistics of the model are recreated by ``prepare_simulation``.

    Parameters:
        model: The model, which is changed in place.
        time_manager: Time manager of the time stepping to the steady state.
        params: Model parameters that are applied while the context is active. The
            original parameters are restored on exit.

    """

    def __init__(
        self,
        model: pp.PorePyModel,
        time_manager: pp.TimeManager,
        params: dict | None = None,
    ) -> None:
        self.model = model
        """The model, which is changed in place."""
        self.time_manager = time_manager
        """Time manager of the time stepping to the steady state."""
        self.params = params or {}
        """Model parameters that replace those of the model in the context."""
        self._original_params: dict = {}
        """Original values of :attr:`params`, or ``_MISSING`` for absent keys."""
        self.original_class = model.__class__
        """Class of the model outside the context."""
        self.original_time_manager = model.time_manager
        original_folder_path = Path(model.params["folder_name"])
        """Time manager of the model outside the context."""
        self.folder_name = original_folder_path.with_name(
            original_folder_path.name + "_initialization"
        )
        """Output folder of the model in the context."""
        data_time = (
            self.original_time_manager.time_init + self.original_time_manager.dt_init
        )
        # Note: The reason why not just time_init, but (time_init + dt_init) is because
        # the lithostatic boundary condition mixin applies the nonzero bc only when
        # t > time_init. This is an implementation detail which can be reconsidered. If
        # so, there is no other particular reason why not to use just time_init.
        self.data_time: float = data_time
        """Time at which time dependent data is evaluated during the initialization."""

        class InitializedModel(self.original_class):  # type: ignore[name-defined]
            """The model with the simplifications of the initialization."""

            def shear_dilation_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
                """No dilation of the fractures during the initialization."""
                return pp.ad.Scalar(0)

            def matrix_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
                """Constant porosity: the reference porosity in each cell."""
                space = pp.ad.OperatorSpace.from_domains(subdomains)
                ones = pp.ad.DenseArray(
                    np.ones(sum(sd.num_cells for sd in subdomains)),
                    source=space,
                    target=space,
                )
                return self.reference_porosity(subdomains) * ones

            def before_time_step(self) -> None:
                """Freeze the clock at the data time during the nonlinear loop.

                The time unfreezes at after_nonlinear_* hooks. It cannot be frozen until
                after_time_step_*: the TimeStepper reads the clock between
                after_nonlinear_* and after_time_step_* to adjust dt, so it needs the
                actual time.

                """
                self.__clock_time = self.time_manager.time
                self.time_manager.time = data_time
                super().before_time_step()

            def after_nonlinear_convergence(self) -> None:
                """Restore the clock."""
                super().after_nonlinear_convergence()
                self.time_manager.time = self.__clock_time

            def after_nonlinear_failure(self) -> None:
                """Restore the clock."""
                super().after_nonlinear_failure()
                self.time_manager.time = self.__clock_time

        self.initialized_class = InitializedModel
        """The subclass of the original model class used in the context."""

    def __enter__(self) -> None:
        """Replace the class, time manager, folder and parameters of the model."""
        self.model.__class__ = self.initialized_class
        self.model.time_manager = self.time_manager
        self._original_params = deepcopy(self.model.params)
        self.model.params["folder_name"] = self.folder_name
        self.model.params.update(self.params)

    def __exit__(self, *args: object) -> None:
        """Restore the class, time manager, folder and parameters of the model.

        Parameters:
            *args: Exception information, not used.

        """
        self.model.__class__ = self.original_class
        self.model.time_manager = self.original_time_manager
        self.model.params = self._original_params
        # The exporter and statistics path were created in the initialization folder.
        # Recreate them in the original one, also if the initialization failed and the
        # model is not prepared again.
        if hasattr(self.model, "exporter"):
            self.model.initialize_data_saving()


def _set_state_as_reference(model: pp.PorePyModel) -> None:
    """Make the current state of the model its reference state.

    The reference values of the variables and of the boundary operators are set to the
    current values, and the reference stress is set to the total stress of the current
    state, such that the stress perturbation from the reference vanishes.

    Parameters:
        model: A prepared model with mechanics.

    """
    equation_system = model.equation_system

    # Compute the reference stress.
    matrix = model.mdg.subdomains(dim=model.nd, return_data=True)
    reference_stress: dict[pp.Grid, np.ndarray] = {}
    for sd, _ in matrix:
        stress = model.stress([sd])  # type: ignore[attr-defined]
        reference_stress[sd] = np.asarray(equation_system.evaluate(stress))

    # Apply the primary variables to the reference state.
    equation_system.set_variable_values(
        equation_system.get_variable_values(iterate_index=0), reference=True
    )

    # Set the reference boundary data.
    model.set_boundary_reference_values()

    # Set the reference stress.
    for sd, data in matrix:
        pp.set_solution_values(
            name=model.reference_stress_key,  # type: ignore[attr-defined]
            values=reference_stress[sd],
            data=data,
            reference=True,
        )


def _run_to_steady_state(
    model: pp.PorePyModel,
    storage: ModelStateStorage,
    nonlinear_solver: pp.solvers.NonlinearSolverBase | None,
    rate_tolerances: dict[str, float],
    residual_tolerances: dict[str, float],
    earliest_stop_time: float,
) -> None:
    """Time step the prepared initialization model to a steady state, and save it.

    The time stepping stops when the criterion
    :class:`~.steady_state_early_stop.SteadyStateEarlyStopCriterion` is fulfilled, i.e.
    both the rate of change of the variables and the residual of the equations are
    below the tolerances, after ``earliest_stop_time``.

    Parameters:
        model: Prepared model in the initialization context.
        storage: Where to save the steady state.
        nonlinear_solver: Solver of the time steps. A default solver of the model runner
            is used if it is None.
        rate_tolerances: Tolerances of the rate of change of variables.
        residual_tolerance: Tolerances of the residual of equations.
        earliest_stop_time: The time stepping is not stopped before this time.

    Raises:
        InitializationError: If a steady state was not reached.

    """
    runner = pp.ModelRunner(
        cast(pp.SolutionStrategy, model),
        params={"prepare_simulation": False},
        nonlinear_solver=nonlinear_solver,
        early_stop_criteria=[
            SteadyStateEarlyStopCriterion(
                rate_tolerances=rate_tolerances,
                residual_tolerances=residual_tolerances,
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
    # The saved state is the reference state of the simulation to be started from it.
    _set_state_as_reference(model)
    storage.save_state(model)


def _validate_steady_state(
    model: pp.PorePyModel, residual_tolerances: dict[str, float]
) -> bool:
    """Check the residual of the equations of the model at its current state.

    The previous time step must equal the current state, as it does for a state that is
    just loaded or set. The residual is then that of the steady problem, without
    accumulation.

    Parameters:
        model: A prepared model.
        residual_tolerances: Tolerances of the residual of equations.

    Returns:
        True if the residual is below the tolerances, False otherwise.

    """
    failed = check_and_log_tolerances(
        metric=pp.EquationBasedLebesgueMetric(model),
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
    rate_tolerances: Optional[dict[str, float]] = None,
    residual_tolerances: Optional[dict[str, float]] = None,
    nonlinear_solver: Optional[pp.solvers.NonlinearSolverBase] = None,
    storage: Optional[ModelStateStorage] = None,
    time_manager: Optional[pp.TimeManager] = None,
    earliest_stop_time: float = 1 * pp.YEAR,
    initialization_params: Optional[dict] = None,
) -> pp.PorePyModel:
    """Initialize a model to a steady state.

    If a steady state was saved before, it is loaded, and used if it is consistent with
    the model and verified to be a steady state. Otherwise it is found by time stepping
    with a simplified model, and saved. The state is then set in the original model as
    its state and reference state, and verified there. See the module documentation for
    details.

    Warning:
        The returned model is already prepared. Do not call
        :meth:`~porepy.models.solution_strategy.SolutionStrategy.prepare_simulation`
        on it afterwards: it discards the obtained initialized reference states.

    Parameters:
        model: A model with mechanics that has not been prepared. It is prepared by the
            pipeline, and is ready to be run when this function returns.
        rate_tolerances: Tolerances of the rate of change of variables, per variable
            name and in model units per second. Defaults to the values provided by
            :func:`default_rate_tolerances`.
        residual_tolerances: Tolerances of the residual of equations, per equation name
            and in model units. Defaults to the values provided by
            :func:`default_residual_tolerances`.
        nonlinear_solver: Solver for the time steps to the steady state. A default
            solver of the model runner is used if it is None.
        storage: Where to save the steady state. Defaults to a file in the simulation
            folder.
        time_manager: Time manager of the time stepping to the steady state. Defaults
            to :func:`default_initialization_time_manager`.
        earliest_stop_time: The time stepping is not stopped before this time.
        initialization_params: Model parameters that replace those of the model during
            the time stepping to the steady state only, e.g. a constant well schedule.
            The original model is checked and run with its own parameters.

    Returns:
        The model, prepared, with the steady state as its initial and reference state.

    Raises:
        ValueError: If the model has no mechanics.
        InitializationError: If no verified steady state was found. The model should
            be discarded then, since its equations are not those of the original model.

    """
    if not hasattr(model, "reference_stress_key"):
        raise ValueError(
            "The initialization requires a model with mechanics (a momentum balance)."
        )

    # Initialize defaults.
    if rate_tolerances is None:
        rate_tolerances = default_rate_tolerances(model)
    if residual_tolerances is None:
        residual_tolerances = default_residual_tolerances()
    if storage is None:
        storage = ModelStateStorage(
            Path(model.params["folder_name"]) / "steady_state.npz"
        )
    if time_manager is None:
        time_manager = default_initialization_time_manager()

    init_context = _InitializationContext(model, time_manager, initialization_params)

    def load_steady_state() -> bool:
        """Load the saved steady state and validate it.

        Returns:
            True if loaded successfully, False otherwise.

        """
        if not storage.exists():
            return False
        logger.info(f"Trying to load steady state from {storage.path}.")
        try:
            storage.load_state(model)
        except LoadedStateMismatch as e:
            # Failed to load.
            logger.warning(f"{e} It is discarded.")
            storage.discard()
            # Reset the model state to the initial condition, to time step from there.
            model.prepare_simulation()
            return False
        if not _validate_steady_state(model, residual_tolerances):
            # Failed to validate.
            storage.discard()
            model.prepare_simulation()
            return False
        return True

    # Find the steady state with the simplified model: load it, or time step to it.
    with init_context:
        model.prepare_simulation()

        # Try to load steady state from file, if cannot load, run initialization.
        if not load_steady_state():
            _run_to_steady_state(
                model=model,
                storage=storage,
                nonlinear_solver=nonlinear_solver,
                rate_tolerances=rate_tolerances,
                residual_tolerances=residual_tolerances,
                earliest_stop_time=earliest_stop_time,
            )

        # The arrays are collected from the simplified model, before the original model
        # is prepared again, since preparing resets the state.
        state_arrays = storage.capture_state(model)

    # Prepare that the full model (enable nonlinearities) and check that it is in a
    # steady state.
    model.prepare_simulation()

    storage.apply_state(model, arrays=state_arrays)
    if not _validate_steady_state(model, residual_tolerances):
        storage.discard()
        raise InitializationError(
            "The model is not in a steady state at the found state."
        )

    logger.info(
        "Initialization is complete. The simulation initial and reference states are "
        "set to the found steady state. Do not call prepare_simulation again, it will "
        "discard the steady state."
    )
    return model
