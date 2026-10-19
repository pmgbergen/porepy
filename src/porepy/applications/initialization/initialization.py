"""Initialization of a model with mechanics to a steady state.

This module finds the steady state of a model, given its boundary conditions and
sources at initial time. The steady state becomes initial *and* the reference state of
the model. In the new reference state, the stress, the porosity and the shear
dilation of the model are the reference values, and the equations of the model are
satisfied without accumulation.

What the pipeline does
======================

:func:`initialization_pipeline` takes a model that is not yet prepared and does the
following:

1. Replaces temporarily the class, the time manager and the output folder of the model
   (see :class:`_InitializationContext`). The simplified model has zero shear dilation
   and constant porosity, which makes the nonlinear problem easier. Its time dependent
   data (boundary values, sources) is evaluated at one fixed time, see below. The
   results of the time stepping are written to ``<folder_name>_initialization``.
2. Prepares the simplified model, and either

   - loads a previously saved state from ``<folder_name>/steady_state.npz``, if it
     exists. It is used if the saved arrays have the shapes of the model and the
     residual of the equations is below the tolerances. Otherwise, it is discarded.
   - or time steps the simplified model until both the rate of change of the variables
     and the residual of the equations are below the tolerances
     (:class:`~.steady_state_early_stop.SteadyStateEarlyStopCriterion`). The time
     stepping starts from the initial condition of the model. The state is saved.

   The found state is made the reference state of the simplified model before it is
   saved: the variables and the boundary values are set as reference values, and the
   total stress as the reference stress.
3. Restores the original model, prepares it again, updates its time dependent data to
   the data time, and sets the saved state, which includes the reference state, in it
   (:class:`~.model_state_storage.ModelStateStorage`).
4. Checks that the residual of the equations of the original model, with the shear
   dilation and porosity that were simplified, is below the tolerances at that state.
   If not, the saved state is discarded and :class:`InitializationError` is raised.

Assumptions about the model
===========================

The model must fulfill the following, otherwise the result is not defined:

- **Mechanics.** The model has a momentum balance, with a stress, and the reference
  stress ``reference_stress_key`` of it. A :class:`ValueError` is raised otherwise.
- **Library constitutive laws.** The stress, porosity and shear dilation are formulated
  as perturbations from the reference state, as the constitutive laws of the library
  are. A model that overrides ``stress``, ``mechanical_stress``, ``porosity`` or
  ``shear_dilation_gap`` and does not do this does not vanish its perturbations in the
  new reference state. If the model has a ``matrix_porosity``, it has a
  ``reference_porosity``.
- **Time dependent data is frozen.** The boundary values and sources are the data of the
  first time step of the simulation, i.e. at the time ``start + initial time step``.
  They are frozen during the initialization (simulation time does not go forward).

"""

import logging
from collections import defaultdict
from pathlib import Path

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

    The time steps grow from a second to a year, within 100 years.

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


_MISSING = object()
"""Marks a model parameter that did not exist before it was replaced."""


class InitializationError(RuntimeError):
    """Raised if the initialization did not end in a verified steady state."""


class _InitializationContext:
    """Temporarily replaces the model class, time manager and output folder.

    The context manager sets those used for initialization.

    The class of the model is replaced by a subclass, with the following changes:

    - The shear dilation is zero, and the matrix porosity is the reference porosity.
      This removes the nonlinearities that depend on the plastic displacement and the
      deformation, so that the time stepping to the steady state converges easily.
    - The clock of the time manager is set to the data time from the start of a time
      step until the end of the nonlinear loop. The model updates its time dependent
      data (boundary values, sources) at the start of a time step from the clock, so
      this data is constant, and is that of the first time step of the simulation. The
      clock is restored at the end of the nonlinear loop.

    The time manager and the folder are restored on exit, also if an error is raised.
    The exporter and statistics of the model are recreated by ``prepare_simulation``.

    Parameters:
        model: The model, which is changed in place.
        time_manager: Time manager of the time stepping to the steady state.
        params: Model parameters that replace those of the model while the context is
            active, e.g. a time independent well schedule. The original values are
            restored on exit; keys that the model did not have are removed.

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
        """Time manager of the model outside the context."""
        self.original_folder_name = model.params["folder_name"]
        """Output folder of the model outside the context."""
        self.folder_name = Path(self.original_folder_name).with_name(
            Path(self.original_folder_name).name + "_initialization"
        )
        """Output folder of the model in the context."""
        # Time at which boundary conditions and sources are evaluated during
        # initialization. The initialization clock runs far beyond the simulation's
        # start, but the reference state must match the data seen by the first time
        # step of the main simulation. This is the first target time rather than the
        # start time, since e.g. lithostatic boundary conditions are different at the
        # initial time itself.
        data_time = (
            self.original_time_manager.time_init + self.original_time_manager.dt_init
        )
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

                The initialization clock runs far beyond the simulation start, but
                time dependent data (boundary values, sources) must be the data of
                the first time step of the simulation. Therefore, the clock is
                frozen at data_time from before_time_step, which updates this data,
                until the nonlinear loop has finished. It cannot be frozen until
                after_time_step_*: the time stepper reads the clock between the
                solve and these hooks to log the step and to choose the next time
                step size, so it needs the actual time.

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
        self.model.params["folder_name"] = self.folder_name
        self._original_params = {
            key: self.model.params.get(key, _MISSING) for key in self.params
        }
        self.model.params.update(self.params)

    def __exit__(self, *args: object) -> None:
        """Restore the class, time manager, folder and parameters of the model.

        Parameters:
            *args: Exception information, not used.

        """
        self.model.__class__ = self.original_class
        self.model.time_manager = self.original_time_manager
        self.model.params["folder_name"] = self.original_folder_name
        for key, value in self._original_params.items():
            if value is _MISSING:
                self.model.params.pop(key, None)
            else:
                self.model.params[key] = value

    def update_time_dependent_data(self) -> None:
        """Update the time dependent data of the original model to the data time.

        A prepared model has the data of the start time. A steady state is found for
        the data of the data time, so the model must have this data to be in it. The
        clock is restored afterwards.

        """
        clock_time = self.model.time_manager.time
        self.model.time_manager.time = self.data_time
        try:
            self.model.update_time_dependent_ad_arrays()
        finally:
            self.model.time_manager.time = clock_time


def _set_state_as_reference(model: pp.PorePyModel) -> None:
    """Make the current state of the model its reference state.

    The reference values of the variables and of the boundary operators are set to the
    current values, and the reference stress is set to the total stress of the current
    state, such that the stress perturbation from the reference vanishes.

    Parameters:
        model: A prepared model with mechanics.

    """
    es = model.equation_system
    # The total stress depends on the reference state, hence it is evaluated first.
    matrix = model.mdg.subdomains(dim=model.nd, return_data=True)
    reference_stress = {
        sd: np.asarray(es.evaluate(model.stress([sd]))) for sd, _ in matrix
    }
    es.set_variable_values(es.get_variable_values(iterate_index=0), reference=True)
    model.set_boundary_reference_values()
    for sd, data in matrix:
        pp.set_solution_values(
            name=model.reference_stress_key,
            values=reference_stress[sd],
            data=data,
            reference=True,
        )


def _run_to_steady_state(
    model: pp.PorePyModel,
    storage: ModelStateStorage,
    nonlinear_solver: pp.solvers.NonlinearSolverBase | None,
    rate_tolerances: dict[str, float],
    residual_tolerance: dict[str, float],
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
    nonlinear_solver: pp.solvers.NonlinearSolverBase | None = None,
    storage: ModelStateStorage | None = None,
    time_manager: pp.TimeManager | None = None,
    earliest_stop_time: float = 1 * pp.YEAR,
    initialization_params: dict | None = None,
) -> pp.PorePyModel:
    """Initialize a model to a steady state.

    If a steady state was saved before, it is loaded, and used if it is consistent with
    the model and verified to be a steady state. Otherwise it is found by time stepping
    with a simplified model, and saved. The state is then set in the original model as
    its state and reference state, and verified there. See the module documentation for
    the steps and the assumptions about the model.

    Warning:
        The returned model is already prepared. Do not call
        :meth:`~porepy.models.solution_strategy.SolutionStrategy.prepare_simulation`
        on it afterwards: it recreates the grids, variables and equations from the
        initial condition of the model, which discards the initialized state and the
        reference state. Run the returned model without preparing it again, e.g.
        ``pp.run_time_dependent_model(model, {"prepare_simulation": False})``.

    Parameters:
        model: A model with mechanics that has not been prepared. It is prepared by the
            pipeline, and is ready to be run when this function returns.
        rate_tolerances: Tolerances of the rate of change of variables, per variable
            name and in model units per second. See
            :func:`default_rate_tolerances`.
        residual_tolerances: Tolerances of the residual of equations, per equation name
            and in model units. See :func:`default_residual_tolerances`.
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
        # The saved state includes reference values, which exist in the data of the
        # model only after they are set. They are replaced by the loaded ones. This is
        # not done before time stepping, since it would change the equations.
        _set_state_as_reference(model)
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
        # A prepared model has the data of the start time. The time stepping updates it,
        # but a loaded state is checked before that.
        init_context.update_time_dependent_data()
        if not load_steady_state():
            _run_to_steady_state(
                model=model,
                storage=storage,
                nonlinear_solver=nonlinear_solver,
                rate_tolerances=rate_tolerances,
                residual_tolerance=residual_tolerances,
                earliest_stop_time=earliest_stop_time,
            )

    # The arrays are collected from the simplified model, before the original model is
    # prepared again, since preparing resets the state.
    state_arrays = storage.capture_state(model)
    # Check that the full model, with the nonlinearities that were disabled, is in a
    # steady state.
    model.prepare_simulation()
    init_context.update_time_dependent_data()
    # The saved state includes reference values, see load_steady_state.
    _set_state_as_reference(model)
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
