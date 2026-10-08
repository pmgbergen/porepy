"""Initialization of a model with mechanics to a steady state.

This module finds the state of a model in which nothing changes in time, given its
boundary conditions and sources, and makes that state the initial *and* the reference
state of the model. In the new reference state, the stress, the porosity and the shear
dilation of the model are the reference values, and the equations of the model are
satisfied without accumulation. A simulation can then start from it, and only changes
that are applied after the start, e.g. injection, change the state.

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

3. Restores the original model, prepares it again, and sets the found state as its
   state and as its reference state. This includes the total stress as the reference
   stress, and the boundary values as reference values of the boundary operators
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
- **Time dependent model.** The model has a time manager with a start time and an
  initial time step. The simulation time is the start time when the model is prepared.
- **Repeated preparation.** :meth:`~porepy.models.solution_strategy.SolutionStrategy.
  prepare_simulation` can be called more than once. The pipeline calls it two or three
  times, which creates the grid, variables and discretizations each time. The model is
  not restarted from a file, since that would overwrite the initial condition.
- **Time dependent data is frozen.** The boundary values and sources are the data of the
  first time step of the simulation, i.e. at the time ``start + initial time step``
  (``params["initialization_data_time"]`` overrides this). They are constant during the
  initialization. This assumes that they are updated from the clock of the time manager
  in ``before_time_step`` or the methods it calls (``update_time_dependent_ad_arrays``,
  ``update_derived_quantities``). Code that reads the clock at other times, e.g. when
  an operator is evaluated or in the hooks after the nonlinear loop, sees the clock of
  the initialization. In the final model, only the boundary values are set to the
  frozen data. Other time dependent data, e.g. a source term, has the values of the
  start time when the model is prepared again, and is updated to the data time only by
  the first time step. Therefore, such data must not differ between the start time and
  the data time, or the check of the final model fails.
- **A steady state exists.** The initialization time must be long enough, and no
  accumulation term can be driven by the (constant) data, e.g. the data must not
  describe an ongoing injection.
- **Hooks are called.** The solver calls ``after_nonlinear_convergence`` or
  ``after_nonlinear_failure`` before the time stepper reads the clock again, as the
  Newton solver of the library does. The model class is a regular class that can be
  subclassed, and its ``params`` contain a ``folder_name``.
- **Names.** The tolerances are given per variable or equation name, i.e. the names in
  the variable and equation indexers of the model, e.g. ``"pressure"`` and
  ``"mass_balance_equation"``.
- **Saved states.** A saved state is only checked for the variable layout and the shapes
  of the arrays. A state of a different model with the same grids and variables, e.g.
  with different material constants, is rejected only by the check of the residual.

Tolerances are in the units of the model and are absolute. They are dictionaries that
return the default for names that are not given, e.g. a ``defaultdict``.

"""

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


def default_initialization_time_manager() -> pp.TimeManager:
    """Time manager of the time stepping to the steady state.

    The time steps grow from a second to a year, within 100 years. The nonlinear solver
    controls the time step size within each interval.

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
    """Tolerances of the rate of change of variables, in the units of the model per
    second. They correspond to the changes of 100 Pa, 1 K and 1 cm per year. Other
    variables use 1e-3."""
    return defaultdict(
        lambda: 1e-3,
        {
            "pressure": model.units.convert_units(100, "Pa") / pp.YEAR,
            "temperature": model.units.convert_units(1, "K") / pp.YEAR,
            "u": model.units.convert_units(1e-2, "m") / pp.YEAR,
        },
    )


def default_residual_tolerances() -> dict[str, float]:
    """Tolerance of the residual of all equations, in the units of the model."""
    return defaultdict(lambda: 1e-9)


class InitializationError(RuntimeError):
    """Raised if the initialization did not end in a verified steady state."""


class _InitializationContext:
    """Context manager that temporarily replaces the model class, time manager and
    output folder with those used for initialization.

    The class of the model is replaced by a subclass, with the following changes:

    - The shear dilation is zero, and the matrix porosity is the reference porosity.
      This removes the nonlinearities that depend on the plastic displacement and the
      deformation, so that the time stepping to the steady state converges easily.
    - The clock of the time manager is set to the data time from the start of a time
      step until the end of the nonlinear loop. The model updates its time dependent
      data (boundary values, sources) at the start of a time step from the clock, so
      this data is constant, and is that of the first time step of the simulation. The
      clock is restored at the end of the nonlinear loop, since the time stepper needs
      the actual time to log the step and to choose the next time step size. The hooks
      after the time step (``after_time_step_*``) are called after that, so they are not
      suitable for restoring it.

    The time manager and the folder are restored on exit, also if an error is raised.
    The exporter and statistics of the model are recreated by ``prepare_simulation``.

    Parameters:
        model: The model, which is changed in place.
        time_manager: Time manager of the time stepping to the steady state.

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
                # No dilation of the fractures during the initialization.
                return pp.ad.Scalar(0)

            def matrix_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
                # Constant porosity: the reference porosity in each cell.
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
    storage.save_state(model)


def _validate_steady_state(
    model: pp.PorePyModel, residual_tolerances: dict[str, float]
) -> bool:
    """Check that the residual of the equations of the model is below the tolerances
    at its current state.

    The previous time step must equal the current state, as it does for a state that is
    just loaded or set. The residual is then that of the steady problem, without
    accumulation.

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
    nonlinear_solver: pp.solvers.NewtonSolver | None = None,
    storage: ModelStateStorage | None = None,
    time_manager: pp.TimeManager | None = None,
    earliest_stop_time: float = 1 * pp.YEAR,
) -> pp.PorePyModel:
    """Initialize a model to a steady state.

    If a steady state was saved before, it is loaded, and used if it is consistent with
    the model and verified to be a steady state. Otherwise it is found by time stepping
    with a simplified model, and saved. The state is then set in the original model as
    its state and reference state, and verified there. See the module documentation for
    the steps and the assumptions about the model.

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

    Returns:
        The model, prepared, with the steady state as its initial and reference state.

    Raises:
        ValueError: If the model has no mechanics.
        InitializationError: If no verified steady state was found. The model should
            be discarded then, since its equations are not those of the original model.

    """
    # The reference stress is the quantity that makes the stress vanish as perturbation
    # in the reference state, and it only exists for models with mechanics.
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
            # Reset the model state to the initial condition, to time step from there.
            model.prepare_simulation()
            return False
        return True

    # Find the steady state with the simplified model: load it, or time step to it.
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

    # The arrays are collected from the simplified model, before the original model is
    # prepared again, since preparing resets the state.
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
