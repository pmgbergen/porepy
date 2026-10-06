from dataclasses import dataclass

import numpy as np
import logging
from pathlib import Path

import pp_solvers
import porepy as pp
from porepy.models.model_runner import _extract_nonlinear_solver_from_params
from thm_fractured_reservoir import ThmFracturedReservoir, set_model_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SCHEDULE_INTERVAL_EQUILIBRATION = "100 years"

CHANGE_PERIOD = pp.YEAR
"""Period over which the change of variables is measured, to compare it with
tolerances."""


def make_initialization_time_manager():
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
                    name=SCHEDULE_INTERVAL_EQUILIBRATION,
                ),
            ],
            t_end=100 * pp.YEAR,
        )
    )


def make_solver_params() -> dict:
    return {
        "nl_max_iterations": 25,
        "nl_convergence_inc_atol": 1e-7,
        "nl_convergence_res_atol": 1e-7,
        "nl_divergence_inc_atol": 1e12,
        "nl_divergence_res_atol": 1e12,
        "nonlinear_solver": pp.solvers.ConstraintLineSearchNonlinearSolver,
        "global_line_search": 1,
        "local_line_search": 0,
    }


def make_nonlinear_solver() -> pp.solvers.NewtonSolver:
    return pp.solvers.NewtonSolver(
        params=make_solver_params(),
        linear_solver=pp_solvers.IterativeLinearSolver(),
    )


DEFAULT_TOLERANCE = 1e-3
"""Absolute tolerance, in model units, of variables without an explicit tolerance."""


def make_tolerances(model: pp.PorePyModel) -> dict[str, float]:
    """Tolerances of variables: changes during :data:`CHANGE_PERIOD` that are considered
    negligible, in the units of the model. Other variables use
    :data:`DEFAULT_TOLERANCE`."""
    return {
        "pressure": model.units.convert_units(100, "Pa") / pp.YEAR,
        "temperature": model.units.convert_units(1, "K") / pp.YEAR,
        "u": model.units.convert_units(1e-2, "m") / pp.YEAR,
    }


# def change_over_period(
#     model: pp.PorePyModel,
#     solver: pp.solvers.NewtonSolver,
#     period: float = CHANGE_PERIOD,
# ) -> np.ndarray:
#     """Change of all variables during a time step of length ``period``.

#     It is the first Newton increment of that time step. Right after a converged time
#     step the previous time step equals the current state, so the increment vanishes in
#     a steady state. The state of the model is not changed.

#     """
#     solver.linear_solver.initialize_with_model(model)
#     model.ad_time_step.set_value(period)
#     try:
#         increment, _ = solver.iteration(model)
#     finally:
#         model.ad_time_step.set_value(model.time_manager.dt)
#     return increment


def check_steady_state(
    model: pp.PorePyModel,
    tolerances: dict[str, float],
    variable_change: np.ndarray,
    default_tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[dict[str, float], dict[str, float]]:
    """Check whether the change of each variable is below its tolerance.

    Parameters:
        model: The model.
        tolerances: Absolute tolerance of variables, by name.
        variable_change: Change of all variables, see :func:`change_over_period`.
        default_tolerance: Tolerance of variables not in ``tolerances``.

    Returns:
        The norm of the change of each variable, and those that exceed the tolerance.

    """
    norms = {}
    failed = {}
    for (
        name,
        domains_dofs,
    ) in model.equation_system.variable_indexer.group_by_name().items():
        indices = np.concatenate(list(domains_dofs.values()))
        norm = np.linalg.norm(variable_change[indices])
        norms[name] = norm
        if norm >= tolerances.get(name, default_tolerance):
            failed[name] = norm
    return norms, failed


def log_steady_state_convergence(
    norms: dict[str, float],
    failed: dict[str, float],
) -> None:
    """Log the range and the change during :data:`CHANGE_PERIOD` of each variable."""

    def signed(value: float) -> str:
        return f"{'-' if value < 0 else ' '}{abs(value):.1e}"

    max_symbols_offset = max(len(key) for key in norms)
    for key, indices in norms.items():
        logger.info(
            f"{key}:{' ' * (1 + max_symbols_offset - len(key))}"
            f"rate ={signed(norms[key])}{'' if key in failed else ', converged'}"
        )


@dataclass
class SteadyStateModelRunnerSuccess(pp.ModelRunnerStatusSuccess):
    pass


class InitializationError(RuntimeError):
    """Raised if the initialization did not end in a verified steady state."""


class SteadyStateEarlyStopCriterion(pp.EarlyStopCriterion):
    """Stops the initialization run once the variables change less than their
    tolerances during :data:`CHANGE_PERIOD`, see :func:`check_steady_state`.

    Parameters:
        tolerances: Absolute tolerances of variables.
        earliest_stop_time: The simulation is not stopped before this time.
        solver: Solver used to compute the change of variables. Created if not given.

    """

    def __init__(
        self,
        rate_tolerances: dict[str, float],
        residual_tolerances: dict[str, float],
        earliest_stop_time: float = 0,
    ) -> None:
        self.rate_tolerances = rate_tolerances
        self.residual_tolerances = residual_tolerances
        self.earliest_stop_time = earliest_stop_time
        self.previous_state = None

    def simulation_should_stop(
        self, model: pp.PorePyModel
    ) -> pp.ModelRunnerStatus | None:
        state = model.equation_system.get_variable_values(iterate_index=0)
        rate = None
        if self.previous_state is not None:
            rate = (
                state - self.previous_state
            ) / model.time_manager.dt  # TODO YZ: this dt is incorrect.
        self.previous_state = state
        failed = {"not_checked": np.inf}
        if rate is not None:
            norms, failed = check_steady_state(model, self.rate_tolerances, rate)
            log_steady_state_convergence(norms, failed)
        if failed or model.time_manager.time < self.earliest_stop_time:
            return None
        return SteadyStateModelRunnerSuccess()


class SteadyStateMismatch(ValueError):
    """The saved steady state does not match the model."""


class ModelStateStorage:
    """Saves the arrays that define the state of a model, and loads them back. Whether
    the state is a steady state is not checked here.

    The arrays are:

    - the values of all variables, together with their names and sizes,
    - the reference stress of each matrix subdomain,
    - the boundary values of each boundary grid, which are the reference values of the
      boundary operators.

    All variable values (including the reference values) are set to the saved state
    when loading. The file is a numpy ``.npz`` archive.

    Parameters:
        path: File to save to and load from, usually in the simulation folder.

    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def discard(self) -> None:
        """Delete the saved state."""
        self.path.unlink(missing_ok=True)

    @staticmethod
    def _variable_layout(model: pp.PorePyModel) -> tuple[np.ndarray, np.ndarray]:
        groups = model.equation_system.variable_indexer.group_by_name()
        names = np.array(list(groups))
        sizes = np.array([sum(i.size for i in g.values()) for g in groups.values()])
        return names, sizes

    @staticmethod
    def _matrix_subdomains(model: pp.PorePyModel) -> list[tuple[pp.Grid, dict]]:
        return list(model.mdg.subdomains(dim=model.nd, return_data=True))

    @staticmethod
    def _boundaries(model: pp.PorePyModel) -> list[tuple[pp.BoundaryGrid, dict]]:
        return list(model.mdg.boundaries(return_data=True))

    def capture_state(self, model: pp.PorePyModel) -> dict[str, np.ndarray]:
        """Collect the arrays defining the current state of the model.

        The stress is the total stress. It is the reference stress once the reference
        state is set, and it is independent of whether the boundary values are already
        stored as reference values.

        """
        names, sizes = self._variable_layout(model)
        arrays: dict[str, np.ndarray] = {
            "variables": model.equation_system.get_variable_values(iterate_index=0),
            "variable_names": names,
            "variable_sizes": sizes,
        }
        for i, (sd, _) in enumerate(self._matrix_subdomains(model)):
            arrays[f"reference_stress_{i}"] = np.asarray(
                model.equation_system.evaluate(model.stress([sd]))
            )
        for j, (_, data) in enumerate(self._boundaries(model)):
            for name, values in data.get(pp.ITERATE_SOLUTIONS, {}).items():
                arrays[f"boundary_{j}_{name}"] = np.array(values[0])
        return arrays

    def save_state(self, model: pp.PorePyModel) -> None:
        """Save the state of the model."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **self.capture_state(model))  # type: ignore[arg-type]
        logger.info(f"Saved steady state to {self.path}.")

    def load_state(self, model: pp.PorePyModel) -> None:
        """Load the saved state into the model and set it as the reference state.

        Raises:
            SteadyStateMismatch: If the arrays are inconsistent with the model, i.e.
                variable layout, array shapes or the set of boundary arrays differ, or
                the values are not finite. The model is not changed in that case.

        """
        try:
            with np.load(self.path) as archive:
                arrays = {key: archive[key] for key in archive.files}
        except (OSError, ValueError) as e:
            raise SteadyStateMismatch(f"Cannot read {self.path}: {e}") from e
        self._check_consistency(model, arrays)
        self._apply_state(model, arrays)

    def _check_consistency(
        self, model: pp.PorePyModel, arrays: dict[str, np.ndarray]
    ) -> None:
        def fail(message: str) -> None:
            raise SteadyStateMismatch(f"Saved steady state is inconsistent: {message}")

        names, sizes = self._variable_layout(model)
        for key in ["variables", "variable_names", "variable_sizes"]:
            if key not in arrays:
                fail(f"missing '{key}'")
        if not (
            np.array_equal(arrays["variable_names"], names)
            and np.array_equal(arrays["variable_sizes"], sizes)
        ):
            fail("variable names or sizes differ")
        if arrays["variables"].shape != (model.equation_system.num_dofs(),):
            fail("number of degrees of freedom differs")

        for i, (sd, _) in enumerate(self._matrix_subdomains(model)):
            saved = arrays.get(f"reference_stress_{i}")
            if saved is None or saved.shape != (sd.num_faces * model.nd,):
                fail(f"reference stress of matrix subdomain {i}")

        expected = {"variables", "variable_names", "variable_sizes"}
        expected |= {
            f"reference_stress_{i}" for i in range(len(self._matrix_subdomains(model)))
        }
        for j, (_, data) in enumerate(self._boundaries(model)):
            for name, values in data.get(pp.ITERATE_SOLUTIONS, {}).items():
                key = f"boundary_{j}_{name}"
                expected.add(key)
                if key not in arrays or arrays[key].shape != np.shape(values[0]):
                    fail(f"boundary array {key}")
        if set(arrays) != expected:
            fail(f"unexpected arrays {sorted(set(arrays) ^ expected)}")
        floats = [a for a in arrays.values() if a.dtype.kind == "f"]
        if not all(np.all(np.isfinite(a)) for a in floats):
            fail("values are not finite")

    def _apply_state(
        self, model: pp.PorePyModel, arrays: dict[str, np.ndarray]
    ) -> None:
        """Set the arrays as the state and as the reference state of the model."""
        for i, (_, data) in enumerate(self._matrix_subdomains(model)):
            pp.set_solution_values(
                name=model.reference_stress_key,
                values=arrays[f"reference_stress_{i}"].copy(),
                data=data,
                reference=True,
            )
        es = model.equation_system
        es.set_variable_values(reference=True, values=arrays["variables"])
        es.set_variable_values(iterate_index=0, values=arrays["variables"])
        # Copy the state to all stored iterates and time steps.
        model.initialize_previous_iterate_and_time_step_values()
        # Boundary values are the state of the boundary grids. They are also the
        # reference values of the boundary operators.
        for j, (_, data) in enumerate(self._boundaries(model)):
            for name in data.get(pp.ITERATE_SOLUTIONS, {}):
                values = arrays[f"boundary_{j}_{name}"]
                for kwargs in [
                    {"reference": True},
                    {"iterate_index": 0},
                    {"time_step_index": 0},
                ]:
                    pp.set_solution_values(
                        name=name, values=values.copy(), data=data, **kwargs
                    )
        model.update_derived_quantities()


class _InitializationContext:
    """Context manager that temporarily replaces the model class, time manager and
    output folder with those used for initialization.

    The initialization model sets shear dilation to zero and porosity to its reference
    value, and evaluates time-dependent data (boundary values, sources) at a fixed time.

    """

    def __init__(self, model: pp.PorePyModel) -> None:
        self.model = model
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

            def update_time_dependent_ad_arrays(self) -> None:
                # Time dependent data is frozen at data_time, while the clock of the
                # time manager advances to find the steady state.
                clock_time = self.time_manager.time
                self.time_manager.time = data_time
                try:
                    super().update_time_dependent_ad_arrays()
                finally:
                    self.time_manager.time = clock_time

        self.initialized_class = InitializedModel

    def __enter__(self) -> None:
        self.model.__class__ = self.initialized_class
        self.model.time_manager = make_initialization_time_manager()
        self.model.params["folder_name"] = self.folder_name

    def __exit__(self, *args: object) -> None:
        self.model.__class__ = self.original_class
        self.model.time_manager = self.original_time_manager
        self.model.params["folder_name"] = self.original_folder_name


def _run_to_steady_state(
    model: pp.PorePyModel,
    storage: ModelStateStorage,
    tolerances: dict[str, float],
) -> None:
    """Time step the prepared initialization model to a steady state, and set the
    steady state as the reference state.

    Raises:
        InitializationError: If a steady state was not reached.

    """
    runner = pp.ModelRunner(
        model,
        params={"prepare_simulation": False},
        nonlinear_solver=make_nonlinear_solver(),  # TODO YZ: Should be injected.
        early_stop_criteria=[
            SteadyStateEarlyStopCriterion(
                rate_tolerances=tolerances,
                earliest_stop_time=1 * pp.YEAR,
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


def validate_steady_state(model: pp.PorePyModel, tolerances: dict[str, float]) -> bool:
    """Check that the variables of the model do not change during
    :data:`CHANGE_PERIOD` from its current state."""
    # change = change_over_period(model, make_nonlinear_solver())
    model.equation_system.assemble(evaluate_jacobian=False)
    norms, failed = check_steady_state(model, tolerances, change)
    log_steady_state_convergence(
        model, model.equation_system.get_variable_values(iterate_index=0), norms, failed
    )
    if failed:
        logger.warning(
            "Not a steady state, change is too large for: "
            + ", ".join(f"{k} ({v:.1e})" for k, v in failed.items())
        )
    return not failed


def initialization_pipeline(
    model: pp.PorePyModel,
    tolerances: dict[str, float],
    storage: ModelStateStorage | None = None,
) -> pp.PorePyModel:
    """Initialize a model to a steady state.

    If a steady state was saved before, it is loaded, and used if it is consistent with
    the model and verified to be a steady state. Otherwise it is found by time stepping
    with a simplified model, and saved. The state is then set in the original model and
    verified there.

    Parameters:
        model: The model.
        tolerances: Tolerances of variables, see :func:`make_tolerances`.
        storage: Where to save the steady state. Defaults to a file in the simulation
            folder.

    Raises:
        InitializationError: If no verified steady state was found.

    """
    if storage is None:
        storage = ModelStateStorage(
            Path(model.params["folder_name"]) / "steady_state.npz"
        )
    init_context = _InitializationContext(model)

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
        if not validate_steady_state(model, tolerances):
            storage.discard()
            # Reset the model state.
            model.prepare_simulation()
            return False
        return True

    with init_context:
        model.prepare_simulation()
        if not load_steady_state():
            _run_to_steady_state(model, storage, tolerances)

        initialization_steady_state_residual = model.equation_system.assemble(evaluate_jacobian=False)

    # Check that the full model, with the nonlinearities that were disabled, is in a
    # steady state.
    model.prepare_simulation()
    storage.load_state(model)
    if not validate_steady_state(model, tolerances):
        storage.discard()
        raise InitializationError(
            "The model is not in a steady state at the found state."
        )

    logger.info(
        "Initialization complete. The simulation initial and reference states are set "
        "to the found steady state."
    )
    return model


def run_main_simulation(model: pp.PorePyModel):
    runner = pp.ModelRunner(
        model=model,
        params={"prepare_simulation": False},
        nonlinear_solver=make_nonlinear_solver(),
    )
    status = runner.run()
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("porepy.numerics.solvers.newton_solver").setLevel(logging.WARNING)
    logging.getLogger("porepy.numerics.solvers.linear_solvers.linear_solver").setLevel(
        logging.WARNING
    )
    logging.getLogger("pp_solvers.porepy_integration").setLevel(logging.WARNING)
    model = ThmFracturedReservoir(set_model_params())

    model = initialization_pipeline(
        model,
        tolerances=make_tolerances(model),
    )

    run_main_simulation(model)

    pass
