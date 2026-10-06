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

RESIDUAL_RTOL = 1e-3
"""Maximum accepted residual relative to the effect of tolerable variable errors."""
TRIAL_STEP_RTOL = 1e-1
"""Maximum accepted Newton increment relative to variable tolerances, i.e. the state is
accepted if it is within 10% of the tolerances from the steady state."""


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


def make_tolerances(model: pp.PorePyModel) -> dict[str, float]:
    """Absolute tolerances of variables, i.e. the changes that are considered
    negligible, in the units of the model. Variables without an entry are measured
    relative to their magnitude, see :func:`variable_tolerance_vector`."""
    return {
        "pressure": model.units.convert_units(100, "Pa"),
        "temperature": model.units.convert_units(1, "K"),
        "u": model.units.convert_units(1e-2, "m"),
    }


# Variables whose magnitude can be tiny (e.g. a passive well carries no flux), so a
# tolerance relative to their own magnitude is meaningless. They are measured relative
# to the magnitude of a variable of the same kind.
SCALE_LIKE = {
    "well_flux": "interface_darcy_flux",
    "well_enthalpy_flux": "interface_enthalpy_flux",
}


def variable_tolerance_vector(
    model: pp.PorePyModel,
    tolerances: dict[str, float],
    relative_tolerance: float = 1e-2,
) -> np.ndarray:
    """Tolerance for each degree of freedom.

    Variables given in ``tolerances`` use that absolute value. Others use
    ``relative_tolerance`` times the largest magnitude of the variable (or of the
    variable it is :data:`SCALE_LIKE`) in the current state, which makes the measure
    independent of units.

    """
    state = model.equation_system.get_variable_values(iterate_index=0)
    vector = np.zeros_like(state)
    groups = {
        name: np.concatenate(list(domains_dofs.values()))
        for name, domains_dofs in (
            model.equation_system.variable_indexer.group_by_name().items()
        )
    }
    for name, indices in groups.items():
        if name in tolerances:
            vector[indices] = tolerances[name]
            continue
        reference = groups.get(SCALE_LIKE.get(name, name), indices)
        if reference.size > 0 and indices.size > 0:
            vector[indices] = relative_tolerance * np.abs(state[reference]).max()
    return vector


def _safe_ratio(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """|values| / scale, where 0 / 0 is 0 and x / 0 is infinity for x != 0."""
    values = np.abs(values)
    ratio = np.zeros_like(values)
    positive = scale > 0
    ratio[positive] = values[positive] / scale[positive]
    ratio[~positive & (values > 0)] = np.inf
    return ratio


def steady_state_residual_ratios(
    model: pp.PorePyModel, tolerance_vector: np.ndarray
) -> dict[str, float]:
    """Largest relative residual per equation, with the maximum norm.

    The residual in each row is compared with the residual that a change of all
    variables by their tolerance would produce in that row, ``|J| @ tolerance``. A
    ratio of 1 thus means that the residual is as large as the effect of tolerable
    errors in the variables. The measure is independent of the units of the equations.

    """
    system = model.equation_system.assemble()
    matrix = abs(system.matrix)  # type: ignore[arg-type]
    ratios = _safe_ratio(system.rhs, matrix @ tolerance_vector)
    indexer = model.equation_system.equation_indexer
    return {
        name: float(ratios[np.concatenate(list(domains_dofs.values()))].max())
        for name, domains_dofs in indexer.group_by_name().items()
    }


def trial_step_ratios(
    model: pp.PorePyModel,
    tolerance_vector: np.ndarray,
    solver: pp.solvers.NewtonSolver | None = None,
) -> dict[str, float]:
    """Largest Newton increment per variable, relative to its tolerance, if one
    iteration was taken from the current state. The state is not changed."""
    if solver is None:
        solver = make_nonlinear_solver()
    solver.linear_solver.initialize_with_model(model)
    increment, _ = solver.iteration(model)
    ratios = _safe_ratio(increment, tolerance_vector)
    indexer = model.equation_system.variable_indexer
    return {
        name: float(ratios[np.concatenate(list(domains_dofs.values()))].max())
        for name, domains_dofs in indexer.group_by_name().items()
    }


def check_steady_state(
    model: pp.PorePyModel,
    tolerances: dict[str, float],
    solver: pp.solvers.NewtonSolver | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Check whether the current state of the model is a steady state.

    The equations are evaluated as they are, so the previous time step must hold the
    state under test: right after a converged time step the accumulation terms then
    vanish and the residual is that of the steady problem.

    Returns:
        Relative residuals per equation (:func:`steady_state_residual_ratios`),
        relative trial steps per variable (:func:`trial_step_ratios`), and the entries
        of both that exceed :data:`RESIDUAL_RTOL` and :data:`TRIAL_STEP_RTOL`.

    """
    tolerance_vector = variable_tolerance_vector(model, tolerances)
    residual_ratios = steady_state_residual_ratios(model, tolerance_vector)
    trial_ratios = trial_step_ratios(model, tolerance_vector, solver)
    failed = {
        **{k: v for k, v in residual_ratios.items() if not v <= RESIDUAL_RTOL},
        **{k: v for k, v in trial_ratios.items() if not v <= TRIAL_STEP_RTOL},
    }
    return residual_ratios, trial_ratios, failed


def log_steady_state_ratios(
    residual_ratios: dict[str, float], trial_ratios: dict[str, float]
) -> None:
    for name, ratios in [("residual", residual_ratios), ("trial step", trial_ratios)]:
        logger.info(
            f"Relative {name}: "
            + ", ".join(f"{key}={value:.1e}" for key, value in ratios.items())
        )


def log_steady_state_convergence(
    model: pp.PorePyModel,
    state: np.ndarray,
    rates: np.ndarray | None,
    trial_ratios: dict[str, float] | None,
) -> None:
    """Log the range and the rate of change of each variable.

    Parameters:
        model: The model.
        state: Current values of all variables.
        rates: Absolute rate of change of all variables, in units per second, or None
            if it is not known yet.
        trial_ratios: Distance to the steady state relative to the tolerance for each
            variable (:func:`trial_step_ratios`), if it was evaluated. A variable is
            marked as converged if it is below :data:`TRIAL_STEP_RTOL`.

    """

    def signed(value: float) -> str:
        return f"{'-' if value < 0 else ' '}{abs(value):.1e}"

    groups = {
        name: np.concatenate(list(domains_dofs.values()))
        for name, domains_dofs in (
            model.equation_system.variable_indexer.group_by_name().items()
        )
    }
    max_symbols_offset = max(len(key) for key in groups)
    for key, indices in groups.items():
        if indices.size == 0:
            continue
        line = (
            f"{key}:{' ' * (1 + max_symbols_offset - len(key))}"
            f"min ={signed(state[indices].min())}, max ={signed(state[indices].max())}"
        )
        if rates is not None:
            line += f", rate ={signed(rates[indices].max())}/s"
        if trial_ratios is not None:
            ratio = trial_ratios[key]
            line += f", distance ={signed(ratio)}"
            line += ", converged" if ratio <= TRIAL_STEP_RTOL else ""
        logger.info(line)


@dataclass
class SteadyStateModelRunnerSuccess(pp.ModelRunnerStatusSuccess):
    pass


class InitializationError(RuntimeError):
    """Raised if the initialization did not end in a verified steady state."""


class SteadyStateEarlyStopCriterion(pp.EarlyStopCriterion):
    """Stops the initialization run once the model is in a steady state.

    After each accepted time step, the previous time step equals the new state, so the
    residual and a trial Newton step are those of the steady problem, see
    :func:`check_steady_state`. This replaces comparing consecutive solutions, which
    depends on the time step size.

    Parameters:
        tolerances: Absolute tolerances of variables, see
            :func:`variable_tolerance_vector`.
        earliest_stop_time: The criterion is not evaluated before this time.
        solver: Solver used for trial steps. Created if not given.

    """

    def __init__(
        self,
        tolerances: dict[str, float],
        earliest_stop_time: float = 0,
        solver: pp.solvers.NewtonSolver | None = None,
    ) -> None:
        self.tolerances = tolerances
        self.earliest_stop_time = earliest_stop_time
        self.solver = solver if solver is not None else make_nonlinear_solver()
        self.previous: tuple[np.ndarray, float] | None = None
        """State and time at the previous call, to compute rates of change."""

    def simulation_should_stop(
        self, model: pp.PorePyModel
    ) -> pp.ModelRunnerStatus | None:
        state = model.equation_system.get_variable_values(iterate_index=0)
        time = model.time_manager.time
        rates = None
        if self.previous is not None and time > self.previous[1]:
            rates = np.abs(state - self.previous[0]) / (time - self.previous[1])
        self.previous = (state, time)

        # The check needs a linear solve, so it is only done when the state may be close
        # to a steady state.
        failed = {"not evaluated": np.inf}
        trial_ratios = None
        if time >= self.earliest_stop_time:
            residual_ratios, trial_ratios, failed = check_steady_state(
                model, self.tolerances, self.solver
            )
            logger.info(
                "Relative residual: "
                + ", ".join(f"{k}={v:.1e}" for k, v in residual_ratios.items())
            )
        log_steady_state_convergence(model, state, rates, trial_ratios)
        if failed:
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
                tolerances=tolerances,
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
    """Check that the equations of the model are in equilibrium at its current state:
    their residual is negligible, and a Newton step from the state would not change it.

    """
    residual_ratios, trial_ratios, failed = check_steady_state(model, tolerances)
    log_steady_state_ratios(residual_ratios, trial_ratios)
    if failed:
        logger.warning(
            "Not a steady state, relative residual / trial step too large for: "
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
        tolerances: Absolute tolerances of variables, see
            :func:`variable_tolerance_vector`.
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
