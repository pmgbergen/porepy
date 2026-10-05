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
    assert system.matrix is not None
    ratios = _safe_ratio(system.rhs, abs(system.matrix) @ tolerance_vector)
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


@dataclass
class SteadyStateModelRunnerSuccess(pp.ModelRunnerStatusSuccess):
    pass


class InitializationError(RuntimeError):
    """Raised if the initialization did not end in a verified steady state.

    Attributes:
        status: Status of the initialization run, if available.
        residual_ratios: Relative residuals per equation, if they were evaluated, see
            :func:`steady_state_residual_ratios`.
        trial_step_ratios: Relative Newton increments per variable, if evaluated.

    """

    def __init__(
        self,
        message: str,
        status: pp.ModelRunnerStatus | None = None,
        residual_ratios: dict[str, float] | None = None,
        trial_step_ratios: dict[str, float] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.residual_ratios = residual_ratios
        self.trial_step_ratios = trial_step_ratios


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

    def simulation_should_stop(
        self, model: pp.PorePyModel
    ) -> pp.ModelRunnerStatus | None:
        if model.time_manager.time < self.earliest_stop_time:
            return None
        residual_ratios, trial_ratios, failed = check_steady_state(
            model, self.tolerances, self.solver
        )
        log_steady_state_ratios(residual_ratios, trial_ratios)
        if failed:
            return None
        return SteadyStateModelRunnerSuccess()


def initialization_pipeline(model: pp.PorePyModel) -> pp.PorePyModel:
    original_model_class = model.__class__
    original_time_manager = model.time_manager
    original_folder_name = model.params["folder_name"]
    # Time at which boundary conditions and sources are evaluated during initialization.
    # The initialization clock runs far beyond the simulation's start, but the reference
    # state must match the data seen by the first time step of the main simulation. This
    # is the first target time rather than the start time, since e.g. lithostatic
    # boundary conditions are different at the initial time itself.
    data_time = model.params.get(
        "initialization_data_time",
        original_time_manager.time + original_time_manager.dt,
    )
    initialization_folder_name = Path(original_folder_name).with_name(
        Path(original_folder_name).name + "_initialization"
    )

    def restore_original_model() -> None:
        model.__class__ = original_model_class
        model.time_manager = original_time_manager
        model.params["folder_name"] = original_folder_name
        # The exporter, the iteration exporter and the solver statistics were created
        # in prepare_simulation with the initialization folder. Recreate them, so that
        # the main simulation writes to the original folder.
        model.set_nonlinear_solver_statistics()
        model.initialize_data_saving()

    class InitializedModel(original_model_class):
        def shear_dilation_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            return pp.ad.Scalar(0)

        def matrix_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            ones = np.ones(sum(sd.num_cells for sd in subdomains))
            return self.reference_porosity(subdomains) * pp.ad.DenseArray(ones)

        def update_time_dependent_ad_arrays(self) -> None:
            # Time dependent data (boundary values, sources) is frozen at data_time,
            # while the clock of the time manager advances to find the steady state.
            clock_time = self.time_manager.time
            self.time_manager.time = data_time
            try:
                super().update_time_dependent_ad_arrays()
            finally:
                self.time_manager.time = clock_time

    model.__class__ = InitializedModel
    model.time_manager = make_initialization_time_manager()
    model.params["folder_name"] = initialization_folder_name

    nonlinear_solver = make_nonlinear_solver()
    tolerances = make_tolerances(model)

    # model.prepare_simulation()
    initialization_runner = pp.ModelRunner(
        model,
        nonlinear_solver=nonlinear_solver,
        early_stop_criteria=[
            SteadyStateEarlyStopCriterion(
                tolerances=tolerances,
                earliest_stop_time=1 * pp.YEAR,
            )
        ],
    )
    try:
        try:
            status = initialization_runner.run()
        except RuntimeError as e:
            if not (e.args and isinstance(e.args[0], pp.ModelRunnerStatus)):
                raise
            status = e.args[0]

        # The reference state is only overwritten if a steady state was found: reaching
        # the final time or a failed time step is not an equilibrium.
        if not isinstance(status, SteadyStateModelRunnerSuccess):
            raise InitializationError(
                f"Steady state was not reached: {status}. The model state is "
                "not initialized.",
                status=status,
            )

        for matrix_domain, data in model.mdg.subdomains(
            dim=model.nd, return_data=True
        ):
            domains = [matrix_domain]
            # Full stress at the steady state. Boundary reference values are not set
            # yet, so the boundary contribution is included in absolute terms.
            stress_val = model.equation_system.evaluate(model.stress(domains))

            pp.set_solution_values(
                name=model.reference_stress_key,
                values=stress_val,
                data=data,
                reference=True,
            )

        steady_state = model.equation_system.get_variable_values(iterate_index=0)
        model.equation_system.set_variable_values(reference=True, values=steady_state)
        # Copy the state to all stored iterates and time steps.
        model.initialize_previous_iterate_and_time_step_values()
        # Anchor the boundary values at the steady state, so that the boundary
        # contribution to mechanical_stress and displacement_divergence cancels there.
        model.set_boundary_reference_values()
    finally:
        restore_original_model()

    # Discard the equations of the initialization model and set the original ones.
    model.rebuild_equations()
    # Fluxes, upwind discretizations etc. of the original equations depend on the state.
    model.update_derived_quantities()

    for domain in model.mdg.subdomains(dim=model.nd):
        boundary_faces = domain.get_all_boundary_faces()
        internal_faces = domain.get_internal_faces()
        domains = [domain]

        stress = model.equation_system.evaluate(
            model.stress(domains),
        ).reshape((model.nd, -1), order="F")
        reference_stress = model.equation_system.evaluate(
            model.reference_stress(domains)
        ).reshape((model.nd, -1), order="F")
        mechanical_stress = model.equation_system.evaluate(
            model.mechanical_stress(domains)
        ).reshape((model.nd, -1), order="F")
        pressure_stress = model.equation_system.evaluate(
            model.pressure_stress(domains)
        ).reshape((model.nd, -1), order="F")
        thermal_stress = model.equation_system.evaluate(
            model.thermal_stress(domains)
        ).reshape((model.nd, -1), order="F")
        diff = model.equation_system.evaluate(
            model.porosity(domains)
        ) - model.equation_system.evaluate(model.reference_porosity(domains))

    # Validate that the original equations are in equilibrium at the found state:
    # their residual is negligible, and a Newton step from the state would not change it.
    residual_ratios, trial_ratios, failed = check_steady_state(model, tolerances)
    log_steady_state_ratios(residual_ratios, trial_ratios)
    if failed:
        raise InitializationError(
            "The original equations are not in equilibrium at the found state, "
            "relative residual / trial step too large for: "
            + ", ".join(f"{k} ({v:.1e})" for k, v in failed.items()),
            status=status,
            residual_ratios=residual_ratios,
            trial_step_ratios=trial_ratios,
        )
    logger.info(
        "Initialization complete. The simulation initial and reference states are set "
        "to the found steady state."
    )

    # prepare_simulation is skipped in the main run, so export the initial state here.
    if model._is_time_dependent():
        model.save_data_time_step()
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

    model = initialization_pipeline(model)

    run_main_simulation(model)

    pass
