from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class SteadyStateModelRunnerSuccess(pp.ModelRunnerStatusSuccess):
    pass


class InitializationError(RuntimeError):
    """Raised if the initialization did not end in a verified steady state.

    Attributes:
        status: Status of the initialization run, if available.
        residual_norms: Residual norms per equation, if they were evaluated.

    """

    def __init__(
        self,
        message: str,
        status: pp.ModelRunnerStatus | None = None,
        residual_norms: dict[str, float] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.residual_norms = residual_norms


class SteadyStateEarlyStopCriterion(pp.EarlyStopCriterion):
    def __init__(
        self,
        metric: pp.Metric,
        tolerances: dict[str, float],
        earliest_stop_time: float = 0,
        default_tolerance: float = 1.0,
        variable_tags: Optional[list[pp.solvers.VariableTag]] = None,
    ) -> None:
        self.metric = metric
        self.tolerances = tolerances
        self.default_tolerance = default_tolerance
        self.previous_solution: np.ndarray | None = None
        self.variable_tags = variable_tags
        self.earliest_stop_time = earliest_stop_time

    def simulation_should_stop(
        self, model: pp.PorePyModel
    ) -> pp.ModelRunnerStatus | None:
        variables = None
        if self.variable_tags is not None:
            variables = model.equation_system.variable_indexer.filter_by_tags(
                model=model, tags=self.variable_tags
            )

        if self.previous_solution is None:
            self.previous_solution = model.equation_system.get_variable_values(
                variables=variables, time_step_index=0
            )
            return None

        current_solution = model.equation_system.get_variable_values(
            variables=variables, time_step_index=0
        )
        norms = self.metric(current_solution - self.previous_solution)

        self.previous_solution = current_solution

        for unknown_key in set(self.tolerances).difference(norms):
            logger.warning(
                "SteadyStateEarlyStopCriterion has an unknown key that is ignored: "
                f"{unknown_key}"
            )

        steady_state_data = {}
        for key, norm in norms.items():
            values = model.equation_system.get_variable_values(
                variables=[key], time_step_index=0
            )
            equilibrated = norm < self.tolerances.get(key, self.default_tolerance)
            steady_state_data[key] = {
                "norm": norm,
                "min": values.min(),
                "max": values.max(),
                "equilibrated": equilibrated,
            }

        log_steady_state_convergence(steady_state_data)
        if (
            all(variable["equilibrated"] for variable in steady_state_data.values())
            and model.time_manager.time >= self.earliest_stop_time
        ):
            return SteadyStateModelRunnerSuccess()
        return None


def log_steady_state_convergence(steady_state_data: dict) -> None:
    def signed(value: float) -> str:
        return f"{'-' if value < 0 else ' '}{abs(value):.1e}"

    max_symbols_offset = max(len(key) for key in steady_state_data)
    for key, data in steady_state_data.items():
        logger.info(
            f"{key}:{' ' * (1 + max_symbols_offset - len(key))}"
            f"min ={signed(data['min'])}, max ={signed(data['max'])}, "
            f"Δ ={signed(data['norm'])}{', converged' if data['equilibrated'] else ''}"
        )


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

    nonlinear_solver = pp.solvers.NewtonSolver(
        params=make_solver_params(),
        linear_solver=pp_solvers.IterativeLinearSolver(),
    )

    # model.prepare_simulation()
    initialization_runner = pp.ModelRunner(
        model,
        nonlinear_solver=nonlinear_solver,
        early_stop_criteria=[
            SteadyStateEarlyStopCriterion(
                metric=pp.VariableBasedEuclideanMetric(model=model),
                tolerances={
                    "pressure": model.units.convert_units(100, "Pa"),  # 10
                    "temperature": model.units.convert_units(1, "K"),  # 1e-2
                    "u": model.units.convert_units(1e-2, "m"),
                    "unknown_key": -1,
                },
                default_tolerance=1e-3,
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

        steady_state = model.equation_system.get_variable_values(time_step_index=0)
        model.equation_system.set_variable_values(reference=True, values=steady_state)
        model.equation_system.set_variable_values(
            time_step_index=0, values=steady_state
        )
        model.equation_system.set_variable_values(iterate_index=0, values=steady_state)
        # Anchor the boundary values at the steady state, so that the boundary
        # contribution to mechanical_stress and displacement_divergence cancels there.
        model.set_boundary_reference_values()
    finally:
        restore_original_model()

    # Reset equations and rediscretize.
    for eq_name in list(model.equation_system.equations.keys()):
        model.equation_system.remove_equation(eq_name)

    # Parts of model.prepare_simulation
    model.set_equations()
    model.update_discretization_parameters()
    model.discretize()
    model.set_nonlinear_discretizations()

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

    residual = model.equation_system.assemble(evaluate_jacobian=False)
    residual_metric = pp.EquationBasedEuclideanMetric(model=model)
    residual_norms = residual_metric(residual)
    residual_tol = 1e-9
    failed = {
        name: norm for name, norm in residual_norms.items() if not norm <= residual_tol
    }
    if failed:
        for equation_name, norm in failed.items():
            logger.error(
                f"Equilibration failed for {equation_name = }, residual {norm = :.1e}."
            )
        raise InitializationError(
            f"Residual of the original equations at the found steady state exceeds "
            f"{residual_tol:.1e} for: {', '.join(failed)}.",
            status=status,
            residual_norms=residual_norms,
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
        nonlinear_solver=pp.solvers.NewtonSolver(
            params=make_solver_params(),
            linear_solver=pp_solvers.IterativeLinearSolver(),
        ),
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
