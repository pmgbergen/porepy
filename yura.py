from turtle import mode
from typing import Optional

import numpy as np
import logging
import porepy as pp
from porepy.models.model_runner import _extract_nonlinear_solver_from_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SCHEDULE_INTERVAL_EQUILIBRATION = "100 years"


class MyModel(pp.Thermoporomechanics):
    def set_fractures(self):
        self._fractures = [pp.LineFracture(np.array([[0.25, 0.75], [0.5, 0.5]]))]


class InitializationRunner:
    def __init__(
        self,
        model: pp.PorePyModel,
        time_stepper: Optional[pp.time_stepper.TimeStepper] = None,
        nonlinear_solver: Optional[pp.solvers.NonlinearSolverBase] = None,
    ) -> None:
        self.model = model
        """Model instance passed at instantiation."""

        if time_stepper is None:
            time_stepper = pp.time_stepper.TimeStepper(
                scheduler=pp.time_stepper.assemble_default_time_scheduler(
                    time_manager=model.time_manager
                ),
                max_attempts=10,
            )
        self.time_stepper: pp.time_stepper.TimeStepper = time_stepper
        """Responsible for the time stepping logic. Used only in time-dependent
        simulations."""

        self.solver: pp.solvers.NonlinearSolverBase = (
            _extract_nonlinear_solver_from_params(
                nonlinear_solver=nonlinear_solver,
                params={},
                is_nonlinear_problem=self.model._is_nonlinear_problem(),
            )
        )

    def run(self) -> pp.ModelRunnerStatus:
        equilibration_tol = 1e-8
        while not self.model.time_manager.final_time_reached():
            # Perform the time step.
            time_step_status = self.time_stepper.perform_time_step(
                self.model, self.solver
            )

            # Abort simulation if time step was stopped.
            if isinstance(time_step_status, pp.time_stepper.TimeStepperStatusFailure):
                logger.error(f"Time stepping failed: {time_step_status.reason}")
                return pp.ModelRunnerStatusFailure(reason=time_step_status.reason)
            interval, _ = self.model.time_manager.schedule.get_current_next_intervals(
                time=self.model.time_manager.time
            )
            accepted_solution = self.model.equation_system.get_variable_values(
                iterate_index=0
            )
            if interval.name == SCHEDULE_INTERVAL_EQUILIBRATION:
                metric = pp.VariableBasedEuclideanMetric(model=self.model)
                reference = self.model.equation_system.get_variable_values(
                    reference=True
                )
                diff_norms = metric(values=accepted_solution - reference)

                is_equilibrated = True
                for variable_name, norm in diff_norms.items():
                    if norm >= equilibration_tol:
                        is_equilibrated = False
                if is_equilibrated:
                    return pp.ModelRunnerStatusSuccess()
            # Continue
            # self.model.equation_system.set_variable_values(
            #     values=accepted_solution, reference=True
            # )

            pressure = self.model.equation_system.get_variable_values(
                variables=["pressure"], iterate_index=0
            )
            pressure = self.model.units.convert_units(pressure, "Pa", to_si=True)
            print(f"p_min: {pressure.min():.1e}, p_max: {pressure.max():.1e}")

            temperature = self.model.equation_system.get_variable_values(
                variables=["temperature"], iterate_index=0
            )
            print(f"T_min: {temperature.min():.1e}, T_max: {temperature.max():.1e}")

            disp = self.model.equation_system.get_variable_values(
                variables=["u"], iterate_index=0
            )
            print(f"u_min: {disp.min():.1e}, u_max: {disp.max():.1e}")

            # lamda = self.model.equation_system.get_variable_values(
            #     variables=["contact_traction"], iterate_index=0
            # )
            # print(f"t_min: {lamda.min():.1e}, t_max: {lamda.max():.1e}")

        # Conclude the simulation status.
        return pp.ModelRunnerStatusFailure("Model did not equilibrate after 100 years.")


if __name__ == "__main__":
    model = MyModel()
    # status = pp.ModelRunner(model=model).run()
    # assert status.is_success()

    model.prepare_simulation()

    original_time_manager = model.time_manager

    model.time_manager = pp.TimeManager(
        pp.time_stepper.Schedule(
            intervals=[
                pp.time_stepper.TimeInterval.create(
                    t_start=0,
                    dt_start=pp.SECOND,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                ),
                pp.time_stepper.TimeInterval.create(
                    t_start=2 * pp.SECOND,
                    dt_start=pp.HOUR,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                ),
                pp.time_stepper.TimeInterval.create(
                    t_start=pp.HOUR,
                    dt_start=pp.DAY,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                ),
                pp.time_stepper.TimeInterval.create(
                    t_start=pp.DAY,
                    dt_start=30 * pp.DAY,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                ),
                pp.time_stepper.TimeInterval.create(
                    t_start=30 * pp.DAY,
                    dt_start=pp.YEAR,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                ),
                pp.time_stepper.TimeInterval.create(
                    t_start=pp.YEAR,
                    dt_start=pp.YEAR,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                    name=SCHEDULE_INTERVAL_EQUILIBRATION,
                ),
            ],
            t_end=100 * pp.YEAR,
        )
    )

    initialization_runner = InitializationRunner(model=model)

    initialization_status = initialization_runner.run()
    pass
