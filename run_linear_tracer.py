import porepy as pp
from tests.functional.setups.linear_tracer import SimplePipe2D, TracerFlowModel_3p

import logging

logging.basicConfig(level=logging.INFO)

cell_size, model_class = SimplePipe2D.pipe_length / 10, TracerFlowModel_3p

# Run verification models and retrieve results for three different times
material_constants = {
    "solid": pp.SolidConstants(porosity=1.0, permeability=1.0, residual_aperture=1),
}
time_manager = pp.TimeManager(
    schedule=[0, 10, 30, 80, 100], dt_init=10, constant_dt=True
)
model_params = {
    "material_constants": material_constants,
    "time_manager": time_manager,
    "meshing_arguments": {"cell_size": cell_size},
    "prepare_simulation": False,
    "times_to_export": [],
}

model = model_class(model_params)
if isinstance(model, TracerFlowModel_3p):
    # To create phase fractions as variables and have a representation fo h_mix
    model.params["equilibrium_condition"] = "dummy"

model.prepare_simulation()

# Setting dt and end time schedule according to cfl condition and approximate
# flow velocity. Works only assuming the test does not work with I/O of times.
sd = model.mdg.subdomains()[0]
dt = model.exact_sol.dt_from_cfl(sd)

time_manager = pp.TimeManager(
    schedule=[0, 3 * dt, 6 * dt, 9 * dt, 10 * dt],
    dt_init=dt,
    constant_dt=True,
)
model.ad_time_step.set_value(dt)
model.time_manager = time_manager
nonlinear_solver = pp.solvers.NewtonSolver(
    linear_solver=pp.solvers.SchurComplementReductionLinearSolver(
        primary_equation_tags=[
            pp.DefaultEquationTags.mass_balance,
            pp.DefaultEquationTags.energy_balance,
            pp.EquationTag(name="component_mass_balance_equation_tracer"),
        ],
        primary_variable_tags=[
            pp.DefaultVariableTags.pressure,
            pp.DefaultVariableTags.enthalpy,
            pp.VariableTag(name="z_tracer"),
        ],
        primary_linear_solver=pp.solvers.LinearSolverDirect(),
    )
)
pp.ModelRunner(model, model_params, nonlinear_solver=nonlinear_solver).run()
