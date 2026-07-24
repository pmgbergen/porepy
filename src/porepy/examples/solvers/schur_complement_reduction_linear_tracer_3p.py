"""This example shows how to apply the Schur complement reduction-based linear solver.

The considered model a simplified model of 3-phase 2-component flow, which do not
represent any real physics and should not be used in modeling.

The setup is heavily inspired by `tests/functional/test_linear_tracer.py`.

"""

import logging

import porepy as pp
from tests.functional.setups.linear_tracer import SimplePipe2D, TracerFlowModel_3p


def run_example():
    logging.basicConfig(level=logging.INFO)

    model_params = {
        "material_constants": {
            "solid": pp.SolidConstants(
                porosity=1.0, permeability=1.0, residual_aperture=1
            ),
        },
        "meshing_arguments": {"cell_size": SimplePipe2D.pipe_length / 10},
        "prepare_simulation": False,  # Need to do it manually to determine CFL.
        "times_to_export": [],
        "equilibrium_condition": "dummy",
    }

    model = TracerFlowModel_3p(model_params)

    # Setting dt and end time schedule according to cfl condition and approximate
    # flow velocity.
    model.prepare_simulation()
    sd = model.mdg.subdomains()[0]
    dt = model.exact_sol.dt_from_cfl(sd)

    time_manager = pp.TimeManager(
        schedule=[0, 3 * dt, 6 * dt, 9 * dt, 10 * dt],
        dt_init=dt,
        constant_dt=True,
    )
    model.ad_time_step.set_value(dt)
    model.time_manager = time_manager

    # Initializing a nonlinear solver with a custom linear solver that applies a Schur
    # complement reduction. It requires us to list the primary variables and equations.
    # The default tags are available for the standard equations and variables in PorePy.
    # This model uses a non-standard equation and variable "z_tracer". Creating custom
    # tags for them.
    nonlinear_solver = pp.solvers.NewtonSolver(
        linear_solver=pp.solvers.SchurComplementReductionLinearSolver(
            primary_equation_tags=[
                pp.solvers.DefaultEquationTags.mass_balance,
                pp.solvers.DefaultEquationTags.energy_balance,
                pp.solvers.EquationTag(name="component_mass_balance_equation_tracer"),
            ],
            primary_variable_tags=[
                pp.solvers.DefaultVariableTags.pressure,
                pp.solvers.DefaultVariableTags.enthalpy,
                pp.solvers.VariableTag(name="z_tracer"),
            ],
            # Using a direct linear solver as the inner algorithm. It can be replaced
            # with an iterative solver.
            primary_linear_solver=pp.solvers.LinearSolverDirect(),
        )
    )
    pp.ModelRunner(model, model_params, nonlinear_solver=nonlinear_solver).run()


if __name__ == "__main__":
    run_example()
