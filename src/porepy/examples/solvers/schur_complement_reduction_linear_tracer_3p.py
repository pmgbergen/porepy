"""This example shows how to apply the Schur complement reduction-based linear solver.

The considered model is a simplified model of single-phase, two-component flow. It does
not represent any real physics and should not be used in modeling.

The setup is heavily inspired by `examples/tracer_flow.py`.

"""

import logging

import porepy as pp
from porepy.examples.tracer_flow import TracerFlowModel


def run_example():
    # The same initialization as in examples/tracer_flow.py.

    # Initial time step 60 seconds.
    dt_init = pp.MINUTE
    # Simulation time 20 minutes.
    T_end = 20 * pp.MINUTE
    # min max time step size is 6 seconds and 10 minutes respectively
    dt_min_max = (0.1 * dt_init, 10 * pp.MINUTE)
    # parameters for Newton solver
    max_iterations = 80
    newton_tol = 1e-6
    newton_tol_increment = newton_tol

    time_manager = pp.TimeManager(
        schedule=[0, T_end],
        dt_init=dt_init,
        dt_min_max=dt_min_max,
        iter_optimal_range=(2, 10),
        iter_relax_factors=(0.8, 1.2),
        recomp_factor=0.8,
    )

    params = {
        "material_constants": {
            # Solid with impermeable fractures.
            "solid": pp.SolidConstants(
                porosity=0.1, permeability=1e-7, normal_permeability=1e-19
            ),
        },
        "fracture_indices": [0, 1],
        "time_manager": time_manager,
        "meshing_arguments": {"cell_size": 0.05},
        "grid_type": "simplex",
    }
    model = TracerFlowModel(params)  # type: ignore[abstract]

    # This part is different from examples/tracer_flow.py:
    # Initializing a nonlinear solver with a custom linear solver that applies a Schur
    # complement reduction. It requires us to list the primary variables and equations.
    # The default tags are available for the standard equations and variables in PorePy.
    # This model uses a non-standard equation and variable "z_tracer". Creating custom
    # tags for them.
    nonlinear_solver = pp.solvers.NewtonSolver(
        linear_solver=pp.solvers.SchurComplementReductionLinearSolver(
            primary_equation_tags=[
                pp.solvers.DefaultEquationTags.mass_balance,
                pp.solvers.DefaultEquationTags.interface_darcy_flux,
                pp.solvers.DefaultEquationTags.well_flux,
                pp.solvers.EquationTag(name="component_mass_balance_equation_tracer"),
            ],
            primary_variable_tags=[
                pp.solvers.DefaultVariableTags.pressure,
                pp.solvers.DefaultVariableTags.interface_darcy_flux,
                pp.solvers.DefaultVariableTags.well_flux,
                pp.solvers.VariableTag(name="z_tracer"),
            ],
            # Using a direct linear solver as the inner algorithm. It can be replaced
            # with an iterative solver.
            primary_linear_solver=pp.solvers.LinearSolverDirect(),
        ),
        params={
            "nl_max_iterations": max_iterations,
            "nl_convergence_inc_atol": newton_tol_increment,
            "nl_convergence_res_atol": newton_tol,
        },
    )
    runner = pp.ModelRunner(model, nonlinear_solver=nonlinear_solver)
    runner.run()
    return [model]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_example()
