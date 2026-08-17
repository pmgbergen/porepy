"""This example shows how to use `SequentialNonlinearSolver`.

It constructs a simple (not physically realistic) poromechanics model and solves it by
decoupling nonlinear solvers for:
- momentum balance;
- fluid mass balance.

This example also shows how to encorporate custom convergence criteria.

"""

import logging

import porepy as pp
from porepy.models.metric import (
    EquationBasedEuclideanMetric,
    EquationBasedLebesgueMetric,
    VariableBasedEuclideanMetric,
)


class FullModel(  # type: ignore
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.Poromechanics,
):
    pass


def run_example():
    logging.basicConfig(level=logging.INFO)

    # Silence the Newton and linear solver convergence info printing, because it becomes
    # poorly readible when both the outer and inner nonlinear solvers log info.
    logging.getLogger("porepy.numerics.solvers.newton_solver").setLevel(logging.WARNING)
    logging.getLogger("porepy.numerics.solvers.linear_solvers.linear_solver").setLevel(
        logging.WARNING
    )

    model_params_2d = {
        "material_constants": {
            "reference_values": {"pressure": 1},
        },
        "fracture_indices": [0, 1],
        "u_north": -0.001,
        "meshing_arguments": {"cell_size": 0.1},
        "time_manager": pp.TimeManager(schedule=[0, 5], dt_init=1, constant_dt=True),
    }
    model_2d = FullModel(model_params_2d)

    # List equations and variables corresponding to the first subsolver (mechanics).
    mechanics_equations = [
        pp.solvers.DefaultEquationTags.momentum_balance,
        pp.solvers.DefaultEquationTags.interface_force_balance,
        pp.solvers.DefaultEquationTags.normal_fracture_deformation,
        pp.solvers.DefaultEquationTags.tangential_fracture_deformation,
    ]
    mechanics_variables = [
        pp.solvers.DefaultVariableTags.displacement,
        pp.solvers.DefaultVariableTags.interface_displacement,
        pp.solvers.DefaultVariableTags.contact_traction,
    ]

    # List equations and variables corresponding to the second subsolver (flow).
    flow_equations = [
        pp.solvers.DefaultEquationTags.mass_balance,
        pp.solvers.DefaultEquationTags.interface_darcy_flux,
        pp.solvers.DefaultEquationTags.well_flux,
    ]
    flow_variables = [
        pp.solvers.DefaultVariableTags.pressure,
        pp.solvers.DefaultVariableTags.interface_darcy_flux,
        pp.solvers.DefaultVariableTags.well_flux,
    ]

    # Set up separate convergence criteria for each subsolver.
    convergence_mechanics = pp.solvers.ConvergenceCriteria(
        {
            "res_rel": pp.solvers.ResidualBasedRelativeCriterion(
                tol=1e-5,
                metric=EquationBasedLebesgueMetric(
                    model=model_2d, equation_tags=mechanics_equations
                ),
            ),
        }
    )
    convergence_flow = pp.solvers.ConvergenceCriteria(
        {
            "inc_abs": pp.solvers.IncrementBasedAbsoluteCriterion(
                tol=1e-5,
                metric=VariableBasedEuclideanMetric(
                    model=model_2d, variable_tags=flow_variables
                ),
            )
        }
    )
    # Set up convergence criteria for the outer solver.
    convergence_full = pp.solvers.ConvergenceCriteria(
        {
            "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                tol=1e-5,
                metric=EquationBasedEuclideanMetric(
                    model=model_2d, equation_tags=mechanics_equations
                ),
            ),
        }
    )

    # Create the sequential solver that takes two subsolvers: Newton for flow and Newton
    # for mechanics.
    nonlinear_solver = pp.solvers.SequentialNonlinearSolver(
        subsolvers=[
            pp.solvers.NewtonSolver(
                equation_tags=mechanics_equations,
                variable_tags=mechanics_variables,
                params={"nl_convergence_criteria": convergence_mechanics},
            ),
            pp.solvers.NewtonSolver(
                equation_tags=flow_equations,
                variable_tags=flow_variables,
                params={"nl_convergence_criteria": convergence_flow},
            ),
        ],
        convergence_criteria=convergence_full,
    )

    pp.ModelRunner(model_2d, nonlinear_solver=nonlinear_solver).run()
    return [model_2d]


if __name__ == "__main__":
    run_example()
