"""This example shows how to use `SequentialNonlinearSolver`.

It constructs a simple (not physically realistic) poromechanics model and solves it by
decoupling nonlinear solvers for:
- momentum balance;
- fluid mass balance.

"""

import logging

import porepy as pp


class FullModel(  # type: ignore
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.Poromechanics,
):
    pass


def run_example():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy.numerics.solvers.nonlinear_solvers").setLevel(
        logging.WARNING
    )
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
    }
    model_2d = FullModel(model_params_2d)

    nonlinear_solver = pp.solvers.SequentialNonlinearSolver(
        subsolvers=[
            pp.solvers.NewtonSolver(
                equation_tags=[
                    pp.solvers.DefaultEquationTags.momentum_balance,
                    pp.solvers.DefaultEquationTags.interface_force_balance,
                    pp.solvers.DefaultEquationTags.normal_fracture_deformation,
                    pp.solvers.DefaultEquationTags.tangential_fracture_deformation,
                ],
                variable_tags=[
                    pp.solvers.DefaultVariableTags.displacement,
                    pp.solvers.DefaultVariableTags.interface_displacement,
                    pp.solvers.DefaultVariableTags.contact_traction,
                ],
            ),
            pp.solvers.NewtonSolver(
                equation_tags=[
                    pp.solvers.DefaultEquationTags.mass_balance,
                    pp.solvers.DefaultEquationTags.interface_darcy_flux,
                    pp.solvers.DefaultEquationTags.well_flux,
                ],
                variable_tags=[
                    pp.solvers.DefaultVariableTags.pressure,
                    pp.solvers.DefaultVariableTags.interface_darcy_flux,
                    pp.solvers.DefaultVariableTags.well_flux,
                ],
            ),
        ]
    )

    pp.ModelRunner(model_2d, nonlinear_solver=nonlinear_solver).run()
    return [model_2d]


if __name__ == "__main__":
    run_example()
