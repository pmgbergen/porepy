"""Example shows how to run a simulation defined in `examples/geothermal_reservoir.py`
with an sequential nonlinear solver that decouples from each other:
- elasticity and contact mechanics;
- energy and mass balance.

Known issues:
- After time stepper hits the schedule, dt is unreasonably small.
- With a small dt, pressure-energy newton solver experience problems. YZ guesses it is
  due to bad energy scaling in lower dimension, but it needs a proper investigation.

"""

import logging


import porepy as pp
from porepy.examples.geothermal_reservoir import (
    GeothermalReservoirWellBCs,
    set_model_params,
)


def run_example() -> list[pp.PorePyModel]:
    """Run the geothermal reservoir example and return the model."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy.numerics.solvers.nonlinear_solvers").setLevel(
        logging.WARNING
    )
    logging.getLogger("porepy.numerics.solvers.linear_solvers.linear_solver").setLevel(
        logging.WARNING
    )
    nonlinear_solver = pp.solvers.SequentialNonlinearSolver(
        max_iterations=25,
        convergence_criteria=pp.solvers.assemble_default_convergence_criteria(
            is_nonlinear_problem=True,
            inc_atol=1e-7,
            inc_rtol=1e-10,
            res_atol=1e-7,
            res_rtol=1e-10,
            metric=pp.EuclideanMetric(),
        ),
        divergence_criteria=pp.solvers.assemble_default_divergence_criteria(
            is_nonlinear_problem=True,
            max_iterations=25,
            inc_div_atol=1e12,
            res_div_atol=1e12,
            metric=pp.EuclideanMetric(),
        ),
        subsolvers=[
            pp.solvers.NewtonSolver(
                params={
                    "nl_max_iterations": 25,
                    "nl_convergence_inc_atol": 1e-7,
                    "nl_convergence_res_atol": 1e-7,
                    "nl_divergence_inc_atol": 1e12,
                    "nl_divergence_res_atol": 1e12,
                },
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
            pp.solvers.ConstraintLineSearchNonlinearSolver(
                params={
                    "nl_max_iterations": 25,
                    "nl_convergence_inc_atol": 1e-7,
                    "nl_convergence_res_atol": 1e-7,
                    "nl_divergence_inc_atol": 1e50,
                    "nl_divergence_res_atol": 1e50,
                    "global_line_search": 1,
                    "local_line_search": 0,
                },
                equation_tags=[
                    pp.solvers.DefaultEquationTags.mass_balance,
                    pp.solvers.DefaultEquationTags.interface_darcy_flux,
                    pp.solvers.DefaultEquationTags.well_flux,
                    pp.solvers.DefaultEquationTags.energy_balance,
                    pp.solvers.DefaultEquationTags.interface_fourier_flux,
                    pp.solvers.DefaultEquationTags.interface_enthalpy_flux,
                    pp.solvers.DefaultEquationTags.well_enthalpy_flux,
                ],
                variable_tags=[
                    pp.solvers.DefaultVariableTags.pressure,
                    pp.solvers.DefaultVariableTags.interface_darcy_flux,
                    pp.solvers.DefaultVariableTags.well_flux,
                    pp.solvers.DefaultVariableTags.temperature,
                    pp.solvers.DefaultVariableTags.interface_fourier_flux,
                    pp.solvers.DefaultVariableTags.interface_enthalpy_flux,
                    pp.solvers.DefaultVariableTags.well_enthalpy_flux,
                ],
            ),
        ],
    )
    models: list[pp.PorePyModel] = []
    model = GeothermalReservoirWellBCs(set_model_params())
    model.time_manager.dt_min_max = (
        0.1 * pp.SECOND,  # decreased the lower bound.
        max(pp.YEAR, model.time_manager.dt_init),
    )
    model.time_manager.iter_optimal_range = (10, 15)
    # Increasing target interval because sequential solver requires more iteration.
    pp.ModelRunner(model, nonlinear_solver=nonlinear_solver).run()
    models.append(model)
    return models


if __name__ == "__main__":
    run_example()
