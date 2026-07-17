from typing import Callable
import numpy as np

import pytest

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.applications.md_grids.mdg_library import (
    cube_with_orthogonal_fractures,
    square_with_orthogonal_fractures,
)
from porepy.applications.test_utils.models import add_mixin
from tests.functional.setups.linear_tracer import TracerFlowModel_3p
# ---------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------


def _get_primary_equ_and_vars_cf(
    model: pp.PorePyModel,
) -> tuple[list[pp.EquationTag], list[pp.VariableTag]]:
    """Return the primary equation and variable tags of a CF model."""

    equ_tags: list[pp.EquationTag] = []
    var_tags: list[pp.VariableTag] = []
    # Overall mass balance.
    if isinstance(model, pp.fluid_mass_balance.FluidMassBalanceEquations):
        equ_tags += [pp.DefaultEquationTags.mass_balance]
        var_tags += [pp.DefaultVariableTags.pressure]

    # Eergy balance. Can have either enthalpy or temperature as a primary variable.
    if isinstance(model, pp.energy_balance.TotalEnergyBalanceEquations):
        equ_tags += [pp.DefaultEquationTags.energy_balance]
    if isinstance(
        model, pp.compositional_flow.SolutionStrategyExtendedFluidMassAndEnergy
    ):
        var_tags += [pp.VariableTag(name=model.enthalpy_variable)]
    elif isinstance(model, pp.energy_balance.SolutionStrategyEnergyBalance):
        var_tags += [pp.VariableTag(name=model.temperature_variable)]

    # Individual component mass balances.
    if isinstance(model, pp.compositional_flow.ComponentMassBalanceEquations):
        equ_tags += [
            pp.EquationTag(name=name)
            for name in model.component_mass_balance_equation_names()
        ]
    if isinstance(model, pp.compositional.CompositionalVariables):
        var_tags += [
            pp.VariableTag(name=name) for name in model.overall_fraction_variables
        ]
        var_tags += [
            pp.VariableTag(name=name) for name in model.tracer_fraction_variables
        ]

    # Fluxes.
    for eq_name in model.equation_system.equations.keys():
        if "_flux" in eq_name:
            equ_tags.append(pp.EquationTag(name=eq_name))
    for var in model.equation_system.variables:
        if "_flux" in var.name:
            var_tags.append(pp.VariableTag(name=var.name))

    return equ_tags, var_tags


@pytest.mark.parametrize(
    "test_model_class,get_primary_tags",
    [
        (TracerFlowModel_3p, _get_primary_equ_and_vars_cf),
    ],
)
@pytest.mark.parametrize(
    "mdg",
    [
        square_with_orthogonal_fractures("cartesian", {"cell_size": 0.25}, [0, 1])[0],
        cube_with_orthogonal_fractures("cartesian", {"cell_size": 0.25}, [0, 1, 2])[0],
    ],
)
def test_schur_complement_reduction_on_model(
    mdg: pp.MixedDimensionalGrid,
    test_model_class: type[pp.PorePyModel],
    get_primary_tags: Callable[
        [pp.PorePyModel], tuple[list[pp.EquationTag], list[pp.VariableTag]]
    ],
):
    """Compare direct and Schur-complement solutions of a model linear system."""

    # NOTE: Depending on what which model class the tests are performed, the local
    # geometry mixin and model parameters need adaption in order to overwrite the
    # geometry and parametrization already contained within the tested model class.
    # The adaption must be consistent with the mdg's the tests are performed on.
    class LocalGeometry(pp.PorePyModel):
        def create_mdg(self) -> None:
            self.mdg = mdg

        def set_domain(self) -> None:
            self._domain = nd_cube_domain(
                mdg.dim_max(), self.units.convert_units(1.0, "m")
            )

    model_params = {
        "equilibrium_condition": "dummy",
        "meshing_arguments": {
            "cell_size": 0.1,
        },
    }

    model_class = add_mixin(LocalGeometry, test_model_class)

    model = model_class(model_params)
    model.prepare_simulation()
    primary_equation_tags, primary_variable_tags = get_primary_tags(model)
    model.equation_system.set_variable_values(
        np.ones(model.equation_system.num_dofs()), iterate_index=0, time_step_index=0
    )

    model.before_time_step()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    linear_system = model.equation_system.assemble()

    direct_solver = pp.solvers.LinearSolverDirect()
    direct_solution, direct_status = direct_solver.solve_linear_system(linear_system)
    assert direct_status.is_success()

    schur_solver = pp.solvers.SchurComplementReductionLinearSolver(
        primary_equation_tags=primary_equation_tags,
        primary_variable_tags=primary_variable_tags,
        primary_linear_solver=pp.solvers.LinearSolverDirect(),
    )
    schur_solution, schur_status = schur_solver.solve_linear_system(linear_system)
    assert schur_status.is_success()

    np.testing.assert_allclose(schur_solution, direct_solution, atol=1e-9, rtol=0.0)
