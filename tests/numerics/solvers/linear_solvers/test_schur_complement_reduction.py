"""Unit tests for the Schur-complement reduction linear solver."""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.applications.md_grids.mdg_library import (
    cube_with_orthogonal_fractures,
    square_with_orthogonal_fractures,
)
from porepy.applications.test_utils.models import add_mixin
from porepy.numerics.solvers.linear_solvers.schur_complement_reduction import (
    _filter_by_tags,
)
from tests.functional.setups.linear_tracer import TracerFlowModel_3p


class MockPrimaryLinearSolver(pp.solvers.LinearSolverBase):
    """Return a prescribed result and record calls made by the outer solver."""

    def __init__(
        self,
        solution: np.ndarray,
        status: pp.solvers.LinearSolverStatus,
    ) -> None:
        self.solution = solution
        self.status = status
        self.linear_systems: list[pp.solvers.LinearSystem] = []
        self.model: object | None = None

    def initialize_with_model(self, model: pp.PorePyModel) -> None:
        self.model = model

    def solve_linear_system(
        self, linear_system: pp.solvers.LinearSystem
    ) -> tuple[np.ndarray, pp.solvers.LinearSolverStatus]:
        self.linear_systems.append(linear_system)
        return self.solution.copy(), self.status


@dataclass
class LinearSystemData:
    """The algebraic system and the tags identifying its primary block."""

    linear_system: pp.solvers.LinearSystem
    primary_equation_tag: pp.solvers.EquationTag
    primary_variable_tag: pp.solvers.VariableTag


@pytest.fixture
def linear_system_data() -> LinearSystemData:
    """Construct a tagged block system without constructing a PorePy model."""
    domain = pp.CartGrid([1])

    secondary_equation_0 = pp.ad.EquationOnDomain("secondary_0", domain)
    primary_equation = pp.ad.EquationOnDomain("primary", domain)
    secondary_equation_1 = pp.ad.EquationOnDomain("secondary_1", domain)
    equation_indexer = pp.ad.EquationIndexer(
        equation_dofs={
            secondary_equation_0: np.array([0]),
            primary_equation: np.array([1, 3]),
            secondary_equation_1: np.array([2]),
        }
    )

    secondary_variable_0 = pp.ad.Variable("secondary_0", {"cells": 1}, domain)
    primary_variable = pp.ad.Variable("primary", {"cells": 1}, domain)
    secondary_variable_1 = pp.ad.Variable("secondary_1", {"cells": 1}, domain)
    variable_indexer = pp.ad.VariableIndexer(
        variable_dofs={
            secondary_variable_0: np.array([0]),
            primary_variable: np.array([1, 3]),
            secondary_variable_1: np.array([2]),
        }
    )

    matrix = csr_matrix(
        [
            [4.0, 2.0, 1.0, 0.0],
            [1.0, 5.0, 0.0, 2.0],
            [1.0, 1.0, 3.0, 1.0],
            [0.0, 2.0, 2.0, 6.0],
        ]
    )
    # This right-hand side is matrix @ [1, -1, 2, 3].
    rhs = np.array([4.0, 2.0, 9.0, 20.0])

    return LinearSystemData(
        linear_system=pp.solvers.LinearSystem(
            matrix=matrix,
            rhs=rhs,
            equation_indexer=equation_indexer,
            variable_indexer=variable_indexer,
        ),
        primary_equation_tag=pp.solvers.EquationTag("primary"),
        primary_variable_tag=pp.solvers.VariableTag("primary"),
    )


def test_filter_by_tags_handles_duplicate_tags() -> None:
    """Must raise ValueError if tags lead to duplicated operators."""

    domain = pp.CartGrid([1])
    variable = pp.ad.Variable("x", {"cells": 1}, domain)
    tags = [pp.solvers.VariableTag("x"), pp.solvers.VariableTag("x")]

    with pytest.raises(ValueError):
        _ = _filter_by_tags([variable], tags)


def test_filter_by_tags_allows_disjoint_tags_with_same_name() -> None:
    """Tags with the same name may select disjoint subsets of domains."""

    @dataclass(frozen=True)
    class OnDomain(pp.solvers.DomainFilter):
        domain: pp.GridLike

        def filter(self, domain: pp.GridLike) -> bool:
            return domain is self.domain

    domains = [pp.CartGrid([1]) for _ in range(3)]
    variables = [pp.ad.Variable("x", {"cells": 1}, domain) for domain in domains]
    tags = [
        pp.solvers.VariableTag("x", defined_on=OnDomain(domain))
        for domain in domains[:2]
    ]

    filtered, not_filtered = _filter_by_tags(variables, tags)

    assert filtered == variables[:2]
    assert not_filtered == variables[2:]


def test_default_primary_solver() -> None:
    """A direct solver is created when no primary solver is supplied."""
    solver = pp.solvers.SchurComplementReductionLinearSolver(
        primary_equation_tags=[],
        primary_variable_tags=[],
    )

    assert isinstance(solver.primary_linear_solver, pp.solvers.LinearSolverDirect)


def test_initialize_with_model_is_forwarded_to_primary_solver() -> None:
    """Initialization only forwards a shallow model mock to the inner solver."""

    class ShallowMockModel:
        pass

    primary_solver = MockPrimaryLinearSolver(
        solution=np.empty(0),
        status=pp.solvers.LinearSolverStatusSuccess(solve_time=0.0),
    )
    solver = pp.solvers.SchurComplementReductionLinearSolver(
        primary_equation_tags=[],
        primary_variable_tags=[],
        primary_linear_solver=primary_solver,
    )
    model = ShallowMockModel()

    solver.initialize_with_model(model)  # type: ignore[arg-type]

    assert primary_solver.model is model


@pytest.mark.parametrize("primary_solver_succeeds", [True, False])
def test_solve_reduces_system_and_wraps_primary_status(
    linear_system_data: LinearSystemData,
    primary_solver_succeeds: bool,
) -> None:
    """The inner solver receives the Schur system and its status is preserved."""
    if primary_solver_succeeds:
        primary_status: pp.solvers.LinearSolverStatus = (
            pp.solvers.LinearSolverStatusSuccess(solve_time=1.0)
        )
    else:
        primary_status = pp.solvers.LinearSolverStatusFailure(reason="mock failure")
    primary_solver = MockPrimaryLinearSolver(
        solution=np.array([-1.0, 3.0]),
        status=primary_status,
    )
    solver = pp.solvers.SchurComplementReductionLinearSolver(
        primary_equation_tags=[linear_system_data.primary_equation_tag],
        primary_variable_tags=[linear_system_data.primary_variable_tag],
        primary_linear_solver=primary_solver,
    )

    solution, status = solver.solve_linear_system(linear_system_data.linear_system)

    np.testing.assert_allclose(solution, [1.0, -1.0, 2.0, 3.0])
    assert len(primary_solver.linear_systems) == 1
    reduced_system = primary_solver.linear_systems[0]
    assert reduced_system.matrix is not None
    np.testing.assert_allclose(
        reduced_system.matrix.toarray(),
        np.array([[50.0, 23.0], [18.0, 58.0]]) / 11.0,
    )
    np.testing.assert_allclose(reduced_system.rhs, np.array([19.0, 156.0]) / 11.0)
    assert reduced_system.equation_indexer.projection_indices(
        list(reduced_system.equation_indexer.equation_dofs)
    ).tolist() == [0, 1]
    assert reduced_system.variable_indexer.projection_indices(
        list(reduced_system.variable_indexer.variable_dofs)
    ).tolist() == [0, 1]

    assert status.primary_solver_status is primary_status
    if primary_solver_succeeds:
        assert isinstance(status, pp.solvers.SchurComplementReductionStatusSuccess)
        assert status.solve_time >= 0.0
    else:
        assert isinstance(status, pp.solvers.SchurComplementReductionStatusFailure)
        assert status.reason == "primary linear solver failed"


def test_solve_delegates_when_secondary_block_is_empty(
    linear_system_data: LinearSystemData,
) -> None:
    """A fully primary system is passed unchanged to the primary solver."""
    primary_status = pp.solvers.LinearSolverStatusSuccess(solve_time=1.0)
    expected_solution = np.array([1.0, -1.0, 2.0, 3.0])
    primary_solver = MockPrimaryLinearSolver(expected_solution, primary_status)
    equation_tags = [
        pp.solvers.EquationTag(equation.name)
        for equation in linear_system_data.linear_system.equation_indexer.equation_dofs
    ]
    variable_tags = [
        pp.solvers.VariableTag(variable.name)
        for variable in linear_system_data.linear_system.variable_indexer.variable_dofs
    ]
    solver = pp.solvers.SchurComplementReductionLinearSolver(
        primary_equation_tags=equation_tags,
        primary_variable_tags=variable_tags,
        primary_linear_solver=primary_solver,
    )

    solution, status = solver.solve_linear_system(linear_system_data.linear_system)

    np.testing.assert_array_equal(solution, expected_solution)
    assert primary_solver.linear_systems == [linear_system_data.linear_system]
    assert isinstance(status, pp.solvers.SchurComplementReductionStatusSuccess)
    assert status.primary_solver_status is primary_status


def test_solve_requires_matrix(linear_system_data: LinearSystemData) -> None:
    """A released matrix is rejected before data initialization."""
    linear_system_data.linear_system.release_matrix_reference()
    solver = pp.solvers.SchurComplementReductionLinearSolver(
        primary_equation_tags=[linear_system_data.primary_equation_tag],
        primary_variable_tags=[linear_system_data.primary_variable_tag],
    )

    with pytest.raises(AssertionError, match="Matrix should be provided"):
        solver.solve_linear_system(linear_system_data.linear_system)


# ---------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------


def _get_primary_equ_and_vars_cf(
    model: pp.PorePyModel,
) -> tuple[list[pp.solvers.EquationTag], list[pp.solvers.VariableTag]]:
    """Return the primary equation and variable tags of a CF model."""

    equ_tags: list[pp.solvers.EquationTag] = []
    var_tags: list[pp.solvers.VariableTag] = []
    # Overall mass balance.
    if isinstance(model, pp.fluid_mass_balance.FluidMassBalanceEquations):
        equ_tags += [pp.solvers.DefaultEquationTags.mass_balance]
        var_tags += [pp.solvers.DefaultVariableTags.pressure]

    # Eergy balance. Can have either enthalpy or temperature as a primary variable.
    if isinstance(model, pp.energy_balance.TotalEnergyBalanceEquations):
        equ_tags += [pp.solvers.DefaultEquationTags.energy_balance]
    if isinstance(
        model, pp.compositional_flow.SolutionStrategyExtendedFluidMassAndEnergy
    ):
        var_tags += [pp.solvers.VariableTag(name=model.enthalpy_variable)]
    elif isinstance(model, pp.energy_balance.SolutionStrategyEnergyBalance):
        var_tags += [pp.solvers.VariableTag(name=model.temperature_variable)]

    # Individual component mass balances.
    if isinstance(model, pp.compositional_flow.ComponentMassBalanceEquations):
        equ_tags += [
            pp.solvers.EquationTag(name=name)
            for name in model.component_mass_balance_equation_names()
        ]
    if isinstance(model, pp.compositional.CompositionalVariables):
        var_tags += [
            pp.solvers.VariableTag(name=name)
            for name in model.overall_fraction_variables
        ]
        var_tags += [
            pp.solvers.VariableTag(name=name)
            for name in model.tracer_fraction_variables
        ]

    # Fluxes.
    for eq_name in model.equation_system.equations.keys():
        if "_flux" in eq_name:
            equ_tags.append(pp.solvers.EquationTag(name=eq_name))
    variable_names = {var.name for var in model.equation_system.variables}
    for var_name in variable_names:
        if "_flux" in var_name:
            var_tags.append(pp.solvers.VariableTag(name=var_name))

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
        [pp.PorePyModel],
        tuple[list[pp.solvers.EquationTag], list[pp.solvers.VariableTag]],
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
