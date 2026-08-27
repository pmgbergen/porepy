"""Unit tests for the sequential nonlinear solver."""

import numpy as np
import pytest

import porepy as pp
from porepy.numerics.ad.indexers import EquationOnDomain
from porepy.numerics.ad.operators import Variable
from porepy.numerics.solvers.convergence_check import ConvergenceStatus
from porepy.numerics.solvers.nonlinear_solvers import NonlinearSolverStatus


class MockEquationSystem:
    """Equation-system mock with an empty solution and residual vectors."""

    variables = []
    equation_indexer = pp.ad.EquationIndexer(indices={})

    def get_variable_values(self, variables, iterate_index):
        return np.zeros(0)

    def assemble(self, evaluate_jacobian, equations):
        assert evaluate_jacobian is False
        return np.zeros(0)


class MockSuccess(pp.solvers.NonlinearSolverStatusConverged):
    """Successful nonlinear-solver status mock."""

    def __init__(self):
        self.convergence_statuses = pp.solvers.ConvergenceStatusCollection()
        self.divergence_statuses = pp.solvers.ConvergenceStatusCollection()

    def number_of_iterations(self) -> int:
        return 1


class MockFailure(pp.solvers.NonlinearSolverStatusFailed):
    """Failed nonlinear-solver status mock."""

    def __init__(self):
        self.convergence_statuses = pp.solvers.ConvergenceStatusCollection()
        self.divergence_statuses = pp.solvers.ConvergenceStatusCollection()

    def number_of_iterations(self) -> int:
        return 1


def _make_var(name: str):
    """Create a variable for tests in this file."""
    return pp.ad.Variable(
        name=name, ndof=pp.ad.GridEntities(cells=1), domain=pp.CartGrid(1)
    )


def _make_eq(name: str):
    """Create an equation for tests in this file."""
    return pp.ad.EquationOnDomain(name=name, domain=pp.CartGrid(1))


class MockInnerSolver(pp.solvers.NonlinearSolverBase):
    """Nonlinear subsolver mock with prescribed success statuses."""

    def __init__(
        self,
        nonlinear_solve_success: list[bool],
        active_equations: list[pp.ad.EquationOnDomain],
        active_variables: list[pp.ad.Variable],
    ) -> None:
        """Store prescribed solve outcomes and active operators."""
        self.nonlinear_solve_success = nonlinear_solve_success
        self.active_equations = active_equations
        self.active_variables = active_variables
        self.solve_index = 0

    def get_active_variables(self, model: pp.PorePyModel) -> list[Variable]:
        return self.active_variables

    def get_active_equations(self, model: pp.PorePyModel) -> list[EquationOnDomain]:
        return self.active_equations

    def solve(self, model: pp.PorePyModel) -> NonlinearSolverStatus:
        """Return the next prescribed solver status."""
        is_success = self.nonlinear_solve_success[self.solve_index]
        self.solve_index += 1

        if is_success:
            return MockSuccess()
        else:
            return MockFailure()


# Variables and equations for the tests.
variables_a = [_make_var(name) for name in ["a1", "a2", "a3"]]
variables_b = [_make_var(name) for name in ["b1", "b2"]]
variables_c = [_make_var(name) for name in ["c1"]]
equations_a = [_make_eq(name) for name in ["A1"]]
equations_b = [_make_eq(name) for name in ["B1", "B2", "B3"]]
equations_c = [_make_eq(name) for name in ["C1", "C2"]]


class MockConvergenceCriterion(pp.solvers.ConvergenceCriterion):
    """Convergence criterion mock that succeeds after a prescribed count."""

    def __init__(self, success_after: int):
        """Set the iteration count needed for convergence."""
        self.success_after = success_after
        self.count = 0

    def check(
        self, *args, **kwargs
    ) -> tuple[ConvergenceStatus, float | dict[str, float]]:
        """Return convergence once the prescribed count is reached."""
        self.count += 1
        if self.count < self.success_after:
            result = ConvergenceStatus.CONTINUE_ITERATING
        else:
            result = ConvergenceStatus.CONVERGED
        return result, 0


@pytest.mark.parametrize("divergence_criteria_provided", [True, False])
def test_failing_sequential_solver(divergence_criteria_provided: bool):
    """Test sequential solver failure when outer convergence is not reached."""
    max_iter = 5
    model = pp.SinglePhaseFlow()
    model.equation_system = MockEquationSystem()
    solver = pp.solvers.SequentialNonlinearSolver(
        max_iterations=max_iter,
        subsolvers=[
            MockInnerSolver(
                nonlinear_solve_success=[True] * max_iter,
                active_equations=equations_a,
                active_variables=variables_a,
            ),
            MockInnerSolver(
                nonlinear_solve_success=[True] * max_iter,
                active_equations=equations_b,
                active_variables=variables_b,
            ),
        ],
        convergence_criteria=pp.solvers.ConvergenceCriteria(
            crit=MockConvergenceCriterion(success_after=max_iter * 2)
        ),
        # Pass None to use default divergence criteria. Otherwise, pass empty divergence
        # criteria.
        divergence_criteria=None
        if divergence_criteria_provided
        else pp.solvers.DivergenceCriteria(),
    )
    status = solver.solve(model=model)

    assert isinstance(status, pp.solvers.SequentialSolverFailed)
    # Number of iterations must be what we expect.
    assert status.number_of_iterations() == max_iter
    # Must fail due to max iterations.
    assert status.divergence_statuses["max_iter"] == ConvergenceStatus.FAILED
    for statuses_per_iteration in status.subsolver_statuses:
        # Must be two subsolvers at each iteration.
        assert len(statuses_per_iteration) == 2
        # Each sub-solver must be successful.
        assert all(x.is_converged() for x in statuses_per_iteration)


def test_sequential_solver_failing_subsolvers():
    """Test outer convergence despite a failing subsolver."""
    iters_to_converge = 3
    model = pp.SinglePhaseFlow()
    model.equation_system = MockEquationSystem()
    solver = pp.solvers.SequentialNonlinearSolver(
        subsolvers=[
            MockInnerSolver(
                nonlinear_solve_success=[True] * iters_to_converge,
                active_equations=equations_a,
                active_variables=variables_a,
            ),
            MockInnerSolver(
                nonlinear_solve_success=[False] * iters_to_converge,
                active_equations=equations_b,
                active_variables=variables_b,
            ),
        ],
        convergence_criteria=pp.solvers.ConvergenceCriteria(
            crit=MockConvergenceCriterion(success_after=iters_to_converge)
        ),
    )
    status = solver.solve(model=model)

    assert isinstance(status, pp.solvers.SequentialSolverConverged)
    # Number of iterations must be what we expect.
    assert status.number_of_iterations() == iters_to_converge
    for statuses_per_iteration in status.subsolver_statuses:
        # Must be two subsolvers at each iteration.
        assert len(statuses_per_iteration) == 2
        # Each sub-solver must fail.
        assert statuses_per_iteration[0].is_converged()
        assert statuses_per_iteration[1].is_failed()


# Parametrized cases cover:
# - Overlapping sets
# - Permutations
# - Empty sets
@pytest.mark.parametrize(
    "active_equations_per_solver",
    [
        [equations_a, equations_b],
        [equations_a + equations_b, equations_b + equations_c],
        [equations_c + equations_b + equations_a, equations_b, [], equations_b],
    ],
)
@pytest.mark.parametrize(
    "active_variables_per_solver",
    [
        [variables_a, variables_b],
        [variables_a + variables_b, variables_b + variables_c],
        [variables_c + variables_a + variables_b, variables_c, [], variables_c],
    ],
)
def test_active_equations_variables(
    active_equations_per_solver: list[list[pp.ad.EquationOnDomain]],
    active_variables_per_solver: list[list[pp.ad.Variable]],
):
    """Test that methods get_active_equations and get_active_variables behave as
    expected.

    """
    solver = pp.solvers.SequentialNonlinearSolver(
        subsolvers=[
            MockInnerSolver(
                nonlinear_solve_success=[True],
                active_equations=active_equations,
                active_variables=active_variables,
            )
            for active_equations, active_variables in zip(
                active_equations_per_solver, active_variables_per_solver
            )
        ],
    )
    model = pp.SinglePhaseFlow()
    model.equation_system = MockEquationSystem()

    # Expected equations and variables should be non-duplicating lists with preserved
    # order.
    actual_active_equations = solver.get_active_equations(model=model)
    expected_active_equations = list(
        dict.fromkeys(eq for eqs in active_equations_per_solver for eq in eqs)
    )
    assert actual_active_equations == expected_active_equations

    actual_active_variables = solver.get_active_variables(model=model)
    expected_active_variables = list(
        dict.fromkeys(var for vars in active_variables_per_solver for var in vars)
    )
    assert actual_active_variables == expected_active_variables
