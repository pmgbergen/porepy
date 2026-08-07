"""Unit tests for the Newton solver."""

import copy
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from deepdiff import DeepDiff
from scipy.sparse import csr_matrix

import porepy as pp
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.numerics.solvers.convergence_check import (
    ConvergenceInfoHistory,
    ConvergenceStatus,
    ConvergenceStatusCollection,
    check_convergence,
)
from porepy.numerics.solvers.newton_solver import (
    NewtonSolverConverged,
    NewtonSolverFailed,
    _summarize_solver_status,
)
from porepy.numerics.solvers.nonlinear_solvers import (
    NonlinearSolverStatus,
    NonlinearSolverStatusConverged,
    NonlinearSolverStatusFailed,
)
from porepy.time_stepper.time_step_status import (
    # TimeStepperStatusContinueIterating,
    TimeStepperStatusFailure,
    TimeStepperStatusSuccess,
)
from porepy.viz import solver_statistics

# ! ---- Auxiliary fixtures and classes ---- ! #


def linear_solver_statuses(num_statuses: int):
    return [pp.solvers.LinearSolverStatusSuccess(solve_time=1.0)] * num_statuses


def time_step_success() -> TimeStepperStatusSuccess:
    """Create a successful time-step status for statistics tests."""
    return TimeStepperStatusSuccess(
        time=1.0,
        dt=0.5,
        nonlinear_solver_status=NewtonSolverConverged(
            linear_solver_statuses=linear_solver_statuses(2),
            convergence_statuses=ConvergenceStatusCollection(),
            divergence_statuses=ConvergenceStatusCollection(),
        ),
    )


def time_step_failure() -> TimeStepperStatusFailure:
    """Create a failed time-step status for statistics tests."""
    return TimeStepperStatusFailure(
        nonlinear_solver_status=NewtonSolverFailed(
            linear_solver_statuses=linear_solver_statuses(2),
            convergence_statuses=ConvergenceStatusCollection(),
            divergence_statuses=ConvergenceStatusCollection(),
        ),
        reason="Nonlinear solver failed.",
    )


# def time_step_status_in_progress() -> TimeStepperStatusContinueIterating:
#     """Create an in-progress time-step status for statistics tests."""
#     return TimeStepperStatusContinueIterating(
#         attempt=0,
#         nonlinear_solver_status=NewtonSolverFailed(
#             linear_solver_statuses=linear_solver_statuses(2),
#             convergence_statuses=ConvergenceStatusCollection(),
#             divergence_statuses=ConvergenceStatusCollection(),
#         ),
#     )


def default_newton_solver(nonlinear_increment_history: Optional[np.ndarray] = None):
    """Create a Newton solver with a mock linear solver.

    Parameters:
        nonlinear_increment_history: Sequence of increments returned by successive
            calls to the mock linear solver. Tests use this to control the nonlinear
            increment norm on each Newton iteration and thereby trigger convergence
            or divergence criteria. If omitted, the mock solver receives an empty
            history.

    """
    if nonlinear_increment_history is None:
        nonlinear_increment_history = np.ndarray(shape=())
    return pp.solvers.NewtonSolver(
        params={
            "nl_convergence_criteria": {
                "inc_abs": pp.solvers.IncrementBasedAbsoluteCriterion(
                    tol=1.0, metric=pp.EuclideanMetric()
                ),
                "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                    tol=1.0, metric=pp.EuclideanMetric()
                ),
            },
            "nl_divergence_criteria": {
                "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=3),
                "inc_inf": pp.solvers.IncrementBasedAbsoluteDivergenceCriterion(
                    tol=10.0, metric=pp.EuclideanMetric()
                ),
                "res_inf": pp.solvers.ResidualBasedAbsoluteDivergenceCriterion(
                    tol=10.0, metric=pp.EuclideanMetric()
                ),
                "inc_nan": pp.solvers.IncrementBasedNanCriterion(),
                "res_nan": pp.solvers.ResidualBasedNanCriterion(),
            },
        },
        linear_solver=MockLinearSolver(nonlinear_increment_history),
    )


class MockEquationSystem:
    residual: np.ndarray
    """Will be set from the outside in tests."""

    equation_indexer = pp.ad.EquationIndexer(
        {pp.ad.EquationOnDomain("y", domain=pp.CartGrid(nx=1)): np.array([0])}
    )
    """Mock equation indexer"""
    variable_indexer = pp.ad.VariableIndexer(
        {
            pp.ad.Variable(
                "x",
                ndof={"cells": 1},
                domain=pp.CartGrid(nx=1),
            ): np.array([0]),
        }
    )
    """Mock variable indexer"""

    def get_variable_values(self, **wkwargs):
        return np.array([1.0])

    def assemble(self, evaluate_jacobian: bool = True, **kwargs):
        if not evaluate_jacobian:
            return self.residual
        return pp.solvers.LinearSystem(
            matrix=csr_matrix(np.array([[1.0]])),
            rhs=np.array([1e-11]),
            equation_indexer=pp.ad.EquationIndexer(indices={}),
            variable_indexer=pp.ad.VariableIndexer(indices={}),
        )


class MockMdg:
    def subdomains(self):
        return []


class MockModel:
    """Mock model for testing the Newton solver.

    Only features:
    - nonlinear_solver_statistics (incl. advance_iteration method, and I/O)
    - equation_system (only interface to assembling residual)
    - return value of the nonlinear increment on solve_linear_system

    """

    def __init__(
        self,
        residual_history: Optional[np.ndarray] = None,
        path: Optional[Path] = None,
        is_nonlinear: bool = True,
    ):
        self.nonlinear_solver_statistics = (
            solver_statistics.SolverStatisticsFactory.create_statistics_type(
                nonlinear=is_nonlinear, time_dependent=False
            )(path=path)
        )
        self.equation_system = MockEquationSystem()
        self.mdg = MockMdg()
        if residual_history is None:
            residual_history = np.ndarray(shape=())
        self.residual_history = residual_history
        self._is_nonlinear = is_nonlinear

    def before_nonlinear_loop(self):
        self.nonlinear_solver_statistics.increase_index()

    def before_nonlinear_iteration(self):
        self.equation_system.residual = np.array(self.residual_history[0])
        self.residual_history = self.residual_history[1:]

    def after_nonlinear_iteration(
        self,
        nonlinear_increment: np.ndarray,
        updated_variables: Optional[list[pp.ad.Variable]] = None,
    ):
        pass

    def after_nonlinear_convergence(self):
        self.nonlinear_solver_statistics.save()

    def after_nonlinear_failure(self):
        self.nonlinear_solver_statistics.save()

    def _is_time_dependent(self):
        return False

    def _is_nonlinear_problem(self):
        return self._is_nonlinear


class MockLinearSolver(pp.solvers.LinearSolverBase):
    """A mockup class for a linear solver. Each call to solve_linear_system returns next
    value of the `nonlinear_increment_history` array.

    """

    def __init__(self, nonlinear_increment_history: np.ndarray):
        self.nonlinear_increment_history = nonlinear_increment_history
        self.iteration_counter = -1
        """Counts the number of times solve_linear_system was called."""

    def solve_linear_system(
        self, linear_system: pp.solvers.LinearSystem
    ) -> tuple[np.ndarray, pp.solvers.LinearSolverStatus]:
        self.iteration_counter += 1
        increment = np.array(self.nonlinear_increment_history[self.iteration_counter])
        return increment, pp.solvers.LinearSolverStatusSuccess(solve_time=0)


class TimeDependentMockModel(MockModel):
    """Use nested lists for convergence history and adapted statistics."""

    def __init__(
        self,
        residual_history=None,
        path=None,
    ):
        super().__init__(residual_history=residual_history, path=path)
        self.nonlinear_solver_statistics = pp.NonlinearSolverAndTimeStatistics(
            path=path
        )
        self.time_manager = pp.TimeManager(
            schedule=[0.0, 1.0], dt_init=0.5, constant_dt=True
        )

    def before_nonlinear_loop(self):
        super().before_nonlinear_loop()
        self.residuals = self.residual_history[0]
        self.residual_history = self.residual_history[1:]

    def before_nonlinear_iteration(self):
        self.equation_system.residual = np.array(self.residuals[0])
        self.residuals = self.residuals[1:]

    def _is_time_dependent(self):
        return True

    def _is_nonlinear_problem(self):
        return True


# ! ---- Unit tests ---- ! #


def test_init_criteria():
    """Test that custom convergence and divergence criteria are set correctly."""
    custom_conv_criteria = {
        "residual_based": pp.solvers.ResidualBasedAbsoluteCriterion(
            tol=1e-6, metric=pp.EuclideanMetric()
        ),
    }
    custom_div_criteria = {
        "inc_nan": pp.solvers.IncrementBasedNanCriterion(),
        "res_nan": pp.solvers.ResidualBasedNanCriterion(),
    }
    solver = pp.solvers.NewtonSolver(
        params={
            "nl_convergence_criteria": custom_conv_criteria,
            "nl_divergence_criteria": custom_div_criteria,
        }
    )
    assert solver.convergence_criteria == custom_conv_criteria
    assert solver.divergence_criteria == custom_div_criteria


@pytest.mark.parametrize(
    ("status_type", "expected"),
    [
        (NewtonSolverConverged, "successful"),
        (NewtonSolverFailed, "failed"),
    ],
)
def test_nonlinear_solver_status_serialization(status_type, expected):
    status = status_type(
        linear_solver_statuses=linear_solver_statuses(2),
        convergence_statuses=ConvergenceStatusCollection(),
        divergence_statuses=ConvergenceStatusCollection(),
    )
    assert status.serialize() == expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (time_step_status_in_progress(), "in_progress"),
        (time_step_success(), "successful"),
        (time_step_failure(), "failed"),
    ],
)
def test_time_stepper_status_serialization(status, expected):
    assert status.serialize() == expected


def test_init_criteria_valid_max_iterations():
    """Test that max_iterations attribute is correctly fetched."""
    solver = pp.solvers.NewtonSolver()
    assert solver.max_iterations == 10  # From default params.
    assert default_newton_solver().max_iterations == 3  # From criteria.


@pytest.mark.parametrize(
    "key, value",
    [
        ("nl_convergence_inc_atol", 5.0),
        ("nl_convergence_res_atol", 5.0),
        ("nl_convergence_inc_rtol", 5.0),
        ("nl_convergence_res_rtol", 5.0),
    ],
)
def test_init_convergence_criteria_sanity_check(key, value):
    """Test sanity check in convergence criteria."""
    with pytest.raises(AssertionError) as e:
        pp.solvers.NewtonSolver(
            params={
                key: value,
                "nl_convergence_criteria": {
                    "inc_abs": pp.solvers.IncrementBasedAbsoluteCriterion(
                        tol=1e-1, metric=pp.EuclideanMetric()
                    ),
                },
            }
        )
        assert (
            """If 'nl_convergence_criteria' is provided, """
            """do not provide individual convergence tolerances.""" in str(e.value)
        )


@pytest.mark.parametrize(
    "key, value",
    [
        ("nl_max_iterations", 5),
        ("nl_divergence_inc_atol", 5.0),
        ("nl_divergence_res_atol", 5.0),
    ],
)
def test_init_divergence_criteria_sanity_check(key, value):
    """Test sanity check in divergence criteria."""
    with pytest.raises(AssertionError) as e:
        pp.solvers.NewtonSolver(
            params={
                key: value,
                "nl_divergence_criteria": {
                    "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=2)
                },
            }
        )
        assert (
            """If 'nl_divergence_criteria' is provided, do not provide """
            """individual divergence tolerances.""" in str(e.value)
        )


def test_increase_iteration_index():
    """Unit test for the advance_iteration method of the Newton solver."""
    # Init solver.
    solver = default_newton_solver()

    # Advance iteration count.
    assert solver.iteration_index == 0
    solver.increase_iteration_index()
    assert solver.iteration_index == 1
    solver.increase_iteration_index()
    assert solver.iteration_index == 2


def test_solve_convergence():
    """Test that the solver returns SUCCESSFUL on convergence."""
    # Init model with convergence after two iterations.
    model = MockModel(residual_history=[1.0, 0.5])
    solver = default_newton_solver(nonlinear_increment_history=[2.0, 0.5])

    # Call solve.
    solver_status = solver.solve(model)

    # Check simulation status.
    assert solver_status.is_converged()


def test_solve_convergence_statistics():
    """Test that the solver statistics are updated correctly on convergence to check
    correct behavior after convergence.

    """
    if Path("solver_statistics.json").exists():
        Path("solver_statistics.json").unlink()
    # Init model with convergence after two iterations.
    model = MockModel(
        residual_history=[1.0, 0.5],
        path=Path("solver_statistics.json"),
    )
    solver = default_newton_solver(nonlinear_increment_history=[2.0, 0.5])

    # Call solve.
    _ = solver.solve(model)

    # Summarize status and save statistics.
    # TODO: Revisit during restructuring of loops.
    simulation_status = time_step_success()
    model.nonlinear_solver_statistics.log_simulation_status(simulation_status)
    model.nonlinear_solver_statistics.save()

    # Check solver statistics.

    with open("solver_statistics.json", "r") as f:
        data = json.load(f)

    assert (
        DeepDiff(
            data,
            {
                "global": {
                    "num_cells": {},
                    "num_domains": {},
                    "simulation_status_history": ["successful"],
                    "final_simulation_status": "successful",
                    "num_entries": 1,
                    "num_iterations_history": [2],
                    "total_num_iterations": 2,
                    "total_num_waisted_iterations": 0,
                    "final_convergence_status": {
                        "inc_abs": "converged",
                        "res_abs": "converged",
                        "max_iter": "converged",
                        "inc_inf": "converged",
                        "res_inf": "converged",
                        "inc_nan": "converged",
                        "res_nan": "converged",
                    },
                },
                "0": {
                    "num_iterations": 2,
                    "simulation_status": "successful",
                    "solver_status": "successful",
                    "convergence_status": {
                        "inc_abs": ["continue_iterating", "converged"],
                        "res_abs": ["continue_iterating", "converged"],
                        "max_iter": ["converged", "converged"],
                        "inc_inf": ["converged", "converged"],
                        "res_inf": ["converged", "converged"],
                        "inc_nan": ["converged", "converged"],
                        "res_nan": ["converged", "converged"],
                    },
                    "convergence_info": {"inc_abs": [2.0, 0.5], "res_abs": [1.0, 0.5]},
                },
            },
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
            ignore_type_in_groups=[(dict, ConvergenceInfoHistory)],
        )
        == {}
    )

    # Clean up.
    Path("solver_statistics.json").unlink()


def test_solve_convergence_time_dependent():
    """Test that the solver returns SUCCESSFUL for converged time-dependent model."""
    # Minimal setup.
    model = TimeDependentMockModel(residual_history=[[1.0, 0.5], [1.0, 1.0, 0.5]])
    solver = default_newton_solver(
        nonlinear_increment_history=[2.0, 0.5, 2.0, 1.0, 0.5]
    )

    # First time step - advance time to log the time step.
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    solver_status = solver.solve(model)

    # Check simulation status.
    assert solver_status.is_converged()

    # Second time step.
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    solver_status = solver.solve(model)

    # Check simulation status.
    assert solver_status.is_converged()


def test_solve_failure():
    """Test that the solver returns FAILED on divergence."""
    # Minimal setup for failure after two iterations.
    model = MockModel(residual_history=[1.0, np.nan])
    solver = default_newton_solver(nonlinear_increment_history=[2.0, 100.0])
    solver_status = solver.solve(model)

    # Check simulation status.
    assert solver_status.is_failed()


def test_solve_failure_statistics():
    """Test that the solver statistics are updated correctly on convergence to check
    correct behavior after failure.

    """
    # Minimal setup for failure after two iterations.
    model = MockModel(
        residual_history=[1.0, np.nan],
        path=Path("solver_statistics.json"),
    )
    solver = default_newton_solver(nonlinear_increment_history=[2.0, 100.0])
    solver_status = solver.solve(model)

    # Check simulation status.
    assert solver_status.is_failed()

    model.nonlinear_solver_statistics.log_simulation_status(time_step_failure())
    model.nonlinear_solver_statistics.save()

    # Check solver statistics.
    with open("solver_statistics.json", "r") as f:
        data = json.load(f)

    assert (
        DeepDiff(
            data,
            {
                "global": {
                    "num_cells": {},
                    "num_domains": {},
                    "simulation_status_history": ["failed"],
                    "final_simulation_status": "failed",
                    "num_entries": 1,
                    "num_iterations_history": [2],
                    "total_num_iterations": 2,
                    "total_num_waisted_iterations": 2,
                    "final_convergence_status": {
                        "inc_abs": "continue_iterating",
                        "res_abs": "continue_iterating",
                        "max_iter": "converged",
                        "inc_inf": "failed",
                        "res_inf": "failed",
                        "inc_nan": "converged",
                        "res_nan": "failed",
                    },
                },
                "0": {
                    "num_iterations": 2,
                    "simulation_status": "failed",
                    "solver_status": "failed",
                    "convergence_status": {
                        "inc_abs": ["continue_iterating", "continue_iterating"],
                        "res_abs": ["continue_iterating", "continue_iterating"],
                        "max_iter": ["converged", "converged"],
                        "inc_inf": ["converged", "failed"],
                        "res_inf": ["converged", "failed"],
                        "inc_nan": ["converged", "converged"],
                        "res_nan": ["converged", "failed"],
                    },
                    "convergence_info": {
                        "inc_abs": [2.0, 100.0],
                        "res_abs": [1.0, np.nan],
                    },
                },
            },
            ignore_numeric_type_changes=True,  # for nan
        )
        == {}
    )

    # Clean up.
    Path("solver_statistics.json").unlink()


def test_solve_failure_time_dependent():
    """Test that the solver returns FAILED on divergence for a time-dependent model,"""
    # Minimal setup for failure for first of three iterations - last two identical.
    model = TimeDependentMockModel(
        residual_history=[[1.0, np.nan], [1.0, 1.0, 0.5], [1.0, 1.0, 0.5]],
    )
    solver = default_newton_solver(
        nonlinear_increment_history=[2.0, 100.0, 2.0, 1.0, 0.5, 2.0, 1.0, 0.5]
    )

    # First time step - advance time to log the time step.
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    solver_status = solver.solve(model)

    # Check simulation status.
    assert not model.time_manager.final_time_reached()
    assert solver_status.is_failed()

    # Retry time step, so do not increase time.
    solver_status = solver.solve(model)

    # Check simulation status.
    assert not model.time_manager.final_time_reached()
    assert solver_status.is_converged()

    # First time step - advance time to log the time step.
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    solver_status = solver.solve(model)

    # Check simulation status.
    assert model.time_manager.final_time_reached()
    assert solver_status.is_converged()


def test_before_nonlinear_loop():
    """Unit test for the before_nonlinear_loop method of the Newton solver.

    Mainly check correct management of indices.

    """
    # Init model and solver.
    model = MockModel()
    solver = default_newton_solver()

    # Mock a situation in the midst of a simulation (after some time step).
    solver.iteration_index = 10
    model.nonlinear_solver_statistics.index = 5

    # Call before_nonlinear_loop.
    solver.before_nonlinear_loop(model)

    # Ensure resetting of iteration index and increase of statistics index.
    assert solver.iteration_index == 0
    assert model.nonlinear_solver_statistics.index == 6


@pytest.mark.parametrize(
    "inc_history, res_history, is_converged, is_failed",
    [
        ([2.0, 2.0], [1.0, 1.0], False, False),  # no convergence after 2 iterations
        ([2.0, 0.5], [1.0, 0.5], True, False),  # convergence in 2 iterations
        ([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], False, True),  # divergence due to max iter.
        ([2.0, 2.0, 0.5], [1.0, 1.0, 0.5], True, True),  # convergence and divergence
        ([2.0, 11.0], [1.0, 1.0], False, True),  # divergence due to increment
        ([2.0, 2.0], [1.0, 11.0], False, True),  # divergence due to residual
        ([2.0, np.nan], [1.0, 1.0], False, True),  # divergence due to increment nan
        ([2.0, 2.0], [1.0, np.nan], False, True),  # divergence due to residual nan
    ],
)
def test_nonlinear_loop(
    inc_history,
    res_history,
    is_converged,
    is_failed,
):
    """Test that the Newton loop exits correctly."""
    model = MockModel(residual_history=res_history)
    solver = default_newton_solver(nonlinear_increment_history=inc_history)

    # Identify number of iterations from history.
    num_iter = len(inc_history)

    # Prepare for Newton loop.
    solver.before_nonlinear_loop(model)

    # Perform Newton loop.
    try:
        convergence_status, divergence_status, linear_solver_statuses = (
            solver.nonlinear_loop(model)
        )

        # Check that the returned statuses match expected values
        if is_converged:
            assert convergence_status.is_converged()
        else:
            assert convergence_status.is_iterating()
        if is_failed:
            assert divergence_status.is_failed()
        else:
            assert divergence_status.is_converged()

        # Check that the number of iterations and recorded linear solves is as expected.
        assert solver.iteration_index == num_iter
        assert len(linear_solver_statuses) == num_iter

    except Exception as e:
        # Newton loop only stops on convergence or divergence.
        # Need to handle the non-convergence and non-divergence case.
        assert not (is_converged or is_failed), f"Unexpected exception: {e}"


@pytest.mark.parametrize(
    "convergence_status, divergence_status, expected_solver_status",
    [
        (
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONVERGED,
            NonlinearSolverStatusConverged,
        ),
        (
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONTINUE_ITERATING,
            NonlinearSolverStatusConverged,
        ),
        (
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.FAILED,
            NonlinearSolverStatusConverged,  # Convergence trumps divergence
        ),
        (
            ConvergenceStatus.CONTINUE_ITERATING,
            ConvergenceStatus.FAILED,
            NonlinearSolverStatusFailed,
        ),
    ],
)
def test_summarize_solver_status(
    convergence_status,
    divergence_status,
    expected_solver_status: type[NonlinearSolverStatus],
):
    """Unit test for the summarize_solver_status method of the Newton solver."""
    # Init model and solver.
    model = MockModel()
    solver = default_newton_solver()

    # Minimal mimicking of loop.
    solver_status = _summarize_solver_status(
        ConvergenceStatusCollection({"convergence": convergence_status}),
        ConvergenceStatusCollection({"divergence": divergence_status}),
        linear_solver_statuses=linear_solver_statuses(2),
    )

    # Check that the returned simulation status matches expected value.
    assert isinstance(solver_status, expected_solver_status)


def test_before_nonlinear_iteration():
    """Unit test for the before_nonlinear_iteration method of the Newton solver."""
    # Init model and solver.
    model = MockModel(residual_history=[1.0])
    solver = default_newton_solver(nonlinear_increment_history=[2.0])

    # Check initial iteration index.
    assert solver.iteration_index == 0

    # Call before_nonlinear_iteration.
    solver.before_nonlinear_iteration(model)

    # Check that the iteration index has been increased.
    assert solver.iteration_index == 1


@pytest.mark.parametrize(
    "inc, res, iteration_index, is_converged, is_failed",
    [
        ([2.0, 1.0, 1, False, False]),  # Not converged nor diverged
        ([2.0, 1.0, 2, False, False]),  # Not converged nor diverged
        ([2.0, 1.0, 3, False, True]),  # Diverged due to max iterations
        ([0.5, 0.5, 1, True, False]),  # Convergence
        ([0.5, 0.5, 2, True, False]),  # Convergence
        ([0.5, 0.5, 3, True, True]),  # Convergence and divergence
        ([11.0, 0.5, 1, False, True]),  # Due to increment divergence
        ([0.5, 11.0, 1, False, True]),  # Due to residual divergence
        ([np.nan, 0.5, 1, False, True]),  # Due to increment nan
        ([0.5, np.nan, 1, False, True]),  # Due to residual nan
    ],
)
def test_after_nonlinear_iteration(
    inc,
    res,
    iteration_index,
    is_converged,
    is_failed,
):
    """Test the after_nonlinear_iteration method of the Newton solver."""
    # Init model and solver.
    model = MockModel()
    solver = default_newton_solver()

    # Mock the nonlinear increment and residual for the last iteration.
    model.nonlinear_increment = np.array([inc])
    model.equation_system.residual = np.array([res])

    # Mock the number of iterations.
    solver.iteration_index = iteration_index

    # Minimal setup needed of the model statistics.
    model.nonlinear_solver_statistics.num_iterations_history = [iteration_index]

    # Check convergence.
    convergence_status, divergence_status = solver.after_nonlinear_iteration(
        model, model.nonlinear_increment
    )

    # Check that the returned statuses match expected values
    if is_converged:
        assert convergence_status.is_converged()
    else:
        assert convergence_status.is_iterating()
    if is_failed:
        assert divergence_status.is_failed()
    else:
        assert divergence_status.is_converged()


@pytest.mark.parametrize(
    "inc, res, iteration_index, is_converged, is_failed",
    [
        ([2.0, 1.0, 1, False, False]),  # Not converged nor diverged
        ([2.0, 1.0, 2, False, False]),  # Not converged nor diverged
        ([2.0, 1.0, 3, False, True]),  # Diverged due to max iterations
        ([0.5, 0.5, 1, True, False]),  # Convergence
        ([0.5, 0.5, 2, True, False]),  # Convergence
        ([0.5, 0.5, 3, True, True]),  # Convergence and divergence
        ([11.0, 0.5, 1, False, True]),  # Due to increment divergence
        ([0.5, 11.0, 1, False, True]),  # Due to residual divergence
        ([np.nan, 0.5, 1, False, True]),  # Due to increment nan
        ([0.5, np.nan, 1, False, True]),  # Due to residual nan
    ],
)
def test_check_convergence(
    inc,
    res,
    iteration_index,
    is_converged,
    is_failed,
):
    """Test the check_convergence function in relation to the Newton solver."""
    # Init model and solver.
    model = MockModel()
    solver = default_newton_solver()

    # Check convergence.
    convergence_status, divergence_status, convergence_info = check_convergence(
        convergence_criteria=solver.convergence_criteria,
        divergence_criteria=solver.divergence_criteria,
        nonlinear_increment=np.array([inc]),
        solution=model.equation_system.get_variable_values(),
        residual=np.array([res]),
        iteration_index=iteration_index,
    )

    # Check that the returned statuses match expected values
    if is_converged:
        assert convergence_status.is_converged()
    else:
        assert convergence_status.is_iterating()
    if is_failed:
        assert divergence_status.is_failed()
    else:
        assert divergence_status.is_converged()
    assert (
        DeepDiff(
            convergence_info,
            {"inc_abs": inc, "res_abs": res},
            ignore_numeric_type_changes=True,
        )
        == {}
    )


def test_update_solver_statistics():
    """Unit test for the update_solver_statistics method of the Newton solver."""
    model = MockModel()
    solver = default_newton_solver()

    # Prepare model for updating solver statistics.
    model.before_nonlinear_loop()

    # Set some inputs for the update_solver_statistics method.
    # Here, we simulate one iteration with not converged status.
    convergence_status = ConvergenceStatusCollection(
        {
            "inc_abs": ConvergenceStatus.CONTINUE_ITERATING,
            "res_abs": ConvergenceStatus.CONTINUE_ITERATING,
            "max_iter": ConvergenceStatus.CONVERGED,
        }
    )
    convergence_info = {"inc_abs": 2.0, "res_abs": 1.0}

    # Call the update_solver_statistics method
    solver.update_solver_statistics(model, convergence_status, convergence_info)

    # Check that the solver statistics have been updated correctly
    assert model.nonlinear_solver_statistics.num_iterations == 1
    assert model.nonlinear_solver_statistics.convergence_status == {
        "inc_abs": ["continue_iterating"],
        "res_abs": ["continue_iterating"],
        "max_iter": ["converged"],
    }
    assert model.nonlinear_solver_statistics.convergence_info == {
        "inc_abs": [2.0],
        "res_abs": [1.0],
    }
    # This field is updated by the linear solver.
    assert model.nonlinear_solver_statistics.simulation_status_history == []


# ! ---- Test integration ---- ! #


@pytest.mark.parametrize("num_iterations", [1, 3])
def test_integration_nonlinear_iteration_count(num_iterations):
    """Test for checking if the nonlinear iterations are counted as expected.

    A pre set value of expected iterations is set, and the test checks that the
    iteration count matches the pre set value after convergence is obtained.

    """
    model = SinglePhaseFlow({"times_to_export": []})

    # Model will not converge within the prescribed number of iterations, we just track
    # the iteration count.
    with pytest.raises(RuntimeError):
        pp.ModelRunner(
            model,
            {
                "nl_convergence_inc_atol": 0,
                "nl_convergence_res_atol": 0,
                "nl_max_iterations": num_iterations,
            },
        ).run()

    assert model.nonlinear_solver_statistics.num_iterations == num_iterations
    for key in model.nonlinear_solver_statistics.convergence_status:
        assert (
            len(model.nonlinear_solver_statistics.convergence_status[key])
            == num_iterations
        )
    for key in model.nonlinear_solver_statistics.convergence_info:
        assert (
            len(model.nonlinear_solver_statistics.convergence_info[key])
            == num_iterations
        )


@pytest.mark.parametrize("is_nonlinear", [False, True])
def test_linear_nonlinear_model(is_nonlinear: bool):
    """Tests that the nonlinear solver performs a single iteration if the problem is
    linear, which is equivalent to a single linear solve. It then returns success
    despite the nonlinear residual is still large (we don't want it to check it for a
    linear problem).

    """
    model = MockModel(
        # Residual is not small after the first iteration!
        residual_history=[5, 0.02, 1e-23],
        is_nonlinear=is_nonlinear,
    )

    # Creating a nonlinear solver with default convergence criteria. They must be
    # different for nonlinear and linear cases.
    solver = pp.solvers.NewtonSolver(
        is_nonlinear_problem=is_nonlinear,
        params=None,
        linear_solver=MockLinearSolver([10, 0.05, 1e-22]),
    )
    status = solver.solve(model)
    assert isinstance(status, NonlinearSolverStatusConverged)

    # The solver will do 1 iteration for a linear problem and 3 for a nonlinear one.
    expected_num_iterations = 3 if is_nonlinear else 1
    assert len(status.linear_solver_statuses) == expected_num_iterations


@pytest.mark.xfail(
    reason="This reproduces a bug https://github.com/pmgbergen/porepy/issues/1713.",
    strict=True,
)
def test_linear_solver_fails():
    """Creates a linear problem and a solver for it. The solver returns nans after the
    first iteration. The solver must return failure status.

    This test is marked as failing. If you are working on this issue, you should make
    this test passing and remove the "xfail" decorator.

    """
    model = MockModel(
        residual_history=[np.nan],
        is_nonlinear=False,
    )
    solver = pp.solvers.NewtonSolver(
        is_nonlinear_problem=False,
        params=None,
        linear_solver=MockLinearSolver([np.nan]),
    )
    status = solver.solve(model)
    assert status.is_failed(), "Must return failure, but returns success instead."
