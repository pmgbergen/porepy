"""Unit tests for the Newton solver."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from deepdiff import DeepDiff

import porepy as pp
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceInfoHistory,
    ConvergenceStatus,
    ConvergenceStatusCollection,
    SimulationStatus,
)

# ! ---- Auxiliary fixtures and classes ---- ! #


@pytest.fixture
def default_newton_solver():
    return pp.NewtonSolver(
        params={
            "nl_convergence_criteria": {
                "inc_abs": pp.IncrementBasedAbsoluteCriterion(
                    tol=1.0, metric=pp.EuclideanMetric()
                ),
                "res_abs": pp.ResidualBasedAbsoluteCriterion(
                    tol=1.0, metric=pp.EuclideanMetric()
                ),
            },
            "nl_divergence_criteria": {
                "max_iter": pp.MaxIterationsCriterion(max_iterations=3),
                "inc_inf": pp.IncrementBasedAbsoluteDivergenceCriterion(
                    tol=10.0, metric=pp.EuclideanMetric()
                ),
                "res_inf": pp.ResidualBasedAbsoluteDivergenceCriterion(
                    tol=10.0, metric=pp.EuclideanMetric()
                ),
                "inc_nan": pp.IncrementBasedNanCriterion(),
                "res_nan": pp.ResidualBasedNanCriterion(),
            },
        }
    )


class MockEquationSystem:
    def get_variable_values(self, **wkwargs):
        return np.array([1.0])

    def assemble(self, evaluate_jacobian=False):
        return self.residual


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
        nonlinear_increment_history=None,
        residual_history=None,
        path=None,
    ):
        self.nonlinear_solver_statistics = pp.NonlinearSolverStatistics(path=path)
        self.equation_system = MockEquationSystem()
        self.mdg = MockMdg()
        self.nonlinear_increment_history = nonlinear_increment_history
        self.residual_history = residual_history

    def before_nonlinear_loop(self):
        self.nonlinear_solver_statistics.increase_index()

    def before_nonlinear_iteration(self):
        self.nonlinear_increment = np.array(self.nonlinear_increment_history[0])
        self.nonlinear_increment_history = self.nonlinear_increment_history[1:]
        self.equation_system.residual = np.array(self.residual_history[0])
        self.residual_history = self.residual_history[1:]

    def after_nonlinear_iteration(self, inc):
        pass

    def after_nonlinear_convergence(self):
        self.nonlinear_solver_statistics.save()

    def after_nonlinear_failure(self):
        self.nonlinear_solver_statistics.save()
        return SimulationStatus.FAILED

    def assemble_linear_system(self):
        pass

    def solve_linear_system(self):
        return self.nonlinear_increment

    def _is_time_dependent(self):
        return False


class TimeDependentMockModel(MockModel):
    """Use nested lists for convergence history and adapted statistics."""

    def __init__(
        self,
        nonlinear_increment_history=None,
        residual_history=None,
        path=None,
    ):
        super().__init__(
            nonlinear_increment_history=nonlinear_increment_history,
            residual_history=residual_history,
            path=path,
        )
        self.nonlinear_solver_statistics = pp.NonlinearSolverAndTimeStatistics(
            path=path
        )
        self.time_manager = pp.TimeManager(
            schedule=[0.0, 1.0], dt_init=0.5, constant_dt=True
        )

    def before_nonlinear_loop(self):
        super().before_nonlinear_loop()
        self.nonlinear_increments = self.nonlinear_increment_history[0]
        self.nonlinear_increment_history = self.nonlinear_increment_history[1:]
        self.residuals = self.residual_history[0]
        self.residual_history = self.residual_history[1:]

    def before_nonlinear_iteration(self):
        self.nonlinear_increment = np.array(self.nonlinear_increments[0])
        self.nonlinear_increments = self.nonlinear_increments[1:]
        self.equation_system.residual = np.array(self.residuals[0])
        self.residuals = self.residuals[1:]

    def _is_time_dependent(self):
        return True


# ! ---- Unit tests ---- ! #


def test_init_criteria():
    """Test that custom convergence and divergence criteria are set correctly."""
    custom_conv_criteria = {
        "residual_based": pp.ResidualBasedAbsoluteCriterion(
            tol=1e-6, metric=pp.EuclideanMetric()
        ),
    }
    custom_div_criteria = {
        "inc_nan": pp.IncrementBasedNanCriterion(),
        "res_nan": pp.ResidualBasedNanCriterion(),
    }
    solver = pp.NewtonSolver(
        params={
            "nl_convergence_criteria": custom_conv_criteria,
            "nl_divergence_criteria": custom_div_criteria,
        }
    )
    assert solver.convergence_criteria == custom_conv_criteria
    assert solver.divergence_criteria == custom_div_criteria


def test_init_criteria_valid_max_iterations(default_newton_solver):
    """Test that max_iterations attribute is correctly fetched."""
    solver = pp.NewtonSolver()
    assert solver.max_iterations == 10  # From default params.
    assert default_newton_solver.max_iterations == 3  # From criteria.


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
        pp.NewtonSolver(
            params={
                key: value,
                "nl_convergence_criteria": {
                    "inc_abs": pp.IncrementBasedAbsoluteCriterion(
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
        pp.NewtonSolver(
            params={
                key: value,
                "nl_divergence_criteria": {
                    "max_iter": pp.MaxIterationsCriterion(max_iterations=2)
                },
            }
        )
        assert (
            """If 'nl_divergence_criteria' is provided, do not provide """
            """individual divergence tolerances.""" in str(e.value)
        )


def test_increase_iteration_index(default_newton_solver):
    """Unit test for the advance_iteration method of the Newton solver."""
    # Init solver.
    solver = default_newton_solver

    # Advance iteration count.
    assert solver.iteration_index == 0
    solver.increase_iteration_index()
    assert solver.iteration_index == 1
    solver.increase_iteration_index()
    assert solver.iteration_index == 2


def test_solve_convergence(default_newton_solver):
    """Test that the solver returns SUCCESSFUL on convergence."""
    # Init model with convergence after two iterations.
    model = MockModel(
        nonlinear_increment_history=[2.0, 0.5],
        residual_history=[1.0, 0.5],
    )
    solver = default_newton_solver

    # Call solve.
    simulation_status = solver.solve(model)

    # Check simulation status.
    assert simulation_status == SimulationStatus.SUCCESSFUL


def test_solve_convergence_statistics(default_newton_solver):
    """Test that the solver statistics are updated correctly on convergence to check
    correct behavior after convergence.

    """
    if Path("solver_statistics.json").exists():
        Path("solver_statistics.json").unlink()
    # Init model with convergence after two iterations.
    model = MockModel(
        nonlinear_increment_history=[2.0, 0.5],
        residual_history=[1.0, 0.5],
        path=Path("solver_statistics.json"),
    )
    solver = default_newton_solver

    # Call solve.
    _ = solver.solve(model)

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
                    "convergence_status": {
                        "inc_abs": ["not_converged", "converged"],
                        "res_abs": ["not_converged", "converged"],
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


def test_solve_convergence_time_dependent(default_newton_solver):
    """Test that the solver returns SUCCESSFUL for converged time-dependent model."""
    # Minimal setup.
    model = TimeDependentMockModel(
        nonlinear_increment_history=[[2.0, 0.5], [2.0, 1.0, 0.5]],
        residual_history=[[1.0, 0.5], [1.0, 1.0, 0.5]],
    )
    solver = default_newton_solver

    # First time step - advance time to log the time step.
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    simulation_status = solver.solve(model)

    # Check simulation status.
    assert simulation_status == SimulationStatus.SUCCESSFUL

    # Second time step.
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    simulation_status = solver.solve(model)

    # Check simulation status.
    assert simulation_status == SimulationStatus.SUCCESSFUL


def test_solve_convergence_time_dependent_statistics(default_newton_solver):
    """Test that the solver statistics are updated correctly on convergence to check
    correct behavior after convergence, for a time-dependent model.

    """
    # Clean up.
    if Path("solver_and_time_statistics.json").exists():
        Path("solver_and_time_statistics.json").unlink()

    # Minimal setup.
    model = TimeDependentMockModel(
        nonlinear_increment_history=[[2.0, 0.5], [2.0, 1.0, 0.5]],
        residual_history=[[1.0, 0.5], [1.0, 1.0, 0.5]],
        path=Path("solver_and_time_statistics.json"),
    )
    solver = default_newton_solver

    # Define the reference solver statistics, for two time steps.
    reference_data_after_1 = {
        "global": {
            "num_cells": {},
            "num_domains": {},
            "simulation_status_history": ["successful"],
            "final_simulation_status": "successful",
            "num_entries": 1,
            "final_time_reached": 0,
            "total_num_time_steps": 1,
            "total_num_failed_time_steps": 0,
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
            "final_time_reached": 0,
            "time_index": 1,
            "time": 0.5,
            "dt": 0.5,
            "num_iterations": 2,
            "simulation_status": "successful",
            "convergence_status": {
                "inc_abs": ["not_converged", "converged"],
                "res_abs": ["not_converged", "converged"],
                "max_iter": ["converged", "converged"],
                "inc_inf": ["converged", "converged"],
                "res_inf": ["converged", "converged"],
                "inc_nan": ["converged", "converged"],
                "res_nan": ["converged", "converged"],
            },
            "convergence_info": {"inc_abs": [2.0, 0.5], "res_abs": [1.0, 0.5]},
        },
    }

    # Add data for the second time step, and update global data.
    reference_data_after_2 = copy.deepcopy(reference_data_after_1)
    reference_data_after_2["global"].update(
        {
            "simulation_status_history": ["successful", "successful"],
            "final_simulation_status": "successful",
            "num_entries": 2,
            "final_time_reached": 1,
            "total_num_time_steps": 2,
            "total_num_failed_time_steps": 0,
            "num_iterations_history": [2, 3],
            "total_num_iterations": 5,
            "total_num_waisted_iterations": 0,
            "final_convergence_status": {
                "inc_abs": "converged",
                "res_abs": "converged",
                "max_iter": "diverged",
                "inc_inf": "converged",
                "res_inf": "converged",
                "inc_nan": "converged",
                "res_nan": "converged",
            },
        }
    )
    reference_data_after_2["1"] = {
        "final_time_reached": 1,
        "time_index": 2,
        "time": 1.0,
        "dt": 0.5,
        "num_iterations": 3,
        "simulation_status": "successful",
        "convergence_status": {
            "inc_abs": ["not_converged", "not_converged", "converged"],
            "res_abs": ["not_converged", "not_converged", "converged"],
            "max_iter": ["converged", "converged", "diverged"],
            "inc_inf": ["converged", "converged", "converged"],
            "res_inf": ["converged", "converged", "converged"],
            "inc_nan": ["converged", "converged", "converged"],
            "res_nan": ["converged", "converged", "converged"],
        },
        "convergence_info": {
            "inc_abs": [2.0, 1.0, 0.5],
            "res_abs": [1.0, 1.0, 0.5],
        },
    }

    # First time step - advance time to log the time step.
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    _ = solver.solve(model)

    # Check solver statistics.
    with open("solver_and_time_statistics.json", "r") as f:
        data = json.load(f)

    # Check only the first time step. Need to adapt the global data accordingly.
    assert DeepDiff(data, reference_data_after_1) == {}

    # Second time step.
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    _ = solver.solve(model)

    # Check solver statistics.
    with open("solver_and_time_statistics.json", "r") as f:
        data = json.load(f)

    # Check all data.
    assert DeepDiff(data, reference_data_after_2) == {}

    # Clean up.
    Path("solver_and_time_statistics.json").unlink()


def test_solve_failure(default_newton_solver):
    """Test that the solver returns FAILED on divergence."""
    # Minimal setup for failure after two iterations.
    model = MockModel(
        nonlinear_increment_history=[2.0, 100.0],
        residual_history=[1.0, np.nan],
    )
    solver = default_newton_solver
    simulation_status = solver.solve(model)

    # Check simulation status.
    assert simulation_status == SimulationStatus.FAILED


def test_solve_failure_statistics(default_newton_solver):
    """Test that the solver statistics are updated correctly on convergence to check
    correct behavior after failure.

    """
    # Minimal setup for failure after two iterations.
    model = MockModel(
        nonlinear_increment_history=[2.0, 100.0],
        residual_history=[1.0, np.nan],
        path=Path("solver_statistics.json"),
    )
    solver = default_newton_solver
    simulation_status = solver.solve(model)

    # Check simulation status.
    assert simulation_status == SimulationStatus.FAILED

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
                        "inc_abs": "not_converged",
                        "res_abs": "not_converged",
                        "max_iter": "converged",
                        "inc_inf": "diverged",
                        "res_inf": "diverged",
                        "inc_nan": "converged",
                        "res_nan": "diverged",
                    },
                },
                "0": {
                    "num_iterations": 2,
                    "simulation_status": "failed",
                    "convergence_status": {
                        "inc_abs": ["not_converged", "not_converged"],
                        "res_abs": ["not_converged", "not_converged"],
                        "max_iter": ["converged", "converged"],
                        "inc_inf": ["converged", "diverged"],
                        "res_inf": ["converged", "diverged"],
                        "inc_nan": ["converged", "converged"],
                        "res_nan": ["converged", "diverged"],
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


def test_solve_failure_time_dependent(default_newton_solver):
    """Test that the solver returns FAILED on divergence for a time-dependent model,"""
    # Minimal setup for failure for first of three iterations - last two identical.
    model = TimeDependentMockModel(
        nonlinear_increment_history=[[2.0, 100.0], [2.0, 1.0, 0.5], [2.0, 1.0, 0.5]],
        residual_history=[[1.0, np.nan], [1.0, 1.0, 0.5], [1.0, 1.0, 0.5]],
    )
    solver = default_newton_solver

    # First time step - advance time to log the time step.
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    simulation_status = solver.solve(model)

    # Check simulation status.
    assert not model.time_manager.final_time_reached()
    assert simulation_status == SimulationStatus.FAILED

    # Retry time step, so do not increase time.
    simulation_status = solver.solve(model)

    # Check simulation status.
    assert not model.time_manager.final_time_reached()
    assert simulation_status == SimulationStatus.SUCCESSFUL

    # First time step - advance time to log the time step.
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    simulation_status = solver.solve(model)

    # Check simulation status.
    assert model.time_manager.final_time_reached()
    assert simulation_status == SimulationStatus.SUCCESSFUL


def test_solve_failure_time_dependent_statistics(default_newton_solver):
    """Test that the solver statistics are updated correctly on convergence to check
    correct behavior after failure, for a time-dependent model.

    """
    # Clean up.
    if Path("solver_and_time_statistics.json").exists():
        Path("solver_and_time_statistics.json").unlink()

    # Minimal setup for failure for first of three iterations - last two identical.
    model = TimeDependentMockModel(
        nonlinear_increment_history=[[2.0, 100.0], [2.0, 1.0, 0.5], [2.0, 1.0, 0.5]],
        residual_history=[[1.0, np.nan], [1.0, 1.0, 0.5], [1.0, 1.0, 0.5]],
        path=Path("solver_and_time_statistics.json"),
    )
    solver = default_newton_solver

    # Define the reference solver statistics, for two time steps (three iterations).
    reference_data_after_1 = {
        "global": {
            "num_cells": {},
            "num_domains": {},
            "simulation_status_history": ["failed"],
            "final_simulation_status": "failed",
            "num_entries": 1,
            "final_time_reached": 0,
            "total_num_time_steps": 1,
            "total_num_failed_time_steps": 1,
            "num_iterations_history": [2],
            "total_num_iterations": 2,
            "total_num_waisted_iterations": 2,
            "final_convergence_status": {
                "inc_abs": "not_converged",
                "res_abs": "not_converged",
                "max_iter": "converged",
                "inc_inf": "diverged",
                "res_inf": "diverged",
                "inc_nan": "converged",
                "res_nan": "diverged",
            },
        },
        "0": {
            "final_time_reached": 0,
            "time_index": 1,
            "time": 0.5,
            "dt": 0.5,
            "num_iterations": 2,
            "simulation_status": "failed",
            "convergence_status": {
                "inc_abs": ["not_converged", "not_converged"],
                "res_abs": ["not_converged", "not_converged"],
                "max_iter": ["converged", "converged"],
                "inc_inf": ["converged", "diverged"],
                "res_inf": ["converged", "diverged"],
                "inc_nan": ["converged", "converged"],
                "res_nan": ["converged", "diverged"],
            },
            "convergence_info": {"inc_abs": [2.0, 100.0], "res_abs": [1.0, np.nan]},
        },
    }

    # Add data for the second time step, and update global data.
    reference_data_after_2 = copy.deepcopy(reference_data_after_1)
    reference_data_after_2["global"].update(
        {
            "simulation_status_history": ["failed", "successful"],
            "final_simulation_status": "successful",
            "num_entries": 2,
            "final_time_reached": 0,
            "total_num_time_steps": 2,
            "total_num_failed_time_steps": 1,
            "num_iterations_history": [2, 3],
            "total_num_iterations": 5,
            "total_num_waisted_iterations": 2,
            "final_convergence_status": {
                "inc_abs": "converged",
                "res_abs": "converged",
                "max_iter": "diverged",
                "inc_inf": "converged",
                "res_inf": "converged",
                "inc_nan": "converged",
                "res_nan": "converged",
            },
        }
    )
    reference_data_after_2["1"] = {
        "final_time_reached": 0,
        "time_index": 1,
        "time": 0.5,
        "dt": 0.5,
        "num_iterations": 3,
        "simulation_status": "successful",
        "convergence_status": {
            "inc_abs": ["not_converged", "not_converged", "converged"],
            "res_abs": ["not_converged", "not_converged", "converged"],
            "max_iter": ["converged", "converged", "diverged"],
            "inc_inf": ["converged", "converged", "converged"],
            "res_inf": ["converged", "converged", "converged"],
            "inc_nan": ["converged", "converged", "converged"],
            "res_nan": ["converged", "converged", "converged"],
        },
        "convergence_info": {
            "inc_abs": [2.0, 1.0, 0.5],
            "res_abs": [1.0, 1.0, 0.5],
        },
    }

    # Add data for the second time step, and update global data.
    reference_data_after_3 = copy.deepcopy(reference_data_after_2)
    reference_data_after_3["global"].update(
        {
            "simulation_status_history": ["failed", "successful", "successful"],
            "final_simulation_status": "successful",
            "num_entries": 3,
            "final_time_reached": 1,
            "total_num_time_steps": 3,
            "total_num_failed_time_steps": 1,
            "num_iterations_history": [2, 3, 3],
            "total_num_iterations": 8,
            "total_num_waisted_iterations": 2,
            "final_convergence_status": {
                "inc_abs": "converged",
                "res_abs": "converged",
                "max_iter": "diverged",
                "inc_inf": "converged",
                "res_inf": "converged",
                "inc_nan": "converged",
                "res_nan": "converged",
            },
        }
    )
    reference_data_after_3["2"] = copy.deepcopy(reference_data_after_2["1"])
    reference_data_after_3["2"].update(
        {
            "final_time_reached": 1,
            "time_index": 2,
            "time": 1.0,
        }
    )

    # First time step - advance time to log the time step.
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    _ = solver.solve(model)

    # Check solver statistics.
    with open("solver_and_time_statistics.json", "r") as f:
        data = json.load(f)

    # Check the first time step (first loop). Adapt the global data accordingly.
    assert (
        DeepDiff(
            data,
            reference_data_after_1,
            ignore_numeric_type_changes=True,  # for nan
        )
        == {}
    )

    # Retry time step (second loop), so do not increase time.
    _ = solver.solve(model)

    # Check solver statistics.
    with open("solver_and_time_statistics.json", "r") as f:
        data = json.load(f)

    # Check all data.
    assert (
        DeepDiff(
            data,
            reference_data_after_2,
            ignore_numeric_type_changes=True,  # for nan
        )
        == {}
    )

    # Second time step (third loop) - advance time to log the time step.
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    _ = solver.solve(model)

    # Check solver statistics.
    with open("solver_and_time_statistics.json", "r") as f:
        data = json.load(f)

    # Check only the first time step. Need to adapt the global data accordingly.
    assert (
        DeepDiff(
            data,
            reference_data_after_3,
            ignore_numeric_type_changes=True,  # for nan
        )
        == {}
    )

    # Clean up.
    Path("solver_and_time_statistics.json").unlink()


def test_before_nonlinear_loop(default_newton_solver):
    """Unit test for the before_nonlinear_loop method of the Newton solver.

    Mainly check correct management of indices.

    """
    # Init model and solver.
    model = MockModel()
    solver = default_newton_solver

    # Mock a situation in the midst of a simulation (after some time step).
    solver.iteration_index = 10
    model.nonlinear_solver_statistics.index = 5

    # Call before_nonlinear_loop.
    solver.before_nonlinear_loop(model)

    # Ensure resetting of iteration index and increase of statistics index.
    assert solver.iteration_index == 0
    assert model.nonlinear_solver_statistics.index == 6


@pytest.mark.parametrize(
    "inc_history, res_history, is_converged, is_diverged",
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
    inc_history, res_history, is_converged, is_diverged, default_newton_solver
):
    """Test that the Newton loop exits correctly."""
    model = MockModel(
        nonlinear_increment_history=inc_history, residual_history=res_history
    )
    solver = default_newton_solver

    # Identify number of iterations from history.
    num_iter = len(inc_history)

    # Prepare for Newton loop.
    solver.before_nonlinear_loop(model)

    # Perform Newton loop.
    try:
        convergence_status, divergence_status = solver.nonlinear_loop(model)

        # Check that the returned statuses match expected values
        if is_converged:
            assert convergence_status.is_converged()
        else:
            assert convergence_status.is_not_converged()
        if is_diverged:
            assert divergence_status.is_diverged()
        else:
            assert divergence_status.is_converged()

        # Check that the number of iterations is as expected.
        assert solver.iteration_index == num_iter

    except Exception as e:
        # Newton loop only stops on convergence or divergence.
        # Need to handle the non-convergence and non-divergence case.
        assert not (is_converged or is_diverged), f"Unexpected exception: {e}"


@pytest.mark.parametrize(
    "convergence_status, divergence_status, expected_simulation_status",
    [
        (
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONVERGED,
            SimulationStatus.SUCCESSFUL,
        ),
        (
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.DIVERGED,
            SimulationStatus.SUCCESSFUL,  # Convergence trumps divergence
        ),
        (
            ConvergenceStatus.NOT_CONVERGED,
            ConvergenceStatus.DIVERGED,
            SimulationStatus.FAILED,
        ),
    ],
)
def test_after_nonlinear_loop(
    convergence_status,
    divergence_status,
    expected_simulation_status,
    default_newton_solver,
):
    """Unit test for the after_nonlinear_loop method of the Newton solver."""
    # Init model and solver.
    model = MockModel()
    solver = default_newton_solver

    # Minimal mimicking of loop.
    model.nonlinear_solver_statistics.simulation_status_history = [
        SimulationStatus.SUCCESSFUL
    ]

    simulation_status = solver.after_nonlinear_loop(
        model, convergence_status, divergence_status
    )

    # Check that the returned simulation status matches expected value.
    assert simulation_status == expected_simulation_status


def test_before_nonlinear_iteration(default_newton_solver):
    """Unit test for the before_nonlinear_iteration method of the Newton solver."""
    # Init model and solver.
    model = MockModel(nonlinear_increment_history=[2.0], residual_history=[1.0])
    solver = default_newton_solver

    # Check initial iteration index.
    assert solver.iteration_index == 0

    # Call before_nonlinear_iteration.
    solver.before_nonlinear_iteration(model)

    # Check that the iteration index has been increased.
    assert solver.iteration_index == 1


@pytest.mark.parametrize(
    "inc, res, iteration_index, is_converged, is_diverged",
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
    inc, res, iteration_index, is_converged, is_diverged, default_newton_solver
):
    """Test the after_nonlinear_iteration method of the Newton solver."""
    # Init model and solver.
    model = MockModel()
    solver = default_newton_solver

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
        assert convergence_status.is_not_converged()
    if is_diverged:
        assert divergence_status.is_diverged()
    else:
        assert divergence_status.is_converged()


@pytest.mark.parametrize(
    "inc, res, iteration_index, is_converged, is_diverged",
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
    inc, res, iteration_index, is_converged, is_diverged, default_newton_solver
):
    """Test the check_convergence method of the Newton solver."""
    # Init model and solver.
    model = MockModel()
    solver = default_newton_solver

    # Mock the nonlinear increment and residual for the last iteration.
    model.nonlinear_increment = np.array([inc])
    model.equation_system.residual = np.array([res])

    # Mock the number of iterations.
    solver.iteration_index = iteration_index

    # Check convergence.
    convergence_status, divergence_status, convergence_info = solver.check_convergence(
        model, model.nonlinear_increment
    )

    # Check that the returned statuses match expected values
    if is_converged:
        assert convergence_status.is_converged()
    else:
        assert convergence_status.is_not_converged()
    if is_diverged:
        assert divergence_status.is_diverged()
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


def test_update_solver_statistics(default_newton_solver):
    """Unit test for the update_solver_statistics method of the Newton solver."""
    model = MockModel()
    solver = default_newton_solver

    # Prepare model for updating solver statistics.
    model.before_nonlinear_loop()

    # Set some inputs for the update_solver_statistics method.
    # Here, we simulate one iteration with not converged status.
    simulation_status = SimulationStatus.IN_PROGRESS
    convergence_status = ConvergenceStatusCollection(
        {
            "inc_abs": ConvergenceStatus.NOT_CONVERGED,
            "res_abs": ConvergenceStatus.NOT_CONVERGED,
            "max_iter": ConvergenceStatus.CONVERGED,
        }
    )
    convergence_info = {"inc_abs": 2.0, "res_abs": 1.0}

    # Call the update_solver_statistics method
    solver.update_solver_statistics(
        model, simulation_status, convergence_status, convergence_info
    )

    # Check that the solver statistics have been updated correctly
    assert model.nonlinear_solver_statistics.num_iterations == 1
    assert model.nonlinear_solver_statistics.convergence_status == {
        "inc_abs": ["not_converged"],
        "res_abs": ["not_converged"],
        "max_iter": ["converged"],
    }
    assert model.nonlinear_solver_statistics.convergence_info == {
        "inc_abs": [2.0],
        "res_abs": [1.0],
    }
    assert model.nonlinear_solver_statistics.simulation_status_history == [
        "in_progress"
    ]


# ! ---- Test integration ---- ! #


@pytest.mark.parametrize("num_iterations", [1, 3])
def test_integration_nonlinear_iteration_count(num_iterations):
    """Test for checking if the nonlinear iterations are counted as expected.

    A pre set value of expected iterations is set, and the test checks that the
    iteration count matches the pre set value after convergence is obtained.

    """
    model = SinglePhaseFlow({"times_to_export": []})
    pp.run_time_dependent_model(
        model,
        {
            "nl_convergence_inc_atol": 0,
            "nl_convergence_res_atol": 0,
            "nl_max_iterations": num_iterations,
        },
    )

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
