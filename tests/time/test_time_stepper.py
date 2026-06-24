"""Integration tests for the TimeStepper class."""

from typing import Literal

import pytest

from porepy.models.protocol import PorePyModel
from porepy.numerics.linear_solvers import LinearSolver
from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceCriterion,
    ConvergenceStatus,
    MaxIterationsCriterion,
)
from porepy.numerics.nonlinear.nonlinear_solvers import NewtonSolver
from porepy.numerics.time_step_control import TimeManager
from porepy.time.time_stepper import TimeStepper
import numpy as np

from porepy.viz.solver_statistics import SolverStatisticsFactory


class MockEquationSystem:
    """Used internally in MockModel."""

    def assemble(self, evaluate_jacobian: bool):
        assert evaluate_jacobian == False
        return np.ones(5)

    def get_variable_values(self, iterate_index: int):
        assert iterate_index == 0
        return np.ones(5)


class MockMixedDimensionalGrid:
    """Used internally in MockModel."""

    def subdomains(self):
        return []


class MockModel(PorePyModel):
    """Used in test_model_delegate_methods_called, read the test docstring."""

    def __init__(self, is_nonlinear: bool):
        self.sequence_of_calls: list[str] = []
        """Each delegate method of the mock model writes its name here when called."""

        self.is_nonlinear: bool = is_nonlinear
        """Used by the LinearSolver class to raise an exception if we are nonlinear."""

        self.equation_system = MockEquationSystem()
        """Used to evaluate convergence criteria."""

        self.mdg = MockMixedDimensionalGrid()
        """Used in LinearSolver.update_solver_statistics."""

        self.nonlinear_solver_statistics = (
            SolverStatisticsFactory.create_statistics_type(
                nonlinear=is_nonlinear, time_dependent=True
            )()
        )
        """Used by the TimeStepper and the NewtonSolver / LinearSolver"""

    def _is_nonlinear_problem(self):
        """Used by the LinearSolver class to raise an exception if we are nonlinear."""
        return self.is_nonlinear

    def before_time_step(self):
        self.sequence_of_calls.append("before_time_step")

    def before_nonlinear_loop(self):
        self.sequence_of_calls.append("before_nonlinear_loop")

        # We need to do it, otherwise will fail with IndexError on attempt to write
        # statistics.
        self.nonlinear_solver_statistics.increase_index()

    def before_nonlinear_iteration(self):
        self.sequence_of_calls.append("before_nonlinear_iteration")

    def assemble_linear_system(self):
        self.sequence_of_calls.append("assemble_linear_system")

    def solve_linear_system(self):
        self.sequence_of_calls.append("solve_linear_system")
        return np.ones(5)

    def after_nonlinear_iteration(self, nonlinear_increment):
        self.sequence_of_calls.append("after_nonlinear_iteration")

    def after_nonlinear_convergence(self):
        self.sequence_of_calls.append("after_nonlinear_convergence")

    def after_nonlinear_failure(self):
        self.sequence_of_calls.append("after_nonlinear_failure")

    def after_time_step_convergence(self):
        self.sequence_of_calls.append("after_time_step_convergence")

    def after_time_step_failure(self):
        self.sequence_of_calls.append("after_time_step_failure")


class MockNonlinearSolver:
    """Used in test_model_delegate_methods_called, read the test docstring."""

    def __init__(self, num_iters_for_success: int):
        self._iter = 0
        """Number of times solve was called."""
        self.num_iters_for_success: int = num_iters_for_success
        """Number of times solve must be called to return success."""

    def solve(self, model) -> ConvergenceStatus:
        self._iter += 1
        if self._iter < self.num_iters_for_success:
            return ConvergenceStatus.FAILED
        return ConvergenceStatus.CONVERGED


class MaxIterationsConvergenceCriterion(ConvergenceCriterion):
    """Accepts solution after the given number of iterations"""

    def __init__(self, max_iter: int) -> None:
        self.max_iter = max_iter
        self.num_iter = 0

    def check(self, **kwargs):
        self.num_iter += 1
        if self.num_iter < self.max_iter:
            return ConvergenceStatus.CONTINUE_ITERATING, 0
        else:
            return ConvergenceStatus.CONVERGED, 0


@pytest.mark.parametrize("solver_type", ["nonlinear", "linear", "mock"])
def test_model_delegate_methods_called(
    solver_type: Literal["nonlinear", "linear", "mock"],
):
    """This integration test is an attempt to solidify the API between the PorePy model
    and the objects that control it from the outside (TimeStepper and/or NewtonSolver).

    PorePy model exposes multiple delegate methods (named before_* and after_*). This
    test replaces a real PorePy model with a MockModel that remembers the order in which
    we called these methods. We make a single time step with TimeStepper and ensure that
    the methods are called in the right order and amount.

    The test covers:
    - NewtonSolver: fails 2 times, each after 2 unsuccessful nonlinear iterations. Then
        converges.
    - LinearSolver: fails 4 times after unsuccessful linear solve. Then converges.
    - MockSolver: fails 2 times then converges. The purpose is that it does not call the
        delegate methods, so we check that the TimeStepper called only those delegate
        methods it is supposed to.

    """
    # Customize convergence criteria.
    solver_params = {
        "nl_convergence_criteria": {
            "accept_num_iter": MaxIterationsConvergenceCriterion(max_iter=5)
        },
        "nl_divergence_criteria": {
            "reject_num_iter": MaxIterationsCriterion(max_iterations=2)
        },
    }

    # Initialize the solver.
    is_nonlinear = True
    if solver_type == "linear":
        solver = LinearSolver(params=solver_params)
        is_nonlinear = False
    elif solver_type == "nonlinear":
        solver = NewtonSolver(params=solver_params)
    elif solver_type == "mock":
        solver = MockNonlinearSolver(num_iters_for_success=5)
    else:
        raise ValueError

    # initialize the real TimeStepper and the MockModel.
    time_stepper = TimeStepper(
        time_manager=TimeManager(
            schedule=[0, 1], dt_init=1, constant_dt=False, dt_min_max=(0.1, 2)
        )
    )
    model = MockModel(is_nonlinear=is_nonlinear)

    # Do the time step.
    time_stepper.perform_time_step(model=model, solver=solver)

    # Build an array of expected results.
    before_main_loop = [
        "before_time_step",
        "before_nonlinear_loop",
    ]
    main_loop = [
        "before_nonlinear_iteration",
        "assemble_linear_system",
        "solve_linear_system",
        "after_nonlinear_iteration",
    ]
    after_main_loop_success = [
        "after_nonlinear_convergence",
        "after_time_step_convergence",
    ]
    after_main_loop_failure = [
        "after_nonlinear_failure",
        "after_time_step_failure",
    ]

    if solver_type == "linear":
        expected_result = (
            # Four unsuccessful attempts with cutting the time step.
            (before_main_loop + main_loop + after_main_loop_failure) * 4
            # And a single successful time step
            + before_main_loop
            + main_loop
            + after_main_loop_success
        )
    elif solver_type == "nonlinear":
        expected_result = (
            # Two unsuccessful time steps, each with 2 nonlinear solves.
            (before_main_loop + main_loop * 2 + after_main_loop_failure) * 2
            # And a single successful time step.
            + before_main_loop
            + main_loop
            + after_main_loop_success
        )
    elif solver_type == "mock":
        expected_result = ["before_time_step", "after_time_step_failure"] * 4 + [
            "before_time_step",
            "after_time_step_convergence",
        ]
    else:
        raise ValueError

    # Compare with expected.
    assert model.sequence_of_calls == expected_result
