"""Integration tests for the TimeStepper class."""

from typing import Literal

import pytest

from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.models.model_runner import ModelRunner, ModelRunnerStatusFailure
from porepy.models.protocol import PorePyModel
from porepy.numerics.linear_solvers import LinearSolver
from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceCriterion,
    ConvergenceInfoCollection,
    ConvergenceStatus,
    ConvergenceStatusCollection,
    MaxIterationsCriterion,
)
from porepy.numerics.nonlinear.nonlinear_solvers import NewtonSolver
from porepy.numerics.time_step_control import TimeManager
from porepy.time.time_stepper import TimeStepper
import numpy as np

from porepy.viz.solver_statistics import (
    NonlinearSolverAndTimeStatistics,
    SolverStatisticsFactory,
)

# MARK: test_model_delegate_methods_called


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


# MARK: test_model_time_step_control


class DynamicTimeStepTestCaseModel(SinglePhaseFlow):
    """A mockup model that stores the lists that control when to converge, when to
    diverge, etc. Used by DynamicNewtonSolver.

    See the description of the input parameters at `test_model_time_step_control`.

    """

    def __init__(
        self,
        num_nonlinear_iterations: list[int],
        time_step_converged: list,
        params: dict,
    ):
        super().__init__(params)
        self.time_step_idx: int = -1
        self.num_nonlinear_iters: int = 0
        self.num_nonlinear_iterations: list[int] = num_nonlinear_iterations
        self.time_step_converged: list = time_step_converged
        self.time_step_history: list = []

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()  # The AD time step is expected to update here.
        self.time_step_idx += 1
        self.num_nonlinear_iters = 0
        self.time_step_history.append(self.time_manager.dt)

    def before_nonlinear_iteration(self):
        super().before_nonlinear_iteration()

        # The AD time step should not change throughout the Newton iterations.
        assert (
            self.equation_system.evaluate(self.ad_time_step) == self.time_manager.dt
        ), "The AD time step value conflicts with the value from the time_manager."

        # The initial guess for the unknown time step values should be equal to the
        # known time step values. See https://github.com/pmgbergen/porepy/issues/1205.
        if self.num_nonlinear_iters == 0:
            iterate_values = self.equation_system.get_variable_values(iterate_index=0)
            state_values = self.equation_system.get_variable_values(time_step_index=0)
            assert np.all(iterate_values == state_values), (
                "Likely, 'iterate' was not reset after the unsuccessful time step."
            )

        self.num_nonlinear_iters += 1

    def _is_nonlinear_problem(self):
        return True

    # Minimizing computational expenses.
    def assemble_linear_system(self) -> None:
        pass

    def solve_linear_system(self) -> np.ndarray:
        return np.ones(self.equation_system.num_dofs())


class DynamicNewtonSolver(NewtonSolver):
    """A mockup Newton solver that returns convergence or divergence based on what
    DynamicTimeStepTestCaseModel prescribes. Used in `test_model_time_step_control`.

    """

    def check_convergence(
        self, model: DynamicTimeStepTestCaseModel, nonlinear_increment
    ) -> tuple[
        ConvergenceStatusCollection,
        ConvergenceStatusCollection,
        ConvergenceInfoCollection,
    ]:
        assert isinstance(
            model.nonlinear_solver_statistics, NonlinearSolverAndTimeStatistics
        )
        if (
            model.nonlinear_solver_statistics.num_iterations
            < model.num_nonlinear_iterations[model.time_step_idx] - 1
        ):
            return (
                ConvergenceStatusCollection(
                    {"crit": ConvergenceStatus.CONTINUE_ITERATING}
                ),
                ConvergenceStatusCollection(
                    {"div_crit": ConvergenceStatus.CONTINUE_ITERATING}
                ),
                ConvergenceInfoCollection({"crit": 1.0}),
            )
        if model.time_step_converged[model.time_step_idx] is True:
            return (
                ConvergenceStatusCollection({"crit": ConvergenceStatus.CONVERGED}),
                ConvergenceStatusCollection(
                    {"div_crit": ConvergenceStatus.CONTINUE_ITERATING}
                ),
                ConvergenceInfoCollection({"crit": 0.0}),
            )
        else:
            return (
                ConvergenceStatusCollection(
                    {"crit": ConvergenceStatus.CONTINUE_ITERATING}
                ),
                ConvergenceStatusCollection({"div_crit": ConvergenceStatus.FAILED}),
                ConvergenceInfoCollection({"crit": np.nan}),
            )


MAX_NONLINEAR_ITER = 10


@pytest.mark.parametrize(
    "params",
    [
        # Case 1: A successful simulation run with dynamic time stepping.
        # Covers these situations:
        # - decrease the time step after diverged
        # - decrease the time step after iteration limit
        # - increase the time step due to few nonlinear iterations
        # - keep the time step due to expected number of nonlinear iterations
        # - decrease the time step due to many nonlinear iterations (after convergence)
        # - decrease the time step to meet the schedule (last time step)
        {
            # Below reads as: time step 0 takes 4 nonlinear iterations, time step 1
            # takes 3 nonlinear iterations, etc.
            "num_nonlinear_iterations": [4, 3, MAX_NONLINEAR_ITER + 2, 1, 6, 9, 1, 1],
            # Time step 0 diverged after 4 iterations, time step 1 converged after 3
            # iterations, etc. "unreachable" means that the convergence check should not
            # be called due to exceeding the iteration limit.
            "time_step_converged": [False, True, "unreachable"] + [True] * 5,
            # Time step magnitudes to compare with. These are known values produced with
            # the settings of the TimeStepper found in the test function below.
            "exported_dt_expected": [1, 0.3, 0.6, 0.18, 0.36, 0.36, 0.144, 0.006],
        },
        # Case 2: constant_dt. Should fail after nonlinear divergence.
        {
            "constant_dt": True,
            "num_nonlinear_iterations": [2, 3],
            "time_step_converged": [True, False],
            "exported_dt_expected": [1, 1],
            "schedule_end": 2,  # Matches the constant dt = 1.
            "failure_reason": "Max retries (1)",
        },
        # Case 3: An unsuccessful simulation with dynamic time stepping. Reached the
        # minimal time step and should fail.
        {
            "num_nonlinear_iterations": [1, 1, 1],
            "time_step_converged": [False, False, False],
            "exported_dt_expected": [1, 0.3, 0.1],
            "failure_reason": "time step achieved its minimum admissible value",
        },
        # Case 4: The time step fails right before the schedule point. Expected to
        # decrease dt and meet the schedule regardless.
        {
            "num_nonlinear_iterations": [1, 1, 1, 1, 1],
            "time_step_converged": [True, False, True, True, True],
            "exported_dt_expected": [1, 0.35, 0.105, 0.21, 0.035],
        },
        # Case 5: Fail because we reach the limit of attempts to cut the time step.
        {
            "num_nonlinear_iterations": [1, 1, 1, 1, 1, 1],
            "time_step_converged": [True, True, False, False, False, False],
            "exported_dt_expected": [1, 2, 4, 1.2, 0.36, 0.108],
            "schedule_end": 10,  # Far beyond possible dt to avoid dt_max clipping.
            "failure_reason": "Max retries (4)",
        },
        # Case 6: All time steps are successful so dt reaches dt_max and remains it.
        {
            "num_nonlinear_iterations": [1, 1, 1, 1, 1, 1],
            "time_step_converged": [True, True, True, True, True, True],
            "exported_dt_expected": [1, 2, 4, 5, 5, 0.1],
            "schedule_end": 17.1,
        },
        # Case 7: All time steps are successful, but take too many nonlinear iterations,
        # so dt reaches dt_min and remains it.
        {
            "num_nonlinear_iterations": [8, 8, 8, 8, 8, 8],
            "time_step_converged": [True, True, True, True, True, True],
            "exported_dt_expected": [1, 0.4, 0.16, 0.1, 0.1, 0.1],
            "schedule_end": 1.86,
        },
        # Case 8: Time step we need to make to meet the schedule is below minimal dt.
        # The current behavior is that we ignore dt_min to meet the schedule.
        # This implies that if we hit this in the middle of the simulation, next dt will
        # be dt_min, which is quite sub-optimal.
        {
            "num_nonlinear_iterations": [1, 1],
            "time_step_converged": [True, True],
            "exported_dt_expected": [1, 1e-5],
            "schedule_end": 1 + 1e-5,
        },
        # Case 9: Successful constant_dt simulation, but Newton makes either too few or
        # to many interations. Check that dt remains the same.
        {
            "constant_dt": True,
            "num_nonlinear_iterations": [1, 8, 1],
            "time_step_converged": [True, True, True],
            "exported_dt_expected": [1, 1, 1],
            "schedule_end": 3,
        },
    ],
)
def test_model_time_step_control(params: dict):
    """The integration test for the `TimeStepper` - how it interacts with a real PorePy
    model."""
    constant_dt = params.get("constant_dt", False)
    num_nonlinear_iterations = params["num_nonlinear_iterations"]
    time_step_converged = params["time_step_converged"]
    exported_dt_expected = params["exported_dt_expected"]
    failure_reason = params.get("failure_reason", "")
    # 1.35 is the default value assumed by most of the tests. It is arbitrary.
    schedule_end = params.get("schedule_end", 1.35)

    should_fail = len(failure_reason) != 0

    time_manager = TimeManager(
        schedule=(0, schedule_end),
        dt_init=1,
        constant_dt=constant_dt,
        dt_min_max=(0.1, 5),
        iter_relax_factors=(0.4, 2),
        iter_optimal_range=(4, 7),
        recomp_factor=0.3,
        recomp_max=4,
    )

    model = DynamicTimeStepTestCaseModel(
        num_nonlinear_iterations=num_nonlinear_iterations,
        time_step_converged=time_step_converged,
        params={
            "time_manager": time_manager,
            "times_to_export": [],  # Suspends export
        },
    )
    solver_params = {
        "nonlinear_solver": DynamicNewtonSolver,
        "nl_convergence_inc_atol": 1e-6,
        "nl_max_iterations": MAX_NONLINEAR_ITER,
    }

    status = ModelRunner(model, solver_params).run()
    assert np.allclose(model.time_step_history, exported_dt_expected)
    assert model.time_manager.final_time_reached() != should_fail
    if should_fail:
        assert isinstance(status, ModelRunnerStatusFailure)
        assert status.reason.find(failure_reason) != -1
    else:
        assert status.is_success()
