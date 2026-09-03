"""Integration tests for the TimeStepper class."""

import json
import tempfile
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pytest
from deepdiff import DeepDiff
from scipy.sparse import csr_matrix

import porepy as pp
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.models.model_runner import ModelRunner, ModelRunnerStatusFailure
from porepy.models.protocol import PorePyModel
from porepy.numerics.ad.indexers import EquationOnDomain
from porepy.numerics.ad.operators import Variable
from porepy.time_stepper.time_step_control import Schedule, TimeInterval, TimeManager
from porepy.time_stepper.time_stepper import TimeStepper
from porepy.viz.solver_statistics import SolverStatisticsFactory

# MARK: test_model_delegate_methods_called


class MockEquationSystem:
    """Used internally in MockModel."""

    equation_indexer = pp.ad.EquationIndexer(indices={})
    variable_indexer = pp.ad.VariableIndexer(indices={})

    def assemble(self, evaluate_jacobian: bool = True, **kwargs):
        if not evaluate_jacobian:
            # Artificially satisfy residual norm convergence criterion.
            return np.array([1e-11])
        return pp.solvers.LinearSystem(
            matrix=csr_matrix(np.array([[1.0]])),
            rhs=np.array([1e-11]),
            equation_indexer=pp.ad.EquationIndexer(indices={}),
            variable_indexer=pp.ad.VariableIndexer(indices={}),
        )

    def get_variable_values(
        self, iterate_index: int, variables: list[pp.ad.Variable] | None = None
    ):
        assert iterate_index == 0
        return np.ones(5)


class MockMixedDimensionalGrid:
    """Used internally in MockModel."""

    def subdomains(self):
        return []


class MockModel(PorePyModel):
    """Used in test_model_delegate_methods_called, read the test docstring."""

    def __init__(self, statistics_path: Optional[Path] = None):
        self.sequence_of_calls: list[str] = []
        """Each delegate method of the mock model writes its name here when called."""

        self.equation_system = MockEquationSystem()
        """Used to evaluate convergence criteria."""

        self.mdg = MockMixedDimensionalGrid()
        """Used in _update_solver_statistics_after_nonlinear_solve."""

        self.nonlinear_solver_statistics = (
            SolverStatisticsFactory.create_statistics_type(
                nonlinear=True, time_dependent=True
            )()
        )
        if statistics_path:
            self.nonlinear_solver_statistics.path = Path(statistics_path)
        """Used by the TimeStepper and the NewtonSolver."""

        self.time_manager = pp.TimeManager(schedule=[0, 1], dt_init=1)
        """Used by the TimeStepper."""

    def before_time_step(self):
        self.sequence_of_calls.append("before_time_step")

    def before_nonlinear_loop(self):
        self.sequence_of_calls.append("before_nonlinear_loop")

        # We need to do it, otherwise will fail with IndexError on attempt to write
        # statistics.
        self.nonlinear_solver_statistics.increase_index()

    def before_nonlinear_iteration(self):
        self.sequence_of_calls.append("before_nonlinear_iteration")

    def after_nonlinear_iteration(
        self,
        nonlinear_increment: np.ndarray,
        updated_variables: Optional[list[pp.ad.Variable]] = None,
    ):
        self.sequence_of_calls.append("after_nonlinear_iteration")

    def after_nonlinear_convergence(self):
        self.sequence_of_calls.append("after_nonlinear_convergence")

    def after_nonlinear_failure(self):
        self.sequence_of_calls.append("after_nonlinear_failure")

    def after_time_step_convergence(self):
        self.sequence_of_calls.append("after_time_step_convergence")
        # Mimick the behavior of the real model.
        self.nonlinear_solver_statistics.save()

    def after_time_step_failure(self):
        self.sequence_of_calls.append("after_time_step_failure")
        # Mimick the behavior of the real model.
        self.nonlinear_solver_statistics.save()


class MockLinearSolver(pp.solvers.LinearSolverBase):
    """A mockup object for a linear solver. Always returns an array of ones of a given
    shape.

    """

    def __init__(self, num_dofs: int) -> None:
        self.num_dofs = num_dofs

    def solve_linear_system(
        self, linear_system: pp.solvers.LinearSystem
    ) -> tuple[np.ndarray, pp.solvers.LinearSolverStatus]:
        return np.ones(self.num_dofs), pp.solvers.LinearSolverStatusSuccess(
            solve_time=0
        )


class MaxIterationsConvergenceCriterion(pp.solvers.ConvergenceCriterion):
    """Accepts solution after the given number of iterations"""

    def __init__(self, max_iter: int) -> None:
        self.max_iter = max_iter
        self.num_iter = 0

    def check(self, **kwargs):
        self.num_iter += 1
        if self.num_iter < self.max_iter:
            # Second value is an arbitrary non-zero number.
            return pp.solvers.ConvergenceStatus.CONTINUE_ITERATING, 0.1
        else:
            return pp.solvers.ConvergenceStatus.CONVERGED, 0


@pytest.mark.parametrize("solver_type", ["nonlinear", "mock"])
def test_model_delegate_methods_called(
    solver_type: Literal["nonlinear", "mock"],
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
            "reject_num_iter": pp.solvers.MaxIterationsCriterion(max_iterations=2)
        },
    }
    model = MockModel()

    # Initialize the solver.
    if solver_type == "nonlinear":
        solver = pp.solvers.NewtonSolver(
            params=solver_params, linear_solver=MockLinearSolver(num_dofs=4)
        )
    elif solver_type == "mock":
        solver = DynamicNewtonSolver(
            num_nonlinear_iterations=[1, 1, 1, 1, 1],
            time_step_converged=[False, False, False, False, True],
            call_model_methods=False,
        )
        # We need to do it, otherwise will fail with IndexError on attempt to write
        # statistics. This is called in model.before_nonlinear_loop.
        model.nonlinear_solver_statistics.increase_index()
    else:
        raise ValueError

    # Initialize the real TimeStepper and the MockModel.
    time_stepper = TimeStepper.with_time_manager(model.time_manager)

    # Do the time step.
    time_stepper.perform_time_step(model=model, solver=solver)

    # Build an array of expected results.
    before_main_loop = [
        "before_time_step",
        "before_nonlinear_loop",
    ]
    main_loop = [
        "before_nonlinear_iteration",
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

    if solver_type == "nonlinear":
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
    """A mockup model used in combination with :class:`DynamicNewtonSolver`.

    See the description of the input parameters at `test_model_time_step_control`.

    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.time_step_history: list = []

        self.nonlinear_solver_statistics = (
            SolverStatisticsFactory.create_statistics_type(
                nonlinear=True, time_dependent=True
            )()
        )

    def before_time_step(self) -> None:
        super().before_time_step()  # The AD time step is expected to update here.
        # We need to do it, otherwise will fail with IndexError on attempt to write
        # statistics.
        self.nonlinear_solver_statistics.increase_index()
        self.num_nonlinear_iters = 0
        self.time_step_history.append(self.time_manager.dt)

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


class DynamicNewtonSolver(pp.solvers.NonlinearSolverBase):
    """A mockup Newton solver that returns convergence or divergence based on
    pre-defined values it takes during initialization. Used in
    :func:`test_model_time_step_control`. and
    :func:`test_model_delegate_methods_called`.

    """

    def __init__(
        self,
        num_nonlinear_iterations: list[int],
        time_step_converged: list,
        call_model_methods: bool = True,
    ):
        self.num_nonlinear_iterations: list[int] = num_nonlinear_iterations
        """Number of iteration in i-th nonlinear solve."""
        self.time_step_converged: list = time_step_converged
        """List of whether i-th nonlinear solve is successful."""
        self.current_idx = 0
        """Internal counter of encountered nonlinear problems."""
        self.call_model_methods = call_model_methods
        """Whether to call model.before_* and model.after_* methods."""

    def get_active_equations(self, model: PorePyModel) -> list[EquationOnDomain]:
        return []

    def get_active_variables(self, model: PorePyModel) -> list[Variable]:
        return []

    def solve(
        self, model: DynamicTimeStepTestCaseModel
    ) -> pp.solvers.NonlinearSolverStatus:
        num_nonlinear_iterations = self.num_nonlinear_iterations[self.current_idx]
        is_success = self.time_step_converged[self.current_idx]

        if self.call_model_methods:
            for _ in range(num_nonlinear_iterations):
                model.before_nonlinear_iteration()
                model.after_nonlinear_iteration(
                    nonlinear_increment=np.zeros(
                        model.equation_system.equation_indexer.size
                    )
                )
            if is_success:
                model.after_nonlinear_convergence()
            else:
                model.after_nonlinear_failure()

        linear_solver_statuses: list[pp.solvers.LinearSolverStatus] = [
            pp.solvers.LinearSolverStatusSuccess(solve_time=0.0)
        ] * num_nonlinear_iterations
        if is_success:
            result = pp.solvers.NewtonSolverConverged(
                convergence_statuses=pp.solvers.ConvergenceStatusCollection(),
                divergence_statuses=pp.solvers.ConvergenceStatusCollection(),
                linear_solver_statuses=linear_solver_statuses,
            )
        else:
            result = pp.solvers.NewtonSolverFailed(
                convergence_statuses=pp.solvers.ConvergenceStatusCollection(),
                divergence_statuses=pp.solvers.ConvergenceStatusCollection(),
                linear_solver_statuses=linear_solver_statuses,
            )

        self.current_idx += 1
        return result


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
            "num_nonlinear_iterations": [4, 3, 12, 1, 6, 9, 1, 1],
            # Time step 0 diverged after 4 iterations, time step 1 converged after 3
            # iterations, etc.
            "time_step_converged": [False, True, False] + [True] * 5,
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
            "failure_reason": "Constant time scheduler cannot decrease time step size",
        },
        # Case 3: An unsuccessful simulation with dynamic time stepping. Reached the
        # minimal time step and should fail.
        {
            "num_nonlinear_iterations": [1, 1, 1],
            "time_step_converged": [False, False, False],
            "exported_dt_expected": [1, 0.3, 0.1],
            "failure_reason": "is lower than the minimum admissible value",
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
            "failure_reason": "Max attempts (4) exhausted; stopping.",
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
    """Test time-step control of a PorePy simulation through using:
    - real ModelRunner;
    - real TimeStepper;
    - mock nonlinear solver;
    - mock PorePy model.

    Prescribed nonlinear iteration counts and convergence outcomes test time-step
    adaptation, schedule alignment, admissible step-size bounds, constant time
    steps, and terminal failures. The test also verifies state rollback after failed
    attempts, synchronization of the AD time step, the time-step history, and the final
    runner status.

    """
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
    )

    model = DynamicTimeStepTestCaseModel(
        params={
            "time_manager": time_manager,
            "times_to_export": [],  # Suspends export
        },
    )

    time_stepper = pp.time_stepper.TimeStepper.with_time_manager(
        model.time_manager, max_attempts=4
    )

    nonlinear_solver = DynamicNewtonSolver(
        num_nonlinear_iterations=num_nonlinear_iterations,
        time_step_converged=time_step_converged,
        call_model_methods=True,
    )
    model_runner = ModelRunner(
        model, nonlinear_solver=nonlinear_solver, time_stepper=time_stepper
    )
    if not should_fail:
        status = model_runner.run()
    else:
        try:
            model_runner.run()
        except RuntimeError as e:
            status = e.args[0]
        else:
            assert False
    assert np.allclose(model.time_step_history, exported_dt_expected)
    assert model.time_manager.final_time_reached() != should_fail
    if should_fail:
        assert isinstance(status, ModelRunnerStatusFailure)
        assert failure_reason in status.reason
    else:
        assert status.is_success()


def test_advanced_scheduler():
    time_manager = TimeManager.with_advanced_schedule(
        schedule=Schedule(
            intervals=[
                TimeInterval.create(
                    name="initialization",
                    t_start=0,
                    dt_start=1,
                    dt_max=1,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                ),
                TimeInterval.create(
                    name="injection",
                    t_start=3,
                    dt_start=0.1,
                    dt_min=0.01,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                ),
                TimeInterval.create(
                    name="relaxation",
                    t_start=3.02,
                    dt_start=1e2,
                    constraints=[
                        pp.time_stepper.TargetNonlinearIterations(),
                    ],
                ),
            ],
            t_end=3e2,
        ),
    )
    model = DynamicTimeStepTestCaseModel(
        params={
            "time_manager": time_manager,
            "times_to_export": [],  # Suspends export
        },
    )
    nonlinear_solver = DynamicNewtonSolver(
        num_nonlinear_iterations=[2] * 10,
        # The first injection step fails and is retried at dt_min. All other
        # attempts converge.
        time_step_converged=[True, True, True, False] + [True] * 6,
        call_model_methods=True,
    )
    model_runner = ModelRunner(model, nonlinear_solver=nonlinear_solver)

    status = model_runner.run()
    assert status.is_success()
    dt_expected = [
        # Initialization.
        1,
        1,
        1,
        # Injection: align with the next interval, then retry at dt_min.
        0.02,
        0.01,
        0.01,
        # Relaxation.
        130,
        166.98,
    ]
    assert np.allclose(model.time_step_history, dt_expected)


# MARK: Statistics


def default_newton_solver(iter_converge: int):
    """Initialize the Newton solver with convergence criteria used in test.

    Parameters:
        iter_converge: Converge after this number of nonlinear iterations. After this,
            will always converge, no matter what the time step is.

    """
    return pp.solvers.NewtonSolver(
        params={
            # MockModel does not decrease residual, so we use mock convergence criteria
            # here. To test statistics writing, we use two of them. Keys are arbitrary.
            "nl_convergence_criteria": {
                "crit1": MaxIterationsConvergenceCriterion(max_iter=1),
                "crit2": MaxIterationsConvergenceCriterion(max_iter=iter_converge),
            },
            "nl_divergence_criteria": {
                "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=2),
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
        linear_solver=MockLinearSolver(num_dofs=4),
    )


@pytest.fixture
def statistics_path(request):
    """Fixture with cleanup of the file. Making the file name unique for each test to
    facilitate runing in parallel without collisions.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        statistics_path = (
            Path(tmpdir) / f"solver_and_time_statistics_{request.node.nodeid}.json"
        )
        yield statistics_path


def test_solve_convergence_time_dependent_statistics(statistics_path: Path):
    """Test that the solver statistics are updated correctly on convergence to check
    correct behavior after convergence, for a time-dependent model. The test is moved
    from test_nonlinear_solvers.py

    """
    # Minimal setup.
    model = MockModel(statistics_path=statistics_path)
    solver = default_newton_solver(iter_converge=2)
    model.time_manager = TimeManager(schedule=[0, 1], dt_init=0.5, constant_dt=True)
    time_stepper = TimeStepper.with_time_manager(model.time_manager)

    # Define the reference solver statistics, for two time steps.
    reference_data = {
        "global": {
            "num_cells": {},
            "num_domains": {},
            "simulation_status_history": [
                "successful",
                "successful",
            ],
            "final_simulation_status": "successful",
            "num_entries": 2,
            "final_time_reached": 1,
            "total_num_time_steps": 2,
            "total_num_failed_time_steps": 0,
            "num_iterations_history": [2, 1],
            "total_num_iterations": 3,
            "total_num_waisted_iterations": 0,
            "final_convergence_status": {
                "crit1": "converged",
                "crit2": "converged",
                "max_iter": "converged",
                "inc_inf": "converged",
                "res_inf": "converged",
                "inc_nan": "converged",
                "res_nan": "converged",
            },
        },
        "0": {
            "final_time_reached": 0,
            "time_index": 1,  # Note that time_index is off-by-one from the dict key.
            "time": 0.5,
            "dt": 0.5,
            "num_iterations": 2,
            "simulation_status": "successful",
            "solver_status": "successful",
            "convergence_status": {
                "crit1": ["converged", "converged"],
                "crit2": ["continue_iterating", "converged"],
                # Note that we both converge and fail, but the current logic treats it
                # as success.
                "max_iter": ["converged", "failed"],
                "inc_inf": ["converged", "converged"],
                "res_inf": ["converged", "converged"],
                "inc_nan": ["converged", "converged"],
                "res_nan": ["converged", "converged"],
            },
            "convergence_info": {"crit1": [0, 0], "crit2": [0.1, 0]},
        },
        "1": {
            "final_time_reached": 1,
            "time_index": 2,
            "time": 1.0,
            "dt": 0.5,
            "num_iterations": 1,
            "simulation_status": "successful",
            "solver_status": "successful",
            "convergence_status": {
                "crit1": ["converged"],
                "crit2": ["converged"],
                "max_iter": ["converged"],
                "inc_inf": ["converged"],
                "res_inf": ["converged"],
                "inc_nan": ["converged"],
                "res_nan": ["converged"],
            },
            "convergence_info": {
                "crit1": [0],
                "crit2": [0],
            },
        },
    }

    # Making two time steps.
    status = time_stepper.perform_time_step(model=model, solver=solver)
    assert status.is_success()

    status = time_stepper.perform_time_step(model=model, solver=solver)
    assert status.is_success()

    # Check solver statistics.
    with open(statistics_path, "r") as f:
        data = json.load(f)

    # Check for both time steps.
    assert DeepDiff(data, reference_data) == {}


def test_solve_failure_time_dependent_statistics(statistics_path: Path):
    """Test that the solver statistics are updated correctly on convergence to check
    correct behavior after failure, for a time-dependent model.

    """
    model = MockModel(statistics_path=statistics_path)
    solver = default_newton_solver(iter_converge=5)
    model.time_manager = TimeManager(
        schedule=[0, 1], dt_init=1, constant_dt=False, dt_min_max=(0.5, 1)
    )
    time_stepper = TimeStepper.with_time_manager(model.time_manager)

    # It will attempt to make a time step twice here, with dt=1 and dt=0.5. Both will
    # fail after two unsuccessful nonlinear iterations.
    status = time_stepper.perform_time_step(model=model, solver=solver)
    assert status.is_failure()

    # The time stepper gave up and the real simulation would be stopped at this moment.
    # But we use a mock convergence criterion, so we can try one more time with the same
    # dt=0.5. This time, the time step will converge after 1 nonlinear iteration.
    status = time_stepper.perform_time_step(model=model, solver=solver)
    assert status.is_success()

    # The 4th time-step attempt will converge after 1 iteration with dt=0.5.
    status = time_stepper.perform_time_step(model=model, solver=solver)
    assert status.is_success()

    # Check solver statistics.
    with open(statistics_path, "r") as f:
        data = json.load(f)

    # Define the reference solver statistics, for 3 time steps.
    reference_data = {
        "global": {
            "num_cells": {},
            "num_domains": {},
            # Four time-step attempts, 0th failed (the time stepper retried), 1st
            # failed (the time stepper gave up), 2nd and 3rd succeeded.
            "simulation_status_history": [
                "in_progress",
                "failed",
                "successful",
                "successful",
            ],
            "final_simulation_status": "successful",
            "num_entries": 4,
            "final_time_reached": 1,
            "total_num_time_steps": 4,
            "total_num_failed_time_steps": 2,
            "num_iterations_history": [2, 2, 1, 1],
            "total_num_iterations": 6,
            "total_num_waisted_iterations": 4,
            "final_convergence_status": {
                "crit1": "converged",
                "crit2": "converged",
                "max_iter": "converged",
                "inc_inf": "converged",
                "res_inf": "converged",
                "inc_nan": "converged",
                "res_nan": "converged",
            },
        },
        "0": {
            "simulation_status": "in_progress",
            "solver_status": "failed",
            "final_time_reached": 1,
            "time_index": 1,
            "time": 1,
            "dt": 1,
            "num_iterations": 2,
            "convergence_status": {
                "crit1": ["converged", "converged"],
                "crit2": ["continue_iterating", "continue_iterating"],
                "max_iter": ["converged", "failed"],
                "inc_inf": ["converged", "converged"],
                "res_inf": ["converged", "converged"],
                "inc_nan": ["converged", "converged"],
                "res_nan": ["converged", "converged"],
            },
            "convergence_info": {"crit1": [0, 0], "crit2": [0.1, 0.1]},
        },
        "1": {
            "simulation_status": "failed",
            "solver_status": "failed",
            "final_time_reached": 0,
            "time_index": 1,
            "time": 0.5,
            "dt": 0.5,
            "num_iterations": 2,
            "convergence_status": {
                "crit1": ["converged", "converged"],
                "crit2": ["continue_iterating", "continue_iterating"],
                "max_iter": ["converged", "failed"],
                "inc_inf": ["converged", "converged"],
                "res_inf": ["converged", "converged"],
                "inc_nan": ["converged", "converged"],
                "res_nan": ["converged", "converged"],
            },
            "convergence_info": {"crit1": [0, 0], "crit2": [0.1, 0.1]},
        },
        "2": {
            "simulation_status": "successful",
            "solver_status": "successful",
            "final_time_reached": 0,
            "time_index": 1,
            "time": 0.5,
            "dt": 0.5,
            "num_iterations": 1,
            "convergence_status": {
                "crit1": ["converged"],
                "crit2": ["converged"],
                "max_iter": ["converged"],
                "inc_inf": ["converged"],
                "res_inf": ["converged"],
                "inc_nan": ["converged"],
                "res_nan": ["converged"],
            },
            "convergence_info": {"crit1": [0], "crit2": [0]},
        },
        "3": {
            "simulation_status": "successful",
            "solver_status": "successful",
            "final_time_reached": 1,
            "time_index": 2,
            "time": 1.0,
            "dt": 0.5,
            "num_iterations": 1,
            "convergence_status": {
                "crit1": ["converged"],
                "crit2": ["converged"],
                "max_iter": ["converged"],
                "inc_inf": ["converged"],
                "res_inf": ["converged"],
                "inc_nan": ["converged"],
                "res_nan": ["converged"],
            },
            "convergence_info": {"crit1": [0], "crit2": [0]},
        },
    }

    assert (
        DeepDiff(
            data,
            reference_data,
            ignore_numeric_type_changes=True,  # to treat 1 and 1.0 as equal.
        )
        == {}
    )
