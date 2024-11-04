"""
This module contains a collection of unit tests for the `TimeManager` class. The
module contains two test classes, namely: `TestParameterInputs` and `TestTimeControl`.
The module also contains the integration test to the models.

`TestParameterInputs` contains test methods that check the sanity of the input
parameters. This includes checks for default parameters in initialization, checks for
admissible parameter values, and checks for admissible parameter types.

`TestTimeControl` contains checks for the correct behaviour of the time-stepping control
algorithm via the `next_time_step()` method. Here, the tests are designed to check that
the time step is indeed adapted (based on iterations or recomputation criteria) and
corrected (based on minimum and maximum allowable time steps or to satisfy required
scheduled times).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.models.fluid_mass_balance import SinglePhaseFlow


class TestParameterInputs:
    """The following tests are written to check the sanity of the input parameters"""

    def test_default_parameters_and_attribute_initialization(self):
        """Test the default parameters and initialization of attributes."""
        time_manager = pp.TimeManager(schedule=[0, 1], dt_init=0.1)
        np.testing.assert_equal(time_manager.schedule, np.array([0, 1]))
        assert time_manager.time_init == 0
        assert time_manager.time_final == 1
        assert time_manager.dt_init == 0.1
        assert time_manager.dt_min_max == (0.001, 0.1)
        assert time_manager.iter_max == 15
        assert time_manager.iter_optimal_range == (4, 7)
        assert time_manager.iter_relax_factors == (0.7, 1.3)
        assert time_manager.recomp_factor == 0.5
        assert time_manager.recomp_max == 10
        assert time_manager.time == 0
        assert time_manager.dt == 0.1

    @pytest.mark.parametrize(
        "schedule", [(0, 0.5, 1), [0, 0.5, 1], np.array([0, 0.5, 1])]
    )
    def test_schedule_argument_type_compatibility(self, schedule):
        """The 'schedule' object is supposed to take any array-like object."""
        time_manager = pp.TimeManager(schedule=schedule, dt_init=0.01)
        assert isinstance(time_manager.schedule, np.ndarray)
        assert (time_manager.schedule == np.array([0.0, 0.5, 1.0])).all()

    @pytest.mark.parametrize(
        "schedule, dt_init",
        [
            ([], 0.1),
            ([1], 0.1),
        ],
    )
    def test_schedule_length_greater_than_2(self, schedule, dt_init):
        """An error should be raised if len(schedule) < 2."""
        msg = "Expected schedule with at least two elements."
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(schedule=schedule, dt_init=dt_init)
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init",
        [
            ([-1, 10], 0.1),
            ([0, -3], 0.1),
            ([1, 2, -100, 3, 4], 0.1),
        ],
    )
    def test_positive_time_in_schedule(self, schedule, dt_init):
        """An error should be raised if a negative time is encountered in the schedule."""
        msg = "Encountered at least one negative time in schedule."
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(schedule=schedule, dt_init=dt_init)
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init",
        [
            ([0, 1, 2, 5, 1], 0.1),
            ([0, 1, 1, 2], 0.1),
            ([100, 200, 50], 0.1),
        ],
    )
    def test_strictly_increasing_time_in_schedule(self, schedule, dt_init):
        """An error should be raised if times in schedule are not strictly increasing."""
        msg = "Schedule must contain strictly increasing times."
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(schedule=schedule, dt_init=dt_init)
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init",
        [
            ([0, 1], 1),
            ([0, 1], 1.0),
            ([0, 1, 2], 1),
            ([0, 0.05, 2.0, 4.0, 10.0], 0.05),
            ([5.0, 10.0, 60.0, 62.5, 70.0], 2.5),
        ],
    )
    def test_constant_dt_compatibility_with_schedule(self, schedule, dt_init):
        """No error should be raised if dt and schedule are compatible"""
        try:
            pp.TimeManager(schedule=schedule, dt_init=dt_init, constant_dt=True)
        except Exception as exc:
            assert False, f"The following exception was raised: {exc}"

    @pytest.mark.parametrize(
        "schedule, dt_init",
        [
            ([0, 1], 0.4),
            ([0, 1], 0.3333),
            ([0, 0.4, 0.5, 0.8, 1], 0.2),
            ([13.1, 13.2, 13.3], 0.2),
        ],
    )
    def test_raise_error_incompatible_dt_and_schedule(self, schedule, dt_init):
        """An error should be raised if the schedule is incompatible with the constant time
        step."""
        msg = (
            "Mismatch between the time step and scheduled time. Make sure the two are "
            "compatible, or consider adjusting the tolerances of the sanity check."
        )
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(schedule=schedule, dt_init=dt_init, constant_dt=True)
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init",
        [
            ([0, 1], -1),
            ([0, 1], 0),
        ],
    )
    def test_positive_initial_time_step(self, schedule, dt_init):
        """An error should be raised if initial time step is non-positive."""
        msg = "Initial time step must be positive."
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(schedule=schedule, dt_init=dt_init)
        assert msg in str(excinfo.value)

    def test_initial_time_step_smaller_than_final_time(self):
        """An error should be raised if initial time step is larger than final time."""
        msg = "Initial time step cannot be larger than final simulation time."
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(schedule=[0, 1], dt_init=1.0001)
        assert msg in str(excinfo.value)

    def test_initial_time_step_larger_than_minimum_time_step(self):
        """An error should be raised if initial time step is less than minimum time step."""
        msg_dtmin = "Initial time step cannot be smaller than minimum time step. "
        msg_unset = (
            "This error was raised since `dt_min_max` was not set on "
            "initialization. Thus, values of dt_min and dt_max were assigned "
            "based on the final simulation time. If you still want to use this "
            "initial time step, consider passing `dt_min_max` explicitly."
        )
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(schedule=[0, 1], dt_init=0.0009)
        assert (msg_dtmin + msg_unset) in str(excinfo.value)

    def test_initial_time_step_smaller_than_maximum_time_step(self):
        """An error should be raised if initial time step is larger than the maximum time
        step."""
        msg_dtmax = "Initial time step cannot be larger than maximum time step. "
        msg_unset = (
            "This error was raised since `dt_min_max` was not set on "
            "initialization. Thus, values of dt_min and dt_max were assigned "
            "based on the final simulation time. If you still want to use this "
            "initial time step, consider passing `dt_min_max` explicitly."
        )
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(schedule=[0, 1], dt_init=0.11)
        assert (msg_dtmax + msg_unset) in str(excinfo.value)

    @pytest.mark.parametrize("iter_max", [0, -1])
    def test_max_number_of_iterations_positive(self, iter_max):
        """An error should be raised if the maximum number of iterations is not positive."""
        msg = "Maximum number of iterations must be positive."
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(schedule=[0, 1], dt_init=0.1, iter_max=iter_max)
        assert msg in str(excinfo.value)

    def test_lower_iter_endpoint_smaller_than_upper_iter_endpoint(self):
        """An error should be raised if the lower endpoint of the optimal iteration range is
        larger than the upper endpoint of the optimal iteration range."""
        iter_optimal_range = (3, 2)
        msg = (
            f"Lower endpoint '{iter_optimal_range[0]}' of optimal iteration "
            f"range cannot be larger than upper endpoint '{iter_optimal_range[1]}'."
        )
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(
                schedule=[0, 1],
                dt_init=0.1,
                iter_max=5,
                iter_optimal_range=iter_optimal_range,
            )
        assert msg in str(excinfo.value)

    def test_upper_iter_endpoint_less_or_equal_than_max_iter(self):
        """An error should be raised if the upper endpoint of the optimal iteration range is
        larger than the maximum number of iterations."""
        iter_optimal_range = (2, 6)
        iter_max = 5
        msg = (
            f"Upper endpoint '{iter_optimal_range[1]}' of optimal iteration "
            f"range cannot be larger than maximum number of iterations "
            f"'{iter_max}'."
        )
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(
                schedule=[0, 1],
                dt_init=0.1,
                iter_max=iter_max,
                iter_optimal_range=iter_optimal_range,
            )
        assert msg in str(excinfo.value)

    def test_lower_iter_endpoint_greater_than_or_equal_to_zero(self):
        """An error should be raised if the lower iteration range is less than zero."""
        iter_optimal_range = (-1, 2)
        msg = (
            f"Lower endpoint '{iter_optimal_range[0]}' of optimal iteration "
            "range cannot be negative."
        )
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(
                schedule=[0, 1],
                dt_init=0.1,
                iter_max=5,
                iter_optimal_range=iter_optimal_range,
            )
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init, iter_relax_factors",
        [
            ([0, 1], 0.1, (1.0, 1.3)),
            ([0, 1], 0.1, (1.05, 1.3)),
        ],
    )
    def test_under_relaxation_factor_less_than_one(
        self, schedule, dt_init, iter_relax_factors
    ):
        """An error should be raised if under-relaxation factor >= 1"""
        msg = "Expected under-relaxation factor < 1."
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(
                schedule,
                dt_init,
                iter_relax_factors=iter_relax_factors,
            )
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init, iter_relax_factors",
        [
            ([0, 1], 0.1, (0.7, 1.0)),
            ([0, 1], 0.1, (0.7, 0.95)),
        ],
    )
    def test_over_relaxation_factor_greater_than_one(
        self, schedule, dt_init, iter_relax_factors
    ):
        """An error should be raised if over-relaxation factor <= 1"""
        msg = "Expected over-relaxation factor > 1."
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(
                schedule,
                dt_init,
                iter_relax_factors=iter_relax_factors,
            )
        assert msg in str(excinfo.value)

    def test_dt_min_times_over_relax_factor_less_than_dt_max(self):
        """An error should be raised if dt_min * over_relax_factor > dt_max."""
        msg_dtmin_over = "Encountered dt_min * over_relax_factor > dt_max. "
        msg_osc = (
            "The algorithm will behave erratically for such a combination of "
            "parameters. See documentation of `dt_min_max` or `iter_relax_factors`."
        )
        msg = msg_dtmin_over + msg_osc
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(
                schedule=[0, 1],
                dt_init=0.1,
                iter_relax_factors=(0.9, 101),
            )
        assert msg in str(excinfo.value)

    def test_dt_max_times_under_relax_factor_greater_than_dt_min(self):
        """An error should be raised if dt_max * under_relax_factor < dt_min."""
        msg_dtmax_under = "Encountered dt_max * under_relax_factor < dt_min. "
        msg_osc = (
            "The algorithm will behave erratically for such a combination of "
            "parameters. See documentation of `dt_min_max` or `iter_relax_factors`."
        )
        msg = msg_dtmax_under + msg_osc
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(
                schedule=[0, 1],
                dt_init=0.1,
                iter_relax_factors=(0.009, 1.3),
            )
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init, recomp_factor",
        [
            ([0, 1], 0.1, 1),
            ([0, 1], 0.1, 1.05),
        ],
    )
    def test_recomputation_factor_less_than_one(self, schedule, dt_init, recomp_factor):
        """An error should be raised if the recomputation factor is greater or equal to one."""
        msg = "Expected recomputation factor < 1."
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(
                schedule=schedule,
                dt_init=dt_init,
                recomp_factor=recomp_factor,
            )
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init, recomp_max",
        [
            ([0, 1], 0.1, -1),
            ([0, 1], 0.1, 0),
        ],
    )
    def test_number_of_recomp_attempts_greater_than_zero(
        self, schedule, dt_init, recomp_max
    ):
        """An error should be raised if the number of recomputation attempts is zero or
        negative."""
        msg = "Number of recomputation attempts must be > 0."
        with pytest.raises(ValueError) as excinfo:
            pp.TimeManager(
                schedule=schedule,
                dt_init=dt_init,
                recomp_max=recomp_max,
            )
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "num_time_steps, final_time",
        [
            (3, 1.0e-14),
            (7, 1.0e-14),
            (3, 1.0),
            (10, 1.0),
        ],
    )
    def test_number_time_steps(self, num_time_steps, final_time):
        """An error should be realised if the number of performed time steps should differ
        from the effective definition via the time step size. Implicitly, this test also
        tests whether the time manager correctly detects final times."""
        time_manager = pp.TimeManager(
            schedule=[0.0, final_time],
            dt_init=final_time / num_time_steps,
            constant_dt=True,
        )

        model_params = {
            "time_manager": time_manager,
            "times_to_export": [],
        }

        model = SinglePhaseFlow(model_params)
        pp.run_time_dependent_model(model)
        performed_time_steps = model.time_manager.time_index

        assert performed_time_steps == num_time_steps


class TestTimeControl:
    """The following tests are written to check the overall behavior of the
    time-stepping algorithm"""

    @pytest.mark.parametrize("recompute_solution", [False, True])
    @pytest.mark.parametrize(
        "time",
        [
            1,  # We reach the final time
            2,  # We are above the final time
        ],
    )
    def test_final_simulation_time(self, recompute_solution: bool, time: int):
        """Test if final simulation time returns None if we do not ask to recompute the
        solution"""
        # Assume we reach the final time
        time_manager = pp.TimeManager(schedule=[0, 1], dt_init=0.1)
        time_manager.time = time
        dt = time_manager.compute_time_step(
            iterations=1000, recompute_solution=recompute_solution
        )
        if recompute_solution:
            assert dt is not None
        else:
            assert dt is None

    @pytest.mark.parametrize(
        "schedule, dt_init, time, time_index, iters, recomp_sol",
        [
            ([0, 1], 0.1, 0.3, 3, 1000, True),
            ([0, 10], 1, 2, 0, None, False),
            ([0, pp.HOUR, 3 * pp.HOUR], 0.5 * pp.HOUR, 0, 0, None, False),
            ([0, pp.HOUR, 3 * pp.HOUR], 0.5 * pp.HOUR, 1.5 * pp.HOUR, 678, 342, True),
        ],
    )
    def test_constant_time_step(
        self, schedule, dt_init, time, time_index, iters, recomp_sol
    ):
        """Test if a constant dt is returned, independent of any configuration or input."""
        time_manager = pp.TimeManager(
            schedule=schedule, dt_init=dt_init, constant_dt=True
        )
        time_manager.time = time
        time_manager.time_index = time_index

        dt = time_manager.compute_time_step(
            iterations=iters, recompute_solution=recomp_sol
        )
        assert time_manager.dt == dt
        assert time_manager.time == time
        assert time_manager.time_index == time_index

    def test_raise_warning_iteration_not_none_for_constant_time_step(self):
        """A warning should be raised if iterations is given and time step is constant"""
        time_manager = pp.TimeManager([0, 1], 0.1, iter_max=10, constant_dt=True)
        iterations = 1
        msg = f"iterations '{iterations}' has no effect if time step is constant."
        with pytest.warns() as record:
            time_manager.compute_time_step(iterations=iterations)
        assert str(record[0].message) == msg

    def test_raise_warning_recompute_solution_true_for_constant_time_step(self):
        """A warning should be raised if recompute_solution is True and time step is
        constant"""
        time_manager = pp.TimeManager([0, 1], 0.1, iter_max=10, constant_dt=True)
        msg = "recompute_solution=True has no effect if time step is constant."
        with pytest.warns() as record:
            time_manager.compute_time_step(recompute_solution=True)
        assert str(record[0].message) == msg

    def test_non_recomputed_solution_conditions(self):
        """Test behaviour of the algorithm when the solution should NOT be recomputed"""
        # Check if internal flag _recomp_sol remains unchanged when recompute_solution=False
        # regardless of the number of iterations provided by the user
        time_manager = pp.TimeManager([0, 1], 0.1)
        time_manager.compute_time_step(iterations=5)
        assert not time_manager._recomp_sol
        time_manager.compute_time_step(iterations=1000)
        assert not time_manager._recomp_sol
        # Check if _recomp_num resets to zero when solution is NOT recomputed
        time_manager = pp.TimeManager([0, 1], 0.1)
        time_manager._recomp_num = 3  # manually change recomputation attempts
        time_manager.compute_time_step(iterations=5)
        assert time_manager._recomp_num == 0
        # Assume recompute_solution=True, but we reach or exceeded maximum number of attempts
        time_manager = pp.TimeManager([0, 1], 0.1, recomp_max=5)
        time_manager._recomp_num = 5
        with pytest.raises(ValueError) as excinfo:
            msg = (
                f"Solution did not converge after {time_manager.recomp_max} recomputing "
                "attempts."
            )
            time_manager.compute_time_step(iterations=5, recompute_solution=True)
        assert time_manager._recomp_sol and (msg in str(excinfo.value))

    def test_recompute_solution_false_by_default(self):
        """ "Checks if recompute solution is False by default"""
        time_manager = pp.TimeManager([0, 1], 0.1)
        time_manager.compute_time_step(iterations=3)
        assert not time_manager._recomp_sol

    def test_recomputed_solutions(self):
        """Test behaviour of the algorithm when the solution should be recomputed. Note
        that this should be independent of the number of iterations that the user passes
        """
        time_manager = pp.TimeManager([0, 100], 2, recomp_factor=0.5)
        time_manager.time = 5
        time_manager.time_index = 13
        time_manager.dt = 1
        time_manager._recomp_num = 6
        time_manager.compute_time_step(iterations=1000, recompute_solution=True)
        # We expect the following actions to occur:
        #     time to be reduced by old dt (time = 5 - 1 = 4)
        #     time index to be reduced by one (time_index = 13 - 1 = 12)
        #     new dt to be half of the old one (dt = 1 * 0.5 = 0.5)
        #     recomputation flag set to True (_recomp_flag = True)
        #     recomputation counter to increase by 1 (_recomp_num = 6 + 1 = 7)
        assert time_manager.time == 4.0
        assert time_manager.time_index == 12
        assert time_manager.dt == 0.5
        assert time_manager._recomp_sol
        assert time_manager._recomp_num == 7

    def test_recomputed_solution_with_calculated_dt_less_than_dt_min(self):
        """Test when a solution is recomputed and the calculated time step is less than
        the minimum allowable time step, the time step is indeed the minimum time step
        """
        time_manager = pp.TimeManager([0, 100], 0.15, recomp_factor=0.5)
        # Emulate the scenario where the solution must be recomputed
        time_manager.time = 5
        time_manager.compute_time_step(iterations=1000, recompute_solution=True)
        # First the algorithm will reduce dt by half (so dt=0.5), but this is less than
        # dt_min. Hence, dt_min should be set.
        assert time_manager.dt == time_manager.dt_min_max[0]

    def test_warning_when_iterations_is_given_and_recomputation_is_true(self):
        """A warning should be raised when iterations is not None and the recomputation flag
        is True"""
        time_manager = pp.TimeManager([0, 1], 0.1, iter_max=10)
        msg = "Number of iterations has no effect in recomputation."
        with pytest.warns() as record:
            time_manager.compute_time_step(iterations=1, recompute_solution=True)
        assert str(record[0].message) == msg

    def test_raise_error_when_adapting_based_on_recomputation_with_dt_equal_to_dt_min(
        self,
    ):
        """An error should be raised when adaption based on recomputation is attempted with
        dt = dt_min"""
        time_manager = pp.TimeManager(schedule=[0, 100], dt_init=1, dt_min_max=(1, 10))

        # For these parameters, we have time_manager.dt = time_manager.dt_init =
        # time_manager.dt_min_max[0] Attempting a recomputation should raise an error
        with pytest.raises(ValueError) as excinfo:
            msg = (
                "Recomputation will not have any effect since the time step "
                f"achieved its minimum admissible value -> dt = dt_min = {time_manager.dt}."
            )
            time_manager.compute_time_step(iterations=5, recompute_solution=True)
        assert time_manager._recomp_sol and (msg in str(excinfo.value))

    def test_raise_error_when_adapting_based_on_iterations_with_iterations_none(self):
        """An error should be raised if adaptation based on iteration is intended but
        iterations is None.
        """
        time_manager = pp.TimeManager(schedule=[0, 100], dt_init=1)
        with pytest.raises(ValueError) as excinfo:
            msg = "Time step cannot be adapted without 'iterations'."
            time_manager.compute_time_step()
        assert not time_manager._recomp_sol and (msg in str(excinfo.value))

    @pytest.mark.parametrize("iterations", [11, 100])
    def test_warning_iteration_is_greater_than_max_iter(self, iterations):
        """Test if a warning is raised when the number of iterations passed to the
        method is greater than max_iter and the solution is not recomputed"""
        time_manager = pp.TimeManager([0, 1], 0.1, iter_max=10)
        warn_msg = (
            f"The given number of iterations '{iterations}' is larger than the maximum "
            f"number of iterations '{time_manager.iter_max}'. This usually means that the "
            f"solver did not converge, but since recompute_solution = False was given, the "
            f"algorithm will adapt the time step anyways."
        )
        with pytest.warns() as record:
            time_manager.compute_time_step(
                iterations=iterations, recompute_solution=False
            )
        assert str(record[0].message) == warn_msg

    @pytest.mark.parametrize("iterations", [1, 3, 5])
    def test_decreasing_time_step(self, iterations):
        """Test if the time step decreases after the number of iterations is less or
        equal than the lower endpoint of the optimal iteration range by its
        corresponding factor.
        """
        time_manager = pp.TimeManager(
            [0, 100],
            2,
            iter_optimal_range=(5, 9),
            iter_relax_factors=(0.7, 1.3),
        )
        time_manager.dt = 1
        time_manager.compute_time_step(iterations=iterations, recompute_solution=False)
        assert time_manager.dt == 1.3

    @pytest.mark.parametrize("iterations", [9, 11, 13])
    def test_increasing_time_step(self, iterations):
        """Test if the time step is restricted after the number of iterations is greater
        or equal than the upper endpoint of the optimal iteration range by its
        corresponding factor."""
        time_manager = pp.TimeManager(
            [0, 100],
            2,
            iter_optimal_range=(5, 9),
            iter_relax_factors=(0.7, 1.3),
        )
        time_manager.dt = 1
        time_manager.compute_time_step(iterations=iterations, recompute_solution=False)
        assert time_manager.dt == 0.7

    @pytest.mark.parametrize("iterations", [6, 7, 8])
    def test_time_step_within_optimal_iteration_range(self, iterations):
        """Test if the time step remains unchanged when the number of iterations lies
        between the optimal iteration range"""
        time_manager = pp.TimeManager(
            [0, 100],
            2,
            iter_optimal_range=(5, 9),
            iter_relax_factors=(0.7, 1.3),
        )
        time_manager.dt = 1
        time_manager.compute_time_step(iterations=iterations)
        assert time_manager.dt == 1

    @pytest.mark.parametrize("dt", [0.13, 0.1, 0.075])
    def test_time_step_less_than_dt_min(self, dt):
        """Test if the algorithm passes dt_min when the calculated dt is less than dt_min"""
        time_manager = pp.TimeManager([0, 100], 2, iter_optimal_range=(4, 7))
        time_manager.dt = dt
        time_manager.compute_time_step(iterations=7)
        assert time_manager.dt == time_manager.dt_min_max[0]

    @pytest.mark.parametrize("dt", [9, 10, 15])
    def test_time_step_greater_than_dt_max(self, dt):
        """Test if the algorithm passes dt_max when the calculated dt is greater than dt_max"""
        time_manager = pp.TimeManager([0, 100], 2, iter_optimal_range=(4, 7))
        time_manager.dt = dt
        time_manager.compute_time_step(iterations=4)
        assert time_manager.dt == time_manager.dt_min_max[1]

    @pytest.mark.parametrize(
        "schedule, dt_init",
        [
            ([0, 1], 0.1),
            ([0, 10, 20, 30], 1),
            ([10, 11, 15, 16, 19, 20], 1),
            (
                [0, 0.01, 1 * pp.HOUR, 2 * pp.HOUR, 100 * pp.HOUR, 101 * pp.HOUR],
                2 * pp.HOUR,
            ),
        ],
    )
    def test_hitting_schedule_times(self, schedule, dt_init):
        """Test if algorithm respects the passed target times from the schedule"""
        time_manager = pp.TimeManager(schedule, dt_init)
        for time in schedule[1:]:
            time_manager.time = 0.99 * time
            time_manager.dt = time_manager.dt_min_max[1]
            time_manager.compute_time_step(iterations=4)
            assert time == time_manager.time + time_manager.dt

    @pytest.mark.parametrize(
        "schedule, dt_init, is_constant, time, dt",
        [
            ([0, 1], 0.1, False, 0.3, 0.2),
            ([0, 1], 0.1, True, 0.3, 0.2),
        ],
    )
    def test_update_time(self, schedule, dt_init, is_constant, time, dt):
        """Checks if time is correctly updated"""
        time_manager = pp.TimeManager(
            schedule=schedule, dt_init=dt_init, constant_dt=is_constant
        )
        time_manager.time = time
        time_manager.dt = dt
        time_manager.increase_time()
        assert time_manager.time == time + dt

    @pytest.mark.parametrize(
        "schedule, dt_init, is_constant, time_index",
        [
            ([0, 1], 0.1, False, 13),
            ([0, 1], 0.1, True, 13),
        ],
    )
    def test_update_time_index(self, schedule, dt_init, is_constant, time_index):
        """Checks if time index is correctly updated"""
        time_manager = pp.TimeManager(
            schedule=schedule, dt_init=dt_init, constant_dt=is_constant
        )
        time_manager.time_index = time_index
        time_manager.increase_time_index()
        assert time_manager.time_index == (time_index + 1)

    @pytest.mark.parametrize("constant_dt", [True, False])
    def test_time_step_match_schedule_exactly(self, constant_dt: bool):
        """Checks the edge case when the dynamic time stepping is on, but the next time
        step matches the schedule exactly.

        See: https://github.com/pmgbergen/porepy/issues/1152

        """
        time_manager = pp.TimeManager(
            dt_init=1, dt_min_max=(0.1, 1), schedule=[0, 1, 2], constant_dt=constant_dt
        )

        while not time_manager.final_time_reached():
            time_manager.increase_time()
            time_manager.increase_time_index()
            time_manager.compute_time_step(iterations=5)
            assert not np.allclose(time_manager.dt, 0)

    def test_io_time_information(self):
        """Check I/O functionality. Checks if writing and loading of time information
        in the form of the evolution of time and dt is performed correctly. Also test
        whether a single time and dt is picked correctly from loaded histories and the
        latter are cut-off accordingly.

        """
        pth = Path("times.json")
        # Imitate a time manager iterating through some consecutive time steps.
        time_manager = pp.TimeManager(schedule=[0, 1], dt_init=0.1, constant_dt=True)
        for i in range(10):
            time_manager.write_time_information(pth)
            time_manager.increase_time_index()
            time_manager.increase_time()

        # Check if the history is generated correctly.
        assert np.all(np.isclose(time_manager.exported_times, np.linspace(0, 0.9, 10)))
        assert np.all(np.isclose(time_manager.exported_dt, 10 * [0.1]))

        # Check if entirely fetched history of time and dt is loaded correctly.
        new_time_manager = pp.TimeManager(
            schedule=[0, 1], dt_init=0.1, constant_dt=True
        )
        new_time_manager.load_time_information(pth)
        assert np.all(
            np.isclose(new_time_manager.exported_times, np.linspace(0, 0.9, 10))
        )
        assert np.all(np.isclose(new_time_manager.exported_dt, 10 * [0.1]))

        # Check if single-chosen time and dt are picked correctly.
        new_time_manager.set_time_and_dt_from_exported_steps(5)
        assert np.isclose(new_time_manager.time, 0.5)
        assert np.isclose(new_time_manager.dt, 0.1)

        # Check if history has been cut-off correctly.
        assert np.all(
            np.isclose(
                np.array(new_time_manager.exported_times),
                np.array([0, 0.1, 0.2, 0.3, 0.4]),
            )
        )
        assert np.all(
            np.isclose(
                np.array(new_time_manager.exported_dt),
                np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
            )
        )
        # Remove temporary file.
        pth.unlink()


class DynamicTimeStepTestCaseModel(SinglePhaseFlow):
    """A mockup model that overrides `check_convergence` and predefines convergence
    behavior after each nonlinear iteration.

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
            self.ad_time_step.value(self.equation_system) == self.time_manager.dt
        ), "The AD time step value conflicts with the value from the time_manager."

        # The initial guess for the unknown time step values should be equal to the
        # known time step values. See https://github.com/pmgbergen/porepy/issues/1205.
        if self.num_nonlinear_iters == 0:
            iterate_values = self.equation_system.get_variable_values(iterate_index=0)
            state_values = self.equation_system.get_variable_values(time_step_index=0)
            assert np.all(
                iterate_values == state_values
            ), "Likely, 'iterate' was not reset after the unsuccessful time step."

        self.num_nonlinear_iters += 1

    def _is_nonlinear_problem(self):
        return True

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        if self.num_nonlinear_iters < self.num_nonlinear_iterations[self.time_step_idx]:
            # Neither converged nor diverged
            return False, False
        if self.time_step_converged[self.time_step_idx] is True:
            # Converged
            return True, False
        if self.time_step_converged[self.time_step_idx] is False:
            # Diverged
            return False, True
        assert (
            False
        ), "Nonlinear solver did not stop iterating after the iteration limit."

    # Minimizing computational expences.

    def assemble_linear_system(self) -> None:
        pass

    def solve_linear_system(self) -> np.ndarray:
        return np.ones(self.equation_system.num_dofs())


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
            "should_fail": True,
            "num_nonlinear_iterations": [2, 3],
            "time_step_converged": [True, False],
            "exported_dt_expected": [1, 1],
        },
        # Case 3: An unsuccessful simulation with dynamic time stepping. Reached the
        # minimal time step and failed.
        {
            "should_fail": True,
            "num_nonlinear_iterations": [1, 1, 1],
            "time_step_converged": [False, False, False],
            "exported_dt_expected": [1, 0.3, 0.1],
        },
        # Case 4: The time step fails right before the schedule point. Expected to
        # decrease dt and meet the schedule regardless.
        {
            "num_nonlinear_iterations": [1, 1, 1, 1, 1],
            "time_step_converged": [True, False, True, True, True],
            "exported_dt_expected": [1, 0.35, 0.105, 0.21, 0.035],
        },
    ],
)
def test_model_time_step_control(params: dict):
    """The integration test of the `TimeManager` class into PorePy models."""
    constant_dt = params.get("constant_dt", False)
    should_fail = params.get("should_fail", False)
    num_nonlinear_iterations = params["num_nonlinear_iterations"]
    time_step_converged = params["time_step_converged"]
    exported_dt_expected = params["exported_dt_expected"]

    schedule_end = 2 if constant_dt else 1.35
    time_manager = pp.TimeManager(
        schedule=(0, schedule_end),
        dt_init=1,
        constant_dt=constant_dt,
        dt_min_max=(0.1, 5),
        iter_relax_factors=(0.4, 2),
        iter_optimal_range=(4, 7),
        recomp_factor=0.3,
    )
    model = DynamicTimeStepTestCaseModel(
        num_nonlinear_iterations=num_nonlinear_iterations,
        time_step_converged=time_step_converged,
        params={
            "time_manager": time_manager,
            "times_to_export": [],  # Suspends export
        },
    )

    if should_fail:
        with pytest.raises(ValueError):
            pp.run_time_dependent_model(model, {"max_iterations": MAX_NONLINEAR_ITER})
    else:
        pp.run_time_dependent_model(model, {"max_iterations": MAX_NONLINEAR_ITER})

    assert np.allclose(model.time_step_history, exported_dt_expected)
