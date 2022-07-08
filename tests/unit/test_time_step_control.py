import pytest

import porepy as pp
from porepy.numerics.time_step_control import TimeSteppingControl as Ts


class TestParameterInputs:
    """The following tests are written to check the sanity of the input parameters"""

    def test_default_parameters_and_attribute_initialization(self):
        """Test the default parameters and initialization of attributes."""
        tsc = Ts(schedule=[0, 1], dt_init=0.2, dt_min_max=(0.1, 0.5))
        assert tsc.schedule == [0, 1]
        assert tsc.time_init == 0
        assert tsc.time_final == 1
        assert tsc.dt_init == 0.2
        assert tsc.dt_min == 0.1
        assert tsc.dt_max == 0.5
        assert tsc.iter_max == 15
        assert tsc.iter_low == 4
        assert tsc.iter_upp == 7
        assert tsc.iter_low_factor == 1.3
        assert tsc.iter_upp_factor == 0.7
        assert tsc.recomp_factor == 0.5
        assert tsc.recomp_max == 10
        assert tsc.time == 0
        assert tsc.dt == 0.2

    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max",
        [
            ([], 0.1, (0.1, 1)),
            ([1], 0.1, (0.1, 1)),
        ],
    )
    def test_schedule_length_greater_than_2(self, schedule, dt_init, dt_min_max):
        """An error should be raised if len(schedule) < 2."""
        msg = "Expected schedule with at least two items."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=schedule, dt_init=dt_init, dt_min_max=dt_min_max)
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max",
        [
            ([-1, 10], 0.1, (0.1, 1)),
            ([0, -3], 0.1, (0.1, 1)),
            ([1, 2, -100, 3, 4], 0.1, (0.1, 1)),
        ],
    )
    def test_positive_time_in_schedule(self, schedule, dt_init, dt_min_max):
        """An error should be raised if a negative time is encountered in the schedule."""
        msg = "Encountered at least one negative time in schedule."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=schedule, dt_init=dt_init, dt_min_max=dt_min_max)
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max",
        [
            ([0, 1, 2, 5, 1], 0.1, (0.1, 1)),
            ([0, 1, 1, 2], 0.1, (0.1, 1)),
            ([100, 200, 50], 0.1, (0.1, 1)),
        ],
    )
    def test_strictly_increasing_time_in_schedule(self, schedule, dt_init, dt_min_max):
        """An error should be raised if times in schedule are not strictly increasing."""
        msg = "Schedule must contain strictly increasing times."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=schedule, dt_init=dt_init, dt_min_max=dt_min_max)
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max",
        [
            ([0, 1], -1, (0.1, 1)),
            ([0, 1], 0, (0.1, 1)),
        ],
    )
    def test_positive_initial_time_step(self, schedule, dt_init, dt_min_max):
        """An error should be raised if initial time step is non-positive."""
        msg = "Initial time step must be positive."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=schedule, dt_init=dt_init, dt_min_max=dt_min_max)
        assert msg in str(excinfo.value)

    def test_initial_time_step_smaller_than_final_time(self):
        """An error should be raised if initial time step is larger than final time."""
        msg = "Initial time step cannot be larger than final simulation time."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=[0, 1], dt_init=1.0001, dt_min_max=(0.1, 0.5))
        assert msg in str(excinfo.value)

    def test_initial_time_step_larger_than_minimum_time_step(self):
        """An error should be raised if initial time step is less than minimum time step."""
        msg = "Initial time step cannot be smaller than minimum time step."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=[0, 1], dt_init=0.09, dt_min_max=(0.1, 0.5))
        assert msg in str(excinfo.value)

    def test_initial_time_step_smaller_than_maximum_time_step(self):
        """An error should be raised if initial time step is larger than the maximum time
        step."""
        msg = "Initial time step cannot be larger than maximum time step."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=[0, 1], dt_init=0.51, dt_min_max=(0.1, 0.5))
        assert msg in str(excinfo.value)

    def test_max_number_of_iterations_non_negative(self):
        """An error should be raised if the maximum number of iterations is negative."""
        msg = "Maximum number of iterations must be non-negative."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=[0, 1], dt_init=0.1, dt_min_max=(0.1, 0.5), iter_max=-1)
        assert msg in str(excinfo.value)

    def test_lower_iter_smaller_than_upper_iter(self):
        """An error should be raised if the lower optimal iteration range is larger than
        the upper optimal iteration range."""
        msg = "Lower optimal iteration range cannot be larger than"
        msg += " upper optimal iteration range."
        with pytest.raises(ValueError) as excinfo:
            Ts([0, 1], 0.1, (0.1, 0.5), iter_max=5, iter_optimal_range=(3, 2))
        assert msg in str(excinfo.value)

    def test_upper_iter_less_or_equal_than_max_iter(self):
        """An error should be raised if the upper optimal iteration range is larger than
        the maximum number of iterations."""
        msg = "Upper optimal iteration range cannot be larger than"
        msg += " maximum number of iterations."
        with pytest.raises(ValueError) as excinfo:
            Ts([0, 1], 0.1, (0.1, 0.5), iter_max=5, iter_optimal_range=(2, 6))
        assert msg in str(excinfo.value)

    def test_lower_iter_greater_or_equal_than_zero(self):
        """An error should be raised if the lower iteration range is less than zero."""
        msg = "Lower optimal iteration range cannot be negative."
        with pytest.raises(ValueError) as excinfo:
            Ts([0, 1], 0.1, (0.1, 0.5), iter_max=5, iter_optimal_range=(-1, 2))
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max, iter_lowupp_factor",
        [
            ([0, 1], 0.1, (0.1, 0.5), (1, 0.7)),
            ([0, 1], 0.1, (0.1, 0.5), (0.95, 0.7)),
        ],
    )
    def test_lower_factor_greater_than_one(
        self, schedule, dt_init, dt_min_max, iter_lowupp_factor
    ):
        """An error should be raised if the lower multiplication factor is less or equal than
        one."""
        msg = "Expected lower multiplication factor > 1."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule, dt_init, dt_min_max, iter_lowupp_factor=iter_lowupp_factor)
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max, iter_lowupp_factor",
        [
            ([0, 1], 0.1, (0.1, 0.5), (1.3, 1)),
            ([0, 1], 0.1, (0.1, 0.5), (1.3, 1.05)),
        ],
    )
    def test_upper_factor_less_than_one(
        self, schedule, dt_init, dt_min_max, iter_lowupp_factor
    ):
        """An error should be raised if the upper multiplication factor is greater or equal
        than one."""
        msg = "Expected upper multiplication factor < 1."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule, dt_init, dt_min_max, iter_lowupp_factor=iter_lowupp_factor)
        assert msg in str(excinfo.value)

    def test_dt_min_times_low_iter_factor_less_than_dt_max(self):
        """An error should be raised if dt_min * iter_low_factor > dt_max."""
        msg = "Encountered dt_min * iter_low_factor > dt_max."
        with pytest.raises(ValueError) as excinfo:
            Ts(
                schedule=[0, 1],
                dt_init=0.1,
                dt_min_max=(0.1, 0.5),
                iter_lowupp_factor=(6, 0.9),
            )
        assert msg in str(excinfo.value)

    def test_dt_max_times_upp_iter_factor_greater_than_dt_min(self):
        """An error should be raised if dt_max * iter_upp_factor < dt_min."""
        msg = "Encountered dt_max * iter_upp_factor < dt_min."
        with pytest.raises(ValueError) as excinfo:
            Ts(
                schedule=[0, 1],
                dt_init=0.1,
                dt_min_max=(0.1, 0.5),
                iter_lowupp_factor=(1.3, 0.01),
            )
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max, recomp_factor",
        [
            ([0, 1], 0.1, (0.1, 0.5), 1),
            ([0, 1], 0.1, (0.1, 0.5), 1.05),
        ],
    )
    def test_recomputation_factor_less_than_one(
        self, schedule, dt_init, dt_min_max, recomp_factor
    ):
        """An error should be raised if the recomputation factor greater or equal to one."""
        msg = "Expected recomputation multiplication factor < 1."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule, dt_init, dt_min_max, recomp_factor=recomp_factor)
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max, recomp_max",
        [
            ([0, 1], 0.1, (0.1, 0.5), -1),
            ([0, 1], 0.1, (0.1, 0.5), 0),
        ],
    )
    def test_number_of_recomp_attempts_greater_than_zero(
        self, schedule, dt_init, dt_min_max, recomp_max
    ):
        """An error should be raised if the number of recomputation attempts is zero or
        negative."""
        msg = "Number of recomputation attempts must be > 0."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule, dt_init, dt_min_max, recomp_max=recomp_max)
        assert msg in str(excinfo.value)


class TestTimeControl:
    """The following tests are written to check the overall behaviour of the time-stepping
    algorithm"""

    def test_final_simulation_time(self):
        """Test if final simulation time returns None, irrespectively of parameters
        passed in next_time_step()."""
        # Assume we reach the final time
        tsc = Ts([0, 1], 0.1, (0.1, 0.5))
        tsc.time = 1
        dt = tsc.next_time_step(iterations=1000, recompute_solution=True)
        assert dt is None
        # Now, assume we are above the final time
        tsc = Ts([0, 1], 0.1, (0.1, 0.5))
        tsc.time = 2
        dt = tsc.next_time_step(iterations=0, recompute_solution=False)
        assert dt is None

    def test_non_recomputed_solution_conditions(self):
        """Test behaviour of the algorithm when the solution should NOT be recomputed"""
        # Check if internal flag _recomp_sol remains unchanged when recompute_solution=False
        # regardless of the number of iterations provided by the user
        tsc = Ts([0, 1], 0.1, (0.1, 0.5))
        tsc.next_time_step(recompute_solution=False, iterations=5)
        assert not tsc._recomp_sol
        tsc.next_time_step(recompute_solution=False, iterations=1000)
        assert not tsc._recomp_sol
        # Check if _recomp_num resets to zero when solution is NOT recomputed
        tsc = Ts([0, 1], 0.1, (0.1, 0.5))
        tsc._recomp_num = 3  # manually change recomputation attempts
        tsc.next_time_step(recompute_solution=False, iterations=5)
        assert tsc._recomp_num == 0
        # Assume recompute_solution=True, but we reach or exceeded maximum number of attempts
        tsc = Ts([0, 1], 0.1, (0.1, 0.5), recomp_max=5)
        tsc._recomp_num = 5
        with pytest.raises(ValueError) as excinfo:
            msg = f"Solution did not converge after {tsc.recomp_max} recomputing attempts."
            tsc.next_time_step(recompute_solution=True, iterations=5)
        assert tsc._recomp_sol and (msg in str(excinfo.value))

    def test_recomputed_solutions(self):
        """Test behaviour of the algorithm when the solution should be recomputed. Note
        that this should be independent of the number of iterations that the user passes"""
        tsc = Ts([0, 100], 2, (0.1, 10), recomp_factor=0.5)
        tsc.time = 5
        tsc.dt = 1
        tsc.next_time_step(recompute_solution=True, iterations=1000)
        # We expect the time step to be reduced half, time to be corrected (decreased
        # accordingly), _recomp_sol == True, and the counter _recomp_num increased by 1.
        assert tsc.time == 4
        assert tsc.dt == 0.5
        assert tsc._recomp_sol
        assert tsc._recomp_num == 1

    def test_recomputed_solution_with_calculated_dt_less_than_dt_min(self):
        """Test when a solution is recomputed and the calculated time step is less than
        the minimum allowable time step, the time step is indeed the minimum time step"""
        tsc = Ts([0, 100], 2, (0.6, 10), recomp_factor=0.5)
        # Emulate the scenario where the solution must be recomputed b
        tsc.time = 5
        tsc.dt = 1
        tsc.next_time_step(recompute_solution=True, iterations=1000)
        # First the algorithm will reduce dt by half (so dt=0.5), but this is less than
        # dt_min. Hence, dt_min should be set.
        assert tsc.dt == tsc.dt_min

    @pytest.mark.parametrize("iterations", [1, 3, 5])
    def test_relaxing_time_step(self, iterations):
        """Test if the time step is relaxed after the number of iterations is less or equal
        than the lower optimal iteration range, by its corresponding factor"""
        tsc = Ts(
            [0, 100],
            2,
            (0.1, 10),
            iter_optimal_range=(5, 9),
            iter_lowupp_factor=(1.3, 0.7),
        )
        tsc.dt = 1
        tsc.next_time_step(recompute_solution=False, iterations=iterations)
        assert tsc.dt == 1.3

    @pytest.mark.parametrize("iterations", [9, 11, 13])
    def test_restricting_time_step(self, iterations):
        """Test if the time step is restricted after the number of iterations is greater or
        equal than the upper optimal iteration range, by its corresponding factor"""
        tsc = Ts(
            [0, 100],
            2,
            (0.1, 10),
            iter_optimal_range=(5, 9),
            iter_lowupp_factor=(1.3, 0.7),
        )
        tsc.dt = 1
        tsc.next_time_step(recompute_solution=False, iterations=iterations)
        assert tsc.dt == 0.7

    @pytest.mark.parametrize("iterations", [6, 7, 8])
    def test_time_step_within_optimal_iteration_range(self, iterations):
        """Test if the time step remains unchanged when the number of iterations lies
        between the optimal iteration range"""
        tsc = Ts(
            [0, 100],
            2,
            (0.1, 10),
            iter_optimal_range=(5, 9),
            iter_lowupp_factor=(1.3, 0.7),
        )
        tsc.dt = 1
        tsc.next_time_step(recompute_solution=False, iterations=iterations)
        assert tsc.dt == 1

    @pytest.mark.parametrize("dt", [0.13, 0.1, 0.075])
    def test_time_step_less_than_dt_min(self, dt):
        """Test if the algorithm passes dt_min when the calculated dt is less than dt_min"""
        tsc = Ts([0, 100], 2, (0.1, 10), iter_optimal_range=(4, 7))
        tsc.dt = dt
        tsc.next_time_step(recompute_solution=False, iterations=7)
        assert tsc.dt == tsc.dt_min

    @pytest.mark.parametrize("dt", [9, 10, 15])
    def test_time_step_greater_than_dt_max(self, dt):
        """Test if the algorithm passes dt_max when the calculated dt is greater than dt_max"""
        tsc = Ts([0, 100], 2, (0.1, 10), iter_optimal_range=(4, 7))
        tsc.dt = dt
        tsc.next_time_step(recompute_solution=False, iterations=4)
        assert tsc.dt == tsc.dt_max

    @pytest.mark.parametrize(
        "schedule",
        [
            [0, 1],
            [0, 10, 20, 30],
            [10, 11, 15, 16, 19, 20],
            [0, 0.01, 1 * pp.HOUR, 2 * pp.HOUR, 100 * pp.HOUR, 101 * pp.HOUR],
        ],
    )
    def test_hitting_schedule_times(self, schedule):
        """Test if algorithm respects the passed target times from the schedule"""
        tsc = Ts(schedule, 0.1, (0.01, 0.1 * schedule[-1]))
        for time in schedule[1:]:
            tsc.time = 0.99 * time
            tsc.dt = tsc.dt_max
            tsc.next_time_step(recompute_solution=False, iterations=4)
            assert time == tsc.time + tsc.dt
