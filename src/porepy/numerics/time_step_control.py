from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["TimeSteppingControl"]


class TimeSteppingControl:
    """Parent class for iteration-based time-stepping control routine.

    Parameters:
        schedule: Array-like object containing the target times for the simulation.
            Unless a constant time step is prescribed, the time-stepping algorithm will adapt
            the time step so that the scheduled times are guaranteed to be hit.

            The `schedule` must contain minimally two elements, corresponding to the
            initial and final simulation times. Schedules of size > 2 must contain strictly
            increasing times. Examples of VALID inputs are: [0, 1], np.array([0, 10, 30,50]),
            and [0, 1*pp.HOUR, 3*pp.HOUR]. Examples of INVALID inputs are: [1], [1,0],
            and np.array([0, 1, 1, 2]).

            If a constant time step is used (`constant_dt = True`), then the time step
            (`dt_init`) is required to be compatible with the scheduled times in `schedule`.
            Otherwise, an error will be raised. Examples of VALID inputs for `constant_dt =
            True` and `dt_init = 2` are: [0, 2] and np.array([0, 4, 6, 10]). Examples of
            INVALID inputs for `constant_dt = True` and `dt_init = 2` are [0, 3] and
            np.array([0, 4, 5, 10]).
        constant_dt: Whether to treat the time step as constant or not.
            If True, then the time-stepping control algorithm is effectively bypassed. The
            algorithm will NOT adapt the time step in any situation, even if the user
            attempts to recompute the solution. Nevertheless, the attributes such as
            scheduled times will still be accesible, provided `dt_init` and `schedule` are
            compatible.
        dt_min_max: Minimum and maximum permissible time steps.
            If None, then the minimum time step is set to 0.1% of the final simulation time
            and the maximum time step is set to 10% of the final simulation time. If given,
            then the first and second elements of the tuple corresponds to the minimum and
            maximum time steps, respectively.

            To avoid oscilations and ensure a stable time step adaptation in combination with
            the relaxation factors, we further require:
                dt_min_max[0] * iter_relax_factors[1] < dt_min_max[1], and
                dt_min_max[1] * iter_relax_factors[0] > dt_min_max[0].
            Note that in practical applications, these conditions are ussually met.
        iter_max: Maximum number of iterations.
        iter_optimal_range: Optimal iteration range.
            The first and second elements of the tuple correspond to the lower and upper
            endpoints of the optimal iteration range.
        iter_relax_factors: Relaxation factors.
            The first and second elements of the tuple corresponds to the under-relaxation and
            over-relaxation factors, respectively. We require the under-relaxation factor
            to be strictly lower than one, whereas the over-relaxation factor is required to
            be strictly greater than one.

            To avoid oscilations and ensure a stable time step adaptation in combination with
            the minimum and maximum allowable time steps, we further require:
                dt_min_max[0] * iter_relax_factors[1] < dt_min_max[1], and
                dt_min_max[1] * iter_relax_factors[0] > dt_min_max[0].
            Note that in practical applications, these conditions are usually met.
        recomp_factor: Failed-to-converge solution recomputation factor.
            Factor by which the time step will be multiplied in case the solution must be
            recomputed. We require `recomp_factor` to be strictly less than one.
        recomp_max: Failed-to-converge maximum recomputation attempts. The maximum allowable
            number of consecutive recomputation attempts.
        print_info. Whether to print on-the-fly time-stepping information or not.

    Example:
        # The following is an example on how to initialize a time-stepping object
        tsc = pp.TimeSteppingControl(
            schedule=[0, 10],
            dt_init=0.5,
            dt_min_max=(0.1, 2),
            iter_max=10,
            iter_optimal_range=(3, 8),
            iter_relax_factors=(0.9, 1.1),
            recomp_factor=0.1,
            recomp_max=5,
            print_info=True
        )
        # To inspect the attributes of the object
        print(tsc)

    Attributes:
        dt (float): Time step.
        dt_init (float): Initial time step.
        dt_min_max (tuple[Union[int, float], Union[int, float]]): Min and max time steps.
        is_constant (bool): Constant time step flag.
        iter_max (int): Maximum number of iterations.
        iter_optimal_range (tuple[int, int]): Optimal iteration range.
        iter_relax_factors (tuple[float, float]): Relaxation factors.
        recomp_factor (float): Recomputation factor. Strictly lower than one.
        recomp_max (int): Maximum number of recomputation attempts.
        schedule (ArrayLike): List of scheduled times including initial and final times.
        time (float): Current time.
        time_final (float): Final simulation time.
        time_init (float): Initial simulation time.

    """

    def __init__(
        self,
        schedule: ArrayLike,
        dt_init: Union[int, float],
        constant_dt: bool = False,
        dt_min_max: Optional[tuple[Union[int, float], Union[int, float]]] = None,
        iter_max: int = 15,
        iter_optimal_range: tuple[int, int] = (4, 7),
        iter_relax_factors: tuple[float, float] = (0.7, 1.3),
        recomp_factor: float = 0.5,
        recomp_max: int = 10,
        print_info: bool = False,
    ) -> None:

        # TODO: Add test to make sure schedule can indeed be an ArrayLike object
        schedule = np.array(schedule)
        # Sanity checks for schedule
        if np.size(schedule) < 2:
            raise ValueError("Expected schedule with at least two elements.")
        elif any(time < 0 for time in schedule):
            raise ValueError("Encountered at least one negative time in schedule.")
        elif not self._is_strictly_increasing(schedule):
            raise ValueError("Schedule must contain strictly increasing times.")

        # Sanity checks for initial time step
        if dt_init <= 0:
            raise ValueError("Initial time step must be positive.")
        elif dt_init > schedule[-1]:
            raise ValueError(
                "Initial time step cannot be larger than final simulation time."
            )

        # If dt_min_max is not given, set dt_min=0.001*final_time and dt_max=0.1*final_time
        dt_min_max_passed = dt_min_max  # store for later use
        if dt_min_max is None:
            dt_min_max = (0.001 * schedule[-1], 0.1 * schedule[-1])

        # More sanity checks below. Note that all the remaining sanity checks (but one) are
        # only needed when constant_dt = False. Thus, to save time when constant_dt = True,
        # we use this rather ugly if-statement from below.

        if not constant_dt:

            # Sanity checks for dt_min and dt_max
            msg_dtmin = "Initial time step cannot be smaller than minimum time step. "
            msg_dtmax = "Initial time step cannot be larger than maximum time step. "
            msg_unset = (
                "This error was raised since `dt_min_max` was not set on "
                "initialization. Thus, values of dt_min and dt_max were assigned "
                "based on the final simulation time. If you still want to use this "
                "initial time step, consider passing `dt_min_max` explictly."
            )

            if dt_init < dt_min_max[0]:
                if dt_min_max_passed is not None:
                    raise ValueError(msg_dtmin)
                else:
                    raise ValueError(msg_dtmin + msg_unset)

            if dt_init > dt_min_max[1]:
                if dt_min_max_passed is not None:
                    raise ValueError(msg_dtmax)
                else:
                    raise ValueError(msg_dtmax + msg_unset)

            # NOTE: The above checks guarantee that minimum time step <= maximum time step

            # Sanity checks for maximum number of iterations.
            # Note that iter_max = 1 is a possibility. This will imply that the solver
            # reaches convergence directly, e.g., as in direct solvers.
            if iter_max <= 0:
                raise ValueError("Maximum number of iterations must be positive.")

            # Sanity checks for optimal iteration range
            if iter_optimal_range[0] > iter_optimal_range[1]:
                msg = (
                    f"Lower endpoint '{iter_optimal_range[0]}' of optimal iteration range "
                    f"cannot be larger than upper endpoint '{iter_optimal_range[1]}'."
                )
                raise ValueError(msg)
            elif iter_optimal_range[1] > iter_max:
                msg = (
                    f"Upper endpoint '{iter_optimal_range[1]}' of optimal iteration range "
                    f"cannot be larger than maximum number of iterations '{iter_max}'."
                )
                raise ValueError(msg)
            elif iter_optimal_range[0] < 0:
                msg = (
                    f"Lower endpoint '{iter_optimal_range[0]}' of optimal iteration range "
                    "cannot be negative."
                )
                raise ValueError(msg)

            # Sanity checks for relaxation factors
            if iter_relax_factors[0] >= 1:
                raise ValueError("Expected under-relaxation factor < 1.")
            elif iter_relax_factors[1] <= 1:
                raise ValueError("Expected over-relaxation factor > 1.")

            # Checks for sensible combinations of iter_optimal_range and iter_relax_factors
            msg_dtmin_over = "Encountered dt_min * over_relax_factor > dt_max. "
            msg_dtmax_under = "Encountered dt_max * under_relax_factor < dt_min. "
            msg_osc = (
                "The algorithm will behave erratically for such a combination of parameters. "
                "See documentation of `dt_min_max` or `iter_relax_factors`."
            )
            if dt_min_max[0] * iter_relax_factors[1] > dt_min_max[1]:
                raise ValueError(msg_dtmin_over + msg_osc)
            elif dt_min_max[1] * iter_relax_factors[0] < dt_min_max[0]:
                raise ValueError(msg_dtmax_under + msg_osc)

            # Sanity check for recomputation factor
            if recomp_factor >= 1:
                raise ValueError("Expected recomputation factor < 1.")

            # Sanity check for maximum number of recomputation attempts
            if recomp_max <= 0:
                raise ValueError("Number of recomputation attempts must be > 0.")

        else:

            # TODO: Add unit test for this sanity check
            # If the time step is constant, check that the scheduled times and the time
            # step are compatible. See documentation of `schedule`.
            sim_times = np.arange(schedule[0], schedule[-1] + dt_init, dt_init)
            intersect = np.intersect1d(sim_times, schedule)
            # If the length of the intersection and scheduled times are unequal, there is a
            # mismatch
            if (len(intersect) - len(schedule)) != 0:
                raise ValueError("Mismatch between the time step and scheduled time.")

        # Schedule, initial, and final times
        self.schedule = schedule
        self.time_init = schedule[0]
        self.time_final = schedule[-1]

        # Initial time step
        self.dt_init = dt_init

        # Minimum and maximum allowable time steps
        self.dt_min_max = dt_min_max

        # Maximum number of iterations
        # TODO: This is really a property of the nonlinear solver. A full integration will
        #  most likely required "pulling" this parameter from the solver side.
        self.iter_max = iter_max

        # Optimal iteration range
        self.iter_optimal_range = iter_optimal_range

        # Relaxation factors
        self.iter_relax_factors = iter_relax_factors

        # Recomputation multiplication factor
        self.recomp_factor = recomp_factor

        # Number of permissible re-computation attempts
        self.recomp_max = recomp_max

        # Constant time step flag
        self.is_constant = constant_dt

        # Time
        self.time: Union[int, float] = self.time_init

        # Time step. Initially, equal to the initial time step
        self.dt: Union[int, float] = self.dt_init

        # Private attributes
        # Number of times the solution has been recomputed
        self._recomp_num: int = 0

        # Index of the next scheduled time
        self._scheduled_idx: int = 1

        # Print information
        # TODO: In the future, printing should be promoted to a logging strategy
        self._print_info: bool = print_info

        # Flag to keep track of recomputed solutions
        # TODO: Is this really needed?
        self._recomp_sol: bool = False

    def __repr__(self) -> str:

        s = "Time-stepping control object with attributes:\n"
        s += f"Initial and final simulation time = ({self.time_init}, {self.time_final})\n"
        s += f"Initial time step = {self.dt_init}\n"
        s += f"Minimum and maximum time steps = {self.dt_min_max}\n"
        s += f"Optimal iteration range = {self.iter_optimal_range}\n"
        s += f"Relaxation factors = {self.iter_relax_factors}\n"
        s += f"Recomputation factor = {self.recomp_factor}\n"
        s += f"Maximum recomputation attempts = {self.recomp_max}\n"
        s += f"Current time step and time are {self.dt} and {self.time}."

        return s

    def next_time_step(
        self, iterations: int, recompute_solution: bool = False
    ) -> Union[float, None]:
        """Determine next time step based on the previous number of iterations.

        For an in-depth explanation of the algorithm, refer to the sections Algorithm Overview
        and Algorithm Workflow from below.

        Parameters:
            iterations: Number of non-linear iterations. In time-dependent simulations,
                this typically represents the number of iterations for a given time step.
            recompute_solution: Whether the solution needs to be recomputed or not. If True,
                then the time step is multiplied by `recomp_factor`. If False, then the time
                step will be tuned accordingly.

        Returns: Next time step if time < final_time. None, otherwise.

        Algorithm Overview: Below, we provide a brief overview of the algorithm.

            Provided `recompute_solution = False`, the algorithm will adapt the time step
            based on `iterations`. If `iterations` is less than the lower endpoint of the
            optimal iteration range, then it will increase the time step by a factor
            `iter_relax_factors[1]`. If `iterations` is greater than the upper endpoint of the
            optimal iteration range it will decrease the time step by a factor
            `iter_relax_factors[0]`. Otherwise, `iterations` lies in the optimal iteration
            range, and time step remains unchanged.

            If `recompute_solution = True`, then the time step will be decreased by a factor
            `recomp_factor` with the hope of achieving convergence in the next time level. To
            avoid an infinite loop, an error will be raised if the method is called more than
            `recomp_max` consecutive times with the flag `recompute_solution = True`.

            Now that the algorithm has determined a new time step, it has to ensure three more
            conditions, (1) the calculated time step cannot be smaller than dt_min,
            (2) the calculated time step cannot be larger than dt_max, and (3) the
            time step cannot be too large such that the next time will exceed a scheduled
            time. These three conditions are implemented in this order of precedence and
            will override any of the previous calculated time steps.

        Algorithm Workflow: For completeness, we include the full algorithm in pseudocode.

            INPUT
                tsc // time step control object properly initialized
                iterations // number of non-linear interations
                recompute_solution // boolean flag

            IF time > final simulation time THEN
                RETURN None
            ENDIF

            IF constant_dt is True THEN
                RETURN dt_init
            ENDIF

            IF recompute_solution is False THEN
                RESET counter that keeps track of number of recomputing attempts
                IF iterations < lower endpoint of optimal iteration range THEN
                    DECREASE dt // multiply by over_relax_factor
                IFELSE iterations > upper endpoint of optimal iteration range THEN
                    INCREASE dt // multiply by under_relax_factor
                ELSE
                    PASS // dt reamains unchanged
                ENDIF
            ELSE
                IF number of recomputing attempts has not been exhausted THEN
                    SUBSTRACT dt from current time // we have to "go back in time"
                    DECREASE dt // multiply by recomp_factor
                    INCREASE counter that keeps track of number of recomputing attempts
                ELSE
                    RAISE Error // maximum number of recomputing attempts has been exhausted
                ENDIF
            ENDIF

            IF dt < dt_min THEN
                SET dt = dt_min
            ENDIF

            IF dt > dt_max THEN
                SET dt = dt_max
            ENDIF

            IF time + dt > a scheduled time THEN
                SET dt = scheduled time - time
            ENDIF

            RETURN dt

        """

        # For bookkeeping reasons, save recomputation flag
        # TODO: Is this really needed?
        self._recomp_sol = recompute_solution

        # First, check if we reach final simulation time
        if self.time >= self.time_final:
            return None

        # If the time step is constant, always return that value
        if self.is_constant:
            return self.dt_init

        # Adapt time step
        if not recompute_solution:
            self._adaptation_based_on_iterations(iterations=iterations)
        else:
            self._adaptation_based_on_recomputation()

        # Correct time step
        self._correction_based_on_dt_min()
        self._correction_based_on_dt_max()
        self._correction_based_on_schedule()

        return self.dt

    def _adaptation_based_on_iterations(self, iterations: int) -> None:
        """Provided convergence, adapt time step based on the number of iterations.

        Parameters:
            iterations: Number of non-linear iterations needed to achieve convergence.

        Raises: Warning if `iterations` > `max_iter`.

        """

        # Sanity check: Make sure the given number of iterations is less or equal than the
        # maximum number of iterations
        # TODO: Add unit test for this sanity check
        if iterations > self.iter_max:
            msg = (
                f"The given number of iterations '{iterations}' is larger than the maximum "
                f"number of iterations '{self.iter_max}'. This usually means that the solver "
                f"did not converge, but since recompute_solution = False was given, the "
                f"algorithm will adapt the time step anyways."
            )
            warnings.warn(msg)

        # Make sure to reset the recomputation counter
        self._recomp_num = 0

        # Proceed to determine the next time step using the following criteria:
        #     (C1) If the number of iterations is less than the lower endpoint of the optimal
        #     iteration range `iter_low`, we can relax the time step by multiplying it by an
        #     over-relaxation factor greater than 1, i.e., `over_relax_factor`.
        #     (C2) If the number of iterations is greater than the upper endpoint of the
        #     optimal iteration range `iter_upp`, we have to decrease the time step by
        #     multiplying it by an under-relaxation factor smaller than 1, i.e.,
        #     `under_relax_factor`.
        #     (C3) If neither of these situations occur, then the number iterations lies
        #     in the optimal iteration range, and the time step remains unchanged.
        if iterations <= self.iter_optimal_range[0]:  # (C1)
            self.dt = self.dt * self.iter_relax_factors[1]
            if self._print_info:
                print(f"Relaxing time step. Next dt = {self.dt}.")
        elif iterations >= self.iter_optimal_range[1]:  # (C2)
            self.dt = self.dt * self.iter_relax_factors[0]
            if self._print_info:
                print(f"Restricting time step. Next dt = {self.dt}.")
        else:
            pass  # (C3)

    def _adaptation_based_on_recomputation(self) -> None:
        """Adapt (decrease) time step when the solution failed to converge.

        Raises: ValueError if recomp_attemps > max_recomp_attempts. That is, when the maximum
            number of recomputation attempts has been exhausted.

        """

        if self._recomp_num < self.recomp_max:
            # If the solution did not converge AND we are allowed to recompute it, then:
            #   (S1) Update simulation time since solution will be recomputed.
            #   (S2) Decrease time step multiplying it by the recomputing factor < 1.
            #   (S3) Increase counter that keeps track of the number of times the solution
            #        was recomputed.

            # Note that iterations is not really used here. So, as long as
            # recompute_solution = True and recomputation_attempts < max_recomp_attempts,
            # the method is entirely agnostic to the number of iterations passed. This
            # design choice was made to give more flexibility, in the sense that we are
            # not limiting the recomputation criteria to _only_ reaching the maximum
            # number of iterations, even though that is the primary intended usage.
            self.time -= self.dt  # (S1)
            self.dt *= self.recomp_factor  # (S2)
            self._recomp_num += 1  # (S3)
            if self._print_info:
                msg = (
                    "Solution did not converge and will be recomputed."
                    f" Recomputing attempt #{self._recomp_num}. Next dt = {self.dt}."
                )
                print(msg)
        else:
            # The solution did not converge AND we exhausted all recomputation attempts
            msg = (
                f"Solution did not converge after {self.recomp_max} recomputing "
                "attempts."
            )
            raise ValueError(msg)

    def _correction_based_on_dt_min(self) -> None:
        """Correct time step if dt < dt_min."""
        if self.dt < self.dt_min_max[0]:
            self.dt = self.dt_min_max[0]
            if self._print_info:
                print(
                    f"Calculated dt < dt_min. Using dt_min = {self.dt_min_max[0]} instead."
                )

    def _correction_based_on_dt_max(self) -> None:
        """Correct time step if dt > dt_max."""
        if self.dt > self.dt_min_max[1]:
            self.dt = self.dt_min_max[1]
            if self._print_info:
                print(
                    f"Calculated dt > dt_max. Using dt_max = {self.dt_min_max[1]} instead."
                )

    def _correction_based_on_schedule(self) -> None:
        """Correct time step if time + dt > scheduled_time."""
        schedule_time = self.schedule[self._scheduled_idx]
        if (self.time + self.dt) > schedule_time:
            self.dt = schedule_time - self.time  # correct  time step

            if self._scheduled_idx < len(self.schedule) - 1:
                self._scheduled_idx += 1  # increase index to catch next scheduled time
                if self._print_info:
                    print(
                        f"Correcting time step to match scheduled time. Next dt = {self.dt}."
                    )
            else:
                if self._print_info:
                    print(
                        f"Correcting time step to match final time. Final dt = {self.dt}."
                    )

    # Helpers
    @staticmethod
    def _is_strictly_increasing(check_array: np.ndarray) -> bool:
        """Checks if a list is strictly increasing.

        Parameters:
            check_array: Array to be tested.

        Returns: True or False.

        """
        return all(a < b for a, b in zip(check_array, check_array[1:]))
