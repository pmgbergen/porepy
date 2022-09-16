from typing import Optional, Tuple, Union

import numpy as np

__all__ = ["TimeSteppingControl"]


class TimeSteppingControl:
    """Parent class for iteration-based time stepping control routine."""

    def __init__(
        self,
        schedule: list,
        dt_init: float,
        constant_dt: bool = False,
        dt_min_max: Optional[Union[Tuple[float, float], None]] = None,
        iter_max: int = 15,
        iter_optimal_range: Tuple[int, int] = (4, 7),
        iter_lowupp_factor: Tuple[float, float] = (1.3, 0.7),
        recomp_factor: float = 0.5,
        recomp_max: int = 10,
        print_info: bool = False,
    ) -> None:
        """Constructor for the TimeSteppingControl class.

        Parameters:
            schedule: List containing the target times for the simulation.
                The time-stepping algorithm will adapt the time step so that the target
                times are guaranteed to be hit/reached. The list must contain minimally
                two elements, corresponding to the initial and final simulation times.
                Lists of length > 2 must contain strictly increasing times.
                Examples of VALID inputs are:
                  [0, 1]
                  [0, 10, 30, 50]
                  [0, 1*pp.HOUR, 3*pp.HOUR].
                Examples of INVALID inputs are:
                  [1]
                  [1, 0]
                  [0, 1, 1, 2]
            dt_init: Initial time step. If `constant_dt` is True, then `dt_init` is taken as
                the constant time step.
            constant_dt: Whether to treat the time step as constant or not. If True, then
                the time-stepping control algorithm is effectively bypassed. To be
                precise, the algorithm won't adapt the time step in any situation.
                Nevertheless, the attributes (such as scheduled times) will still be accesible.
            dt_min_max: Minimum and maximum permissible time steps. If None, then the
                minimum time step is set to 0.1% of the final simulation time and the
                maximum time step is set to 10% of the final simulation time. If given, then
                the first and second elements of the tuple corresponds to the minimum and
                maximum time steps, respectively.
            iter_max: Maximum number of iterations.
            iter_optimal_range: Optimal iteration range. The first and second elements of the
                tuple corresponds to the lower and upper bounds of the optimal iteration range.
            iter_lowupp_factor: Lower and upper multiplication factors. The first and second
                elements of the tuple corresponds to the lower and upper multiplication
                factors, respectively. We require the lower multiplication factor to be
                strictly greater than one, whereas the upper multiplication factor is
                required to be strictly less than one.
            recomp_factor: Failed-to-converge recomputation factor. Factor by which the
                time step will be multiplied in case the solution must be recomputed (see
                documentation of `next_time_step` method).
            recomp_max: Failed-to-converge maximum recomputation attempts. The maximum
                allowable number of consecutive recomputation attempts. If `recomp_max` is
                exceeded, an error will be raised.
            print_info. Print time-stepping information.

        Example:
            # The following is an example on how to construct a time-stepping object
            tsc = pp.TimeSteppingControl(
                schedule=[0, 10],
                dt_init=0.5,
                dt_min_max=(0.1, 2),
                iter_max=10,
                iter_optimal_range=(4, 7),
                iter_lowupp_factor=(1.1, 0.9),
                recomp_factor=0.5,
                recomp_max=5,
                print_info=True
            )
            # To inspect the current attributes of the object
            print(tsc)

        """

        # Sanity checks for schedule
        if len(schedule) < 2:
            raise ValueError("Expected schedule with at least two items.")
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

        # Set dt_min_max if necessary
        if dt_min_max is None:
            dt_min_max = (0.01 * schedule[-1], 0.1 * schedule[1])

        # More sanity checks below. Note that all the remaining sanity checks (but one) are
        # only needed when constant_dt = False. Thus, to save time when constant_dt = True,
        # we use this rather ugly if-statement from below.

        if not constant_dt:

            # Sanity checks for dt_min and dt_max
            if dt_init < dt_min_max[0]:
                raise ValueError(
                    "Initial time step cannot be smaller than minimum time step."
                )
            elif dt_init > dt_min_max[1]:
                raise ValueError(
                    "Initial time step cannot be larger than maximum time step."
                )

            # NOTE: The above checks guarantee that minimum time step <= maximum time step

            # Sanity checks for maximum number of iterations.
            # Note that 0 is a possibility. This will imply that the solver reaches
            # convergence directly, e.g., as in linear solvers.
            if iter_max < 0:
                raise ValueError("Maximum number of iterations must be non-negative.")

            # Sanity checks for optimal iteration range
            if iter_optimal_range[0] > iter_optimal_range[1]:
                s = "Lower optimal iteration range cannot be larger than"
                s += " upper optimal iteration range."
                raise ValueError(s)
            elif iter_optimal_range[1] > iter_max:
                s = "Upper optimal iteration range cannot be larger than"
                s += " maximum number of iterations."
                raise ValueError(s)
            elif iter_optimal_range[0] < 0:
                s = "Lower optimal iteration range cannot be negative."
                raise ValueError(s)

            # Sanity checks for lower and upper multiplication factors
            if iter_lowupp_factor[0] <= 1:
                raise ValueError("Expected lower multiplication factor > 1.")
            elif iter_lowupp_factor[1] >= 1:
                raise ValueError("Expected upper multiplication factor < 1.")

            # Checks for sensible combinations of iter_optimal_range and iter_lowupp_factor
            if dt_min_max[0] * iter_lowupp_factor[0] > dt_min_max[1]:
                raise ValueError("Encountered dt_min * iter_low_factor > dt_max.")
            elif dt_min_max[1] * iter_lowupp_factor[1] < dt_min_max[0]:
                raise ValueError("Encountered dt_max * iter_upp_factor < dt_min.")

            # Sanity check for recomputation factor
            if recomp_factor >= 1:
                raise ValueError("Expected recomputation multiplication factor < 1.")

            # Sanity check for maximum number of recomputation attempts
            if recomp_max <= 0:
                raise ValueError("Number of recomputation attempts must be > 0.")

        else:

            # If the time step is constant, check that the scheduled times and the time
            # step are compatible.
            # E.g. dt=2 works with schedule = [0,2,4,8], but dt=3 does not.
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
        self.dt_min, self.dt_max = dt_min_max

        # Maximum number of iterations
        # TODO: This is really a property of the nonlinear solver. A full integration will
        #  most likely requiring "pulling" this parameter from the solver side.
        self.iter_max = iter_max

        # Optimal iteration range
        # TODO: This is really a property of the nonlinear solver. A full integration will
        #  most likely requiring "pulling" this parameter from the solver side.
        self.iter_low, self.iter_upp = iter_optimal_range

        # Lower and upper multiplication factors
        self.iter_low_factor, self.iter_upp_factor = iter_lowupp_factor

        # Re-computation multiplication factor
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
        self._recomp_sol: bool = False

    def __repr__(self) -> str:

        s = "Time-stepping control object with atributes:\n"
        s += f"Initial and final simulation time = ({self.time_init}, {self.time_final})\n"
        s += f"Initial time step = {self.dt_init}\n"
        s += f"Minimum and maximum time steps = ({self.dt_min}, {self.dt_max})\n"
        s += f"Optimal iteration range = ({self.iter_low}, {self.iter_upp})\n"
        s += (
            f"Lower and upper multiplication factors = ({self.iter_low_factor}, "
            f"{self.iter_upp_factor})\n"
        )
        s += f"Recompute solution multiplication factor = {self.recomp_factor}\n"
        s += f"Maximum recomputation attempts = {self.recomp_max}\n"
        s += f"Current time step and time are {self.dt} and {self.time}."

        return s

    def next_time_step(
        self, iterations: int, recompute_solution: bool
    ) -> Union[float, None]:
        """
        Determines the next time step based on the previous number of iterations needed
        to reach convergence. If convergence was not achieved, then the time step is
        reduced by recomp_factor. The time-stepping control routine will recompute the
        solution recomp_max times. Otherwise, an error will be raised and the simulation
        stopped.

        Parameters
        iterations (int): Number of non-linear iterations. In time-dependent simulations,
            this tipically represent the number of iterations for a given time step.
        recompute_solution (bool): Wheter the solution needs to be recomputed or not. If
            True, then the time step is multiplied by recomp_factor. If False, the time
            step will be tuned accordingly.

        Returns
        dt (float or None):  Next time step if time < final_simulation time. None otherwise.

        """

        # For bookkeeping reasons, save recomputation flag
        self._recomp_sol = recompute_solution

        # First, check if we reach final simulation time
        if self.time >= self.time_final:
            return None

        # if time step is constant, always return that value
        if self.is_constant:
            # TODO if time step is constant, this can lead to an infinite loop
            return self.dt_init

        # If the solution did not convergence and we are allow to recompute it:
        #   Update simulation time (since solution will be recomputed).
        #   Decrease time step multiplying it by the recomputing factor < 1.
        #   Increase counter that tracks the number of times that the solution was recomputed.
        #   Check if calculated time step is larger than dt_min. Otherwise, use dt_min.
        # Note that iterations is not really used here. So, as long as the passed
        # recompute_solution==True and recomputation_attempts < max_recomp_attempts, the user
        # can pass any number of iterations. This, in principle, allows for more flexibility,
        # in the sense that we are no restricting the recomputation criteria to only reaching
        # the maximum number of iterations, even though that is the primary intended usage.
        if recompute_solution and self._recomp_num < self.recomp_max:
            self.time -= self.dt
            self.dt *= self.recomp_factor
            self._recomp_num += 1
            if self._print_info:
                s = "Solution did not converge and will be recomputed."
                s += f" Recomputing attempt #{self._recomp_num}. Next dt = {self.dt}."
                print(s)
            if self.dt < self.dt_min:
                self.dt = self.dt_min
                if self._print_info:
                    print(
                        f"Calculated dt < dt_min. Using dt_min = {self.dt_min} instead."
                    )
            return self.dt
        elif not recompute_solution:  # we reach convergence, set recomp_num to zero
            self._recomp_num = 0
        else:  # number of recomputing attempts has been exhausted
            msg = f"Solution did not converge after {self.recomp_max} recomputing attempts."
            # TODO: Should this be a RunTimeError perhaps?
            raise ValueError(msg)

        # If iters < max_iter. Proceed to determine the next time step using the
        # following criteria.
        # If iters is less than the lower optimal iteration range "iter_low", we can relax
        # the time step, and multiply by a lower multiplication factor greater than 1,
        # i.e., "factor_low". If the number of iterations is greater than the upper optimal
        # iteration range "iter_upp", we have to decrease the time step by multiplying by an
        # upper multiplication factor smaller than 1, i.e., "factor_upp". If neither of these
        # situations occur, then the number iterations lies between the optimal iteration
        # range and the time step remains unchanged.
        if iterations <= self.iter_low:
            self.dt = self.dt * self.iter_low_factor
            if self._print_info:
                print(f"Relaxing time step. Next dt = {self.dt}.")
        elif iterations >= self.iter_upp:
            self.dt = self.dt * self.iter_upp_factor
            if self._print_info:
                print(f"Restricting time step. Next dt = {self.dt}.")

        # Check if the calculated time step is less than the minimum allowable time step
        if self.dt < self.dt_min:
            self.dt = self.dt_min
            if self._print_info:
                print(f"Calculated dt < dt_min. Using dt_min = {self.dt_min} instead.")

        # Check if the calculated time step is greater than the maximum allowable time step
        if self.dt > self.dt_max:
            self.dt = self.dt_max
            if self._print_info:
                print(f"Calculated dt > dt_max. Using dt_max = {self.dt_max} instead.")

        # Check if we reach a scheduled time
        schedule_time = self.schedule[self._scheduled_idx]
        if (self.time + self.dt) > schedule_time:
            self.dt = schedule_time - self.time  # adapt time step

            if self._scheduled_idx < len(self.schedule) - 1:
                self._scheduled_idx += 1  # increase index to catch next scheduled time
                if self._print_info:
                    print(
                        f"Correcting time step to match scheduled time. Next dt = {self.dt}."
                    )
            else:
                if self._print_info:
                    print(
                        f"Correcting time step to match final time. Next dt = {self.dt}."
                    )
        return self.dt

    # Helpers
    @staticmethod
    def _is_strictly_increasing(check_list: list) -> bool:
        """Checks if a list is strictly increasing

        Parameters
            check_list (List): List to be tested

        Returns
            (bool): True or False

        """
        return all(a < b for a, b in zip(check_list, check_list[1:]))
