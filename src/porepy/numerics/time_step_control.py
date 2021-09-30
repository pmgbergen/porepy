from typing import Optional, Tuple, Union

__all__ = ["TimeSteppingControl"]


class TimeSteppingControl:
    """Parent class for iteration-based time stepping control routine."""

    def __init__(
        self,
        schedule: list,
        dt_init: float,
        dt_min_max: Tuple[float, float],
        iter_max: int,
        iter_optimal_range: Tuple[int, int],
        iter_lowupp_factor: Optional[Tuple[float, float]] = None,
        recomp_factor: Optional[float] = None,
        recomp_max: Optional[int] = None,
        print_info: Optional[bool] = None,
    ):
        """Computes the next time step based on the number of non-linear iterations.

        Parameters:
            schedule (List): List containing the target times for the simulation.
                The time-stepping algorithm will adapt the time step so that the target
                times are guaranteed to be hit/reached. The list must contain minimally
                two elements, corresponding to the the initial and final simulation times.
                Lists of length > 2 must contain strictly increasing times.
                Examples of valid inputs are:
                  [0, 1], [0, 10, 30, 50], [0, 1*pp.HOUR, 3*pp.HOUR].
                Examples of invalid inputs are:
                  [1], [1, 0], [0, 1, 1, 2].
            dt_init (float): Initial time step.
            dt_min_max (Tuple of float): Minimum and maximum permissible time steps.
            iter_max (int): Maximum number of iterations.
            iter_optimal_range (Tuple of int): Lower and upper optimal iteration range.
            iter_lowupp_factor (Tuple of float, optional): Lower and upper multiplication
                factors. Default is (1.3, 0.7).
            recomp_factor (float). Failed-to-converge recomputation factor. Default is 0.5.
            recomp_max (int). Failed-to-converge maximum recomputation attempts. Default is 10.
            print_info (bool). Print time-stepping information. Default is True.

        """

        # Sanity checks
        if len(schedule) < 2:
            s = "Schedule list must have at least two items, representing the initial and"
            s += " final simualtion time."
            raise ValueError(s)
        elif schedule[0] < 0:
            raise ValueError("Initial time cannot be negative.")
        elif schedule[-1] < schedule[0]:
            raise ValueError("Final time cannot be smaller than initial time.")
        elif not self._is_strictly_increasing(schedule):
            raise ValueError("Schedule must contain strictly increasing times.")

        if dt_init <= 0:
            raise ValueError("Initial time step must be positive")
        elif dt_init > schedule[-1]:
            raise ValueError(
                "Inital time step cannot be larger than final simulation time."
            )
        elif dt_init < dt_min_max[0]:
            raise ValueError(
                "Intial time step cannot be smaller than minimum time step."
            )
        elif dt_init > dt_min_max[1]:
            raise ValueError(
                "Intial time step cannot be larger than maximum time step."
            )

        if dt_min_max[0] > dt_min_max[1]:
            s = "Minimum time step cannot be larger than maximum time step."
            raise ValueError(s)

        if iter_max <= 0:
            raise ValueError("Maximum amount of iterations must be a postive integer")

        if iter_optimal_range[0] > iter_optimal_range[1]:
            s = "Lower optimal iteration range cannot be larger than"
            s += " upper optimal iteration range."
            raise ValueError(s)
        elif iter_optimal_range[1] > iter_max:
            s = "Upper optimal iteration range cannot be larger than"
            s += " maximum amount of iterations."
            raise ValueError(s)

        if iter_lowupp_factor is not None and (iter_lowupp_factor[0] <= 1):
            raise ValueError("Expected lower multiplication factor > 1")
        elif iter_lowupp_factor is not None and (iter_lowupp_factor[1] >= 1):
            raise ValueError("Expected upper multiplication factor < 1")

        if (recomp_factor is not None) and recomp_factor >= 1:
            raise ValueError("Expected recomputation factor < 1")

        if (recomp_max is not None) and recomp_max <= 0:
            raise ValueError(
                "Number of recomputation attempts must be a positive integer"
            )

        # Schedule, initial, and final times
        self.schedule = schedule
        self.time_init = schedule[0]
        self.time_final = schedule[-1]

        # Initial time step
        self.dt_init = dt_init

        # Minimum and maximum allowable time steps
        self.dt_min, self.dt_max = dt_min_max

        # Maximum amount of iterations
        self.iter_max = iter_max

        # Target iteration range
        self.iter_low, self.iter_upp = iter_optimal_range

        # Lower and upper multiplication factors
        if iter_lowupp_factor is not None:
            self.iter_low_factor, self.iter_upp_factor = iter_lowupp_factor
        else:
            self.iter_low_factor = 1.3
            self.iter_upp_factor = 0.7

        # Re-computation multiplication factor
        if recomp_factor is not None:
            self.recomp_factor = recomp_factor
        else:
            self.recomp_factor = 0.5

        # Number of permissible re-computation attempts
        if recomp_max is not None:
            self.recomp_max = recomp_max
        else:
            self.recomp_max = 10

        # Print information
        if print_info is None:
            self._print_info = True
        else:
            self._print_info = print_info

        # Time
        self.time = self.time_init

        # Time step. Initially, equal to the initial time step
        self.dt = self.dt_init

        # Recompute solution flag
        self.recomp_sol = False

        # Private attributes
        # Number of times the solution has been recomputed
        self._recomp_num = 0

        # Index of the next scheduled time
        self._scheduled_idx = 1

    def __repr__(self) -> str:

        s = "Time-stepping control object with atributes:\n"
        s += f"Initial simulation time = {self.time_init}\n"
        s += f"Final simulation time = {self.time_final}\n"
        s += f"Initial time step = {self.dt_init}\n"
        s += f"Minimum time step = {self.dt_min}\n"
        s += f"Maximum time step = {self.dt_max}\n"
        s += f"Lower optimal iteration range = {self.iter_low}\n"
        s += f"Upper optimal iteration range = {self.iter_upp}\n"
        s += f"Lower multiplication factor = {self.iter_low_factor}\n"
        s += f"Upper multiplication factor = {self.iter_upp_factor}\n"
        s += f"Recompute solution multiplication factor = {self.recomp_factor}\n"
        s += f"Maxximum recomputing attempts = {self.recomp_max}"

        return s

    def next_time_step(self, iters: int) -> Union[float, None]:
        """
        Determines the next time step based on the previous amount of iterations needed
        to reach convergence. If convergence was not achieved, then the time step is
        reduced by recomp_factor. The time-stepping control routine will recompute the
        solution recomp_max times. Otherwise, an error will be raised and the simulation
        stopped.

        Parameters
        iters (int): Number of non-linear iterations. In time-dependent simulations,
            this tipically represent the number of iterations for a time step.

        Returns
        -------
        dt: float
            Next time step
        """

        # First, check if we are allowed to recompute the solution
        if self._recomp_num > self.recomp_max:
            s = f"Solution did not convergece after {self.recomp_max}"
            s += " recomputing attempts."
            raise ValueError(s)

        # Check if we reach final simulation time
        if self.time == self.time_final:
            return None

        # If iters == max_iter:
        #   Decrease time step by the recomputing factor
        #   Update time (since solution will be recomputed)
        #   Set to True the re-computation flag
        #   Increase counter that keeps track of how many times the solution was recomputed
        if iters == self.iter_max:
            if self._print_info:
                s = "Solution did not converge. Reducing time step and recomputing solution."
                s += f" Next dt = {self.dt}."
                print(s)
            self.time -= self.dt  # reduce time
            self.dt = self.dt * self.recomp_factor  # reduce time step
            self.recomp_sol = True
            self._recomp_num += 1
            return self.dt
        else:
            self.recomp_sol = False
            self._recomp_num = 0

        # If iters < max_iter. Proceed to determine the next time step using the
        # following criteria.
        # If iters is less than the lower optimal iteration range "iter_low", we can relax
        # the time step, and multiply by a lower multiplication factor greater than 1,
        # i.e., "factor_low". If the number of iterations is greater than the upper optimal
        # iteration range "iter_upp", we have to decrease the time step by multiplying
        # by an upper multiplication factor smaller than 1, i.e., "factor_upp". If neither
        # of these situations occur, then the number iterations lies between the optimal
        # iteration range, and the time step remains unchanged.
        if iters <= self.iter_low:
            self.dt = self.dt * self.iter_low_factor
            if self._print_info:
                print(f"Relaxing time step. Next dt = {self.dt}.")
        elif iters >= self.iter_upp:
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

        # Check if we reach a scheduled time, and match time step
        schedule_time = self.schedule[self._scheduled_idx]
        if (self.time + self.dt) > schedule_time:
            self.dt = schedule_time - self.time  # adapt time step
            if self._scheduled_idx < len(self.schedule) - 1:
                self._scheduled_idx += 1  # increase index to catch next scheduled time
                if self._print_info:
                    print(
                        f"Correcting dt to match scheduled time. Next dt = {self.dt}."
                    )
            else:
                if self._print_info:
                    print(f"Correcting dt to match final time. Final dt = {self.dt}.")

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
