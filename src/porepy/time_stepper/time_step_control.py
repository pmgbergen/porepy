"""
This module contains a routine for iteration-based time-stepping control.

The algorithm is heavily inspired by [1], which was later used in [2].

[1] Simunek, J., Van Genuchten, M. T., & Sejna, M. (2005). The HYDRUS-1D software
    package for simulating the one-dimensional movement of water, heat, and multiple
    solutes in variably-saturated media. University of California-Riverside Research
    Reports, 3, 1-240.

[2] Varela, J., Gasda, S. E., Keilegavlen, E., & Nordbotten, J. M. (2021). A
    Finite-Volume-Based Module for Unsaturated Poroelasticity. Advanced Modeling with
    the MATLAB Reservoir Simulation Toolbox.

Algorithm Overview:

    Provided `recompute_solution = False`, the algorithm will adapt the time step based
    on `iterations`. If `iterations` is less than the lower endpoint of the optimal
    iteration range, then it will increase the time step by a factor
    `iter_relax_factors[1]`. If `iterations` is greater than the upper endpoint of the
    optimal iteration range it will decrease the time step by a factor
    `iter_relax_factors[0]`. Otherwise, `iterations` lies in the optimal iteration
    range, and time step remains unchanged.

    If `recompute_solution = True`, then the time step will be reduced by a factor
    `recomp_factor` with the hope of achieving convergence in the next time level. The
    algorithm will keep decreasing the time step unless: (1) the time step is equal to
    the minimum admissible time step or (2) the number of recomputing attempts has been
    exhausted. In both cases, an error will be raised.

    Now that the algorithm has determined a new time step, it has to ensure three more
    conditions, (1) the calculated time step cannot be smaller than dt_min, (2) the
    calculated time step cannot be larger than dt_max, and (3) the time step cannot be
    too large such that the next time will exceed a scheduled time. These three
    conditions are implemented in this order of precedence and will override any of the
    previous calculated time steps.

Algorithm Workflow in Pseudocode:

    INPUT
        time_manager // time step control object properly initialized iterations //
        number of non-linear iterations recompute_solution // boolean flag

    IF time > final simulation time THEN
        RETURN None
    ENDIF

    IF constant_dt is True THEN
        RETURN dt_init
    ENDIF

    IF recompute_solution is False THEN
        RESET counter that keeps track of number of recomputing attempts IF iterations <
        lower endpoint of optimal iteration range THEN
            DECREASE dt // multiply by an over relaxation factor > 1
        IFELSE iterations > upper endpoint of optimal iteration range THEN
            INCREASE dt // multiply by an under relaxation factor < 1
        ELSE
            PASS // dt remains unchanged
        ENDIF
    ELSE
        IF number of recomputing attempts has not been exhausted THEN
            IF dt is equal to dt_min THEN
                RAISE Error // since recomputation will not have any effect
            ENDIF
            SUBTRACT dt from current time // we have to "go back in time"
            DECREASE dt // multiply by recomputation factor < 1
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

from __future__ import annotations

from typing import Optional, Union
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["TimeManager"]


class TimeManager:
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
        rtol: float = 1e-10,
        atol: float = 1e-16,
    ) -> None:
        warn(message="", category=FutureWarning, stacklevel=2)
        self.schedule = np.array(schedule)
        self.time_init = self.schedule[0]
        self.time_final = self.schedule[-1]
        self.dt_init = dt_init
        self.dt_min_max = dt_min_max
        self.iter_optimal_range = iter_optimal_range
        self.iter_relax_factors = iter_relax_factors
        self.recomp_factor = recomp_factor
        self.is_constant = constant_dt

        # Time
        self.time: Union[int, float] = self.time_init

        # Time step. Initially, equal to the initial time step
        self.dt: Union[int, float] = self.dt_init

        # Time index
        self.time_index: int = 0

        # Private attributes
        # Number of times the solution has been recomputed
        self._recomp_num: int = 0

        # Index of the next scheduled time
        self._scheduled_idx: int = 1

        # Print information
        # TODO: In the future, printing should be promoted to a logging strategy
        self._print_info: bool = print_info

        # Keep track of recomputed solutions and current number of iterations
        self._recomp_sol: bool = False
        self._iters: Union[int, None] = None

        # Bookkeeping of saved time steps for restarting purposes.
        self.exported_dt: list[pp.number] = []
        """A list of time steps for the simulation states that were saved on disk with
        `write_time_information` for restarting purposes. Completeness and lack of
        duplication are NOT guaranteed.

        NOTE: This property cannot be inferred from `exported_times`, consider the case
        when not every time step is saved.

        """
        self.exported_times: list[pp.number] = []
        """A list of time points for the simulation states that were saved on disk with
        `write_time_information` for restarting purposes. Completeness and lack of
        duplication are NOT guaranteed.

        """

    def __repr__(self) -> str:
        s = "Time-stepping control object with attributes:\n"
        s += "Initial and final simulation time = "
        s += f"({self.time_init}, {self.time_final})\n"
        s += f"Initial time step = {self.dt_init}\n"
        s += f"Minimum and maximum time steps = {self.dt_min_max}\n"
        s += f"Optimal iteration range = {self.iter_optimal_range}\n"
        s += f"Relaxation factors = {self.iter_relax_factors}\n"
        s += f"Recomputation factor = {self.recomp_factor}\n"
        s += f"Maximum recomputation attempts = {self.recomp_max}\n"
        s += f"Current time step and time are {self.dt} and {self.time}."

        return s

    def elapsed_time(self) -> float:
        """Return the elapsed simulation time."""
        return self.time - self.time_init

    def is_at_initial_time(self) -> bool:
        """Check whether the time manager is at the initial time."""
        return self.time < self.time_init or np.isclose(
            self.time, self.time_init, rtol=self.rtol, atol=0.5 * self.dt_min_max[0]
        )

    def final_time_reached(self) -> bool:
        """Check whether the time manager has reached the end of the schedule.

        Returns:
            Whether the final time has reached or been overstepped.

        """
        return self.time > self.time_final or np.isclose(
            self.time, self.time_final, rtol=self.rtol, atol=self.atol
        )

    def compute_time_step(
        self, iterations: Optional[int] = None, recompute_solution: bool = False
    ) -> Union[float, None]:
        """Determine next time step based on the previous number of iterations.

        See also Algorithm Overview and Algorithm Workflow from the module
        documentation.

        Parameters:
            iterations: Number of non-linear iterations. In time-dependent simulations,
                this typically represents the number of iterations for a given time
                step. A warning is raised if `iterations` is given when
                `recompute_solution = True` or `constant_dt = True`.
            recompute_solution: Whether the solution needs to be recomputed or not. If
                True, then the time step is multiplied by `recomp_factor`. If False,
                then the time step will be tuned accordingly.

        Returns:
            Next time step if time < final_time. None, otherwise.

        """

        # For bookkeeping reasons, save recomputation and iterations
        self._recomp_sol = recompute_solution
        self._iters = iterations

        # First, check if we reach final simulation time with a valid solution. This
        # works as a safeguard and should not be removed if the logic below is not
        # reconsidered.
        if not recompute_solution and self.final_time_reached():
            return None

        # If the time step is constant, always return that value. TimeStepper currently
        # may ask to recompute dt even for the constant time step. This is temporary,
        # the time manager will soon stop being responsible for recomputing the time
        # step. Keeping the original behavior for now.
        if self.is_constant:
            # Some sanity checks
            if iterations is not None:
                msg = (
                    f"iterations '{iterations}' has no effect if time step is constant."
                )
                warnings.warn(msg)
            if recompute_solution:
                msg = "recompute_solution=True has no effect if time step is constant."
                warnings.warn(msg)

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

    def increase_time(self) -> None:
        """Increase simulation time by the current time step."""
        self.time += self.dt

    def increase_time_index(self) -> None:
        """Increase time index counter by one."""
        self.time_index += 1

    def _adaptation_based_on_iterations(self, iterations: Optional[int]) -> None:
        """Provided convergence, adapt time step based on the number of iterations.

        Parameters:
            iterations: Number of non-linear iterations needed to achieve convergence.

        Raises:
            ValueError if `iterations` is None.
            Warning if `iterations` > `max_iter`.

        """

        # Sanity check: Make sure iterations is given
        if iterations is None:
            msg = "Time step cannot be adapted without 'iterations'."
            raise ValueError(msg)

        # Sanity check: Make sure the given number of iterations is less or equal than
        # the maximum number of iterations
        if iterations > self.iter_max:
            msg = (
                f"The given number of iterations '{iterations}' is larger than the "
                f"maximum number of iterations '{self.iter_max}'. This usually means "
                "that the solver did not converge, but since recompute_solution ="
                " False was given, the algorithm will adapt the time step anyways."
            )
            warnings.warn(msg)

        # Proceed to determine the next time step using the following criteria: (C1) If
        #     the number of iterations is less than the lower endpoint of the optimal
        #     iteration range `iter_low`, we can relax the time step by multiplying it
        #     by an over-relaxation factor greater than 1, i.e., `over_relax_factor`.
        #     (C2) If the number of iterations is greater than the upper endpoint of the
        #     optimal iteration range `iter_upp`, we have to decrease the time step by
        #     multiplying it by an under-relaxation factor smaller than 1, i.e.,
        #     `under_relax_factor`. (C3) If neither of these situations occur, then the
        #     number iterations lies in the optimal iteration range, and the time step
        #     remains unchanged.
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
        """Adapt (decrease) time step when `recompute_solution` = True.

        Raises:
            ValueError if dt = dt_min, since any recomputation attempt will be
                pointless.

        """
        # If dt = dt_min, adaptation based on recomputation won't have any effect in
        # the next iteration (any decrease in time step will be corrected to dt_min
        # by self.correction_based_on_dt_min() in a subsequent correction step).
        # Thus, to avoid pointless iterations, we raise an error.
        if self.dt == self.dt_min_max[0]:
            msg = (
                "Recomputation will not have any effect since the time step "
                f"achieved its minimum admissible value -> dt = dt_min = {self.dt}."
            )
            raise ValueError(msg)

        # Raise a warning if iterations is not None.
        if self._iters is not None:
            msg = "Number of iterations has no effect in recomputation."
            warnings.warn(msg)

        # If the solution did not converge AND we are allowed to recompute it, then:
        #   (S1) Decrease time step multiplying it by the recomputing factor < 1.
        #   (S2) Step back in the schedule if we expected to meet the next schedule
        #        point.

        self.dt *= self.recomp_factor  # (S1)

        # When we refactor this into the TimeStepper's responsibility, it should be made
        # less complex and more robust, by not using indices.
        if self._is_about_to_hit_schedule:  # (S2)
            self._scheduled_idx -= 1

        if self._print_info:
            msg = (
                "Solution did not converge and will be recomputed."
                f" Recomputing attempt #{self._recomp_num}. Next dt = {self.dt}."
            )
            print(msg)

    def _correction_based_on_dt_min(self) -> None:
        """Correct time step if dt < dt_min."""
        if self.dt < self.dt_min_max[0]:
            self.dt = self.dt_min_max[0]
            if self._print_info:
                print(
                    f"Calculated dt < dt_min. Using dt_min = {self.dt_min_max[0]}"
                    " instead."
                )

    def _correction_based_on_dt_max(self) -> None:
        """Correct time step if dt > dt_max."""
        if self.dt > self.dt_min_max[1]:
            self.dt = self.dt_min_max[1]
            if self._print_info:
                print(
                    f"Calculated dt > dt_max. Using dt_max = {self.dt_min_max[1]}"
                    " instead."
                )

    def _correction_based_on_schedule(self) -> None:
        """Correct time step if time + dt > scheduled_time."""
        # When moving this to the TimeStepper, we should make this more efficient and
        # robust by keeping track of the next scheduled time, instead of the index
        # and updating it every time we hit the schedule. This way, we would also avoid
        # any issues related to the index, e.g., out of bounds, and we would not need to
        # step back in the schedule if we expected to hit it but did not due to
        # recomputation.
        schedule_time = self.schedule[self._scheduled_idx]

        self._is_about_to_hit_schedule = False

        if self.time + self.dt > schedule_time:
            self._is_about_to_hit_schedule = True
            self._scheduled_idx += 1  # Increase index to catch next scheduled time.

            if np.isclose(self.time, schedule_time, rtol=self.rtol, atol=self.atol):
                # Scheduled time will be reached within tol, no need for correction.
                if self._print_info:
                    print(
                        f"Not correcting time step to match scheduled time. Next dt ="
                        f" {self.dt}."
                    )
                return

            # Consider dt=1, t=0.999999, schedule_time=1. We'll decrease the time step
            # to dt=1e-6 and have to slowly increase it. When refactored, we should
            # track of previous dt, and return to that value if we expected to hit the
            # schedule. There is no reason to start increasing the dt from scratch.

            # Use a reset of previous dt to avoid oscillations and ensure a
            # stable time step adaptation in combination with the relaxation factors.
            self.dt = schedule_time - self.time  # Correcting time step.

            if self._scheduled_idx < len(self.schedule) - 1:
                if self._print_info:
                    print(
                        f"Correcting time step to match scheduled time. Next dt ="
                        f" {self.dt}."
                    )
            else:
                if self._print_info:
                    print(
                        f"Correcting time step to match final time. Final dt ="
                        f" {self.dt}."
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

    @staticmethod
    def is_schedule_in_simulated_times(
        schedule: np.ndarray,
        sim_times: np.ndarray,
        rtol: float = 1e-10,
        atol: float = 1e-16,
    ) -> bool:
        """Checks if ``schedule`` is a proper subset of ``sim_times`` for given
           tolerances

        Reference: https://github.com/numpy/numpy/issues/7784#issuecomment-848036186

        Parameters:
            schedule: First array.
            sim_times: Second array.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            True if all times in ``schedule`` intersect the elements of ``sim_times``
            with relative tolerance ``rtol`` and absolute tolerance ``atol`. False
            otherwise.

        """
        ss = np.searchsorted(schedule[1:-1], sim_times, side="left")
        in1d = np.isclose(schedule[ss], sim_times, rtol=rtol, atol=atol) | np.isclose(
            schedule[ss + 1], sim_times, rtol=rtol, atol=atol
        )
        return schedule.size == np.sum(in1d)

    # I/O
    def write_time_information(self, path: Path) -> None:
        """Keep track of history of time and time step size and store as json file
        storing lists the evolution of both as lists.

        NOTE: The history only contains time and dt for all occasions when this routine
        is called. This routine does neither guarantee completeness, nor duplicated.

        Parameters:
            path: Specified path for storing time and dt.

        """

        # Bookkeeping
        self.exported_times.append(
            int(self.time) if isinstance(self.time, np.integer) else float(self.time)
        )
        self.exported_dt.append(
            int(self.dt) if isinstance(self.dt, np.integer) else float(self.dt)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as out_file:
            json.dump({"time": self.exported_times, "dt": self.exported_dt}, out_file)

    def load_time_information(self, path: Path) -> None:
        """Keep track of history of time and time step size and store.

        Mirrors :meth:`write_time_information`.

        Parameters:
            path: Specified path for retrieving time and dt.

        """
        with path.open("r") as in_file:
            data = json.load(in_file)
            self.exported_times = data["time"]
            self.exported_dt = data["dt"]

    def set_time_and_dt_from_exported_steps(self, time_index: int = -1) -> None:
        """Load time and dt (time step) and cut off all later times and time steps.

        NOTE: This method by itself does NOT update the simulation state arrays.

        NOTE: It is implicitly assumed that the first entry of the history corresponds
        to the initial solution.

        Parameters:
            time_index: reference index addressing the currently stored history. By
                default, the latest accessible time and dt is retrieved.

        Raises:
            ValueError

        """
        if not hasattr(self, "exported_times") or not hasattr(self, "exported_dt"):
            raise ValueError(
                """The time manager does not hold information on previously used time
                and dt."""
            )

        self.time = self.exported_times[time_index]
        self.dt = self.exported_dt[time_index]

        self.exported_times = self.exported_times[:time_index]
        self.exported_dt = self.exported_dt[:time_index]
