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
        time_index: int = 0,
    ) -> None:
        warn(message="", category=FutureWarning, stacklevel=2)
        self.schedule = np.array(schedule, dtype=float)
        self.time_init = float(self.schedule[0])
        self._time_final = float(self.schedule[-1])
        self.dt_init = float(dt_init)
        self.dt_min_max = dt_min_max
        self.iter_optimal_range = iter_optimal_range
        self.iter_relax_factors = iter_relax_factors
        self.recomp_factor = float(recomp_factor)
        self.is_constant = constant_dt
        self._time = float(self.time_init)
        self._dt = float(self.dt_init)
        self.time_index = time_index

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, val):
        raise ValueError(
            "model.time_manager is deprecated. Please, set model.time_data.time to "
            "change simulation time manually."
        )

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        raise ValueError(
            "model.time_manager is deprecated. Please, set model.time_data.dt to "
            "change simulation time manually."
        )

    @property
    def time_final(self):
        return self._time_final

    @time_final.setter
    def time_final(self, val):
        raise ValueError(
            "model.time_manager is deprecated. Please, control the simulation schedule "
            "with ModelRunner(time_stepper=TimeStepper(scheduler="
            "assemble_default_time_scheduler(schedule=[t_start, ..., t_end])))."
        )

    @property
    def exported_times(self):
        raise ValueError(
            "model.time_manager is deprecated. Please, access the simulation schedule "
            "through model_runner.time_stepper.scheduler."
        )
