from abc import ABC, abstractmethod
from logging import getLogger
from typing import cast

import numpy as np

import porepy as pp

__all__ = [
    "TimeStepConstraint",
    "TargetNonlinearIterations",
    "CourantTimeStepConstraint",
]

logger = getLogger(__name__)


class TimeStepConstraint(ABC):
    @abstractmethod
    def suggest_dt(self, dt: float, context: dict) -> float:
        pass


class TargetNonlinearIterations(TimeStepConstraint):
    def __init__(
        self,
        dt_min: float,
        iter_min: int = 4,
        iter_max: int = 7,
        increase_factor: float = 1.3,
        decrease_factor: float = 0.7,
        retry_factor: float = 0.5,
        t_snap: float = 1e-6,
    ) -> None:
        if iter_min > iter_max:
            raise ValueError(
                f"Incorrect optimal iteration range: [{iter_min, {iter_max}}]."
            )
        self.dt_min: float = dt_min
        self.iter_min = iter_min
        self.iter_max = iter_max
        if (
            not (increase_factor >= 1)
            or not (0 < decrease_factor <= 1)
            or not (0 < retry_factor < 1)
        ):
            raise ValueError(
                f"Incorrect adjustment factors: {increase_factor = }, "
                f"{decrease_factor = }, {retry_factor = }."
            )
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.retry_factor = retry_factor
        self.t_snap: float = t_snap

    def suggest_dt(self, dt: float, context: dict) -> float:
        status: pp.solvers.NonlinearSolverStatus | None = context.get(
            "nonlinear_solver_status", None
        )
        if status is None:
            # TODO YZ: Should warn
            assert False

        if status.is_converged():
            num_iter = status.number_of_iterations()
            if num_iter < self.iter_min:
                return dt * self.increase_factor
            elif num_iter > self.iter_max:
                # Decrease dt, but not below dt_min.
                return max(dt * self.decrease_factor, self.dt_min)
            else:
                return dt
        else:
            if abs(dt - self.dt_min) < self.t_snap:
                return dt * self.retry_factor
            return max(dt * self.retry_factor, self.dt_min)


class CourantTimeStepConstraint(TimeStepConstraint):
    def __init__(self, target_cfl: float = 1.0, tol: float = 1e-10) -> None:
        self.target_cfl = target_cfl
        self.tol = tol

    def suggest_dt(self, dt: float, context: dict) -> float:
        model = cast(pp.PorePyModel | None, context.get("model", None))
        if model is None:
            # TODO YZ: Should warn
            assert False
        dt = float("inf")
        for subdomain in model.mdg.subdomains():
            v = np.max(model.equation_system.evaluate(model.darcy_flux([subdomain])))
            if v < self.tol:
                continue
            x = subdomain.cell_diameters(cell_wise=False, func=np.max).min()
            dt = min(dt, self.target_cfl * x / v)
        return dt