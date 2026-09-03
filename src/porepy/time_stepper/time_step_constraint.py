"""Module defines constraints that adjust the simulation time step as a feedback to
simulation behavior. Provides the base class `TimeStepConstraint` and two
implementations:
- TargetNonlinearIterations
- CourantTimeStepConstraint


"""

from abc import ABC, abstractmethod
from typing import cast

import numpy as np

import porepy as pp

__all__ = [
    "TimeStepConstraint",
    "TargetNonlinearIterations",
    "CourantTimeStepConstraint",
]


class TimeStepConstraint(ABC):
    @abstractmethod
    def suggest_dt(self, dt: float, context: dict) -> float:
        """The constraint suggests the maximum dt, permitted for the next time step.

        In other words, for the returned `dt_new` the permitted time step range is
        `(0, dt_new]`.

        Parameters:
            dt: The time step magnitude that was used for the current time step attempt.
            context: Simulation context to make the adjustment based on. Implementations
                define what they expect to see in this dictionary.

        Returns:
            `dt_new`.

        """


class TargetNonlinearIterations(TimeStepConstraint):
    """Increases / decreases dt if the number of nonlinear solver iterations if below
    `iter_min` / above `iter_max`, respectively.

    Expects "nonlinear_solver_status" (:class:`pp.solvers.NonlinearSolverStatus`) in
    context.

    Parameters:
        iter_min: Left bound of the nonlinear solver's desired iterations range.
        iter_max: Right bound of the nonlinear solver's desired iterations range.
        increase_factor: How much to increase dt if iterations are below `iter_min`.
        decrease_factor: How much to decrease dt if iterations are above `iter_max`.
        retry_factory: How much to decrease dt if the nonlinear solver failed.
        dt_min: Sets the dt point that is guaranteed to try when decreasing dt. If
            `dt == dt_min`, will decrease dt below `dt_min`. The practical use case is
            during repeating failures, where it will ensure that it will attempt to make
            the time step with `dt_min`, before stopping the simulation with failure. If
            the value is not specified (default), does not affect.
        t_snap: Snapping time. If time difference is below it, treats it as zero.

    """

    def __init__(
        self,
        iter_min: int = 4,
        iter_max: int = 7,
        increase_factor: float = 1.3,
        decrease_factor: float = 0.7,
        retry_factor: float = 0.5,
        dt_min: float = 0,
        t_snap: float = 1e-6,
    ) -> None:
        if iter_min > iter_max:
            raise ValueError(
                f"Incorrect optimal iteration range: [{iter_min, {iter_max}}]."
            )
        self.dt_min: float = dt_min
        """Sets the dt point that is guaranteed to try when decreasing dt. If
        `dt == dt_min`, will decrease dt below `dt_min`. The practical use case is
        during repeating failures, where it will ensure that it will attempt to make
        the time step with `dt_min`, before stopping the simulation with failure. If
        the value is not specified (default), does not affect.

        """
        self.iter_min = iter_min
        """Left bound of the nonlinear solver's desired iterations range."""
        self.iter_max = iter_max
        """Right bound of the nonlinear solver's desired iterations range."""
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
        """How much to increase dt if iterations are below `iter_min`."""
        self.decrease_factor = decrease_factor
        """How much to decrease dt if iterations are above `iter_max`."""
        self.retry_factor = retry_factor
        """How much to decrease dt if the nonlinear solver failed."""
        self.t_snap: float = t_snap
        """Snapping time. If time difference is below it, treats it as zero."""

    def suggest_dt(self, dt: float, context: dict) -> float:
        status = context.get("nonlinear_solver_status")
        if not isinstance(status, pp.solvers.NonlinearSolverStatus):
            raise ValueError(
                "TargetNonlinearIterations requires 'nonlinear_solver_status' in "
                "context."
            )

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
    """Adjust dt to follow the provided target CFL = max(v) * dt / min(x), where v is
    the fluid velocity and x is the cell diameter.

    Expects "model" (:class:`pp.PorePyModel`) in context.

    Parameters:
        target_cfl: Target dimensionless value.
        atol: Velosity tolerance, below treated as zero.

    """

    def __init__(self, target_cfl: float = 1.0, atol: float = 1e-10) -> None:
        self.target_cfl = target_cfl
        """Target dimensionless value."""
        self.atol = atol
        """Velosity tolerance, below treated as zero."""

    def suggest_dt(self, dt: float, context: dict) -> float:
        model = cast(pp.PorePyModel | None, context.get("model", None))
        if not isinstance(model, pp.SolutionStrategy):
            raise ValueError("CourantTimeStepConstraint requires 'model' in context.")

        dt = float("inf")
        for subdomain in model.mdg.subdomains():
            v = np.max(model.equation_system.evaluate(model.darcy_flux([subdomain])))
            if v < self.atol:
                # Velosity is zero in this subdomain, not applying constraint.
                continue
            x = subdomain.cell_diameters(cell_wise=False, func=np.max).min()
            dt = min(dt, self.target_cfl * x / v)
        return dt
