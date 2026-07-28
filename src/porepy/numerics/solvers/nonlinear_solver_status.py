from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .convergence_check import ConvergenceStatusCollection

__all__ = [
    "NonlinearSolverStatus",
    "NonlinearSolverStatusConverged",
    "NonlinearSolverStatusFailed",
]


@dataclass
class NonlinearSolverStatus(ABC):
    """A status object used to indicate the nonlinear solver state.

    This is an enum of two allowed states: either success or failure. Each state can
    have data associated with it. `NonlinearSolverStatusConverged` and
    `NonlinearSolverStatusFailed` can be subclassed to (i) introduce specific cases of
    success or failure and (ii) associate additional data with these cases. The base
    class `NonlinearSolverStatus` should NOT be subclassed.

    """

    @abstractmethod
    def serialize(self) -> str:
        """Return the stable string representation used in stored statistics."""

    @abstractmethod
    def number_of_iterations(self) -> int:
        """Number of iterations of a nonlinear solver.

        This is a method and not a property, because different implementation can deduce
        it from their other data, so there is no need to duplicate it.

        It assumes that every nonlinear solver we ever encounter is iterative, and
        algorithms that rely on it, e.g., :class:`porepy.TimeStepper` and
        :class:`porepy.ModelRunner`, can always request the number of its iterations and
        adjust based on this information. It is important, however, that the meaning of
        a single iteration is different in Newton's method, or a sequential iterative
        solver, or a nonlinear Richardson iteration. YZ has an uneasy feeling about
        introducing this as a part of the API.

        """

    def is_converged(self) -> bool:
        """Whether the nonlinear system is solved successfully."""
        # Developer note: This breaks the OOP principle that the base class should not
        # know of its children, but we agreed on having these methods (is_success and
        # is_failure) for convenience. One can think of NonlinearSolverStatus as a
        # closed enum of two cases (success and failure), which in this case justifies
        # this binding with child classes.
        return isinstance(self, NonlinearSolverStatusConverged)

    def is_failed(self) -> bool:
        """Whether the nonlinear system is not solved successfully."""
        return isinstance(self, NonlinearSolverStatusFailed)


@dataclass
class NonlinearSolverStatusConverged(NonlinearSolverStatus):
    """The nonlinear solver solved the problem successfully."""

    convergence_statuses: ConvergenceStatusCollection
    divergence_statuses: ConvergenceStatusCollection

    def serialize(self) -> str:
        return "successful"


@dataclass
class NonlinearSolverStatusFailed(NonlinearSolverStatus):
    """The nonlinear failed to solve the problem."""

    convergence_statuses: ConvergenceStatusCollection
    divergence_statuses: ConvergenceStatusCollection

    def serialize(self) -> str:
        return "failed"
