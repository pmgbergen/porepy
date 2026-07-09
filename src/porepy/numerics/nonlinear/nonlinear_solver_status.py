from abc import ABC, abstractmethod
from dataclasses import dataclass

from porepy.numerics.nonlinear.convergence_check import ConvergenceStatusCollection


@dataclass
class NonlinearSolverStatus(ABC):
    """A status object used to indicate the NewtonSolver state. This
    is an enum of two allowed states: either success or failure. Each state can have
    data associated with it. `NonlinearSolverStatusConverged` and
    `NonlinearSolverStatusFailed` can be subclassed to (i) introduce specific cases of
    success or failure and (ii) associate additional data with these cases. The base
    class `NonlinearSolverStatus` should NOT be subclassed.

    """

    @abstractmethod
    def serialize(self) -> str:
        """Return the stable string representation used in stored statistics."""

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
    """The NewtonSolver solved the problem successfully."""

    convergence_statuses: ConvergenceStatusCollection
    divergence_statuses: ConvergenceStatusCollection
    num_nonlinear_iterations: int

    def serialize(self) -> str:
        return "successful"


@dataclass
class NonlinearSolverStatusFailed(NonlinearSolverStatus):
    """The NewtonSolver failed to solve the problem."""

    convergence_statuses: ConvergenceStatusCollection
    divergence_statuses: ConvergenceStatusCollection
    num_nonlinear_iterations: int

    def serialize(self) -> str:
        return "failed"
