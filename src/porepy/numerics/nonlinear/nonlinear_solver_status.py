from abc import ABC
from dataclasses import dataclass

from porepy.numerics.nonlinear.convergence_check import ConvergenceStatusCollection


@dataclass
class NonlinearSolverStatus(ABC):
    """A status object used to indicate the NewtonSolver (or LinearSolver) state."""

    def is_converged(self) -> bool:
        """Whether the nonlinear system is solved successfully."""
        return isinstance(self, NonlinearSolverStatusConverged)

    def is_failed(self) -> bool:
        """Whether the nonlinear system is not solved successfully."""
        return isinstance(self, NonlinearSolverStatusFailed)


@dataclass
class NonlinearSolverStatusConverged(NonlinearSolverStatus):
    """The NewtonSolver solved the problem successfully."""

    convergence_statuses: ConvergenceStatusCollection
    divergence_statuses: ConvergenceStatusCollection


@dataclass
class NonlinearSolverStatusFailed(NonlinearSolverStatus):
    """The NewtonSolver failed to solve the problem."""

    convergence_statuses: ConvergenceStatusCollection
    divergence_statuses: ConvergenceStatusCollection
