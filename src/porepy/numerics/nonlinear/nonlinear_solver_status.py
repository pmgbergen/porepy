from abc import ABC
from dataclasses import dataclass

from porepy.numerics.nonlinear.convergence_check import ConvergenceStatusCollection


@dataclass
class NonlinearSolverStatus(ABC):
    def is_converged(self) -> bool:
        return isinstance(self, NonlinearSolverStatusConverged)

    def is_failed(self) -> bool:
        return isinstance(self, NonlinearSolverStatusFailed)


@dataclass
class NonlinearSolverStatusConverged(NonlinearSolverStatus):
    convergence_statuses: ConvergenceStatusCollection
    divergence_statuses: ConvergenceStatusCollection


@dataclass
class NonlinearSolverStatusFailed(NonlinearSolverStatus):
    convergence_statuses: ConvergenceStatusCollection
    divergence_statuses: ConvergenceStatusCollection
