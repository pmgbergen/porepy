"""Nonlinear solvers to be used with PorePy models, ModelRunner and TimeStepper.

Implemented classes:
    NonlinearSolverBase - abstract class describing the nonlinear solver interface.
    NonlinearSolverStatusConverged - status dataclass describing the nonlinear solver
        success.
    NonlinearSolverStatusFailed - status dataclass describing the nonlinear solver
        failure.
    NonlinearSolverBase - abstract class describing the nonlinear solver status.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import porepy as pp
from porepy.numerics.solvers.convergence_check import ConvergenceStatusCollection

__all__ = [
    "NonlinearSolverStatus",
    "NonlinearSolverStatusConverged",
    "NonlinearSolverStatusFailed",
    "NonlinearSolverBase",
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
        solver, or a nonlinear Richardson iteration.

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


class NonlinearSolverBase(ABC):
    """Abstract base class defining the interface for nonlinear solvers.

    Do not add method implementations or fields into it; this class should remain purely
    abstract.

    """

    @abstractmethod
    def solve(self, model: pp.PorePyModel) -> NonlinearSolverStatus:
        """Solve a nonlinear problem."""

    @abstractmethod
    def get_active_equations(
        self, model: pp.PorePyModel
    ) -> list[pp.ad.EquationOnDomain]:
        """A list of atomic equations this nonlinear solver operates on."""

    @abstractmethod
    def get_active_variables(self, model: pp.PorePyModel) -> list[pp.ad.Variable]:
        """A list of atomic variables this nonlinear solver operates on."""
