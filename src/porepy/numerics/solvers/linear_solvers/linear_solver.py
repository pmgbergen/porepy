"""Linear solvers to be used inside nonlinear sovlers.

Implemented classes:
    LinearSystem - container for an assembled matrix and right-hand side.
    LinearSolverBase - abstract class describing the linear solver interface.
    LinearSolverDirect - a direct solver with multiple supported backends.

"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import DEBUG, getLogger
from typing import Literal, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

import porepy as pp

try:
    import scikits.umfpack  # type: ignore

    IS_UMFPACK_INSTALLED = True
except ImportError:
    IS_UMFPACK_INSTALLED = False

try:
    from pypardiso import spsolve as pypardiso_spsolve  # type: ignore

    IS_PYPARDISO_INSTALLED = True
except ImportError:
    pypardiso_spsolve = lambda mat, rhs: rhs
    IS_PYPARDISO_INSTALLED = False

logger = getLogger(__name__)

DirectSolverBackends = Literal[
    "pypardiso",
    "umfpack",
    "scipy_sparse",
]

__all__ = [
    "LinearSolverStatus",
    "LinearSolverStatusSuccess",
    "LinearSolverStatusFailure",
    "LinearSolverStatusNotConverged",
    "LinearSystem",
    "LinearSolverBase",
    "LinearSolverDirect",
]


@dataclass
class LinearSolverStatus(ABC):
    """Base class for information produced by a linear solver invocation."""

    def is_success(self) -> bool:
        # Developer note: This breaks the OOP principle that the base class should not
        # know of its children, but we agreed on having these methods (is_success,
        # is_failure and is_not_converged) for convenience. One can think of
        # LinearSolverStatus as a closed enum of three cases (success, failure and not
        # converged), which in this case justifies this binding with child classes.
        """Whether the linear system is solved successfully."""
        return isinstance(self, LinearSolverStatusSuccess)

    def is_failure(self) -> bool:
        """Whether the linear system is not solved successfully."""
        return isinstance(self, LinearSolverStatusFailure)

    def is_not_converged(self) -> bool:
        """Whether the solver stopped short of its tolerance, but returned a finite
        solution that may be used as an inexact solution."""
        return isinstance(self, LinearSolverStatusNotConverged)


@dataclass
class LinearSolverStatusSuccess(LinearSolverStatus):
    """Status returned when a linear system was solved successfully."""

    solve_time: float
    """Wall-clock time spent solving the linear system, in seconds."""


@dataclass
class LinearSolverStatusFailure(LinearSolverStatus):
    """Status returned when a linear solver failed."""

    reason: str
    """Human-readable description of the failure."""


@dataclass
class LinearSolverStatusNotConverged(LinearSolverStatus):
    """Status returned when a solver stopped before reaching its tolerance, but without
    breaking down, so that the returned solution is finite and possibly useful.

    This is the third outcome of a linear solve, next to success and failure. It is
    needed by iterative solvers, which can stop in two very different ways:

    - The solution is unusable, e.g. it contains NaN or infinite values, or the
      preconditioner could not be set up. This is a failure
      (:class:`LinearSolverStatusFailure`); the caller should not use the solution.
    - The solver ran out of iterations, or its Krylov process broke down, before the
      requested residual reduction was reached. The solution is finite and usually
      reduces the residual, only by less than requested. This is the present status.

    Reporting the second case as a failure would discard a usable solution; reporting
    it as a success would hide that the tolerance was not met. The status therefore
    leaves the decision to the caller, which has the information to make it. In
    particular, for a Newton method, the solution is an inexact Newton step, which is
    a legitimate update as long as the nonlinear convergence and divergence criteria
    are checked afterwards (inexact Newton methods rely on this, see e.g. Eisenstat and
    Walker, SIAM J. Sci. Comput. 17(1), 1996).
    :class:`~porepy.numerics.solvers.NewtonSolver` applies such updates and continues
    iterating. This mirrors PETSc's nonlinear solvers, which
    always stop on a NaN or infinite linear solve, but can be allowed to continue past
    other linear-solver failures (option ``-snes_max_linear_solve_fail``).

    Implementations of :class:`LinearSolverBase` should return this status only when
    the returned solution is finite. Direct solvers do not use it: they either solve
    the system or fail.

    """

    reason: str
    """Human-readable description of why the solver stopped."""


@dataclass
class LinearSystem:
    """Container for an assembled matrix and right-hand side vector.

    The matrix may be released to reduce memory use, so callers should check that
    :attr:`matrix` is not ``None`` before using it.

    To actually deallocate the matrix and free the memory, you need to ensure that no
    other Python object has a reference to :attr:`matrix`. Having a reference to the
    ``LinearSystem`` object is fine. Use :meth:`release_matrix_reference` to remove the
    matrix from this container.

    """

    matrix: Optional[csr_matrix]
    rhs: np.ndarray
    equation_indexer: pp.ad.EquationIndexer
    """Indexer to map equations and their definition domains to the DoFs of the `matrix`
    and the `rhs`.

    """
    variable_indexer: pp.ad.VariableIndexer
    """Indexer to map variables and their definition domains to the DoFs of the
    `matrix`.

    """

    def release_matrix_reference(self) -> None:
        """Release this container's reference to the matrix.

        This method does not invoke garbage collection. The caller is responsible for
        triggering garbage collection if needed.

        """
        self.matrix = None


class LinearSolverBase(ABC):
    """Abstract base class defining the interface for linear solvers.

    Do not add method implementations or fields into it; this class should remain purely
    abstract.

    """

    def initialize_with_model(self, model: pp.PorePyModel) -> None:
        """Initialize model-dependent solver state.

        Solvers without model-dependent state may use this default no-op implementation.

        Note: This is needed in practice to construct a dof manager from a model in
        the iterative solver. Due to the foreseen changes (introducing tags and indexers
        on porepy side), this api might change.

        Parameters:
            model: Model whose linearized systems will be solved.

        """

    @abstractmethod
    def solve_linear_system(
        self, linear_system: LinearSystem
    ) -> tuple[np.ndarray, LinearSolverStatus]:
        """Solve an assembled linear system.

        Parameters:
            linear_system: System containing the matrix and right-hand side vector.

        Returns:
            The solution vector and a status describing the solver outcome.

        """


class LinearSolverDirect(LinearSolverBase):
    """Direct linear solver class.

    Parameters:
        backend: String specifying a direct linear solver implementation. "pypardiso"
            (default) and "umfpack" require additional dependencies to be installed,
            "pip install pypardiso" and "pip install scikit-umfpack", respectively.
            These backends can improve linear solver performance. If these libraries are
            not installed, falls back to the scikit implementation.

    """

    def __init__(
        self,
        backend: DirectSolverBackends = "pypardiso",
    ) -> None:
        if backend == "pypardiso" and not IS_PYPARDISO_INSTALLED:
            logger.debug(
                "PyPardiso could not be imported, falling back on 'umfpack' backend."
            )
            backend = "umfpack"
        if backend == "umfpack" and not IS_UMFPACK_INSTALLED:
            logger.debug(
                "scikits.umfpack could not be imported, falling back on 'scipy_sparse' "
                "backend"
            )
            backend = "scipy_sparse"
        self.backend: DirectSolverBackends = backend
        """String specifying a direct linear solver implementation."""

    def solve_linear_system(
        self, linear_system: LinearSystem
    ) -> tuple[np.ndarray, LinearSolverStatus]:
        """Solve linear system with a direct solver.

        Parameters:
            linear_system: System containing the matrix and right-hand side vector.

        Returns:
            The solution vector and a status describing the solver outcome.

        """
        t_0 = time.time()
        mat = linear_system.matrix
        rhs = linear_system.rhs
        if mat is None:
            raise ValueError("Cannot solve a linear system whose matrix was released.")

        # Log debugging statistics. Can be expensive for large matrices, so computing
        # only if needed.
        if logger.isEnabledFor(DEBUG):
            abs_mat = abs(mat)
            row_sums = np.sum(abs_mat, axis=1)
            logger.debug(f"Max element in A {np.max(abs_mat):.2e}")
            logger.debug(
                f"Max {np.max(row_sums):.2e} and min {np.min(row_sums):.2e} A sum."
            )

        if self.backend not in ["pypardiso", "umfpack", "scipy_sparse"]:
            raise ValueError(f"Unknown linear solver backend: {self.backend}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                if self.backend == "pypardiso":
                    assert IS_PYPARDISO_INSTALLED
                    x = pypardiso_spsolve(mat, rhs)

                elif self.backend == "umfpack":
                    assert IS_UMFPACK_INSTALLED
                    # Following may be needed:
                    # A.indices = A.indices.astype(np.int64)
                    # A.indptr = A.indptr.astype(np.int64)
                    x = spsolve(mat, rhs, use_umfpack=True)
                else:
                    x = spsolve(mat, rhs, use_umfpack=False)
        except Exception as e:
            logger.exception(e)
            return np.full_like(rhs, np.nan), LinearSolverStatusFailure(
                reason=f"{type(e).__name__}: {e}"
            )

        solve_time = time.time() - t_0
        logger.info(f"Solved linear system in {solve_time:.2e} seconds.")
        return np.atleast_1d(x), LinearSolverStatusSuccess(solve_time=solve_time)
