from abc import ABC, abstractmethod
from logging import DEBUG, getLogger
import time
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

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


class LinearSolverBase(ABC):
    """Abstract base class defining the interface for linear solvers.

    Do not add method implementations or fields into it; this class should remain purely
    abstract.

    """

    @abstractmethod
    def solve_linear_system(self, mat: csr_matrix, rhs: np.ndarray) -> np.ndarray:
        """Solver a linear system defined by a matrix `mat` and a right-hand side vector
        `rhs`.

        Returns:
            np.ndarray: Solution vector.

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

    def solve_linear_system(self, mat: csr_matrix, rhs: np.ndarray) -> np.ndarray:
        """Solve linear system with a direct solver.

        Returns:
            np.ndarray: Solution vector.

        """
        t_0 = time.time()

        # Log debugging statistics. Can be expensive for large matrices, so computing
        # only if needed.
        if logger.isEnabledFor(DEBUG):
            abs_mat = abs(mat)
            row_sums = np.sum(abs_mat, axis=1)
            logger.debug(f"Max element in A {np.max(abs_mat):.2e}")
            logger.debug(
                f"Max {np.max(row_sums):.2e} and min {np.min(row_sums):.2e} A sum."
            )

        if self.backend == "pypardiso":
            assert IS_PYPARDISO_INSTALLED
            x = pypardiso_spsolve(mat, rhs)

        elif self.backend == "umfpack":
            assert IS_UMFPACK_INSTALLED
            # Following may be needed:
            # A.indices = A.indices.astype(np.int64)
            # A.indptr = A.indptr.astype(np.int64)
            x = spsolve(mat, rhs, use_umfpack=True)
        elif self.backend == "scipy_sparse":
            x = spsolve(mat, rhs, use_umfpack=False)
        else:
            raise ValueError(f"Unknown linear solver backend: {self.backend}")

        x = np.atleast_1d(x)

        logger.info(f"Solved linear system in {time.time() - t_0:.2e} seconds.")
        return x
