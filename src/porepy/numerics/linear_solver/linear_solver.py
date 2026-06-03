"""This module contains classes that describe the components of the PETSc KSP and PC.
These classes do not produce PETSc options by themselves, they instead generate a dict
of PETSc options, and a dict of instruction, used to assemble PETSc objects in
`options_parser.py`.

This module also defines the default linear solver configurations for PorePy models.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Literal, Optional, Sequence
import numpy as np


from porepy.numerics.linear_solver.dof_manager import DofManager
from porepy.numerics.linear_solver.equation_variable_groups import EquationVariableGroup
from porepy.numerics.linear_solver.block_linear_system import BlockLinearSystem
from pp_solvers.transformations import LinearSystemTransformation
from scipy import sparse as sps

from logging import getLogger

logger = getLogger(__name__)


class LinearSolverFactory(ABC):
    @abstractmethod
    def build(
        self,
        block_linear_system: BlockLinearSystem,
        dof_manager: DofManager,
        user_options: dict,
    ):
        pass


class DirectSolver(LinearSolverFactory):
    def __init__(
        self, backend: Literal["scipy_sparse", "pypardiso", "umfpack"]
    ) -> None:
        self.backend: Literal["scipy_sparse", "pypardiso", "umfpack"] = backend

    def build(
        self,
        block_linear_system: BlockLinearSystem,
        dof_manager: DofManager,
        user_options: dict,
    ):
        def solve_linear_system(b: np.ndarray) -> np.ndarray:
            A = block_linear_system.mat
            logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
            logger.debug(
                f"""Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min
                {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."""
            )

            solver = self.backend
            if solver == "pypardiso":
                # This is the default option which is invoked unless explicitly overridden
                # by the user. We need to check if the pypardiso package is available.
                try:
                    from pypardiso import spsolve as sparse_solver  # type: ignore
                except ImportError:
                    # Fall back on the standard scipy sparse solver.
                    sparse_solver = sps.linalg.spsolve
                    logger.warning(
                        """PyPardiso could not be imported,
                        falling back on scipy.sparse.linalg.spsolve"""
                    )
                x = sparse_solver(A, b)
            elif solver == "umfpack":
                # Following may be needed:
                # A.indices = A.indices.astype(np.int64)
                # A.indptr = A.indptr.astype(np.int64)
                x = sps.linalg.spsolve(A, b, use_umfpack=True)
            elif solver == "scipy_sparse":
                x = sps.linalg.spsolve(A, b)
            else:
                raise ValueError(
                    f"AbstractModel does not know how to apply the linear solver {solver}"
                )

            x = np.atleast_1d(x)

            return x

        return solve_linear_system


@dataclass
class LinearSolverConfiguration:
    solver: LinearSolverAlgorithmConfiguration
    groups: list[EquationVariableGroup]
    transformations: list[LinearSystemTransformation] = field(
        default_factory=lambda: []
    )
