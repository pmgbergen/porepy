from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeVar
from time import time

import porepy as pp
import numpy as np

from porepy.numerics.ad.equation_system import EquationIndexer, VariableIndexer
from porepy.numerics.linalg.matrix_operations import invert_permuted_block_diag_matrix
from porepy.numerics.solvers.linear_solvers.linear_solver import (
    LinearSolverBase,
    LinearSolverDirect,
    LinearSolverStatus,
    LinearSolverStatusFailure,
    LinearSolverStatusSuccess,
    LinearSystem,
)

__all__ = [
    "SchurComplementReductionStatusSuccess",
    "SchurComplementReductionStatusFailure",
    "SchurComplementReductionLinearSolver",
]


@dataclass
class SchurComplementReductionStatusSuccess(LinearSolverStatusSuccess):
    primary_solver_status: LinearSolverStatus


@dataclass
class SchurComplementReductionStatusFailure(LinearSolverStatusFailure):
    primary_solver_status: LinearSolverStatus


_T = TypeVar("_T")


class SchurComplementReductionLinearSolver(LinearSolverBase):
    def __init__(
        self,
        primary_equation_tags: list[pp.EquationTag],
        primary_variable_tags: list[pp.VariableTag],
        primary_linear_solver: Optional[LinearSolverBase] = None,
    ) -> None:
        self.primary_equation_tags = primary_equation_tags
        self.primary_variable_tags = primary_variable_tags
        if primary_linear_solver is None:
            primary_linear_solver = LinearSolverDirect()
        self.primary_linear_solver = primary_linear_solver

        self._is_initialized = False

    def initialize(self, linear_system: LinearSystem) -> None:
        def split_primary_secondary(
            primary_tags: list[pp.EquationTag] | list[pp.VariableTag],
            dofs_dict: dict[_T, np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray, dict[_T, np.ndarray], dict[_T, np.ndarray]]:
            primary_dofs = []
            secondary_dofs = []
            primary_dofs_map = {}
            primary_offset = 0
            secondary_dofs_map = {}
            secondary_offset = 0
            primary_tag_name_lookup = {tag.name: tag for tag in primary_tags}
            for eq, dofs in dofs_dict.items():
                tag = primary_tag_name_lookup.get(eq.name, None)
                if tag is not None and tag.defined_on.filter(eq.domain):
                    # This is primary
                    primary_dofs.append(dofs)
                    primary_dofs_map[eq] = np.arange(dofs.size) + primary_offset
                    primary_offset += dofs.size
                else:
                    # This is secondary
                    secondary_dofs.append(dofs)
                    secondary_dofs_map[eq] = np.arange(dofs.size) + secondary_offset
                    secondary_offset += dofs.size

            return (
                _concatenate_safe(primary_dofs),
                _concatenate_safe(secondary_dofs),
                primary_dofs_map,
                secondary_dofs_map,
            )

        (
            self.primary_eq_dofs,
            self.secondary_eq_dofs,
            primary_eq_map,
            secondary_eq_map,
        ) = split_primary_secondary(
            primary_tags=self.primary_equation_tags,
            dofs_dict=linear_system.equation_indexer.equation_dofs,
        )
        (
            self.primary_var_dofs,
            self.secondary_var_dofs,
            primary_var_map,
            secondary_var_map,
        ) = split_primary_secondary(
            primary_tags=self.primary_variable_tags,
            dofs_dict=linear_system.variable_indexer.variable_dofs,
        )
        self.primary_eq_indexer = EquationIndexer(equation_dofs=primary_eq_map)
        self.primary_var_indexer = VariableIndexer(variable_dofs=primary_var_map)

        secondary_eq_indexer = EquationIndexer(equation_dofs=secondary_eq_map)
        secondary_var_indexer = VariableIndexer(variable_dofs=secondary_var_map)
        self.secondary_eq_perm, self.secondary_var_perm = (
            rearrange_matrix_as_array_of_structures(
                eq_indexer=secondary_eq_indexer, var_indexer=secondary_var_indexer
            )
        )
        bs = len(secondary_eq_indexer.equation_dofs)
        if bs != 0:
            assert self.secondary_eq_dofs.size % bs == 0
            self.secondary_block_sizes = (
                np.ones(self.secondary_eq_dofs.size // bs, dtype=np.int64) * bs
            )
        else:
            self.secondary_block_sizes = np.empty(0)

        self._is_initialized = True

    def solve_linear_system(
        self, linear_system: LinearSystem
    ) -> tuple[np.ndarray, LinearSolverStatus]:
        """TODO YZ Solve an assembled linear system.

        Parameters:
            linear_system: System containing the matrix and right-hand side vector.

        Returns:
            The solution vector and a status describing the solver outcome.

        """
        t0 = time()
        assert linear_system.matrix is not None

        if not self._is_initialized:
            self.initialize(linear_system)

        # TODO YZ: Shortcut
        if len(self.secondary_eq_dofs) == 0 or len(self.secondary_var_dofs) == 0:
            return self.primary_linear_solver.solve_linear_system(linear_system)

        # || A_ss A_sp ||  <- eliminating
        # || A_ps A_pp ||

        A_p = linear_system.matrix[self.primary_eq_dofs, :]
        A_pp = A_p[:, self.primary_var_dofs]
        A_ps = A_p[:, self.secondary_var_dofs]
        del A_p
        A_s = linear_system.matrix[self.secondary_eq_dofs, :]
        A_sp = A_s[:, self.primary_var_dofs]
        A_ss = A_s[:, self.secondary_var_dofs]
        del A_s
        rhs_s = linear_system.rhs[self.secondary_eq_dofs]

        A_ss_inv = invert_permuted_block_diag_matrix(
            A_ss,
            row_permutation=self.secondary_eq_perm,
            col_permutation=self.secondary_var_perm,
            block_sizes=self.secondary_block_sizes,
        )

        A_ps_mul_Ass_inv = A_ps @ A_ss_inv

        S_pp = A_pp - A_ps_mul_Ass_inv @ A_sp
        rhs_p = linear_system.rhs[self.primary_eq_dofs] - A_ps_mul_Ass_inv @ rhs_s

        sol_p, primary_solver_status = self.primary_linear_solver.solve_linear_system(
            LinearSystem(
                matrix=S_pp,
                rhs=rhs_p,
                equation_indexer=self.primary_eq_indexer,
                variable_indexer=self.primary_var_indexer,
            )
        )

        sol_s = A_ss_inv @ (rhs_s - A_sp @ sol_p)
        solution = np.empty(len(sol_s) + len(sol_p))
        solution[self.primary_var_dofs] = sol_p
        solution[self.secondary_var_dofs] = sol_s
        if primary_solver_status.is_success():
            return solution, SchurComplementReductionStatusSuccess(
                solve_time=time() - t0,
                primary_solver_status=primary_solver_status,
            )
        else:
            return solution, SchurComplementReductionStatusFailure(
                primary_solver_status=primary_solver_status,
                reason="primary linear solver failed",
            )


def rearrange_matrix_as_array_of_structures(
    eq_indexer: EquationIndexer, var_indexer: VariableIndexer
):
    eq_dofs = list(eq_indexer.equation_dofs.values())
    var_dofs = list(var_indexer.variable_dofs.values())
    assert len(eq_dofs) == len(var_dofs)
    if len(eq_dofs) == 0:
        return np.empty(0), np.empty(0)
    return np.stack(eq_dofs).ravel(order="F"), np.stack(var_dofs).ravel(order="F")


def _concatenate_safe(arrays: list[np.ndarray]) -> np.ndarray:
    if len(arrays) > 0:
        return np.concatenate(arrays)
    return np.empty(0)
