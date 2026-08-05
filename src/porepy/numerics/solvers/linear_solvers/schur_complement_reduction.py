from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import time
from typing import Collection, Optional, Sequence

import numpy as np

import porepy as pp
from porepy.numerics.linalg.matrix_operations import invert_permuted_block_diag_matrix
from porepy.numerics.solvers.equation_variable_tags import EquationTag, VariableTag
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


class SchurComplementReductionLinearSolver(LinearSolverBase):
    """A linear solver for a linear system with the following block structure:
    ```
    | A_ss A_sp |   | x_s |   | b_s |
    | A_ps A_pp | * | x_p | = | b_p |,
    ```
    where subscripts 'p' and 's' denote primary and secondary blocks, respectively. The
    primary block is defined by parameters `primary_equation_tags` and
    `primary_variable_tags`. The secondary block is a complement to it.

    The submatrix `A_ss` must be invertible, and computing its inverse must be cheap
    computationally, e.g., it should be block diagonal. Then, the Schur complement
    of this system is assembled:
    ```
    S_pp = A_pp - A_ps * (A_ss)^-1 * A_sp.
    ```
    The solution algorithm involves two steps:
    1. Solving the linear system with respect to `x_p` with the `primary_linear_solver`:
    ```
    S_pp * x_p = b_p - A_ps * (A_ss)^-1 * b_s
    ```
    2. Computing the secondary solution block:
    ```
    x_s = (A_ss)^-1 * (b_s - A_sp * x_p)
    ```

    The implementation assumes that the shape and arrangement of the block matrix won't
    change between subsequent linear systems.

    Parameters:
        primary_equation_tags: Tags that define primary equations (rows).
        primary_variable_tags: Tags that define primary variables (columns).
        primary_linear_solver: A linear solver for the linear system based on `S_pp`. If
            None (default), a direct solver is applied.

    """

    def __init__(
        self,
        primary_equation_tags: list[EquationTag],
        primary_variable_tags: list[VariableTag],
        primary_linear_solver: Optional[LinearSolverBase] = None,
    ) -> None:
        self.primary_equation_tags = primary_equation_tags
        """Tags that define primary equations (rows)."""
        self.primary_variable_tags = primary_variable_tags
        """Tags that define primary variables (columns)."""
        if primary_linear_solver is None:
            primary_linear_solver = LinearSolverDirect()
        self.primary_linear_solver = primary_linear_solver
        """A linear solver for the linear system based on `S_pp`."""

        self._data: _SchurComplementReductionData | None = None
        """The index arrays needed for the algorithm to operate. Initialized lazily in
        :meth:`_initialize_data` the first time :meth:`solve_linear_system` is invoked.

        It assumes that the arrangement in the linear system does not change between
        iterations.

        """

    def _initialize_data(
        self, linear_system: LinearSystem
    ) -> _SchurComplementReductionData:
        """Construct index arrays based on the linear system indexers."""
        primary_eqs, secondary_eqs = _filter_by_tags(
            all_operators=linear_system.equation_indexer.indices,
            tags=self.primary_equation_tags,
        )
        primary_vars, secondary_vars = _filter_by_tags(
            all_operators=linear_system.variable_indexer.indices,
            tags=self.primary_variable_tags,
        )

        eq_indexer = linear_system.equation_indexer
        var_indexer = linear_system.variable_indexer

        secondary_eq_perm, secondary_var_perm, block_sizes = (
            rearrange_matrix_as_array_of_structures(
                eq_indexer=eq_indexer.construct_restricted_indexer(secondary_eqs),
                var_indexer=var_indexer.construct_restricted_indexer(secondary_vars),
            )
        )

        return _SchurComplementReductionData(
            primary_eq_indexer=eq_indexer.construct_restricted_indexer(primary_eqs),
            primary_var_indexer=var_indexer.construct_restricted_indexer(primary_vars),
            primary_eq_dofs=eq_indexer.projection_indices(primary_eqs),
            primary_var_dofs=var_indexer.projection_indices(primary_vars),
            secondary_eq_dofs=eq_indexer.projection_indices(secondary_eqs),
            secondary_var_dofs=var_indexer.projection_indices(secondary_vars),
            secondary_eq_perm=secondary_eq_perm,
            secondary_var_perm=secondary_var_perm,
            secondary_block_sizes=block_sizes,
        )

    def initialize_with_model(self, model: pp.PorePyModel) -> None:
        """SchurComplementReductionLinearSolver does not need initialization by itself,
        but the inner linear solver might need it."""
        return self.primary_linear_solver.initialize_with_model(model)

    def solve_linear_system(
        self, linear_system: LinearSystem
    ) -> tuple[np.ndarray, LinearSolverStatus]:
        """Solve an assembled linear system applying the Schur complement factorization.

        Parameters:
            linear_system: System containing the matrix and right-hand side vector.

        Returns:
            The solution vector and a status describing the solver outcome.

        """
        t0 = time()
        assert linear_system.matrix is not None, "Matrix should be provided."

        # Lazy initialization.
        if self._data is None:
            self._data = self._initialize_data(linear_system)
        data = self._data

        # Shortcut if the secondary block is empty.
        if len(data.secondary_eq_dofs) == 0 or len(data.secondary_var_dofs) == 0:
            solution, status = self.primary_linear_solver.solve_linear_system(
                linear_system
            )
            return solution, _wrap_primary_solver_status(status, solve_time=time() - t0)

        # Slice the following submatrices and right-hand side vectors:
        # | A_ss A_sp |  | rhs_s |
        # | A_ps A_pp |, | rhs_p |
        A_p = linear_system.matrix[data.primary_eq_dofs, :]
        A_pp = A_p[:, data.primary_var_dofs]
        A_ps = A_p[:, data.secondary_var_dofs]
        del A_p
        A_s = linear_system.matrix[data.secondary_eq_dofs, :]
        A_sp = A_s[:, data.primary_var_dofs]
        A_ss = A_s[:, data.secondary_var_dofs]
        del A_s
        rhs_s = linear_system.rhs[data.secondary_eq_dofs]

        # Compute (A_ss)^-1. This inverter is hard-coded for now.
        A_ss_inv = invert_permuted_block_diag_matrix(
            A_ss,
            row_permutation=data.secondary_eq_perm,
            col_permutation=data.secondary_var_perm,
            block_sizes=data.secondary_block_sizes,
        )

        # A_sp * (A_ss)^-1, pre-computed because we will need it multiple times.
        A_ps_mul_Ass_inv = A_ps @ A_ss_inv
        # S_pp = A_pp - A_ps * (A_ss)^-1 * A_sp.
        S_pp = A_pp - A_ps_mul_Ass_inv @ A_sp
        # Modified right-hand side: rhs_p - A_sp * (A_ss)^-1 * rhs_s.
        rhs_p = linear_system.rhs[data.primary_eq_dofs] - A_ps_mul_Ass_inv @ rhs_s

        # Solve for the primary block solution.
        sol_p, status = self.primary_linear_solver.solve_linear_system(
            LinearSystem(
                matrix=S_pp,
                rhs=rhs_p,
                equation_indexer=data.primary_eq_indexer,
                variable_indexer=data.primary_var_indexer,
            )
        )

        # Compute the secondary block solution.
        sol_s = A_ss_inv @ (rhs_s - A_sp @ sol_p)

        # Concatenate the primary and the secondary solution blocks.
        solution = np.empty(len(sol_s) + len(sol_p), dtype=sol_s.dtype)
        solution[data.primary_var_dofs] = sol_p
        solution[data.secondary_var_dofs] = sol_s

        return solution, _wrap_primary_solver_status(status, solve_time=time() - t0)


def rearrange_matrix_as_array_of_structures(
    eq_indexer: pp.ad.EquationIndexer, var_indexer: pp.ad.VariableIndexer
):
    """Build permutation indices that rearrange the matrix in a block-diagonal form.

    The original matrix must be a block matrix:
    ```
    A_00 A_01 . A_0n
    A_10 A_11 . A_1n
     .    .   .  .
    A_n0 A_n1 . A_nn
    ```
    Each block A_ij corresponds to a single equation-variable pair defined on a single
    grid. All equation-variable pairs must be defined on the same set of grids.

    There is no requirement on how blocks A_ij should be arranged, e.g., different
    equation-variable pairs can live next to each other, and the order of grids can be
    different for different variables.

    Parameters:
        eq_indexer: Equation indexer representing the block rows of the matrix.
        var_indexer: Variable indexer representing the block columns of the matrix.

    Raises:
        ValueError: If not all equation-variable pairs are defined on the same set of
            grids.

    Returns:
        A tuple of 3 elements:
        - row_permutation: array to permute the rows of the matrix;
        - col_permutation: array to permute the columns of the matrix;
        - block_sizes: the sizes `n` of small `(n x n)` blocks on the diagonal of the
            permuted matrix.

    """
    unique_grids_equations = {eq.domain for eq in eq_indexer.indices}
    unique_grids_variables = {var.domain for var in var_indexer.indices}
    assert len(eq_indexer.indices) == len(var_indexer.indices)
    assert unique_grids_equations == unique_grids_variables

    eq_dofs_by_grid: dict[pp.GridLike, list[np.ndarray]] = {
        grid: [] for grid in unique_grids_equations
    }
    for eq, dofs in eq_indexer.indices.items():
        eq_dofs_by_grid[eq.domain].append(dofs)

    var_dofs_by_grid: dict[pp.GridLike, list[np.ndarray]] = {
        grid: [] for grid in unique_grids_variables
    }
    for var, dofs in var_indexer.indices.items():
        var_dofs_by_grid[var.domain].append(dofs)

    # Sanity check that assumes that the number of equations and variables is the same
    # on every grid. If it turns out that we define some equations only on certain
    # grids, this function can be extended. So far, no such case is known to YZ.
    bs = 0
    for list_of_dofs in eq_dofs_by_grid.values():
        if bs == 0:
            bs = len(list_of_dofs)
        if bs != len(list_of_dofs):
            raise ValueError(
                "All equations must be defined on the same set of domains."
            )
    for list_of_dofs in var_dofs_by_grid.values():
        if bs != len(list_of_dofs):
            raise ValueError(
                "All variables must be defined on the same set of domains."
            )

    if bs == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=int)

    assert var_indexer.size % bs == 0
    block_sizes = np.full(var_indexer.size // bs, bs, dtype=np.int64)

    eq_dofs_by_grid_rearranged = [
        np.stack(dofs).ravel(order="F") for dofs in eq_dofs_by_grid.values()
    ]
    var_dofs_by_grid_rearranged = [
        np.stack(dofs).ravel(order="F") for dofs in var_dofs_by_grid.values()
    ]
    return (
        _concatenate_safe(eq_dofs_by_grid_rearranged),
        _concatenate_safe(var_dofs_by_grid_rearranged),
        block_sizes,
    )


@dataclass
class _SchurComplementReductionData:
    """Data structures needed for the Schur complement reduction algorithm."""

    primary_eq_indexer: pp.ad.EquationIndexer
    """Equation indexer of the primary submatrix."""
    primary_var_indexer: pp.ad.VariableIndexer
    """Variable indexer of the primary submatrix."""

    primary_eq_dofs: np.ndarray
    """Indices that map the full equations vector to the primary equations vector."""
    primary_var_dofs: np.ndarray
    """Indices that map the full variables vector to the primary variables vector."""
    secondary_eq_dofs: np.ndarray
    """Indices that map the full equations vector to the secondary equations vector."""
    secondary_var_dofs: np.ndarray
    """Indices that map the full variables vector to the secondary variables vector."""

    secondary_eq_perm: np.ndarray
    """Equations (rows) permutation of the secondary block to the block-diagonal form.

    """
    secondary_var_perm: np.ndarray
    """Variables (columns) permutation of the secondary block to the block-diagonal
    form.

    """
    secondary_block_sizes: np.ndarray
    """Array with block sizes of each small dense square block on the permuted secondary
    matrix diagonal.

    """


def _concatenate_safe(arrays: list[np.ndarray]) -> np.ndarray:
    """Concatenate a list of arrays and return an empty array if the list is empty."""
    if len(arrays) > 0:
        return np.concatenate(arrays)
    return np.empty(0)


def _wrap_primary_solver_status(
    primary_solver_status: LinearSolverStatus, solve_time: float
) -> LinearSolverStatus:
    """Wrap a primary linear solver status with a Schur complement solver status."""
    if primary_solver_status.is_success():
        return SchurComplementReductionStatusSuccess(
            solve_time=solve_time,
            primary_solver_status=primary_solver_status,
        )
    else:
        return SchurComplementReductionStatusFailure(
            primary_solver_status=primary_solver_status,
            reason="primary linear solver failed",
        )


def _filter_by_tags[T: (pp.ad.EquationOnDomain, pp.ad.Variable)](
    all_operators: Collection[T], tags: Sequence[EquationTag | VariableTag]
) -> tuple[list[T], list[T]]:
    """Split atomic equations or variables into two groups: the one covered by `tags`
    and its complement.

    Parameters:
        all_operators: Atomic variables or equations to split into groups.

    Result:
        Two groups of atomic variables or equations.

    """
    # Organizing tags by eq/var names. Multiple tags with a single name are allowed.
    # E.g. pressure on wells and pressure not on wells.
    tags_by_name: dict[str, list[EquationTag | VariableTag]] = defaultdict(list)
    for tag in tags:
        tags_by_name[tag.name].append(tag)

    # Two groups.
    filtered_operators: list[T] = []
    not_filtered_operators: list[T] = []

    for operator in all_operators:
        tags_for_this_name = tags_by_name[operator.name]
        matching_tags = [
            tag for tag in tags_for_this_name if tag.defined_on.filter(operator.domain)
        ]
        if len(matching_tags) > 1:
            # Overlapping tags would place the operator in the filtered group more than
            # once.
            raise ValueError(f"Duplicated operators: [{operator}]")
        if len(matching_tags) == 1:
            filtered_operators.append(operator)
        else:
            not_filtered_operators.append(operator)

    return filtered_operators, not_filtered_operators
