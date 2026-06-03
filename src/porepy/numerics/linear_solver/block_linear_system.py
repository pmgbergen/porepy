"""This module contains the `BlockLinearSystem` class. It wraps a linear system (sparse
matrix and the rhs vector), where each degree of freedom corresponds to a particular
part of a problem. E.g., for a coupled poromechanics problem, DoFs 0 - 999 correspond to
the mass balance equation, and DoFs 1000 - 3999 correspond to the momentum balance
equation. This class provides convenient indexing to operate with these submatrices.

Internally, slicing of submatrices is handled by the `LinearSystemIndexer` class, which
stores all the DoFs information.

While PorePy is treated as the primary provided of block linear systems, the classes
here do not depend on it, and can be used without establishing a PorePy model.

"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import scipy.sparse
from numpy.typing import DTypeLike
from scipy.sparse import coo_matrix, csr_matrix, spmatrix

from porepy.numerics.linear_solver.plot_linear_system import (
    color_spy,
    matshow,
    plot_max,
    plot_vector,
)

__all__ = ["BlockLinearSystem", "LinearSystemIndexer"]


class LinearSystemIndexer:
    """This class bookkeeps the information about row/column degrees of freedom of
    groups of a block linear system.

    """

    def __init__(
        self,
        dofs_row: list[np.ndarray],
        dofs_col: list[np.ndarray],
        original_dofs_row: Optional[list[np.ndarray]] = None,
        original_dofs_col: Optional[list[np.ndarray]] = None,
        group_names_row: Optional[list[str]] = None,
        group_names_col: Optional[list[str]] = None,
        enabled_groups_row: Optional[list[int]] = None,
        enabled_groups_col: Optional[list[int]] = None,
    ) -> None:
        assert len(dofs_row) == len(dofs_col)
        self.dofs_row: list[np.ndarray] = dofs_row
        """List of row indices of a block linear systems. i-th element is an array of
        indices that points to the rows of a linear system, corresponding to i-th group.

        """
        self.dofs_col: list[np.ndarray] = dofs_col
        """List of column indices of a block linear systems. i-th element is an array of
        indices that points to the columns of a linear system, corresponding to i-th
        group.

        """

        if enabled_groups_row is None:
            enabled_groups_row = list(range(len(dofs_row)))
        self.enabled_groups_row: list[int] = enabled_groups_row
        """List of row groups that are enabled. A group is enabled by default, but can
        be disabled if we slice a submatrix without this group. Also indicates the order
        of row permutations.
 
        """
        if enabled_groups_col is None:
            enabled_groups_col = list(range(len(dofs_col)))
        self.enabled_groups_col: list[int] = enabled_groups_col
        """List of column groups that are enabled. A group is enabled by default, but
        can be disabled if we slice a submatrix without this group. Also indicates the
        order of column permutations.
 
        """
        if original_dofs_row is None:
            original_dofs_row = [x.copy() for x in dofs_row]
        self.original_dofs_row: list[np.ndarray] = original_dofs_row
        """Same as `dofs_row`, but this list does not change after slicing or
        permutations. Needed for reverse transformations to the original PorePy
        arrangement.
 
        """
        if original_dofs_col is None:
            original_dofs_col = [x.copy() for x in dofs_col]
        self.original_dofs_col: list[np.ndarray] = original_dofs_col
        """Same as `dofs_col`, but this list does not change after slicing or
        permutations. Needed for reverse transformations to the original PorePy
        arrangement.
 
        """

        if group_names_row is None:
            group_names_row = [str(i) for i in range(len(dofs_row))]
        self.group_names_row: list[str] = group_names_row
        """List of group names for the rows. They typically represent equation names in
        a multiphysics simulation, however, this information is stored here only for
        debugging purposes and does not affect the behavior of the class. These names
        are not necesserily the same as PorePy equation names. 

        """
        if group_names_col is None:
            group_names_col = [str(i) for i in range(len(dofs_col))]

        self.group_names_col: list[str] = group_names_col
        """List of group names for the columns. They typically represent variable names
        in a multiphysics simulation, however, this information is stored here only for
        debugging purposes and does not affect the behavior of the class. These names
        are not necesserily the same as PorePy variable names. 

        """

    def __getitem__(self, key: list | slice | tuple) -> LinearSystemIndexer:
        """Get a sub-indexer corresponding to the given group indices.

        Parameters:
            key: The key to index the matrix. See `BlockLinearSystem.__getitem__` for
            permissible formats.

        Raises:
            IndexError: If a disabled or out-of-bounds group is selected.

        Returns:
            A block linear system object containing the selected groups.

        """
        # Unifying and validating the passed key.
        key = self.correct_validate_getitem_key(key)
        groups_row, groups_col = key

        # Creating a new index for the sliced matrix, as it was likely permuted.
        new_dofs_row, new_dofs_col = self._make_permutation_after_slicing(key)

        # Return a new indexer object with the selected groups. Compared to
        # the current inexer, the new object potentially has a subset of enabled groups,
        # with a corresponding subset of local indices (dofs_row and dofs_col).
        return LinearSystemIndexer(
            dofs_row=new_dofs_row,
            dofs_col=new_dofs_col,
            original_dofs_row=self.original_dofs_row,  # unchanged
            original_dofs_col=self.original_dofs_col,  # unchanged
            group_names_col=self.group_names_col,  # unchanged
            group_names_row=self.group_names_row,  # unchanged
            enabled_groups_row=groups_row,
            enabled_groups_col=groups_col,
        )

    def get_dofs_of_groups(
        self, key: list | slice | tuple
    ) -> tuple[np.ndarray, np.ndarray]:
        """Builds indices that can be used to slice a submatrix, corresponding to the
        provided groups.

        Parameters:
            key: The groups of rows and columns.

        Returns:
            Two arrays, corresponding to row and column indices.

        """
        groups_row, groups_col = self.correct_validate_getitem_key(key)

        dofs_row = []
        for group in groups_row:
            dofs_row.append(self.dofs_row[group])

        dofs_col = []
        for group in groups_col:
            dofs_col.append(self.dofs_col[group])

        return concatenate_dof_indices(dofs_row), concatenate_dof_indices(dofs_col)

    def _make_permutation_after_slicing(
        self, key: tuple[list[int], list[int]]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Produces new indices that correspond to a new submatrix, generated by slicing
        in with the given key.

        Parameters:
            key: The groups of rows and columns. Does not validate input, so if key is
            passed in arbitrary format (e.g. from `__getitem__`), it must be formatted
            on the caller side by calling the `correct_validate_getitem_key` method.

        Returns:
            Two arrays, corresponding to row and column indices.

        """
        groups_row, groups_col = key

        dofs_row = [np.array([], dtype=int) for _ in self.dofs_row]
        counter = 0
        for group in groups_row:
            end = counter + len(self.dofs_row[group])
            dofs_row[group] = np.arange(counter, end)
            counter = end

        dofs_col = [np.array([], dtype=int) for _ in self.dofs_col]
        counter = 0
        for group in groups_col:
            end = counter + len(self.dofs_col[group])
            dofs_col[group] = np.arange(counter, end)
            counter = end

        return dofs_row, dofs_col

    def correct_validate_getitem_key(
        self, key: list | slice | tuple
    ) -> tuple[list[int], list[int]]:
        """Helper function to unify the format of the key, passed into __getitem__ and
        __setitem__. See `BlockLinearSystem.__getitem__` for permissible formats.

        """
        # Since the key is defined as a single argument (see __getitem__), passing
        # multiple arguments (e.g., both row and column indices) will be interpreted as
        # a tuple. If the key is a list or a slice, we will assign the same key to the
        # row and column indices.
        if isinstance(key, list):
            key = key, key
        if isinstance(key, slice):
            key = key, key

        # By now, we should have a tuple with two elements, corresponding to the row and
        # column block indices to be extracted.
        assert isinstance(key, tuple)
        assert len(key) == 2

        def correct_key(
            key_: slice | int, enabled_groups: list[int], total: int
        ) -> list[int]:
            # Convert slice or int to list of indices. Total is the maximum upper bound
            # of a slice, in case it is given on the form `1:` or similar.
            if isinstance(key_, slice):
                start = key_.start or 0
                stop = key_.stop or total
                step = key_.step or 1
                if step < 1:
                    raise NotImplementedError("Negative step is not supported.")
                result = [i for i in range(start, stop, step) if i in enabled_groups]
            else:
                try:
                    # Try to iterate over the key. If not successful (which means this
                    # is an int?), convert to a list.
                    iter(key_)  # type: ignore
                    result = key_
                except TypeError:
                    result = [key_]

            # Correct negative indices.
            result = [x if x >= 0 else total + x for x in result]

            # Validate indices in bounds.
            for x in result:
                if not (0 <= x < total):
                    raise IndexError(
                        f"Block matrix index out of bounds, expected 0 <= {x} < {total}"
                    )
                if not x in enabled_groups:
                    raise IndexError(f"Taking disabled group {x}")
            return result

        groups_row, groups_col = key
        # Convert the key to a list of indices.
        groups_row = correct_key(
            groups_row, enabled_groups=self.enabled_groups_row, total=len(self.dofs_row)
        )
        groups_col = correct_key(
            groups_col, enabled_groups=self.enabled_groups_col, total=len(self.dofs_col)
        )
        return groups_row, groups_col


class BlockLinearSystem:
    """Storage class for block linear systems with utility methods for indexing.

    The indexing of a block linear systems is done in terms of groups, e.g.
    `bmat[1, 2]` produces the sub-linear system (submatrix and rhs) that corresponds to
    row (equation) group 1 and column (variable) group 2. Indexing is 0-based.

    One group typically corresponds to a subproblem that should be treated by a single
    preconditioner inside a block-preconditioner framework.

    See `__getitem__` for all permissible indexing formats.

    """

    def __init__(
        self,
        mat: csr_matrix,
        rhs: np.ndarray,
        indexer: LinearSystemIndexer,
        validate_input: bool = True,
    ):
        self.mat: csr_matrix = mat
        """The matrix itself."""
        self.rhs: np.ndarray = rhs
        """The right-hand side vector."""
        self.indexer: LinearSystemIndexer = indexer
        """The indexer object that contains information about indices that correspond to
        different groups."""

        if validate_input:
            validate_block_matrix(self)

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the matrix."""
        return self.mat.shape

    def __repr__(self) -> str:
        rows = len(self.indexer.enabled_groups_row)
        cols = len(self.indexer.enabled_groups_col)
        return (
            f"BlockLinearSystem of shape {self.shape} with {self.mat.nnz} elements "
            f"with {rows}x{cols} enabled groups."
        )

    # MARK: Slicing

    def __getitem__(self, key: list | slice | tuple) -> BlockLinearSystem:
        """Get a block submatrix corresponding to the given group indices.

        The following indexing is supported:

        - `1, 2`: Get the submatrix corresponding to the single row group index 1 and
           column group index 2. Results in submatrix [J_12].
        - `[1, 2]`: Get the submatrix corresponding to multiple row group indices 1 and
           2 and column group indices 1 and 2. Results in the submatrix
           [[J_11, J_12], [J_21, J_22]].
        - `([1, 2], [3, 4])`: Get the groups corresponding to row group indices 1 and 2
           and column group indices 3 and 4. Results in the submatrix
           [[J_13, J_14], [J_23, J_24]].
        - `:, [1, 2]: Get all row groups and column groups 1 and 2. Results in the
           submatrix [[J_11, J_12], [J_21, J_22], ..., [J_m1, J_m2]], where m is the
           maximum row group index.
        - `[1, 2], :`: Get row groups 1 and 2 and all column groups. Results in the
           submatrix [[J_11, J_12, ..., J_1n], [J_21, J_22, ..., J_2n]], where n is the
           maximum column group index.
        - `[1, 2], 1:4`: Get row groups 1 and 2 and column blocks 1 to 3. Results in
           the submatrix [[J_11, J_12, J_13], [J_21, J_22, J_23]].

        Indices can be given by tuples as well as lists. The indexing is 0-based.

        Only enabled groups can be taken. That is, such expressions are not permitted:
        `A = bmat[1, 2]; B = A[2, 3]`. You cannot take a disabled group (2, 3) from a
        submatrix A, which has only group (1, 2) enabled.

        Parameters:
            key: The key to index the matrix. See above for permissible formats.

        Raises:
            IndexError: If a disabled or out-of-bounds group is selected.

        Returns:
            A block linear system object containing the selected groups.

        """
        # Preparing the indices for slicing.
        groups_dofs_row, groups_dofs_col = self.indexer.get_dofs_of_groups(key)

        # Slicing the matrix and the rhs.
        sliced_matrix = self.mat[groups_dofs_row][:, groups_dofs_col]
        rhs = self.rhs[groups_dofs_row]

        if not sliced_matrix.data.flags.owndata:
            # Slicing of a sparse matrix sometimes produces a view with considerable
            # higher memory consumption than the original matrix. Copy the data to
            # reclaim ownership and reduce memory consumption. We assume that the
            # incurred overhead is minor compared to the benefits in saving memory.
            sliced_matrix.data = sliced_matrix.data.copy()

        # Return a new block linear system object with the selected blocks. Compared to
        # the current object, the new object potentially has a subset of enabled groups,
        # with a corresponding subset of local indices (dofs_row and dofs_col).
        return BlockLinearSystem(
            mat=sliced_matrix,
            rhs=rhs,
            indexer=self.indexer[key],
            validate_input=False,
        )

    def __setitem__(
        self, key: list | slice | tuple, value: BlockLinearSystem | spmatrix
    ) -> None:
        """Updates the submatrix corresponding to the row and column groups passed in
        the `key` parameter with the `value`.

        See `__getitem__` for permissible formats of the key.

        Warning:
            This implementation is inefficient because at some point SciPy casts the
            assigned matrix to a dense matrix. Use it only for experimentation, and
            avoid it in performance-critical parts of the code. See methods
            `set_zeros`, `set_diagonal` for efficiency.

        Parameters:
            key: The key to index the matrix. See above for permissible formats.
            value: The value to set. This can be a BlockMatrixStorage object, or a
                sparse matrix.

        Raises:
            IndexError: If a disabled or out-of-bounds group is selected.

        """
        # Preparing the indices for slicing.
        key = self.indexer.correct_validate_getitem_key(key)
        dofs_row_for_slicing, dofs_col_for_slicing = self.indexer.get_dofs_of_groups(
            key
        )
        rows_expanded, cols_expanded = np.meshgrid(
            dofs_row_for_slicing,
            dofs_col_for_slicing,
            sparse=True,
            indexing="ij",
            copy=False,
        )
        # Updating the matrix slice.
        self.mat[rows_expanded, cols_expanded] = value

    def copy(self) -> BlockLinearSystem:
        """Deep-copies the underlying matrix and right-hand side and passes references
        to the indexes to blocks and groups (does not copy their memory).

        Returns:
            A new identical block matrix storage object.

        """
        res = self.empty_container()
        res.mat = self.mat.copy()
        res.rhs = self.rhs.copy()
        return res

    def empty_container(self) -> BlockLinearSystem:
        """Create container with the same structure as the current one, but with an
        empty matrix. Typically used for creating new linear systems with a similar
        structure.

        One exceptional usage - to cheaply create a `LinearSystemIndexer` object that
        corresponds to a submatrix, without performing expensive matrix slicing.

        """

        return BlockLinearSystem(
            mat=scipy.sparse.csr_matrix(self.mat.shape),
            rhs=np.zeros(self.mat.shape[0]),
            indexer=self.indexer,
            validate_input=False,
        )

    def permute_left_vector_to_original(self, vec: np.ndarray) -> np.ndarray:
        """The block linear system undergoes some permutations / slicing. This method
        permutes the left vector (right-hand side) to restore the original arrangement
        that was used when the `BlockLinearSystem` was originally created. If some
        information is lost due to slicing, these values are filled with zeros.

        """
        return self._permute_vec_to_original(vec, side="left")

    def permute_right_vector_to_original(self, vec: np.ndarray) -> np.ndarray:
        """The block linear system undergoes some permutations / slicing. This method
        permutes the right vector (solution) to restore the original arrangement that
        was used when the `BlockLinearSystem` was originally created. If some
        information is lost due to slicing, these values are filled with zeros.

        """
        return self._permute_vec_to_original(vec, side="right")

    def _permute_vec_to_original(
        self, vec: np.ndarray, side: Literal["left", "right"]
    ) -> np.ndarray:
        """See `permute_left_vector_to_original` and `permute_right_vector_to_original`
        methods.

        """
        if side == "left":
            enabled_groups = self.indexer.enabled_groups_row
            original_dofs = self.indexer.original_dofs_row
        elif side == "right":
            enabled_groups = self.indexer.enabled_groups_col
            original_dofs = self.indexer.original_dofs_col
        else:
            raise ValueError(f"{side = }")
        dofs = [original_dofs[i] for i in enabled_groups]
        result = np.zeros(sum(x.size for x in original_dofs))
        result[concatenate_dof_indices(dofs)] = vec
        return result

    # MARK: Matrix construction

    def set_zeros(
        self, group_row_idx: list[int] | int, group_col_idx: list[int] | int
    ) -> None:
        """Set the values in the given block rows and columns to zeros. Does not change
        the sparsity pattern, so this is much cheaper than doing it in the naive way."""
        key = self.indexer.correct_validate_getitem_key((group_row_idx, group_col_idx))
        dofs_row, dofs_col = self.indexer.get_dofs_of_groups(key)
        nonzero_idx = get_nonzero_indices(
            A=self.mat, row_indices=dofs_row, col_indices=dofs_col
        )
        self.mat.data[nonzero_idx] = 0

    def set_diagonal(
        self,
        groups: list[int] | int,
        values: np.ndarray | float,
        additive: bool = False,
    ) -> None:
        """Adds the values to the main diagonal of the given groups. This method avoids
        allocating a dense matrix.

        This is expected to work only with the groups on the main diagonal. If you take
        J[3, 4] and try to set diagonal, this method should raise an error.

        """
        key = self.indexer.correct_validate_getitem_key(groups)
        dofs_row, dofs_col = self.indexer.get_dofs_of_groups(key)
        try:
            len(values)
        except Exception:
            # This is a scalar.
            values = np.full(shape=dofs_row.shape, fill_value=values)
        assert dofs_row.size == dofs_col.size == values.size

        if not additive:
            values = values - csr_matrix(self.mat[dofs_row, dofs_col]).toarray().ravel()
        tmp = coo_matrix((values, (dofs_row, dofs_col)), shape=self.mat.shape).tocsr()
        self.mat += tmp

    # MARK: Visualization

    def color_spy(
        self,
        show=True,
        aspect: Literal["equal", "auto"] = "equal",
        marker=None,
        color=True,
        hatch=False,
        draw_marker=True,
        alpha=0.3,
    ):
        """Draws a sparse matrix stencil and colors the rows and columns to distinguish
        different groups.

        Parameters:
            show: Whether to call `plt.show()`.
            aspect: Passed to `plt.spy()`. "auto" can be useful if the aspect ratio is
                too high. Otherwise, "equal" is better.
            marker: Which marker to use for the matrix stencil. Default value changes
                based on the matrix size.
            color: Whether to color the background.
            hatch: Whether to hatch the background.
            draw_marker: If not, only colors / hatches the background.
            alpha: Background transparency.

        """
        color_spy(
            bmat=self,
            show=show,
            aspect=aspect,
            marker=marker,
            alpha=alpha,
            color=color,
            hatch=hatch,
            draw_marker=draw_marker,
        )

    def matshow(
        self,
        log=True,
        show=True,
        threshold: float = 1e-30,
        aspect: Literal["equal", "auto"] = "equal",
    ):
        """Displays the values of a sparse matrix. Converts it to a dense matrix, so
        should not be called for large matrices.

        Parameters:
            log: Whether to use the log scale.
            show: Whether to call `plt.show()`.
            threshold: Does not display the values with `abs(x) < threshold`.
            aspect: Passed to `plt.matshow()`. "auto" can be useful if the aspect ratio
            is too high. Otherwise, "equal" is better.

        """
        matshow(self.mat, log=log, show=show, threshold=threshold, aspect=aspect)

    def matshow_groups(self, log=True, show=True):
        """The combination of `matshow` and `color_spy`. Shows the values in the matrix
        and hatches the background to see different groups.

        Parameters:
            log: Whether to use the log scale.
            show: Whether to call `plt.show()`.

        """
        self.matshow(log=log, show=False)
        self.color_spy(show=show, color=False, hatch=True, draw_marker=False)

    def plot_max(self, annot=True, mean=False):
        """Displays a table, where each cell represents a matrix group. Useful to get
        the idea about the groups and which of them are not empty.

        Parameters:
            annot: Display the aggregated max / mean value for each cell.
            mean: Whether to show the mean value. Otherwise, shows the abs(max) value.

        """
        plot_max(bmat=self, annot=annot, mean=mean)

    def plot_solution(self, permuted_solution, log: bool = True):
        """Displays a solution vector and colors the background to see different groups.

        Parameters:
            permuted_solution: The solution vector, in the same arrangement as this
                block linear system object.
            log: Whether to use the log scale.

        """
        plot_vector(bmat=self, vec=permuted_solution, side="right", log=log)

    def plot_rhs(self, permuted_rhs: Optional[np.ndarray] = None, log: bool = True):
        """Displays the right-hand side (RHS) vector and colors the background to show
        different groups.

        Parameters:
            permuted_rhs: The RHS vector to display, arranged in the same order as this
                block linear system.
                If None, uses the system's stored RHS vector.
            log: Whether to use the log scale for visualization.

        """
        if permuted_rhs is None:
            permuted_rhs = self.rhs
        plot_vector(bmat=self, vec=permuted_rhs, side="left", log=log)


def validate_block_matrix(bmat: BlockLinearSystem) -> None:
    """Checks that the block matrix is initialized correctly.

    Raises:
        ValueError: If disabled groups have nonzero size.
        ValueError: If the groups (submatrices) on the diagonal are not square.
        ValueError: If the number of groups is different from the number of group names.
        ValueError: If the matrix shape is inconsistent with the indexer.
        ValueError: If the matrix shape is inconsistent with the rhs shape.
    """
    # Checking that the disabled groups are empty and the diagonal enabled groups are
    # square matrices. First, preparing the data structures for this.
    indexer = bmat.indexer
    is_enabled_group_row = [False] * len(indexer.dofs_row)
    for i in indexer.enabled_groups_row:
        is_enabled_group_row[i] = True
    is_enabled_group_col = [False] * len(indexer.dofs_col)
    for i in indexer.enabled_groups_col:
        is_enabled_group_col[i] = True
    # Then, perform the check.
    for i in range(len(indexer.dofs_row)):
        if not is_enabled_group_row[i] and len(indexer.dofs_row[i]) != 0:
            raise ValueError(f"Disabled row group {i} has nonzero size.")
        if not is_enabled_group_col[i] and len(indexer.dofs_col[i]) != 0:
            raise ValueError(f"Disabled column group {i} has nonzero size.")
        if (
            is_enabled_group_row[i]
            and is_enabled_group_col[i]
            and (len(indexer.dofs_row[i]) != len(indexer.dofs_col[i]))
        ):
            raise ValueError(f"Diagonal group ({i},{i}) is not a square matrix.")

    # Making sure that the number of groups is consistent.
    if (len(indexer.group_names_row) != len(indexer.dofs_row)) or (
        len(indexer.group_names_col) != len(indexer.dofs_col)
    ):
        raise ValueError(
            "The number of groups should be the same as the number of group names."
        )

    # Making sure that the sum of indexed shapes matches the original matrix shape.
    shape_sum_row = 0
    shape_sum_col = 0
    expected_shape = bmat.shape
    for dofs_row in indexer.dofs_row:
        shape_sum_row += len(dofs_row)
        if np.any(dofs_row > expected_shape[0]):
            raise ValueError(
                f"Bad DoFs index for a matrix with shape {expected_shape}."
            )
    for dofs_col in indexer.dofs_col:
        shape_sum_col += len(dofs_col)
        if np.any(dofs_col > expected_shape[1]):
            raise ValueError(
                f"Bad DoFs index for a matrix with shape {expected_shape}."
            )
    result_shape = (shape_sum_row, shape_sum_col)
    if result_shape != expected_shape:
        raise ValueError(
            f"Matrix shape {expected_shape} is inconsistent with the groups shape "
            f"{result_shape}."
        )

    # Checking the RHS.
    if bmat.rhs.shape[0] != bmat.mat.shape[0]:
        raise ValueError(
            f"Inconsistent RHS shape: {bmat.rhs.shape = }, {bmat.mat.shape = }."
        )


def get_nonzero_indices(
    A: csr_matrix, row_indices: np.ndarray, col_indices: np.ndarray
) -> list[int]:
    """
    Get the indices of A.data that correspond to the specified subset of rows and
    columns.

    Parameters:
        A: The input sparse matrix.
        row_indices: The array of row indices to consider.
        col_indices: The array of column indices to consider.

    Returns:
        Indices in A.data corresponding to non-zero elements in the specified subset.
    """
    result_indices = []
    col_set = set(col_indices)  # For quick lookup

    for row in row_indices:
        start_ptr = A.indptr[row]
        end_ptr = A.indptr[row + 1]

        for data_idx in range(start_ptr, end_ptr):
            col_idx = A.indices[data_idx]
            if col_idx in col_set:
                result_indices.append(data_idx)

    return result_indices


def concatenate_dof_indices(
    x: list[np.ndarray], dtype: DTypeLike = np.int64
) -> np.ndarray:
    """Helper function for `np.concatenate` that handles empty input."""
    if len(x) == 0:
        return np.array([], dtype=dtype)
    return np.concatenate(x, dtype=dtype)
