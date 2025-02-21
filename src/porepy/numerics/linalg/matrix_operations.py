"""
module for operations on sparse matrices
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as sps
from typing import Literal

import porepy as pp

from porepy.utils.mcolon import mcolon

try:
    from numba import njit, prange

    numba_available = True
except ImportError:
    numba_available = False


def zero_columns(A: sps.csc_matrix, cols: np.ndarray) -> None:
    """
    Function to zero out columns in matrix A. Note that this function does not
    change the sparcity structure of the matrix, it only changes the column
    values to 0.

    The matrix is modified in place.

    Parameter
    ---------
    A (scipy.sparse.spmatrix): A sparce matrix
    cols (ndarray): A numpy array of columns that should be zeroed

    Return
    ------
    None


    """

    if A.getformat() != "csc":
        raise ValueError("Need a csc matrix")
    indptr = A.indptr
    col_indptr = mcolon(indptr[cols], indptr[cols + 1])
    A.data[col_indptr] = 0


def zero_rows(A: sps.csr_matrix, rows: np.ndarray) -> None:
    """
    Function to zero out rows in matrix A. Note that this function does not
    change the sparcity structure of the matrix, it only changes the row
    values to 0.

    The matrix is modified in place.

    Parameter
    ---------
    A (scipy.sparse.spmatrix): A sparce matrix
    rows (ndarray): A numpy array of columns that should be zeroed

    Return
    ------
    None

    """

    if A.getformat() != "csr":
        raise ValueError("Need a csr matrix")
    indptr = A.indptr
    row_indptr = mcolon(indptr[rows], indptr[rows + 1])
    A.data[row_indptr] = 0


def merge_matrices(
    A: sps.spmatrix,
    B: sps.spmatrix,
    lines_to_replace: np.ndarray,
    matrix_format: Literal["csr", "csc"],
) -> None:
    """Replace rows/coloms of matrix A with rows/cols of matrix B.

    If the matrix format is csc, this function is equivalent with

        A[:, lines_to_replace] = B

    If the matrix format is csr, this funciton is equivalent iwth

        A[lines_to_replace, :] = B

    Replacement is done in place.

    Parameter
    ---------
    A (scipy.sparse.spmatrix): A sparse matrix
    B (scipy.sparse.spmatrix): A sparse matrix
    lines_to_replace (ndarray): Lines of A to be replaced by B.
    matrix_format (str): Should be either 'csr' or 'csc'. Both A and B should adhere
        to the respective format.

    Return
    ------
    None

    """
    # Run a set of checks on the input, it easy to get this wrong.
    if not all((s == matrix_format for s in (A.getformat(), B.getformat()))):
        raise ValueError(
            f"Both matrices should be of the specified format {matrix_format}"
        )
    if matrix_format == "csr":
        if A.shape[1] != B.shape[1]:
            raise ValueError(
                f"Unequal number of matrix columns: {A.shape[1]} and {B.shape[1]}"
            )
        if lines_to_replace.size != B.shape[0]:
            raise ValueError("B.shape[0] must equal size of lines")
    if matrix_format == "csc":
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                f"Unequal number of matrix columns: {A.shape[0]} and {B.shape[0]}"
            )
        if lines_to_replace.size != B.shape[1]:
            raise ValueError("B.shape[1] must equal size of lines")
    if np.unique(lines_to_replace).size != lines_to_replace.size:
        raise ValueError("Can only merge unique lines")
    indptr = A.indptr
    indices = A.indices
    data = A.data

    ind_ix = mcolon(indptr[lines_to_replace], indptr[lines_to_replace + 1])

    # First we remove the old data
    num_rem = np.zeros(indptr.size, dtype=np.int32)
    num_rem[lines_to_replace + 1] = (
        indptr[lines_to_replace + 1] - indptr[lines_to_replace]
    )
    num_rem = np.cumsum(num_rem, dtype=num_rem.dtype)

    indptr = indptr - num_rem

    keep = np.ones(A.data.size, dtype=bool)
    keep[ind_ix] = False
    indices = indices[keep]
    data = data[keep]

    # Then we add the new
    b_indptr = B.indptr
    b_indices = B.indices
    b_data = B.data

    num_added = np.zeros(indptr.size, dtype=np.int32)
    num_added[lines_to_replace + 1] = b_indptr[1:] - b_indptr[:-1]
    num_added = np.cumsum(num_added, dtype=num_added.dtype)

    rep = np.diff(b_indptr)
    indPos = np.repeat(indptr[lines_to_replace], rep)

    A.indices = np.insert(indices, indPos, b_indices)
    A.data = np.insert(data, indPos, b_data)
    A.indptr = indptr + num_added


def stack_mat(A: sps.spmatrix, B: sps.spmatrix):
    """Stack matrix B at the end of matrix A.

    If A and B are csc matrices this function is equivalent to
        A = scipy.sparse.hstack((A, B))
    If A and B are csr matrices this function is equivalent to
        A = scipy.sparse.vstack((A, B))

    Parameters:
        A: A sparse matrix.
        B: A sparse matrix

    """
    if A.getformat() != "csc" and A.getformat() != "csr":
        raise ValueError("Need a csc or csr matrix")
    elif A.getformat() != B.getformat():
        raise ValueError("A and B must be of same matrix type")
    if A.getformat() == "csc":
        if A.shape[0] != B.shape[0]:
            raise ValueError("A.shape[0] must equal B.shape[0]")
    if A.getformat() == "csr":
        if A.shape[1] != B.shape[1]:
            raise ValueError("A.shape[0] must equal B.shape[0]")
    if B.indptr.size == 1:
        return
    A.indptr = np.append(A.indptr, B.indptr[1:] + A.indptr[-1])
    A.indices = np.append(A.indices, B.indices)
    A.data = np.append(A.data, B.data)

    if A.getformat() == "csc":
        A._shape = (A._shape[0], A._shape[1] + B._shape[1])
    if A.getformat() == "csr":
        A._shape = (A._shape[0] + B._shape[0], A._shape[1])


def copy(A: sps.spmatrix) -> sps.spmatrix:
    """Create a new matrix C that is a copy of matrix A.

    This function is equivalent to A.copy(), but does not change the ordering of
    A.indices for csc and csr matrices.

    Parameters:
        A: A sparce matrix.

    Returns:
        A sparce matrix copy of A.

    """
    if A.getformat() == "csc":
        return sps.csc_matrix((A.data, A.indices, A.indptr), shape=A.shape)
    elif A.getformat() == "csr":
        return sps.csr_matrix((A.data, A.indices, A.indptr), shape=A.shape)
    else:
        return A.copy()


def stack_diag(A: sps.spmatrix, B: sps.spmatrix) -> sps.spmatrix:
    """Create a new matrix C that contains matrix A and B at the diagonal.

    C = [[A, 0], [0, B]]
    This function is equivalent to sps.block_diag((A, B), format=A.format), but does not
    change the ordering of A.indices or B.indices.

    Parameters:
        A: A sparse matrix.
        B: A sparse matrix.

    Returns:
        A sparse matrix containing A and B at the diagonal.

    """
    if A.getformat() != "csc" and A.getformat() != "csr":
        raise ValueError("Need a csc or csr matrix")
    elif A.getformat() != B.getformat():
        raise ValueError("A and B must be of same matrix type")
    if B.indptr.size == 1:
        return A
    C = A.copy()

    if A.getformat() == "csc":
        indices_offset = A.shape[0]
    else:
        indices_offset = A.shape[1]
    C.indptr = np.append(A.indptr, B.indptr[1:] + A.indptr[-1])
    C.indices = np.append(A.indices, B.indices + indices_offset)
    C.data = np.append(A.data, B.data)

    C._shape = (A._shape[0] + B._shape[0], A._shape[1] + B._shape[1])
    return C


def slice_indices(
    A: sps.spmatrix, slice_ind: np.ndarray, return_array_ind: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, Union[np.ndarray, slice]]]:
    """Function for slicing sparse matrix along rows or columns.

    If A is a csc_matrix, it will be sliced along columns. If A is a csr_matrix, it will
    be sliced along the rows.

    Parameters:
        A: A sparse matrix.
        slice_ind: Array containing indices to be sliced.

    Returns:
        Tuple of indices and array_ind.

            indices:
            If A is csc_matrix, the nonzero row indices or columns slice_ind.
            If A is csr_matrix, the nonzero columns indices or rows slice_ind.

            array_ind: The indices in the compressed storage format (csc or csr)
            corresponding to the slice; so that, if A is csr, A.indices[array_ind] gives
            the columns of the slice (represented in indices), and the corresponding
            data can be accessed as A.data[array_ind]. Only returned if return_array_ind
            is True.

    Example:

        >>> A = sps.csc_matrix(np.eye(10))
        >>> rows = slice_indices(A, np.array([0,2,3]))

    """
    assert A.getformat() == "csc" or A.getformat() == "csr"
    if np.asarray(slice_ind).dtype == "bool":
        # Convert to indices.
        # First check for dimension.
        if slice_ind.size != A.indptr.size - 1:
            raise IndexError("boolean index did not match indexed array")
        slice_ind = np.where(slice_ind)[0]
    if isinstance(slice_ind, int):
        array_ind = slice(A.indptr[slice_ind], A.indptr[slice_ind + 1])
        indices: np.ndarray = A.indices[array_ind]
    elif isinstance(slice_ind, np.generic):
        # Special case for single index.
        assert isinstance(slice_ind, np.integer)  # For mypy.
        array_ind = slice(A.indptr[slice_ind], A.indptr[slice_ind + 1])
        indices = A.indices[array_ind]
    else:
        array_ind = mcolon(A.indptr[slice_ind], A.indptr[slice_ind + 1])
        indices = A.indices[array_ind]
    if return_array_ind:
        return indices, array_ind
    else:
        return indices


def slice_sparse_matrix(A: sps.spmatrix, ind: np.ndarray | int) -> sps.spmatrix:
    """Function for slicing sparse matrix along rows or columns.

    If the matrix is a csc_matrix it will be sliced along columns, while if matrix is a
    csr_matrix it will be sliced along the rows.

    Parameters:
        A: A sparse matrix. Should be either csc or csr.
        ind: Array containing indices to be sliced.

    Raises:
        ValueError: If the matrix is not csc or csr.

    Returns:
        A sliced matrix.
            If A is a csc_matrix A_sliced = A[:, ind]
            If A is a csr_matrix A_sliced = A[ind, :]

    """
    if A.getformat() != "csc" and A.getformat() != "csr":
        raise ValueError("Need a csc or csr matrix")

    # The slicing will be done based on a numpy array of indices (row and column for csr
    # and csc, respectively). The provided index can be an array of booleans, or a
    # single integer. Convert these to numpy arrays of indices.
    if np.asarray(ind).dtype == "bool":
        ind = np.where(ind)[0]
    if isinstance(ind, int):
        ind = np.array([ind])

    # Dimension of the sliced matrix along the axis of the slicing.
    N = ind.size
    # Expand the indices along the compressed axis. To understand this command, it is
    # necessary to be familiar with the compressed storage format.
    ind_slice = mcolon(A.indptr[ind], A.indptr[ind + 1])
    # Pick out the subset of the indices from A that are also in the slice.
    indices = A.indices[ind_slice]
    # Make a new indptr array and fill it with the relevant parts of the original indptr
    # array.
    indptr = np.zeros(ind.size + 1)
    indptr[1:] = np.cumsum(A.indptr[ind + 1] - A.indptr[ind])
    # Data can be extracted directly from the data array of A.
    data = A.data[ind_slice]

    if A.getformat() == "csc":
        return sps.csc_matrix((data, indices, indptr), shape=(A.shape[0], N))
    elif A.getformat() == "csr":
        return sps.csr_matrix((data, indices, indptr), shape=(N, A.shape[1]))


class MatrixSlicer:
    """Class for slicing (sparse) matrices, vectors, and AD arrays.

    Operating with this class is equivalent to using a projection matrix. However, using
    this class avoids both the construction of the projection matrix and the
    matrix-matrix product (alt. matrix-vector product when applied to a matrix). This
    can give significant speedups for large matrices.

    Example:
        Let y be a numpy array, scipy sparse matrix, or an AdArray, with size, say, 4.
        That is, if y is a 1d numpy array, it has shape (4,), if it is an AdArray, it
        value attribute has shape (4,), and if it is a sparse matrix (or a numpy array
        of more than one dimension, though that case is not a primary motivation for the
        slicer), it has 4 rows, while the number of columns is arbitrary. The slicer
        then permits the following classes of operations:

        1. Restrict y to elements 0 and 2:
            >>> S = MatrixSlicer(domain_indices=np.array([0, 2]))
            >>> y_restricted = S @ y
            This is a mapping from R^4 to R^2 (or the natural generalization if y is a
            matrix).
        2. Map y to a larger space, so that element 0 goes to dimension 0, 1 -> 2, 2 ->
            4, 3 -> 1: >>> S = MatrixSlicer(range_indices=np.array([0, 2, 4, 1])) >>>
            y_mapped = S @ y This is a mapping from R^4 to R^5, since the highest index
            in range_indices is 4 (and it is 0-offset).
        3. Do the same operation as in 2., but leave out the mapping of element 1:
            >>> S = MatrixSlicer(domain_indices=np.array([0, 2, 3]),
                                 range_indices=np.array([0, 4, 1])
                                )
            >>> y_mapped = S @ y
        4. Set the size of the range space explicitly (it must be at least as large as
            the size implied by range_indices): >>> S =
            MatrixSlicer(domain_indices=np.array([0, 2, 3]),
                                 range_indices=np.array([0, 4, 1]), range_size=7
                                )
            >>> y_mapped = S @ y
            This is a mapping from R^4 to R^7.
            The range_size parameter can also be used in the case where only the domain
            indices are given.

    Warning:
        Since this class is not constructed to be used as an operand in general
        arithmetic operations, care is needed when using it in combination with other
        operands. Consider the expression

            A x S @ y

        where S is a MatrixSlicer instance and y is a quantity to be sliced. Depending
        on the operator x, and following Python's rules for operator precedence, the
        expression will be evaluated as either 'A x (S @ y)' or '(A x S) @ y', where the
        former is the only reasonable interpretation. Unfortunately, if x has equal
        precedence with @, Python will evaluate the expression from the left, that is,
        '(A x S) @ y'. This can of course be enforced by using parentheses, but doing so
        throughout the code will become cumbersome and error-prone (note that the need
        for parantheses will carry over the Ad operators when a representation of the
        MatrixSlicer is introduced in that framework). If x has higher precedence than
        @, parantheses around S @ y are needed.

        As a partial remedy, the MatrixSlicer implements methods __rmatmul__, __rmul__,
        and __rtruediv__ to handle cases where it is the right operand. These are the
        relevant operands (e.g., not integer division or the modulus operator) that have
        equal precedence with @, where issues can arise. These special methods use
        delayed evaluation to first carry out the slicing (S @ y), and then apply 'A x'
        to the result.

        *However*, this only works if the methods __rmul__ etc. are called in the first
        place. Fundamental data types in python (int, float) will do so, as will scipy
        sparse matrices. *Numpy arrays will most likely not do so*. Instead, it will use
        its own __mul__ method with the MatrixSlicer as the right operand, and probably
        return a numpy array with data type object. Some rules of thumb therefore apply:
            1. Be careful when using the MatrixSlicer in chained operations, in
               particular with numpy arrays.
            2. If in doubt, use paranthesis.

    """

    def __init__(
        self,
        domain_indices: Optional[np.ndarray] = None,
        range_indices: Optional[np.ndarray] = None,
        range_size: Optional[int] = None,
        domain_size: Optional[int] = None,
    ) -> None:
        if range_indices is None and domain_indices is None:
            # We need to know what we are mapping from or to (or both).
            raise ValueError("Either range_indices or domain_indices must be set.")

        is_onto = False
        if domain_indices is not None and range_indices is None:
            # If only domain indices are given, the range is assumed to be the same size
            # as the domain. The slicing will then be a simple restriction of the
            # domain.
            range_indices = np.arange(domain_indices.size)
            if range_size is None:
                is_onto = True

        elif range_indices is not None and domain_indices is None:
            # If only range indices are given, the domain is assumed to be the same size
            # as the range. The slicing will then be a simple prolongation of the
            # domain.

            domain_indices = np.arange(range_indices.size)
            is_onto = False

        if range_size is None:
            # If range_size is not given, it is assumed to be the maximum of the range
            # indices plus one (since the indices are 0-offset).
            range_size = range_indices.max() + 1

        # Store the indices and size.
        self._domain_indices: np.ndarray = domain_indices
        self._range_indices: np.ndarray = range_indices
        self._range_size: int = range_size
        self._domain_size: int = (
            domain_size if domain_size is not None else domain_indices.max() + 1
        )

        self._is_onto: bool = is_onto

        # Precompute the sorting of the range indices. This is needed when slicing a
        # matrix, and can be done independently of the matrix to be sliced.
        self._sort_ind_range = np.argsort(range_indices)

        # Variable to store pending operations and operand; see class documentation for
        # description.
        self._pending_operation = None
        self._pending_operand = None

        self._is_transposed = False

    @property
    def domain_indices(self) -> np.ndarray:
        return self._domain_indices

    @property
    def range_indices(self) -> np.ndarray:
        return self._range_indices

    @property
    def range_size(self) -> int:
        return self._range_size

    @property
    def domain_size(self) -> int:
        return self._domain_size

    def transpose(self) -> MatrixSlicer:
        """Return a transposed MatrixSlicer.

        A transposed MatrixSlicer will slice the matrix along columns instead of rows.
        The domain and range indices will refer to columns instead of rows. The range
        size will be the number of columns in the resulting matrix. The transpose
        operation has no effect on the slicing of vectors, while, if applied to an
        AdArray, an error will be raised, since the Jacobian of an AdArray should be
        treated in a row-wise manner.

        Returns:
            A transposed MatrixSlicer.

        """
        obj = MatrixSlicer(
            domain_indices=self._range_indices,
            range_indices=self._domain_indices,
            range_size=self._domain_size,
            domain_size=self._range_size,
        )

        obj._is_transposed = not obj._is_transposed
        return obj

    def __getattr__(self, name: str) -> MatrixSlicer:
        """Implement the transpose operation as an attribute. This enables the user to
        write S.T instead of S.transpose().
        """
        if name == "T":
            return self.transpose()
        raise AttributeError(f"MatrixSlicer has no attribute {name}")

    def __repr__(self) -> str:
        s = "MatrixSlicer object\n"
        s += f"Domain size: {self._domain_size}, "
        s += f"number of domain indices: {self._domain_indices.size}.\n"
        s += f"Range size: {self._range_size}, "
        s += f"number of range indices: {self._range_indices.size}.\n"
        s += f"Is onto: {self._is_onto}, is transposed: {self._is_transposed}.\n"
        return s

    def copy(self) -> MatrixSlicer:
        """Create a copy of the MatrixSlicer instance.

        Returns:
            A new instance of MatrixSlicer with the same domain and range indices,
            range size, and state of `is_onto` and `is_transpose`.

        """
        slicer = MatrixSlicer(
            domain_indices=self._domain_indices,
            range_indices=self._range_indices,
            range_size=self._range_size,
        )
        slicer._is_onto = self._is_onto
        slicer._is_transposed = self._is_transposed

        slicer._pending_operation = self._pending_operation
        slicer._pending_operand = self._pending_operand

        return slicer

    def __matmul__(
        self, x: np.ndarray | sps.spmatrix | pp.ad.AdArray
    ) -> np.ndarray | sps.spmatrix | pp.ad.AdArray:
        # Separate handling for different types of input.
        if isinstance(x, MatrixSlicer):
            x._pending_operand = self
            x._pending_operation = "@"
            return x
        if isinstance(x, np.ndarray):
            sliced = self._slice_vector(x)
        elif isinstance(x, (sps.spmatrix, sps.sparray)):
            sliced = self._slice_matrix(x)
        elif isinstance(x, pp.ad.AdArray):
            val = self._slice_vector(x.val)
            jac = self._slice_matrix(x.jac)
            sliced = pp.ad.AdArray(val, jac)

        if self._pending_operand is not None:
            # If there is a pending operand, we need to apply it to the sliced matrix.
            product = eval(f"self._pending_operand {self._pending_operation} sliced")
            return product
        else:
            return sliced

    def __rmatmul__(self, other):
        slicer = self.copy()
        slicer._pending_operand = other
        slicer._pending_operation = "@"
        return slicer

    def __rmul__(self, other):
        slicer = self.copy()
        slicer._pending_operand = other
        slicer._pending_operation = "*"
        return slicer

    def __rtruediv__(self, other):
        slicer = self.copy()
        slicer._pending_operand = other
        slicer._pending_operation = "/"
        return slicer

    def __mul__(self, other):
        """There are of course other operations that also are not supported, but we
        explicitly raise a ValueError for this case, since the user is likely to try
        slicing by multiplication.
        """
        raise ValueError("MatrixSlicer does not support multiplication. Use @ instead.")

    def _slice_vector(self, x: np.ndarray) -> np.ndarray:
        """Slice a vector.

        Parameters:
            x: Vector to be sliced.

        Returns:
            The sliced vector.

        """
        if self._is_onto:
            return x[self._domain_indices]

        if x.ndim == 1:
            vec = np.zeros(self._range_size)
        elif x.ndim == 2:
            # 2d dense arrays are not really intended used in the Ad framework, which is
            # the primary client of the MatrixSlicer. However, we can handle them, and
            # they are invaluable for testing.
            vec = np.zeros((self._range_size, x.shape[1]))
        else:
            raise ValueError("Only 1d and 2d dense arrays are supported")
        vec[self._range_indices] = x[self._domain_indices]
        return vec

    def _slice_matrix(self, A: sps.csr_matrix) -> sps.csr_matrix:
        """Slice a matrix along rows, accessing data and indices directly.

        Parameters:
            A: Matrix to be sliced.

        Returns:
            The sliced matrix.

        """
        A = A.tocsr()
        container = sps.csr_matrix

        if self._is_onto:
            return A[self._domain_indices]

        # Data storage for the matrix to be sliced.
        indptr = A.indptr

        # Sorting indices for the range indices. This was precomputed, since it does not
        # depend on the matrix to be sliced.
        sort_ind_range = self._sort_ind_range

        # Number of non-zero elements in each row in the domain.
        num_elem_per_row_domain = np.take(indptr, self._domain_indices + 1) - np.take(
            indptr, self._domain_indices
        )

        # Number of non-zero elements in the range matrix. We will eventually use this
        # in a cumsum operation, which should be 0 at the start. We therefore prepend a
        # 0, to avoid having to use array concatenation. The assignment to the sorted
        # range indices (next statement) starts at 1.
        num_elem_per_row = np.zeros(self._range_size + 1, dtype=int)

        # Assignment to the sorted range indices.
        num_elem_per_row[self._range_indices[sort_ind_range] + 1] = (
            num_elem_per_row_domain[sort_ind_range]
        )
        # Cumulative sum to get the indptr array. Reuse the array
        # num_elem_per_row_domain to store the indptr array to avoid memory allocation.
        np.cumsum(num_elem_per_row, out=num_elem_per_row)
        # Make a reference to the indptr array with a more descriptive name.
        new_indptr = num_elem_per_row

        # Get the indices (referring to the fields A.data and A.indices) of the non-zero
        # elements in the target rows. This requires that we
        sorted_domain_indices = self._domain_indices[sort_ind_range]
        sub_indices = mcolon(
            indptr[sorted_domain_indices], indptr[sorted_domain_indices + 1]
        )

        # Fetch the data and indices from the original matrix. NOTE: This part will in
        # many cases (among the) most time-consuming part of the slicing operation.
        # After a long day of experimenting, the below code is the fastest EK could come
        # up with. Notably, using take is about 1/3 faster than using fancy indexing
        # (~A.data[sub_indices]). Ideally, we would have used a view of the arrays, but
        # numpy could not be convinced to work that way.
        new_data = np.take(A.data, sub_indices)
        new_indices = np.take(A.indices, sub_indices)
        new_num_rows = self._range_size

        shape = (new_num_rows, A.shape[1])

        return container((new_data, new_indices, new_indptr), shape=shape)


def optimized_compressed_storage(A: sps.spmatrix) -> sps.spmatrix:
    """Choose an optimal storage format (csr or csc) for a sparse matrix.

    The format is chosen depending on whether A.shape[0] > A.shape[1] or not.

    For very sparse matrices where the number of rows and columns differs significantly
    (e.g., projection matrices), there can be substantial memory gains by choosing the
    right storage format, by reducing the number of equal

    As an illustration, consider a matrix with shape 1 x N with 1 element: If stored in
    csc format, this will require an indptr array of size N, while csr format requires
    only size 2.

    Parameters:
        A: Matrix to be reformatted.

    Returns:
        sps.spmatrix: The matrix represented in optimal storage format.

    """
    if A.shape[0] > A.shape[1]:
        return A.tocsc()
    else:
        return A.tocsr()


def sparse_dia_from_sparse_blocks(blocks: list[sps.dia_matrix]) -> sps.dia_matrix:
    """Construct a sparse diagonal matrix from a list of sparse diagonal matrices.

    This is a shortcut that can be used for fast construction of block diagonal matrices
    where the individual blocks are known to be diagonal. The blocks are concatenated
    along the main diagonal, hence the resulting matrix will be structurally equal to an
    identity matrix.

    For block diagonal matrices with a more general structure of the matrix blocks
    (those that have non-zero elements off the main diagonal), use the methods
    cs{r,c}_matrix_from_sparse_blocks().

    Parameters:
        blocks: List of diagonal matrices. Should be of dia-format and only have data
            along the main diagonal.

    Raises:
        ValueError: If the blocks are not of dia-format or if the blocks have data
            off the main diagonal.

    Returns:
        A sparse diagonal matrix.

    """
    data_array = []
    for mat in blocks:
        if mat.getformat() != "dia":
            raise ValueError("All blocks must be in dia format")
        if mat.offsets.size != 1 or mat.offsets[0] != 0:
            raise ValueError("All blocks must be diagonal.")
        data_array.append(mat.data.ravel())

    data = np.concatenate(data_array)
    return sps.dia_matrix((data, 0), shape=(data.size, data.size))


def csr_matrix_from_sparse_blocks(blocks: list[sps.spmatrix]) -> sps.spmatrix:
    """Create a csr representation of a block diagonal matrix from a list of sparse
    matrices.

    The function is equivalent to, but can be significantly faster than, the call

        sps.block_diag(blocks)

    Parameters:
        blocks: List of sparse matrices to be included. This can be of any sparse
            format, but the function will be faster if the blocks are of csr format.

    Returns:
        csr representation of the block matrix.

    See also:
        csc_matrix_from_sparse_blocks() for an equivalent function that is optimized for
            csc matrices.
        csr_matrix_from_dense_blocks(), csc_matrix_from_dense_blocks() for functions
            that create sparse block diagonal matrices from dense data.

    Example:
        >>> block_1 = sps.csr_matrix([[1, 2], [3, 4]])
        >>> block_2 = sps.csr_matrix([[5, 6, 7], [8, 9, 10]])
        >>> csr_matrix_from_sparse_blocks([block_1, block_2]).toarray()
        array([[1, 2, 0, 0, 0],
               [3, 4, 0, 0, 0],
               [0, 0, 5, 6, 7],
               [0, 0, 8, 9, 10]])

    """
    return _csx_matrix_from_sparse_blocks(blocks, "csr")


def csc_matrix_from_sparse_blocks(blocks: list[sps.spmatrix]) -> sps.spmatrix:
    """Create a csc representation of a block diagonal matrix from a list of sparse
    matrices.

    The function is equivalent to, but can be significantly faster than, the call

        sps.block_diag(blocks)

    Parameters:
        blocks: List of sparse matrices to be included. This can be of any sparse
            format, but the function will be faster if the blocks are of csc format.

    Returns:
        csc representation of the block matrix.

    See also:
        csr_matrix_from_sparse_blocks() for an equivalent function that is optimized for
            csr matrices.
        csr_matrix_from_dense_blocks(), csc_matrix_from_dense_blocks() for functions
            that create sparse block diagonal matrices from dense data.

    Example:
        >>> block_1 = sps.csc_matrix([[1, 2], [3, 4]])
        >>> block_2 = sps.csc_matrix([[5, 6, 7], [8, 9, 10]])
        >>> csc_matrix_from_sparse_blocks([block_1, block_2]).toarray()
        array([[1, 2, 0, 0, 0],
               [3, 4, 0, 0, 0],
               [0, 0, 5, 6, 7],
               [0, 0, 8, 9, 10]])

    """
    return _csx_matrix_from_sparse_blocks(blocks, "csc")


def _csx_matrix_from_sparse_blocks(
    blocks: list[sps.spmatrix], matrix_format: Literal["csr", "csc"]
) -> sps.spmatrix:
    """Create a csr or csc representation of a block diagonal matrix from a list of
    sparse matrices.

    The function is equivalent to, but can be significantly faster than, the call

        sps.block_diag(blocks)

    Parameters:
        blocks: List of sparse matrices to be added.
        matrix_format: type of matrix to be created. Should be either 'csr' or 'csc'.

    Raises:
        ValueError: If the size of the data does not match the blocks size and number
            of blocks.

    Returns:
        sps.csr_matrix: csr representation of the block matrix.

    """
    if matrix_format == "csr":
        container = sps.csr_matrix
    elif matrix_format == "csc":
        container = sps.csc_matrix
    else:
        raise ValueError('matrix_format must be either "csr" or "csc".')

    # Convert the blocks to the correct format. NOTE: Timing shows that for general
    # matrices, this is the main bottleneck of the function. Thus if possible, make sure
    # all the blocks are constructed in the correct format.
    for i in range(len(blocks)):
        if blocks[i].getformat() != matrix_format:
            blocks[i] = blocks[i].asformat(matrix_format)

    # Shortcut. We know the format is right.
    if len(blocks) == 1:
        return blocks[0]

    # Calculate the size of the block matrix.
    num_rows = sum([m.shape[0] for m in blocks])
    num_cols = sum([m.shape[1] for m in blocks])

    # CSC and CSR matrices are constructed and operated on in a very similar way; the
    # difference is in which dimension the indices represent. We need this to get the
    # offsets right.
    indices_dim = 0 if matrix_format == "csc" else 1
    indices_offset = np.cumsum([0] + [m.shape[indices_dim] for m in blocks])
    # The indptr offset is the same for both formats.
    indptr_offset = np.cumsum([0] + [m.indptr[-1] for m in blocks])

    # Now we can make lists of the data, indices and indptr arrays, using offsets for
    # the indices and indptr arrays.
    indices = [m.indices + indices_offset[i] for i, m in enumerate(blocks)]
    indptr = [np.array([0])] + [
        m.indptr[1:] + indptr_offset[i] for i, m in enumerate(blocks)
    ]
    data = [m.data for m in blocks]
    # Concatenate the arrays and create the block matrix.
    block_mat = container(
        (np.concatenate(data), np.concatenate(indices), np.concatenate(indptr)),
        shape=(num_rows, num_cols),
    )
    return block_mat


def csr_matrix_from_dense_blocks(
    data: np.ndarray, block_size: int, num_blocks: int
) -> sps.spmatrix:
    """Create a csr representation of a block diagonal matrix from an array of data.

    The block-diagonal matrix is constructed by inserting the data in the blocks in
    a row-wise fashion. The blocks are assumed to be square and of the same size. See
    the example below for an illustration.

    The function is equivalent to, but orders of magnitude faster than, the call

        sps.block_diag(blocks)

    Parameters:
        data: Matrix values, sorted column-wise.
        block_size: The size of *all* the blocks.
        num_blocks: Number of blocks to be added.

    Raises:
        ValueError: If the size of the data does not match the blocks size and number
            of blocks.

    Returns:
        sps.csr_matrix: csr representation of the block matrix.

    See also:
        csc_matrix_from_dense_blocks() for an equivalent function that is optimized for
            csc matrices.
        csr_matrix_from_sparse_blocks(), csc_matrix_from_sparse_blocks() for functions
            that create block diagonal matrices from sparse matrices.

    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> block_size, num_blocks = 2, 2
        >>> csr_matrix_from_dense_blocks(data, block_size, num_blocks).toarray()
        array([[1, 2, 0, 0],
               [3, 4, 0, 0],
               [0, 0, 5, 6],
               [0, 0, 7, 8]])

    """
    return _csx_matrix_from_dense_blocks(data, block_size, num_blocks, sps.csr_matrix)


def csc_matrix_from_dense_blocks(
    data: np.ndarray, block_size: int, num_blocks: int
) -> sps.spmatrix:
    """Create a csc representation of a block diagonal matrix from an array of data.

    The block-diagonal matrix is constructed by inserting the data in the blocks in
    a column-wise fashion. The blocks are assumed to be square and of the same size. See
    the example below for an illustration.

    The function is equivalent to, but orders of magnitude faster than, the call

        sps.block_diag(blocks)

    Parameters:
        data: Matrix values, sorted column-wise.
        block_size: The size of *all* the blocks.
        num_blocks: Number of blocks to be added.

    Raises:
        ValueError: If the size of the data does not match the blocks size and number
            of blocks.

    Returns:
        sps.csc_matrix: csc representation of the block matrix.

    See also:
        csr_matrix_from_dense_blocks() for an equivalent function that is optimized for
            csc matrices.
        csr_matrix_from_sparse_blocks(), csc_matrix_from_sparse_blocks() for functions
            that create block diagonal matrices from sparse matrices.

    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> block_size, num_blocks = 2, 2
        >>> csc_matrix_from_dense_blocks(data, block_size, num_blocks).toarray()
        array([[1, 3, 0, 0],
               [2, 4, 0, 0],
               [0, 0, 5, 7],
               [0, 0, 6, 8]])

    """
    return _csx_matrix_from_dense_blocks(data, block_size, num_blocks, sps.csc_matrix)


def _csx_matrix_from_dense_blocks(
    data: np.ndarray, block_size: int, num_blocks: int, matrix_format
) -> sps.spmatrix:
    """Create a csr or csc representation of a block diagonal matrix of uniform block
    size.

    The function is equivalent to, but orders of magnitude faster than, the call

        sps.block_diag(blocks)

    Parameters:
        data: Matrix values, sorted column-wise.
        block_size: The size of *all* the blocks.
        num_blocks: Number of blocks to be added.
        matrix_format: type of matrix to be created. Should be either sps.csc_matrix
            or sps.csr_matrix

    Returns:
        sps.csr_matrix: csr representation of the block matrix.

    Raises:
        ValueError: If the size of the data does not match the blocks size and number
            of blocks.

    """
    if not data.size == block_size**2 * num_blocks:
        raise ValueError("Incompatible input to generate block matrix")
    # The block structure of the matrix allows for a unified construction of compressed
    # column and row matrices. The difference will simply be in how the data is
    # interpreted.

    # The new columns or rows start with intervals of block_size.
    indptr = np.arange(0, block_size**2 * num_blocks + 1, block_size)

    # To get the indices in the compressed storage format requires some more work.
    if block_size > 1:
        # First create indices for each of the blocks.
        #  The inner tile creates arrays
        #   [0, 1, ..., block_size-1, 0, 1, ... block_size-1, ... ]
        #   The size of the inner tile is block_size^2, and forms the indices of a
        # single block.
        #  The outer tile repeats the inner tile, num_blocks times
        #  The size of base is thus block_size^2 * num_blocks
        base = np.tile(
            np.tile(np.arange(block_size), (block_size, 1)).reshape((1, -1)), num_blocks
        )[0]
        # Next, increase the index in base, so as to create a block diagonal matrix
        # the first block_size^2 elements (e.g. the elemnets of the first block are
        # unperturbed.
        # The next block_size elements are increased by block_size^2 etc.
        block_increase = (
            np.tile(np.arange(num_blocks), (block_size**2, 1)).reshape(
                (1, -1), order="F"
            )[0]
            * block_size
        )
        indices = base + block_increase
    else:
        indices = np.arange(num_blocks, dtype=int)
    mat = matrix_format(
        (data, indices, indptr),
        shape=(num_blocks * block_size, num_blocks * block_size),
    )
    return mat


def invert_diagonal_blocks(
    mat: sps.spmatrix, s: np.ndarray, method: Optional[str] = None
) -> Union[sps.csr_matrix, sps.csc_matrix]:
    """Invert block diagonal matrix.

    Three implementations are available, either pure numpy, or a speedup using
    numba or cython. If none is specified, the function will try to use numba,
    then cython. The python option will only be invoked if explicitly asked
    for; it will be very slow for general problems.

    Parameters:
        mat: sps.csr or sps.csc matrix to be inverted.
        s: block size. Must be int64 for the numba acceleration to work
        method: Choice of method. Either numba (default), cython or 'python'.
            Defaults to None, in which case first numba, then cython is tried.

    Returns:
        imat: Inverse matrix.

    Raises:
        ImportError: If numba or cython implementation is invoked without numba or
        cython being available on the system.

    """

    def invert_diagonal_blocks_python(a: sps.spmatrix, size: np.ndarray) -> np.ndarray:
        """
        Invert block diagonal matrix using pure python code.

        Parameters:
            a: Block diagonal sparse matrix
            size: Size of individual blocks

        Returns:
            inv_a: Flattened nonzero values of the inverse matrix
        """

        # This function only supports CSR anc CSC format.
        if not (sps.isspmatrix_csr(a) or sps.isspmatrix_csc(a)):
            raise TypeError("Sparse array type not implemented: ", type(a))

        # Construction of simple data structures (low complexity).
        # Indices for block positions, flattened inverse block positions and nonzeros.
        # Expanded block positions.
        idx_blocks = np.cumsum([0] + list(size))
        # Expanded nonzero positions for flattened inverse blocks.
        idx_inv_blocks = np.cumsum([0] + list(size * size))
        # Nonzero positions for the given matrix data (i.e. a.data).
        idx_nnz = np.searchsorted(a.indices, idx_blocks)

        # Retrieve global indices (low complexity).
        if sps.isspmatrix_csr(a):
            # cols are in fact a.indices
            cols = a.indices
            # row_reps is a structure containing the number of repetitions for a row
            row_reps = a.indptr[1 : a.indptr.size] - a.indptr[0 : a.indptr.size - 1]
            # rows are in fact a vector of global indices, i.e.
            # row_i repeated row_reps[i] times
            rows = np.repeat(np.arange(a.shape[0], dtype=np.int32), row_reps)
        else:
            # rows are in fact a.indices
            rows = a.indices
            # col_reps is a structure containing the number of repetitions for a column
            col_reps = a.indptr[1 : a.indptr.size] - a.indptr[0 : a.indptr.size - 1]
            # cols are in fact a vector of global indices, i.e.
            # col_j repeated col_reps[j] times
            cols = np.repeat(np.arange(a.shape[1], dtype=np.int32), col_reps)

        # Nonzero entries
        data = a.data

        # Flattened nonzero values of the dense inverse.
        inv_a = np.zeros(idx_inv_blocks[-1])

        def operate_on_block(ib: int):
            """
            Retrieve, invert, and assign dense block inverse values.

            Parameters:
                ib: the block index

            """

            # To avoid creation of new data
            # this line retrieves nnz for the block with index ib
            flat_block = inv_a[idx_inv_blocks[ib] : idx_inv_blocks[ib + 1]]

            # Retrieve global block position
            idx_shift = idx_blocks[ib]
            # Transform from global to local rows positions
            l_row = rows[idx_nnz[ib] : idx_nnz[ib + 1]] - idx_shift
            # Transform from global to local cols positions
            l_col = cols[idx_nnz[ib] : idx_nnz[ib + 1]] - idx_shift
            # Construct flattened local positions (major order of non-zeros)
            sequence_ij = l_row * size[ib] + l_col
            # Assigning flattened positions directly from the matrix data (a.data)
            flat_block[sequence_ij] = data[idx_nnz[ib] : idx_nnz[ib + 1]]
            # Reshape flattened block to squared dense block of size[ib]
            dense_block = np.reshape(flat_block, (size[ib], size[ib]))
            # Perform inversion and assigning values from a 1-D iterator (ndarray.flat)
            inv_a[idx_inv_blocks[ib] : idx_inv_blocks[ib + 1]] = np.linalg.inv(
                dense_block
            ).flat

        # Trigger computations from np.fromiter
        np.fromiter(map(operate_on_block, range(size.size)), dtype=np.ndarray)
        return inv_a

    def invert_diagonal_blocks_numba(a: sps.csr_matrix, size: np.ndarray) -> np.ndarray:
        """
        It is the parallel function of the Python inverter.  Using numba support and a
        single call to numba.prange, parallelization is achieved.

        Parameters:
            a : Block diagonal sparse matrix
            size : Size of individual blocks

        Returns:
            inv_a: Flattened nonzero values of the inverse matrix.

        """

        # This function only supports CSR anc CSC format.
        if not (sps.isspmatrix_csr(a) or sps.isspmatrix_csc(a)):
            raise TypeError("Sparse array type not implemented: ", type(a))

        is_csr_q = sps.isspmatrix_csr(a)
        # Matrix information
        data = a.data
        indices = a.indices
        indptr = a.indptr

        # Extended block sizes structure
        sz = np.insert(size, 0, 0).astype(np.int32)

        @njit(
            "f8[::1](b1,f8[::1],i4[::1],i4[::1],i4[::1])",
            cache=True,
            parallel=True,
        )
        def inv_compiled_function(is_csr_q, data, indices, indptr, sz):
            # Construction of simple data structures (low complexity)
            # Indices for block positions, flattened inverse block positions and nonzeros
            # Expanded block positions
            idx_blocks = np.cumsum(sz).astype(np.int32)
            # Expanded nonzero positions for flattened inverse blocks
            idx_inv_blocks = np.cumsum(np.square(sz)).astype(np.int32)
            # Nonzero positions for the given matrix data (i.e. a.data)
            idx_nnz = np.searchsorted(indices, idx_blocks).astype(np.int32)

            # Retrieve global indices (low complexity)
            if is_csr_q:
                cols = indices
                row_reps = indptr[1 : indptr.size] - indptr[0 : indptr.size - 1]
            else:
                rows = indices
                col_reps = indptr[1 : indptr.size] - indptr[0 : indptr.size - 1]

            # flattened nonzero values of the dense inverse (low complexity)
            # Numba np.zeros support ensures v is a contiguous array (C-contiguous)
            v = np.zeros(idx_inv_blocks[-1])

            for ib in prange(sz.size - 1):
                v_range = np.arange(idx_inv_blocks[ib], idx_inv_blocks[ib + 1])
                flat_block = v[v_range]
                # Retrieve global block position
                idx_shift = idx_blocks[ib]
                idx_block = idx_blocks[np.array([ib, ib + 1])]

                # Transform from global to local rows and cols positions
                if is_csr_q:
                    l_row = (
                        np.repeat(
                            np.arange(idx_block[0], idx_block[1]),
                            row_reps[idx_block[0] : idx_block[1]],
                        ).astype(np.int32)
                        - idx_shift
                    )
                    l_col = cols[idx_nnz[ib] : idx_nnz[ib + 1]] - idx_shift
                else:
                    l_row = rows[idx_nnz[ib] : idx_nnz[ib + 1]] - idx_shift
                    l_col = (
                        np.repeat(
                            np.arange(idx_block[0], idx_block[1]),
                            col_reps[idx_block[0] : idx_block[1]],
                        ).astype(np.int32)
                        - idx_shift
                    )
                # Construct flattened local positions (major order of non-zeros)
                sequence_ij = l_row * sz[ib + 1] + l_col
                # Assigning flattened positions directly from the matrix data (a.data)
                flat_block[sequence_ij] = data[idx_nnz[ib] : idx_nnz[ib + 1]]
                # Reshape flattened block to squared dense block of size[ib]
                dense_block = np.reshape(flat_block, (sz[ib + 1], sz[ib + 1]))
                # Perform inversion and assigning values from a 1-D ravelled array
                v[v_range] = np.ravel(np.linalg.inv(dense_block))
            return v

        inv_a = inv_compiled_function(is_csr_q, data, indices, indptr, sz)
        return inv_a

    def invert_diagonal_blocks_numba_old(
        a: sps.csr_matrix, size: np.ndarray
    ) -> np.ndarray:
        """Invert block diagonal matrix by invoking numba acceleration of a simple
        for-loop based algorithm.

        Currently not used, but may be resurrected if the new implementation fails.
        Parameters:
            a : sps.csr matrix
            size : Size of individual blocks

        Returns:
            inv_a: Flattened nonzero values of the inverse matrix.

        """

        # This function only supports CSR format.
        if not sps.isspmatrix_csr(a):
            raise TypeError("Sparse array type not implemented: ", type(a))

        ptr = a.indptr
        indices = a.indices
        dat = a.data

        # Just in time compilation
        @njit("f8[:](i4[:],i4[:],f8[:],i8[:])", cache=True, parallel=True)
        def inv_python(indptr, ind, data, sz):
            """
            Invert block matrices by explicitly forming local matrices. The code
            in itself is not efficient, but it is hopefully well suited for
            speeding up with numba.

            IMPLEMENTATION NOTES BELOW

            The code consists of a loop over the blocks. For each block, a local square
            matrix is formed, the inverse is computed using numpy (which again will
            invoke LAPACK), and the inverse is stored in an (raveled) array. The most
            complex part of the code is the formation of the local matrix: Since the
            original matrix is sparse, there may be zero elements in the blocks
            which may not be explicitly represented in the data, and the order of the
            columns in the sparse format may not be linear. To deal with this, we do a
            double loop to fill in the local matrix.

            Profiling (June 2022) showed that the overhead in filling in the local
            matrix by for-loops was minimal; specifically, attempts at speeding up the
            computations by forcing a full block structure of the matrices (with
            explicit zeros and linear ordering of columns), so that the local matrix
            could be formed by a reshape, failed.

            """

            # Index of where the rows start for each block.
            block_row_starts_ind = np.zeros(sz.size, dtype=np.int32)
            block_row_starts_ind[1:] = np.cumsum(sz[:-1])

            # Number of columns per row. Will change from one column to the
            # next
            num_cols_per_row = indptr[1:] - indptr[0:-1]
            # Index to where the columns start for each row (NOT blocks)
            row_cols_start_ind = np.zeros(num_cols_per_row.size + 1, dtype=np.int32)
            row_cols_start_ind[1:] = np.cumsum(num_cols_per_row)

            # Index to where the (full) data starts. Needed, since the
            # inverse matrix will generally be full
            full_block_starts_ind = np.zeros(sz.size + 1, dtype=np.int32)
            full_block_starts_ind[1:] = np.cumsum(np.square(sz))
            # Structure to store the solution
            inv_vals = np.zeros(np.sum(np.square(sz)))

            # Loop over all blocks. Do this in parallel, this has shown significant
            # speedups by numba.
            for iter1 in prange(sz.size):
                n = sz[iter1]

                loc_mat = np.zeros((n, n))
                # Fill in non-zero elements in local matrix
                # This requires some work, since not all elements in the local matrix
                # are represented in the data array (elements may be zero). Also, the
                # ordering of the data may not correspond to a linear ordering of the
                # columns.
                for iter2 in range(n):  # Local rows
                    global_row = block_row_starts_ind[iter1] + iter2
                    data_counter = row_cols_start_ind[global_row]

                    # Loop over local columns. Getting the number of columns
                    #  for each row is a bit involved
                    for _ in range(
                        num_cols_per_row[iter2 + block_row_starts_ind[iter1]]
                    ):
                        loc_col = ind[data_counter] - block_row_starts_ind[iter1]
                        loc_mat[iter2, loc_col] = data[data_counter]
                        data_counter += 1
                # Compute inverse using np.linalg.inv, which will again invoke an
                # appropriate lapack function.
                inv_mat = np.ravel(np.linalg.inv(loc_mat))

                # Store data in the output
                loc_ind = np.arange(
                    full_block_starts_ind[iter1], full_block_starts_ind[iter1 + 1]
                )
                inv_vals[loc_ind] = inv_mat

            return inv_vals

        inv_a = inv_python(ptr, indices, dat, size)
        return inv_a

    # Remove blocks of size 0
    s = s[s > 0]
    # Select numba function
    if (method == "numba" or method is None) and numba_available:
        try:
            inv_vals = invert_diagonal_blocks_numba(mat, s)
        except np.linalg.LinAlgError:
            raise ValueError("Error in inversion of local linear systems")
    # Select python vectorized function
    elif (
        method == "python"
        or (method == "numba" or method is None)
        and not numba_available
    ):
        if (method == "numba" or method is None) and not numba_available:
            warnings.warn(
                "Numba is not available falling back to python inverter.", UserWarning
            )
        inv_vals = invert_diagonal_blocks_python(mat, s)
    else:
        raise ValueError(f"Unknown type of block inverter {method}")
    ia = block_diag_matrix(inv_vals, s)
    return ia


def block_diag_matrix(vals: np.ndarray, sz: np.ndarray) -> sps.spmatrix:
    """Construct block diagonal matrix based on matrix elements and block sizes.

    Parameters:
        vals: matrix values
        sz: size of matrix blocks

    Returns:
        sps.csr matrix.

    """
    indices = block_diag_index(sz)
    # This line recovers starting indices of the rows.
    indptr = np.hstack((np.zeros(1), np.cumsum(rldecode(sz, sz)))).astype(np.int32)
    n = np.sum(sz)
    return sps.csr_matrix((vals, indices, indptr), shape=(n, n))


def block_diag_index(
    m: np.ndarray, n: Optional[np.ndarray] = None
) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Get row and column indices for block diagonal matrix

    This is intended as the equivalent of the corresponding method in MRST.

    Examples:
    >>> m = np.array([2, 3])
    >>> n = np.array([1, 2])
    >>> i, j = block_diag_index(m, n)
    >>> i, j
    (array([0, 1, 2, 3, 4, 2, 3, 4]), array([0, 0, 1, 1, 1, 2, 2, 2]))
    >>> a = np.array([1, 3])
    >>> i, j = block_diag_index(a)
    >>> i, j
    (array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3]), array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))

    Parameters:
        m - ndarray, dimension 1
        n - ndarray, dimension 1, defaults to m

    """
    # Case for squared but irregular block sizes
    if n is None:
        # Construction of auxiliary structures (low complexity)
        n = np.insert(m, 0, 0).astype(np.int32)
        idx_blocks = np.cumsum(n, dtype=np.int32)
        idx_inv_blocks = np.cumsum(np.square(n), dtype=np.int32)
        i = np.zeros(idx_inv_blocks[-1], dtype=np.int32, order="C")

        def retrieve_indices(ib):
            i_range = np.arange(idx_blocks[ib], idx_blocks[ib + 1])
            # this two step operations is one of the fastest way to:
            # duplicate an array n-times and concatenate the duplicates in a flattened
            # structure
            i_val = np.empty((n[ib + 1], *i_range.shape), i_range.dtype)
            np.copyto(i_val, i_range)
            # Finally assign values
            i[idx_inv_blocks[ib] : idx_inv_blocks[ib + 1]] = i_val.flat

        # Trigger computations from np.fromiter
        np.fromiter(map(retrieve_indices, range(n.size - 1)), dtype=np.ndarray)
        return i

    start = np.hstack((np.zeros(1, dtype="int"), m))
    pos = np.cumsum(start)
    p1 = pos[0:-1]
    p2 = pos[1:] - 1
    p1_full = rldecode(p1, n)
    p2_full = rldecode(p2, n)

    i = mcolon(p1_full, p2_full + 1)
    sumn = np.arange(np.sum(n))
    m_n_full = rldecode(m, n)
    j = rldecode(sumn, m_n_full)
    return i, j


def rlencode(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compress matrix by looking for identical columns.

    Example usage: Convert a full set of (row or column) indices of a sparse matrix into
    compressed storage.

    Acknowledgement: The code is heavily inspired by MRST's function with the same name,
    however, requirements on the shape of functions are probably somewhat different.

    Parameters:
        A: Matrix to be compressed. Should be 2d. Compression   will be along the second
        axis.

    Returns:
        np.ndarray: The compressed array, size n x m.
        np.ndarray: Number of times each row in the first output array should
            be repeated to restore the original array.

    See also:
        rldecode

    """
    comp = A[::, 0:-1] != A[::, 1::]
    i = np.any(comp, axis=0)
    i = np.hstack((np.argwhere(i).ravel(), (A.shape[1] - 1)))

    num = np.diff(np.hstack((np.array([-1]), i)))

    return A[::, i], num


def rldecode(A: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Decode compressed information in indices.

    Example usage: Convert the index pointers in compressed matrix storage
    (row or column) to a full set of indices.

    Acknowledgement: The code is heavily inspired by MRST's function with the
    same name, however, requirements on the shape of functions are probably
    somewhat different.

    >>> rldecode(np.array([1, 2, 3]), np.array([2, 3, 1]))
    [1, 1, 2, 2, 2, 3]

    >>> rldecode(np.array([1, 2]), np.array([1, 3]))
    [1, 2, 2, 2]

    Parameters:
        A (double, m x k), compressed matrix to be recovered. The
        compression should be along dimension 1
        n (int): Number of occurences for each element

    Returns:
        B: The restored array.

    See also:
        rlencode

    """
    r = n > 0
    i = np.cumsum(np.hstack((np.zeros(1, dtype=int), n[r])), dtype=int)
    j = np.zeros(i[-1], dtype=int)
    j[i[1:-1:]] = 1
    B = A[np.cumsum(j)]
    return B


def sparse_kronecker_product(matrix: sps.spmatrix, nd: int) -> sps.spmatrix:
    """Convert the scalar projection to a vector quantity.

    Used to expand projection matrices from scalar versions to a form applicable for
    projection of nd-vector quantities.

    Parameters:
        matrix: Matrix to be expanded using a Kronecker product.
        nd: The dimension to which matrix is expanded. If the prescribed dimension
            is 1, the projection matrix is returned without changes.

    """
    if nd == 1:
        # No need to do expansion for 1d variables.
        return matrix
    else:
        return sps.kron(matrix, sps.eye(nd)).tocsc()


def sparse_array_to_row_col_data(
    A: sps.sparse, remove_nz: Optional[bool] = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Function to retrieve indices and values of a matrix.

    Parameters:
        A: A sparse matrix.

        remove_nz: Optional directive for removing explicit zeros.

    Returns:
        A triplet of rows, columns, and values.

    """

    mat_copy = sps.coo_matrix(A, copy=True)
    if remove_nz:
        nz_mask = mat_copy.data != 0
        return (mat_copy.row[nz_mask], mat_copy.col[nz_mask], mat_copy.data[nz_mask])
    else:
        return (mat_copy.row, mat_copy.col, mat_copy.data)
