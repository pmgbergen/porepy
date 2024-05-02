"""
module for operations on sparse matrices
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as sps
from typing_extensions import Literal

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
    """
    Stack matrix B at the end of matrix A.
    If A and B are csc matrices this function is equivalent to
        A = scipy.sparse.hstack((A, B))
    If A and B are csr matrices this function is equivalent to
        A = scipy.sparse.vstack((A, B))

    Parameters:
    -----------
    A (scipy.sparse.spmatrix): A sparse matrix
    B (scipy.sparse.spmatrix): A sparse matrix

    Return
    ------
    None


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
    """
    Create a new matrix C that is a copy of matrix A
    This function is equivalent to
    A.copy(), but does not change the ordering
    of the A.indices for csc and csr matrices

    Parameters:
    -----------
    A (scipy.sparse.spmatrix): A sparce matrix


    Return
    ------
        A (scipy.sparse.spmatrix): A sparce matrix
    """
    if A.getformat() == "csc":
        return sps.csc_matrix((A.data, A.indices, A.indptr), shape=A.shape)
    elif A.getformat() == "csr":
        return sps.csr_matrix((A.data, A.indices, A.indptr), shape=A.shape)
    else:
        return A.copy()


def stack_diag(A: sps.spmatrix, B: sps.spmatrix) -> sps.spmatrix:
    """
    Create a new matrix C that contains matrix A and B at the diagonal:
    C = [[A, 0], [0, B]]
    This function is equivalent to
    sps.block_diag((A, B), format=A.format), but does not change the ordering
    of the A.indices or B.indices

    Parameters:
    -----------
    A (scipy.sparse.spmatrix): A sparce matrix
    B (scipy.sparse.spmatrix): A sparce matrix

    Return
    ------
    None


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
    """
    Function for slicing sparse matrix along rows or columns.
    If A is a csc_matrix A will be sliced along columns, while if A is a
    csr_matrix A will be sliced along the rows.

    Parameters
    ----------
    A (scipy.sparse.csc/csr_matrix): A sparse matrix.
    slice_ind (np.ndarray): Array containing indices to be sliced

    Returns
    -------
    indices (np.ndarray): If A is csc_matrix:
                            The nonzero row indices or columns slice_ind
                          If A is csr_matrix:
                            The nonzero columns indices or rows slice_ind
    array_ind (np.ndarray or slice): The indices in the compressed storage format (csc
                            or csr) corresponding to the slice; so that, if A is csr,
                            A.indices[array_ind] gives the columns of the slice
                            (represented in indices), and the corresponding data can be
                            accessed as A.data[array_ind]. Only returned if
                            return_array_ind is True.

    Examples
    --------
    A = sps.csc_matrix(np.eye(10))
    rows = slice_indices(A, np.array([0,2,3]))
    """
    assert A.getformat() == "csc" or A.getformat() == "csr"
    if np.asarray(slice_ind).dtype == "bool":
        # convert to indices.
        # First check for dimension
        if slice_ind.size != A.indptr.size - 1:
            raise IndexError("boolean index did not match indexed array")
        slice_ind = np.where(slice_ind)[0]
    if isinstance(slice_ind, int):
        array_ind = slice(A.indptr[int(slice_ind)], A.indptr[int(slice_ind + 1)])
        indices: np.ndarray = A.indices[array_ind]
    elif slice_ind.size == 1:
        array_ind = slice(A.indptr[int(slice_ind)], A.indptr[int(slice_ind + 1)])
        indices = A.indices[array_ind]
    else:
        array_ind = mcolon(A.indptr[slice_ind], A.indptr[slice_ind + 1])
        indices = A.indices[array_ind]
    if return_array_ind:
        return indices, array_ind
    else:
        return indices


def slice_mat(A: sps.spmatrix, ind: np.ndarray) -> sps.spmatrix:
    """
    Function for slicing sparse matrix along rows or columns.
    If A is a csc_matrix A will be sliced along columns, while if A is a
    csr_matrix A will be sliced along the rows.

    Parameters
    ----------
    A (scipy.sparse.csc/csr_matrix): A sparse matrix.
    ind (np.array): Array containing indices to be sliced.

    Returns
    -------
    A_sliced (scipy.sparse.csc/csr_matrix): The sliced matrix
        if A is a csc_matrix A_sliced = A[:, ind]
        if A is a csr_matrix A_slice = A[ind, :]

    Examples
    --------
    A = sps.csc_matrix(np.eye(10))
    rows = slice_mat(A, np.array([0,2,3]))
    """
    assert A.getformat() == "csc" or A.getformat() == "csr"

    if np.asarray(ind).dtype == "bool":
        # convert to indices.
        # First check for dimension
        if ind.size != A.indptr.size - 1:
            raise IndexError("boolean index did not match indexed array")
        ind = np.where(ind)[0]
    if isinstance(ind, int):
        N = 1
        indptr = np.zeros(2)
        ind_slice = slice(A.indptr[int(ind)], A.indptr[int(ind + 1)])
    elif ind.size == 1:
        N = 1
        indptr = np.zeros(2)
        ind_slice = slice(A.indptr[int(ind)], A.indptr[int(ind + 1)])
    else:
        N = ind.size
        indptr = np.zeros(ind.size + 1)
        ind_slice = mcolon(A.indptr[ind], A.indptr[ind + 1])
    indices = A.indices[ind_slice]
    indptr[1:] = np.cumsum(A.indptr[ind + 1] - A.indptr[ind])
    data = A.data[ind_slice]

    if A.getformat() == "csc":
        return sps.csc_matrix((data, indices, indptr), shape=(A.shape[0], N))
    elif A.getformat() == "csr":
        return sps.csr_matrix((data, indices, indptr), shape=(N, A.shape[1]))


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
        A (sps.spmatrix): Matrix to be reformatted.

    Returns:
        sps.spmatrix: The matrix represented in optimal storage format.

    """
    if A.shape[0] > A.shape[1]:
        return A.tocsc()
    else:
        return A.tocsr()


def csr_matrix_from_blocks(
    data: np.ndarray, block_size: int, num_blocks: int
) -> sps.spmatrix:
    """Create a csr representation of a block diagonal matrix of uniform block size.

    The function is equivalent to, but orders of magnitude faster than, the call

        sps.block_diag(blocks)

    Parameters:
        data (np.array): Matrix values, sorted column-wise.
        block_size (int): The size of *all* the blocks.
        num_blocks (int): Number of blocks to be added.

    Returns:
        sps.csr_matrix: csr representation of the block matrix.

    Raises:
        ValueError: If the size of the data does not match the blocks size and number
            of blocks.

    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> block_size, num_blocks = 2, 2
        >>> csr_matrix_from_blocks(data, block_size, num_blocks).toarray()
        array([[1, 2, 0, 0],
               [3, 4, 0, 0],
               [0, 0, 5, 6],
               [0, 0, 7, 8]])

    """
    return _csx_matrix_from_blocks(data, block_size, num_blocks, sps.csr_matrix)


def csc_matrix_from_blocks(
    data: np.ndarray, block_size: int, num_blocks: int
) -> sps.spmatrix:
    """Create a csc representation of a block diagonal matrix of uniform block size.

    The function is equivalent to, but orders of magnitude faster than, the call

        sps.block_diag(blocks)

    Parameters:
        data (np.array): Matrix values, sorted column-wise.
        block_size (int): The size of *all* the blocks.
        num_blocks (int): Number of blocks to be added.

    Returns:
        sps.csc_matrix: csr representation of the block matrix.

    Raises:
        ValueError: If the size of the data does not match the blocks size and number
            of blocks.

    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> block_size, num_blocks = 2, 2
        >>> csc_matrix_from_blocks(data, block_size, num_blocks).toarray()
        array([[1, 3, 0, 0],
               [2, 4, 0, 0],
               [0, 0, 5, 7],
               [0, 0, 6, 8]])

    """
    return _csx_matrix_from_blocks(data, block_size, num_blocks, sps.csc_matrix)


def _csx_matrix_from_blocks(
    data: np.ndarray, block_size: int, num_blocks: int, matrix_format
) -> sps.spmatrix:
    """Create a csr representation of a block diagonal matrix of uniform block size.

    The function is equivalent to, but orders of magnitude faster than, the call

        sps.block_diag(blocks)

    Parameters:
        data (np.array): Matrix values, sorted column-wise.
        block_size (int): The size of *all* the blocks.
        num_blocks (int): Number of blocks to be added.
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
    # interpreted

    # The new columns or rows start with intervals of block_size
    indptr = np.arange(0, block_size**2 * num_blocks + 1, block_size)

    # To get the indices in the compressed storage format requires some more work
    if block_size > 1:
        # First create indices for each of the blocks
        #  The inner tile creates arrays
        #   [0, 1, ..., block_size-1, 0, 1, ... block_size-1, ... ]
        #   The size of the inner tile is block_size^2, and forms the indices of a
        # single block
        #  The outer tile repeats the inner tile, num_blocks times
        #  The size of base is thus block_size^2 * num_blocks
        base = np.tile(
            np.tile(np.arange(block_size), (block_size, 1)).reshape((1, -1)), num_blocks
        )[0]
        # Next, increase the index in base, so as to create a block diagonal matrix
        # the first block_size^2 elements (e.g. the elemnets of the first block are
        # unperturbed.
        # the next block_size elements are increased by block_size^2 etc.
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
) -> Union[sps.csr, sps.csc]:
    """
    Invert block diagonal matrix.

    Three implementations are available, either pure numpy, or a speedup using
    numba or cython. If none is specified, the function will try to use numba,
    then cython. The python option will only be invoked if explicitly asked
    for; it will be very slow for general problems.

    Parameters
    ----------
    mat: sps.csr or sps.csc matrix to be inverted.
    s: block size. Must be int64 for the numba acceleration to work
    method: Choice of method. Either numba (default), cython or 'python'.
        Defaults to None, in which case first numba, then cython is tried.

    Returns
    -------
    imat: Inverse matrix

    Raises
    -------
    ImportError: If numba or cython implementation is invoked without numba or
        cython being available on the system.

    """

    def invert_diagonal_blocks_python(a: sps.spmatrix, size: np.ndarray) -> np.ndarray:
        """
        Invert block diagonal matrix using pure python code.

        Parameters
        ----------
        a : Block diagonal sparse matrix
        size : Size of individual blocks

        Returns
        -------
        inv_a: Flattened nonzero values of the inverse matrix
        """

        # This function only supports CSR anc CSC format.
        if not (sps.isspmatrix_csr(a) or sps.isspmatrix_csc(a)):
            raise TypeError("Sparse array type not implemented: ", type(a))

        # Construction of simple data structures (low complexity)
        # Indices for block positions, flattened inverse block positions and nonzeros
        # Expanded block positions
        idx_blocks = np.cumsum([0] + list(size))
        # Expanded nonzero positions for flattened inverse blocks
        idx_inv_blocks = np.cumsum([0] + list(size * size))
        # Nonzero positions for the given matrix data (i.e. a.data)
        idx_nnz = np.searchsorted(a.indices, idx_blocks)

        # Retrieve global indices (low complexity)
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

        # flattened nonzero values of the dense inverse
        inv_a = np.zeros(idx_inv_blocks[-1])

        def operate_on_block(ib: int):
            """
            Retrieve, invert, and assign dense block inverse values.

            Parameters
            ----------
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

        Parameters
        ----------
        a : Block diagonal sparse matrix
        size : Size of individual blocks

        Returns
        -------
        inv_a: Flattened nonzero values of the inverse matrix
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
        """
        Invert block diagonal matrix by invoking numba acceleration of a simple
        for-loop based algorithm.

        Parameters
        ----------
        a : sps.csr matrix
        size : Size of individual blocks

        Returns
        -------
        inv_a: Flattened nonzero values of the inverse matrix
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
    """
    Construct block diagonal matrix based on matrix elements and block sizes.

    Parameters
    ----------
    vals: matrix values
    sz: size of matrix blocks

    Returns
    -------
    sps.csr matrix
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

    Example usage: Convert a full set of (row or column) indices of a
    sparse matrix into compressed storage.

    Acknowledgement: The code is heavily inspired by MRST's function with the
    same name, however, requirements on the shape of functions are probably
    somewhat different.

    Parameters:
        A (np.ndarray): Matrix to be compressed. Should be 2d. Compression
            will be along the second axis.

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


def invert_permutation(perm):
    """Invert permutation array.

    Parameters:
        perm : permutation array

    Returns:
        inv_perm: Permuted sparse array

    """

    x = np.empty_like(perm)
    x[perm] = np.arange(len(perm), dtype=perm.dtype)
    return x


def sparse_permute(
    a: sps.spmatrix, row_perm: np.ndarray, col_perm: np.ndarray, inplace_q: bool = False
) -> sps.spmatrix:
    """Permute a sparse array.

    Parameters:
        a : Sparse array
        row_perm : Rows permutation dense array
        col_perm : Columns permutation dense array
        inplace_q: Apply permutation in place

    Returns:
        a_perm: Permuted sparse array

    """

    # This function only supports CSR anc CSC format.
    if not (sps.isspmatrix_csr(a) or sps.isspmatrix_csc(a)):
        raise TypeError("Sparse array type not implemented: ", type(a))

    not_equal_length_q = row_perm.shape[0] != a.shape[0]
    if not_equal_length_q:
        raise IndexError(
            "Row Permutation should have length equal to the number of rows."
        )

    not_equal_length_q = col_perm.shape[0] != a.shape[1]
    if not_equal_length_q:
        raise IndexError(
            "Column Permutation should have length equal to the number of columns."
        )

    # Operating with unsigned int
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uintc
    row_perm = invert_permutation(row_perm).astype(dtype=np.uintc)
    col_perm = invert_permutation(col_perm).astype(dtype=np.uintc)

    o_indptr = a.indptr.astype(dtype=np.uintc)
    o_indices = a.indices.astype(dtype=np.uintc)

    # Retrieve global indices (low complexity)
    if sps.isspmatrix_csr(a):
        row_reps = o_indptr[1 : o_indptr.size] - o_indptr[0 : o_indptr.size - 1]
        rows = np.repeat(np.arange(a.shape[0], dtype=np.uintc), row_reps)
        cols = o_indices

        rows = row_perm[rows]
        cols = col_perm[cols]
        sorted_idx = np.argsort(rows).astype(dtype=np.uintc)

        count = np.bincount(rows)
        indptr = np.cumsum(np.insert(count, 0, 0)).astype(dtype=np.int32)
        indices = cols[sorted_idx].astype(dtype=np.int32)
        data = a.data[sorted_idx]

    else:
        col_reps = o_indptr[1 : o_indptr.size] - o_indptr[0 : o_indptr.size - 1]
        cols = np.repeat(np.arange(a.shape[1], dtype=np.uintc), col_reps)
        rows = a.indices

        rows = row_perm[rows]
        cols = col_perm[cols]
        sorted_idx = np.argsort(cols).astype(dtype=np.uintc)

        count = np.bincount(cols)
        indptr = np.cumsum(np.insert(count, 0, 0)).astype(dtype=np.int32)
        indices = rows[sorted_idx].astype(dtype=np.int32)
        data = a.data[sorted_idx]

    # Applies inplace directive
    if inplace_q:
        a.indptr = indptr
        a.indices = indices
        a.data = data
        return a
    else:
        if sps.isspmatrix_csr(a):
            return sps.csr_matrix((data, indices, indptr), shape=a.shape)
        else:
            return sps.csc_matrix((data, indices, indptr), shape=a.shape)
