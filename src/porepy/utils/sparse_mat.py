"""
module for operations on sparse matrices
"""
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.utils.mcolon import mcolon

module_sections = ["gridding", "discretization", "matrix", "numerics"]


@pp.time_logger(sections=module_sections)
def zero_columns(A, cols):
    """
    Function to zero out columns in matrix A. Note that this function does not
    change the sparcity structure of the matrix, it only changes the column
    values to 0

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


@pp.time_logger(sections=module_sections)
def zero_rows(A, rows):
    """
    Function to zero out rows in matrix A. Note that this function does not
    change the sparcity structure of the matrix, it only changes the row
    values to 0

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


@pp.time_logger(sections=module_sections)
def merge_matrices(A, B, lines):
    """
    Replace rows/coloms of matrix A with rows/cols of matrix B.
    If A and B are csc matrices this function is equivalent with
    A[:, lines] = B
    If A and B are csr matrices this funciton is equivalent iwth
    A[lines, :] = B

    Parameter
    ---------
    A (scipy.sparse.spmatrix): A sparce matrix
    B (scipy.sparse.spmatrix): A sparce matrix
    lines (ndarray): Lines of A to be replaced by B.

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

    if B.getformat() == "csc":
        if lines.size != B.shape[1]:
            raise ValueError("B.shape[1] must equal size of lines")
    if B.getformat() == "csr":
        if lines.size != B.shape[0]:
            raise ValueError("B.shape[0] must equal size of lines")

    if np.unique(lines).shape != lines.shape:
        raise ValueError("Can only merge unique lines")

    indptr = A.indptr
    indices = A.indices
    data = A.data

    ind_ix = mcolon(indptr[lines], indptr[lines + 1])

    # First we remove the old data
    num_rem = np.zeros(indptr.size, dtype=np.int32)
    num_rem[lines + 1] = indptr[lines + 1] - indptr[lines]
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
    num_added[lines + 1] = b_indptr[1:] - b_indptr[:-1]
    num_added = np.cumsum(num_added, dtype=num_added.dtype)

    rep = np.diff(b_indptr)
    indPos = np.repeat(indptr[lines], rep)

    A.indices = np.insert(indices, indPos, b_indices)
    A.data = np.insert(data, indPos, b_data)
    A.indptr = indptr + num_added


@pp.time_logger(sections=module_sections)
def stack_mat(A, B):
    """
    Stack matrix B at the end of matrix A.
    If A and B are csc matrices this function is equivalent to
        A = scipy.sparse.hstack((A, B))
    If A and B are csr matrices this function is equivalent to
        A = scipy.sparse.vstack((A, B))

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


@pp.time_logger(sections=module_sections)
def slice_indices(A, slice_ind, return_array_ind=False):
    """
    Function for slicing sparse matrix along rows or columns.
    If A is a csc_matrix A will be sliced along columns, while if A is a
    csr_matrix A will be sliced along the rows.

    Parameters
    ----------
    A (scipy.sparse.csc/csr_matrix): A sparse matrix.
    slice_ind (np.array): Array containing indices to be sliced

    Returns
    -------
    indices (np.array): If A is csc_matrix:
                            The nonzero row indices or columns slice_ind
                        If A is csr_matrix:
                            The nonzero columns indices or rows slice_ind
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
        indices = A.indices[array_ind]
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


@pp.time_logger(sections=module_sections)
def slice_mat(A, ind):
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


@pp.time_logger(sections=module_sections)
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


@pp.time_logger(sections=module_sections)
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


@pp.time_logger(sections=module_sections)
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
    if not data.size == block_size ** 2 * num_blocks:
        raise ValueError("Incompatible input to generate block matrix")

    # The block structure of the matrix allows for a unified construction of compressed
    # column and row matrices. The difference will simply be in how the data is
    # interpreted

    # The new columns or rows start with intervals of block_size
    indptr = np.arange(0, block_size ** 2 * num_blocks + 1, block_size)

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
            np.tile(np.arange(num_blocks), (block_size ** 2, 1)).reshape(
                (1, -1), order="F"
            )[0]
            * block_size
        )
        indices = base + block_increase
    else:
        indices = np.arange(num_blocks, dytpe=int)

    mat = matrix_format(
        (data, indices, indptr),
        shape=(num_blocks * block_size, num_blocks * block_size),
    )
    return mat
