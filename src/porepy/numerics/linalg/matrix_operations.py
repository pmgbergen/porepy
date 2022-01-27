"""
module for operations on sparse matrices
"""
from typing import Optional, Union, Tuple
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.utils.mcolon import mcolon


def zero_columns(
    A: sps.csc_matrix, cols: np.ndarray, diag: Optional[np.ndarray] = None
) -> None:
    """
    Function to zero out columns in matrix A. Note that this function does not
    change the sparcity structure of the matrix, it only changes the column
    values to 0.

    The matrix is modified in place.

    Parameter
    ---------
    A (scipy.sparse.spmatrix): A sparce matrix
    cols (ndarray): A numpy array of columns that should be zeroed
    diag (np.ndarray, double, optional): Values to be set to the diagonal
        on the eliminated cols.

    Return
    ------
    None


    """

    if A.getformat() != "csc":
        raise ValueError("Need a csc matrix")

    indptr = A.indptr
    col_indptr = mcolon(indptr[cols], indptr[cols + 1])
    A.data[col_indptr] = 0

    if diag is not None:
        # now we set the diagonal
        diag_vals = np.zeros(A.shape[1])
        diag_vals[cols] = diag
        A += sps.dia_matrix((diag_vals, 0), shape=A.shape)


def zero_rows(
    A: sps.csr_matrix, rows: np.ndarray, diag: Optional[np.ndarray] = None
) -> None:
    """
    Function to zero out rows in matrix A. Note that this function does not
    change the sparcity structure of the matrix, it only changes the row
    values to 0.

    The matrix is modified in place.

    Parameter
    ---------
    A (scipy.sparse.spmatrix): A sparce matrix
    rows (ndarray): A numpy array of columns that should be zeroed
    diag (np.ndarray, double, optional): Values to be set to the diagonal
        on the eliminated rows.

    Return
    ------
    None

    """

    if A.getformat() != "csr":
        raise ValueError("Need a csr matrix")

    indptr = A.indptr
    row_indptr = mcolon(indptr[rows], indptr[rows + 1])
    A.data[row_indptr] = 0

    if diag is not None:
        # now we set the diagonal
        diag_vals = np.zeros(A.shape[1])
        diag_vals[rows] = diag
        A += sps.dia_matrix((diag_vals, 0), shape=A.shape)


def merge_matrices(A: sps.spmatrix, B: sps.spmatrix, lines: np.ndarray) -> None:
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
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
        indices = np.arange(num_blocks, dtype=int)

    mat = matrix_format(
        (data, indices, indptr),
        shape=(num_blocks * block_size, num_blocks * block_size),
    )
    return mat


def invert_diagonal_blocks(
    mat: sps.spmatrix, s: np.ndarray, method: Optional[str] = None
) -> sps.spmatrix:
    """
    Invert block diagonal matrix.

    Three implementations are available, either pure numpy, or a speedup using
    numba or cython. If none is specified, the function will try to use numba,
    then cython. The python option will only be invoked if explicitly asked
    for; it will be very slow for general problems.

    Parameters
    ----------
    mat: sps.csr matrix to be inverted.
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

    def invert_diagonal_blocks_python(a: sps.spmatrix, sz: np.ndarray) -> np.ndarray:
        """
        Invert block diagonal matrix using pure python code.

        The implementation is slow for large matrices, consider to use the
        numba-accelerated method invert_invert_diagagonal_blocks_numba instead

        Parameters
        ----------
        A sps.crs-matrix, to be inverted
        sz - size of the individual blocks

        Returns
        -------
        inv_a inverse matrix
        """
        v = np.zeros(np.sum(np.square(sz)))
        p1 = 0
        p2 = 0
        for b in range(sz.size):
            n = sz[b]
            n2 = n * n
            i = p1 + np.arange(n + 1)
            # Picking out the sub-matrices here takes a lot of time.
            v[p2 + np.arange(n2)] = np.linalg.inv(
                a[i[0] : i[-1], i[0] : i[-1]].A
            ).ravel()
            p1 = p1 + n
            p2 = p2 + n2
        return v

    def invert_diagonal_blocks_cython(a: sps.spmatrix, sz: np.ndarray) -> np.ndarray:
        """Invert block diagonal matrix using code wrapped with cython."""
        try:
            import porepy.numerics.fv.cythoninvert as cythoninvert
        except ImportError:
            raise ImportError(
                """Compiled Cython module not available. Is cython installed?"""
            )

        a.sorted_indices()
        ptr = a.indptr
        indices = a.indices
        dat = a.data

        v = cythoninvert.inv_python(ptr, indices, dat, sz)
        return v

    def invert_diagonal_blocks_numba(a: sps.spmatrix, size: np.ndarray) -> np.ndarray:
        """
        Invert block diagonal matrix by invoking numba acceleration of a simple
        for-loop based algorithm.

        This approach should be more efficient than the related method
        invert_diagonal_blocks_python for larger problems.

        Parameters
        ----------
        a : sps.csr matrix
        size : Size of individual blocks

        Returns
        -------
        ia: inverse of a
        """
        try:
            import numba
        except ImportError:
            raise ImportError("Numba not available on the system")

        # Sort matrix storage before pulling indices and data
        a.sorted_indices()
        ptr = a.indptr
        indices = a.indices
        dat = a.data

        # Just in time compilation
        @numba.jit("f8[:](i4[:],i4[:],f8[:],i8[:])", nopython=True, cache=True)
        def inv_python(indptr, ind, data, sz):
            """
            Invert block matrices by explicitly forming local matrices. The code
            in itself is not efficient, but it is hopefully well suited for
            speeding up with numba.

            It may be possible to restruct the code to further help numba,
            this has not been investigated.

            The computation can easily be parallelized, consider this later.
            """

            # Index of where the rows start for each block.
            # block_row_starts_ind = np.hstack((np.array([0]),
            #                                   np.cumsum(sz[:-1])))
            block_row_starts_ind = np.zeros(sz.size, dtype=np.int32)
            block_row_starts_ind[1:] = np.cumsum(sz[:-1])

            # Number of columns per row. Will change from one column to the
            # next
            num_cols_per_row = indptr[1:] - indptr[0:-1]
            # Index to where the columns start for each row (NOT blocks)
            # row_cols_start_ind = np.hstack((np.zeros(1),
            #                                 np.cumsum(num_cols_per_row)))
            row_cols_start_ind = np.zeros(num_cols_per_row.size + 1, dtype=np.int32)
            row_cols_start_ind[1:] = np.cumsum(num_cols_per_row)

            # Index to where the (full) data starts. Needed, since the
            # inverse matrix will generally be full
            # full_block_starts_ind = np.hstack((np.array([0]),
            #                                    np.cumsum(np.square(sz))))
            full_block_starts_ind = np.zeros(sz.size + 1, dtype=np.int32)
            full_block_starts_ind[1:] = np.cumsum(np.square(sz))
            # Structure to store the solution
            inv_vals = np.zeros(np.sum(np.square(sz)))

            # Loop over all blocks
            for iter1 in range(sz.size):
                n = sz[iter1]
                loc_mat = np.zeros((n, n))
                # Fill in non-zero elements in local matrix
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

                # Compute inverse. np.linalg.inv is supported by numba (May
                # 2016), it is not clear if this is the best option. To be
                # revised
                inv_mat = np.ravel(np.linalg.inv(loc_mat))

                loc_ind = np.arange(
                    full_block_starts_ind[iter1], full_block_starts_ind[iter1 + 1]
                )
                inv_vals[loc_ind] = inv_mat
                # Update fields
            return inv_vals

        v = inv_python(ptr, indices, dat, size)
        return v

    # Remove blocks of size 0
    s = s[s > 0]
    # Variable to check if we have tried and failed with numba
    try_cython = False
    if method == "numba" or method is None:
        try:
            inv_vals = invert_diagonal_blocks_numba(mat, s)
        except np.linalg.LinAlgError:
            raise ValueError("Error in inversion of local linear systems")
        except Exception:
            # This went wrong, fall back on cython
            try_cython = True
    # Variable to check if we should fall back on python
    if method == "cython" or try_cython:
        try:
            inv_vals = invert_diagonal_blocks_cython(mat, s)
        except ImportError as e:
            raise e
    elif method == "python":
        inv_vals = invert_diagonal_blocks_python(mat, s)

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
    row, _ = block_diag_index(sz)
    # This line recovers starting indices of the rows.
    indptr = np.hstack(
        (np.zeros(1), np.cumsum(pp.utils.matrix_compression.rldecode(sz, sz)))
    ).astype("int32")
    return sps.csr_matrix((vals, row, indptr))


def block_diag_index(
    m: np.ndarray, n: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
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
    if n is None:
        n = m

    start = np.hstack((np.zeros(1, dtype="int"), m))
    pos = np.cumsum(start)
    p1 = pos[0:-1]
    p2 = pos[1:] - 1
    p1_full = pp.utils.matrix_compression.rldecode(p1, n)
    p2_full = pp.utils.matrix_compression.rldecode(p2, n)

    i = mcolon(p1_full, p2_full + 1)
    sumn = np.arange(np.sum(n))
    m_n_full = pp.utils.matrix_compression.rldecode(m, n)
    j = pp.utils.matrix_compression.rldecode(sumn, m_n_full)
    return i, j
