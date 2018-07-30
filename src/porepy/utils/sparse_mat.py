"""
module for operations on sparse matrices
"""
import numpy as np
import scipy.sparse as sps

from porepy.utils.mcolon import mcolon


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

    keep = np.ones(A.data.size, dtype=np.bool)
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


def slice_indices(A, slice_ind):
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
    if isinstance(slice_ind, int):
        indices = A.indices[
            slice(A.indptr[int(slice_ind)], A.indptr[int(slice_ind + 1)])
        ]
    elif slice_ind.size == 1:
        indices = A.indices[
            slice(A.indptr[int(slice_ind)], A.indptr[int(slice_ind + 1)])
        ]
    else:
        indices = A.indices[mcolon(A.indptr[slice_ind], A.indptr[slice_ind + 1])]
    return indices


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
