"""
module for operations on sparse matrices
"""
import numpy as np
import scipy.sparse as sps

from porepy.utils.mcolon import mcolon


def zero_columns(A, cols):
    '''
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


    '''

    if A.getformat() != 'csc':
        raise ValueError('Need a csc matrix')

    indptr = A.indptr
    col_indptr = mcolon(indptr[cols], indptr[cols + 1])
    A.data[col_indptr] = 0


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
    assert A.getformat() == 'csc' or A.getformat() == 'csr'
    if isinstance(slice_ind, int):
        indices = A.indices[slice(
            A.indptr[int(slice_ind)], A.indptr[int(slice_ind + 1)])]
    elif slice_ind.size == 1:
        indices = A.indices[slice(
            A.indptr[int(slice_ind)], A.indptr[int(slice_ind + 1)])]
    else:
        indices = A.indices[mcolon(
            A.indptr[slice_ind], A.indptr[slice_ind + 1])]
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
    assert A.getformat() == 'csc' or A.getformat() == 'csr'

    if isinstance(ind, int):
        N = 1
        indptr = np.zeros(2)
        ind_slice = slice(
            A.indptr[int(ind)], A.indptr[int(ind + 1)])
    elif ind.size == 1:
        N = 1
        indptr = np.zeros(2)
        ind_slice = slice(
            A.indptr[int(ind)], A.indptr[int(ind + 1)])
    else:
        N = ind.size
        indptr = np.zeros(ind.size + 1)
        ind_slice = mcolon(
            A.indptr[ind], A.indptr[ind + 1])

    indices = A.indices[ind_slice]
    indptr[1:] = np.cumsum(A.indptr[ind + 1] - A.indptr[ind])
    data = A.data[ind_slice]

    if A.getformat() == 'csc':
        return sps.csc_matrix((data, indices, indptr), shape=(A.shape[0], N))
    elif A.getformat() == 'csr':
        return sps.csr_matrix((data, indices, indptr), shape=(N, A.shape[1]))
