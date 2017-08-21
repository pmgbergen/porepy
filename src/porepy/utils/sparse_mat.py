"""
module for operations on sparse matrices
"""
from porepy.utils.mcolon import mcolon


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
    indices = A.indices[mcolon(A.indptr[slice_ind], A.indptr[slice_ind + 1])]
    return indices
