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
