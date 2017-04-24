import numpy as np
cimport numpy as np

def inv_python(indptr, int[:] ind, double[:] data, long[:] sz):
    """
    Invert block matrices by explicitly forming local matrices. The code
    in itself is not efficient, but it is hopefully well suited for
    speeding up with numba.

    It may be possible to restruct the code to further help numba,
    this has not been investigated.

    The computation can easily be parallelized, consider this later.
    """

    cdef unsigned int iter1, iter2, iter3

    inv_vals = np.zeros(np.sum(np.square(sz)))

    # Bookkeeping
    num_per_row = indptr[1:] - indptr[0:-1]
    cdef int row_next = 0
    cdef int global_counter = 0
    cdef int block_size_prev = 0
    cdef int next_imat = 0

    # Loop over all blocks
    for iter1 in range(sz.size):
        n = sz[iter1]
        loc_mat = np.zeros((n, n))
        row_loc = 0
        # Fill in non-zero elements in local matrix
        for iter2 in range(n):
            for iter3 in range(num_per_row[row_next]):
                loc_col = ind[global_counter] - block_size_prev
                loc_mat[row_loc, loc_col] = data[global_counter]
                global_counter += 1
            row_next += 1
            row_loc += 1

        # Compute inverse. np.linalg.inv is supported by numba (May
        # 2016), it is not clear if this is the best option. To be
        # revised
        inv_mat = np.linalg.inv(loc_mat).reshape(n**2)

        loc_ind = next_imat + np.arange(n**2)
        inv_vals[loc_ind] = inv_mat
        # Update fields
        next_imat += n**2
        block_size_prev += n
    return inv_vals
