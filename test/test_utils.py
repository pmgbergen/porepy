""" Utility functions for the tests.

Access: from test import test_utils.
"""

import numpy as np
import scipy.sparse as sps


def permute_matrix_vector(A, rhs, block_dof, full_dof, grids, variables):
    """ Permute the matrix and rhs from assembler order to a specified order.

    Args:
        A: global solution matrix as returned by Assembler.assemble_matrix_rhs.
        rhs: global rhs vector as returned by Assembler.assemble_matrix_rhs.
        block_dof: Map coupling a (grid, variable) pair to an block index of A, as
            returned by Assembler.assemble_matrix_rhs.
        full_dof: Number of DOFs for each pair in block_dof, as returned by
            Assembler.assemble_matrix_rhs.

    Returns:
        sps.bmat(A.size): Permuted matrix.
        np.ndarray(b.size): Permuted rhs vector.
    """
    sz = len(block_dof)
    mat = np.empty((sz, sz), dtype=np.object)
    b = np.empty(sz, dtype=np.object)
    dof = np.empty(sz, dtype=np.object)
    # Initialize dof vector
    dof[0] = np.arange(full_dof[0])
    for i in range(1, sz):
        dof[i] = dof[i - 1][-1] + 1 + np.arange(full_dof[i])

    for row in range(sz):
        # Assembler index 0
        i = block_dof[(grids[row], variables[row])]
        b[row] = rhs[dof[i]]
        for col in range(sz):
            # Assembler index 1
            j = block_dof[(grids[col], variables[col])]
            # Put the A block indexed by i and j in mat of running indexes row and col
            mat[row, col] = A[dof[i]][:, dof[j]]

    return sps.bmat(mat, format="csr"), np.concatenate(tuple(b))
