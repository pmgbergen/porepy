""""Task: Efficient inversion of Matrix, which is a permuted block-diagonal with blocks
of fixed size s, in csr format.

1. Permutation without permuation matrices (slicing of data, indptr and indices)
2. parallel inversion of blocks
3. reverting permutation without permutation matrices

Notes:

    - Sparse matrix -> blocks can have zero elements which do not appear in csr format

"""

import numpy as np
import scipy.sparse as sps
import numba as nu
import matplotlib.pyplot as plt

from scipy.linalg import block_diag

# block size / number of (scalar cell-wise) variables / number of equations per cell
block_size = 3
# number of cells per grid
nc = np.array([3, 2])
# DOF indices per variable
# Can actually be computed with knowledge about structure in AD parsing
dofs = np.array(
    [
        np.array([0, 1, 2, 9, 10]),  # dofs per variable, as returned by equ system
        np.array([3, 4, 5, 11, 12]),
        np.array([6, 7, 8, 13, 14]),
    ]
)


M1 = np.array(
      # |--------- grid 1 --------|----- grid 2 ----|
    [ # |--v11--|--v12---|--v13---|-v21-|-v22-|-v23-|  # ---|---
        [1, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], #  | | |
        [0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0], # g1 | |
        [0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0], #  | | |
                                                       # ---| eq1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 1, 0], # g2 | |
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 1], #  | | |
                                                       # ---|---
        [4, 0, 0, 5, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0], #  | | |
        [0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0], # g1 | |
        [0, 0, 4, 0, 0, 4, 0, 0, 6, 0, 0, 0, 0, 0, 0], #  | | |
                                                       # ---| eq2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 5, 0, 4, 0], # g2 | |
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 5, 0, 4], #  | | |
                                                       # ---|---
        [7, 0, 0, 8, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0], #  | | |
        [0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0], # g1 | |
        [0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 0, 0, 0, 0], #  | | |
                                                       # ---| eq3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 8, 0, 7, 0], # g2 | |
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 8, 0, 7], #  | | |
    ],                                                 # ---|---
    dtype=float
)
"""Example matrix, mimicing an AD.jac with 3 unknowns, on 2 grids and 3 equations.
27 x 27

For now, the blocks in block-diagonal form have no zero values.

- grid 1: 3 cells
- grid 2: 2 cells

All equations depend on all unknowns, defined on all grids.

"""

M_target = block_diag(
    *tuple(
        3 * [np.array([[1,2,3],[4,5,6],[7,8,9]])]
        + 2 * [np.array([[3,2,1],[6,5,4],[9,8,7]])]
    )
)
"""Matrix which should result from the algorithm performed on ``M1``.

A bloack-diagonal matrix, where the first 3 blocks (grid 1)
look like
[
    [1,2,3],
    [4,5,6],
    [7,8,9],
]
and the last two blocks (grid 2) look like
[
    [3,2,1],
    [6,5,4],
    [9,8,7],
]

"""


@nu.njit(
    nu.types.Tuple(nu.f8[:], nu.f8[:], nu.f8[:])(
        nu.f8[:],
        nu.f8[:],
        nu.f8[:],
        nu.i8,
        nu.f8[:,:],
    ),
    cache=True,
)
def permute(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    s: int,
    dofs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a matrix in csr format, permutes the representation into block-diagonal
    form.

    Meant for Jacobian assembled by ``porepy.ad``.

    Assumes that all variables are part of all equations and grids, i.e. blocks are
    square and the system of equations is not over- or underdetermined.

    Parameters:


    """
    assert M1.shape[0] % s == 0  # simple validation of input
    assert M1.shape[1] % s == 0
    assert M1.shape[0] == M1.shape[1]

    nb = int(M1.shape[0] / s)  # number of blocks


M1s = sps.csr_matrix(M1)
d, idx, iptr = permute(M1s.data, M1s.indices, M1s.indptr, block_size, dofs)

M2 = sps.csr_matrix((d, idx, iptr), shape=M1s.shape)
M2 = np.eye(M1.shape[0], dtype=float)

print("--- Input Matrix:")
print(M1)
print("--- Target Matrix:")
print(M_target)
print("--- Output Matrix:")
print(M2)
print("---")

fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax1.set_title("Input Matrix")
img = ax1.spy(M1)
ax2 = fig.add_subplot(1,3,2)
ax2.spy(M_target)
ax2.set_title("Target matrix")
ax3 = fig.add_subplot(1,3,3)
ax3.spy(M2)
ax3.set_title("Output Matrix")

fig.text(0.25, 0.1, f"Number of entries deviating value-wise: {(M_target != M2).sum()}")
fig.tight_layout()
fig.show()

print('done')