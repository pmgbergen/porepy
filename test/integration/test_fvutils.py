import numpy as np
import scipy.sparse as sps

from porepy_new.src.porepy.numerics.fv import fvutils
from porepy_new.src.porepy.grids import structured, simplex


def test_block_matrix_inverters_full_blocks():
    """
    Test inverters for block matrices

    """

    a = np.vstack((np.array([[1, 3], [4, 2]]), np.zeros((3, 2))))
    b = np.vstack((np.zeros((2, 3)),
                   np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]])))
    block = sps.csr_matrix(np.hstack((a, b)))

    sz = np.array([2, 3], dtype='i8')
    iblock_python = fvutils.invert_diagonal_blocks(block, sz, 'python')
    iblock_numba = fvutils.invert_diagonal_blocks(block, sz)
    iblock_ex = np.linalg.inv(block.toarray())

    assert np.allclose(iblock_ex, iblock_python.toarray())
    assert np.allclose(iblock_ex, iblock_numba.toarray())


def test_block_matrix_invertes_sparse_blocks():
    """
    Invert the matrix

    A = [1 2 0 0 0
         3 0 0 0 0
         0 0 3 0 3
         0 0 0 7 0
         0 0 0 1 2]

    Contrary to test_block_matrix_inverters_full_blocks, the blocks of A
    will be sparse. This turned out to give problems
    """

    rows = np.array([0, 0, 1, 2, 2, 3, 4, 4])
    cols = np.array([0, 1, 0, 2, 4, 3, 3, 4])
    data = np.array([1, 2, 3, 3, 3, 7, 1, 2], dtype=np.float64)
    block = sps.coo_matrix((data, (rows, cols))).tocsr()
    sz = np.array([2, 3], dtype='i8')

    iblock_python = fvutils.invert_diagonal_blocks(block, sz, 'python')
    iblock_numba = fvutils.invert_diagonal_blocks(block, sz)
    iblock_ex = np.linalg.inv(block.toarray())

    assert np.allclose(iblock_ex, iblock_python.toarray())
    assert np.allclose(iblock_ex, iblock_numba.toarray())

