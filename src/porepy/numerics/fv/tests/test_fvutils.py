import numpy as np
import scipy.sparse as sps

from fvdiscr import fvutils
from core.grids import structured, simplex


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


def test_subcell_topology_2d_cart_1():
    x = np.ones(2)
    g = structured.CartGrid(x)

    subcell_topology = fvutils.SubcellTopology(g)

    assert np.all(subcell_topology.cno == 0)

    ncum = np.bincount(subcell_topology.nno,
                       weights=np.ones(subcell_topology.nno.size))
    assert np.all(ncum == 2)

    fcum = np.bincount(subcell_topology.fno,
                       weights=np.ones(subcell_topology.fno.size))
    assert np.all(fcum == 2)

    # There is only one cell, thus only unique subfno
    usubfno = np.unique(subcell_topology.subfno)
    assert usubfno.size == subcell_topology.subfno.size

    assert np.all(np.in1d(subcell_topology.subfno, subcell_topology.subhfno))


def test_subcell_mapping_2d_simplex_1():
    p = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    g = simplex.TriangleGrid(p)

    subcell_topology = fvutils.SubcellTopology(g)

    ccum = np.bincount(subcell_topology.cno,
                       weights=np.ones(subcell_topology.cno.size))
    assert np.all(ccum == 6)

    ncum = np.bincount(subcell_topology.nno,
                       weights=np.ones(subcell_topology.nno.size))
    assert ncum[0] == 2
    assert ncum[1] == 4
    assert ncum[2] == 2
    assert ncum[3] == 4

    fcum = np.bincount(subcell_topology.fno,
                       weights=np.ones(subcell_topology.fno.size))
    assert np.sum(fcum == 4) == 1
    assert np.sum(fcum == 2) == 4

    subfcum = np.bincount(subcell_topology.subfno,
                          weights=np.ones(subcell_topology.subfno.size))
    assert np.sum(subfcum == 2) == 2
    assert np.sum(subfcum == 1) == 8
