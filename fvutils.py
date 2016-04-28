# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:04:16 2016

@author: eke001
"""

import numpy as np
import scipy.sparse as sps

from utils import matrix_compression
from utils import mcolon


def compute_dist_face_cell(g, cno, fno, nno, subhfno, eta):
    _, blocksz = matrix_compression.rlencode(np.vstack((cno, nno)))
    dims = g.dim

    i, j = np.meshgrid(subhfno, np.arange(dims))
    j += matrix_compression.rldecode(np.cumsum(blocksz)-blocksz[0], blocksz)
    
    eta_vec = eta*np.ones(fno.size)
    # Set eta values to zero at the boundary
    bnd = np.argwhere(np.abs(g.cellFaces).sum(axis=1).A.squeeze() 
                            == 1).squeeze()
    eta_vec[bnd] = 0
    cp = g.faceCenters[:, fno] + eta_vec * (g.nodes[:, nno] -
                                g.faceCenters[:, fno])
    dist = cp - g.cellCenters[:, cno]
    return sps.coo_matrix((dist.ravel(), (i.ravel(), j.ravel()))).tocsr()


def invert_diagonal_blocks(A, sz):
    # TODO: Try using numba on this code
    v = np.zeros(np.sum(np.square(sz)))
    p1 = 0
    p2 = 0
    for b in range(sz.size):
        n = sz[b]
        n2 = n * n
        i = p1 + np.arange(n+1)
        vals = A[i[0]:i[-1], i[0]:i[-1]].A
        v[p2 + np.arange(n2)] = np.linalg.inv(A[i[0]:i[-1], i[0]:i[-1]].A)
        p1 = p1 + n
        p2 = p2 + n2
    iA = block_diag_matrix(v, sz)
    return iA


def block_diag_matrix(v, sz):
    i, j = block_diag_index(sz)
    return sps.coo_matrix((v, (j, i))).tocsr()


def block_diag_index(m, n=None):
    """
    >>> m = np.array([2, 3])
    >>> n = np.array([1, 2])
    >>> i, j = block_diag_index(m, n)
    >>> i, j
    (array([0, 1, 2, 3, 4, 2, 3, 4]), array([0, 0, 1, 1, 1, 2, 2, 2]))
    >>> a = np.array([1, 3])
    >>> i, j = block_diag_index(a)
    >>> i, j
    (array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3]), array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
    """
    if n is None:
        n = m

    start = np.hstack((np.zeros(1), m))
    pos = np.cumsum(start)
    p1 = pos[0:-1]
    p2 = pos[1:]-1
    p1_full = matrix_compression.rldecode(p1, n)
    p2_full = matrix_compression.rldecode(p2, n)

    i = mcolon.mcolon(p1_full, p2_full)
    sumn = np.arange(np.sum(n))
    m_n_full = matrix_compression.rldecode(m, n)
    j = matrix_compression.rldecode(sumn, m_n_full)
    return i, j


def scalar_divergence(g):
    """
    Get divergence operator for a grid.

    The operator is easily accessible from the grid itself, so we keep it
    here for completeness.

    See also vector_divergence(g)

    Parameters
    ----------
    g grid

    Returns
    -------
    divergence operator
    """
    return g.cellFaces.T


def vector_divergence(g):
    """
    Get vector divergence operator for a grid g

    It is assumed that the first column corresponds to the x-equation of face
    0, second column is y-equation etc. (and so on in nd>2). The next column is
    then the x-equation for face 1. Correspondingly, the first row
    represents x-component in first cell etc.

    Parameters
    ----------
    g grid

    Returns
    -------
    vector_div (sparse csr matrix), dimensions: nd * (num_cells, num_faces)
    """
    # Scalar divergence
    scalar_div = g.cellFaces

    # Vector extension, convert to coo-format to avoid odd errors when one
    # grid dimension is 1 (this may return a bsr matrix)
    # The order of arguments to sps.kron is important.
    block_div = sps.kron(scalar_div, sps.eye(g.dim)).tocsr()

    return block_div.transpose()


if __name__ == '__main__':
    block_diag_index(np.array([2, 3]))
