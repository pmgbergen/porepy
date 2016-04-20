# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 18:10:43 2016

@author: keile
"""

import numpy as np
import scipy.sparse as sps

from utils import matrix_compression 


def create_mapping(g):

    g.cellFaces.sort_indices()
    fi, ci = g.cellFaces.nonzero()

    nfn = np.diff(g.faceNodes.indptr)
    cells_duplicated = matrix_compression.rldecode(ci, nfn[fi])
    faces_duplicated = matrix_compression.rldecode(fi, nfn[fi])

    M = sps.coo_matrix((np.ones(fi.size), (fi, np.arange(fi.size))),
                       shape=(fi.max()+1, fi.size))
    nodes_duplicated = g.faceNodes * M
    nodes_duplicated = nodes_duplicated.indices

    indptr = g.faceNodes.indptr
    indices = g.faceNodes.indices
    data = np.arange(indices.size)+1
    subFaceMat = sps.csc_matrix((data, indices, indptr))
    subFaces = subFaceMat * M
    subFaces = subFaces.data-1

    # Sort data
    idx = np.lexsort((subFaces, faces_duplicated, nodes_duplicated,
                      cells_duplicated))
    nno = nodes_duplicated[idx]
    cno = cells_duplicated[idx]
    fno = faces_duplicated[idx]
    subfno = subFaces[idx].astype(int)
    subhfno = np.arange(idx.size, dtype='>i4')

    return nno, cno, fno, subfno, subhfno
