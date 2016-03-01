# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 18:10:43 2016

@author: keile
"""

import numpy as np
import scipy.sparse as sps

from utils import matrix_compression 


def createMapping(g):

    g.cellFaces.sort_indices()
    fi, ci = g.cellFaces.nonzero()

    nfn = np.diff(g.faceNodes.indptr)
    cellsDuplicated = matrix_compression.rldecode(ci, nfn[fi])
    facesDuplicated = matrix_compression.rldecode(fi, nfn[fi])

    M = sps.coo_matrix((np.ones(fi.size), (fi, np.arange(fi.size))),
                       shape=(fi.max()+1, fi.size))
    nodesDuplicated = g.faceNodes * M
    nodesDuplicated = nodesDuplicated.indices

    indptr = g.faceNodes.indptr
    indices = g.faceNodes.indices
    data = np.arange(indices.size)+1
    subFaceMat = sps.csc_matrix((data, indices, indptr))
    subFaces = subFaceMat * M
    subFaces = subFaces.data-1

    # Sort data
    idx = np.argsort(nodesDuplicated)
    nno = nodesDuplicated[idx]
    cno = cellsDuplicated[idx]
    fno = facesDuplicated[idx]
    subfno = subFaces[idx].astype(int)
    subhfno = np.arange(idx.size, dtype='>i4')
    return nno, cno, fno, subfno, subhfno
