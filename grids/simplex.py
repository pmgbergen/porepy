# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:31:31 2016

@author: keile
"""

import numpy as np
import scipy.sparse as sps
import scipy.spatial

from grid import Grid, GridType
from utils import setmembership
from utils import accumarray


class TriangleGrid(Grid):

    def __init__(self, p, T=None):

        self.dim = 2
        self.type = GridType.triangle

        # Transform points to column vector if necessary (scipy.Delaunay
        # requires this format)
        pdims = p.shape
        if pdims[0] < pdims[1]:
            p = p.T

        if p.shape[1] != 2:
            raise NotImplementedError("Have not yet implemented triangle grids "
                                      "embeded in 2D")
        if T is None:
            Tri = scipy.spatial.Delaunay(p)
            T = Tri.simplices

        self.nodes = p.T
        self.Nn = self.nodes.shape[1]
        assert self.Nn > 2   # Check of transposes of point array

        # Face node relations
        faceNodes = np.vstack((T[::, [0, 1]],
                               T[::, [1, 2]],
                               T[::, [2, 0]]))
        faceNodes.sort(axis=1)
        faceNodes, tmp, cellFaces = setmembership.unique_rows(faceNodes)

        self.Nf = faceNodes.shape[0]
        self.Nc = T.shape[0]

        nNodesPerFace = 2
        faceNodes = faceNodes.ravel(0)
        indptr = np.hstack((np.arange(0, nNodesPerFace*self.Nf, nNodesPerFace),
                           nNodesPerFace * self.Nf))
        data = np.ones(faceNodes.shape, dtype=bool)
        self.faceNodes = sps.csc_matrix((data, faceNodes, indptr),
                                        shape=(self.Nn, self.Nf))

        # Cell face relation
        nFacesPerCell = 3
        cellFaces = cellFaces.reshape(nFacesPerCell, self.Nc).ravel(1)
        indptr = np.hstack((np.arange(0, nFacesPerCell*self.Nc, nFacesPerCell),
                            nFacesPerCell * self.Nc))
        data = -np.ones(cellFaces.shape)
        tmp, sgns = np.unique(cellFaces, return_index=True)
        data[sgns] = 1
        self.cellFaces = sps.csc_matrix((data, cellFaces, indptr),
                                        shape=(self.Nf, self.Nc))

    def cell_node_matrix(self):
        """ Get cell-node relations in a Nc x 3 matrix
        Perhaps move this method to a superclass when tet-grids are implemented
        """

        # Absolute value needed since cellFaces can be negative
        cn = self.faceNodes * np.abs(self.cellFaces) * sps.eye(self.Nc)
        row, col = cn.nonzero()
        scol = np.argsort(col)

        # Consistency check
        assert np.all(accumarray.accum(col, np.ones(col.size)) ==
                      (self.dim + 1))

        return row[scol].reshape(self.Nc, 3)


