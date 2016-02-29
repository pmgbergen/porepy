# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:55:43 2016

@author: keile
"""

import numpy as np
from utils import accumarray
from enum import Enum
from scipy import sparse as sps


class GridType(Enum):
    """
    Enumeration to define types of grids. Not quite sure what I want to use
    them for; right now the primary motivation is to test which type of grid
    this is.
    Possible future usage could assign dimension to the fields, etc.
    """
    triangle = 1
    cartesian_2D = 2
    tensor_2D = 3


class Grid(object):

    def __init__(self, nodes, faces, cells):
        self.dim = nodes.dim
        self.nodes = nodes
        self.faces = faces
        self.cells = cells
        self.name = 'Hei'

    def computeGeometry(self):
        if self.dim == 2:
            self.computeGeometry2D()
        else:
            raise NotImplementedError('3D not handled yet')

    def computeGeometry2D(self):

        xn = self.nodes

        fn = self.faceNodes.indices
        edge1 = fn[::2]
        edge2 = fn[1::2]

        xe1 = xn[:, edge1]
        xe2 = xn[:, edge2]

        edgeLengthX = xe2[0] - xe1[0]
        edgeLengthY = xe2[1] - xe1[1]
        self.faceAreas = np.sqrt(np.power(edgeLengthX, 2) +
                                 np.power(edgeLengthY, 2))
        self.faceCenters = 0.5 * (xe1 + xe2)
        self.faceNormals = np.vstack((edgeLengthY, -edgeLengthX))

        cellFaces, cellno = self.cellFaces.nonzero()

        nCellFaces = accumarray.accum(cellno, np.ones(cellno.shape))

        cx = accumarray.accum(cellno, self.faceCenters[0, cellFaces])
        cy = accumarray.accum(cellno, self.faceCenters[1, cellFaces])
        cCenters = np.vstack((cx, cy)) / nCellFaces

        a = xe1[:, cellFaces] - cCenters[:, cellno]
        b = xe2[:, cellFaces] - cCenters[:, cellno]

        subVol = 0.5 * np.abs(a[0] * b[1] - a[1] * b[0])
        self.cellVolumes = accumarray.accum(cellno, subVol)

        subCentroid = (cCenters[:, cellno] + 2 *
                       self.faceCenters[:, cellFaces]) / 3

        ccx = accumarray.accum(cellno, subVol * subCentroid[0])
        ccy = accumarray.accum(cellno, subVol * subCentroid[1])

        self.cellCenters = np.vstack((ccx, ccy)) / self.cellVolumes

        # Ensure that normal vector direction corresponds with sign convention
        # in self.cellFaces
        def nrm(u):
            return np.sqrt(u[0]*u[0] + u[1]*u[1])

        [fi, ci, val] = sps.find(self.cellFaces)
        _, idx = np.unique(fi, return_index=True)
        sgn = val[idx]
        fc = self.faceCenters[:, fi[idx]]
        cc = self.cellCenters[:, ci[idx]]
        v = fc - cc
        # Prolong the vector from cell to face center in the direction of the
        # normal vector. If the prolonged vector is shorter, the normal should
        # flipped
        vn = v + nrm(v) * self.faceNormals[:, fi[idx]] * 0.001
        flip = np.logical_or(np.logical_and(nrm(v) > nrm(vn), sgn > 0),
                             np.logical_and(nrm(v) < nrm(vn), sgn < 0))
        self.faceNormals[:, flip] *= -1  # Is it correct not to have fi here?
