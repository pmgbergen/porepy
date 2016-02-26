# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 21:23:30 2016

@author: keile
"""
import numpy as np
import scipy as sp
import scipy.sparse as sps


from grid import Grid, GridType


class TensorGrid(Grid):

    def __init__(self, x, y, z=None):
        self.type = GridType.tensor2D
        if z is None:
            self.create2Dgrid(x, y)
        else:
            raise NotImplementedError('Only 2D supported for now')

    def create2Dgrid(self, x, y):
        self.dim = 2

        sx = x.size - 1
        sy = y.size - 1

        numC = sx * sy
        numN = (sx + 1) * (sy + 1)
        numFX = (sx + 1) * sy
        numFY = sx * (sy + 1)
        numF = numFX + numFY

        self.Nc = numC
        self.Nf = numF
        self.Nn = numN

        xCoord, yCoord = sp.meshgrid(x, y)

        self.nodes = np.vstack((xCoord.flatten(), yCoord.flatten()))

        # Face nodes
        N = np.arange(0, numN).reshape(sy+1, sx+1)
        fn1 = N[:-1, ::].ravel()
        fn2 = N[1:, ::].ravel()
        faceNodesX = np.vstack((fn1, fn2)).ravel(1)

        fn1 = N[::, :-1].ravel()
        fn2 = N[::, 1:].ravel()
        faceNodesY = np.vstack((fn1, fn2)).ravel(1)

        nNodesPerFace = 2
        indptr = np.append(np.arange(0, nNodesPerFace*numF, nNodesPerFace),
                           nNodesPerFace * numF)
        faceNodes = np.hstack((faceNodesX, faceNodesY))
        data = np.ones(faceNodes.shape, dtype=bool)
        self.faceNodes = sps.csc_matrix((data, faceNodes, indptr),
                                        shape=(numN, numF))

        # Cell faces
        faceX = np.arange(numFX).reshape(sy, sx+1)
        faceY = numFX + np.arange(numFY).reshape(sy+1, sx)

        FW = faceX[::, :-1].ravel(0)
        FE = faceX[::, 1:].ravel(0)
        FS = faceY[:-1, ::].ravel(0)
        FN = faceY[1:, ::].ravel(0)

        cellFaces = np.vstack((FW, FE, FS, FN)).ravel(1)

        nFacesPerCell = 4
        indptr = np.append(np.arange(0, nFacesPerCell*numC, nFacesPerCell),
                           nFacesPerCell * numC)
        data = np.ones(cellFaces.shape, dtype=bool)
        self.cellFaces = sps.csc_matrix((data, cellFaces, indptr),
                                        shape=(numF, numC))

"""
        # Face neighbors
        # Let index -1 define boundary
        con = -np.ones((sy+2,sx+2))
        con[1:sy+1,1:sx+1] = np.arange(numC).reshape(sy,sx)

        NX1 = con[1:-1,:-1]
        NX2 = con[1:-1,1:]
        neighsX = np.vstack((NX1.ravel(),NX2.ravel()))

        NY1 = con[:-1,1:-1]
        NY2 = con[1:,1:-1]
        neighsY = np.vstack((NY1.ravel(),NY2.ravel()))

        neighs = np.hstack((neighsX,neighsY))

        a = 2

"""


class CartGrid(TensorGrid):
    def __init__(self, nx, physdims=None):

        self.type = GridType.cartesian_2D

        if any(i < 1 for i in nx):
            raise ValueError('All dimensions should be positive')

        if physdims is None:
            physdims = nx

        dims = nx.shape
        if dims[0] == 1:
            raise NotImplementedError('only 2D supported for now')
        elif dims[0] == 2:
            x = np.linspace(0, physdims[0], nx[0]+1)
            y = np.linspace(0, physdims[1], nx[1]+1)
            super(self.__class__, self).__init__(x, y)
        elif dims[0] == 3:
            raise NotImplementedError('only 2D supported for now')
        else:
            raise ValueError('Cartesian grid only implemented for up to three \
            dimensions')
