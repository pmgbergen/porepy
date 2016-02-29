# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:09:29 2016

@author: keile
"""

import numpy as np
import scipy.sparse as sps

from core.constit import second_order_tensor as perm
from utils import accumarray

def tpfa(g, K, bc=None, faces=None):
    """  Discretize the second order elliptic equation using two-point flux

    Args
        g (grid): Grid, or a subclass, with geometry fields computed.
        K (second_order_tensor): Permeability. Cell-wise.
        bc (): Boundary conditions
        faces (np.array, int): Index of faces where TPFA should be applied.
            Currently unused, and defaults to all faces in the grid.
    """
    fi, ci, sgn = sps.find(g.cellFaces)

    # Normal vectors and permeability for each face (here and there side)
    n = g.faceNormals[:, fi]
    n *= sgn
    perm = K.perm[::, ::, ci]

    # Distance from face center to cell center
    fc_cc = g.faceCenters[::, fi] - g.cellCenters[::, ci]

    # Transpose normal vectors to match the shape of K

    nK = perm * n
    nK = nK.sum(axis=0)    
    nK *= fc_cc
    T = nK.sum(axis=0)    

    nrmFC = np.power(fc_cc, 2).sum(axis=0)
    
    T= np.divide(T,nrmFC)

    # Return horamonic average
    T = 1 / accumarray.accum(fi, 1/T)
    T[bc.isNeu] = 0
    F = sps.coo_matrix((T[fi] * sgn,(fi,ci)))
    
    return F
    
