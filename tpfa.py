# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:09:29 2016

@author: keile
"""

import numpy as np
import scipy.sparse as sps


def tpfa(g, k, bc, faces=None):
    """  Discretize the second order elliptic equation using two-point flux

    Args:
        g (grid): Grid, or a subclass, with geometry fields computed.
        k (second_order_tensor): Permeability. Cell-wise.
        bc (): Boundary conditions
        faces (np.array, int): Index of faces where TPFA should be applied.
            Currently unused, and defaults to all faces in the grid.
    """
    fi, ci, sgn = sps.find(g.cell_faces)

    # Normal vectors and permeability for each face (here and there side)
    n = g.face_normals[:, fi]
    n *= sgn
    perm = k.perm[::, ::, ci]

    # Distance from face center to cell center
    fc_cc = g.face_centers[::, fi] - g.cell_centers[::, ci]

    # Transpose normal vectors to match the shape of K

    nk = perm * n
    nk = nk.sum(axis=0)
    nk *= fc_cc
    t_face = nk.sum(axis=0)

    dist_face_cell = np.power(fc_cc, 2).sum(axis=0)
    
    t_face = np.divide(t_face, dist_face_cell)

    # Return horamonic average
    t = 1 / np.bincount(fi, weights=1/t_face)
    t[bc.is_neu] = 0
    flux = sps.coo_matrix((t[fi] * sgn, (fi, ci)))
    
    return flux
