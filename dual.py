# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""

import numpy as np
import scipy.sparse as sps

#------------------------------------------------------------------------------#

def matrix(g, k, bc):
    """  Discretize the second order elliptic equation using dual virtual
    element method.

    Args:
        g (grid): Grid, or a subclass, with geometry fields computed.
        k (second_order_tensor): Permeability. Cell-wise.
        bc (): Boundary conditions
    """
    faces, cells, sgn = sps.find(g.cell_faces)
    tol = 1e-10

    # Normal vectors, permeability, and diameter for each face of cell
    diams = g.cell_diameters()

    size = np.sum(np.square(g.cell_faces.indptr[1:] - g.cell_faces.indptr[:-1]))
    I = np.empty(size,dtype=np.int)
    J = np.empty(size,dtype=np.int)
    data = np.empty(size)
    idx = 0

    for c in np.arange(g.num_cells):
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])

        c_center = g.cell_centers[:,c]
        mono = np.array([lambda pt,i=i: (pt[i] - c_center[i])/diams[c] \
                                                     for i in np.arange(g.dim)])
        grad = np.eye(g.dim)/diams[c]

        K = k.perm[0:g.dim, 0:g.dim, c]
        normals = g.face_normals[0:g.dim, faces[loc]]

        # local matrix D
        D = np.array([np.dot(normals.T, np.dot(K, g)) for g in grad]).T

        # local matrix G
        G = np.dot(grad, np.dot(K, grad.T))*g.cell_volumes[c]

        # local matrix F
        faces_loc = faces[loc]
        sgn_loc = sgn[loc]
        F = np.array([ s*m( g.face_centers[:,f] ) for m in mono \
                        for s,f in zip(sgn_loc,faces_loc)] ).reshape((g.dim,-1))

        assert np.all( np.abs( G - np.dot(F,D) ) < tol )

        # local matrix Pi
        Pi_s = np.dot(np.linalg.inv(G), F)
        ndof = faces_loc.size
        I_Pi = np.eye(ndof) - np.dot(D, Pi_s)

        # local Hdiv-Stiffness matrix
        w = np.linalg.norm(np.linalg.inv(K),np.inf)
        A = np.dot(Pi_s.T, np.dot(G, Pi_s)) + w * np.dot(I_Pi.T, I_Pi)

        # save values for Hdiv-Stiffness matrix
        cols = np.tile(faces_loc, (ndof,1))
        loc_idx = slice(idx,idx+cols.size)
        I[loc_idx] = cols.T.ravel()
        J[loc_idx] = cols.ravel()
        data[loc_idx] = A.ravel()
        idx += cols.size

    # construct the global matrices
    mass = sps.coo_matrix((data,(I,J))).tocsr()
    div = -g.cell_faces.T

    return sps.bmat([[mass, div.T],[div,None]])

#------------------------------------------------------------------------------#

def rhs(g, f, bc):
    size = g.num_faces + g.num_cells
    rhs = np.zeros(size)
    rhs[size-g.num_cells:] = -np.multiply( g.cell_volumes, f( g.cell_centers ) )
    return rhs

#------------------------------------------------------------------------------#

def extract(up, g):
    return up[:g.num_faces], up[g.num_faces:]

#------------------------------------------------------------------------------#

def projectU(u, g, k):
    """  Discretize the second order elliptic equation using dual virtual
    element method.

    Args:
        g (grid): Grid, or a subclass, with geometry fields computed.
        k (second_order_tensor): Permeability. Cell-wise.
    """
    faces, cells, sgn = sps.find(g.cell_faces)
    tol = 1e-10

    # Normal vectors, permeability, and diameter for each face of cell
    diams = g.cell_diameters()

    P0u = np.zeros((3,g.num_cells))
    idx = 0

    for c in np.arange(g.num_cells):
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])

        c_center = g.cell_centers[:,c]
        mono = np.array([lambda pt,i=i: (pt[i] - c_center[i])/diams[c] \
                                                     for i in np.arange(g.dim)])
        grad = np.eye(g.dim)/diams[c]

        K = k.perm[0:g.dim, 0:g.dim, c]
        normals = g.face_normals[0:g.dim, faces[loc]]

        # local matrix D
        D = np.array([np.dot(normals.T, np.dot(K, g)) for g in grad]).T

        # local matrix G
        G = np.dot(grad, np.dot(K, grad.T))*g.cell_volumes[c]

        # local matrix F
        faces_loc = faces[loc]
        sgn_loc = sgn[loc]
        F = np.array([ s*m( g.face_centers[:,f] ) for m in mono \
                        for s,f in zip(sgn_loc,faces_loc)] ).reshape((g.dim,-1))

        assert np.all( np.abs( G - np.dot(F,D) ) < tol )

        # local matrix Pi
        Pi_s = np.dot(np.linalg.inv(G), F)

        # extract the velocity for the current cell
        P0u[:g.dim,c] = np.dot(Pi_s, u[faces_loc]) / diams[c]
#        u_full = I_rotation * [ u_tang; 0 ];
#        O_Pu_h( cellId, : ) = u_full';

    return P0u

#------------------------------------------------------------------------------#
