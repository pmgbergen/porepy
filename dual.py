# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""

import numpy as np
import scipy.sparse as sps
from compgeom import basics as cg

#------------------------------------------------------------------------------#

def matrix(g, k, bc=None):
    """  Discretize the second order elliptic equation using dual virtual
    element method.

    Args:
        g (grid): Grid, or a subclass, with geometry fields computed.
        k (second_order_tensor): Permeability. Cell-wise.
        bc (): Boundary conditions (optional)
    """
    faces, cells, sgn = sps.find(g.cell_faces)

    cell_centers = g.cell_centers
    face_normals = g.face_normals
    face_centers = g.face_centers

    if g.dim != 3:
        R = cg.project_plane_matrix(g.nodes)
        cell_centers = np.dot(R, cell_centers)
        face_normals = np.dot(R, face_normals)
        face_centers = np.dot(R, face_centers)

    diams = g.cell_diameters()

    size = np.sum(np.square(g.cell_faces.indptr[1:] - g.cell_faces.indptr[:-1]))
    I = np.empty(size,dtype=np.int)
    J = np.empty(size,dtype=np.int)
    data = np.empty(size)
    idx = 0

    for c in np.arange(g.num_cells):
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
        mono = np.array([lambda pt,i=i: (pt[i] - cell_centers[i, c])/diams[c] \
                                                     for i in np.arange(g.dim)])
        grad = np.eye(g.dim)/diams[c]

        K = k.perm[0:g.dim, 0:g.dim, c]
        normals = face_normals[0:g.dim, faces[loc]]

        # local matrix D
        D = np.array([np.dot(normals.T, np.dot(K, g)) for g in grad]).T

        # local matrix G
        G = np.dot(grad, np.dot(K, grad.T))*g.cell_volumes[c]

        # local matrix F
        faces_loc = faces[loc]
        sgn_loc = sgn[loc]
        F = np.array([ s*m( face_centers[:,f] ) for m in mono \
                        for s,f in zip(sgn_loc,faces_loc)] ).reshape((g.dim,-1))

        assert np.allclose(G, np.dot(F,D))

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
    mass = sps.coo_matrix((data,(I,J)))
    div = -g.cell_faces.T

    return sps.bmat([[mass, div.T],[div,None]], format='csr')

#------------------------------------------------------------------------------#

def rhs(g, f, bc=None):
    """  Discretize the source term for a dual virtual element method.

    Args:
        g (grid): Grid, or a subclass, with geometry fields computed.
        f (np.array): Scalar source term.
        bc (): Boundary conditions (optional)
    """

    size = g.num_faces + g.num_cells
    rhs = np.zeros(size)
    rhs[size-g.num_cells:] = -np.multiply(g.cell_volumes, f)
    return rhs

#------------------------------------------------------------------------------#

def extract(g, up):
    """  Extract the velocity and the pressure from a dual virtual element
    solution.

    Args:
        g (grid): Grid, or a subclass, with geometry fields computed.
        up (np.array): Solution, stored as [velocity,pressure]
    """
    return up[:g.num_faces], up[g.num_faces:]

#------------------------------------------------------------------------------#

def projectU(g, k, u):
    """  Project the velocity computed with a dual vem solver to obtain a
    piecewise constant vector field, one triplet for each cell.

    Args:
        g (grid): Grid, or a subclass, with geometry fields computed.
        k (second_order_tensor): Permeability. Cell-wise.
        u (np.array): Velocity computed from a dual virtual element method.
    """
    faces, cells, sgn = sps.find(g.cell_faces)

    cell_centers = g.cell_centers
    face_normals = g.face_normals
    face_centers = g.face_centers
    invR = np.eye(3)

    if g.dim != 3:
        R = cg.project_plane_matrix(g.nodes)
        invR = np.linalg.inv(R)
        cell_centers = np.dot(R, cell_centers)
        face_normals = np.dot(R, face_normals)
        face_centers = np.dot(R, face_centers)

    diams = g.cell_diameters()

    P0u = np.zeros((3,g.num_cells))
    idx = 0

    for c in np.arange(g.num_cells):
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
        mono = np.array([lambda pt,i=i: (pt[i] - cell_centers[i, c])/diams[c] \
                                                     for i in np.arange(g.dim)])
        grad = np.eye(g.dim)/diams[c]

        K = k.perm[0:g.dim, 0:g.dim, c]
        normals = face_normals[0:g.dim, faces[loc]]

        # local matrix D
        D = np.array([np.dot(normals.T, np.dot(K, g)) for g in grad]).T

        # local matrix G
        G = np.dot(grad, np.dot(K, grad.T))*g.cell_volumes[c]

        # local matrix F
        faces_loc = faces[loc]
        sgn_loc = sgn[loc]
        F = np.array([ s*m( face_centers[:,f] ) for m in mono \
                        for s,f in zip(sgn_loc,faces_loc)] ).reshape((g.dim,-1))

        assert np.allclose(G, np.dot(F,D))

        # local matrix Pi
        Pi_s = np.dot(np.linalg.inv(G), F)

        # extract the velocity for the current cell
        P0u[:g.dim,c] = np.dot(Pi_s, u[faces_loc]) / diams[c]
        P0u[:,c] = np.dot(invR, P0u[:,c])

    return P0u

#------------------------------------------------------------------------------#
