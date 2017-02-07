# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""

import numpy as np
import scipy.sparse as sps
from compgeom import basics as cg

#------------------------------------------------------------------------------#

def matrix_rhs(g, k, f, bc=None):
    """  Discretize the second order elliptic equation using dual virtual
    element method, both matrix and source term.

    Args:
        g (grid): Grid, or a subclass, with geometry fields computed.
        k (second_order_tensor): Permeability. Cell-wise.
        f (np.array): Scalar source term.
        bc (): Boundary conditions (optional)
    """
    return matrix(g, k, bc), rhs(g, f, bc)

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
    c_centers, f_normals, f_centers, _ = cg.map_grid(g)

    weight = np.ones(g.num_cells) if g.dim != 1 else g.cell_volumes
    diams = g.cell_diameters()

    size = np.sum(np.square(g.cell_faces.indptr[1:] - g.cell_faces.indptr[:-1]))
    I = np.empty(size,dtype=np.int)
    J = np.empty(size,dtype=np.int)
    data = np.empty(size)
    idx = 0

    for c in np.arange(g.num_cells):
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
        faces_loc = faces[loc]

        K = k.perm[0:g.dim, 0:g.dim, c]
        sgn_loc = sgn[loc]
        normals = f_normals[:, faces_loc]

        A, _ = massHdiv(K, c_centers[:,c], g.cell_volumes[c],
                        f_centers[:,faces_loc], normals, sgn_loc, diams[c],
                        weight[c])

        # save values for Hdiv-Stiffness matrix
        cols = np.tile(faces_loc, (faces_loc.size,1))
        loc_idx = slice(idx,idx+cols.size)
        I[loc_idx] = cols.T.ravel()
        J[loc_idx] = cols.ravel()
        data[loc_idx] = A.ravel()
        idx += cols.size

    # construct the global matrices
    mass = sps.coo_matrix((data,(I,J)))
    div = -g.cell_faces.T

    return sps.bmat([[mass, div.T], [div,None]], format='csr')

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
    c_centers, f_normals, f_centers, R = cg.map_grid(g)

    diams = g.cell_diameters()

    P0u = np.zeros((3,g.num_cells))
    idx = 0

    for c in np.arange(g.num_cells):
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
        faces_loc = faces[loc]

        K = k.perm[0:g.dim, 0:g.dim, c]
        sgn_loc = sgn[loc]
        normals = f_normals[:, faces_loc]

        _, Pi_s = massHdiv(K, c_centers[:,c], g.cell_volumes[c],
                           f_centers[:,faces_loc], normals, sgn_loc, diams[c])

        # extract the velocity for the current cell
        P0u[:g.dim,c] = np.dot(Pi_s, u[faces_loc]) / diams[c]
        P0u[:,c] = np.dot(R.T, P0u[:,c])

    return P0u

#------------------------------------------------------------------------------#

def massHdiv(K, c_center, c_volume, f_centers, normals, sgn_loc, diam, weight=0):

    dim = K.shape[0]
    mono = np.array([lambda pt,i=i: (pt[i] - c_center[i])/diam \
                                                       for i in np.arange(dim)])
    grad = np.eye(dim)/diam

    # local matrix D
    D = np.array([np.dot(normals.T, np.dot(K, g)) for g in grad]).T

    # local matrix G
    G = np.dot(grad, np.dot(K, grad.T))*c_volume

    # local matrix F
    F = np.array([ s*m(f) for m in mono \
                    for s,f in zip(sgn_loc,f_centers.T)] ).reshape((dim,-1))

    assert np.allclose(G, np.dot(F,D))

    # local matrix Pi
    Pi_s = np.dot(np.linalg.inv(G), F)
    I_Pi = np.eye(f_centers.shape[1]) - np.dot(D, Pi_s)

    # local Hdiv-Mass matrix
    w = weight * np.linalg.norm(np.linalg.inv(K),np.inf)
    A = np.dot(Pi_s.T, np.dot(G, Pi_s)) + w * np.dot(I_Pi.T, I_Pi)

    return A, Pi_s

#------------------------------------------------------------------------------#
