# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""

import numpy as np
from numpy.linalg import solve

import scipy.sparse as sps

from compgeom import basics as cg
from vem import dual

#------------------------------------------------------------------------------#

def matrix_rhs(g, k, f, bc=None):
    """  Discretize the second order elliptic equation using hybrid-dual virtual
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
    rhs = np.zeros(g.num_faces)

    idx = 0

    for c in np.arange(g.num_cells):
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
        faces_loc = faces[loc]
        ndof = faces_loc.size

        K = k.perm[0:g.dim, 0:g.dim, c]
        normals = np.multiply(np.tile(sgn[loc], (g.dim,1)),
                              f_normals[:, faces_loc])

        A, _ = dual.massHdiv(K, c_centers[:, c], g.cell_volumes[c],
                             f_centers[:, faces_loc], normals, np.ones(ndof),
                             diams[c], weight[c])
        B = -np.ones((ndof,1))
        C = np.eye(ndof,ndof)

        invA = np.linalg.inv(A)
        S = 1/np.dot(B.T, np.dot(invA, B))
        L = np.dot(np.dot(invA, np.dot(B, np.dot(S, B.T))), invA)
        L = np.dot(np.dot(C.T, L - invA), C)

        f_loc = f[c]*g.cell_volumes[c]
        rhs[faces_loc] += np.dot(C.T, np.dot(invA, np.dot(B, np.dot(S,
                                                                  f_loc))))[:,0]

        # save values for hybrid matrix
        cols = np.tile(faces_loc, (faces_loc.size,1))
        loc_idx = slice(idx,idx+cols.size)
        I[loc_idx] = cols.T.ravel()
        J[loc_idx] = cols.ravel()
        data[loc_idx] = L.ravel()
        idx += cols.size

    # construct the global matrices
    H = sps.coo_matrix((data,(I,J))).tocsr()

    # Apply the boundary conditions
    faces_bd = g.get_boundary_faces()
    H[faces_bd, :] *= 0
    H[faces_bd, faces_bd] = 1
    rhs[faces_bd] = np.sin(2*np.pi*f_centers[0,faces_bd])*\
                    np.sin(2*np.pi*f_centers[1,faces_bd])

    return H, rhs

#------------------------------------------------------------------------------#

def computePU(g, l, k, f):
    """  Discretize the second order elliptic equation using hybrid-dual virtual
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

    p = np.zeros(g.num_cells)
    u = np.zeros(g.num_faces)

    for c in np.arange(g.num_cells):
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
        faces_loc = faces[loc]
        ndof = faces_loc.size

        K = k.perm[0:g.dim, 0:g.dim, c]
        normals = np.multiply(np.tile(sgn[loc], (g.dim,1)),
                              f_normals[:, faces_loc])

        A, _ = dual.massHdiv(K, c_centers[:, c], g.cell_volumes[c],
                             f_centers[:, faces_loc], normals, np.ones(ndof),
                             diams[c], weight[c])
        B = -np.ones((ndof,1))
        C = np.eye(ndof,ndof)

        S = 1/np.dot(B.T, solve(A, B))
        f_loc = f[c]*g.cell_volumes[c]

        p[c] = np.dot(S, f_loc-np.dot(B.T, solve(A, np.dot(C, l[faces_loc]))))
        u[faces_loc] = -np.multiply(sgn[loc], solve(A, np.dot(B, p[c]) +
                                                       np.dot(C, l[faces_loc])))

    return p, u

