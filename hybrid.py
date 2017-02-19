# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""

import numpy as np
from numpy.linalg import solve

import scipy.sparse as sps

from core.solver.solver import *
from compgeom import basics as cg
from vem import dual

class HybridDualVEM(Solver):

#------------------------------------------------------------------------------#

    def ndof(self, g): return g.num_faces

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        """
        Return the matrix and righ-hand side for a discretization of a second order
        elliptic equation using hybdrid dual virtual element method.

        Parameters
        ----------
        g : grid
            Grid, or a subclass, with geometry fields computed.
        k : second_order_tensor)
            Permeability. Cell-wise.
        f : array (g.num_cells)
            Scalar source term.
        bc :
            Boundary conditions (optional)
        bc_val : dictionary
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.

        Return
        ------
        matrix: sparse csr (g.num_faces, g.num_faces)
            Spd matrix obtained from the discretization.
        rhs: array (g.num_faces)
            Right-hand side which contains the boundary conditions and the scalar
            source term.

        Examples
        --------
        b_faces_neu = ... # id of the Neumann faces
        b_faces_dir = ... # id of the Dirichlet faces
        bnd = bc.BoundaryCondition(g, np.hstack((b_faces_dir, b_faces_neu)),
                                ['dir']*b_faces_dir.size + ['neu']*b_faces_neu.size)
        bnd_val = {'dir': fun_dir(g.face_centers[:, b_faces_dir]),
                   'neu': fun_neu(f.face_centers[:, b_faces_neu])}

        H, rhs = hybrid.matrix_rhs(g, perm, f, bnd, bnd_val)
        l = sps.linalg.spsolve(H, rhs)
        u, p = hybrid.computeUP(g, l, perm, f)
        P0u = dual.projectU(g, perm, u)

        """
        if g.dim == 0:
            return sps.identity(self.ndof(g), format="csr"), np.zeros(1)

        k, f = data['k'], data['f']
        bc, bc_val = data.get('bc'), data.get('bc_val')

        faces, cells, sgn = sps.find(g.cell_faces)
        c_centers, f_normals, f_centers, _, _ = cg.map_grid(g)

        weight = np.ones(g.num_cells) if g.dim != 1 else g.cell_volumes
        diams = g.cell_diameters()

        size = np.sum(np.square(g.cell_faces.indptr[1:] - \
                                g.cell_faces.indptr[:-1]))
        I = np.empty(size,dtype=np.int)
        J = np.empty(size,dtype=np.int)
        data = np.empty(size)
        rhs = np.zeros(g.num_faces)

        idx = 0
        massHdiv = dual.DualVEM().massHdiv

        for c in np.arange(g.num_cells):
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
            faces_loc = faces[loc]
            ndof = faces_loc.size

            K = k.perm[0:g.dim, 0:g.dim, c]
            normals = np.multiply(np.tile(sgn[loc], (g.dim,1)),
                                  f_normals[:, faces_loc])

            A, _ = massHdiv(K, c_centers[:, c], g.cell_volumes[c],
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
        if bc is not None:
            # remap the dictionary such that the key is lowercase
            keys = [k for k in bc_val.keys()]
            bc_val = {k.lower(): bc_val[k] for k in keys}
            keys = [k.lower() for k in keys]

            if np.any(bc.is_dir):
                norm = sps.linalg.norm(H, np.inf)
                H[bc.is_dir, :] *= 0
                H[bc.is_dir, bc.is_dir] = norm
                rhs[bc.is_dir] = norm*bc_val['dir']

            if np.any(bc.is_neu):
                faces, _, sgn = sps.find(g.cell_faces)
                sgn = sgn[np.unique(faces, return_index=True)[1]]
                rhs[bc.is_neu] += sgn[bc.is_neu]*bc_val['neu']*\
                                  g.face_areas[bc.is_neu]

        return H, rhs

#------------------------------------------------------------------------------#

    def computeUP(self, g, l, data):
        """
        Return the velocity and pressure computed from the hybrid variables.

        Parameters
        ----------
        g : grid
            Grid, or a subclass, with geometry fields computed.
        l : array (g.num_faces)
            Hybrid solution of the system.
        k : second_order_tensor)
            Permeability. Cell-wise.
        f : array (g.num_cells)
            Scalar source term.

        Return
        ------
        u : array (g.num_faces)
            Velocity at each face.
        p : array (g.num_cells)
            Pressure at each cell.

        matrix: sparse csr (g.num_faces, g.num_faces)
            Spd matrix obtained from the discretization.
        rhs: array (g.num_faces)
            Right-hand side which contains the boundary conditions and the scalar
            source term.

        Examples
        --------
        H, rhs = hybrid.matrix_rhs(g, perm, f, bc)
        l = sps.linalg.spsolve(H, rhs)
        u, p = hybrid.computeUP(g, l, perm, f)
        P0u = dual.projectU(g, perm, u)

        """
        k, f = data['k'], data['f']

        faces, cells, sgn = sps.find(g.cell_faces)
        c_centers, f_normals, f_centers, _, _ = cg.map_grid(g)

        weight = np.ones(g.num_cells) if g.dim != 1 else g.cell_volumes
        diams = g.cell_diameters()

        p = np.zeros(g.num_cells)
        u = np.zeros(g.num_faces)
        massHdiv = dual.DualVEM().massHdiv

        for c in np.arange(g.num_cells):
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
            faces_loc = faces[loc]
            sgn_loc = sgn[loc].reshape((-1,1))
            ndof = faces_loc.size

            K = k.perm[0:g.dim, 0:g.dim, c]
            normals = np.multiply(np.tile(sgn_loc.T, (g.dim,1)),
                                  f_normals[:, faces_loc])

            A, _ = massHdiv(K, c_centers[:, c], g.cell_volumes[c],
                            f_centers[:, faces_loc], normals, np.ones(ndof),
                            diams[c], weight[c])
            B = -np.ones((ndof,1))
            C = np.eye(ndof,ndof)

            S = 1/np.dot(B.T, solve(A, B))
            f_loc = f[c]*g.cell_volumes[c]
            l_loc = l[faces_loc].reshape((-1,1))

            p[c] = np.dot(S, f_loc - np.dot(B.T, solve(A, np.dot(C, l_loc))))
            u[faces_loc] = -np.multiply(sgn_loc, solve(A, np.dot(B, p[c]) + \
                                                            np.dot(C, l_loc)))

        return u, p

#------------------------------------------------------------------------------#
