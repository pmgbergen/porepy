# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""

import numpy as np
import scipy.sparse as sps
from compgeom import basics as cg

from core.solver.solver import *

class DualVEM(Solver):

#------------------------------------------------------------------------------#

    def ndof(self, g): return g.num_cells + g.num_faces

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        """
        Return the matrix and righ-hand side for a discretization of a second
        order elliptic equation using dual virtual element method.

        Parameters
        ----------
        g : grid
            Grid, or a subclass, with geometry fields computed.
        k : second_order_tensor)
            Permeability. Cell-wise.
        f : array (self.g.num_cells)
            Scalar source term.
        bc :
            Boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.

        Return
        ------
        matrix: sparse csr (g.num_faces+g_num_cells, g.num_faces+g_num_cells)
            Saddle point matrix obtained from the discretization.
        rhs: array (g.num_faces+g_num_cells)
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

        D, rhs = dual.matrix_rhs(g, perm, f, bnd, bnd_val)
        up = sps.linalg.spsolve(D, rhs)
        u, p = dual.extract(g, up)
        P0u = dual.projectU(g, perm, u)

        """
        M, bc_weight = self.matrix(g, data, bc_weight=True)
        return M, self.rhs(g, data, bc_weight)

#------------------------------------------------------------------------------#

    def matrix(self, g, data, bc_weight=False):
        """
        Return the matrix for a discretization of a second order elliptic equation
        using dual virtual element method.

        Parameters
        ----------
        g : grid
            Grid, or a subclass, with geometry fields computed.
        k : second_order_tensor)
            Permeability. Cell-wise.
        bc :
            Boundary conditions (optional)
        bc_weight: bool
            Decide if the diagonal entries associated to Neumann boundary conditions
            should be weighted by the infinity norm of the mass-Hdiv or not.
            If bc_weight is True than the weight is returned to be used in the
            construction of the right-hand side.

        Return
        ------
        matrix: sparse csr (g.num_faces+g_num_cells, g.num_faces+g_num_cells)
            Saddle point matrix obtained from the discretization.
        weight: scalar (optional)
            Returned only if bc_weight is True. It represents the infinity norm
            of the mass-Hdiv block of the globla matrix.

        Examples
        --------
        b_faces_neu = ... # id of the Neumann faces
        b_faces_dir = ... # id of the Dirichlet faces
        bnd = bc.BoundaryCondition(g, np.hstack((b_faces_dir, b_faces_neu)),
                                ['dir']*b_faces_dir.size + ['neu']*b_faces_neu.size)
        bnd_val = {'dir': fun_dir(g.face_centers[:, b_faces_dir]),
                   'neu': fun_neu(f.face_centers[:, b_faces_neu])}

        D, weight = dual.matrix(g, perm, bnd, bc_weight=True)
        rhs = dual.rhs(g, f, bnd, bnd_val, weight)
        up = sps.linalg.spsolve(D, rhs)
        u, p = dual.extract(g, up)
        P0u = dual.projectU(g, perm, u)

        """
        k, bc = data['k'], data.get('bc')
        faces, cells, sgn = sps.find(g.cell_faces)

        c_centers, f_normals, f_centers, _, _ = cg.map_grid(g)

        weight = np.ones(g.num_cells) if g.dim != 1 else g.cell_volumes
        diams = g.cell_diameters()

        size = np.sum(np.square(g.cell_faces.indptr[1:]-\
                                g.cell_faces.indptr[:-1]))
        I = np.empty(size,dtype=np.int)
        J = np.empty(size,dtype=np.int)
        dataIJ = np.empty(size)
        idx = 0

        for c in np.arange(g.num_cells):
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
            faces_loc = faces[loc]

            K = k.perm[0:g.dim, 0:g.dim, c]
            sgn_loc = sgn[loc]
            normals = f_normals[:, faces_loc]

            A, _ = self.massHdiv(K, c_centers[:,c], g.cell_volumes[c],
                                 f_centers[:,faces_loc], normals, sgn_loc,
                                 diams[c], weight[c])

            # save values for Hdiv-Stiffness matrix
            cols = np.tile(faces_loc, (faces_loc.size,1))
            loc_idx = slice(idx, idx+cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # construct the global matrices
        mass = sps.coo_matrix((dataIJ, (I, J)))
        div = -g.cell_faces.T
        M = sps.bmat([[mass, div.T], [div,None]], format='csr')

        norm = sps.linalg.norm(mass, np.inf) if bc_weight else 1

        # assign the Neumann boundary conditions
        if bc and np.any(bc.is_neu):
            is_neu = np.hstack((bc.is_neu,
                                np.zeros(g.num_cells,dtype=np.bool)))
            M[is_neu, :] *= 0
            M[is_neu, is_neu] = norm

        if bc_weight: return M, norm
        else:         return M

#------------------------------------------------------------------------------#

    def rhs(self, g, data, bc_weight=1):
        """
        Return the righ-hand side for a discretization of a second order elliptic
        equation using dual virtual element method.

        Parameters
        ----------
        g : grid
            Grid, or a subclass, with geometry fields computed.
        f : array (g.num_cells)
            Scalar source term.
        bc :
            Boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.
        bc_weight: scalar (optional)
            Weight for the entries associated to Neumann boundary conditions.

        Return
        ------
        rhs: array (g.num_faces+g_num_cells)
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

        D, weight = dual.matrix(g, perm, bnd, bc_weight=True)
        rhs = dual.rhs(g, f, bnd, bnd_val, weight)
        up = sps.linalg.spsolve(D, rhs)
        u, p = dual.extract(g, up)
        P0u = dual.projectU(g, perm, u)

        """
        f, bc, bc_val = data['f'], data.get('bc'), data.get('bc_val')
        assert not( bool(bc is None) != bool(bc_val is None) )

        rhs = np.zeros(self.ndof(g))
        is_p = np.hstack((np.zeros(g.num_faces,dtype=np.bool),
                          np.ones(g.num_cells,dtype=np.bool)))

        rhs[is_p] = -f*g.cell_volumes
        if bc is None: return rhs

        # remap the dictionary such that the key is lowercase
        keys = [k for k in bc_val.keys()]
        bc_val = {k.lower(): bc_val[k] for k in keys}
        keys = [k.lower() for k in keys]

        if 'dir' in keys:
            is_dir = np.hstack((bc.is_dir,
                                np.zeros(g.num_cells,dtype=np.bool)))
            faces, _, sgn = sps.find(g.cell_faces)
            sgn = sgn[np.unique(faces, return_index=True)[1]]
            rhs[is_dir] = -sgn[bc.is_dir]*bc_val['dir']

        if 'neu' in keys:
            is_neu = np.hstack((bc.is_neu,
                                np.zeros(g.num_cells,dtype=np.bool)))
            rhs[is_neu] = bc_weight*bc_val['neu']*g.face_areas[bc.is_neu]

        return rhs

#------------------------------------------------------------------------------#

    def extractU(self, g, up):
        """  Extract the velocity and the pressure from a dual virtual element
        solution.

        Parameters
        ----------
        g : grid
            Grid, or a subclass, with geometry fields computed.
        up : array (g.num_faces+g.num_cells)
            Solution, stored as [velocity,pressure]

        Return
        ------
        u : array (g.num_faces)
            Velocity at each face.
        p : array (g.num_cells)
            Pressure at each cell.

        Examples
        --------
        D, rhs = dual.matrix_rhs(g, perm, f, bc)
        up = sps.linalg.spsolve(D, rhs)
        u, p = dual.extract(g, up)
        P0u = dual.projectU(g, perm, u)

        """
        return up[:g.num_faces]

#------------------------------------------------------------------------------#

    def extractP(self, g, up):
        """  Extract the velocity and the pressure from a dual virtual element
        solution.

        Parameters
        ----------
        g : grid
            Grid, or a subclass, with geometry fields computed.
        up : array (g.num_faces+g.num_cells)
            Solution, stored as [velocity,pressure]

        Return
        ------
        u : array (g.num_faces)
            Velocity at each face.
        p : array (g.num_cells)
            Pressure at each cell.

        Examples
        --------
        D, rhs = dual.matrix_rhs(g, perm, f, bc)
        up = sps.linalg.spsolve(D, rhs)
        u, p = dual.extract(g, up)
        P0u = dual.projectU(g, perm, u)

        """
        return up[g.num_faces:]

#------------------------------------------------------------------------------#

    def projectU(self, g, u, data):
        """  Project the velocity computed with a dual vem solver to obtain a
        piecewise constant vector field, one triplet for each cell.

        Parameters
        ----------
        g : grid
            Grid, or a subclass, with geometry fields computed.
        k : second_order_tensor)
            Permeability. Cell-wise.
        u : array (g.num_faces)
            Velocity at each face.

        Return
        ------
        P0u : ndarray (3, g.num_faces)
            Velocity at each cell.

        Examples
        --------
        D, rhs = dual.matrix_rhs(g, perm, f, bc)
        up = sps.linalg.spsolve(D, rhs)
        u, p = dual.extract(g, up)
        P0u = dual.projectU(g, perm, u)

        """
        k = data['k']
        faces, cells, sgn = sps.find(g.cell_faces)
        c_centers, f_normals, f_centers, R, dim = cg.map_grid(g)

        diams = g.cell_diameters()

        P0u = np.zeros((3, g.num_cells))
        idx = 0

        for c in np.arange(g.num_cells):
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
            faces_loc = faces[loc]

            K = k.perm[0:g.dim, 0:g.dim, c]
            sgn_loc = sgn[loc]
            normals = f_normals[:, faces_loc]

            _, Pi_s = self.massHdiv(K, c_centers[:,c], g.cell_volumes[c],
                                    f_centers[:,faces_loc], normals, sgn_loc,
                                    diams[c])

            # extract the velocity for the current cell
            P0u[dim, c] = np.dot(Pi_s, u[faces_loc]) / diams[c]
            P0u[:, c] = np.dot(R.T, P0u[:, c])

        return P0u

#------------------------------------------------------------------------------#

    def massHdiv(self, K, c_center, c_volume, f_centers, normals, sgn, diam,
                 weight=0):
        """ Compute the local mass Hdiv matrix using the mixed vem approach.

        Parameters
        ----------
        K : ndarray (g.dim, g.dim)
            Permeability of the cell.
        c_center : array (g.dim)
            Cell center.
        c_volume : scalar
            Cell volume.
        f_centers : ndarray (g.dim, num_faces_of_cell)
            Center of the cell faces.
        normals : ndarray (g.dim, num_faces_of_cell)
            Normal of the cell faces weighted by the face areas.
        sgn : array (num_faces_of_cell)
            +1 or -1 if the normal is inward or outward to the cell.
        diam : scalar
            Diameter of the cell.
        weight : scalar
            weight for the stabilization term. Optional, default = 0.

        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass Hdiv matrix.
        """

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
                            for s,f in zip(sgn,f_centers.T)] ).reshape((dim,-1))

        assert np.allclose(G, np.dot(F,D))

        # local matrix Pi
        Pi_s = np.linalg.solve(G, F)
        I_Pi = np.eye(f_centers.shape[1]) - np.dot(D, Pi_s)

        # local Hdiv-Mass matrix
        w = weight * np.linalg.norm(np.linalg.inv(K),np.inf)
        A = np.dot(Pi_s.T, np.dot(G, Pi_s)) + w * np.dot(I_Pi.T, I_Pi)

        return A, Pi_s

#------------------------------------------------------------------------------#
