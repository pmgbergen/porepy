# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""
import warnings
import numpy as np
from numpy.linalg import solve
import scipy.sparse as sps

from porepy.numerics.mixed_dim.solver import Solver
from porepy.utils import comp_geom as cg
from porepy.numerics.vem import vem_dual as dual
from porepy.params import tensor


class HybridDualVEM(Solver):

    # ------------------------------------------------------------------------------#

    def __init__(self, physics="flow"):
        self.physics = physics

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of faces (hybrid dofs).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_faces

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        """
        Return the matrix and righ-hand side for a discretization of a second
        order elliptic equation using hybrid dual virtual element method.
        The name of data in the input dictionary (data) are:
        perm : tensor.SecondOrderTensor
            Permeability defined cell-wise. If not given a identity permeability
            is assumed and a warning arised.
        source : array (self.g.num_cells)
            Scalar source term defined cell-wise. If not given a zero source
            term is assumed and a warning arised.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

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

        data = {'perm': perm, 'source': f, 'bc': bnd, 'bc_val': bnd_val}

        H, rhs = hybrid.matrix_rhs(g, data)
        l = sps.linalg.spsolve(H, rhs)
        u, p = hybrid.compute_up(g, l, data)
        P0u = dual.project_u(g, perm, u)

        """
        # pylint: disable=invalid-name

        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            return sps.identity(self.ndof(g), format="csr"), np.zeros(1)

        param = data["param"]
        k = param.get_tensor(self)
        f = param.get_source(self)
        bc = param.get_bc(self)
        bc_val = param.get_bc_val(self)
        a = param.aperture

        faces, _, sgn = sps.find(g.cell_faces)

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        c_centers, f_normals, f_centers, _, _, _ = cg.map_grid(g)

        # Weight for the stabilization term
        diams = g.cell_diameters()
        weight = np.power(diams, 2 - g.dim)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.sum(np.square(g.cell_faces.indptr[1:] - g.cell_faces.indptr[:-1]))
        I = np.empty(size, dtype=np.int)
        J = np.empty(size, dtype=np.int)
        data = np.empty(size)
        rhs = np.zeros(g.num_faces)

        idx = 0
        massHdiv = dual.DualVEM().massHdiv

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its faces
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            faces_loc = faces[loc]
            ndof = faces_loc.size

            # Retrieve permeability and normals assumed outward to the cell.
            sgn_loc = sgn[loc].reshape((-1, 1))
            normals = np.multiply(
                np.tile(sgn_loc.T, (g.dim, 1)), f_normals[:, faces_loc]
            )

            # Compute the H_div-mass local matrix
            A = massHdiv(
                k.perm[0 : g.dim, 0 : g.dim, c],
                c_centers[:, c],
                a[c] * g.cell_volumes[c],
                f_centers[:, faces_loc],
                a[c] * normals,
                np.ones(ndof),
                diams[c],
                weight[c],
            )[0]
            # Compute the Div local matrix
            B = -np.ones((ndof, 1))
            # Compute the hybrid local matrix
            C = np.eye(ndof, ndof)

            # Perform the static condensation to compute the hybrid local matrix
            invA = np.linalg.inv(A)
            S = 1 / np.dot(B.T, np.dot(invA, B))
            L = np.dot(np.dot(invA, np.dot(B, np.dot(S, B.T))), invA)
            L = np.dot(np.dot(C.T, L - invA), C)

            # Compute the local hybrid right using the static condensation
            rhs[faces_loc] += np.dot(C.T, np.dot(invA, np.dot(B, np.dot(S, f[c]))))[
                :, 0
            ]

            # Save values for hybrid matrix
            cols = np.tile(faces_loc, (faces_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            data[loc_idx] = L.ravel()
            idx += cols.size

        # construct the global matrices
        H = sps.coo_matrix((data, (I, J))).tocsr()

        # Apply the boundary conditions
        if bc is not None:

            if np.any(bc.is_dir):
                norm = sps.linalg.norm(H, np.inf)
                is_dir = np.where(bc.is_dir)[0]

                H[is_dir, :] *= 0
                H[is_dir, is_dir] = norm
                rhs[is_dir] = norm * bc_val[is_dir]

            if np.any(bc.is_neu):
                faces, _, sgn = sps.find(g.cell_faces)
                sgn = sgn[np.unique(faces, return_index=True)[1]]

                is_neu = np.where(bc.is_neu)[0]
                rhs[is_neu] += sgn[is_neu] * bc_val[is_neu] * g.face_areas[is_neu]

        return H, rhs

    # ------------------------------------------------------------------------------#

    def compute_up(self, g, l, data):
        """
        Return the velocity and pressure computed from the hybrid variables.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        l : array (g.num_faces) Hybrid solution of the system.
        data: dictionary to store the data. See self.matrix_rhs for a detaild
            description.

        Return
        ------
        u : array (g.num_faces) Velocity at each face.
        p : array (g.num_cells) Pressure at each cell.

        """
        # pylint: disable=invalid-name

        if g.dim == 0:
            return 0, l[0]

        param = data["param"]
        k = param.get_tensor(self)
        f = param.get_source(self)
        a = param.aperture

        faces, _, sgn = sps.find(g.cell_faces)

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        c_centers, f_normals, f_centers, _, _, _ = cg.map_grid(g)

        # Weight for the stabilization term
        diams = g.cell_diameters()
        weight = np.power(diams, 2 - g.dim)

        # Allocation of the pressure and velocity vectors
        p = np.zeros(g.num_cells)
        u = np.zeros(g.num_faces)
        massHdiv = dual.DualVEM().massHdiv

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its faces
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            faces_loc = faces[loc]
            ndof = faces_loc.size

            # Retrieve permeability and normals assumed outward to the cell.
            sgn_loc = sgn[loc].reshape((-1, 1))
            normals = np.multiply(
                np.tile(sgn_loc.T, (g.dim, 1)), f_normals[:, faces_loc]
            )

            # Compute the H_div-mass local matrix
            A = massHdiv(
                k.perm[0 : g.dim, 0 : g.dim, c],
                c_centers[:, c],
                a[c] * g.cell_volumes[c],
                f_centers[:, faces_loc],
                a[c] * normals,
                np.ones(ndof),
                diams[c],
                weight[c],
            )[0]
            # Compute the Div local matrix
            B = -np.ones((ndof, 1))
            # Compute the hybrid local matrix
            C = np.eye(ndof, ndof)

            # Perform the static condensation to compute the pressure and velocity
            S = 1 / np.dot(B.T, solve(A, B))
            l_loc = l[faces_loc].reshape((-1, 1))

            p[c] = np.dot(S, f[c] - np.dot(B.T, solve(A, np.dot(C, l_loc))))
            u[faces_loc] = -np.multiply(
                sgn_loc, solve(A, np.dot(B, p[c]) + np.dot(C, l_loc))
            )

        return u, p


# ------------------------------------------------------------------------------#
