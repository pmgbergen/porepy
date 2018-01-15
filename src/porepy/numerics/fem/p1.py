import warnings
import numpy as np
import scipy.sparse as sps
import scipy.linalg as linalg

from porepy.params import tensor

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.numerics.mixed_dim.abstract_coupling import AbstractCoupling

from porepy.utils import comp_geom as cg

#------------------------------------------------------------------------------#

class P1(Solver):

#------------------------------------------------------------------------------#

    def __init__(self, physics='flow'):
        self.physics = physics

#------------------------------------------------------------------------------#

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of nodes.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_nodes

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        """
        Return the matrix and righ-hand side for a discretization of a second
        order elliptic equation using P1 method on simplices.
        The name of data in the input dictionary (data) are:
        perm : second_order_tensor
            Permeability defined cell-wise.
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
        matrix: sparse csr (g.num_nodes, g.num_nodes)
            Matrix obtained from the discretization.
        rhs: array (g.num_nodes)
            Right-hand side which contains the boundary conditions and the scalar
            source term.

        """
        M, bc_weight = self.matrix(g, data, bc_weight=True)
        return M, self.rhs(g, data, bc_weight)

#------------------------------------------------------------------------------#

    def matrix(self, g, data, bc_weight=False):
        """
        Return the matrix for a discretization of a second order elliptic equation
        using P1 method. See self.matrix_rhs for a detaild description.

        Additional parameter:
        --------------------
        bc_weight: to compute the infinity norm of the matrix and use it as a
            weight to impose the boundary conditions. Default True.

        Additional return:
        weight: if bc_weight is True return the weight computed.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            M = sps.identity(self.ndof(g))
            if bc_weight:
                return M, 1
            return M

        # Retrieve the permeability, boundary conditions, and aperture
        # The aperture is needed in the hybrid-dimensional case, otherwise is
        # assumed unitary
        param = data['param']
        k = param.get_tensor(self)
        bc = param.get_bc(self)
        a = param.get_aperture()

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        c_centers, f_normals, f_centers, R, dim, node_coords = cg.map_grid(g)

        if not data.get('is_tangential', False):
                # Rotate the permeability tensor and delete last dimension
                if g.dim < 3:
                    k = k.copy()
                    k.rotate(R)
                    remove_dim = np.where(np.logical_not(dim))[0]
                    k.perm = np.delete(k.perm, (remove_dim), axis=0)
                    k.perm = np.delete(k.perm, (remove_dim), axis=1)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(g.dim+1, 2)*g.num_cells
        I = np.empty(size, dtype=np.int)
        J = np.empty(size, dtype=np.int)
        dataIJ = np.empty(size)
        idx = 0

        cell_nodes = g.cell_nodes()
        nodes, cells, _ = sps.find(cell_nodes)

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c+1])

            nodes_loc = nodes[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the stiff-H1 local matrix
            A = self.stiffH1(a[c]*k.perm[0:g.dim, 0:g.dim, c],
                             g.cell_volumes[c], coord_loc, g.dim)

            # Save values for Hdiv-mass local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx+cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        M = sps.csr_matrix((dataIJ, (I, J)))

        norm = sps.linalg.norm(M, np.inf) if bc_weight else 1

        # assign the Dirichlet boundary conditions
        if bc and np.any(bc.is_dir):
            nodes, _, _, = sps.find(g.face_nodes)

            dir_nodes = np.array([\
                         nodes[g.face_nodes.indptr[f]:g.face_nodes.indptr[f+1]]\
                        for f in np.where(bc.is_dir)[0]]).ravel()

            M[dir_nodes, :] *= 0
            M[dir_nodes, dir_nodes] = norm

        if bc_weight:
            return M, norm
        return M

#------------------------------------------------------------------------------#

    def rhs(self, g, data, bc_weight=1):
        """
        Return the righ-hand side for a discretization of a second order elliptic
        equation using RT0-P0 method. See self.matrix_rhs for a detaild
        description.

        Additional parameter:
        --------------------
        bc_weight: to use the infinity norm of the matrix to impose the
            boundary conditions. Default 1.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        param = data['param']
        f = param.get_source(self)

        if g.dim == 0:
            return np.hstack(([0], f))

        bc = param.get_bc(self)
        bc_val = param.get_bc_val(self)

        assert not bool(bc is None) != bool(bc_val is None)

        rhs = np.zeros(self.ndof(g))
        if bc is None:
            return rhs

        is_p = np.hstack((np.zeros(g.num_faces, dtype=np.bool),
                          np.ones(g.num_cells, dtype=np.bool)))

        if np.any(bc.is_dir):
            is_dir = np.where(bc.is_dir)[0]
            faces, _, sign = sps.find(g.cell_faces)
            sign = sign[np.unique(faces, return_index=True)[1]]
            rhs[is_dir] += -sign[is_dir] * bc_val[is_dir]

        if np.any(bc.is_neu):
            is_neu = np.where(bc.is_neu)[0]
            rhs[is_neu] = bc_weight * bc_val[is_neu]

        return rhs

#------------------------------------------------------------------------------#

    def stiffH1(self, K, c_volume, coord, dim):
        """ Compute the local stiffness H1 matrix using the P1 Lagrangean approach.

        Parameters
        ----------
        K : ndarray (g.dim, g.dim)
            Permeability of the cell.
        c_volume : scalar
            Cell volume.

        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass Hdiv matrix.
        """
        # Allow short variable names in this function
        # pylint: disable=invalid-name

        Q = np.hstack((np.ones((dim+1, 1)), coord.T))
        dphi = np.linalg.inv(Q)[1:, :]

        return c_volume*np.dot(dphi.T, np.dot(K, dphi))

#------------------------------------------------------------------------------#
