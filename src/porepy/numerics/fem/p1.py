import numpy as np
import scipy.sparse as sps
import logging

import porepy as pp

# Module-wide logger
logger = logging.getLogger(__name__)


class P1:

    # ------------------------------------------------------------------------------#

    def __init__(self, keyword="flow"):
        self.keyword = keyword

    # ------------------------------------------------------------------------------#

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
        if isinstance(g, pp.Grid):
            return g.num_nodes
        elif isinstance(g, pp.MortarGrid):
            return g.num_cells
        else:
            raise ValueError

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        """ Return the matrix and righ-hand side for a discretization of a second
        order elliptic equation using P1 method on simplices.

        We assume the following sub-dictionary to be present in the data dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].

        parameter_dictionary contains the entries:
            second_order_tensor: (pp.SecondOrderTensor) Permeability defined cell-wise.
            bc: (pp.BoundaryConditionNode) node-wise boundary conditions.
            bc_values: array (self.ndof) boundary condition values.

        Parameters
        ----------
        g: grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Returns
        ------
        matrix: sparse csr (g.num_nodes, g.num_nodes) Matrix obtained from the
            discretization.
        rhs: array (g.num_nodes) Right-hand side which contains the boundary conditions.
        """
        M, bc_weight = self.matrix(g, data, bc_weight=True)
        return M, self.rhs(g, data, bc_weight)

    # ------------------------------------------------------------------------------#

    def matrix(self, g, data, bc_weight=False):
        """ Return the matrix for a discretization of a second order elliptic equation
        using P1 method.

        Parameters
        ----------
        g: grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data. See self.matrix_rhs for required contents.

        Returns
        ------
        matrix: sparse csr (g.num_nodes, g.num_nodes)
            Matrix obtained from the discretization.

        Optional parameters:
        --------------------
        is_tangential: bool, whether the lower-dimensional permeability tensor has been
            rotated to the fracture plane. Defaults to False. Stored in data dictionary.
        bc_weight: bool, whether to compute the infinity norm of the matrix and use it
            as a weight to impose the boundary conditions. Default False.

        Additional return:
        weight: if bc_weight is True return the weight computed.

        Raises:
        ------
        ValueError if the boundary condition is not defined node-wise.
        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            M = sps.csr_matrix((self.ndof(g), self.ndof(g)))
            if bc_weight:
                return M, 1
            return M

        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the permeability, boundary conditions
        k = parameter_dictionary["second_order_tensor"]
        bc = parameter_dictionary["bc"]
        if not isinstance(bc, pp.BoundaryConditionNode):
            raise ValueError("Consider pp.BoundaryConditionNode to assign bc")

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        c_centers, f_normals, f_centers, R, dim, node_coords = pp.map_geometry.map_grid(
            g
        )

        if not data.get("is_tangential", False):
            # Rotate the permeability tensor and delete last dimension
            if g.dim < 3:
                k = k.copy()
                k.rotate(R)
                remove_dim = np.where(np.logical_not(dim))[0]
                k.values = np.delete(k.values, (remove_dim), axis=0)
                k.values = np.delete(k.values, (remove_dim), axis=1)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(g.dim + 1, 2) * g.num_cells
        I = np.empty(size, dtype=np.int)
        J = np.empty(size, dtype=np.int)
        dataIJ = np.empty(size)
        idx = 0

        cell_nodes = g.cell_nodes()
        nodes, cells, _ = sps.find(cell_nodes)

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = nodes[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the stiff-H1 local matrix
            A = self.stiffH1(
                k.values[0 : g.dim, 0 : g.dim, c], g.cell_volumes[c], coord_loc, g.dim
            )

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        M = sps.csr_matrix((dataIJ, (I, J)))

        norm = sps.linalg.norm(M, np.inf) if bc_weight else 1

        # assign the Dirichlet boundary conditions
        if bc and np.any(bc.is_dir):
            dir_nodes = np.where(bc.is_dir)[0]
            # set in an efficient way the essential boundary conditions, by
            # clear the rows and put norm in the diagonal
            for row in dir_nodes:
                M.data[M.indptr[row] : M.indptr[row + 1]] = 0.0

            d = M.diagonal()
            d[dir_nodes] = norm
            M.setdiag(d)

        if bc_weight:
            return M, norm
        return M

    # ------------------------------------------------------------------------------#

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

        if g.dim == 0:
            return np.zeros(1)

        # Retrieve the boundary conditions and values
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        bc_val = parameter_dictionary["bc_values"]
        bc = parameter_dictionary["bc"]

        rhs = np.zeros(self.ndof(g))
        if bc is None:
            return rhs

        #        if np.any(bc.is_neu):
        #            is_dir = np.where(bc.is_dir)[0]
        #            faces, _, sign = sps.find(g.cell_faces)
        #            sign = sign[np.unique(faces, return_index=True)[1]]
        #            rhs[is_dir] += -sign[is_dir] * bc_val[is_dir]

        if np.any(bc.is_dir):
            is_dir = np.where(bc.is_dir)[0]
            rhs[is_dir] = bc_weight * bc_val[is_dir]

        return rhs

    # ------------------------------------------------------------------------------#

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

        Q = np.hstack((np.ones((dim + 1, 1)), coord.T))
        dphi = np.linalg.inv(Q)[1:, :]

        return c_volume * np.dot(dphi.T, np.dot(K, dphi))

    # ------------------------------------------------------------------------------#


# class P1Coupling(AbstractCoupling):
#
#    # ------------------------------------------------------------------------------#
#
#    def __init__(self, discr):
#        self.discr_ndof = discr.ndof
#
#    # ------------------------------------------------------------------------------#
#
#    def matrix_rhs(self, matrix, g_h, g_l, data_h, data_l, data_edge):
#        """
#        Construct the matrix (and right-hand side) for the coupling conditions.
#        Note: the right-hand side is not implemented now.
#
#        Parameters:
#            g_h: grid of higher dimension
#            g_l: grid of lower dimension
#            data_h: dictionary which stores the data for the higher dimensional
#                grid
#            data_l: dictionary which stores the data for the lower dimensional
#                grid
#            data: dictionary which stores the data for the edges of the grid
#                bucket
#
#        Returns:
#            cc: block matrix which store the contribution of the coupling
#                condition. See the abstract coupling class for a more detailed
#                description.
#        """
#        # pylint: disable=invalid-name
#
#        # Retrieve the number of degrees of both grids
#        # Create the block matrix for the contributions
#        mg = data_edge["mortar_grid"]
#        dof, cc = self.create_block_matrix([g_h, g_l, mg])
#
#        # Recover the information for the grid-grid mapping
#        faces_h, cells_h, _ = sps.find(g_h.cell_faces)
#        ind_faces_h = np.unique(faces_h, return_index=True)[1]
#        cells_h = cells_h[ind_faces_h]
#
#        # Mortar mass matrix
#        inv_M = sps.diags(1.0 / mg.cell_volumes)
#
#        # Projection matrix from hight/lower grid to mortar
#        hat_P = mg.high_to_mortar_avg()
#        check_P = mg.low_to_mortar_avg()
#
#        hat_P0 = g_h.face_nodes.T.astype(np.float) / g_h.dim
#        check_P0 = g_l.cell_nodes().T.astype(np.float) / (g_l.dim + 1)
#
#        # Normal permeability and aperture of the intersection
#        kn = data_h[pp.keywords.PARAMETERS][self.keyword]["normal_diffusivity"]
#        inv_k = 1.0 / (2.0 * kn)
#        aperture_h = data_h[pp.keywords.PARAMETERS][self.keyword]["aperture"]
#
#        # Inverse of the normal permability matrix
#        Eta = sps.diags(np.divide(inv_k, hat_P * aperture_h[cells_h]))
#
#        # Compute the mortar variables rows
#        cc[2, 0] = -hat_P * hat_P0
#        cc[2, 1] = check_P * check_P0
#        cc[2, 2] = Eta * inv_M
#
#        # Compute the high dimensional grid coupled to mortar grid term
#        cc[0, 2] = -cc[2, 0].T
#        cc[1, 2] = -cc[2, 1].T
#
#        return matrix + cc
#
#
## ------------------------------------------------------------------------------#
