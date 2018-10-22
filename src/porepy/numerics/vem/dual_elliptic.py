#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module contains common functionalities for discretization based on the mixed
variational formulation.
"""

import numpy as np
import scipy.sparse as sps

import porepy as pp


class DualElliptic(pp.numerics.mixed_dim.elliptic_discretization.EllipticDiscretization):
    """ Parent class for methods based on the mixed variational form of the
    elliptic equation. The class should not be used by itself, but provides a
    sheared implementation of central methods.

    Known subclasses that can be used for actual discretization are MVEM and RT0.

    """

    def __init__(self, keyword):
        self.keyword = keyword

        # @ALL: We kee the physics keyword for now, or else we completely
        # break the parameter assignment workflow. The physics keyword will go
        # to be replaced by a more generalized approach, but one step at a time
        self.physics = keyword


    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of faces (velocity dofs) plus the number of cells
        (pressure dof). If a mortar grid is given the number of dof are equal to
        the number of cells, we are considering an inter-dimensional interface
        with flux variable as mortars.

        Parameter
        ---------
        g: grid.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        if isinstance(g, pp.Grid):
            return g.num_cells + g.num_faces
        else:
            raise ValueError


    def assemble_rhs(self, g, data, bc_weight=1):
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

        param = data["param"]
        f = param.get_source(self)

        if g.dim == 0:
            return np.hstack(([0], f))

        bc = param.get_bc(self)
        bc_val = param.get_bc_val(self)

        assert not bool(bc is None) != bool(bc_val is None)

        rhs = np.zeros(self.ndof(g))
        if bc is None:
            return rhs

        # For dual discretizations, internal boundaries
        # are handled by assigning Dirichlet conditions. Thus, we remove them
        # from the is_neu (where they belong by default). As the dirichlet
        # values are simply added to the rhs, and the internal Dirichlet
        # conditions on the fractures SHOULD be homogeneous, we exclude them
        # from the dirichlet condition as well.
        is_neu = np.logical_and(bc.is_neu, np.logical_not(bc.is_internal))
        is_dir = np.logical_and(bc.is_dir, np.logical_not(bc.is_internal))

        faces, _, sign = sps.find(g.cell_faces)
        sign = sign[np.unique(faces, return_index=True)[1]]

        if np.any(is_dir):
            is_dir = np.where(is_dir)[0]
            rhs[is_dir] += -sign[is_dir] * bc_val[is_dir]

        if np.any(is_neu):
            is_neu = np.where(is_neu)[0]
            rhs[is_neu] = sign[is_neu] * bc_weight * bc_val[is_neu]

        return rhs

    def _assemble_neumann_common(self, g, data, M, mass, bc_weight=None):
        """ Impose Neumann boundary discretization on an already assembled
        system matrix.

        Common implementation for VEM and RT0. The parameter mass should be
        adapted to the discretization method in question

        """

        norm = sps.linalg.norm(mass, np.inf) if bc_weight else 1

        param = data["param"]
        bc = param.get_bc(self)

        # assign the Neumann boundary conditions
        # For dual discretizations, internal boundaries
        # are handled by assigning Dirichlet conditions. THus, we remove them
        # from the is_neu (where they belong by default) and add them in
        # is_dir.
        is_neu = np.logical_and(bc.is_neu, np.logical_not(bc.is_internal))
        if bc and np.any(is_neu):
            is_neu = np.hstack((is_neu, np.zeros(g.num_cells, dtype=np.bool)))
            is_neu = np.where(is_neu)[0]

            # set in an efficient way the essential boundary conditions, by
            # clear the rows and put norm in the diagonal
            for row in is_neu:
                M.data[M.indptr[row] : M.indptr[row + 1]] = 0.

            d = M.diagonal()
            d[is_neu] = norm
            M.setdiag(d)

        if bc_weight:
            return M, norm
        return M


    def _velocity_dof(self, g, g_m):
        # Recover the information for the grid-grid mapping
        faces_h, cells_h, sign_h = sps.find(g.cell_faces)
        ind_faces_h = np.unique(faces_h, return_index=True)[1]
        cells_h = cells_h[ind_faces_h]
        sign_h = sign_h[ind_faces_h]

        # Velocity degree of freedom matrix
        U = sps.diags(sign_h)

        shape = (g.num_cells, g_m.num_cells)
        hat_E_int = g_m.mortar_to_master_int()
        hat_E_int = sps.bmat([[U * hat_E_int], [sps.csr_matrix(shape)]])
        return hat_E_int


    def assemble_int_bound_pressure_trace(self, g_master, data_master, data_edge, grid_swap, cc, matrix, self_ind=0):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a higher-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
        """
        mg = data_edge['mortar_grid']
        hat_E_int = self._velocity_dof(g_master, mg)

        cc[2, self_ind] -= hat_E_int.T * matrix[0, 0]
        cc[2, 2] -= hat_E_int.T * matrix[0, 0] * hat_E_int


    def assemble_int_bound_flux(self, g_master, data_master, data_edge, grid_swap, cc, matrix, self_ind=0):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a flux boundary condition.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a higher-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.

        """

        # The matrix must be the VEM discretization matrix.
        mg = data_edge['mortar_grid']
        hat_E_int = self._velocity_dof(g_master, mg)

        cc[self_ind, 2] += matrix[self_ind, self_ind] * hat_E_int

    def assemble_int_bound_pressure_cell(self, g_slave, data_slave, data_edge, grid_swap, cc, matrix, self_ind=1):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a condition on the cell pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.

        """
        mg = data_edge['mortar_grid']
        proj = mg.slave_to_mortar_avg()

        A = proj.T
        shape = (g_slave.num_faces, A.shape[1])

        cc[2, self_ind] -= sps.bmat([[sps.csr_matrix(shape)], [A]]).T

    def assemble_int_bound_source(self, g_slave, data_slave, data_edge, grid_swap, cc, matrix, self_ind=1):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a source term.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.

        """
        mg = data_edge['mortar_grid']
        proj = mg.slave_to_mortar_avg()

        A = proj.T
        shape = (g_slave.num_faces, A.shape[1])
        cc[self_ind, 2] += sps.bmat([[sps.csr_matrix(shape)], [A]])


    def enforce_neumann_int_bound(self, g_master, data_edge, matrix):
        """ Enforce Neumann boundary conditions on a given system matrix.

        Methods based on a mixed variational form need this function to
        implement essential boundary conditions.

        The discretization matrix should be modified in place.

        Parameters:
            g (Grid): On which the equation is discretized
            data (dictionary): Of data related to the discretization.
            matrix (scipy.sparse.matrix): Discretization matrix to be modified.

        """
        mg = data_edge['mortar_grid']

        hat_E_int = self._velocity_dof(g_master, mg)

        dof = np.where(hat_E_int.sum(axis=1).A.astype(np.bool))[0]
        norm = np.linalg.norm(matrix[0, 0].diagonal(), np.inf)

        for row in dof:
            idx = slice(matrix[0, 0].indptr[row], matrix[0, 0].indptr[row + 1])
            matrix[0, 0].data[idx] = 0.

            idx = slice(matrix[0, 2].indptr[row], matrix[0, 2].indptr[row + 1])
            matrix[0, 2].data[idx] = 0.

        d = matrix[0, 0].diagonal()
        d[dof] = norm
        matrix[0, 0].setdiag(d)


    def extract_flux(self, g, up, d=None):
        """  Extract the velocity from a dual virtual element solution.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        up : array (g.num_faces+g.num_cells)
            Solution, stored as [velocity,pressure]
        d: data dictionary associated with the grid.
            Unused, but included for consistency reasons.

        Return
        ------
        u : array (g.num_faces)
            Velocity at each face.

        """
        # pylint: disable=invalid-name
        return up[: g.num_faces]


    def extract_pressure(self, g, up, d=None):
        """  Extract the pressure from a dual virtual element solution.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        up : array (g.num_faces+g.num_cells)
            Solution, stored as [velocity,pressure]
        d: data dictionary associated with the grid.
            Unused, but included for consistency reasons.

        Return
        ------
        p : array (g.num_cells)
            Pressure at each cell.

        """
        # pylint: disable=invalid-name
        return up[g.num_faces :]
