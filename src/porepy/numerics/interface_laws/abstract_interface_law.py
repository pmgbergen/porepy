"""
Mother class for all interface laws.
"""
import numpy as np
import scipy.sparse as sps
import porepy as pp


class AbstractInterfaceLaw:
    """ Partial implementation of an interface (between two grids) law. Any full
    interface law must implement the missing functions.

    Attributes:
        keyword (str): Used to identify the right parameter dictionary from the full
            data dictionary of this grid.
        edge_coupling_via_high_dim (boolean): If True, assembly will allow for a direct
            coupling between different edges. The class must then implement the function
            assemble_edge_coupling_via_high_dim().
        edge_coupling_via_low_dim (boolean): If True, assembly will allow for a direct
            coupling between different edges. The class must then implement the function
            assemble_edge_coupling_via_low_dim().

    """

    def __init__(self, keyword):
        self.keyword = keyword
        self.edge_coupling_via_high_dim = False
        self.edge_coupling_via_low_dim = False

    def _key(self):
        return self.keyword + "_"

    def _discretization_key(self):
        return self._key() + pp.DISCRETIZATION

    def ndof(self, mg):
        """ Get the number of degrees of freedom of this interface law for a
        given mortar grid.

        Parameters:
            mg (pp.MortarGrid): Mortar grid of an interface.

        Returns:
            int: Number of degrees of freedom.

        """
        raise NotImplementedError("Must be implemented by any real interface law")

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """ Discretize the interface law and store the discretization in the
        edge data.

        The discretization matrix will be stored in the data dictionary of this
        interface.

        Parameters:
            g_h: Grid of the master domanin.
            g_l: Grid of the slave domain.
            data_h: Data dictionary for the master domain.
            data_l: Data dictionary for the slave domain.
            data_edge: Data dictionary for the edge between the domains.

        """
        raise NotImplementedError("Must be implemented by any real interface law")

    def assemble_matrix_rhs(
        self, g_master, g_slave, data_master, data_slave, data_edge, matrix
    ):
        """ Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.

        The matrix will be

        Parameters:
            g_master: Grid on one neighboring subdomain.
            g_slave: Grid on the other neighboring subdomain.
            data_master: Data dictionary for the master suddomain
            data_slave: Data dictionary for the slave subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix_master: original discretization for the master subdomain

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the master, slave and mortar variable, respectively.
            np.array: Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the master,
                slave and mortar variable, respectively.

        """
        raise NotImplementedError("Must be implemented by any real interface law")

    def _define_local_block_matrix(
        self, g_master, g_slave, discr_master, discr_slave, mg, matrix
    ):
        """ Initialize a block matrix and right hand side for the local linear
        system of the master and slave grid and the interface.

        The generated block matrix is 3x3, where each block is initialized as
        a sparse matrix with size corresponding to the number of dofs for
        the master, slave and mortar variables for this interface law.

        Parameters:
            g_master: Grid on one neighboring subdomain.
            g_slave: Grid on the other neighboring subdomain.
            data_master: Data dictionary for the master suddomain
            data_slave: Data dictionary for the slave subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix_master: original discretization for the master subdomain

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the master, slave and mortar variable, respectively.
                Each of the blocks have an empty sparse matrix with size
                corresponding to the number of dofs of the grid and variable.
            np.array: Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the master,
                slave and mortar variable, respectively.

        """

        master_ind = 0
        slave_ind = 1
        mortar_ind = 2

        dof_master = discr_master.ndof(g_master)
        dof_slave = discr_slave.ndof(g_slave)
        dof_mortar = self.ndof(mg)

        if not dof_master == matrix[master_ind, master_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the master discretization given
            in the coupling discretization must match the number of dofs given by the matrix
            """
            )
        elif not dof_slave == matrix[master_ind, slave_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the slave discretization given
            in the coupling discretization must match the number of dofs given by the matrix
            """
            )
        elif not self.ndof(mg) == matrix[master_ind, mortar_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the edge discretization given
            in the coupling discretization must match the number of dofs given by the matrix
            """
            )
        # We know the number of dofs from the master and slave side from their
        # discretizations
        dof = np.array([dof_master, dof_slave, dof_mortar])
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        # The rhs is just zeros
        rhs = np.empty(3, dtype=np.object)
        rhs[master_ind] = np.zeros(dof_master)
        rhs[slave_ind] = np.zeros(dof_slave)
        rhs[mortar_ind] = np.zeros(dof_mortar)

        return cc, rhs

    def _define_local_block_matrix_edge_coupling(
        self, g, discr_grid, mg_primary, mg_secondary, matrix
    ):
        """ Initialize a block matrix and right hand side for the local linear
        system of the master and slave grid and the interface.

        The generated block matrix is 3x3, where each block is initialized as
        a sparse matrix with size corresponding to the number of dofs for
        the master, slave and mortar variables for this interface law.

        Parameters:
            g_master: Grid on one neighboring subdomain.
            g_slave: Grid on the other neighboring subdomain.
            data_master: Data dictionary for the master suddomain
            data_slave: Data dictionary for the slave subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix_master: original discretization for the master subdomain

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the master, slave and mortar variable, respectively.
                Each of the blocks have an empty sparse matrix with size
                corresponding to the number of dofs of the grid and variable.
            np.array: Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the master,
                slave and mortar variable, respectively.

        """

        grid_ind = 0
        primary_ind = 1
        secondary_ind = 2

        dof_grid = discr_grid.ndof(g)
        dof_mortar_primary = self.ndof(mg_primary)
        dof_mortar_secondary = self.ndof(mg_secondary)

        if not dof_grid == matrix[grid_ind, grid_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the master discretization given
            in the coupling discretization must match the number of dofs given by the matrix
            """
            )
        elif not dof_mortar_primary == matrix[grid_ind, primary_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the slave discretization given
            in the coupling discretization must match the number of dofs given by the matrix
            """
            )
        elif not dof_mortar_secondary == matrix[grid_ind, secondary_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the edge discretization given
            in the coupling discretization must match the number of dofs given by the matrix
            """
            )
        # We know the number of dofs from the master and slave side from their
        # discretizations
        dof = np.array([dof_grid, dof_mortar_primary, dof_mortar_secondary])
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        # The rhs is just zeros
        rhs = np.empty(3, dtype=np.object)
        rhs[grid_ind] = np.zeros(dof_grid)
        rhs[primary_ind] = np.zeros(dof_mortar_primary)
        rhs[secondary_ind] = np.zeros(dof_mortar_secondary)

        return cc, rhs

    def assemble_edge_coupling_via_high_dim(
        self,
        g_between,
        data_between,
        edge_primary,
        data_edge_primary,
        edge_secondary,
        data_edge_secondary,
        matrix,
    ):
        """ Method to assemble the contribution from one interface to another one.

        The method must be implemented for subclasses of AbstractInterfaceLaw which has
        the attribute edge_coupling_via_high_dim set to True. For classes where the
        variable is False, there is no need for action.

        Note that the mixed-dimensional modeling framework does not allow for direct
        couplings between interfaces. However, there may be cases where an interface law
        is dependent on variables on the boundary between the higher dimensional
        grid and another interface. As we normally associate these boundary values
        with the variable on the secondary interface, this method is available
        as an alternative.

        For more details on how this function is invoked see pp.Assembler.
        Note that the coupling currently only can be invoked if the variables
        on the primary and secondary interface have the same name.

        Any discretization operation should be done as part of self.discretize().

        Parameters:
            g_between (pp.Grid): Grid of the higher dimensional neighbor to the
                main interface
            data_between (dict): Data dictionary of the intermediate grid.
            edge_primary (tuple of grids): The grids of the primary edge
            data_edge_primary (dict): Data dictionary of the primary interface.
            edge_secondary (tuple of grids): The grids of the secondary edge.
            data_edge_secondary (dict): Data dictionary of the secondary interface.
            matrix: original discretization.

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the master grid, the primary and secondary interface,
                respectively.
            np.array: Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the master grid,
                the primary and secondary interface, respectively.

        """
        if self.edge_coupling_via_high_dim:
            raise NotImplementedError(
                """Interface laws with edge couplings via the high
                                      dimensional grid must implement this model"""
            )
        else:
            pass

    def assemble_edge_coupling_via_low_dim(
        self,
        g_between,
        data_between,
        edge_primary,
        data_edge_primary,
        edge_secondary,
        data_edge_secondary,
        matrix,
    ):
        """ Method to assemble the contribution from one interface to another one.

        The method must be implemented for subclasses of AbstractInterfaceLaw which has
        the attribute edge_coupling_via_low_dim set to True. For classes where the
        variable is False, there is no need for action.

        Note that the mixed-dimensional modeling framework does not allow for direct
        couplings between interfaces. However, there may be cases where an interface law
        is dependent on variables on the boundary between the lower-dimensional
        grid and another interface. As we normally associate these boundary values
        with the variable on the secondary interface, this method is available
        as an alternative.

        For more details on how this function is invoked see pp.Assembler.
        Note that the coupling currently only can be invoked if the variables
        on the primary and secondary interface have the same name.

        Any discretization operation should be done as part of self.discretize().

        Parameters:
            g_between (pp.Grid): Grid of the lower-dimensional neighbor to the
                main interface
            data_between (dict): Data dictionary of the intermediate grid.
            edge_primary (tuple of grids): The grids of the primary edge
            data_edge_primary (dict): Data dictionary of the primary interface.
            edge_secondary (tuple of grids): The grids of the secondary edge.
            data_edge_secondary (dict): Data dictionary of the secondary interface.
            matrix: original discretization.

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the master, slave and mortar variable, respectively.
            np.array: Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the master,
                slave and mortar variable, respectively.

        """
        if self.edge_coupling_via_low_dim:
            raise NotImplementedError(
                """Interface laws with edge couplings via the high
                                      dimensional grid must implement this model"""
            )
        else:
            pass
