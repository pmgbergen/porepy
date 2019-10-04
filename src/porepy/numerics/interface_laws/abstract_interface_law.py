"""
Mother class for all interface laws.
"""
import numpy as np
import scipy.sparse as sps
import porepy as pp

class AbstractInterfaceLaw:
    
    def _key(self):
        return self.keyword + "_"

    def _discretization_key(self):
        return self._key() + pp.DISCRETIZATION    
    
    def ndof(self, mg):
        raise NotImplementedError("Must be implemented by any real interface law")
        
    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """ Discretize the interface law and store the discretization in the
        edge data.

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

        Parameters:
            g_master: Grid on one neighboring subdomain.
            g_slave: Grid on the other neighboring subdomain.
            data_master: Data dictionary for the master suddomain
            data_slave: Data dictionary for the slave subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix_master: original discretization for the master subdomain
            matrix_slave: original discretization for the slave subdomain

            The discretization matrices must be included since they will be
            changed by the imposition of Neumann boundary conditions on the
            internal boundary in some numerical methods (Read: VEM, RT0)

        """        
        raise NotImplementedError("Must be implemented by any real interface law")

    def _define_local_block_matrix(
        self, g_master, g_slave, discr_master, discr_slave, mg, matrix
    ):

        master_ind = 0
        slave_ind = 1

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
        elif not self.ndof(mg) == matrix[master_ind, 2].shape[1]:
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
        rhs[2] = np.zeros(dof_mortar)

        return cc, rhs
