"""
Mother class for all interface laws.
"""
import numpy as np
import scipy.sparse as sps


class AbstractInterfaceLaw:
    def ndof(self, mg):
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
        #        dof = np.array([dof_master, dof_slave, mg.num_cells])
        dof = np.array([dof_master, dof_slave, dof_mortar])
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        # The rhs is just zeros
        # EK: For some reason, the following lines were necessary to apease python
        rhs = np.empty(3, dtype=np.object)
        rhs[master_ind] = np.zeros(dof_master)
        rhs[slave_ind] = np.zeros(dof_slave)
        rhs[2] = np.zeros(dof_mortar)

        return cc, rhs
