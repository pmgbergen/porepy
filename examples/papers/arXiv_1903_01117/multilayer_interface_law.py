import numpy as np
import scipy.sparse as sps

import porepy as pp


class RobinCouplingMultiLayer(object):
    """ A condition with resistance to flow between subdomains. Implementation
        of the model studied (though not originally proposed) by Martin et
        al 2005.

        @ALL: We should probably make an abstract superclass for all couplers,
        similar to for all elliptic discretizations, so that new
        implementations know what must be done.

    """

    def __init__(self, keyword, discr_master, discr_slave=None):
        self.keyword = keyword
        if discr_slave is None:
            discr_slave = discr_master
        self.discr_master = discr_master
        self.discr_slave = discr_slave

    def _key(self):
        return self.keyword + "_"

    def _discretization_key(self):
        return self._key() + pp.DISCRETIZATION

    @staticmethod
    def ndof(mg):
        return mg.num_cells

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """ Discretize the interface law and store the discretization in the
        edge data.

        TODO: Right now, we are a bit unclear on whether it is required that g_h
        represents the higher-dimensional domain. It should not need to do so.
        TODO: Clean up in the aperture concept.

        Parameters:
            g_h: Grid of the master domanin.
            g_l: Grid of the slave domain.
            data_h: Data dictionary for the master domain.
            data_l: Data dictionary for the slave domain.
            data_edge: Data dictionary for the edge between the domains.

        """
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]
        # Mortar data structure.
        mg = data_edge["mortar_grid"]

        inv_M = sps.diags(1.0 / mg.cell_volumes)

        # Normal permeability and aperture of the intersection
        inv_k = 1.0 / (parameter_dictionary_edge["normal_diffusivity"])
        Eta = sps.diags(inv_k)

        matrix_dictionary_edge["Robin_discr"] = -inv_M * Eta

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
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]
        if not "Robin_discr" in matrix_dictionary_edge:
            self.discretize(g_master, g_slave, data_master, data_slave, data_edge)

        master_ind = 0
        slave_ind = 1

        # Generate matrix for the coupling. This can probably be generalized
        # once we have decided on a format for the general variables
        mg = data_edge["mortar_grid"]

        dof_master = self.discr_master.ndof(g_master)
        dof_slave = self.discr_slave.ndof(g_slave)

        if not dof_master == matrix[master_ind, master_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the master discretization given
            in RobinCoupling must match the number of dofs given by the matrix
            """
            )
        elif not dof_slave == matrix[slave_ind, slave_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the slave discretization given
            in RobinCoupling must match the number of dofs given by the matrix
            """
            )
        elif not mg.num_cells == matrix[master_ind, 2].shape[1]:
            raise ValueError(
                """The number of dofs of the edge discretization given
            in RobinCoupling must match the number of dofs given by the matrix
            """
            )
        # We know the number of dofs from the master and slave side from their
        # discretizations
        dof = np.array(
            [
                matrix[master_ind, master_ind].shape[1],
                matrix[slave_ind, slave_ind].shape[1],
                mg.num_cells,
            ]
        )
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        # The rhs is just zeros
        # EK: For some reason, the following lines were necessary to apease python
        rhs = np.empty(3, dtype=np.object)
        rhs[master_ind] = np.zeros(dof_master)
        rhs[slave_ind] = np.zeros(dof_slave)
        rhs[2] = np.zeros(mg.num_cells)

        # The convention, for now, is to put the higher dimensional information
        # in the first column and row in matrix, lower-dimensional in the second
        # and mortar variables in the third
        cc[2, 2] = matrix_dictionary_edge["Robin_discr"]

        self.discr_master.assemble_int_bound_pressure_cell(
            g_master, data_master, data_edge, True, cc, matrix, master_ind, 1.0
        )
        self.discr_master.assemble_int_bound_source(
            g_master, data_master, data_edge, True, cc, matrix, master_ind, -1.0
        )

        self.discr_slave.assemble_int_bound_pressure_cell(
            g_slave, data_slave, data_edge, False, cc, matrix, slave_ind, -1.0
        )
        self.discr_slave.assemble_int_bound_source(
            g_slave, data_slave, data_edge, False, cc, matrix, slave_ind, 1.0
        )

        matrix += cc

        return matrix, rhs


# ------------------------------------------------------------------------------
