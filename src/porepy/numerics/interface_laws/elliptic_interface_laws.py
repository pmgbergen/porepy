"""
Coupling conditions between subdomains for elliptic equations.

Current content:
    Robin-type couplings, as decsribed by Martin et al 2005.
    Full continuity conditions between subdomains
"""
import numpy as np
import scipy.sparse as sps

import porepy as pp


class RobinCoupling(object):
    """ A condition with resistance to flow between subdomains. Implementation
        of the model studied (though not originally proposed) by Martin et
        al 2005.

        @ALL: We should probably make an abstract superclass for all couplers,
        similar to for all elliptic discretizations, so that new
        implementations know what must be done.

    """

    def __init__(self, keyword, discr_master, discr_slave=None):
        # @ALL should the node discretization default to Tpfa?
        self.keyword = keyword
        if discr_slave is None:
            discr_slave = discr_master
        self.discr_master = discr_master
        self.discr_slave = discr_slave

    def _key(self):
        return self.keyword + "_"

    def _discretization_key(self):
        return self._key() + pp.DISCRETIZATION

    def ndof(self, mg):
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
        parameter_dictionary_h = data_h[pp.PARAMETERS][self.keyword]
        # Mortar data structure.
        mg = data_edge["mortar_grid"]

        faces_h, cells_h, _ = sps.find(g_h.cell_faces)
        ind_faces_h = np.unique(faces_h, return_index=True)[1]
        cells_h = cells_h[ind_faces_h]

        inv_M = sps.diags(1.0 / mg.cell_volumes)

        # Normal permeability and aperture of the intersection
        inv_k = 1.0 / (2.0 * parameter_dictionary_edge["normal_diffusivity"])
        aperture_h = parameter_dictionary_h["aperture"]

        proj = mg.master_to_mortar_avg()

        Eta = sps.diags(np.divide(inv_k, proj * aperture_h[cells_h]))

        # @ALESSIO, @EIRIK: the tpfa and vem couplers use different sign
        # conventions here. We should be very careful.
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

        grid_swap = g_master.dim < g_slave.dim
        if grid_swap:
            g_master, g_slave = g_slave, g_master
            data_master, data_slave = data_slave, data_master

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
        elif not dof_slave == matrix[master_ind, slave_ind].shape[1]:
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
        #        dof = np.array([dof_master, dof_slave, mg.num_cells])
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

        self.discr_master.assemble_int_bound_pressure_trace(
            g_master, data_master, data_edge, grid_swap, cc, matrix, master_ind
        )
        self.discr_master.assemble_int_bound_flux(
            g_master, data_master, data_edge, grid_swap, cc, matrix, master_ind
        )

        self.discr_slave.assemble_int_bound_pressure_cell(
            g_slave, data_slave, data_edge, grid_swap, cc, matrix, slave_ind
        )
        self.discr_slave.assemble_int_bound_source(
            g_slave, data_slave, data_edge, grid_swap, cc, matrix, slave_ind
        )

        matrix += cc

        self.discr_master.enforce_neumann_int_bound(
            g_master, data_edge, matrix, False, master_ind
        )

        return matrix, rhs


# ------------------------------------------------------------------------------


class FluxPressureContinuity(RobinCoupling):
    """ A condition for flux and pressure continuity between two domains of equal
    dimension. This can be used to specify full continuity between fractures,
    two domains or a periodic boundary condition for a single domain. The faces
    coupled by flux and pressure condition must be specified by a MortarGrid on
    a graph edge.
    For each face we will impose
    v_m = lambda
    v_s = -lambda
    p_m - p_s = 0
    where subscript m and s is for master and slave, v is the flux, p the pressure,
    and lambda the mortar variable.

    @Allesio, Eirik:
    TODO: It might only works for methods that do not change the discretization
          matrix. We flip the sign of the pressure and flux on the slave side,
          and I don't know if this will effect the changed to the discretization
          matrix.
    """

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """ Nothing really to do here

        Parameters:
            g_h: Grid of the master domanin.
            g_l: Grid of the slave domain.
            data_h: Data dictionary for the master domain.
            data_l: Data dictionary for the slave domain.
            data_edge: Data dictionary for the edge between the domains.

        """
        pass

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

        if not g_master.dim == g_slave.dim:
            raise AssertionError("Slave and master must have same dimension")

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
            in FluxPressureContinuity must match the number of dofs given by the matrix
            """
            )
        elif not dof_slave == matrix[master_ind, slave_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the slave discretization given
            in FluxPressureContinuity must match the number of dofs given by the matrix
            """
            )
        elif not mg.num_cells == matrix[master_ind, 2].shape[1]:
            raise ValueError(
                """The number of dofs of the edge discretization given
            in FluxPressureContinuity must match the number of dofs given by the matrix
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
        cc_master = cc.reshape((3, 3))
        cc_slave = cc_master.copy()

        # The rhs is just zeros
        # EK: For some reason, the following lines were necessary to apease python
        rhs = np.empty(3, dtype=np.object)
        rhs[master_ind] = np.zeros(dof_master)
        rhs[slave_ind] = np.zeros(dof_slave)
        rhs[2] = np.zeros(mg.num_cells)

        # The convention, for now, is to put the master grid information
        # in the first column and row in matrix, slave grid in the second
        # and mortar variables in the third
        # If master and slave is the same grid, they should contribute to the same
        # row and coloumn. When the assembler assigns matrix[idx] it will only add
        # the slave information because of duplicate indices (master and slave is the same).
        # We therefore write the both master and slave info to the slave index.
        if g_master == g_slave:
            master_ind = 1
        else:
            master_ind = 0

        self.discr_master.assemble_int_bound_pressure_trace(
            g_master, data_master, data_edge, False, cc_master, matrix, master_ind
        )
        self.discr_master.assemble_int_bound_flux(
            g_master, data_master, data_edge, False, cc_master, matrix, master_ind
        )

        self.discr_slave.assemble_int_bound_pressure_trace(
            g_slave, data_slave, data_edge, True, cc_slave, matrix, slave_ind
        )

        self.discr_slave.assemble_int_bound_flux(
            g_slave, data_slave, data_edge, True, cc_slave, matrix, slave_ind
        )
        # We now have to flip the sign of some of the matrices
        # First we flip the sign of the slave flux because the mortar flux points
        # from the master to the slave, i.e., flux_s = -mortar_flux
        cc_slave[slave_ind, 2] = -cc_slave[slave_ind, 2]
        # Then we flip the sign for the pressure continuity since we have
        # We have that p_m - p_s = 0.
        cc_slave[2, slave_ind] = -cc_slave[2, slave_ind]

        # Note that cc_slave[2, 2] is fliped twice, first for pressure continuity
        # now, the matrix cc = cc_slave + cc_master expresses the flux and pressure
        # continuities over the mortars.
        # cc[0] -> flux_m = mortar_flux
        # cc[1] -> flux_s = -mortar_flux
        # cc[2] -> p_m - p_s = 0
        matrix += cc_master + cc_slave

        self.discr_master.enforce_neumann_int_bound(
            g_master, data_edge, matrix, False, master_ind
        )
        self.discr_slave.enforce_neumann_int_bound(
            g_slave, data_edge, matrix, True, slave_ind
        )

        return matrix, rhs


class RobinContact(object):
    """Contact condition for elastic problem
    """

    def __init__(self, keyword, discr_master, discr_slave=None):
        # @ALL should the node discretization default to Tpfa?
        self.keyword = keyword
        if discr_slave is None:
            discr_slave = discr_master
        self.discr_master = discr_master
        self.discr_slave = discr_slave

    def _key(self):
        return self.keyword + "_"

    def _discretization_key(self):
        return self._key() + pp.keywords.DISCRETIZATION

    def ndof(self, mg):
        return (mg.dim + 1) * mg.num_cells

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """ Nothing really to do here

        Parameters:
            g_h: Grid of the master domanin.
            g_l: Grid of the slave domain.
            data_h: Data dictionary for the master domain.
            data_l: Data dictionary for the slave domain.
            data_edge: Data dictionary for the edge between the domains.

        """
        # Mortar data structure.
        mg = data_edge["mortar_grid"]

        # Zero in normal direction and ones in tangential
        mortar_weight = sps.block_diag(data_edge["mortar_weight"])
        robin_weight = sps.block_diag(data_edge["robin_weight"])
        data_edge[self._key() + "mortar_weight"] = mortar_weight
        data_edge[self._key() + "robin_weight"] = robin_weight

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
        if not self._key() + "contact_discr" in data_edge.keys():
            self.discretize(g_master, g_slave, data_master, data_slave, data_edge)

        if not g_master.dim == g_slave.dim:
            raise AssertionError("Slave and master must have same dimension")

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
            in FluxPressureContinuity must match the number of dofs given by the matrix
            """
            )
        elif not dof_slave == matrix[master_ind, slave_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the slave discretization given
            in FluxPressureContinuity must match the number of dofs given by the matrix
            """
            )
        elif not self.ndof(mg) == matrix[master_ind, 2].shape[1]:
            raise ValueError(
                """The number of dofs of the edge discretization given
            in FluxPressureContinuity must match the number of dofs given by the matrix
            """
            )
        # We know the number of dofs from the master and slave side from their
        # discretizations
        dof = np.array(
            [
                matrix[master_ind, master_ind].shape[1],
                matrix[slave_ind, slave_ind].shape[1],
                self.ndof(mg),
            ]
        )
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc_master = cc.reshape((3, 3))
        cc_slave = cc_master.copy()
        cc_mortar = cc_master.copy()

        # The rhs is just zeros
        # EK: For some reason, the following lines were necessary to apease python
        rhs = np.empty(3, dtype=np.object)
        rhs[master_ind] = np.zeros(dof_master)
        rhs[slave_ind] = np.zeros(dof_slave)
        rhs[2] = data_edge["rhs"]

        # The convention, for now, is to put the master grid information
        # in the first column and row in matrix, slave grid in the second
        # and mortar variables in the third
        # If master and slave is the same grid, they should contribute to the same
        # row and coloumn. When the assembler assigns matrix[idx] it will only add
        # the slave information because of duplicate indices (master and slave is the same).
        # We therefore write the both master and slave info to the slave index.
        if g_master == g_slave:
            master_ind = 1
        else:
            master_ind = 0

        self.discr_master.assemble_int_bound_displacement_trace(
            g_master, data_master, data_edge, False, cc_master, matrix, master_ind
        )

        self.discr_master.assemble_int_bound_stress(
            g_master, data_master, data_edge, False, cc_master, matrix, master_ind
        )

        self.discr_slave.assemble_int_bound_displacement_trace(
            g_slave, data_slave, data_edge, True, cc_slave, matrix, slave_ind
        )

        self.discr_slave.assemble_int_bound_stress(
            g_slave, data_slave, data_edge, True, cc_slave, matrix, slave_ind
        )
        # We now have to flip the sign of some of the matrices
        # First we flip the sign of the master stress because the mortar stress
        # is defined from the slave stress. Then, flux_m = -mortar_flux
        cc_master[master_ind, 2] = -cc_master[slave_ind, 2]
        # Then we flip the sign for the displacement continuity since we have
        # We have that p_m - p_s = 0.
        cc_master[2, master_ind] = -cc_master[2, master_ind]
        # Note that cc_master[2, 2] is fliped twice, first for displacement continuity
        # now, the matrix cc = cc_slave + cc_master expresses the flux and pressure
        # continuities over the mortars.
        # cc[0] -> flux_m = mortar_stress
        # cc[1] -> flux_s = -mortar_stress
        # cc[1] -> u_s - u_m = 0

        # Finally we add the mortar discretization
        cc_mortar[2, 2] = data_edge[self._key() + "mortar_weight"]

        # multiply by robin weight
        robin_weight = data_edge[self._key() + "robin_weight"]
        cc_sm = cc_master + cc_slave
        for i in range(3):
            cc_sm[2, i] = robin_weight * cc_sm[2, i]

        matrix += cc_sm + cc_mortar

        self.discr_master.enforce_neumann_int_bound(
            g_master, data_edge, matrix, False, master_ind
        )
        self.discr_slave.enforce_neumann_int_bound(
            g_slave, data_edge, matrix, True, slave_ind
        )

        return matrix, rhs


class StressDisplacementContinuity(RobinContact):
    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """ Nothing really to do here

        Parameters:
            g_h: Grid of the master domanin.
            g_l: Grid of the slave domain.
            data_h: Data dictionary for the master domain.
            data_l: Data dictionary for the slave domain.
            data_edge: Data dictionary for the edge between the domains.

        """
        pass

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
        if not self._key() + "contact_discr" in data_edge.keys():
            self.discretize(g_master, g_slave, data_master, data_slave, data_edge)

        if not g_master.dim == g_slave.dim:
            raise AssertionError("Slave and master must have same dimension")

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
            in FluxPressureContinuity must match the number of dofs given by the matrix
            """
            )
        elif not dof_slave == matrix[master_ind, slave_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the slave discretization given
            in FluxPressureContinuity must match the number of dofs given by the matrix
            """
            )
        elif not self.ndof(mg) == matrix[master_ind, 2].shape[1]:
            raise ValueError(
                """The number of dofs of the edge discretization given
            in FluxPressureContinuity must match the number of dofs given by the matrix
            """
            )
        # We know the number of dofs from the master and slave side from their
        # discretizations
        dof = np.array(
            [
                matrix[master_ind, master_ind].shape[1],
                matrix[slave_ind, slave_ind].shape[1],
                self.ndof(mg),
            ]
        )
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc_master = cc.reshape((3, 3))
        cc_slave = cc_master.copy()
        # The rhs is just zeros
        # EK: For some reason, the following lines were necessary to apease python
        rhs = np.empty(3, dtype=np.object)
        rhs[master_ind] = np.zeros(dof_master)
        rhs[slave_ind] = np.zeros(dof_slave)
        rhs[2] = np.zeros(self.ndof(mg))

        # The convention, for now, is to put the master grid information
        # in the first column and row in matrix, slave grid in the second
        # and mortar variables in the third
        # If master and slave is the same grid, they should contribute to the same
        # row and coloumn. When the assembler assigns matrix[idx] it will only add
        # the slave information because of duplicate indices (master and slave is the same).
        # We therefore write the both master and slave info to the slave index.
        if g_master == g_slave:
            master_ind = 1
        else:
            master_ind = 0

        self.discr_master.assemble_int_bound_displacement_trace(
            g_master, data_master, data_edge, False, cc_master, matrix, master_ind
        )
        self.discr_master.assemble_int_bound_stress(
            g_master, data_master, data_edge, False, cc_master, matrix, master_ind
        )

        self.discr_slave.assemble_int_bound_displacement_trace(
            g_slave, data_slave, data_edge, True, cc_slave, matrix, slave_ind
        )

        self.discr_slave.assemble_int_bound_stress(
            g_slave, data_slave, data_edge, True, cc_slave, matrix, slave_ind
        )
        # We now have to flip the sign of some of the matrices
        # First we flip the sign of the master stress because the mortar stress
        # is defined from the slave stress. Then, flux_m = -mortar_flux
        cc_master[master_ind, 2] = -cc_master[slave_ind, 2]
        # Then we flip the sign for the displacement continuity since we have
        # We have that p_m - p_s = 0.
        cc_master[2, master_ind] = -cc_master[2, master_ind]

        # Note that cc_master[2, 2] is fliped twice, first for displacement continuity
        # now, the matrix cc = cc_slave + cc_master expresses the flux and pressure
        # continuities over the mortars.
        # cc[0] -> flux_m = mortar_stress
        # cc[1] -> flux_s = -mortar_stress
        # cc[1] -> u_s - u_m = 0

        #        cc_master[2, 2] = sps.eye(self.ndof(mg))
        matrix += cc_master + cc_slave

        self.discr_master.enforce_neumann_int_bound(
            g_master, data_edge, matrix, False, master_ind
        )
        self.discr_slave.enforce_neumann_int_bound(
            g_slave, data_edge, matrix, True, slave_ind
        )

        return matrix, rhs
