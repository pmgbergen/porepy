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
        parameter_dictionary_h = data_h[pp.PARAMETERS][self.discr_master.keyword]
        # Mortar data structure.
        mg = data_edge["mortar_grid"]

        faces_h, cells_h, _ = sps.find(g_h.cell_faces)
        ind_faces_h = np.unique(faces_h, return_index=True)[1]
        cells_h = cells_h[ind_faces_h]

        inv_M = sps.diags(1.0 / mg.cell_volumes)

        # Normal permeability and aperture of the intersection
        inv_k = 1.0 / (parameter_dictionary_edge["normal_diffusivity"])
        aperture_h = parameter_dictionary_h["aperture"]

        proj = mg.master_to_mortar_avg()

        Eta = sps.diags(np.divide(inv_k, proj * aperture_h[cells_h]))

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
            g_master, data_master, data_edge, grid_swap, cc, matrix, rhs, master_ind
        )
        self.discr_master.assemble_int_bound_flux(
            g_master, data_master, data_edge, grid_swap, cc, matrix, rhs, master_ind
        )

        self.discr_slave.assemble_int_bound_pressure_cell(
            g_slave, data_slave, data_edge, grid_swap, cc, matrix, rhs, slave_ind
        )
        self.discr_slave.assemble_int_bound_source(
            g_slave, data_slave, data_edge, grid_swap, cc, matrix, rhs, slave_ind
        )

        matrix += cc

        self.discr_master.enforce_neumann_int_bound(
            g_master, data_edge, matrix, False, master_ind
        )

        return matrix, rhs


# ------------------------------------------------------------------------------


class FluxPressureContinuity(RobinCoupling):
    """ A condition for flux and pressure continuity between two domains. A particular
    attention is devoted in the case if these domanins are of equal
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

        """

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
        elif not dof_slave == matrix[slave_ind, slave_ind].shape[1]:
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
        rhs_slave = np.empty(3, dtype=np.object)
        rhs_slave[master_ind] = np.zeros(dof_master)
        rhs_slave[slave_ind] = np.zeros(dof_slave)
        rhs_slave[2] = np.zeros(mg.num_cells)
        # I got some problems with pointers when doing rhs_master = rhs_slave.copy()
        # so just reconstruct everything.
        rhs_master = np.empty(3, dtype=np.object)
        rhs_master[master_ind] = np.zeros(dof_master)
        rhs_master[slave_ind] = np.zeros(dof_slave)
        rhs_master[2] = np.zeros(mg.num_cells)

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
            g_master,
            data_master,
            data_edge,
            False,
            cc_master,
            matrix,
            rhs_master,
            master_ind,
        )
        self.discr_master.assemble_int_bound_flux(
            g_master,
            data_master,
            data_edge,
            False,
            cc_master,
            matrix,
            rhs_master,
            master_ind,
        )

        if g_master.dim == g_slave.dim:
            # Consider this terms only if the grids are of the same dimension, by
            # imposing the same condition with a different sign, due to the normal
            self.discr_slave.assemble_int_bound_pressure_trace(
                g_slave,
                data_slave,
                data_edge,
                True,
                cc_slave,
                matrix,
                rhs_slave,
                slave_ind,
            )

            self.discr_slave.assemble_int_bound_flux(
                g_slave,
                data_slave,
                data_edge,
                True,
                cc_slave,
                matrix,
                rhs_slave,
                slave_ind,
            )
            # We now have to flip the sign of some of the matrices
            # First we flip the sign of the slave flux because the mortar flux points
            # from the master to the slave, i.e., flux_s = -mortar_flux
            cc_slave[slave_ind, 2] = -cc_slave[slave_ind, 2]
            # Then we flip the sign for the pressure continuity since we have
            # We have that p_m - p_s = 0.
            cc_slave[2, slave_ind] = -cc_slave[2, slave_ind]
            rhs_slave[2] = -rhs_slave[2]
            # Note that cc_slave[2, 2] is fliped twice, first for pressure continuity
        else:
            # Consider this terms only if the grids are of different dimension, by
            # imposing pressure trace continuity and conservation of the normal flux
            # through the lower dimensional object.
            self.discr_slave.assemble_int_bound_pressure_cell(
                g_slave,
                data_slave,
                data_edge,
                False,
                cc_slave,
                matrix,
                rhs_slave,
                slave_ind,
            )

            self.discr_slave.assemble_int_bound_source(
                g_slave,
                data_slave,
                data_edge,
                False,
                cc_slave,
                matrix,
                rhs_slave,
                slave_ind,
            )

        # Now, the matrix cc = cc_slave + cc_master expresses the flux and pressure
        # continuities over the mortars.
        # cc[0] -> flux_m = mortar_flux
        # cc[1] -> flux_s = -mortar_flux
        # cc[2] -> p_m - p_s = 0
        matrix += cc_master + cc_slave
        rhs = rhs_master + rhs_slave

        self.discr_master.enforce_neumann_int_bound(
            g_master, data_edge, matrix, False, master_ind
        )

        # Consider this terms only if the grids are of the same dimension
        if g_master.dim == g_slave.dim:
            self.discr_slave.enforce_neumann_int_bound(
                g_slave, data_edge, matrix, True, slave_ind
            )

        return matrix, rhs


# ------------------------------------------------------------------------------

class RobinContact(object):
    """
    Contact condition for elastic problem. This condition defines a Robin condition
    for the stress and displacement jump between slave and master boundaries. g_slave
    and g_master must have the same dimension.

    The contact condition is Newton's third law
    \sigma \cdot n_slave = -\sigma \cdot n_master,
    i.e., traction on the two sides must be equal and oposite, and a Robin-type condition
    on the displacement jump
    MW * \lambda + RW [u] = robin_rhs
    where MW and RW are matrices of size (g_slave.dim, g.slave.dim), and
    \labmda = \sigma \cdot \n_slave.
    The jump operator [\cdot] is given by
    [v] = v_slave - v_master,
    and robin_rhs is a given rhs.
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
        return self._key() + pp.keywords.DISCRETIZATION

    def ndof(self, mg):
        return (mg.dim + 1) * mg.num_cells

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Discretize the Mortar coupling.
        We assume the following two sub-dictionaries to be present in the data_edge
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]

        parameter_dictionary contains the entries:
            robin_weigth (list): a list of mortar_grid.num_cells np.ndarrays of
                shape (mortar_grid.dim + 1, mortar_grid.dim + 1) giving the displacement
                jump weight.
            mortar_weigth (list): a list of mortar_grid.num_cells np.ndarrays of
                shape (mortar_grid.dim + 1, mortar_grid.dim + 1) giving the mortar

        matrix_dictionary will be updated with the following entries:
            mortar_weigth: sps.csc_matrix (mg.num_cells * mg.dim, mg.num_cells * mg.dim)
                The weight matrix applied to the mortar variables.
            robin_weigth: sps.csc_matrix (mg.num_cells * mg.dim, mg.num_cells * mg.dim)
                The weight matrix applied to the displacement jump.

        Parameters:
            g_h: Grid of the master domanin.
            g_l: Grid of the slave domain.
            data_h: Data dictionary for the master domain.
            data_l: Data dictionary for the slave domain.
            data_edge: Data dictionary for the edge between the domains.

        """
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]

        mortar_weight = sps.block_diag(parameter_dictionary_edge["mortar_weight"])
        robin_weight = sps.block_diag(parameter_dictionary_edge["robin_weight"])
        robin_rhs = parameter_dictionary_edge["robin_rhs"]
        matrix_dictionary_edge["mortar_weight"] = mortar_weight
        matrix_dictionary_edge["robin_weight"] = robin_weight
        matrix_dictionary_edge["robin_rhs"] = robin_rhs

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

        """
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]

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
            in RobinContact must match the number of dofs given by the matrix
            """
            )
        elif not dof_slave == matrix[master_ind, slave_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the slave discretization given
            in RobinContact must match the number of dofs given by the matrix
            """
            )
        elif not self.ndof(mg) == matrix[master_ind, 2].shape[1]:
            raise ValueError(
                """The number of dofs of the edge discretization given
            in RobinContact must match the number of dofs given by the matrix
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

        # EK: For some reason, the following lines were necessary to apease python
        rhs_slave = np.empty(3, dtype=np.object)
        rhs_slave[master_ind] = np.zeros(dof_master)
        rhs_slave[slave_ind] = np.zeros(dof_slave)
        rhs_slave[2] = np.zeros(self.ndof(mg))
        # I got some problems with pointers when doing rhs_master = rhs_slave.copy()
        # so just reconstruct everything.
        rhs_master = np.empty(3, dtype=np.object)
        rhs_master[master_ind] = np.zeros(dof_master)
        rhs_master[slave_ind] = np.zeros(dof_slave)
        rhs_master[2] = np.zeros(self.ndof(mg))

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

        # Obtain the displacement trace u_master
        self.discr_master.assemble_int_bound_displacement_trace(
            g_master,
            data_master,
            data_edge,
            False,
            cc_master,
            matrix,
            rhs_master,
            master_ind,
        )
        # set \sigma_master = -\lamba
        self.discr_master.assemble_int_bound_stress(
            g_master,
            data_master,
            data_edge,
            False,
            cc_master,
            matrix,
            rhs_master,
            master_ind,
        )
        # Obtain the displacement trace u_slave
        self.discr_slave.assemble_int_bound_displacement_trace(
            g_slave, data_slave, data_edge, True, cc_slave, matrix, rhs_slave, slave_ind
        )
        # set \sigma_slave = \lamba
        self.discr_slave.assemble_int_bound_stress(
            g_slave, data_slave, data_edge, True, cc_slave, matrix, rhs_slave, slave_ind
        )
        # We now have to flip the sign of some of the matrices
        # First we flip the sign of the master stress because the mortar stress
        # is defined from the slave stress. Then, stress_master = -\lambda
        cc_master[master_ind, 2] = -cc_master[master_ind, 2]
        # Then we flip the sign for the master displacement since the displacement
        # jump is defined as u_slave - u_master
        cc_master[2, master_ind] = -cc_master[2, master_ind]
        rhs_master[2] = -rhs_master[2]
        # Note that cc_master[2, 2] is fliped twice, first in Newton's third law,
        # then for the displacement jump.

        # now, the matrix cc = cc_slave + cc_master expresses the stress and displacement
        # continuities over the mortar grid.
        # cc[0] -> stress_master = mortar_stress
        # cc[1] -> stress_slave = -mortar_stress
        # cc[2] -> mortar_weight * lambda + robin_weight * (u_slave - u_master) = robin_rhs

        # We don't want to enforce the displacement jump, but a Robin condition.
        # We therefore add the mortar variable to the last equation.
        cc_mortar[2, 2] = matrix_dictionary_edge["mortar_weight"]

        # The displacement jump is scaled by a matrix in the Robin condition:
        robin_weight = matrix_dictionary_edge["robin_weight"]
        cc_sm = cc_master + cc_slave
        rhs = rhs_slave + rhs_master
        rhs[2] = robin_weight * rhs[2]
        for i in range(3):
            cc_sm[2, i] = robin_weight * cc_sm[2, i]

        # Now define the complete Robin condition:
        # mortar_weight * \lambda + "robin_weight" * [u] = robin_rhs
        matrix += cc_sm + cc_mortar
        rhs[2] += matrix_dictionary_edge["robin_rhs"]

        # The following two functions might or might not be needed when using
        # a finite element discretization (see RobinCoupling for flow).
        self.discr_master.enforce_neumann_int_bound(
            g_master, data_edge, matrix, False, master_ind
        )
        self.discr_slave.enforce_neumann_int_bound(
            g_slave, data_edge, matrix, True, slave_ind
        )

        return matrix, rhs


class StressDisplacementContinuity(RobinContact):
    """
    Contact condition for elastic problem. This condition defines continuity for
    the stress and displacement jump between slave and master boundaries. g_slave
    and g_master must have the same dimension.

    This contact condition is equivalent as if the slave and master domain was
    a single connected domain (the discrete solution will be different as the
    discretization will be slightly different).
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
        ----------
            g_master: Grid on one neighboring subdomain.
            g_slave: Grid on the other neighboring subdomain.
            data_master: Data dictionary for the master suddomain
            data_slave: Data dictionary for the slave subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix_master: original discretization for the master subdomain
            matrix_slave: original discretization for the slave subdomain

        """

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
        rhs_slave = np.empty(3, dtype=np.object)
        rhs_slave[master_ind] = np.zeros(dof_master)
        rhs_slave[slave_ind] = np.zeros(dof_slave)
        rhs_slave[2] = np.zeros(self.ndof(mg))
        # I got some problems with pointers when doing rhs_master = rhs_slave.copy()
        # so just reconstruct everything.
        rhs_master = np.empty(3, dtype=np.object)
        rhs_master[master_ind] = np.zeros(dof_master)
        rhs_master[slave_ind] = np.zeros(dof_slave)
        rhs_master[2] = np.zeros(self.ndof(mg))

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

        # Obtain the displacement trace u_master
        self.discr_master.assemble_int_bound_displacement_trace(
            g_master,
            data_master,
            data_edge,
            False,
            cc_master,
            matrix,
            rhs_master,
            master_ind,
        )
        # set \sigma_master = -\lamba
        self.discr_master.assemble_int_bound_stress(
            g_master,
            data_master,
            data_edge,
            False,
            cc_master,
            matrix,
            rhs_master,
            master_ind,
        )
        # Obtain the displacement trace u_slave
        self.discr_slave.assemble_int_bound_displacement_trace(
            g_slave, data_slave, data_edge, True, cc_slave, matrix, rhs_slave, slave_ind
        )
        # set \sigma_slave = \lamba
        self.discr_slave.assemble_int_bound_stress(
            g_slave, data_slave, data_edge, True, cc_slave, matrix, rhs_slave, slave_ind
        )
        # We now have to flip the sign of some of the matrices
        # First we flip the sign of the master stress because the mortar stress
        # is defined from the slave stress. Then, stress_master = -\lambda
        cc_master[master_ind, 2] = -cc_master[master_ind, 2]
        # Then we flip the sign for the master displacement since the displacement
        # jump is defined as u_slave - u_master
        cc_master[2, master_ind] = -cc_master[2, master_ind]
        rhs_master[2] = -rhs_master[2]
        # Note that cc_master[2, 2] is fliped twice, first in Newton's third law,
        # then for the displacement jump.

        # now, the matrix cc = cc_slave + cc_master expresses the stress and displacement
        # continuities over the mortar grid.
        # cc[0] -> stress_master = mortar_stress
        # cc[1] -> stress_slave = -mortar_stress
        # cc[2] -> u_slave - u_master = 0

        matrix += cc_master + cc_slave
        rhs = rhs_slave + rhs_master
        # The following two functions might or might not be needed when using
        # a finite element discretization (see RobinCoupling for flow).
        self.discr_master.enforce_neumann_int_bound(
            g_master, data_edge, matrix, False, master_ind
        )

        # Consider this terms only if the grids are of the same dimension
        if g_master.dim == g_slave.dim:
            self.discr_slave.enforce_neumann_int_bound(
                g_slave, data_edge, matrix, True, slave_ind
            )

        return matrix, rhs


class RobinContactBiotPressure(RobinContact):
    """
    This condition adds the fluid pressure contribution to the Robin contact condition.
    The Robin condition says:
    MW * lambda + RW * [u] = robin_rhs,
    where MW (mortar_weight) and RW (robin_weight) are two matrices, and
    [u] = u_slave - u_master is the displacement jump from the slave to the master.
    In Biot the displacement on the  contact boundary (u_slave and u_master) will be a
    linear function of cell center displacement (u), mortar stress (lambda) and cell
    centere fluid pressure (p):
        A * u + B * p + C * lam = u_slave/u_master
    This class adds the contribution B.

    To enforce the full continuity this interface law must be used in combination with
    the RobinContact conditions which adds the contributions A and C
    """

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """ Discretize the robin weight (RW)

        Parameters:
            g_h: Grid of the master domanin.
            g_l: Grid of the slave domain.
            data_h: Data dictionary for the master domain.
            data_l: Data dictionary for the slave domain.
            data_edge: Data dictionary for the edge between the domains.

        """
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]

        robin_weight = sps.block_diag(parameter_dictionary_edge["robin_weight"])
        matrix_dictionary_edge["robin_weight"] = robin_weight

    def assemble_matrix_rhs(
        self, g_master, g_slave, data_master, data_slave, data_edge, matrix
    ):
        """ Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.
        Parameters:
        ----------
            g_master: Grid on one neighboring subdomain.
            g_slave: Grid on the other neighboring subdomain.
            data_master: Data dictionary for the master suddomain
            data_slave: Data dictionary for the slave subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix_master: original discretization for the master subdomain
            matrix_slave: original discretization for the slave subdomain

        """
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]

        self.discretize(g_master, g_slave, data_master, data_slave, data_edge)

        if not g_master.dim == g_slave.dim:
            raise AssertionError("Slave and master must have same dimension")

        master_ind = 0
        slave_ind = 1

        # Generate matrix for the coupling. This can probably be generalized
        # once we have decided on a format for the general variables
        mg = data_edge["mortar_grid"]

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
        rhs[master_ind] = np.zeros(matrix[master_ind, master_ind].shape[1])
        rhs[slave_ind] = np.zeros(matrix[slave_ind, slave_ind].shape[1])
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

        # Obtain the contribution of the cell centered pressure on the displacement
        # trace u_master
        self.discr_master.assemble_int_bound_displacement_trace(
            g_master, data_master, data_edge, False, cc_master, matrix, rhs, master_ind
        )
        # Equivalent for u_slave
        self.discr_slave.assemble_int_bound_displacement_trace(
            g_slave, data_slave, data_edge, True, cc_slave, matrix, rhs, slave_ind
        )
        # We now have to flip the sign of some of the matrices
        # First we flip the sign of the master stress because the mortar stress
        # is defined from the slave stress. Then, stress_master = -\lambda
        cc_master[master_ind, 2] = -cc_master[master_ind, 2]
        # Then we flip the sign for the master displacement since the displacement
        # jump is defined as u_slave - u_master
        cc_master[2, master_ind] = -cc_master[2, master_ind]

        matrix += cc_master + cc_slave

        # The displacement jump is scaled by a matrix in the Robin condition:
        robin_weight = matrix_dictionary_edge["robin_weight"]

        for i in range(3):
            matrix[2, i] = robin_weight * matrix[2, i]

        # The following two functions might or might not be needed when using
        # a finite element discretization (see RobinCoupling for flow).
        self.discr_master.enforce_neumann_int_bound(
            g_master, data_edge, matrix, False, master_ind
        )
        self.discr_slave.enforce_neumann_int_bound(
            g_slave, data_edge, matrix, True, slave_ind
        )

        return matrix, rhs


class DivU_StressMortar(RobinContactBiotPressure):
    """
    This condition adds the stress mortar contribution to the div u term in the
    fluid mass conservation equation of the Biot equations. When fractures are
    present the divergence of u (div_u) will be a function of cell centere displacement,
    boundary conditions and the stress mortar (lambda):
        div_u = A * u + B * u_bc_val + C * lambda
    The class adds the contribution C, while the DivD discretization adds A and B.
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
        ----------
            g_master: Grid on one neighboring subdomain.
            g_slave: Grid on the other neighboring subdomain.
            data_master: Data dictionary for the master suddomain
            data_slave: Data dictionary for the slave subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix_master: original discretization for the master subdomain
            matrix_slave: original discretization for the slave subdomain

        """

        if not g_master.dim == g_slave.dim:
            raise AssertionError("Slave and master must have same dimension")

        master_ind = 0
        slave_ind = 1

        # Generate matrix for the coupling. This can probably be generalized
        # once we have decided on a format for the general variables
        mg = data_edge["mortar_grid"]

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

        # When we do time stepping in Biot the mortar variable from the previous
        # time step will add a contribution to the rhs due to Backward Euler:
        # \partial div_u / \partial_t = (\div_u^k - \div_u^{k-1})/dt.
        rhs_slave = np.empty(3, dtype=np.object)
        rhs_slave[master_ind] = np.zeros(matrix[master_ind, master_ind].shape[1])
        rhs_slave[slave_ind] = np.zeros(matrix[slave_ind, slave_ind].shape[1])
        rhs_slave[2] = np.zeros(self.ndof(mg))
        # I got some problems with pointers when doing rhs_master = rhs_slave.copy()
        # so just reconstruct everything.
        rhs_master = np.empty(3, dtype=np.object)
        rhs_master[master_ind] = np.zeros(matrix[master_ind, master_ind].shape[1])
        rhs_master[slave_ind] = np.zeros(matrix[slave_ind, slave_ind].shape[1])
        rhs_master[2] = np.zeros(self.ndof(mg))

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

        # lambda acts as a boundary condition on the div_u term. Assemble it for the master.
        self.discr_master.assemble_int_bound_stress(
            g_master,
            data_master,
            data_edge,
            False,
            cc_master,
            matrix,
            rhs_master,
            master_ind,
        )
        # Equivalent for the slave
        self.discr_slave.assemble_int_bound_stress(
            g_slave, data_slave, data_edge, True, cc_slave, matrix, rhs_slave, slave_ind
        )
        # We now have to flip the sign of some of the matrices
        # First we flip the sign of the master stress because the mortar stress
        # is defined from the slave stress. Then, stress_master = -\lambda
        cc_master[master_ind, 2] = -cc_master[master_ind, 2]
        rhs_master[master_ind] = -rhs_master[master_ind]

        matrix += cc_master + cc_slave
        rhs = [s + m for s, m in zip(rhs_slave, rhs_master)]

        # The following two functions might or might not be needed when using
        # a finite element discretization (see RobinCoupling for flow).
        self.discr_master.enforce_neumann_int_bound(
            g_master, data_edge, matrix, False, master_ind
        )
        self.discr_slave.enforce_neumann_int_bound(
            g_slave, data_edge, matrix, True, slave_ind
        )

        return matrix, rhs
