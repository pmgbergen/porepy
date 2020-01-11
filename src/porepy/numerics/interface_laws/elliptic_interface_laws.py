"""
Coupling conditions between subdomains for elliptic equations.

Current content:
    Robin-type couplings, as decsribed by Martin et al 2005.
    Full continuity conditions between subdomains
"""
import numpy as np
import scipy.sparse as sps

import porepy as pp
import porepy.numerics.interface_laws.abstract_interface_law


class RobinCoupling(
    porepy.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw
):
    """ A condition with resistance to flow between subdomains. Implementation
        of the model studied (though not originally proposed) by Martin et
        al 2005.

    """

    def __init__(self, keyword, discr_master, discr_slave=None):
        super(RobinCoupling, self).__init__(keyword)
        if discr_slave is None:
            discr_slave = discr_master
        self.discr_master = discr_master
        self.discr_slave = discr_slave

        # This interface law will have direct interface coupling to represent
        # the influence of the flux boundary condition of the secondary
        # interface on the pressure trace on the first interface.
        self.edge_coupling_via_high_dim = True
        # No coupling via lower-dimensional interfaces.
        self.edge_coupling_via_low_dim = False

    def ndof(self, mg):
        return mg.num_cells

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
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]
        # Mortar data structure.
        mg = data_edge["mortar_grid"]

        faces_h, cells_h, _ = sps.find(g_h.cell_faces)
        ind_faces_h = np.unique(faces_h, return_index=True)[1]
        cells_h = cells_h[ind_faces_h]

        inv_M = sps.diags(1.0 / mg.cell_volumes)

        inv_k = 1.0 / (parameter_dictionary_edge["normal_diffusivity"])

        # If normal diffusivity is given as a constant, parse to np.array
        if not isinstance(inv_k, np.ndarray):
            inv_k *= np.ones(mg.num_cells)

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
            matrix: original discretization

            The discretization matrices must be included since they will be
            changed by the imposition of Neumann boundary conditions on the
            internal boundary in some numerical methods (Read: VEM, RT0)

        """
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]

        mg = data_edge["mortar_grid"]

        master_ind = 0
        slave_ind = 1
        cc, rhs = self._define_local_block_matrix(
            g_master, g_slave, self.discr_master, self.discr_slave, mg, matrix
        )

        # The convention, for now, is to put the higher dimensional information
        # in the first column and row in matrix, lower-dimensional in the second
        # and mortar variables in the third
        cc[2, 2] = matrix_dictionary_edge["Robin_discr"]

        self.discr_master.assemble_int_bound_pressure_trace(
            g_master, data_master, data_edge, cc, matrix, rhs, master_ind
        )
        self.discr_master.assemble_int_bound_flux(
            g_master, data_master, data_edge, cc, matrix, rhs, master_ind
        )

        self.discr_slave.assemble_int_bound_pressure_cell(
            g_slave, data_slave, data_edge, cc, matrix, rhs, slave_ind
        )
        self.discr_slave.assemble_int_bound_source(
            g_slave, data_slave, data_edge, cc, matrix, rhs, slave_ind
        )

        matrix += cc

        self.discr_master.enforce_neumann_int_bound(
            g_master, data_edge, matrix, master_ind
        )

        return matrix, rhs

    def assemble_edge_coupling_via_high_dim(
        self,
        g,
        data_grid,
        edge_primary,
        data_primary_edge,
        edge_secondary,
        data_secondary_edge,
        matrix,
    ):
        """ Represent the impact on a primary interface of the mortar (thus boundary)
        flux on a secondary interface.

        Parameters:
            g (pp.Grid): Grid of the higher dimensional neighbor to the main interface.
            data_grid (dict): Data dictionary of the intermediate grid.
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
        mg_primary = data_primary_edge["mortar_grid"]
        mg_secondary = data_secondary_edge["mortar_grid"]

        # Normally, the projections will be pressure from the master (high-dim node)
        # to the primary mortar, and flux from secondary mortar to master
        proj_pressure = mg_primary.master_to_mortar_avg()
        proj_flux = mg_secondary.mortar_to_master_int()

        # If the primary and / or secondary mortar is a boundary mortar grid, things
        # become more complex. This probably assumes that a FluxPressureContinuity
        # discretization is applied on the relevant mortar grid.
        if isinstance(mg_primary, pp.BoundaryMortar) and edge_primary[0] == g:
            proj_pressure = mg_primary.slave_to_mortar_avg()
        if isinstance(mg_secondary, pp.BoundaryMortar) and edge_secondary[0] == g:
            proj_flux = mg_secondary.mortar_to_slave_int()

        cc, rhs = self._define_local_block_matrix_edge_coupling(
            g, self.discr_master, mg_primary, mg_secondary, matrix
        )

        return self.discr_master.assemble_int_bound_pressure_trace_between_interfaces(
            g, data_grid, proj_pressure, proj_flux, cc, matrix, rhs
        )


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

    def __init__(self, keyword, discr_master, discr_slave=None):
        if discr_slave is None:
            discr_slave = discr_master
        self.discr_master = discr_master
        self.discr_slave = discr_slave

        # This interface law will have direct interface coupling to represent
        # the influence of the flux boundary condition of the secondary
        # interface on the pressure trace on the first interface.
        self.edge_coupling_via_high_dim = False
        # No coupling via lower-dimensional interfaces.
        self.edge_coupling_via_low_dim = False

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

        # Generate matrix for the coupling.
        mg = data_edge["mortar_grid"]
        cc_master, rhs_master = self._define_local_block_matrix(
            g_master, g_slave, self.discr_master, self.discr_slave, mg, matrix
        )

        cc_slave = cc_master.copy()

        # I got some problems with pointers when doing rhs_master = rhs_slave.copy()
        # so just reconstruct everything.
        rhs_slave = np.empty(3, dtype=np.object)
        rhs_slave[master_ind] = np.zeros_like(rhs_master[master_ind])
        rhs_slave[slave_ind] = np.zeros_like(rhs_master[slave_ind])
        rhs_slave[2] = np.zeros_like(rhs_master[2])

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
            g_master, data_master, data_edge, cc_master, matrix, rhs_master, master_ind
        )
        self.discr_master.assemble_int_bound_flux(
            g_master, data_master, data_edge, cc_master, matrix, rhs_master, master_ind
        )

        if g_master.dim == g_slave.dim:
            # Consider this terms only if the grids are of the same dimension, by
            # imposing the same condition with a different sign, due to the normal
            self.discr_slave.assemble_int_bound_pressure_trace(
                g_slave,
                data_slave,
                data_edge,
                cc_slave,
                matrix,
                rhs_slave,
                slave_ind,
                use_slave_proj=True,
            )

            self.discr_slave.assemble_int_bound_flux(
                g_slave,
                data_slave,
                data_edge,
                cc_slave,
                matrix,
                rhs_slave,
                slave_ind,
                use_slave_proj=True,
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
                g_slave, data_slave, data_edge, cc_slave, matrix, rhs_slave, slave_ind
            )

            self.discr_slave.assemble_int_bound_source(
                g_slave, data_slave, data_edge, cc_slave, matrix, rhs_slave, slave_ind
            )

        # Now, the matrix cc = cc_slave + cc_master expresses the flux and pressure
        # continuities over the mortars.
        # cc[0] -> flux_m = mortar_flux
        # cc[1] -> flux_s = -mortar_flux
        # cc[2] -> p_m - p_s = 0
        matrix += cc_master + cc_slave
        rhs = rhs_master + rhs_slave

        self.discr_master.enforce_neumann_int_bound(
            g_master, data_edge, matrix, master_ind
        )

        # Consider this terms only if the grids are of the same dimension
        if g_master.dim == g_slave.dim:
            self.discr_slave.enforce_neumann_int_bound(
                g_slave, data_edge, matrix, slave_ind
            )

        return matrix, rhs
