#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coupling conditions between subdomains for elliptic equations.

Current content:
    Robin-type couplings, as decsribed by Martin et al 2005.

Future content:
    Full continuity conditions between subdomains, to replace the old concept
    of 'DFN' discretizations
    @RUNAR: The periodic conditions you defined should also enter here, don't
    you think?

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

    def __init__(self, keyword):
        self.keyword = keyword

    def key(self):
        return self.keyword + '_'

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

        # Mortar data structure.
        mg = data_edge["mortar_grid"]

        faces_h, cells_h, sign_h = sps.find(g_h.cell_faces)
        ind_faces_h = np.unique(faces_h, return_index=True)[1]
        cells_h = cells_h[ind_faces_h]

        inv_M = sps.diags(1. / mg.cell_volumes)

        # Normal permeability and aperture of the intersection
        inv_k = 1. / (2. * data_edge["kn"])
        aperture_h = data_h["param"].get_aperture()

        proj = mg.high_to_mortar_avg()

        Eta = sps.diags(np.divide(inv_k, proj * aperture_h[cells_h]))

        data_edge[self.key() + 'Robin_discr'] = inv_M * Eta


    def assemble_matrix(self, g_master, g_slave, data_master, data_slave, data_edge, matrix_master, matrix_slave):
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
        assert g_master.dim != g_slave.dim
        grid_swap = g_master.dim < g_slave.dim
        if grid_swap:
            g_master, g_slave = g_slave, g_master
            data_master, data_slave = data_slave, data_master
            matrix_master, matrix_slave = matrix_slave, matrix_master

        _, cc = self.create_block_matrix([g_master, g_slave,
                                          data_edge["mortar_grid"]])

        # The convention, for now, is to put the higher dimensional information
        # in the first column and row in matrix, lower-dimensional in the second
        # and mortar variables in the third
        cc[2, 2] = data_edge[self.key() + 'Robin_discr']

        discr_master = data_master[self.key() + 'discr']

        discr_master.assemble_int_bound_pressure_trace(g_master, data_master, data_edge, grid_swap, cc, matrix_master, self_ind=0)
        discr_master.assemble_int_bound_flux(g_master, data_master, data_edge, grid_swap, cc, matrix_master, self_ind=0)

        discr_slave = data_slave[self.key() + 'discr']

        discr_slave.assemble_int_bound_pressure_cell(g_slave, data_slave, data_edge, grid_swap, cc, matrix_slave, self_ind=1)
        discr_slave.assemble_int_bound_source(g_slave, data_slave, data_edge, grid_swap, cc, matrix_slave, self_ind=1)

        return cc

