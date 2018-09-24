#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module contains superclass for mpfa and tpfa.
"""
import numpy as np
import scipy.sparse as sps

import porepy as pp


class FVElliptic(object):
    """
    """

    def __init__(self, keyword):
        self.keyword = keyword

    def key(self):
        return self.keyword + '_'


    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells (pressure dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

    # ------------------------------------------------------------------------------#

    def assemble_matrix_rhs(self, g, data):
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    def assemble_matrix(self, g, data):
        """
        Return the matrix and right-hand side for a discretization of a second
        order elliptic equation using a FV method with a multi-point flux
        approximation.

        The name of data in the input dictionary (data) are:
        k : second_order_tensor
            Permeability defined cell-wise.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data. For details on necessary keywords,
            see method discretize()

        Return
        ------
        matrix: sparse csr (g_num_cells, g_num_cells)
            Discretization matrix.

        """
        if not self.key() + 'flux' in data.keys():
            self.discretize(g, data)

        div = pp.fvutils.scalar_divergence(g)
        flux = data["flux"]
        M = div * flux

        return M

    # ------------------------------------------------------------------------------#

    def assemble_rhs(self, g, data):
        """
        Return the righ-hand side for a discretization of a second order elliptic
        equation using the MPFA method. See self.matrix_rhs for a detaild
        description.
        """
        if not self.key() + 'bound_flux' in data.keys():
            self.discretize(g, data)


        bound_flux = data[self.key() + "bound_flux"]

        param = data["param"]

        bc_val = param.get_bc_val(self)

        div = g.cell_faces.T

        return -div * bound_flux * bc_val

    def assemble_int_bound_flux(self, g, data, data_edge, grid_swap, matrix, self_ind):

        div = g.cell_faces.T

        # Projection operators to grid
        mg = data_edge['mortar_grid']

        if grid_swap:
            proj = mg.slave_to_mortar_avg()
        else:
            proj = mg.master_to_mortar_avg()

        matrix[self_ind, 2] += div * data[self.key() + 'bound_flux'] * proj.T

    def assemble_int_bound_source(self, g, data, data_edge, grid_swap, matrix, self_ind):

        mg = data_edge['mortar_grid']

        if grid_swap:
            proj = mg.master_to_mortar_avg()
        else:
            proj = mg.slave_to_mortar_avg()

        matrix[self_ind, 2] += -proj.T

    def assemble_int_bound_pressure_trace(self, g, data, data_edge, grid_swap, cc, matrix, self_ind):
        """ Assemble operators to represent the pressure trace.
        """
        mg = data_edge['mortar_grid']

        # TODO: this should become first or second or something
        if grid_swap:
            proj = mg.slave_to_mortar_avg()
        else:
            proj = mg.master_to_mortar_avg()

        bp = data[self.key() + 'bound_pressure_cell']
        cc[2, self_ind] += proj * bp
        cc[2, 2] += proj * data[self.key() + 'bound_pressure_face'] * proj.T


    def assemble_int_bound_pressure_cell(self, g, data, data_edge, grid_swap, cc, matrix, self_ind):
        mg = data_edge['mortar_grid']

        if grid_swap:
            proj = mg.master_to_mortar_avg()
        else:
            proj = mg.slave_to_mortar_avg()

        cc[self_ind, 2] += proj.T


    def assemble_internal_boundary_flux(self, g, data_g, data_edge, is_higher, cc, matrix, mortar_is_flux=True):

        #
        if not mortar_is_flux:
            raise NotImplementedError('It is assumed that the mortar variable represents a flux')

        if is_higher:
            div_h = pp.fvutils.scalar_divergence(g)
            cc[0, 2] += div_h * data_edge["mortar_to_hat_bc"]

        else:
            cc[1, 2] -= data_edge["jump"]


    def assemble_internal_boundary_pressure(self, g, data_g, data_edge, is_higher, cc, matrix, mortar_is_flux=True):

        if not mortar_is_flux:
            raise NotImplementedError('It is assumed that the mortar variable represents a flux')

        if is_higher:
            matrix[2, 0] += data_edge["hat_P_to_mortar"]

            hat_P = data_edge["mortar_grid"].high_to_mortar_avg()
            bound_pressure_face_h = data_g["bound_pressure_face"]
            cc[2, 2] += hat_P * bound_pressure_face_h * hat_P.T

        else:
            cc[2, 1] -= data_edge["check_P_to_mortar"]

    def discretize_coupling(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Computes the coupling terms for the faces between cells in g_h and g_l
        using the two-point flux approximation.
        see matrix_rhs for parameters iformation

        @ALL: This method is kept for now, but I want to delete it: It is only
        a rearrangement of more basic information; I consider it better then
        to either construct these operators on-the-fly, or provide some
        utility contruct for easy access.

        """
        # Mortar data structure.
        mg = data_edge["mortar_grid"]

        # Discretization of boundary conditions
        bound_flux_h = data_h["bound_flux"]

        # Matrices for reconstruction of face pressures.
        # Contribution from cell center values
        bound_pressure_cc_h = data_h["bound_pressure_cell"]
        # Contribution from boundary value
        #bound_pressure_face_h = data_h["bound_pressure_face"]

        # Recover the information for the grid-grid mapping
        faces_h, cells_h, _ = sps.find(g_h.cell_faces)
        ind_faces_h = np.unique(faces_h, return_index=True)[1]
        cells_h = cells_h[ind_faces_h]

        # Projection from mortar grid to upper dimension
        hat_P = mg.high_to_mortar_avg()
        # Projection from mortar grid to lower dimension
        check_P = mg.low_to_mortar_avg()

        # Create the block matrix for the contributions
        hat_P_to_mortar = hat_P * bound_pressure_cc_h
        # Normal permeability and aperture of the intersection
        #inv_k = 1. / (2. * data_edge["kn"])
        #aperture_h = data_h["param"].get_aperture()

        # Inverse of the normal permability matrix
        #Eta = sps.diags(np.divide(inv_k, hat_P * aperture_h[cells_h]))

        # Mortar mass matrix
        #M = sps.diags(1. / mg.cell_volumes)
        #mortar_weight = hat_P * bound_pressure_face_h * hat_P.T
        #mortar_weight -= Eta * M

        # store results
        check_size = (g_l.num_faces, mg.num_cells)
        data_edge["mortar_to_hat_bc"] = bound_flux_h * hat_P.T
        data_edge["mortar_to_check_bc"] = sps.csc_matrix(check_size)
        data_edge["jump"] = check_P.T
        data_edge["hat_P_to_mortar"] = hat_P_to_mortar
        data_edge["check_P_to_mortar"] = check_P
        #data_edge["mortar_weight"] = mortar_weight

