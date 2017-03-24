import numpy as np
import scipy.sparse as sps

from core.solver.abstract_coupling import *

class UpwindCoupling(AbstractCoupling):

#------------------------------------------------------------------------------#

    def __init__(self, solver):
        self.solver = solver

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Construct the matrix (and right-hand side) for the coupling conditions.
        Note: the right-hand side is not implemented now.

        Parameters:
            g_h: grid of higher dimension
            g_l: grid of lower dimension
            data_h: dictionary which stores the data for the higher dimensional
                grid
            data_l: dictionary which stores the data for the lower dimensional
                grid
            data: dictionary which stores the data for the edges of the grid
                bucket

        Returns:
            cc: block matrix which store the contribution of the coupling
                condition in the following order:
                [ cc_hh  cc_hl ]
                [ cc_lh  cc_ll ]
                where:
                - cc_hh is the contribution to be added to the global block
                  matrix in related to the grid of higher dimension (g_h).
                - cc_ll is the contribution to be added to the global block
                  matrix in related to the grid of lower dimension (g_l).
                  In this case the term is null.
                - cc_hl is the contribution to be added to the global block
                  matrix in related to the coupling between grid of higher
                  dimension (g_h) and the grid of lower dimension (g_l).
                - cc_lh is the contribution to be added to the global block
                  matrix in related to the coupling between grid of lower
                  dimension (g_l) and the grid of higher dimension (g_h).
                  In this case cc_lh is the transpose of cc_hl.

        """

        # Normal component of the velocity from the higher dimensional grid
        beta_n = data_edge['beta_n']
        assert beta_n is not None

        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions
        dof, cc = self.create_block_matrix(g_h, g_l)

        # Recover the information for the grid-grid mapping
        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])

        print( "cells_l", cells_l )
        print( "faces_h", faces_h )

        faces, _, sgn = sps.find(g_h.cell_faces)
        sgn = sgn[np.unique(faces, return_index=True)[1]]

        print( "beta_n", beta_n[faces_h] )

        flow_faces = g_h.cell_faces.copy()
        flow_faces.data *= 0
        flow_faces.data[faces_h] = sgn[faces_h] * beta_n[faces_h]
        flow_faces.data = flow_faces.data[flow_faces.indices]

        if_inflow_faces = flow_faces.copy()
        if_inflow_faces.data = np.sign(if_inflow_faces.data.clip(max=0))

        cc[0,0] = if_inflow_faces.transpose() * flow_faces

        print( sps.find(cc[0,0] ) )

        # Compute the diagonal terms
##        dataIJ = 1./np.multiply(g_h.face_areas[faces_h], kn[cells_l])
##        I, J = faces_h, faces_h
##        cc[0,0] = sps.csr_matrix((dataIJ, (I, J)), (dof[0], dof[0]))

#        cc[1,1] = ...


        return cc

#------------------------------------------------------------------------------#
