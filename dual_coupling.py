import numpy as np
import scipy.sparse as sps

class DualCoupling(object):

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

        # Normal permeability of the intersection
        kn = data_edge['kn']
        assert kn is not None

        # Retrieve the number of degrees of both grids
        dof = np.array([self.solver.ndof(g_h), self.solver.ndof(g_l)])

        # Recover the information for the grid-grid mapping
        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])
        faces, _, sgn = sps.find(g_h.cell_faces)
        sgn = sgn[np.unique(faces, return_index=True)[1]]
        sgn = sgn[faces_h]

        # Create the block matrix for the contributions
        cc = np.array([sps.coo_matrix((i,j)) for i in dof for j in dof]\
                     ).reshape((2, 2))

        # Compute the off-diagonal terms
        dataIJ, I, J = sgn, g_l.num_faces+cells_l, faces_h
        cc[1,0] = sps.csr_matrix((dataIJ, (I, J)), (dof[1], dof[0]))
        cc[0,1] = cc[1,0].T

        # Compute the diagonal terms
        dataIJ = 1./np.multiply(g_h.face_areas[faces_h], kn[cells_l])
        I, J = faces_h, faces_h
        cc[0,0] = sps.csr_matrix((dataIJ, (I, J)), (dof[0], dof[0]))

        return cc

#------------------------------------------------------------------------------#
