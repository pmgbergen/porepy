import numpy as np
import scipy.sparse as sps

from porepy.numerics.mixed_dim.abstract_coupling import AbstractCoupling

class DualCoupling(AbstractCoupling):

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
                condition. See the abstract coupling class for a more detailed
                description.
        """
        # pylint: disable=invalid-name

        # Normal permeability and aperture of the intersection to
        # compute the effective normal permeability
        ln = 2*np.divide(data_edge['kn'], data_l['a'])

        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions
        dof, cc = self.create_block_matrix(g_h, g_l)

        # Recover the information for the grid-grid mapping
        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])
        faces, _, sgn = sps.find(g_h.cell_faces)
        sgn = sgn[np.unique(faces, return_index=True)[1]]
        sgn = sgn[faces_h]

        # Compute the off-diagonal terms
        dataIJ, I, J = sgn, g_l.num_faces+cells_l, faces_h
        cc[1, 0] = sps.csr_matrix((dataIJ, (I, J)), (dof[1], dof[0]))
        cc[0, 1] = cc[1, 0].T

        # Compute the diagonal terms
        dataIJ = 1./np.multiply(g_h.face_areas[faces_h], ln[cells_l])
        I, J = faces_h, faces_h
        cc[0, 0] = sps.csr_matrix((dataIJ, (I, J)), (dof[0], dof[0]))

        return cc

#------------------------------------------------------------------------------#
