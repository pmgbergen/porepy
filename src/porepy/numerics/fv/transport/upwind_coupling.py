import numpy as np
import scipy.sparse as sps

from porepy_new.src.porepy.numerics.mixed_dim.abstract_coupling import *

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
                condition. See the abstract coupling class for a more detailed
                description.
        """

        # Normal component of the velocity from the higher dimensional grid
        beta_n = data_edge['beta_n']
        assert beta_n is not None

        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions
        dof, cc = self.create_block_matrix(g_h, g_l)

        # Recover the information for the grid-grid mapping
        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])

        # Recover the correct sign for the velocity
        faces, _, sgn = sps.find(g_h.cell_faces)
        sgn = sgn[np.unique(faces, return_index=True)[1]]
        beta_n = sgn[faces_h] * beta_n[faces_h]

        # Determine which are the corresponding cells of the faces_h
        cell_faces_h = g_h.cell_faces.tocsr()[faces_h, :]
        cells_h = cell_faces_h.nonzero()[1]

        diag_cc11 = np.zeros(g_l.num_cells)
        np.add.at(diag_cc11, cells_l, np.sign(beta_n.clip(max=0))*beta_n)
        cc[1,1] = sps.dia_matrix((diag_cc11, 0), shape=(dof[1], dof[1]))

        # Compute the outflow from the higher to the lower dimensional grid
        cc[1,0] = sps.coo_matrix((-beta_n.clip(min=0), (cells_l, cells_h)),
                                 shape=(dof[1], dof[0]))

        # Compute the inflow from the higher to the lower dimensional grid
        cc[0,1] = sps.coo_matrix((beta_n.clip(max=0), (cells_h, cells_l)),
                                 shape=(dof[0], dof[1]))

        return cc

#------------------------------------------------------------------------------#
