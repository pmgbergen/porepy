""" Module contains super class for all coupling conditions between grids
in mixed-dimensional problems.

The class partly acts as an interface (Java-style), with some methods that are
not implemented, but rather intended as a guide for the development of
concrete couplers.

"""
import numpy as np
import scipy.sparse as sps


class AbstractCoupling(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, discr=None, discr_ndof=None):

        if discr_ndof is None:
            self.discr_ndof = discr.ndof
        else:
            self.discr_ndof = discr_ndof

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Abstract method.
        Return the matrix and righ-hand side for a suitable discretization.

        Parameters:
        -----------
        g_h: grid of higher dimension
        g_l: grid of lower dimension
        data_h: dictionary which stores the data for the higher dimensional
            grid
        data_l: dictionary which stores the data for the lower dimensional
            grid
        data: dictionary which stores the data for the edges of the grid
            bucket

        Returns:
        --------
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
        raise NotImplementedError("Method not implemented")

    # ------------------------------------------------------------------------------#

    def create_block_matrix(self, gs):
        """
        Create the block matrix structure descibed in self.matrix_rhs

        Parameters
        ----------
        gs: grids

        Return
        ------
        dof: array containing the number of dofs for the higher and lower
             dimensional grids, respectively.
        matrix: sparse empty block matrix from the discretization.
        """
        gs = np.atleast_1d(np.asarray(gs))

        # Retrieve the number of degrees of both grids
        dof = np.array([self.discr_ndof(g) for g in gs])

        # Create the block matrix for the contributions
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])

        return dof, cc.reshape((gs.size, gs.size))


# ------------------------------------------------------------------------------#
