import numpy as np
import scipy.sparse as sps

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler

# ------------------------------------------------------------------------------#


class MassMatrixMixedDim(SolverMixedDim):
    def __init__(self, physics="flow"):
        self.physics = physics

        self.discr = MassMatrix(self.physics)

        self.solver = Coupler(self.discr)


# ------------------------------------------------------------------------------#


class MassMatrix(Solver):

    # ------------------------------------------------------------------------------#

    def __init__(self, physics="flow"):
        self.physics = physics

    # ------------------------------------------------------------------------------#

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        """
        Return the matrix and righ-hand side (null) for a discretization of a
        L2-mass bilinear form with constant test and trial functions.

        The name of data in the input dictionary (data) are:
        phi: array (self.g.num_cells)
            Scalar values which represent the porosity.
            If not given assumed unitary.
        deltaT: Time step for a possible temporal discretization scheme.
            If not given assumed unitary.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Return
        ------
        matrix: sparse dia (g.num_cells, g_num_cells)
            Mass matrix obtained from the discretization.
        rhs: array (g_num_cells)
            Null right-hand side.

        Examples
        --------
        data = {'deltaT': 1e-2, 'phi': 0.3*np.ones(g.num_cells)}
        M, _ = mass.MassMatrix().matrix_rhs(g, data)

        """
        ndof = self.ndof(g)
        phi = data["param"].get_porosity()
        aperture = data["param"].get_aperture()
        coeff = g.cell_volumes * phi / data["deltaT"] * aperture

        return sps.dia_matrix((coeff, 0), shape=(ndof, ndof)), np.zeros(ndof)


##########################################################################


class InvMassMatrixMixedDim(SolverMixedDim):
    def __init__(self, physics="flow"):
        self.physics = physics

        self.discr = InvMassMatrix(self.physics)

        self.solver = Coupler(self.discr)


# ------------------------------------------------------------------------------#


class InvMassMatrix(Solver):

    # ------------------------------------------------------------------------------#

    def __init__(self, physics="flow"):
        self.physics = physics

    # ------------------------------------------------------------------------------#

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        """
        Return the inverse of the matrix and righ-hand side (null) for a
        discretization of a L2-mass bilinear form with constant test and trial
        functions.

        The name of data in the input dictionary (data) are:
        phi: array (self.g.num_cells)
            Scalar values which represent the porosity.
            If not given assumed unitary.
        deltaT: Time step for a possible temporal discretization scheme.
            If not given assumed unitary.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Return
        ------
        matrix: sparse dia (g.num_cells, g_num_cells)
            Inverse of mass matrix obtained from the discretization.
        rhs: array (g_num_cells)
            Null right-hand side.

        Examples
        --------
        data = {'deltaT': 1e-2, 'phi': 0.3*np.ones(g.num_cells)}
        M, _ = mass.InvMassMatrix().matrix_rhs(g, data)

        """
        M, rhs = MassMatrix(physics=self.physics).matrix_rhs(g, data)
        return sps.dia_matrix((1. / M.diagonal(), 0), shape=M.shape), rhs


# ------------------------------------------------------------------------------#
