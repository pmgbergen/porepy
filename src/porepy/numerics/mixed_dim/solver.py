import numpy as np

""" Module containing super class for all single-domain solvers.

In reality, this is kind of a contract (in the Java sense), there is
implementation, but the methods are meant to guide the implementation of
solvers, with an eye to the needs of mixed-dimensional couplings.

"""


class Solver(object):

    # ------------------------------------------------------------------------------#

    def ndof(self, g):
        """
        Abstract method.
        Return the number of degrees of freedom associated to the method.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        raise NotImplementedError("Method not implemented")

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        """
        Abstract method.
        Return the matrix and righ-hand side for a suitable discretization.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Return
        ------
        matrix: sparse matrix (self.ndof x self.ndof) from the discretization.
        rhs: array (self.ndof)
            Right-hand side of the problem.
        """
        raise NotImplementedError("Method not implemented")


# ------------------------------------------------------------------------------#


class SolverMixedDim(object):
    def __init__(self):
        pass

    def matrix_rhs(self, gb, **kwargs):
        return self.solver.matrix_rhs(gb, **kwargs)

    def split(self, gb, key, values, **kwargs):
        return self.solver.split(gb, key, values, **kwargs)

    def ndof(self, gb):
        return np.sum([self.discr_ndof(g) for g, _ in gb])


# ------------------------------------------------------------------------------#
