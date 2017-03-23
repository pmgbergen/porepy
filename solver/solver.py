class Solver(object):

#------------------------------------------------------------------------------#

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

#------------------------------------------------------------------------------#

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

#------------------------------------------------------------------------------#
