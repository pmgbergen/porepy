"""
Discretization of the source term of an equation for FV methods.
"""

import numpy as np
import scipy.sparse as sps

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler


class Integral(Solver):
    """
    Discretization of the integrated source term
    int q * dx
    over each grid cell.

    All this function does is returning a zero lhs and
    rhs = param.get_source.physics.
    """
    def __init__(self, keyword="flow"):
        """ Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword = keyword
        self.known_keywords = ["flow", "transport", "mechanics"]

        # The physics keyword is kept for consistency for now, but will soon be purged.
        self.physics = keyword

    # ------------------------------------------------------------------------------#

    def _key(self):
        """ Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + '_'

    # ------------------------------------------------------------------------------#

    def ndof(self, g):
        """ Return the number of degrees of freedom associated to the method.
        For scalar equations, the ndof equals the number of cells. For vector equations,
        we multiply by the dimension.

        Parameter:
            g: grid, or a subclass.

        Returns:
            int: the number of degrees of freedom.

        """
        if self.keyword == "flow":
            return g.num_cells
        elif self.keyword == "transport":
            return g.num_cells
        elif self.keyword == "mechanics":
            return g.num_cells * g.dim
        else:
            raise ValueError('Unknown keyword "%s".\n Possible keywords are: %s'
                             % (self.keyword, self.known_keywords))

    # ------------------------------------------------------------------------------#

    def assemble_matrix_rhs(self, g, data):
        """ Return the (null) matrix and right-hand side for a discretization of the
        integrated source term. Also discretize the necessary operators if the data
        dictionary does not contain a source term.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Returns:
            lhs (sparse dia, self.ndof x self.ndof): Null lhs.
            sources (array, self.ndof): Right-hand side vector.

        The names of data in the input dictionary (data) are:
        param (Parameter Class) with the source field set for self.keyword. The assigned
            source values are assumed to be integrated over the cell volumes.
        """
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    # ------------------------------------------------------------------------------#

    def assemble_matrix(self, g, data):
        """ Return the (null) matrix and for a discretization of the integrated source
        term. Also discretize the necessary operators if the data dictionary does not
        contain a source term.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix (self.ndof x self.ndof): Null system matrix of this
                discretization.
        """
        if not self._key() + "source" in data.keys():
            self.discretize(g, data)

        return data[self._key() + "source"]

    # ------------------------------------------------------------------------------#

    def assemble_rhs(self, g, data):
        """ Return the rhs for a discretization of the integrated source term. Also
        discretize the necessary operators if the data dictionary does not contain a
        source term.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix (self.ndof): Right hand side vector representing the
                source.
        """
        if not self._key() + "bound_source" in data.keys():
            self.discretize(g, data)
        return data[self._key() + "bound_source"]

# ------------------------------------------------------------------------------#

    def discretize(self, g, data, faces=None):
        """ Discretize an integrated source term.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Stores:
            lhs (sparse dia, self.ndof x self.ndof): Null lhs, stored as
                self._key() + "source".
            sources (array, self.ndof): Right-hand side vector, stored as
                self._key() + "bound_source".

        The names of data in the input dictionary (data) are:
        param (Parameter Class) with the source field set for self.keyword. The assigned
            source values are assumed to be integrated over the cell volumes.
        """
        param = data["param"]
        sources = param.get_source(self)
        lhs = sps.csc_matrix((self.ndof(g), self.ndof(g)))
        assert sources.size == self.ndof(g), \
            "There should be one source value for each cell"
        data[self._key() + "source"] = lhs
        data[self._key() + "bound_source"] = sources

# ------------------------------------------------------------------------------


class IntegralMixedDim(SolverMixedDim):
    def __init__(self, physics="flow", coupling=None):
        self.physics = physics

        self.discr = Integral(self.physics)
        self.discr_ndof = self.discr.ndof
        self.coupling_conditions = coupling

        self.solver = Coupler(self.discr, self.coupling_conditions)
        SolverMixedDim.__init__(self)


# ------------------------------------------------------------------------------


class IntegralDFN(SolverMixedDim):
    def __init__(self, dim_max, physics="flow"):
        # NOTE: There is no flow along the intersections of the fractures.

        self.physics = physics
        self.dim_max = dim_max

        self.discr = Integral(self.physics)
        self.coupling_conditions = None

        kwargs = {"discr_ndof": self.discr.ndof, "discr_fct": self.__matrix_rhs__}
        self.solver = Coupler(coupling=None, **kwargs)
        SolverMixedDim.__init__(self)

    def __matrix_rhs__(self, g, data):
        # The highest dimensional problem compute the matrix and rhs, the lower
        # dimensional problem and empty matrix. For the latter, the size of the
        # matrix is the number of cells.
        if g.dim == self.dim_max:
            return self.discr.assemble_matrix_rhs(g, data)
        else:
            ndof = self.discr.ndof(g)
            return sps.csr_matrix((ndof, ndof)), np.zeros(ndof)


# ------------------------------------------------------------------------------
