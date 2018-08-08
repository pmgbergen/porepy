"""
Discretization of the flux term of an equation.
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

    def __init__(self, physics="flow"):
        self.physics = physics
        Solver.__init__(self)

    def ndof(self, g):
        if self.physics is "flow":
            return g.num_cells

        if self.physics is "transport":
            return g.num_cells

        if self.physics is "mechanics":
            return g.num_cells * g.dim

    def matrix_rhs(self, g, data):
        param = data["param"]
        sources = param.get_source(self)
        lhs = sps.csc_matrix((self.ndof(g), self.ndof(g)))
        assert sources.size == self.ndof(
            g
        ), "There should be one soure value for each cell"
        return lhs, sources


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
            return self.discr.matrix_rhs(g, data)
        else:
            ndof = self.discr.ndof(g)
            return sps.csr_matrix((ndof, ndof)), np.zeros(ndof)


# ------------------------------------------------------------------------------
