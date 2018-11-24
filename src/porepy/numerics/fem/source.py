"""
Discretization of the flux term of an equation.
"""

import numpy as np
import scipy.sparse as sps

import porepy as pp

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler

# ------------------------------------------------------------------------------#


class P1SourceMixedDim(SolverMixedDim):
    def __init__(self, keyword="flow"):
        self.keyword = keyword

        self.discr = P1Source(self.keyword)
        self.discr_ndof = self.discr.ndof
        self.coupling_conditions = None

        self.solver = Coupler(self.discr)
        SolverMixedDim.__init__(self)


# ------------------------------------------------------------------------------#


class P1Source(Solver):
    """
    Discretization of the integrated source term
    int q * dx
    over each grid cell.

    All this function does is returning a zero lhs and
    rhs = param.get_source.keyword.
    """

    def __init__(self, keyword="flow"):
        self.keyword = keyword
        Solver.__init__(self)

    def ndof(self, g):
        return g.num_nodes

    def matrix_rhs(self, g, data):
        source = data[pp.keywords.PARAMETERS][self.keyword]["source"]

        solver = pp.P1MassMatrix(keyword=self.keyword)
        M = solver.matrix(g, data)

        lhs = sps.csc_matrix((self.ndof(g), self.ndof(g)))

        return lhs, M.dot(source)


# ------------------------------------------------------------------------------#
