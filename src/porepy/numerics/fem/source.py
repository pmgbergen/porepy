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
    def __init__(self, physics="flow"):
        self.physics = physics

        self.discr = P1Source(self.physics)
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
    rhs = param.get_source.physics.
    """

    def __init__(self, physics="flow"):
        self.physics = physics
        Solver.__init__(self)

    def ndof(self, g):
        return g.num_nodes

    def matrix_rhs(self, g, data):
        param = data["param"]

        solver = pp.P1MassMatrix(physics=self.physics)
        M = solver.matrix(g, data)

        lhs = sps.csc_matrix((self.ndof(g), self.ndof(g)))

        return lhs, M.dot(param.get_source(self))

# ------------------------------------------------------------------------------#
