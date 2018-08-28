"""
Discretization of the source term of an equation.
"""

import numpy as np
import scipy.sparse as sps

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler


class DualSource(Solver):
    """
    Discretization of the integrated source term
    int q * dx
    over each grid cell.

    All this function does is returning a zero lhs and
    rhs = - param.get_source.physics in a saddle point fashion.
    """

    def __init__(self, physics="flow"):
        self.physics = physics
        Solver.__init__(self)

    def ndof(self, g):
        return g.num_cells + g.num_faces

    def matrix_rhs(self, g, data):
        param = data["param"]
        sources = param.get_source(self)
        lhs = sps.csc_matrix((self.ndof(g), self.ndof(g)))
        assert (
            sources.size == g.num_cells
        ), "There should be one soure value for each cell"

        rhs = np.zeros(self.ndof(g))
        is_p = np.hstack(
            (np.zeros(g.num_faces, dtype=np.bool), np.ones(g.num_cells, dtype=np.bool))
        )

        rhs[is_p] = -sources

        return lhs, rhs


# ------------------------------------------------------------------------------


class DualSourceMixedDim(SolverMixedDim):
    def __init__(self, physics="flow", coupling=None):
        self.physics = physics

        self.discr = DualSource(self.physics)
        self.discr_ndof = self.discr.ndof
        self.coupling_conditions = coupling

        self.solver = Coupler(self.discr, self.coupling_conditions)
        SolverMixedDim.__init__(self)


# ------------------------------------------------------------------------------#


class DualSourceDFN(SolverMixedDim):
    def __init__(self, dim_max, physics="flow"):
        # NOTE: There is no flow along the intersections of the fractures.
        # In this case a mixed solver is considered. We assume only two
        # (contiguous) dimensions active. In the higher dimensional grid the
        # physical problem is discretise with a vem solver and in the lower
        # dimensional grid Lagrange multiplier are used to "glue" the velocity
        # dof at the interface between the fractures.
        # For this reason the matrix_rhs and ndof need to be carefully taking
        # care.

        self.physics = physics
        self.dim_max = dim_max

        self.discr = DualSource(self.physics)
        self.coupling_conditions = None

        kwargs = {"discr_ndof": self.__ndof__, "discr_fct": self.__matrix_rhs__}
        self.solver = Coupler(coupling=None, **kwargs)
        SolverMixedDim.__init__(self)

    def __ndof__(self, g):
        # The highest dimensional problem has the standard number of dof
        # associated with the solver. For the lower dimensional problems, the
        # number of dof is the number of cells.
        if g.dim == self.dim_max:
            return self.discr.ndof(g)
        else:
            return g.num_cells

    def __matrix_rhs__(self, g, data):
        # The highest dimensional problem compute the matrix and rhs, the lower
        # dimensional problem and empty matrix. For the latter, the size of the
        # matrix is the number of cells.
        if g.dim == self.dim_max:
            return self.discr.matrix_rhs(g, data)
        else:
            ndof = self.__ndof__(g)
            return sps.csr_matrix((ndof, ndof)), np.zeros(ndof)


# ------------------------------------------------------------------------------#


class Integral(Solver):
    """
    Discretization of the integrated source term
    int q * dx
    over each grid cell.

    All this function does is returning a zero lhs and
    rhs = - param.get_source.physics in a saddle point fashion.
    """

    def __init__(self, physics="flow"):
        self.physics = physics
        Solver.__init__(self)

    def ndof(self, g):
        return g.num_cells + g.num_faces

    def matrix_rhs(self, g, data):
        param = data["param"]
        sources = param.get_source(self)
        lhs = sps.csc_matrix((self.ndof(g), self.ndof(g)))
        assert (
            sources.size == g.num_cells
        ), "There should be one soure value for each cell"

        rhs = np.zeros(self.ndof(g))
        is_p = np.hstack(
            (np.zeros(g.num_faces, dtype=np.bool), np.ones(g.num_cells, dtype=np.bool))
        )

        rhs[is_p] = -sources

        return lhs, rhs


# ------------------------------------------------------------------------------#
