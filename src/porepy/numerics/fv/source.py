import numpy as np
import scipy.sparse as sps

from porepy.numerics.mixed_dim.solver import Solver
from porepy.numerics.mixed_dim.coupler import Coupler


class Integral(Solver):

    def __init__(self, physics='flow'):
        self.physics = physics
        Solver.__init__(self)

    def ndof(self, g):
        return g.num_cells

    def matrix_rhs(self, g, data):
        param = data['param']
        sources = param.get_source(self)
        lhs = sps.csc_matrix((g.num_cells, g.num_cells))
        assert sources.size == g.num_cells, 'There should be one soure value for each cell'
        return lhs, sources


class Density(Solver):
    def __init__(self, physics='flow'):
        self.physics = physics
        Solver.__init__(self)

    def ndof(self, g):
        return g.num_cells

    def matrix_rhs(self, g, data):
        param = data['param']
        sources = param.get_source(self)
        lhs = sps.csc_matrix((g.num_cells, g.num_cells))
        assert sources.size == g.num_cells, 'There should be one soure value for each cell'
        rhs = sources * g.cell_volumes

        return lhs, rhs


class IntegralMultiDim(Solver):
    def __init__(self, physics='flow'):
        self.physics = physics
        discr = Integral(self.physics)
        self.solver = Coupler(discr)
        Solver.__init__(self)

    def ndof(self, gb):
        ndof = 0
        for g, _ in gb:
            ndof += g.num_cells

    def matrix_rhs(self, gb):
        return self.solver.matrix_rhs(gb)

    def split(self, gb, names, var):
        return self.solver.split(gb, names, var)


class DensityMultiDim(Solver):
    def __init__(self, physics='flow'):
        self.physics = physics
        discr = Density(self.physics)
        self.solver = Coupler(discr)
        Solver.__init__(self)

    def ndof(self, gb):
        ndof = 0
        for g, _ in gb:
            ndof += g.num_cells

    def matrix_rhs(self, gb):
        return self.solver.matrix_rhs(gb)

    def split(self, gb, names, var):
        return self.solver.split(gb, names, var)
