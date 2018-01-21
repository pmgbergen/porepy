'''
Discretization of the flux term of an equation.
'''

import numpy as np
import scipy.sparse as sps

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler


class Integral(Solver):
    '''
    Discretization of the integrated source term
    int q * dx
    over each grid cell.

    All this function does is returning a zero lhs and
    rhs = param.get_source.physics.
    '''

    def __init__(self, physics='flow'):
        self.physics = physics
        Solver.__init__(self)

    def ndof(self, g):
        return g.num_nodes

    def matrix_rhs(self, g, data):
        param = data['param']
        sources = param.get_source(self)

        cell_nodes = g.cell_nodes()
        nodes, cells, _ = sps.find(cell_nodes)

        rhs = np.zeros(self.ndof(g))

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c+1])
            nodes_loc = nodes[loc]

            rhs[nodes_loc] += sources[c]/(g.dim+1)

        lhs = sps.csc_matrix((self.ndof(g), self.ndof(g)))
        return lhs, rhs

#------------------------------------------------------------------------------

class IntegralMixedDim(SolverMixedDim):
    def __init__(self, physics='flow'):
        self.physics = physics

        self.discr = Integral(self.physics)
        self.discr_ndof = self.discr.ndof
        self.coupling_conditions = None

        self.solver = Coupler(self.discr)
        SolverMixedDim.__init__(self)

#------------------------------------------------------------------------------
