"""
Discretization of the flux term of an equation.
"""

import scipy.sparse as sps

import porepy as pp


class P1Source():
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
        source = data[pp.PARAMETERS][self.keyword]["source"]

        solver = pp.P1MassMatrix(keyword=self.keyword)
        M = solver.matrix(g, data)

        lhs = sps.csc_matrix((self.ndof(g), self.ndof(g)))

        return lhs, M.dot(source)