"""
Discretization of the source term of an equation.
"""

import numpy as np
import scipy.sparse as sps

import porepy as pp


class DualIntegral():
    """
    Discretization of the integrated source term
    int q * dx
    over each grid cell.

    All this function does is returning a zero lhs and
    rhs = - param.get_source.keyword in a saddle point fashion.
    """

    def __init__(self, keyword="flow"):
        self.keyword = keyword

    def ndof(self, g):
        return g.num_cells + g.num_faces

    def assemble_matrix_rhs(self, g, data):
        sources = data[pp.PARAMETERS][self.keyword]["source"]
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