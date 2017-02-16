import numpy as np
import scipy.sparse as sps

class Coupler(object):

#------------------------------------------------------------------------------#

    def __init__(self, solver, coupling):
        self.solver = solver
        self.coupling = coupling

#------------------------------------------------------------------------------#

    def matrix_rhs(self, gb, data, matrix_format="csr"):

        dof = np.array([self.solver.ndof(g) for g, _ in gb])

        matrix = np.array([sps.coo_matrix((i,j)) for i in dof for j in dof]\
                         ).reshape((gb.size, gb.size))
        rhs = np.array([np.empty(i) for i in dof])

        for g, v in gb:
            pos = int(v)
            matrix[pos, pos], rhs[pos] = self.solver.matrix_rhs(g, data[0][v])

        # loop on the edges
        for e in gb.e():
            v_h, v_l = gb.v_of_e(e)
            g_h, g_l = gb.grids[v_h], gb.grids[v_l]
            idx = np.ix_([int(v_h), int(v_l)], [int(v_h), int(v_l)])

            matrix[idx] += self.coupling.matrix_rhs(g_h, g_l, data[1][e])

        return sps.bmat(matrix, matrix_format), np.concatenate(tuple(rhs))

#------------------------------------------------------------------------------#

    def split(self, gb, values):

        values_split = gb.g_prop()
        ndof = 0
        for g, v in gb:
            dof_v = self.solver.ndof(g)
            values_split[v] = values[slice(ndof, ndof+dof_v)]
            ndof += dof_v
        return values_split

#------------------------------------------------------------------------------#
