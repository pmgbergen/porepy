import numpy as np
import scipy.sparse as sps

class Coupler(object):

#------------------------------------------------------------------------------#

    def __init__(self, solver, coupling):
        self.solver = solver
        self.coupling = coupling

#------------------------------------------------------------------------------#

    def matrix_rhs(self, gb, matrix_format="csr"):

        gb.assign_node_ordering()
        gb.add_node_prop('dof')
        for g, d in gb:
            d['dof'] = self.solver.ndof(g)

        # Initialize the global matrix and rhs to store the local problems
        matrix = np.empty((gb.size(), gb.size()), dtype=np.object)
        rhs = np.empty(gb.size(), dtype=np.object)
        for g_i, d_i in gb:
            pos_i = d_i['node_number']
            rhs[pos_i] = np.empty(d_i['dof'])
            for g_j, d_j in gb:
                pos_j = d_j['node_number']
                matrix[pos_i, pos_j] = sps.coo_matrix((d_i['dof'], d_j['dof']))

        # Loop over the grids and compute the problem matrix
        for g, data in gb:
            pos = data['node_number']
            matrix[pos, pos], rhs[pos] = self.solver.matrix_rhs(g, data)

        # Loop over the edges of the graph (pair of connected grids) to compute
        # the coupling conditions
        for e, data in gb.edges_props():
            g_l, g_h = gb.sorted_nodes_of_edge(e)
            pos_l, pos_h = gb.nodes_prop([g_l, g_h], 'node_number')
            idx = np.ix_([pos_h, pos_l], [pos_h, pos_l])

            fc = data['face_cells']
            matrix[idx] += self.coupling.matrix_rhs(g_h, g_l, fc, data)

        return sps.bmat(matrix, matrix_format), np.concatenate(tuple(rhs))

#------------------------------------------------------------------------------#

    def split(self, gb, key, values):

        dofs = np.empty(gb.size(), dtype=int)
        for _, d in gb:
            dofs[d['node_number']] = d['dof']
        dofs = np.r_[0, np.cumsum(dofs)]

        gb.add_node_prop(key)
        for g, d in gb:
            i = d['node_number']
            d[key] = values[slice(dofs[i], dofs[i+1])]

#------------------------------------------------------------------------------#
