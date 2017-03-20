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

        matrix = np.empty((gb.size(), gb.size()), dtype=np.object)
        for g_i, d_i in gb:
            for g_j, d_j in gb:
                pos_i, pos_j = gb.nodes_prop([g_i, g_j], 'node_number')
                matrix[pos_i, pos_j] = sps.coo_matrix((d_i['dof'], d_j['dof']))

        rhs = np.array([np.empty(d['dof']) for _, d in gb])

        for g, d in gb:
            pos = d['node_number']
            data = gb.node_props(g)
            matrix[pos, pos], rhs[pos] = self.solver.matrix_rhs(g, data)

        for e in gb.edges():
            g_l, g_h = gb.sorted_nodes_of_edge(e)
            pos_l, pos_h = gb.nodes_prop([g_l, g_h], 'node_number')
            idx = np.ix_([pos_h, pos_l], [pos_h, pos_l])

            data = gb.edge_props(e)
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
