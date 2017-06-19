import numpy as np
import scipy.sparse as sps

class Coupler(object):

#------------------------------------------------------------------------------#

    def __init__(self, solver, coupling=None):
        self.solver = solver
        self.coupling = coupling

#------------------------------------------------------------------------------#

    def ndof(self, gb, data_key=None):
        """
        Store in the grid bucket the number of degrees of freedom associated to
        the method.
        It requires the key "dof" in the grid bucket as reserved.

        Parameter
        ---------
        gb: grid bucket.

        """
        if data_key is None:
            gb.add_node_prop('dof')

        for g, d in gb:
            k_d = self.__get_data(d, data_key)
            k_d['dof'] = self.solver.ndof(g)

#------------------------------------------------------------------------------#

    def matrix_rhs(self, gb, matrix_format="csr", data_key=None):
        """
        Return the matrix and righ-hand side for a suitable discretization, where
        a hierarchy of grids are considered. The matrices are stored in the
        global matrix and right-hand side according to the numeration given by
        "node_number".
        It requires the key "dof" in the grid bucket as reserved.
        It requires the key "node_number" be present in the grid bucket, see
        GridBucket.assign_node_ordering().

        Parameters
        ----------
        gb : grid bucket with geometry fields computed.
        matrix_format: (optional, default is csr) format of the sparse matrix.

        Return
        ------
        matrix: sparse matrix from the discretization.
        rhs: array right-hand side of the problem.
        """
        self.ndof(gb, data_key)
        
        # Initialize the global matrix and rhs to store the local problems
        matrix = np.empty((gb.size(), gb.size()), dtype=np.object)
        rhs = np.empty(gb.size(), dtype=np.object)
        for g_i, d_i in gb:
            k_d_i = self.__get_data(d_i, data_key)
            pos_i = d_i['node_number']
            rhs[pos_i] = np.empty(k_d_i['dof'])
            for g_j, d_j in gb:
                k_d_j = self.__get_data(d_j, data_key)
                pos_j = d_j['node_number']
                matrix[pos_i, pos_j] = sps.coo_matrix((k_d_i['dof'],
                                                       k_d_j['dof']))

        # Loop over the grids and compute the problem matrix
        for g, data in gb:
            k_data = self.__get_data(data, data_key)
            pos = data['node_number']
            matrix[pos, pos], rhs[pos] = self.solver.matrix_rhs(g, k_data)

        # Handle special case of 1-element grids, that give 0-d arrays
        rhs = np.array([np.atleast_1d(a) for a in tuple(rhs)])

        # if the coupling conditions are not given fill only the diagonal part
        if self.coupling is None:
            return sps.bmat(matrix, matrix_format), np.concatenate(tuple(rhs))

        # Loop over the edges of the graph (pair of connected grids) to compute
        # the coupling conditions
        for e, data in gb.edges_props():
            g_l, g_h = gb.sorted_nodes_of_edge(e)
            pos_l, pos_h = gb.nodes_prop([g_l, g_h], 'node_number')
            idx = np.ix_([pos_h, pos_l], [pos_h, pos_l])

            data_l, data_h = gb.node_props(g_l), gb.node_props(g_h)

            k_data_l = self.__get_data(data_l, data_key)
            k_data_h = self.__get_data(data_h, data_key)

            matrix[idx] += self.coupling.matrix_rhs(g_h, g_l, k_data_h,
                                                    k_data_l, data)

        return sps.bmat(matrix, matrix_format), np.concatenate(tuple(rhs))

#------------------------------------------------------------------------------#

    def split(self, gb, key, values, data_key=None):
        """
        Store in the grid bucket the vector, split in the function, solution of
        the problem. The values are extracted from the global solution vector
        according to the numeration given by "node_number".

        Parameters
        ----------
        gb : grid bucket with geometry fields computed.
        key: new name of the solution to be stored in the grid bucket.
        values: array, global solution.

        """
        dofs = np.empty(gb.size(), dtype=int)
        for _, data in gb:
            k_data = self.__get_data(data, data_key)
            dofs[data['node_number']] = k_data['dof']
        dofs = np.r_[0, np.cumsum(dofs)]

        if data_key is None:
            gb.add_node_prop(key)

        for g, d in gb:
            k_data = self.__get_data(data, data_key)
            i = data['node_number']
            k_data[key] = values[slice(dofs[i], dofs[i+1])]

#------------------------------------------------------------------------------#
    def __get_data(self, data, data_key):
        if data_key is not None:
            k_data = data[data_key]
        else:
            k_data = data
        return k_data
        
