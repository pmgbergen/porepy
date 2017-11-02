""" Implementation of a coupler of discretizations for mixed-dimensional
problems.

The class Coupler act as a general purpose coupler for various discretizations
(for now assuming the same disrcetization is applied on all grids). The actual
discretization is constructed by a Solver (within each grid) and an
AbstractCoupler (between grids). In practice, an extension of the two classes
is needed. The Coupler acts as a bookkeeper that knows the global numbering,
and puts submatrices in their right places.

"""
import numpy as np
import scipy.sparse as sps


class Coupler(object):

    #------------------------------------------------------------------------------#

    def __init__(self, solver, coupling=None):
        self.solver = solver
        self.coupling = coupling

#------------------------------------------------------------------------------#

    def ndof(self, gb):
        """
        Store in the grid bucket the number of degrees of freedom associated to
        the method.
        It requires the key "dof" in the grid bucket as reserved.

        Parameter
        ---------
        gb: grid bucket.

        """

        gb.add_node_prop('dof')
        for g, d in gb:
            d['dof'] = self.solver.ndof(g)

#------------------------------------------------------------------------------#

    def matrix_rhs(self, gb, matrix_format="csr"):
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
        self.ndof(gb)

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
            matrix[idx] += self.coupling.matrix_rhs(
                g_h, g_l, data_h, data_l, data)

        return sps.bmat(matrix, matrix_format), np.concatenate(tuple(rhs))

#------------------------------------------------------------------------------#

    def split(self, gb, key, values):
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
        dofs = self._dof_start_of_grids(gb)

        gb.add_node_prop(key)
        for g, d in gb:
            i = d['node_number']
            d[key] = values[slice(dofs[i], dofs[i + 1])]

#------------------------------------------------------------------------------#
    def merge(self, gb, key):
        """
        Merge the stored split function stored in the grid bucket to a vector.
        The values are put into the global  vector according to the numeration
        given by "node_number".

        Parameters
        ----------
        gb : grid bucket with geometry fields computed.
        key: new name of the solution to be stored in the grid bucket.

        Returns
        -------
        values: (ndarray) the values stored in the bucket as an array
        """

        dofs = self._dof_start_of_grids(gb)
        values = np.zeros(dofs[-1])

        for g, d in gb:
            i = d['node_number']
            values[slice(dofs[i], dofs[i + 1])] = d[key]

        return values
#------------------------------------------------------------------------------#

    def _dof_start_of_grids(self ,gb):
        " Helper method to get first global dof for all grids. "
        self.ndof(gb)
        dofs = np.empty(gb.size(), dtype=int)
        for _, d in gb:
            dofs[d['node_number']] = d['dof']
        return np.r_[0, np.cumsum(dofs)]
#------------------------------------------------------------------------------#

    def dof_of_grid(self, gb, g):
        """ Obtain global indices of dof associated with a given grid.

        Parameters:
            gb: Grid_bucket representation of mixed-dimensional data.
            g: Grid, one member of gb.

        Returns:
            np.array of ints: Indices of all dof for the given grid

        """
        dof_list = self._dof_start_of_grids
        nn = gb.node_props(g)['node_number']
        return np.arange(dof_list[nn], dof_list[nn+1])
