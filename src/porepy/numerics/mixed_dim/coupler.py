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

    # ------------------------------------------------------------------------------#

    def __init__(self, discr=None, coupling=None, **kwargs):

        # Consider the dofs
        discr_ndof = kwargs.get("discr_ndof")
        if discr_ndof is None:
            self.discr_ndof = discr.ndof
        else:
            self.discr_ndof = discr_ndof

        # Consider the solver for each dimension
        discr_fct = kwargs.get("discr_fct")
        if discr_fct is None:
            self.discr_fct = discr.matrix_rhs
        else:
            self.discr_fct = discr_fct

        # Consider the coupling between dimensions
        coupling_fct = kwargs.get("coupling_fct")

        # Choose coupling or coupling_fct
        if coupling is None and coupling_fct is None:
            # if both are None, assign None coupling
            self.coupling_fct = [None]
        elif coupling_fct is not None:
            # make sure coupling_fct is list
            if not isinstance(coupling_fct, list):
                coupling_fct = [coupling_fct]
            self.coupling_fct = [c for c in coupling_fct]
        else: # we assign coupling as coupling function
            # Make sure coupling is list
            if not isinstance(coupling, list):
                coupling = [coupling]
            coupling_matrix_rhs = []
            for c in coupling:
                if c is None: # we assign None coupling
                    coupling_matrix_rhs.append(c)
                else: # Assign the matrix rhs coupling function
                    coupling_matrix_rhs.append(c.matrix_rhs)
            self.coupling_fct = [c for c in coupling_matrix_rhs]

        # assign how many sets of mortars we need
        self.num_mortars = len(self.coupling_fct)

    # ------------------------------------------------------------------------------#

    def ndof(self, gb):
        """
        Store in the grid bucket the number of degrees of freedom associated to
        the method included the mortar variables.
        It requires the key "dof" in the grid bucket as reserved for the nodes
        and for the edges.

        Parameter
        ---------
        gb: grid bucket.

        """
        gb.add_node_props("dof")
        for g, d in gb:
            d["dof"] = self.discr_ndof(g)

        gb.add_edge_props("dof")
        for _, d in gb.edges():
            d["dof"] = [d["mortar_grid"].num_cells] * self.num_mortars

    # ------------------------------------------------------------------------------#

    def initialize_matrix_rhs(self, gb):
        """
        Initialize the block global matrix. Including the part for the mortars.
        We assume that the first (0:gb.num_graph_nodes()) blocks are reserved for
        the physical domains, while the last (gb.num_graph_nodes() +
        0:gb.num_graph_edges()) are reserved for the mortar coupling

        Parameter:
            gb: grid bucket.
        Return:
            matrix: the block global matrix.
            rhs: the global right-hand side.
        """
        # Initialize the global matrix and rhs to store the local problems
        num_nodes = gb.num_graph_nodes()
        num_edges = gb.num_graph_edges()
        size = num_nodes + self.num_mortars * num_edges
        shapes = np.empty((size, size), dtype=np.object)

        # Initialize the shapes for the matrices and rhs for all the sub-blocks
        for _, d_i in gb:
            pos_i = d_i["node_number"]
            for _, d_j in gb:
                pos_j = d_j["node_number"]
                shapes[pos_i, pos_j] = (d_i["dof"], d_j["dof"])
            for _, d_e in gb.edges():
                for i in range(self.num_mortars):
                    pos_e = d_e["edge_number"] + num_nodes + i * num_edges
                    shapes[pos_i, pos_e] = (d_i["dof"], d_e["dof"][i])
                    shapes[pos_e, pos_i] = (d_e["dof"][i], d_i["dof"])

        for _, d_e in gb.edges():
            for i in range(self.num_mortars):
                pos_e = d_e["edge_number"] + num_nodes + i * num_edges
                dof_e = d_e["dof"][i]
                for _, d_f in gb.edges():
                    for j in range(self.num_mortars):
                        pos_f = d_f["edge_number"] + num_nodes + j * num_edges
                        shapes[pos_e, pos_f] = (dof_e, d_f["dof"][j])

        # initialize the matrix and rhs
        matrix = np.empty(shapes.shape, dtype=np.object)
        rhs = np.empty(shapes.shape[0], dtype=np.object)

        for i in np.arange(shapes.shape[0]):
            rhs[i] = np.zeros(shapes[i, i][0])
            for j in np.arange(shapes.shape[1]):
                matrix[i, j] = sps.coo_matrix(shapes[i, j])

        return matrix, rhs

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self, gb, matrix_format="csr", return_bmat=False):
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
        matrix, rhs = self.initialize_matrix_rhs(gb)

        # Loop over the grids and compute the problem matrix
        for g, d in gb:
            pos = d["node_number"]
            matrix[pos, pos], rhs[pos] = self.discr_fct(g, d)

        # Handle special case of 1-element grids, that give 0-d arrays
        rhs = np.array([np.atleast_1d(a) for a in tuple(rhs)])

        # if the coupling conditions are not given fill only the diagonal part
        if self.num_mortars == 1 and self.coupling_fct[0] is None:
            if return_bmat:
                return matrix, rhs
            else:
                return sps.bmat(matrix, matrix_format), np.concatenate(tuple(rhs))

        # Loop over the edges of the graph (pair of connected grids) to compute
        # the coupling conditions
        num_nodes = gb.num_graph_nodes()
        num_edges = gb.num_graph_edges()
        for e, d in gb.edges():
            for i in range(self.num_mortars):
                g_l, g_h = gb.nodes_of_edge(e)
                pos_l = gb.node_props(g_l, "node_number")
                pos_h = gb.node_props(g_h, "node_number")
                pos_m = d["edge_number"] + num_nodes + i * num_edges
                if pos_h == pos_l:
                    idx = np.ix_([pos_h, pos_m], [pos_h, pos_m])
                else:
                    idx = np.ix_([pos_h, pos_l, pos_m], [pos_h, pos_l, pos_m])

                data_l, data_h = gb.node_props(g_l), gb.node_props(g_h)
                if self.coupling_fct[i] is None:
                    # if coupling is not given, skip
                    pass
                else:
                    matrix[idx] = self.coupling_fct[i](
                        matrix[idx], g_h, g_l, data_h, data_l, d
                    )

        if return_bmat:
            return matrix, rhs
        else:
            return sps.bmat(matrix, matrix_format), np.concatenate(tuple(rhs))

    # ------------------------------------------------------------------------------#

    def split(self, gb, key, values, mortar_key="mortar_solution"):
        """
        Store in the grid bucket the vector, split in the function, solution of
        the problem. The values are extracted from the global solution vector
        according to the numeration given by "node_number".

        Parameters
        ----------
        gb : grid bucket with geometry fields computed.
        key: new name of the solution to be stored in the nodes of the grid
            bucket.
        values: array, global solution.
        mortar_key: (optional) new name of the mortar solution to be stored in
            the edges of the grid bucket

        """
        dofs = self._dof_start_of_grids(gb)

        gb.add_node_props(key)
        for g, d in gb:
            i = d["node_number"]
            d[key] = values[slice(dofs[i], dofs[i + 1])]

        gb.add_edge_props(mortar_key)
        for e, d in gb.edges():
            for j in range(self.num_mortars):
                i = d["edge_number"] + gb.num_graph_nodes() + gb.num_graph_edges() * j
                d[mortar_key] = values[slice(dofs[i], dofs[i + 1])]

    # ------------------------------------------------------------------------------#

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
            i = d["node_number"]
            values[slice(dofs[i], dofs[i + 1])] = d[key]

        return values

    # ------------------------------------------------------------------------------#

    def _dof_start_of_grids(self, gb):
        " Helper method to get first global dof for all grids. "
        self.ndof(gb)
        size = gb.num_graph_nodes() + self.num_mortars * gb.num_graph_edges()
        dofs = np.zeros(size, dtype=int)

        for _, d in gb:
            dofs[d["node_number"]] = d["dof"]

        for j in range(self.num_mortars):
            for e, d in gb.edges():
                i = d["edge_number"] + gb.num_graph_nodes() + gb.num_graph_edges() * j
                dofs[i] = d["dof"][j]

        return np.r_[0, np.cumsum(dofs)]

    # ------------------------------------------------------------------------------#

    def dof_of_grid(self, gb, g):
        """ Obtain global indices of dof associated with a given grid.

        Parameters:
            gb: Grid_bucket representation of mixed-dimensional data.
            g: Grid, one member of gb.

        Returns:
            np.array of ints: Indices of all dof for the given grid

        """
        dof_list = self._dof_start_of_grids(gb)
        nn = gb.node_props(g)["node_number"]
        return np.arange(dof_list[nn], dof_list[nn + 1])


# ------------------------------------------------------------------------------#
