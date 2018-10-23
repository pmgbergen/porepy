#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:33:14 2018

@author: eke001
"""
import numpy as np
import scipy.sparse as sps


class AbstractAssembler(object):

    def __init__(self):
        pass

    def initialize_matrix_rhs(self, gb):
        """
        @ALL: coupied from Coupler. Sholud be revised; ultimately, this should
        be done by a general method that only considers the number and types of
        variables. Of course, it should also be moved somewhere else.

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
        size = num_nodes + num_edges
        shapes = np.empty((size, size), dtype=np.object)

        dof = self._key() + "dof"
        # First loop over all nodes and edges. Assign dof numbers if not already
        # present
        for g, d in gb:
            if not dof in d.keys():
                d[dof] = d[self._discretization_key()].ndof(g)

        for e, d in gb.edges():
            if not dof in d.keys():
                mg = d["mortar_grid"]
                d[dof] = d[self._discretization_key()].ndof(mg)

        # Initialize the shapes for the matrices and rhs for all the sub-blocks
        for _, d_i in gb:
            pos_i = d_i["node_number"]
            for _, d_j in gb:
                pos_j = d_j["node_number"]
                shapes[pos_i, pos_j] = (d_i[dof], d_j[dof])
            for _, d_e in gb.edges():
                pos_e = d_e["edge_number"] + num_nodes
                shapes[pos_i, pos_e] = (d_i[dof], d_e[dof])
                shapes[pos_e, pos_i] = (d_e[dof], d_i[dof])

        for _, d_e in gb.edges():
            pos_e = d_e["edge_number"] + num_nodes
            dof_e = d_e[dof]
            for _, d_f in gb.edges():
                pos_f = d_f["edge_number"] + num_nodes
                shapes[pos_e, pos_f] = (dof_e, d_f[dof])

        # initialize the matrix and rhs
        matrix = np.empty(shapes.shape, dtype=np.object)
        rhs = np.empty(shapes.shape[0], dtype=np.object)

        for i in np.arange(shapes.shape[0]):
            rhs[i] = np.zeros(shapes[i, i][0])
            for j in np.arange(shapes.shape[1]):
                matrix[i, j] = sps.coo_matrix(shapes[i, j])

        return matrix, rhs

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
            i = d["edge_number"] + gb.num_graph_nodes()
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
#        self.ndof(gb)
        size = gb.num_graph_nodes() + gb.num_graph_edges()
        dofs = np.zeros(size, dtype=int)

        for _, d in gb:
            dofs[d["node_number"]] = d[self._key() + "dof"]

        for e, d in gb.edges():
            i = d["edge_number"] + gb.num_graph_nodes()
            dofs[i] = d[self._key() + "dof"]

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

