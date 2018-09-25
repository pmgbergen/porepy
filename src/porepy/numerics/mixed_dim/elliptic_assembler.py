#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import scipy.sparse as sps

import porepy as pp


class EllipticAssembler(object):
    """ A class that assembles a mixed-dimensional elliptic equation.
    """

    def __init__(self, keyword):
        # The keyword should be the same as for all discretization objects
        self.keyword = keyword()

    def key(self):
        return self.keyword + "_"

    def assemble_matrix(self, gb):


        # Initialize the global matrix. In this case, we know there is a single
        # variable (or two, depending on how we end up interpreting the mixed
        # methods) for each node and edge. This concept must be made more
        # general quite soon.
        matrix, rhs = self.initialize_matrix_rhs(gb)


        # Loop over all grids, discretize (if necessary) and assemble. This
        # will populate the main diagonal of the equation.
        for g, data in gb:

            pos = data['node_number']

            discr = data[self.key() + "discr"]

            # Assemble the matrix and right hand side. This will also
            # discretize if not done before.
            loc_A, loc_b = discr.assemble_matrix_rhs(g, data)

            # Assign values in global matrix
            matrix[pos, pos] = loc_A
            rhs[pos] = loc_b


        num_nodes = gb.num_graph_nodes()

        # Loop over all edges
        for e, data_edge in gb.edges():
            discr = data[self.key() + "discr"]
            g_slave, g_master = gb.nodes_of_edge(e)
            data_slave = gb.node_props(g_slave)
            data_master = gb.node_props(g_master)

            pos_master = data_master["node_number"]
            pos_slave = data_slave["node_number"]

            # @RUNAR: I know this breaks with multiple mortar variables.
            # To be improved.
            pos_edge = data_edge["edge_number"] + num_nodes

            idx = np.ix_([pos_master, pos_slave, pos_edge],

                         [pos_master, pos_slave, pos_edge])
            matrix_master = matrix[pos_master, pos_master]
            matrix_slave = matrix[pos_slave, pos_slave]

            loc_A = discr.discretize(g_master, g_slave, data_master, data_slave, matrix_master, matrix_slave)

            matrix[idx] = loc_A

        return matrix, rhs


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
