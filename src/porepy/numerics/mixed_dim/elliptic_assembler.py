#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import scipy.sparse as sps

import porepy as pp


class EllipticAssembler(pp.numerics.mixed_dim.abstract_assembler.AbstractAssembler):
    """ A class that assembles a mixed-dimensional elliptic equation.
    """

    def __init__(self, keyword):
        # The keyword should be the same as for all discretization objects
        self.keyword = keyword

    def key(self):
        return self.keyword + "_"

    def discretization_key(self):
        return self.key() + pp.keywords.DISCRETIZATION

    def assemble_matrix_rhs(self, gb, matrix_format='csr'):


        # Initialize the global matrix. In this case, we know there is a single
        # variable (or two, depending on how we end up interpreting the mixed
        # methods) for each node and edge. This concept must be made more
        # general quite soon.
        matrix, rhs = self.initialize_matrix_rhs(gb)


        # Loop over all grids, discretize (if necessary) and assemble. This
        # will populate the main diagonal of the equation.
        for g, data in gb:

            pos = data['node_number']

            discr = data[self.discretization_key()]

            # Assemble the matrix and right hand side. This will also
            # discretize if not done before.
            loc_A, loc_b = discr.assemble_matrix_rhs(g, data)

            # Assign values in global matrix
            matrix[pos, pos] = loc_A
            rhs[pos] = loc_b

        num_nodes = gb.num_graph_nodes()

        # Loop over all edges
        for e, data_edge in gb.edges():
            discr = data_edge[self.discretization_key()]
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

            loc_matrix = matrix[idx]

            matrix[idx] = discr.assemble_matrix(g_master, g_slave, data_master, data_slave, data_edge, loc_matrix)

        return sps.bmat(matrix, matrix_format), np.concatenate(tuple(rhs))


    def extract_flux(self, gb, pressure_flux_keyword, flux_keyword):
        """ Extract the flux variable from a solution of the elliptic equation.

        This function should be called after self.split()

        @ALL: This I believe replaces the old compute_discharge function for
        mpfa and tpfa, if that is still alive.

        @ALL: I don't like the word extract here - it makes sense for mixed
        formulations, but not so for finite volumes, where this is a post
        processing step that computes the flux. Other possible names are
        'isolate_flux', ?. Opinions?

        @ALL: I am not convinced about the word flux in the function name either,
        but this is not that bad: We would talk about a heat flux in heat transport.

        @ALL: Better names for the last two parameters are required.

        @ALL: With a general variable definition in place, this function can
            be pulled out to a generic assembler.

        Parameters:
            gb: GridBucket, mixed-dimensional grid.
            pressure_flux_keyword (str): Keyword used to identify the solution
                field distribtued in the GridBucket data on each node, e.g.
                the same keyword as given to self.split()
            flux_keyword (str): Keyword to be used to identify the flux field
                in the data structure.

        """
        gb.add_node_props([flux_keyword])
        for g, d in gb:
            discretization = d[self.discretization_key()]
            d[flux_keyword] = discretization.extract_flux(g, d[pressure_flux_keyword], d)


    def extract_pressure(self, gb, presssure_flux_keyword, pressure_keyword):
        """ Extract the pressure variable from a solution of the elliptic equation.

        This function should be called after self.split()

        @ALL: This I believe replaces the old compute_discharge function for
        mpfa and tpfa, if that is still alive.

        @ALL: I don't like the word extract here - it makes sense for mixed
        formulations, but not so for finite volumes, where this is a post
        processing step that computes the flux. Other possible names are
        'isolate_pressure', ?. Opinions?

        @ALL: I am not at all convinced about the word pressure in the function name either.
        Potential is a tempting name, but that carries a specific, and potentially
        misleading, meaning for flow problems with gravity.

        @ALL: Better names for the last two parameters are required.

        @ALL: With a general variable definition in place, this function can
            be pulled out to a generic assembler.

        Parameters:
            gb: GridBucket, mixed-dimensional grid.
            pressure_flux_keyword (str): Keyword used to identify the solution
                field distribtued in the GridBucket data on each node, e.g.
                the same keyword as given to self.split()
            flux_keyword (str): Keyword to be used to identify the flux field
                in the data structure.

        """
        gb.add_node_props([pressure_keyword])
        for g, d in gb:
            discretization = d[self.discretization_key()]
            d[pressure_keyword] = discretization.extract_pressure(g, d[presssure_flux_keyword], d)