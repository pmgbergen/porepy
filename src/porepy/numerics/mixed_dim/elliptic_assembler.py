#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import scipy.sparse as sps

import porepy as pp


class EllipticAssembler(pp.numerics.mixed_dim.AbstractAssembler):
    """ A class that assembles a mixed-dimensional elliptic equation.

    In terms of functionality, this is a specialized version of pp.Assembler(),
    dedicated to assembly of scalar elliptic equations. Compared to the general
    approach, this class has:
        1) A much simpler syntaxt for defining the variable, based on a single
           keyword.
        2) The number of degrees of freedom on each node is not specified, but
           instead inferred from the discretization applied. This is heavily
           motivated by experimentation with heterogeneous discretizations.

    For details on how to use the function, see the method assemble_matrix_rhs().

    Attributes:
        keyword (str): String identifying all quantities (discretization,
                parameters) this object will work on.

    """

    def __init__(self, keyword):
        """ Create an assembler object associated with a specific keyword that
        identifies its variable, discretization and parameters.

        Parameters:
            keyword (str): String used for identifying all quantities this
                object will work on. See assemble_matrix_rhs() for details.
        """
        self.keyword = keyword

    def _key(self):

        return self.keyword + "_"

    def _discretization_key(self):
        # Convenience method to get a string representation for whatever
        return self._key() + pp.DISCRETIZATION

    def assemble_matrix_rhs(self, gb, matrix_format="csr"):
        """ Assemble the system matrix and right hand side  for the elliptic
        equation.

        The function loops over all nodes in the GridBucket and looks for

        """

        # Initialize the global matrix. In this case, we know there is a single
        # variable (or two, depending on how we end up interpreting the mixed
        # methods) for each node and edge. This concept must be made more
        # general quite soon.
        matrix, rhs = self.initialize_matrix_rhs(gb)

        # Loop over all grids, discretize (if necessary) and assemble. This
        # will populate the main diagonal of the equation.
        for g, data in gb:

            # The structure of the GridBucked nodes in the system matrix is
            # based on the keyword node_number.
            # TODO: We should rather use a
            pos = data["node_number"]

            discr = data[self._discretization_key()]

            # Assemble the matrix and right hand side. This will also
            # discretize if not done before.
            loc_A, loc_b = discr.assemble_matrix_rhs(g, data)

            # Assign values in global matrix
            matrix[pos, pos] = loc_A
            rhs[pos] = loc_b

        num_nodes = gb.num_graph_nodes()

        # Loop over all edges
        for e, data_edge in gb.edges():
            discr = data_edge[self._discretization_key()]
            g_slave, g_master = gb.nodes_of_edge(e)
            data_slave = gb.node_props(g_slave)
            data_master = gb.node_props(g_master)

            pos_master = data_master["node_number"]
            pos_slave = data_slave["node_number"]

            # @RUNAR: I know this breaks with multiple mortar variables.
            # To be improved.
            pos_edge = data_edge["edge_number"] + num_nodes

            idx = np.ix_(
                [pos_master, pos_slave, pos_edge], [pos_master, pos_slave, pos_edge]
            )

            rhs_idx = [[pos_master, pos_slave, pos_edge]]
            loc_matrix = matrix[idx]
            matrix[idx], loc_rhs = discr.assemble_matrix_rhs(
                g_master, g_slave, data_master, data_slave, data_edge, loc_matrix
            )
            rhs[rhs_idx] += loc_rhs

        return sps.bmat(matrix, matrix_format), np.concatenate(tuple(rhs))

    def extract_flux(self, gb, pressure_flux_keyword, flux_keyword):
        """ Extract the flux variable from a solution of the elliptic equation.

        This function should be called after self.split()

        @ALL: This I believe replaces the old compute_darcy_flux function for
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
            discretization = d[self._discretization_key()]
            d[flux_keyword] = discretization.extract_flux(
                g, d[pressure_flux_keyword], d
            )

    def extract_pressure(self, gb, presssure_flux_keyword, pressure_keyword):
        """ Extract the pressure variable from a solution of the elliptic equation.

        This function should be called after self.split()

        @ALL: This I believe replaces the old compute_darcy_flux function for
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
            discretization = d[self._discretization_key()]
            d[pressure_keyword] = discretization.extract_pressure(
                g, d[presssure_flux_keyword], d
            )
