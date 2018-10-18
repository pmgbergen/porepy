#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import scipy.sparse as sps
import collections

import porepy as pp


class Assembler(pp.numerics.mixed_dim.AbstractAssembler):
    """ A class that assembles a mixed-dimensional elliptic equation.
    """

    def __init__(self):
        pass

    def discretization_key(self, row, col=None):
        if col is None or row == col:
            return row + "_" + pp.keywords.DISCRETIZATION
        else:
            return row + "_" + col + "_" + pp.keywords.DISCRETIZATION


    def assemble_matrix_rhs(self, gb, matrix_format='csr', variables=None):

        # Initialize the global matrix. In this case, we know there is a single
        # variable (or two, depending on how we end up interpreting the mixed
        # methods) for each node and edge. This concept must be made more
        # general quite soon.
        matrix, rhs, block_dof = self._initialize_matrix_rhs(gb, variables)


        # Loop over all grids, discretize (if necessary) and assemble. This
        # will populate the main diagonal of the equation.
        for g, data in gb:
            loc_var = self._local_variables(data, variables)
            for row in loc_var.keys():
                for col in loc_var.keys():

                    ri = block_dof[(g, row)]
                    ci = block_dof[(g, col)]

                    discr = data.get(self.discretization_key(row, col), None)

                    if discr is None:
                        continue
                    else:
                        # Loop over all discretizations
                        for d in self._iterable(discr):
                            # Assemble the matrix and right hand side. This will also
                            # discretize if not done before.
                            loc_A, loc_b = d.assemble_matrix_rhs(g, data)

                            # Assign values in global matrix
                            matrix[ri, ci] += loc_A
                            rhs[ri] += loc_b

        # Loop over all edges
        for e, data_edge in gb.edges():

            # Grids and data dictionaries for master and slave
            g_slave, g_master = gb.nodes_of_edge(e)
            data_slave = gb.node_props(g_slave)
            data_master = gb.node_props(g_master)

            mg = data_edge["mortar_grid"]

            # Extract the local variables for edge and neighboring nodes
            active_edge_var = self._local_variables(data_edge, variables)
            active_master_var = self._local_variables(data_master, variables)
            active_slave_var = self._local_variables(data_slave, variables)

            # First discretize interaction between edge variables locally.
            # This is in direct analogue with the corresponding operation on
            # nodes.
            for row in active_edge_var.keys():
                for col in active_edge_var.keys():
                    ri = block_dof[(mg, row)]
                    ci = block_dof[(mg, col)]

                    discr = data_edge.get(self.discretization_key(row, col), None)

                    if discr is None:
                        continue
                    else:
                        # Loop over all discretizations
                        for d in self._iterable(discr):
                            # Assemble the matrix and right hand side. This will also
                            # discretize if not done before.
                            loc_A, loc_b = d.assemble_matrix_rhs(mg, data_edge)

                            # Assign values in global matrix
                            matrix[ri, ci] += loc_A
                            rhs[ri] += loc_b

            # Then, discretize the interaction between the edge variables of
            # this edge, and the adjacent node variables.

            # First, loop over all active edge variables
            for edge_key in active_edge_var.keys():
                ei = block_dof[(mg, edge_key)]
                # Loop over all decleared dependencies of this variable

                couplings = active_edge_var[edge_key].get("node_couplings", None)
                if couplings is None:
                    continue
                if not isinstance(couplings, list):
                    couplings = [couplings]
                for dep in couplings:
                    # Get the dependencies of each
                    master_key = dep.get(g_master, None)
                    slave_key = dep.get(g_slave, None)

                    if master_key is None and slave_key is None:
                        raise ValueError('A coupling need at least one dependency')

                    # If either the master or slave dependency is not among the
                    # active variables, we skip this coupling.
                    if (master_key is not None and not master_key in active_master_var.keys()) or \
                        (slave_key is not None and not slave_key in active_slave_var.keys()):
                            continue

                    mi = block_dof.get((g_master, master_key), None)
                    si = block_dof.get((g_slave, slave_key), None)

                    if mi is not None and si is not None:
                        idx = np.ix_([mi, si, ei], [mi, si, ei])
                        for discr in self._iterable(dep[pp.keywords.DISCRETIZATION]):
                            matrix[idx], loc_rhs = discr.assemble_matrix_rhs(g_master, g_slave, data_master, data_slave, data_edge, matrix[idx])
                            rhs[[mi, si, ei]] += loc_rhs

                    elif mi is not None:
                        idx = np.ix_([mi, ei])
                        for discr in self._iterable(dep[pp.keywords.DISCRETIZATION]):
                            matrix[idx], loc_rhs = discr.assemble_matrix_rhs(g_master, data_master, data_edge, matrix[idx])
                            rhs[[mi, ei]] += loc_rhs

                    elif si is not None:
                        idx = np.ix_([si, ei])
                        for discr in self._iterable(dep[pp.keywords.DISCRETIZATION]):
                            matrix[idx], loc_rhs = discr.assemble_matrix_rhs(g_slave, data_slave, data_edge, matrix[idx])
                            rhs[[si, ei]] += loc_rhs


        return sps.bmat(matrix, matrix_format), np.concatenate(tuple(rhs))


    def _initialize_matrix_rhs(self, gb, variables=None):
        """
        Parameter:
            gb: grid bucket.

        Return:
            matrix: the block global matrix.
            rhs: the global right-hand side.
        """

        block_dof_counter = 0
        block_dof = {}
        full_dof = []
        for g, d in gb:
            for k, v in self._local_variables(d, variables).items():
                block_dof[(g, k)] = block_dof_counter
                loc_dof = g.num_cells * v.get('cells', 0) + \
                          g.num_faces * v.get('faces', 0) + \
                          g.num_nodes * v.get('nodes', 0)
                full_dof.append(loc_dof)
                block_dof_counter += 1

        for e, d in gb.edges():
            mg = d['mortar_grid']
            for k, v in self._local_variables(d, variables).items():
                block_dof[(mg, k)] = block_dof_counter
                # We only allow for cell variables on the mortar grid.
                # This will not change in the forseable future
                loc_dof = mg.num_cells * v.get('cells', 0)
                full_dof.append(loc_dof)
                block_dof_counter += 1

        num_blocks = block_dof_counter

        matrix = np.empty((num_blocks, num_blocks), dtype=np.object)
        rhs = np.empty(num_blocks, dtype=np.object)

        for ri in range(num_blocks):
            rhs[ri] = np.zeros(full_dof[ri])
            for ci in range(num_blocks):
                matrix[ri, ci] = sps.coo_matrix((full_dof[ri], full_dof[ci]))

        return matrix, rhs, block_dof

    def assemble_operator(self, gb, operator_name):
        """ @RUNAR: Placeholder method for use for non-linear problems
        """
        pass


    def _local_variables(self, d, variables):
        # Short-hand function to check if a limited number of admissible
        # variables is given, and if so, if a certain value is contained in
        # the set
        loc_variables = d.get(pp.keywords.PRIMARY_VARIABLES, None)
        if variables is None:
            return loc_variables
        else:
            if not isinstance(variables, list):
                variables = [variables]
            var = {}
            for key, val in loc_variables.items():
                if key in variables:
                    var[key] = val
            return var

    def _iterable(self, x):
        if isinstance(x, collections.Iterable):
            return x
        else:
            return (x,)

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