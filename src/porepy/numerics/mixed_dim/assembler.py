#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import scipy.sparse as sps
import collections

import porepy as pp


class Assembler(pp.numerics.mixed_dim.AbstractAssembler):
    """ A class that assembles multi-physics problems on mixed-dimensional
    domains.

    The class is designed to combine different variables on different grids,
    different discretizations for the same variable, various coupling schemes
    between the grids etc. To use the functionality, discretization schemes
    for the individual terms in the equation must be defined and follow certain
    rules. For further description, see the documentation of self.assemble_matrix_rhs().

    """

    def __init__(self):
        pass

    def discretization_key(self, row, col=None):
        if col is None or row == col:
            return row
        else:
            return row + "_" + col

    def _variable_term_key(self, term, key_1, key_2, key_3=None):
        if key_3 is None:
            # Internal to a node or an edge
            if key_1 == key_2:
                return "_".join([term, key_1])
            else:
                return "_".join([term, key_1, key_2])
        else:
            # Coupling between edge and node
            return "_".join([term, key_1, key_2, key_3])

    def assemble_matrix_rhs(
        self, gb, matrix_format="csr", variables=None, add_matrices=True
    ):
        """ Assemble the system matrix and right hand side for a general
        multi-physics problem, and return a block matrix and right hand side.

        Variables can be defined on nodes in the GridBucket (grids in a certain
        dimension) or on the edge between nodes (e.g. on the mortar grid). Node
        variables can be defined on a combination of cells, faces and
        grid-nodes, while edge variables live only on the cells of the mortar
        grid. It is not possible to define a variable only on a subset of these
        objects (say, only on some of the cells in the grid), but it is allowed
        to have a cell variable living on one node in the GridBucket, but not
        in another.

        Variables for a node or edge, together with information on how many
        degrees of freedom they define, are defined on the relevant data
        dictionary, with the following syntax:

            d[pp.keywords.PRIMARY_VARIABLE] = {'var_1': {'cells': 3},
                                               'var_2': {'cells': 1, 'faces': 2}}

        This defines a variable identified by the string 'var_1' as living on
        this object (node or edge), having 3 degrees of freedom per cell in the
        corresponding grid (and tacitly no face or node variables). Similarly,
        'var_2' identifies a variable with one degree of freedom per cell and
        two per face.

        To define a discretization for a variable, the data dictionary should
        contain one or a list of discretization objects that can be accessed by
        the call

            d['var_1' + '_' + pp.keywords.DISCRETIZATION]

        The object should have a method termed assemble_matrix_rhs, that takes
        arguments grid (on a node) or a mortar grid (on an edge) and the
        corresponding data dictionary, and returns the system matrix and right
        hand side for this variable on this node / edge. If several
        discretization objcets are provided as a list, their values will be
        added.

        Coupling between variables on the same node are similarly defined by
        one or a list of discretizations, identified by the fields

            d['var_1' + '_' + 'var_2' + '_' + pp.keywords.DISCRETIZATION]
            d['var_2' + '_' + 'var_1' + '_' + pp.keywords.DISCRETIZATION]

        Here, the first item represents the impact of var_2 on var_1, stored in
        block (var_1, var_2); the definition of the second term is similar.

        Discretization of edge-node couplings are similar to internal interactions
        on nodes and edges, but more general, and thus with a more complex
        syntax. The variable definition on mortar variables has, in addition to
        fields for degrees of freedom, a dictionary of dependencies on
        variables on the neighboring nodes (or list of dictionaries if there is
        more than one dependency). This inner dictionary has fields

            {g1: 'var_1', g2: 'var_2', pp.keywords.DISCRETIZATION: foo()}

        Here, 'var_1' (str) is the identifier of the coupled variable on the
        node identified by the neighboring grid g1, similar for g2. foo() is a
        discretization object for the coupling term. It should have a method
        assemble_matrix_rhs, that takes the arguments

            (g1, g2, data_1, data_2, data_edge, local_matrix)

        where local_matrix is the current (partly assembled) discretization
        matrix for the coupling of these terms. A coupling between a mortar
        variable and only one of its neighboring grids can be constructed by
        letting the inactive grid map to None, e.g.

            {g1: 'var_1', g2: None, pp.keywords.DISCRETIZATION: foo()}

        In this case, foo should have a method assemble_matrix_rhs that takes
        arguments

            (g1, data_1, data_edge, local_matrix)

        """

        # Initialize the global matrix.
        matrix, rhs, block_dof, full_dof = self._initialize_matrix_rhs(gb, variables)

        if len(full_dof) == 0:
            if add_matrices:
                mat, vec = self._assign_matrix_vector(full_dof)
                return mat, vec, block_dof
            else:
                return matrix, rhs, block_dof

        # Loop over all grids, discretize (if necessary) and assemble. This
        # will populate the main diagonal of the equation.
        for g, data in gb:
            loc_var = self._local_variables(data, variables)
            for row in loc_var.keys():
                for col in loc_var.keys():

                    ri = block_dof[(g, row)]
                    ci = block_dof[(g, col)]

                    discr_data = data.get(pp.keywords.DISCRETIZATION, None)
                    if discr_data is None:
                        continue
                    discr = discr_data.get(self.discretization_key(row, col), None)

                    if discr is None:
                        continue
                    else:
                        # Loop over all discretizations
                        for term, d in discr.items():
                            # Assemble the matrix and right hand side. This will also
                            # discretize if not done before.
                            loc_A, loc_b = d.assemble_matrix_rhs(g, data)

                            # Assign values in global matrix
                            var_key_name = self._variable_term_key(term, row, col)
                            matrix[var_key_name][ri, ci] += loc_A
                            rhs[var_key_name][ri] += loc_b

        # Loop over all edges
        for e, data_edge in gb.edges():

            # Grids and data dictionaries for master and slave
            g_slave, g_master = gb.nodes_of_edge(e)
            data_slave = gb.node_props(g_slave)
            data_master = gb.node_props(g_master)

            mg = data_edge["mortar_grid"]

            # Extract the local variables for edge and neighboring nodes
            active_edge_var = self._local_variables(data_edge, variables)

            # First discretize interaction between edge variables locally.
            # This is in direct analogue with the corresponding operation on
            # nodes.
            for row in active_edge_var.keys():
                for col in active_edge_var.keys():
                    ri = block_dof[(mg, row)]
                    ci = block_dof[(mg, col)]

                    discr_data = data_edge.get(pp.keywords.DISCRETIZATION)
                    if discr_data is None:
                        continue
                    discr = discr_data.get(self.discretization_key(row, col), None)

                    if discr is None:
                        continue
                    else:
                        # Loop over all discretizations
                        for term, d in discr.items():
                            # Assemble the matrix and right hand side. This will also
                            # discretize if not done before.
                            loc_A, loc_b = d.assemble_matrix_rhs(g, data_edge)

                            # Assign values in global matrix
                            var_key_name = self._variable_term_key(term, row, col)
                            matrix[var_key_name][ri, ci] += loc_A
                            rhs[var_key_name][ri] += loc_b

            # Then, discretize the interaction between the edge variables of
            # this edge, and the adjacent node variables.
            discr = data_edge.get(pp.keywords.COUPLING_DISCRETIZATION, None)
            if discr is None:
                continue

            for key, terms in discr.items():
                edge_vals = terms.get(e)
                edge_key = edge_vals[0]
                ei = block_dof[(mg, edge_key)]

                master_vals = terms.get(g_master)
                if master_vals is None:
                    master_key = ""
                    mi = None
                else:
                    master_key = master_vals[0]
                    mi = block_dof.get((g_master, master_key))

                slave_vals = terms.get(g_slave)
                if slave_vals is None:
                    slave_key = ""
                    si = None
                else:
                    slave_key = slave_vals[0]
                    si = block_dof.get((g_slave, slave_key))

                mat_key = self._variable_term_key(key, edge_key, slave_key, master_key)

                e_discr = edge_vals[1]
                if mi is not None and si is not None:
                    idx = np.ix_([mi, si, ei], [mi, si, ei])

                    matrix[mat_key][idx], loc_rhs = e_discr.assemble_matrix_rhs(
                        g_master,
                        g_slave,
                        data_master,
                        data_slave,
                        data_edge,
                        matrix[mat_key][idx],
                    )
                    rhs[mat_key][[mi, si, ei]] += loc_rhs

                elif mi is not None:
                    idx = np.ix_([mi, ei], [mi, ei])
                    matrix[mat_key][idx], loc_rhs = e_discr.assemble_matrix_rhs(
                        g_master, data_master, data_edge, matrix[mat_key][idx]
                    )
                    rhs[mat_key][[mi, ei]] += loc_rhs

                elif si is not None:
                    idx = np.ix_([si, ei], [si, ei])
                    matrix[mat_key][idx], loc_rhs = e_discr.assemble_matrix_rhs(
                        g_slave, data_slave, data_edge, matrix[mat_key][idx]
                    )
                    rhs[mat_key][[si, ei]] += loc_rhs

                else:
                    raise ValueError(
                        "Invalid combination of variables on node-edge relation"
                    )

        if add_matrices:

            full_matrix, full_rhs = self._assign_matrix_vector(full_dof)
            for mat in matrix.values():
                full_matrix += mat
            for vec in rhs.values():
                full_rhs += vec

            return (
                sps.bmat(full_matrix, matrix_format),
                np.concatenate(tuple(full_rhs)),
                block_dof,
            )
        else:
            for k, v in matrix.items():
                matrix[k] = sps.bmat(v, matrix_format)
            for k, v in rhs.items():
                rhs[k] = np.concatenate(tuple(v))

            return matrix, rhs, block_dof

    def _initialize_matrix_rhs(self, gb, variables=None):
        """
        Initialize local matrices for all combinations of variables and operators.

        Parameter:
            gb: grid bucket.

        Return:
            matrix: the block global matrix.
            rhs: the global right-hand side.
        """

        block_dof_counter = 0
        block_dof = {}
        full_dof = []

        variable_combinations = []

        for g, d in gb:

            # Loop over variables, count dofs and identify variable-term combinations internal to the node
            for k, v in self._local_variables(d, variables).items():

                # First count the number of dofs per variable
                block_dof[(g, k)] = block_dof_counter
                loc_dof = (
                    g.num_cells * v.get("cells", 0)
                    + g.num_faces * v.get("faces", 0)
                    + g.num_nodes * v.get("nodes", 0)
                )
                full_dof.append(loc_dof)
                block_dof_counter += 1

                # Then identify all discretization terms for this variable
                for k2 in self._local_variables(d, variables).keys():
                    if k == k2:
                        key = k
                    else:
                        key = k + "_" + k2
                    discr = d.get(pp.keywords.DISCRETIZATION, None)
                    if discr is None:
                        continue
                    terms = discr.get(key, None)
                    if terms is None:
                        continue
                    for term in terms.keys():
                        variable_combinations.append(
                            self._variable_term_key(term, k, k2)
                        )

        for e, d in gb.edges():
            mg = d["mortar_grid"]

            # Loop over variables, count dofs and identify variable-term combinations internal to the edge
            for k, v in self._local_variables(d, variables).items():

                # First count the number of dofs per variable
                block_dof[(mg, k)] = block_dof_counter
                # We only allow for cell variables on the mortar grid.
                # This will not change in the forseable future
                loc_dof = mg.num_cells * v.get("cells", 0)
                full_dof.append(loc_dof)
                block_dof_counter += 1

                # Then identify all discretization terms for this variable
                for k2 in self._local_variables(d, variables).keys():
                    if k == k2:
                        key = k
                    else:
                        key = k + "_" + k2
                    discr = d.get(pp.keywords.DISCRETIZATION, None)
                    if discr is None:
                        continue
                    terms = discr.get(key, None)
                    if terms is None:
                        continue
                    for term in terms.keys():
                        variable_combinations.append(
                            self._variable_term_key(term, k, k2)
                        )

            # Finally, identify variable combinations for coupling terms
            g_slave, g_master = gb.nodes_of_edge(e)
            discr = d.get(pp.keywords.COUPLING_DISCRETIZATION, None)
            if discr is None:
                continue
            for term, val in discr.items():
                # Identify this term in the discretization by the variable names
                # on the edge, the variable names of the first and second grid
                # in that order, and finally the term name.
                # There is a tacit assumption here that gb.nodes_of_edge return the
                # grids in the same order here and in the assembly. This should be
                # okay. The consequences for the methods if this is no longer the case is unclear.
                k = val.get(e)[0]
                k2 = val.get(g_slave)
                if k2 is not None:
                    k2 = k2[0]
                else:
                    k2 = ""
                k3 = val.get(g_master)
                if k3 is not None:
                    k3 = k3[0]
                else:
                    k3 = ""
                variable_combinations.append(self._variable_term_key(term, k, k2, k3))

        # We will have one discretization matrix per variable
        matrix_dict = {}
        rhs_dict = {}

        # Uniquify list of variable combinations. Then iterate over all variable
        # combinations and initialize matrices of the right size
        for var in list(set(variable_combinations)):
            matrix, rhs = self._assign_matrix_vector(full_dof)
            matrix_dict[var] = matrix
            rhs_dict[var] = rhs

        return matrix_dict, rhs_dict, block_dof, full_dof

    def _assign_matrix_vector(self, dof):
        # Assign a block matrix and vector with specified number of dofs per block
        num_blocks = len(dof)
        matrix = np.empty((num_blocks, num_blocks), dtype=np.object)
        rhs = np.empty(num_blocks, dtype=np.object)

        for ri in range(num_blocks):
            rhs[ri] = np.zeros(dof[ri])
            for ci in range(num_blocks):
                matrix[ri, ci] = sps.coo_matrix((dof[ri], dof[ci]))

        return matrix, rhs

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
            d[flux_keyword] = discretization.extract_flux(
                g, d[pressure_flux_keyword], d
            )

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
            d[pressure_keyword] = discretization.extract_pressure(
                g, d[presssure_flux_keyword], d
            )
