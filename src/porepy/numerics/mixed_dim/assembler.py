#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import scipy.sparse as sps

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

            d[pp.PRIMARY_VARIABLE] = {'var_1': {'cells': 3},
                                               'var_2': {'cells': 1, 'faces': 2}}

        This defines a variable identified by the string 'var_1' as living on
        this object (node or edge), having 3 degrees of freedom per cell in the
        corresponding grid (and tacitly no face or node variables). Similarly,
        'var_2' identifies a variable with one degree of freedom per cell and
        two per face.

        To define a discretization for a variable, the data dictionary should
        contain one or a list of discretization objects that can be accessed by
        the call

            d['var_1' + '_' + pp.DISCRETIZATION]

        The object should have a method termed assemble_matrix_rhs, that takes
        arguments grid (on a node) or a mortar grid (on an edge) and the
        corresponding data dictionary, and returns the system matrix and right
        hand side for this variable on this node / edge. If several
        discretization objcets are provided as a list, their values will be
        added.

        Coupling between variables on the same node are similarly defined by
        one or a list of discretizations, identified by the fields

            d['var_1' + '_' + 'var_2' + '_' + pp.DISCRETIZATION]
            d['var_2' + '_' + 'var_1' + '_' + pp.DISCRETIZATION]

        Here, the first item represents the impact of var_2 on var_1, stored in
        block (var_1, var_2); the definition of the second term is similar.

        Discretization of edge-node couplings are similar to internal interactions
        on nodes and edges, but more general, and thus with a more complex
        syntax. The variable definition on mortar variables has, in addition to
        fields for degrees of freedom, a dictionary of dependencies on
        variables on the neighboring nodes (or list of dictionaries if there is
        more than one dependency). This inner dictionary has fields

            {g1: 'var_1', g2: 'var_2', pp.DISCRETIZATION: foo()}

        Here, 'var_1' (str) is the identifier of the coupled variable on the
        node identified by the neighboring grid g1, similar for g2. foo() is a
        discretization object for the coupling term. It should have a method
        assemble_matrix_rhs, that takes the arguments

            (g1, g2, data_1, data_2, data_edge, local_matrix)

        where local_matrix is the current (partly assembled) discretization
        matrix for the coupling of these terms. A coupling between a mortar
        variable and only one of its neighboring grids can be constructed by
        letting the inactive grid map to None, e.g.

            {g1: 'var_1', g2: None, pp.DISCRETIZATION: foo()}

        In this case, foo should have a method assemble_matrix_rhs that takes
        arguments

            (g1, data_1, data_edge, local_matrix)

        """
        # Define the matrix format, common for all the sub-matrices
        if matrix_format == "csc":
            sps_matrix = sps.csc_matrix
        else:
            sps_matrix = sps.csr_matrix

        # Initialize the global matrix.
        matrix, rhs, block_dof, full_dof = self._initialize_matrix_rhs(gb, variables, sps_matrix)
        if len(full_dof) == 0:
            if add_matrices:
                mat, vec = self._assign_matrix_vector(full_dof, sps_matrix)
                return mat, vec, block_dof, full_dof
            else:
                return matrix, rhs, block_dof, full_dof

        # Loop over all grids, discretize (if necessary) and assemble. This
        # will populate the main diagonal of the equation.
        for g, data in gb:
            loc_var = self._local_variables(data, variables)
            for row in loc_var.keys():
                for col in loc_var.keys():

                    ri = block_dof[(g, row)]
                    ci = block_dof[(g, col)]

                    discr_data = data.get(pp.DISCRETIZATION, None)
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
                            # Check if the current block is None or not, it could
                            # happend based on the problem setting. Better to stay
                            # on the safe side.
                            if matrix[var_key_name][ri, ci] is None:
                                matrix[var_key_name][ri, ci] = loc_A
                            else:
                                matrix[var_key_name][ri, ci] += loc_A
                            rhs[var_key_name][ri] += loc_b

        # Loop over all edges
        for e, data_edge in gb.edges():

            # Grids and data dictionaries for master and slave
            g_slave, g_master = gb.nodes_of_edge(e)
            data_slave = gb.node_props(g_slave)
            data_master = gb.node_props(g_master)

            # Extract the local variables for edge and neighboring nodes
            active_edge_var = self._local_variables(data_edge, variables)

            # First discretize interaction between edge variables locally.
            # This is in direct analogue with the corresponding operation on
            # nodes.
            for row in active_edge_var.keys():
                for col in active_edge_var.keys():
                    ri = block_dof[(e, row)]
                    ci = block_dof[(e, col)]

                    discr_data = data_edge.get(pp.DISCRETIZATION)
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
                            # Check if the current block is None or not, it could
                            # happend based on the problem setting. Better to stay
                            # on the safe side.
                            if matrix[var_key_name][ri, ci] is None:
                                matrix[var_key_name][ri, ci] = loc_A
                            else:
                                matrix[var_key_name][ri, ci] += loc_A
                            rhs[var_key_name][ri] += loc_b

            # Then, discretize the interaction between the edge variables of
            # this edge, and the adjacent node variables.
            discr = data_edge.get(pp.COUPLING_DISCRETIZATION, None)
            if discr is None:
                continue

            for key, terms in discr.items():
                edge_vals = terms.get(e)
                edge_key = edge_vals[0]
                ei = block_dof[(e, edge_key)]

                master_vals = terms.get(g_master)
                if master_vals is None:
                    master_key = ""
                    mi = None
                else:
                    master_key = master_vals[0]
                    mi = block_dof.get((g_master, master_key))

                    # Also define the key to access the matrix of the discretization of
                    # the master variable on the master node.
                    mat_key_master = self._variable_term_key(
                        master_vals[1], master_key, master_key
                    )

                slave_vals = terms.get(g_slave)
                if slave_vals is None:
                    slave_key = ""
                    si = None
                else:
                    slave_key = slave_vals[0]
                    si = block_dof.get((g_slave, slave_key))
                    # Also define the key to access the matrix of the discretization of
                    # the slave variable on the slave node.
                    mat_key_slave = self._variable_term_key(
                        slave_vals[1], slave_key, slave_key
                    )

                # Key to the matrix dictionary used to access this coupling
                # discretization.
                mat_key = self._variable_term_key(key, edge_key, slave_key, master_key)

                e_discr = edge_vals[1]

                if mi is not None and si is not None:

                    # Assign a local matrix, which will be populated with the
                    # current state of the local system.
                    # Local here refers to the variable and term on the two
                    # nodes, together with the relavant mortar variable and term

                    # Associate the first variable with master, the second with
                    # slave, and the final with edge.
                    loc_mat, _ = self._assign_matrix_vector(full_dof[[mi, si, ei]], sps_matrix)

                    # Pick out the discretizations on the master and slave node
                    # for the relevant variables.
                    # There should be no contribution or modification of the
                    # [0, 1] and [1, 0] terms, since the variables are only
                    # allowed to communicate via the edges.
                    loc_mat[0, 0] = matrix[mat_key_master][mi, mi]
                    loc_mat[1, 1] = matrix[mat_key_slave][si, si]

                    # Run the discretization, and assign the resulting matrix
                    # to a temporary construct
                    tmp_mat, loc_rhs = e_discr.assemble_matrix_rhs(
                        g_master, g_slave, data_master, data_slave, data_edge, loc_mat
                    )
                    # The edge column and row should be assigned to mat_key
                    matrix[mat_key][(ei), (mi, si, ei)] = tmp_mat[(2), (0, 1, 2)]
                    matrix[mat_key][(mi, si), (ei)] = tmp_mat[(0, 1), (2)]

                    # Also update the discretization on the master and slave
                    # nodes
                    matrix[mat_key_master][mi, mi] = tmp_mat[0, 0]
                    matrix[mat_key_slave][si, si] = tmp_mat[1, 1]

                    # Finally take care of the right hand side
                    rhs[mat_key][[mi, si, ei]] += loc_rhs

                elif mi is not None:
                    # si is None
                    loc_mat, _ = self._assign_matrix_vector(full_dof[[mi, ei]], sps_matrix)
                    loc_mat[0, 0] = matrix[mat_key_master][mi, mi]
                    tmp_mat, loc_rhs = e_discr.assemble_matrix_rhs(
                        g_master, data_master, data_edge, loc_mat
                    )
                    matrix[mat_key][(ei), (mi, ei)] = tmp_mat[(1), (0, 1)]
                    matrix[mat_key][mi, ei] = tmp_mat[0, 1]

                    # Also update the discretization on the master and slave
                    # nodes
                    matrix[mat_key_master][mi, mi] = tmp_mat[0, 0]

                    rhs[mat_key][[mi, ei]] += loc_rhs

                elif si is not None:
                    # mi is None
                    loc_mat, _ = self._assign_matrix_vector(full_dof[[si, ei]], sps_matrix)
                    loc_mat[0, 0] = matrix[mat_key_slave][si, si]
                    tmp_mat, loc_rhs = e_discr.assemble_matrix_rhs(
                        g_slave, data_slave, data_edge, loc_mat
                    )
                    matrix[mat_key][ei, (si, ei)] = tmp_mat[1, (0, 1)]
                    matrix[mat_key][si, ei] = tmp_mat[0, 1]

                    # Also update the discretization on the master and slave
                    # nodes
                    matrix[mat_key_slave][si, si] = tmp_mat[0, 0]

                    rhs[mat_key][[si, ei]] += loc_rhs

                else:
                    raise ValueError(
                        "Invalid combination of variables on node-edge relation"
                    )

        if add_matrices:
            size = np.sum(full_dof)
            full_matrix = sps_matrix((size, size))
            full_rhs = np.zeros(size)

            for mat in matrix.values():
                full_matrix += sps.bmat(mat, matrix_format)

            for vec in rhs.values():
                full_rhs += np.concatenate(tuple(vec))

            return full_matrix, full_rhs, block_dof, full_dof

        else:
            for k, v in matrix.items():
                matrix[k] = sps.bmat(v, matrix_format)
            for k, v in rhs.items():
                rhs[k] = np.concatenate(tuple(v))

            return matrix, rhs, block_dof, full_dof

    def _initialize_matrix_rhs(self, gb, variables, sps_matrix):
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
                    discr = d.get(pp.DISCRETIZATION, None)
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
                block_dof[(e, k)] = block_dof_counter
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
                    discr = d.get(pp.DISCRETIZATION, None)
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
            discr = d.get(pp.COUPLING_DISCRETIZATION, None)
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

        full_dof = np.array(full_dof)

        # Uniquify list of variable combinations. Then iterate over all variable
        # combinations and initialize matrices of the right size
        num_blocks = len(full_dof)
        for var in list(set(variable_combinations)):

            matrix_dict[var] = np.empty((num_blocks, num_blocks), dtype=np.object)
            rhs_dict[var] = np.empty(num_blocks, dtype=np.object)

            for di in np.arange(num_blocks):
                # Initilize the block diagonal parts, this is useful for the bmat done
                # at the end of assemble_matrix_rhs to know the correct shape of the full_matrix
                matrix_dict[var][di, di] = sps_matrix((full_dof[di], full_dof[di]))
                rhs_dict[var][di] = np.zeros(full_dof[di])

        return matrix_dict, rhs_dict, block_dof, full_dof

    def _assign_matrix_vector(self, dof, sps_matrix):
        # Assign a block matrix and vector with specified number of dofs per block
        num_blocks = len(dof)
        matrix = np.empty((num_blocks, num_blocks), dtype=np.object)
        rhs = np.empty(num_blocks, dtype=np.object)

        for ri in range(num_blocks):
            rhs[ri] = np.zeros(dof[ri])
            for ci in range(num_blocks):
                matrix[ri, ci] = sps_matrix((dof[ri], dof[ci]))

        return matrix, rhs

    def assemble_operator(self, gb, operator_name):
        """ @RUNAR: Placeholder method for use for non-linear problems
        """
        pass

    def _local_variables(self, d, variables):
        # Short-hand function to check if a limited number of admissible
        # variables is given, and if so, if a certain value is contained in
        # the set
        loc_variables = d.get(pp.PRIMARY_VARIABLES, None)
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

    def distribute_variable(self, gb, var, block_dof, full_dof):
        """ Distribute a vector to the nodes and edges in the GridBucket.

        The intended use is to split a multi-physics solution vector into its
        component parts.

        Parameters:
            gb (GridBucket): Where the variables should be distributed
            var (np.array): Vector to be split.
            block_dof (dictionary from tuples, each item on the form (grid, str), to ints):
                The Grid (or MortarGrid) identifies GridBucket elements, the
                string the local variable name. The values signifies the block
                index of this grid-variable combination.
            full_dof (list of ints): Number of dofs for a variable combination.
                The ordering of the list corresponds to block_dof.

        """
        dof = np.cumsum(np.append(0, np.asarray(full_dof)))

        for pair, bi in block_dof.items():
            g = pair[0]
            var_name = pair[1]
            if isinstance(g, pp.Grid):
                data = gb.node_props(g)
            else:  # This is really an edge
                data = gb.edge_props(g)
            data[var_name] = var[dof[bi] : dof[bi + 1]]

    def merge_variable(self, gb, var, block_dof, full_dof):
        """ Merge a vector to the nodes and edges in the GridBucket.

        The intended use is to merge the component parts of a vector into
        its correct position in the global solution vector.

        Parameters:
            gb (GridBucket): Where the variables should be distributed
            var ('string'): Name of vector to be merged. Should be located at the nodes and
                edges.
            block_dof (dictionary from tuples, each item on the form (grid, str), to ints):
                The Grid (or MortarGrid) identifies GridBucket elements, the
                string the local variable name. The values signifies the block
                index of this grid-variable combination.
            full_dof (list of ints): Number of dofs for a variable combination.
                The ordering of the list corresponds to block_dof.

        """
        dof = np.cumsum(np.append(0, np.asarray(full_dof)))

        values = np.zeros(dof[-1])
        for pair, bi in block_dof.items():
            g = pair[0]
            var_name = pair[1]
            if isinstance(g, pp.Grid):
                data = gb.node_props(g)
            else:  # This is really an edge
                data = gb.edge_props(g)
            if var_name == var:
                loc_value = data[var_name]
            else:
                loc_value = 0
            values[dof[bi] : dof[bi + 1]] = loc_value
        return values
