#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The module contains the Assembler class, which is responsible for assembly of
system matrix and right hand side for a general multi-domain, multi-physics problem.
"""
import numpy as np
import scipy.sparse as sps

import porepy as pp


class Assembler:
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

    def _variable_term_key(
        self, term: str, key_1: str, key_2: str, key_3: str = None
    ) -> str:
        """ Get the key-variable combination used to identify a specific term in the equation.

        For nodes and internally to edges in the GridBucket (i.e. fixed-dimensional grids),
        the variable name is formed by combining the name of one or two primary variables,
        and the name of term (all of which are defined in the data dictionary
        of this node / edge. As examples:
            - An advection-diffusion equation will typically have two terms, say,
                advection_temperature, diffusion_temperature
            - For a coupled flow-temperature discretization, the coupling (off-diagonal)
                terms may have identifiers 'coupling_temperature_flow' and
                'coupling_flow_temperature'

        For couplings between edges and nodes, a three variable combination is needed,
        identifying variable names on the edge and the respective neighboring nodes.

        NOTE: The naming of variables and terms are left to the user. For examples
        on how to set this up, confer the tutorial parameter_asignment_assembler_setup

        Parameters:
            term (str): Identifier of a discretization operation.
            key_1 (str): Variable name.
            key_2 (str): Variable name
            key_3 (str, optional): Variable name. If not provided, a 2-variable
                identifier is returned, that is, we are not working on a node-edge
                coupling.

        Returns:
            str: Identifier for this combination of term and variables.

        """
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
        self, gb, matrix_format="csr", active_variables=None, add_matrices=True
    ):
        """ Assemble the system matrix and right hand side for a general linear
        multi-physics problem, and return a block matrix and right hand side.

        For examples on how to use the assembler, confer the tutorial
        parameter_assignment_assembler_setup.ipynb. Here, we list the main capabilities
        of the assembler:
            * Assign an arbitrary number of variables on each node and edge in the grid
              bucket. Allow for general couplings between the variables internal to each
              node / edge.
            * Assign general coupling schemes between edges and one or both neighboring
              nodes. There are no limitations on variable naming conventions in the
              coupling.
            * Construct a system matrix that only consideres a subset of the variables
              defined in the GridBucket data dictionary.
            * Return either a single discretization matrix covering all variables and
              terms, or one matrix per term per variable. The latter is useful e.g. in
              operator splitting or time stepping schemes.

        In all cases, it is assumed that a discretization object for the relevant terms
        is available. It is up to the user to ensure that the resulting problem is
        well posed.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid where the equations are
                discretized. The data dictionaries on nodes and edges should contain
                variable and discretization information, see tutorial for details.
            matrix_format (str, optional): Matrix format used for the system matrix.
                Defaults to CSR.
            active_variables (list of str, optional): Names of variables to be assembled.
                If provided, only decleared primary variables with a name found in the
                list will be assembled, and the size of the matrix will be reduced
                accordingly.
            add_matrices (boolean, optional): If True, a single system matrix is added,
                else, separate matrices for each variable and term are returned in a
                dictionary.

        Returns:
            scipy sparse matrix, or dictionary of matrices: Discretization matrix,
                dictionary is returned if add_matrices=False.
            np.ndarray, or dictionary of arrays: Right hand side terms. Dictionary is
                returned if add_matrices=False.
            dictionary: Mapping from GridBucket nodes / edges + variables to the
                corresponding block index. The keys on a node is defined by tuples
                (grid, variable_name), while on an edge e, the key is (e, variable_name).
            np.ndarray: For each matrix block, the number of degrees of freedom.

        """
        # Define the matrix format, common for all the sub-matrices
        if matrix_format == "csc":
            sps_matrix = sps.csc_matrix
        else:
            sps_matrix = sps.csr_matrix

        # Initialize the global matrix.
        # This gives us a set of matrices (essentially one per term per variable)
        # and a simial set of rhs vectors. Furthermore, we get block indices
        # of variables on individual nodes and edges, and count the number of
        # dofs per local variable.
        # For details, and some nuances, see documentation of the funciton
        # _initialize_matrix_rhs.
        matrix, rhs, block_dof, full_dof = self._initialize_matrix_rhs(
            gb, active_variables, sps_matrix
        )
        # If there are no variables - most likely if the active_variables do not
        # match any of the decleared variables, we can return now.
        if len(full_dof) == 0:
            if add_matrices:
                # If a single returned value is expected, (summed matrices) it is most easy
                # to generate a new, empty matrix, of the right size.
                mat, vec = self._assign_matrix_vector(full_dof, sps_matrix)
                return mat, vec, block_dof, full_dof
            else:
                return matrix, rhs, block_dof, full_dof

        # Loop over all grids, discretize (if necessary) and assemble. This
        # will populate the main diagonal of the equation.
        for g, data in gb:
            loc_var = self._local_variables(data, active_variables)
            for row in loc_var.keys():
                for col in loc_var.keys():
                    # Block indices, row and column
                    ri = block_dof[(g, row)]
                    ci = block_dof[(g, col)]

                    # Get the specified discretization, if any.
                    discr_data = data.get(pp.DISCRETIZATION, None)
                    if discr_data is None:
                        continue
                    discr = discr_data.get(self.discretization_key(row, col), None)

                    if discr is None:
                        continue
                    else:
                        # Only if non-empty discretization operators are defined,
                        # we should do something.
                        # Loop over all discretizations
                        for term, d in discr.items():
                            # Assemble the matrix and right hand side. This will also
                            # discretize if not done before.
                            loc_A, loc_b = d.assemble_matrix_rhs(g, data)

                            # Assign values in global matrix: Create the same key used
                            # defined when initializing matrices (see that function)
                            var_key_name = self._variable_term_key(term, row, col)

                            # Check if the current block is None or not, it could
                            # happend based on the problem setting. Better to stay
                            # on the safe side.
                            if matrix[var_key_name][ri, ci] is None:
                                matrix[var_key_name][ri, ci] = loc_A
                            else:
                                matrix[var_key_name][ri, ci] += loc_A
                            # The right hand side vector is always initialized.
                            rhs[var_key_name][ri] += loc_b

        # Loop over all edges
        for e, data_edge in gb.edges():

            # Grids and data dictionaries for master and slave
            g_slave, g_master = gb.nodes_of_edge(e)
            data_slave = gb.node_props(g_slave)
            data_master = gb.node_props(g_master)

            # Extract the active local variables for edge
            active_edge_var = self._local_variables(data_edge, active_variables)

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

                # Global block index associated with this edge variable
                ei = block_dof[(e, edge_key)]

                # Get variable name and block index of the master variable.
                # NOTE: We do no test on whether the master variable is among
                # the active ones here. If we ever deactive the master variable
                # but assign a coupling discretization that requires the master
                # variable to be there, we will get an error message somewhere
                # below.
                master_vals = terms.get(g_master)
                if master_vals is None:
                    # An empty identifying string will create no problems below.
                    master_key = ""
                    # If the master variable index is None, this signifies that
                    # the master variable index is not active
                    mi = None
                else:
                    # Name of the relevant variable on the master grid
                    master_key = master_vals[0]
                    # Global index associated with the master variable
                    mi = block_dof.get((g_master, master_key))

                    # Also define the key to access the matrix of the discretization of
                    # the master variable on the master node.
                    mat_key_master = self._variable_term_key(
                        master_vals[1], master_key, master_key
                    )
                # Do similar operations for the slave variable.
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

                # Edge discretization object
                e_discr = edge_vals[1]

                # Now there are three options (and a fourth, invalid one):
                # The standard case is that both slave and master variables
                # are used in the coupling. Alternatively, only one of the master or slave is
                # used. The fourth alternative, none of them are active, is not
                # considered valid, and raises an error message.
                if mi is not None and si is not None:

                    # Assign a local matrix, which will be populated with the
                    # current state of the local system.
                    # Local here refers to the variable and term on the two
                    # nodes, together with the relavant mortar variable and term
                    # Associate the first variable with master, the second with
                    # slave, and the final with edge.
                    loc_mat, _ = self._assign_matrix_vector(
                        full_dof[[mi, si, ei]], sps_matrix
                    )

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

                    # And add coupling directly between master and slave
                    # nodes
                    matrix[mat_key_master][mi, si] = tmp_mat[0, 1]
                    matrix[mat_key_slave][si, mi] = tmp_mat[1, 0]

                    # Finally take care of the right hand side
                    rhs[mat_key][[mi, si, ei]] += loc_rhs

                elif mi is not None:
                    # si is None
                    # The operation is a simplified version of the full option above.
                    loc_mat, _ = self._assign_matrix_vector(
                        full_dof[[mi, ei]], sps_matrix
                    )
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
                    # The operation is a simplified version of the full option above.
                    loc_mat, _ = self._assign_matrix_vector(
                        full_dof[[si, ei]], sps_matrix
                    )
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

        # At this stage, all assembly is done. The remaining step is optionally to
        # add the matrices associated with different terms, and anyhow convert
        # the matrix to a sps. block matrix.
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

    def _initialize_matrix_rhs(self, gb, active_variables, sps_matrix):
        """
        Initialize local matrices for all combinations of variables and operators.

        The function serves three purposes:
            1. Identify all variables and their discretizations defined on individual nodes
               and edges in the GridBucket
            2. To each combination of a node / edge, and a variable, assign an
               index. This will define the ordering of the blocks in the system matrix.
            3. Initialize a set of matrices (for left hand sides) and vectors (rhs)
               for all operators associated with a variable (example: a temperature
               variable in an advection-diffusion problem will typically have two
               operators, one for advection, one for diffusion).

        It is useful to differ between the discretization matrices of different
        variables and terms for at least two reasons:
          1) It is useful in time stepping methods, where only some terms
             are time dependent
          2) In some discretization schemes, the coupling discretization can
             override discretizations on the neighboring nodes. It is critical
             that this only overrides values associated with the relevant terms.
        We therefore generate one discretization matrix and right hand side
        per term, as identified in variable_combinations.
        NOTE: It is possible to construct cases where variable and discretization
        names give unfortunate consequences. However, it does not seem worth
        the effort to split the matrix even further.

        Parameters:
            gb (GridBukcet): Mixed-dimensional grid.
            active_variables (list of str): Name of active variables. If empty,
                or None, all variables are considered active.
            sps_matrix (class): Class for sparse matrices, used to initialize
                individual blokcs in the matrix.

        Returns:
            dict: Global system matrices, on block form (one per node/edge per
                variable). There is one item per term (e.g. diffusion/advection)
                per variable.
            dict: Right hand sides. Similar to the system matrix.
            dict: Giving the block index of a variable on a specific node/edge.
                The dictionary keys take the form of a tuple (grid, variable_name)
                on GridBucket nodes, (edge, variable_name) on edges.
            np.array: For each variable on each node/edge, the number of dofs
                needed.

        """
        # Implementation note: To fully understand the structure of this function
        # it is useful to consider an example of a data dictionary with decleared
        # primary variables and discretization operators.
        # The function needs to dig deep into the dictionaries used in these
        # declarations, thus the code is rather involved.

        # Counter for block index
        block_dof_counter = 0
        # Dictionary that maps node/edge + variable combination to an index.
        block_dof = {}
        # Storage for number of dofs per variable per node/edge, with respect
        # to the ordering specified in block_dof
        full_dof = []

        # Store all combinations of variable pairs (e.g. row-column indices in
        # the global system matrix), and identifiers of discretization operations
        # (e.g. advection or diffusion).
        # Note: This list is common for all nodes / edges.
        variable_combinations = []

        # Loop over all nodes in the grid bucket, identify its local and active
        # variables.
        for g, d in gb:

            # Loop over variables, count dofs and identify variable-term
            # combinations internal to the node
            for key_1, v in self._local_variables(d, active_variables).items():

                # First assign a block index.
                # Note that the keys in the dictionary is a tuple, with a grid
                # and a variable name (str)
                block_dof[(g, key_1)] = block_dof_counter
                block_dof_counter += 1

                # Count number of dofs for this variable on this grid and store it
                loc_dof = (
                    g.num_cells * v.get("cells", 0)
                    + g.num_faces * v.get("faces", 0)
                    + g.num_nodes * v.get("nodes", 0)
                )
                full_dof.append(loc_dof)

                # Next, identify all defined discretization terms for this variable.
                # Do a second loop over the variables of the grid, the combination
                # of the two keys givs us all coupling terms (e.g. an off-diagonal
                # block in the global matrix)
                for key_2 in self._local_variables(d, active_variables).keys():
                    # We need to identify identify individual discretization terms
                    # defined for this equaton. This are identified either by
                    # the key k (for variable dependence on itself), or the
                    # combination k_k2 if the variables are mixed
                    if key_1 == key_2:
                        merged_key = key_1
                    else:
                        merged_key = key_1 + "_" + key_2
                    # Get hold of the discretization operators defined for this
                    # node / edge; we really just need the keys in the
                    # discretization map.
                    # The default assumption is that no discretization has
                    # been defined, in which case we do nothing.
                    discr = d.get(pp.DISCRETIZATION, None)
                    if discr is None:
                        continue
                    # Loop over all the discretization operations, if any, and
                    # add it to the list of observed variables.
                    # We will take care of duplicates below.
                    terms = discr.get(merged_key, None)
                    if terms is None:
                        continue
                    for term in terms.keys():
                        variable_combinations.append(
                            self._variable_term_key(term, key_1, key_2)
                        )

        # Next do the equivalent operation for edges in the grid.
        # Most steps are identical to the operations on the nodes, we comment
        # only on edge-specific aspects; see above loop for more information
        for e, d in gb.edges():
            mg = d["mortar_grid"]

            for key_1, v in self._local_variables(d, active_variables).items():

                # First count the number of dofs per variable. Note that the
                # identifier here is a tuple of the edge and a variable str.
                block_dof[(e, key_1)] = block_dof_counter
                block_dof_counter += 1

                # We only allow for cell variables on the mortar grid.
                # This will not change in the forseable future
                loc_dof = mg.num_cells * v.get("cells", 0)
                full_dof.append(loc_dof)

                # Then identify all discretization terms for this variable
                for key_2 in self._local_variables(d, active_variables).keys():
                    if key_1 == key_2:
                        merged_key = key_1
                    else:
                        merged_key = key_1 + "_" + key_2
                    discr = d.get(pp.DISCRETIZATION, None)
                    if discr is None:
                        continue
                    terms = discr.get(merged_key, None)
                    if terms is None:
                        continue
                    for term in terms.keys():
                        variable_combinations.append(
                            self._variable_term_key(term, key_1, key_2)
                        )

            # Finally, identify variable combinations for coupling terms.
            # This involves both the neighboring grids
            g_slave, g_master = gb.nodes_of_edge(e)
            discr = d.get(pp.COUPLING_DISCRETIZATION, None)
            if discr is None:
                continue
            for term, val in discr.items():
                # term identifies the discretization operator (e.g. advection or
                # diffusion), val contains the coupling information

                # Identify this term in the discretization by the variable names
                # on the edge, the variable names of the slave and master grid
                # in that order, and finally the term name.
                # There is a tacit assumption here that gb.nodes_of_edge return the
                # grids in the same order here and in the assembly. This should be
                # okay. The consequences for the methods if this is no longer the case is unclear.

                # Get the name of the edge variable (it is the first item in
                # a tuple)
                key_edge = val.get(e)[0]
                # Get name of the edge variable, if it exists
                key_slave = val.get(g_slave)
                if key_slave is not None:
                    key_slave = key_slave[0]
                else:
                    # This can happen if the the coupling is one-sided, e.g.
                    # it does not consider the slave grid.
                    # An empty string will give no impact on the generated
                    # combination of variable names and discretizaiton terms
                    key_slave = ""

                key_master = val.get(g_master)
                if key_master is not None:
                    key_master = key_master[0]
                else:
                    key_master = ""
                variable_combinations.append(
                    self._variable_term_key(term, key_edge, key_slave, key_master)
                )

        # By now, we have identified all active variables on nodes and edges in
        # the GridBucket, and assigned a block index to ecah variable on each
        # node/edge. Finally, we need to assign matrix and right hand sides
        # for these terms.
        # We will have one discretization matrix per variable
        matrix_dict = {}
        rhs_dict = {}

        # Array version of the number of dofs per node/edge and variable
        full_dof = np.array(full_dof)
        num_blocks = len(full_dof)

        # Uniquify list of variable combinations. Then iterate over all variable
        # combinations and initialize matrices of the right size
        for var in list(set(variable_combinations)):

            # Generate a block matrix
            matrix_dict[var] = np.empty((num_blocks, num_blocks), dtype=np.object)
            rhs_dict[var] = np.empty(num_blocks, dtype=np.object)

            # Loop over all blocks, initialize the diagonal block.
            # We could also initialize off-diagonal blocks, however, this turned
            # out to be computationally expensive.
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

    def _local_variables(self, d, active_variables):
        """ Find variables defined in a data dictionary, and do intersection
        with defined active variables.

        If no active variables are specified, returned all decleared variables.

        Parameters:
            d (dict): Data dictionary defined on a GridBucket node or edge
            active_variables (list of str, or str): Active variables.

        Returns:
            dict: With variable names and information (#dofs of various kinds), as
                specified by user, but possibly restricted to the active variables

        """

        # Active variables
        loc_variables = d.get(pp.PRIMARY_VARIABLES, None)
        if active_variables is None:
            # No restriction necessary.
            return loc_variables
        else:
            # Find intersection with decleared active variables.
            if not isinstance(active_variables, list):
                active_variables = [active_variables]
            var = {}
            for key, val in loc_variables.items():
                if key in active_variables:
                    var[key] = val
            return var

    def distribute_variable(self, gb, values, block_dof, full_dof, variable_names=None):
        """ Distribute a vector to the nodes and edges in the GridBucket.

        The intended use is to split a multi-physics solution vector into its
        component parts.

        Parameters:
            gb (GridBucket): Where the variables should be distributed
            values (np.array): Vector to be split. It is assumed that it corresponds
                to the ordering implied in block_dof and full_dof, e.g. that it is
                the solution of a linear system assembled with the assembler.
            block_dof (dictionary from tuples, each item on the form (grid, str), to ints):
                The Grid (or MortarGrid) identifies GridBucket elements, the
                string the local variable name. The values signifies the block
                index of this grid-variable combination.
            full_dof (list of ints): Number of dofs for a variable combination.
                The ordering of the list corresponds to block_dof.
            variable_names (list of str, optional): Names of the variable to be
                distributed. If not provided, all variables found in block_dof
                will be distributed

        """
        if variable_names is None:
            variable_names = []
            for pair in block_dof.keys():
                variable_names.append(pair[1])

        dof = np.cumsum(np.append(0, np.asarray(full_dof)))

        for var_name in set(variable_names):
            for pair, bi in block_dof.items():
                g = pair[0]
                name = pair[1]
                if name != var_name:
                    continue
                if isinstance(g, pp.Grid):
                    data = gb.node_props(g)
                else:  # This is really an edge
                    data = gb.edge_props(g)
                data[var_name] = values[dof[bi] : dof[bi + 1]]

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
