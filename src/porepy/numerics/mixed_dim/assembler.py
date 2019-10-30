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

    def __init__(self, gb, active_variables=None):
        """ Construct an assembler for a given GridBucket on a given set of variables.

        Parameters:
            self.gb (pp.GridBucket): Mixed-dimensional grid where the equations are
                discretized. The data dictionaries on nodes and edges should contain
                variable and discretization information, see tutorial for details.
            active_variables (list of str, optional): Names of variables to be assembled.
                If provided, only decleared primary variables with a name found in the
                list will be assembled, and the size of the matrix will be reduced
                accordingly.
                NOTE: For edge coupling terms where the edge variable is defined
                as active, all involved node variables must also be active.

        Raises:
            ValueError: If an edge_coupling is defined with an active edge variable
                 but with an inactive node variable.

        """
        self.gb = gb

        if active_variables is None:
            self.active_variables = active_variables
        else:
            if not isinstance(active_variables, list):
                active_variables = [active_variables]
            self.active_variables = active_variables

        self._identify_dofs()

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

    def assemble_matrix_rhs(self, matrix_format="csr", add_matrices=True):
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
            matrix_format (str, optional): Matrix format used for the system matrix.
                Defaults to CSR.
            add_matrices (boolean, optional): If True, a single system matrix is added,
                else, separate matrices for each variable and term are returned in a
                dictionary.

        Returns:
            scipy sparse matrix, or dictionary of matrices: Discretization matrix,
                dictionary is returned if add_matrices=False.
            np.ndarray, or dictionary of arrays: Right hand side terms. Dictionary is
                returned if add_matrices=False.

        """
        # Define the matrix format, common for all the sub-matrices
        if matrix_format == "csc":
            sps_matrix = sps.csc_matrix
        else:
            sps_matrix = sps.csr_matrix

        # If there are no variables - most likely if the active_variables do not
        # match any of the decleared variables, we can return now.
        if len(self.full_dof) == 0:
            if add_matrices:
                # If a single returned value is expected, (summed matrices) it is most easy
                # to generate a new, empty matrix, of the right size.
                mat, vec = self._assign_matrix_vector(self.full_dof, sps_matrix)
                return mat, vec
            else:
                return self._initialize_matrix_rhs(sps_matrix)

        # Assemble
        matrix, rhs = self._operate_on_gb("assemble", matrix_format=matrix_format)

        # At this stage, all assembly is done. The remaining step is optionally to
        # add the matrices associated with different terms, and anyhow convert
        # the matrix to a sps. block matrix.
        if add_matrices:
            size = np.sum(self.full_dof)
            full_matrix = sps_matrix((size, size))
            full_rhs = np.zeros(size)

            for mat in matrix.values():
                full_matrix += sps.bmat(mat, matrix_format)

            for vec in rhs.values():
                full_rhs += np.concatenate(tuple(vec))

            return full_matrix, full_rhs

        else:
            for k, v in matrix.items():
                matrix[k] = sps.bmat(v, matrix_format)
            for k, v in rhs.items():
                rhs[k] = np.concatenate(tuple(v))

            return matrix, rhs

    def discretize(self, variable_filter=None, term_filter=None, grid=None):
        """ Run the discretization operation on discretizations specified in
        the mixed-dimensional grid.

        Only active variables will be considered. Moreover, the discretization
        operation can be filtered to only consider specified variables, or terms.
        If the variable filter is active, only discretizations where all variables
        survive the filter will be discretized (for diagonal terms, the variable
        must survive, for off-diagonal terms, both terms must survive).

        Filtering on terms works on the more detailed levels of indivdiual terms
        in a multi-physics discretization (say, zoom-in on the advection term
        in a advection-diffusion system).

        The filters can be combined to select specified terms for specified equations.

        Example (discretization internal to a node or edge:
            For a discretizaiton of the form

            data[pp.DISCRETIZATION] = {'temp': {'advection': Foo(), 'diffusion': Bar()},
                                       'pressure' : {'diffusion': FlowFoo()}}

            variable_filter = ['temp'] will discretize all temp terms

            term_filter = ['diffusion'] will discretize duffusion for both the temp and
                pressure variable

            variable_filter = ['temp'], term_filter = ['diffusion'] will only discretize
                the diffusion term for variable temp

        Example (coupling terms):
            Variable filter works as intenal to nodes / edges.
            The term filter acts on the identifier of a coupling, so

            dd[[pp.COUPLING_DISCRETIZATION]] = {'coupling_id' : {g1: {'temp': 'diffusion'},
                                                                 g2:  {'pressure': diffusion'},
                                                                 (g1, g2): {'coupling_variable': FooBar()}}}

            will survive term_filter = ['coupling_id']


        Parameters:
            variable_filter (optional): List of variables to be discretized. If
                None (default), all active variables are discretized.
            term_filter (optional): List of terms to be discretized. If None
                (default), all terms for all active variables are discretized.
            g (pp.Grid, optional): Grid in GridBucket. If specified, only this
                grid will be considered.

        """
        self._operate_on_gb(
            "discretize",
            variable_filter=variable_filter,
            term_filter=term_filter,
            grid=grid,
        )

    def _operate_on_gb(self, operation, **kwargs):
        """ Helper method, loop over the GridBucket, identify nodes / edges
        variables and discretizations, and perform an operation on these.

        Implemented actions are discretizaiton and assembly.

        """

        if operation == "assemble":

            # Initialize the global matrix.
            # This gives us a set of matrices (essentially one per term per variable)
            # and a simial set of rhs vectors. Furthermore, we get block indices
            # of variables on individual nodes and edges, and count the number of
            # dofs per local variable.
            # For details, and some nuances, see documentation of the funciton
            # _initialize_matrix_rhs.
            matrix_format = kwargs.get("matrix_format", "csc")
            if matrix_format == "csc":
                sps_matrix = sps.csc_matrix
            else:
                sps_matrix = sps.csr_matrix

            matrix, rhs = self._initialize_matrix_rhs(sps_matrix)

            term_filter = None
            variable_filter = None
            target_grid = kwargs.get("grid", None)

        elif operation == "discretize":

            variable_keys = kwargs.get("variable_filter", None)
            if variable_keys is None:
                variable_filter = lambda x: True
            else:
                variable_filter = lambda x: x in variable_keys
            term_keys = kwargs.get("term_filter", None)
            if term_keys is None:
                term_filter = lambda x: True
            else:
                term_filter = lambda x: x in term_keys

            matrix = None
            rhs = None
            target_grid = kwargs.get("grid", None)
            sps_matrix = None

        else:
            # We will only reach this if someone has invoked this private method
            # from the outside.
            raise ValueError("Unknown gb operation " + str(operation))

        self._operate_on_node(
            operation, matrix, rhs, variable_filter, term_filter, target_grid
        )

        self._operate_on_edge(operation, matrix, rhs, variable_filter, term_filter)

        self._operate_on_edge_coupling(
            operation, matrix, rhs, variable_filter, term_filter, sps_matrix
        )

        if operation == "assemble":
            return matrix, rhs
        else:
            return None

    def _operate_on_node(
        self, operation, matrix, rhs, variable_filter, term_filter, target_grid
    ):

        # Loop over all grids, discretize (if necessary) and assemble. This
        # will populate the main diagonal of the equation.
        for g, data in self.gb:
            if target_grid is not None and g is not target_grid:
                continue
            loc_var = self._local_variables(data)
            for row in loc_var.keys():
                for col in loc_var.keys():
                    # Block indices, row and column
                    ri = self.block_dof[(g, row)]
                    ci = self.block_dof[(g, col)]

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

                            if operation == "discretize":
                                if (
                                    variable_filter(row)
                                    and variable_filter(col)
                                    and term_filter(term)
                                ):
                                    d.discretize(g, data)
                            elif operation == "assemble":
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

    def _operate_on_edge(self, operation, matrix, rhs, variable_filter, term_filter):
        for e, data_edge in self.gb.edges():

            # Extract the active local variables for edge
            active_edge_var = self._local_variables(data_edge)

            # First discretize interaction between edge variables locally.
            # This is in direct analogue with the corresponding operation on
            # nodes.
            for row in active_edge_var.keys():
                for col in active_edge_var.keys():
                    ri = self.block_dof[(e, row)]
                    ci = self.block_dof[(e, col)]

                    discr_data = data_edge.get(pp.DISCRETIZATION)
                    if discr_data is None:
                        continue
                    discr = discr_data.get(self.discretization_key(row, col), None)

                    if discr is None:
                        continue
                    else:
                        # Loop over all discretizations
                        for term, d in discr.items():
                            if operation == "discretize":
                                if (
                                    variable_filter(row)
                                    and variable_filter(col)
                                    and term_filter(term)
                                ):
                                    d.discretize(data_edge)
                            elif operation == "assemble":
                                # Assemble the matrix and right hand side. This will also
                                # discretize if not done before.

                                loc_A, loc_b = d.assemble_matrix_rhs(data_edge)

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

    def _operate_on_edge_coupling(
        self, operation, matrix, rhs, variable_filter, term_filter, sps_matrix
    ):
        # Loop over all edges
        for e, data_edge in self.gb.edges():

            # Grids and data dictionaries for master and slave
            g_slave, g_master = self.gb.nodes_of_edge(e)
            data_slave = self.gb.node_props(g_slave)
            data_master = self.gb.node_props(g_master)

            # Then, discretize the interaction between the edge variables of
            # this edge, and the adjacent node variables.
            discr = data_edge.get(pp.COUPLING_DISCRETIZATION, None)
            if discr is None:
                continue

            for coupling_key, terms in discr.items():
                edge_vals = terms.get(e)
                edge_key = edge_vals[0]

                # Only continue if this is an active variable
                if not self._is_active_variable(edge_key):
                    continue

                # Global block index associated with this edge variable
                ei = self.block_dof[(e, edge_key)]

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

                    # Only continue if this is an active variable
                    if not self._is_active_variable(master_key):
                        continue

                    # Global index associated with the master variable
                    mi = self.block_dof.get((g_master, master_key))

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
                    # Only continue if this is an active variable
                    if not self._is_active_variable(slave_key):
                        continue

                    si = self.block_dof.get((g_slave, slave_key))
                    # Also define the key to access the matrix of the discretization of
                    # the slave variable on the slave node.
                    mat_key_slave = self._variable_term_key(
                        slave_vals[1], slave_key, slave_key
                    )

                # Key to the matrix dictionary used to access this coupling
                # discretization.
                mat_key = self._variable_term_key(
                    coupling_key, edge_key, slave_key, master_key
                )

                # Edge discretization object
                e_discr = edge_vals[1]

                # Now there are three options (and a fourth, invalid one):
                # The standard case is that both slave and master variables
                # are used in the coupling. Alternatively, only one of the master or slave is
                # used. The fourth alternative, none of them are active, is not
                # considered valid, and raises an error message.
                if mi is not None and si is not None:
                    if operation == "discretize":
                        if (
                            variable_filter(master_key)
                            and variable_filter(slave_key)
                            and variable_filter(edge_key)
                        ):
                            e_discr.discretize(
                                g_master, g_slave, data_master, data_slave, data_edge
                            )

                    elif operation == "assemble":

                        # Assign a local matrix, which will be populated with the
                        # current state of the local system.
                        # Local here refers to the variable and term on the two
                        # nodes, together with the relavant mortar variable and term
                        # Associate the first variable with master, the second with
                        # slave, and the final with edge.
                        loc_mat, _ = self._assign_matrix_vector(
                            self.full_dof[[mi, si, ei]], sps_matrix
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
                            g_master,
                            g_slave,
                            data_master,
                            data_slave,
                            data_edge,
                            loc_mat,
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
                    # The operation is a simplified version of the full option above.
                    if operation == "discretize":
                        if variable_filter(master_key) and variable_filter(edge_key):
                            e_discr.discretize(g_master, data_master, data_edge)
                    elif operation == "assemble":

                        loc_mat, _ = self._assign_matrix_vector(
                            self.full_dof[[mi, ei]], sps_matrix
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
                    if operation == "discretize":
                        if variable_filter(slave_key) and variable_filter(edge_key):
                            e_discr.discretize(g_slave, data_slave, data_edge)
                    elif operation == "assemble":

                        loc_mat, _ = self._assign_matrix_vector(
                            self.full_dof[[si, ei]], sps_matrix
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

                # Finally, discretize direct couplings between this edge and other
                # edges.
                # The below lines allow only for very specific coupling types:
                #    i) The discretization type of the two edges should be the same
                #   ii) The variable name should be the same for both edges
                #  iii) Only the block edge_ind - other_edge_ind can be filled in.
                # These restrictions may be loosned somewhat in the future, but a
                # general coupling between different edges will not be implemented.
                if operation == "assemble" and e_discr.edge_coupling_via_high_dim:
                    for other_edge, data_other in self.gb.edges_of_node(g_master):

                        # Skip the case where the primary and secondary edge is the same
                        if other_edge == e:
                            continue

                        # Avoid coupling between mortar grids of different dimensions.
                        if (
                            data_other["mortar_grid"].dim
                            != data_edge["mortar_grid"].dim
                        ):
                            continue

                        # Only consider terms where the primary and secondary edge have
                        # the same variable name. This is an intended restriction of the
                        # flexibility of the code: Direct edge couplings are implemented
                        # only to replace explicit variables for boundary conditions on
                        # external boundaries, for which the current implementation
                        # should suffice. While more advanced couplings could easily be
                        # introduced, it will violate the modeling framework for mixed-
                        # dimensional problems.
                        # Although different variable names for the same physics is
                        # permitted in the modeling framework, the current restriction
                        # is considered reasonable for the time being.
                        oi = self.block_dof.get((other_edge, edge_key), None)
                        if oi is None:
                            continue

                        # Assign a local matrix, which will be populated with the
                        # current state of the local system.
                        # Local here refers to the variable and term on the two
                        # nodes, together with the relavant mortar variable and term
                        # Associate the first variable with master, the second with
                        # slave, and the final with edge.
                        loc_mat, _ = self._assign_matrix_vector(
                            self.full_dof[[mi, ei, oi]], sps_matrix
                        )
                        tmp_mat, loc_rhs = e_discr.assemble_edge_coupling_via_high_dim(
                            g_master,
                            data_master,
                            e,
                            data_edge,
                            other_edge,
                            data_other,
                            loc_mat,
                        )
                        matrix[mat_key][ei, oi] = tmp_mat[1, 2]
                        rhs[mat_key][ei] += loc_rhs[1]

                if operation == "assemble" and e_discr.edge_coupling_via_low_dim:
                    for other_edge, data_other in self.gb.edges_of_node(g_slave):

                        # Skip the case where the primary and secondary edge is the same
                        if other_edge == e:
                            continue

                        if (
                            data_other["mortar_grid"].dim
                            != data_edge["mortar_grid"].dim
                        ):
                            continue

                        # Only consider terms where the primary and secondary edge have
                        # the same variable name. This is an intended restriction of the
                        # flexibility of the code: Direct edge couplings are implemented
                        # only to replace explicit variables for boundary conditions on
                        # external boundaries, for which the current implementation
                        # should suffice. While more advanced couplings could easily be
                        # introduced, it will violate the modeling framework for mixed-
                        # dimensional problems.
                        # Although different variable names for the same physics is
                        # permitted in the modeling framework, the current restriction
                        # is considered reasonable for the time being.
                        oi = self.block_dof.get((other_edge, edge_key), None)
                        if oi is None:
                            continue

                        # Assign a local matrix, which will be populated with the
                        # current state of the local system.
                        # Local here refers to the variable and term on the two
                        # nodes, together with the relavant mortar variable and term
                        # Associate the first variable with master, the second with
                        # slave, and the final with edge.
                        loc_mat, _ = self._assign_matrix_vector(
                            self.full_dof[[si, ei, oi]], sps_matrix
                        )
                        tmp_mat, loc_rhs = e_discr.assemble_edge_coupling_via_high_dim(
                            g_slave, data_slave, data_edge, data_other, loc_mat
                        )
                        matrix[mat_key][ei, oi] = tmp_mat[1, 2]
                        rhs[mat_key][ei] += loc_rhs[1]

    def _identify_dofs(self):
        """
        Initialize local matrices for all combinations of variables and operators.

        The function serves three purposes:
            1. Identify all variables and their discretizations defined on individual nodes
               and edges in the GridBucket
            2. To each combination of a node / edge, and a variable, assign an
               index. This will define the ordering of the blocks in the system matrix.

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
            self.gb (GridBukcet): Mixed-dimensional grid.
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
        for g, d in self.gb:

            # Loop over variables, count dofs and identify variable-term
            # combinations internal to the node
            if self._local_variables(d) is None:
                continue
            for key_1, v in self._local_variables(d).items():

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
                for key_2 in self._local_variables(d).keys():
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
        for e, d in self.gb.edges():
            mg = d["mortar_grid"]

            if self._local_variables(d) is None:
                continue
            for key_1, v in self._local_variables(d).items():

                # First count the number of dofs per variable. Note that the
                # identifier here is a tuple of the edge and a variable str.
                block_dof[(e, key_1)] = block_dof_counter
                block_dof_counter += 1

                # We only allow for cell variables on the mortar grid.
                # This will not change in the forseable future
                loc_dof = mg.num_cells * v.get("cells", 0)
                full_dof.append(loc_dof)

                # Then identify all discretization terms for this variable
                for key_2 in self._local_variables(d).keys():
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
            g_slave, g_master = self.gb.nodes_of_edge(e)
            discr = d.get(pp.COUPLING_DISCRETIZATION, None)
            if discr is None:
                continue
            for term, val in discr.items():
                # term identifies the discretization operator (e.g. advection or
                # diffusion), val contains the coupling information

                # Identify this term in the discretization by the variable names
                # on the edge, the variable names of the slave and master grid
                # in that order, and finally the term name.
                # There is a tacit assumption here that self.gb.nodes_of_edge return the
                # grids in the same order here and in the assembly. This should be
                # okay. The consequences for the methods if this is no longer the case is unclear.

                # Get the name of the edge variable (it is the first item in
                # a tuple)
                key_edge = val.get(e)[0]
                if not self._is_active_variable(key_edge):
                    continue

                # Get name of the edge variable, if it exists
                key_slave = val.get(g_slave)
                if key_slave is not None:
                    key_slave = key_slave[0]
                    if not self._is_active_variable(key_slave):
                        raise ValueError(
                            "Edge variable "
                            + key_edge
                            + " is coupled to an inactive node variable "
                            + key_slave
                        )

                else:
                    # This can happen if the the coupling is one-sided, e.g.
                    # it does not consider the slave grid.
                    # An empty string will give no impact on the generated
                    # combination of variable names and discretizaiton terms
                    key_slave = ""

                key_master = val.get(g_master)
                if key_master is not None:
                    key_master = key_master[0]
                    if not self._is_active_variable(key_master):
                        raise ValueError(
                            "Edge variable "
                            + key_edge
                            + " is coupled to an inactive node variable "
                            + key_master
                        )
                else:
                    key_master = ""

                variable_combinations.append(
                    self._variable_term_key(term, key_edge, key_slave, key_master)
                )

        # Array version of the number of dofs per node/edge and variable
        self.full_dof = np.array(full_dof)
        self.block_dof = block_dof
        self.variable_combinations = variable_combinations

    def _initialize_matrix_rhs(self, sps_matrix):
        """
        Initialize a set of matrices (for left hand sides) and vectors (rhs)
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
            sps_matrix (class): Class for sparse matrices, used to initialize
                individual blokcs in the matrix.

        Returns:
            dict: Global system matrices, on block form (one per node/edge per
                variable). There is one item per term (e.g. diffusion/advection)
                per variable.
            dict: Right hand sides. Similar to the system matrix.

        """
        # We will have one discretization matrix per variable
        matrix_dict = {}
        rhs_dict = {}

        num_blocks = len(self.full_dof)

        # Uniquify list of variable combinations. Then iterate over all variable
        # combinations and initialize matrices of the right size
        for var in list(set(self.variable_combinations)):

            # Generate a block matrix
            matrix_dict[var] = np.empty((num_blocks, num_blocks), dtype=np.object)
            rhs_dict[var] = np.empty(num_blocks, dtype=np.object)

            # Loop over all blocks, initialize the diagonal block.
            # We could also initialize off-diagonal blocks, however, this turned
            # out to be computationally expensive.
            for di in np.arange(num_blocks):
                # Initilize the block diagonal parts, this is useful for the bmat done
                # at the end of assemble_matrix_rhs to know the correct shape of the full_matrix
                matrix_dict[var][di, di] = sps_matrix(
                    (self.full_dof[di], self.full_dof[di])
                )
                rhs_dict[var][di] = np.zeros(self.full_dof[di])

        return matrix_dict, rhs_dict

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

    def assemble_operator(self, keyword, operator_name):
        """
        Assemble a global agebraic operator from the local algebraic operators on
        the nodes or edges of a grid bucket. The global operator is a block diagonal
        matrix with the local operators on the diagonal.

        Parameters:
            keyword (string): Keyword for the dictionary in
                d[pp.DISCRETIZATION_MATRICES] for which the operator is stored.
            operator_name (string): keyword for the operator in the
                d[pp.DISCRETIZATION_MATRICES][keyword] dictionary.

        Returns:
            Operator (sps.block_diag): Global algebraic operator.

        """
        operator = []

        def _get_operator(d, keyword, operator_name):
            loc_disc = d[pp.DISCRETIZATION_MATRICES].get(keyword, None)
            if loc_disc is None:  # Return if keyword is not found
                return None
            loc_op = loc_disc.get(operator_name, None)
            return loc_op

        # Loop ever nodes in the gb to find the local operators
        for _, d in self.gb:
            op = _get_operator(d, keyword, operator_name)
            # If a node does not have the keyword or operator, do not add it.
            if op is None:
                continue
            operator.append(op)

        # Loop over edges in the gb to find the local operators
        for _, d in self.gb.edges():
            op = _get_operator(d, keyword, operator_name)
            # If an edge does not have the keyword or operator, do not add it.
            if op is None:
                continue
            operator.append(op)

        if len(operator) == 0:
            raise ValueError(
                "Could not find operator: " + operator_name + " for keyword: " + keyword
            )
        return sps.block_diag(operator)

    def assemble_parameter(self, keyword, parameter_name):
        """
        Assemble a global parameter from the local parameters defined on
        the nodes or edges of a grid bucket. The global parameter is a nd-vector
        of the stacked local parameters.

        Parameters:
        keyword (string): Keyword to access the dictionary
            d[pp.PARAMETERS][keyword] for which the parameters are stored.
        operator_name (string): keyword of the parameter. Will access
            d[pp.DISCRETIZATION_MATRICES][keyword][parameter.
        Returns:
        Operator (sps.block_diag): Global parameter.
        """
        parameter = []
        for _, d in self.gb:
            parameter.append(d[pp.PARAMETERS][keyword][parameter_name])
        return np.hstack(parameter)

    def _local_variables(self, d):
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
        if self.active_variables is None:
            # No restriction necessary.
            return loc_variables
        else:
            # Find intersection with decleared active variables.
            var = {}
            for key, val in loc_variables.items():
                if key in self.active_variables:
                    var[key] = val
            return var

    def _is_active_variable(self, key):
        """ Check if a key denotes an active variable

        Parameters:
            key (str): Variable identifier.
            active_variables (list of str, or str): Active variables.

        Returns:
            boolean: True if key is in active_variables, or active_variables
                is None.

        """
        if self.active_variables is None:
            return True
        else:
            return key in self.active_variables

    def distribute_variable(self, values, variable_names=None, use_state=True):
        """ Distribute a vector to the nodes and edges in the GridBucket.

        The intended use is to split a multi-physics solution vector into its
        component parts.

        Parameters:
            values (np.array): Vector to be split. It is assumed that it corresponds
                to the ordering implied in block_dof and full_dof, e.g. that it is
                the solution of a linear system assembled with the assembler.
            variable_names (list of str, optional): Names of the variable to be
                distributed. If not provided, all variables found in block_dof
                will be distributed
            use_state (boolean, optional): If True (default), the data will be stored in
                data[pp.STATE][variable_name]. If not, store it directly in the data
                dictionary on the components of the GridBucket.

        """
        if variable_names is None:
            variable_names = []
            for pair in self.block_dof.keys():
                variable_names.append(pair[1])

        dof = np.cumsum(np.append(0, np.asarray(self.full_dof)))

        for var_name in set(variable_names):
            for pair, bi in self.block_dof.items():
                g = pair[0]
                name = pair[1]
                if name != var_name:
                    continue
                if isinstance(g, tuple):
                    # This is really an edge
                    data = self.gb.edge_props(g)
                else:
                    data = self.gb.node_props(g)

                if pp.STATE in data.keys():
                    data[pp.STATE][var_name] = values[dof[bi] : dof[bi + 1]]
                else:
                    data[pp.STATE] = {var_name: values[dof[bi] : dof[bi + 1]]}

    def merge_variable(self, var):
        """ Merge a vector to the nodes and edges in the GridBucket.

        The intended use is to merge the component parts of a vector into
        its correct position in the global solution vector.

        Parameters:
            var ('string'): Name of vector to be merged. Should be located at the nodes and
                edges.

        """
        dof = np.cumsum(np.append(0, np.asarray(self.full_dof)))

        values = np.zeros(dof[-1])
        for pair, bi in self.block_dof.items():
            g = pair[0]
            var_name = pair[1]
            if isinstance(g, tuple):
                # This is really an edge
                data = self.gb.edge_props(g)
            else:
                data = self.gb.node_props(g)
            if var_name == var:
                loc_value = data[pp.STATE][var_name]
            else:
                loc_value = 0
            values[dof[bi] : dof[bi + 1]] = loc_value
        return values

    def dof_ind(self, g, name):
        """ Get the indices in the global system of variables associated with a
        given node / edge (in the GridBucket sense) and a given variable.

        Parameters:
            g (pp.Grid or pp.GridBucket edge): Either a grid, or an edge in the
                GridBucket.
            name (str): Name of a variable. Should be an active variable.

        Returns:
            np.array (int): Index of degrees of freedom for this variable.

        """
        block_ind = self.block_dof[(g, name)]
        dof_start = np.hstack((0, np.cumsum(self.full_dof)))
        return np.arange(dof_start[block_ind], dof_start[block_ind + 1])

    def num_dof(self):
        """ Get total number of unknowns of the identified variables.

        Returns:
            int: Number of unknowns. Size of solution vector.
        """
        return self.full_dof.sum()
