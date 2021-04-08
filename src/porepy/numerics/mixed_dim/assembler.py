"""
The module contains the Assembler class, which is responsible for assembly of
system matrix and right hand side for a general multi-domain, multi-physics problem.
"""
from collections import namedtuple
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

module_sections = ["numerics", "discretization", "assembly"]

csc_or_csr_matrix = Union[sps.csc_matrix, sps.csr_matrix]

GridVariableTerm = namedtuple("GridVariableTerm", ["grid", "row", "col", "term"])
GridVariableTerm.__doc__ += (
    "Combinations of grids variables and terms found in GridBucket"
)
GridVariableTerm.grid.__doc__ = "Item in GridBucket. Can be node or edge."
GridVariableTerm.row.__doc__ = "Variable name of the row for this term."
GridVariableTerm.col.__doc__ = (
    "Variable name of the column for this term. "
    + "Differs from row for variable couplings."
)
GridVariableTerm.term.__doc__ = "Term for this discretization"

CouplingVariableTerm = namedtuple(
    "CouplingVariableTerm", ["coupling", "edge", "primary", "secondary", "term"]
)


class Assembler:
    """A class that assembles multi-physics problems on mixed-dimensional
    domains.

    The class is designed to combine different variables on different grids,
    different discretizations for the same variable, various coupling schemes
    between the grids etc. To use the functionality, discretization schemes
    for the individual terms in the equation must be defined and follow certain
    rules. For further description, see the documentation of self.assemble_matrix_rhs().

    """

    @pp.time_logger(sections=module_sections)
    def __init__(
        self, gb: pp.GridBucket, dof_manager: Optional[pp.DofManager] = None
    ) -> None:
        """Construct an assembler for a given GridBucket on a given set of variables.

        Parameters:
            self.gb (pp.GridBucket): Mixed-dimensional grid where the equations are
                discretized. The data dictionaries on nodes and edges should contain
                variable and discretization information, see tutorial for details.

        """
        self.gb = gb

        if dof_manager is None:
            dof_manager = pp.DofManager(gb)

        self._dof_manager = dof_manager

        # Identify all variable couplings in the GridBucket, and assign degrees of
        # freedom for each block.
        self._identify_variable_combinations()

    @staticmethod
    @pp.time_logger(sections=module_sections)
    def _discretization_key(row: str, col: str = None) -> str:
        if col is None or row == col:
            return row
        else:
            return row + "_" + col

    @staticmethod
    @pp.time_logger(sections=module_sections)
    def _variable_term_key(term: str, key_1: str, key_2: str, key_3: str = None) -> str:
        """Get the key-variable combination used to identify a specific term in the equation.

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

    @pp.time_logger(sections=module_sections)
    def assemble_matrix_rhs(
        self,
        filt: Optional[pp.assembler_filters.AssemblerFilter] = None,
        matrix_format: str = "csr",
        add_matrices: bool = True,
        only_matrix: bool = False,
        only_rhs: bool = False,
    ) -> Union[
        Tuple[Union[csc_or_csr_matrix, np.ndarray], np.ndarray],
        Tuple[Dict[str, sps.spmatrix], Dict[str, np.ndarray]],
    ]:
        """Assemble the system matrix and right hand side for a general linear
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
              @pp.time_logger(sections=module_sections)
              defined in the GridBucket data dictionary.
            * Return either a single discretization matrix covering all variables and
              terms, or one matrix per term per variable. The latter is useful e.g. in
              operator splitting or time stepping schemes.

        The latter two effects can be achieved by applying a filter to the assembly
        operation, see for instance pp.assembler_filters.ListFilter.

        In all cases, it is assumed that a discretization object for the relevant terms
        is available. It is up to the user to ensure that the resulting problem is
        well posed.

        Parameters:
            filt (pp.assembler_filters.AssemblerFilter, optional): Filter to invoke
                selected discretizations. Defaults to a PassAllFilter, which will
                lead to discretization of all terms in the entire GridBucket.
            matrix_format (str, optional): Matrix format used for the system matrix.
                Defaults to CSR.
            add_matrices (boolean, optional): If True, a single system matrix is added,
                else, separate matrices for each variable and term are returned in a
                dictionary.
            only_matrix (boolean, optional). If True, only the matrix will be assembled.
                Note that some discretization methods will still invoke its full
                assemble_matrix_rhs method. This method will still return a (zero)
                rhs vector.
            only_rhs (boolean, optional). If True, only the rhs will be assembled.
                Note that some discretization methods will still invoke its full
                assemble_matrix_rhs method. This method will still return a (zero)
                discretization matrix.

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

        if only_matrix and only_rhs:
            raise ValueError(
                "At most one of 'only_matrix' and 'only_rhs' should be True"
            )

        # If there are no variables, w can return now
        if self._dof_manager.full_dof.size == 0:
            if add_matrices:
                # If a single returned value is expected, (summed matrices) it is most easy
                # to generate a new, empty matrix, of the right size.
                mat, vec = self._assign_matrix_vector(
                    self._dof_manager.full_dof, sps_matrix
                )
                return mat, vec
            else:
                return self._initialize_matrix_rhs(sps_matrix)

        # Assemble
        matrix, rhs = self._operate_on_gb(  # type:ignore
            operation="assemble",
            filt=filt,
            matrix_format=matrix_format,
            assemble_matrix_only=only_matrix,
            assemble_rhs_only=only_rhs,
        )

        # At this stage, all assembly is done. The remaining step is optionally to
        # add the matrices associated with different terms, and anyhow convert
        # the matrix to a sps. block matrix.
        if add_matrices:
            size = np.sum(self._dof_manager.full_dof)
            full_matrix = sps_matrix((size, size))
            full_rhs = np.zeros(size)  # type: ignore

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

    @pp.time_logger(sections=module_sections)
    def update_discretization(
        self, filt: Optional[pp.assembler_filters.AssemblerFilter] = None
    ) -> None:
        """Update discretizations without a full rediscretization.

        The method will invoke the update_discretization() method on discretizations
        on all grids which have the parameter partial_update set to True in its data
        dictionary. If a Filter is given to this function, the partial update will
        be used as an additional filter.

        Parameters:
            filt (pp.assembler_filters.AssemblerFilter, optional): Filter.

        """
        # Only those grids that are marked for update will be updated. This is
        # implemented as an additional filtering step.

        if filt is None or isinstance(filt, pp.assembler_filters.AllPassFilter):

            # If no filter is provided, make a ListFilter that effectively is AllPass.
            # The grid list must be constructed explicitly, we may remove items below.
            grid_list_type = Union[
                Union[pp.Grid, List[pp.Grid]],
                Tuple[pp.Grid, pp.Grid],
                Tuple[pp.Grid, pp.Grid, Tuple[pp.Grid, pp.Grid]],
            ]

            grid_list: List[grid_list_type] = [g for g, _ in self.gb]
            grid_list += [e for e, _ in self.gb.edges()]
            grid_list += [(e[0], e[1], e) for e, _ in self.gb.edges()]
            variable_list: List[str] = []
            term_list: List[str] = []

        elif isinstance(filt, pp.assembler_filters.ListFilter):
            # Pick from ListFilter
            grid_list = filt._grid_list
            variable_list = filt._variable_list
            term_list = filt._term_list

        else:
            raise NotImplementedError(
                "Discretization update cannot be combined with non-standard filter"
            )

        # Keep track of which grids are marked or update
        update_grid: Dict[pp.Grid, bool] = {}

        # Represent as set for easy removal of grids
        grid_set = set(grid_list)

        # Loop over all nodes, either register them as marked for update, or remove
        # from the grid_set.
        for g, d in self.gb:
            if d.get("partial_update", False):
                update_grid[g] = True
            else:
                if g in grid_set:
                    grid_set.remove(g)
                update_grid[g] = False

        for e, d_e in self.gb.edges():
            # if edge not marked for partial update, remove
            update_edge = d_e.get("partial_update", False)
            if not update_edge and e in grid_set:
                grid_set.remove(e)

            # The coupling should be updated if the edge or any of the neigboring
            # grids is marked for update
            if (
                not (update_grid[e[0]] or update_grid[e[1]] or update_edge)
                and (e[0], e[1], e) in grid_set
            ):
                grid_set.remove((e[0], e[1], e))

        # Create a new filter with only grids marked for update.
        new_filt = pp.assembler_filters.ListFilter(
            grid_list=list(grid_set), variable_list=variable_list, term_list=term_list
        )

        self._operate_on_gb(operation="update_discretization", filt=new_filt)

    @pp.time_logger(sections=module_sections)
    def discretize(
        self, filt: Optional[pp.assembler_filters.AssemblerFilter] = None
    ) -> None:
        """Run the discretization operation on discretizations specified in
        the mixed-dimensional grid.

        Discretization can be applied selectively to specific discretization objcets
        in the GridBucket by passing an appropriate filter. See pp.assembler_filters
        for details, in particular the class ListFilter.

        Parameters:
            filt (pp.assembler_filters.AssemblerFilter, optional): Filter to invoke
                selected discretizations. Defaults to a PassAllFilter, which will
                lead to discretization of all terms in the entire GridBucket.

        """
        self._operate_on_gb("discretize", filt=filt)

    @pp.time_logger(sections=module_sections)
    def _operate_on_gb(
        self,
        operation: str,
        filt: Optional[pp.assembler_filters.AssemblerFilter] = None,
        **kwargs,
    ) -> Union[
        Tuple[csc_or_csr_matrix, np.ndarray],
        Tuple[Dict[str, csc_or_csr_matrix], Dict[str, np.ndarray]],
        None,
    ]:
        """Helper method, loop over the GridBucket, identify nodes / edges
        variables and discretizations, and perform an operation on these.

        Implemented actions are discretization and assembly.

        """
        if filt is None:
            # If the filter is not specified, do no filtering.
            filt = pp.assembler_filters.AllPassFilter()

        # Both assemble and discretize relies on t
        if operation == "assemble":
            # Initialize the global matrix.
            # This gives us a set of matrices (essentially one per term per variable)
            # and a similar set of rhs vectors. Furthermore, we get block indices
            # of variables on individual nodes and edges, and count the number of
            # dofs per local variable.
            # For details, and some nuances, see documentation of the function
            # _initialize_matrix_rhs.
            matrix_format: str = kwargs.get("matrix_format", "csc")
            if matrix_format == "csc":
                sps_matrix = sps.csc_matrix
            else:
                sps_matrix = sps.csr_matrix

            matrix, rhs = self._initialize_matrix_rhs(sps_matrix)

            extra_args = {
                key: kwargs[key]
                for key in ["assemble_matrix_only", "assemble_rhs_only"]
            }

            # Make term and variable filters that let everything through

        elif operation == "discretize" or operation == "update_discretization":
            matrix = None  # type:ignore
            rhs = None  # type:ignore
            sps_matrix = None
            extra_args = {}
        else:
            # We will only reach this if someone has invoked this private method
            # from the outside.
            raise ValueError("Unknown gb operation " + str(operation))

        # First take care of operations internal to nodes and edges
        self._operate_on_nodes_and_edges(filt, operation, matrix, rhs, **extra_args)

        # Next, handle coupling over edges
        self._operate_on_edge_coupling(
            filt, operation, matrix, rhs, sps_matrix, **extra_args
        )

        # Return type depends on operation
        if operation == "assemble":
            return matrix, rhs
        else:
            return None

    @pp.time_logger(sections=module_sections)
    def _operate_on_nodes_and_edges(
        self,
        filt: pp.assembler_filters.AssemblerFilter,
        operation: str,
        matrix: Optional[Dict[str, np.ndarray]] = None,
        rhs: Optional[Dict[str, np.ndarray]] = None,
        assemble_matrix_only: Optional[bool] = False,
        assemble_rhs_only: Optional[bool] = False,
    ):
        for combination in self._grid_variable_term_combinations:
            # Coupling terms should not be considered here
            if isinstance(combination, CouplingVariableTerm):
                continue
            if not filt.filter(
                grids=[combination.grid],
                variables=[combination.row, combination.col],
                terms=[combination.term],
            ):
                continue

            # The grid-like quantity is either a grid or an interface.
            # The two require slightly different function calls etc.

            grid = combination.grid
            is_node = isinstance(grid, pp.Grid)
            # Get hold of data dictionary
            if is_node:
                data = self.gb.node_props(grid)
            else:
                data = self.gb.edge_props(grid)

            # Discretization
            # NOTE: For the edge, this is not the coupling discretization,
            # for that see self._operate_on_nodes_and_edges().
            discr = data[pp.DISCRETIZATION][
                self._discretization_key(combination.row, combination.col)
            ][combination.term]

            # Either discretize (full or update) or assemble
            if operation == "discretize":
                if is_node:
                    discr.discretize(grid, data)
                else:
                    discr.discretize(data)
            elif operation == "update_discretization":
                if is_node:
                    discr.update_discretization(grid, data)
                else:
                    discr.update_discretization(data)
            else:  # assemble
                # Assemble the matrix and right hand side. This will also
                # discretize if not done before.
                # Call appropriate assembler for nodes and edges, respectively.
                if assemble_matrix_only:
                    if is_node:
                        loc_A = discr.assemble_matrix(grid, data)
                    else:
                        loc_A = discr.assemble_matrix(data)
                elif assemble_rhs_only:
                    if is_node:
                        loc_b = discr.assemble_rhs(grid, data)
                    else:
                        loc_b = discr.assemble_rhs(data)
                else:
                    if is_node:
                        loc_A, loc_b = discr.assemble_matrix_rhs(grid, data)
                    else:
                        loc_A, loc_b = discr.assemble_matrix_rhs(data)

                # Assign values in global matrix: Create the same key used
                # defined when initializing matrices (see that function)
                ri = self._dof_manager.block_dof[(grid, combination.row)]
                ci = self._dof_manager.block_dof[(grid, combination.col)]
                var_key_name = self._variable_term_key(
                    combination.term, combination.row, combination.col
                )

                # Check if the current block is None or not, it could
                # based on the problem setting. Better to stay
                # the safe side.
                if not assemble_rhs_only:
                    if matrix[var_key_name][ri, ci] is None:  # type:ignore
                        matrix[var_key_name][ri, ci] = loc_A  # type:ignore
                    else:
                        matrix[var_key_name][ri, ci] += loc_A  # type:ignore
                if not assemble_matrix_only:
                    rhs[var_key_name][ri] += loc_b  # type:ignore

    @pp.time_logger(sections=module_sections)
    def _operate_on_edge_coupling(
        self,
        filt: pp.assembler_filters.AssemblerFilter,
        operation: str,
        matrix: Optional[Dict[str, np.ndarray]],
        rhs: Dict[str, np.ndarray],
        sps_matrix: Type[csc_or_csr_matrix],
        assemble_matrix_only: Optional[bool] = False,
        assemble_rhs_only: Optional[bool] = False,
    ) -> None:
        """Perform operation on all edge-node couplings.

        This method should not be invoked directly, but instead accessed via the public
        methods discretize() or assemble_matrix_rhs()

        Parameters:
            operation (str): Should be 'assemble' or 'discretize'.
            matrix (dict): Dictionary that maps strings of variable combinations to the
                block matrix. The keys are variable combinations, found in
                self.variable_combinations. The values are block matrices, stored as
                np.ndarrays, with each array item defined as a sps.spmatrix.
                Only needed if operation == 'assemble'.
            rhs (dict): Dictionary that maps strings of variable combinations to the
                block rhsx. The keys are variable combinations, found in
                self.variable_combinations. The values are block vectors, stored as
                np.ndarrays, with each array item defined as an np.ndarray.
                Only needed if operation == 'assemble'.
            sps_matrix(csc or csr sparse matrix): The sparse matrix format.
                Should be csc or csr.

            variable_filter, term_filter: Parameters that can be used for
                partial discretization or assembly. The usage of these terms is
                currently unclear. Use with care.

        Examples
        --------

        For reference, the following is an example of the contents of
        pp.COUPLING_DISCRETIZATION, annotated for the code below:

            d[pp.COUPLING_DISCRETIZATION] = {
                "scalar_coupling_term": {                           <-- coupling_key
                    g_h: ("pressure", "diffusion"),                 <-- (primary_var_key,
                                                                         primary_term_key)
                    g_l: ("pressure", "diffusion"),                 <-- (secondary_var_key,
                                                                         secondary_term_key)
                    e: (
                        "mortar_pressure",                          <-- edge_var_key
                        pp.RobinCoupling("flow", pp.Mpfa("flow"),   <-- edge_discr
                    ),
                },
            }

        """
        for combination in self._grid_variable_term_combinations:
            # Only CouplingVariableTerms are considered
            if isinstance(combination, GridVariableTerm):
                continue
            # Check if this is filtered out
            if not filt.filter(
                grids=[combination.coupling],
                variables=[
                    combination.primary,
                    combination.secondary,
                    combination.edge,
                ],
                terms=[combination.term],
            ):
                continue

            g_primary, g_secondary, e = combination.coupling
            data_secondary = self.gb.node_props(g_secondary)
            data_primary = self.gb.node_props(g_primary)
            data_edge = self.gb.edge_props(e)

            term_key = combination.term
            coupling = data_edge[pp.COUPLING_DISCRETIZATION][term_key]

            # Get edge coupling discretization
            edge_vals: Tuple[str, Any] = coupling.get(e)
            edge_var_key, edge_discr = edge_vals

            # Global block index associated with this edge variable
            edge_idx: int = self._dof_manager.block_dof[(e, edge_var_key)]

            # Get variable name and block index of the primary grid variable.
            primary_vals: Tuple[str, str] = coupling.get(g_primary, None)
            if primary_vals is None:
                # An empty identifying string will create no problems below.
                primary_var_key = ""
                primary_term_key = ""
                # If the primary variable index is None, this signifies that
                # the primary grid variable index is not active
                primary_idx = None

                # If operation is 'assemble', set mat_key_primary to None here
                # and throw and error when the 'matrix' dictionary is accessed.
                mat_key_primary = None
            else:
                # Name of the relevant variable on the primary grid
                primary_var_key, primary_term_key = primary_vals

                # Global index associated with the primary variable
                primary_idx = self._dof_manager.block_dof.get(
                    (g_primary, primary_var_key)
                )

                # Also define the key to access the matrix of the discretization of
                # the primary variable on the primary node.
                mat_key_primary = self._variable_term_key(
                    primary_term_key, primary_var_key, primary_var_key
                )
            # Do similar operations for the secondary variable.
            secondary_vals: Tuple[str, str] = coupling.get(g_secondary, None)
            if secondary_vals is None:
                secondary_var_key = ""
                secondary_term_key = ""
                secondary_idx = None

                # If operation is 'assemble', set mat_key_secondary to None here
                # and throw and error when the 'matrix' dictionary is accessed.
                mat_key_secondary = None
            else:
                secondary_var_key, secondary_term_key = secondary_vals

                secondary_idx = self._dof_manager.block_dof.get(
                    (g_secondary, secondary_var_key)
                )
                # Also define the key to access the matrix of the discretization of
                # the secondary variable on the secondary node.
                mat_key_secondary = self._variable_term_key(
                    secondary_term_key, secondary_var_key, secondary_var_key
                )

            # Key to the matrix dictionary used to access this coupling
            # discretization.
            mat_key = self._variable_term_key(
                term_key, edge_var_key, secondary_var_key, primary_var_key
            )

            # Now there are three options (and a fourth, invalid one):
            # The standard case is that both secondary and primary variables
            # are used in the coupling. Alternatively, only one of the primary or secondary is
            # used. The fourth alternative, none of them are active, is not
            # considered valid, and raises an error message.
            if primary_idx is not None and secondary_idx is not None:
                if operation == "discretize":
                    edge_discr.discretize(
                        g_primary, g_secondary, data_primary, data_secondary, data_edge
                    )

                elif operation == "update_discretization":
                    edge_discr.discretize(
                        g_primary, g_secondary, data_primary, data_secondary, data_edge
                    )

                elif operation == "assemble":
                    # Assign a local matrix, which will be populated with the
                    # current state of the local system.
                    # Local here refers to the variable and term on the two
                    # nodes, together with the relavant mortar variable and term
                    # Associate the first variable with primary, the second with
                    # secondary, and the final with edge.
                    if not assemble_rhs_only:
                        loc_mat, _ = self._assign_matrix_vector(
                            self._dof_manager.full_dof[
                                [primary_idx, secondary_idx, edge_idx]
                            ],
                            sps_matrix,
                        )
                    # Pick out the discretizations on the primary and secondary node
                    # for the relevant variables.
                    # There should be no contribution or modification of the
                    # [0, 1] and [1, 0] terms, since the variables are only
                    # allowed to communicate via the edges.
                    if mat_key_primary:
                        loc_mat[0, 0] = matrix[mat_key_primary][  # type:ignore
                            primary_idx, primary_idx
                        ]
                    else:
                        raise ValueError(
                            f"No discretization found on the primary grid "
                            f"of dimension {g_primary.dim}, for the "
                            f"coupling term {term_key}."
                        )
                    if mat_key_secondary:
                        loc_mat[1, 1] = matrix[mat_key_secondary][  # type:ignore
                            secondary_idx, secondary_idx
                        ]
                    else:
                        raise ValueError(
                            f"No discretization found on the secondary grid "
                            f"of dimension {g_secondary.dim}, for the "
                            f"coupling term {term_key}."
                        )

                        # Pick out the discretizations on the primary and secondary node
                        # for the relevant variables.
                        # There should be no contribution or modification of the
                        # [0, 1] and [1, 0] terms, since the variables are only
                        # allowed to communicate via the edges.
                        if mat_key_primary:
                            loc_mat[0, 0] = matrix[mat_key_primary][  # type:ignore
                                primary_idx, primary_idx
                            ]
                        else:
                            raise ValueError(
                                f"No discretization found on the primary grid "
                                f"of dimension {g_primary.dim}, for the "
                                f"coupling term {term_key}."
                            )
                        if mat_key_secondary:
                            loc_mat[1, 1] = matrix[mat_key_secondary][  # type:ignore
                                secondary_idx, secondary_idx
                            ]
                        else:
                            raise ValueError(
                                f"No discretization found on the secondary grid "
                                f"of dimension {g_secondary.dim}, for the "
                                f"coupling term {term_key}."
                            )

                    # Run the discretization, and assign the resulting matrix
                    # to a temporary construct
                    if assemble_matrix_only:
                        tmp_mat = edge_discr.assemble_matrix(
                            g_primary,
                            g_secondary,
                            data_primary,
                            data_secondary,
                            data_edge,
                            loc_mat,
                        )
                    elif assemble_rhs_only:
                        loc_rhs = edge_discr.assemble_rhs(
                            g_primary,
                            g_secondary,
                            data_primary,
                            data_secondary,
                            data_edge,
                            None,  # The local matrix should not be used
                        )
                    else:
                        tmp_mat, loc_rhs = edge_discr.assemble_matrix_rhs(
                            g_primary,
                            g_secondary,
                            data_primary,
                            data_secondary,
                            data_edge,
                            loc_mat,
                        )
                    if not assemble_rhs_only:
                        # The edge column and row should be assigned to mat_key
                        matrix[mat_key][  # type:ignore
                            edge_idx, (primary_idx, secondary_idx, edge_idx)
                        ] = tmp_mat[2, (0, 1, 2)]
                        matrix[mat_key][  # type:ignore
                            (primary_idx, secondary_idx), edge_idx
                        ] = tmp_mat[(0, 1), 2]
                        # Also update the discretization on the primary and secondary
                        # nodes
                        matrix[mat_key_primary][  # type:ignore
                            primary_idx, primary_idx
                        ] = tmp_mat[0, 0]
                        matrix[mat_key_secondary][  # type:ignore
                            secondary_idx, secondary_idx
                        ] = tmp_mat[1, 1]

                    if not assemble_matrix_only:
                        # Finally take care of the right hand side
                        rhs[mat_key][[primary_idx, secondary_idx, edge_idx]] += loc_rhs

            elif primary_idx is not None:
                # TODO: Term filters are not applied to this case
                # secondary_idx is None
                # The operation is a simplified version of the full option above.
                if operation in ("discretize", "update_discretization"):
                    edge_discr.discretize(g_primary, data_primary, data_edge)
                elif operation == "assemble":

                    loc_mat, _ = self._assign_matrix_vector(
                        self._dof_manager.full_dof[[primary_idx, edge_idx]], sps_matrix
                    )
                    loc_mat[0, 0] = matrix[mat_key_primary][  # type:ignore
                        primary_idx, primary_idx
                    ]

                    if assemble_matrix_only:
                        tmp_mat = edge_discr.assemble_matrix(
                            g_primary, data_primary, data_edge, loc_mat
                        )

                    elif assemble_rhs_only:
                        loc_rhs = edge_discr.assemble_rhs(
                            g_primary, data_primary, data_edge, loc_mat
                        )

                    else:
                        tmp_mat, loc_rhs = edge_discr.assemble_matrix_rhs(
                            g_primary, data_primary, data_edge, loc_mat
                        )
                    if not assemble_rhs_only:
                        matrix[mat_key][  # type:ignore
                            edge_idx, (primary_idx, edge_idx)
                        ] = tmp_mat[1, (0, 1)]
                        matrix[mat_key][primary_idx, edge_idx] = tmp_mat[  # type:ignore
                            0, 1
                        ]

                        # Also update the discretization on the primary and secondary
                        # nodes
                        matrix[mat_key_primary][  # type:ignore
                            primary_idx, primary_idx
                        ] = tmp_mat[0, 0]

                    if not assemble_matrix_only:
                        rhs[mat_key][[primary_idx, edge_idx]] += loc_rhs

            elif secondary_idx is not None:
                # TODO: Term filters are not applied to this case
                # primary_idx is None
                # The operation is a simplified version of the full option above.
                if operation in ("discretize", "update_discretization"):
                    edge_discr.discretize(g_secondary, data_secondary, data_edge)
                elif operation == "assemble":

                    loc_mat, _ = self._assign_matrix_vector(
                        self._dof_manager.full_dof[[secondary_idx, edge_idx]],
                        sps_matrix,
                    )
                    loc_mat[0, 0] = matrix[mat_key_secondary][  # type:ignore
                        secondary_idx, secondary_idx
                    ]
                    if assemble_matrix_only:
                        tmp_mat = edge_discr.assemble_matrix(
                            g_secondary, data_secondary, data_edge, loc_mat
                        )

                    elif assemble_rhs_only:
                        loc_rhs = edge_discr.assemble_rhs(
                            g_secondary, data_secondary, data_edge, loc_mat
                        )

                    else:
                        tmp_mat, loc_rhs = edge_discr.assemble_matrix_rhs(
                            g_secondary, data_secondary, data_edge, loc_mat
                        )

                    if not assemble_rhs_only:
                        matrix[mat_key][  # type:ignore
                            edge_idx, (secondary_idx, edge_idx)
                        ] = tmp_mat[1, (0, 1)]
                        matrix[mat_key][  # type:ignore
                            secondary_idx, edge_idx
                        ] = tmp_mat[0, 1]

                        # Also update the discretization on the primary and secondary
                        # nodes
                        matrix[mat_key_secondary][  # type:ignore
                            secondary_idx, secondary_idx
                        ] = tmp_mat[0, 0]
                    if not assemble_matrix_only:
                        rhs[mat_key][[secondary_idx, edge_idx]] += loc_rhs

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
            # These restrictions may be loosened somewhat in the future, but a
            # general coupling between different edges will not be implemented.
            if operation == "assemble" and edge_discr.edge_coupling_via_high_dim:
                for other_edge, data_other in self.gb.edges_of_node(g_primary):

                    # Skip the case where the primary and secondary edge is the same
                    if other_edge == e:
                        continue

                    # Avoid coupling between mortar grids of different dimensions.
                    if data_other["mortar_grid"].dim != data_edge["mortar_grid"].dim:
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
                    oi = self._dof_manager.block_dof.get(
                        (other_edge, edge_var_key), None
                    )
                    if oi is None:
                        continue

                    assemble_matrix, assemble_rhs = True, True
                    if assemble_matrix_only:
                        assemble_rhs = False
                    if assemble_rhs_only:
                        assemble_matrix = False

                    # Assign a local matrix, which will be populated with the
                    # current state of the local system.
                    # Local here refers to the variable and term on the two
                    # nodes, together with the relavant mortar variable and term
                    # Associate the first variable with primary, the second with
                    # secondary, and the final with edge.
                    if assemble_matrix:
                        loc_mat, _ = self._assign_matrix_vector(
                            self._dof_manager.full_dof[[primary_idx, edge_idx, oi]],
                            sps_matrix,
                        )

                        (
                            tmp_mat,
                            loc_rhs,
                        ) = edge_discr.assemble_edge_coupling_via_high_dim(
                            g_primary,
                            data_primary,
                            e,
                            data_edge,
                            other_edge,
                            data_other,
                            loc_mat,
                            assemble_matrix=assemble_matrix,
                            assemble_rhs=assemble_rhs,
                        )
                        matrix[mat_key][edge_idx, oi] = tmp_mat[1, 2]  # type:ignore
                    else:
                        loc_mat = None
                        (
                            tmp_mat,
                            loc_rhs,
                        ) = edge_discr.assemble_edge_coupling_via_high_dim(
                            g_primary,
                            data_primary,
                            e,
                            data_edge,
                            other_edge,
                            data_other,
                            loc_mat,
                            assemble_matrix=assemble_matrix,
                            assemble_rhs=assemble_rhs,
                        )

                    rhs[mat_key][edge_idx] += loc_rhs[1]  # type:ignore

            if operation == "assemble" and edge_discr.edge_coupling_via_low_dim:
                for other_edge, data_other in self.gb.edges_of_node(g_secondary):

                    # Skip the case where the primary and secondary edge is the same
                    if other_edge == e:
                        continue

                    if data_other["mortar_grid"].dim != data_edge["mortar_grid"].dim:
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
                    oi = self._dof_manager.block_dof.get(
                        (other_edge, edge_var_key), None
                    )
                    if oi is None:
                        continue

                    # Assign a local matrix, which will be populated with the
                    # current state of the local system.
                    # Local here refers to the variable and term on the two
                    # nodes, together with the relavant mortar variable and term
                    # Associate the first variable with primary, the second with
                    # secondary, and the final with edge.
                    loc_mat, _ = self._assign_matrix_vector(
                        self._dof_manager.full_dof[[secondary_idx, edge_idx, oi]],
                        sps_matrix,
                    )
                    assemble_matrix, assemble_rhs = True, True
                    if assemble_matrix_only:
                        assemble_rhs = False
                    if assemble_rhs_only:
                        assemble_matrix = False

                    (tmp_mat, loc_rhs) = edge_discr.assemble_edge_coupling_via_low_dim(
                        g_secondary,
                        data_secondary,
                        data_edge,
                        data_other,
                        loc_mat,
                        assemble_matrix=assemble_matrix,
                        assemble_rhs=assemble_rhs,
                    )
                    matrix[mat_key][edge_idx, oi] = tmp_mat[1, 2]  # type:ignore
                    rhs[mat_key][edge_idx] += loc_rhs[1]

    @pp.time_logger(sections=module_sections)
    def _identify_variable_combinations(self) -> None:
        """
        Initialize local matrices for all combinations of variables and operators.

        The function serves two purposes:
            1. Identify all variables and their discretizations defined on individual nodes
               and edges in the GridBucket

        At the end of this function, self has been assigned variable_combinations.
        This is a list of strings that define all couplings ofvariables  found in the problem
        specification. This includes both diagonal terms in the system block matrix,
        coupling terms within nodesand edges, and couplings between edges and nodes.

        """
        # Implementation note: To fully understand the structure of this function
        # it is useful to consider an example of a data dictionary with declared
        # primary variables and discretization operators.
        # The function needs to dig deep into the dictionaries used in these
        # declarations, thus the code is rather involved.

        # Store all combinations of variable pairs (e.g. row-column indices in
        # the global system matrix), and identifiers of discretization operations
        # (e.g. advection or diffusion).
        # Note: This list is common for all nodes / edges.
        # This list is used to access discretization matrices for different
        # variables and terms
        variable_combinations: List[str] = []

        # Combinations of grids-like features (grid, edge, edge-grid coupling),
        # variables and terms. Used to track discretizations, and which grids and
        # variables they act on. Also used to filter discretizations for partial
        # discretizations etc.
        # IMPLEMENTATION NOTE: variable_combinations and grid_variable_term_combinations
        # are sort of overlapping in the information they contained (the former is
        # a subset of the latter), but for implementation convenience it is useful
        # to keep both.
        grid_variable_term_combinations: List[
            Union[CouplingVariableTerm, GridVariableTerm]
        ] = []

        # Loop over all nodes in the grid bucket, identify its local and active
        # variables.
        for g, d in self.gb:

            # If for some reason there are no primary variables defined for this grid,
            # skip it.
            if pp.PRIMARY_VARIABLES not in d:
                continue

            # Loop over variables, count dofs and identify variable-term
            # combinations internal to the node
            for local_var in d[pp.PRIMARY_VARIABLES]:

                # Identify all defined discretization terms for this variable.
                # Do a second loop over the variables of the grid, the combination
                # of the two variables gives us all coupling terms (e.g. an off-diagonal
                # block in the global matrix)
                for other_local_var in d[pp.PRIMARY_VARIABLES]:
                    # We need to identify identify individual discretization terms
                    # defined for this equation. These are identified either by
                    # the variable k (for variable dependence on itself), or the
                    # combination var1_var2 if the variables are mixed
                    merged_vars = self._discretization_key(local_var, other_local_var)

                    # Get hold of the discretization operators defined for this
                    # node / edge; we really just need the keys in the
                    # discretization map.
                    # The default assumption is that no discretization has
                    # been defined, in which case we do nothing.
                    discr: Dict[str, Any] = d.get(pp.DISCRETIZATION, None)

                    # It may be that there is no discretization specified
                    if discr is None:
                        continue

                    # Loop over all the discretization operations, if any, and
                    # add it to the list of observed variables.
                    # We will take care of duplicates below.
                    terms = discr.get(merged_vars, None)
                    if terms is None:
                        continue

                    for term in terms:
                        variable_combinations.append(
                            self._variable_term_key(term, local_var, other_local_var)
                        )
                        grid_variable_term_combinations.append(
                            GridVariableTerm(g, local_var, other_local_var, term)
                        )

        # Next do the equivalent operation for edges in the grid.
        # Most steps are identical to the operations on the nodes, we comment
        # only on edge-specific aspects; see above loop for more information
        for e, d in self.gb.edges():

            # If for some reason there are no primary variables defined for this edge,
            # skip it.
            if pp.PRIMARY_VARIABLES not in d:
                continue

            for local_var in d[pp.PRIMARY_VARIABLES]:

                # Identify all discretization terms for this variable
                for other_local_var in d[pp.PRIMARY_VARIABLES]:
                    merged_vars = self._discretization_key(local_var, other_local_var)
                    discr = d.get(pp.DISCRETIZATION, None)
                    if discr is None:
                        continue
                    terms = discr.get(merged_vars, None)
                    if terms is None:
                        continue
                    for term in terms.keys():
                        variable_combinations.append(
                            self._variable_term_key(term, local_var, other_local_var)
                        )
                        grid_variable_term_combinations.append(
                            GridVariableTerm(e, local_var, other_local_var, term)
                        )
            # Finally, identify variable combinations for coupling terms.
            # This involves both the neighboring grids
            g_secondary, g_primary = self.gb.nodes_of_edge(e)

            discr = d.get(pp.COUPLING_DISCRETIZATION, None)
            if discr is None:
                continue

            for term, val in discr.items():
                # term identifies the discretization operator (e.g. advection or
                # diffusion), val contains the coupling information

                # Identify this term in the discretization by the variable names
                # on the edge, the variable names of the secondary and primary grid
                # in that order, and finally the term name.
                # There is a tacit assumption here that self.gb.nodes_of_edge return the
                # grids in the same order here and in the assembly. This should be okay.
                # The consequences for the methods if this is no longer the case is unclear.

                # Get the name of the edge variable (it is the first item in a tuple)
                key_edge: str = val.get(e)[0]

                # Get name of the edge variable, if it exists
                key_secondary = val.get(g_secondary)
                if key_secondary is not None:
                    key_secondary = key_secondary[0]

                else:
                    # This can happen if the the coupling is one-sided, e.g.
                    # it does not consider the secondary grid.
                    # An empty string will give no impact on the generated
                    # combination of variable names and discretizaiton terms
                    key_secondary = ""

                key_primary = val.get(g_primary)
                if key_primary is not None:
                    key_primary = key_primary[0]
                else:
                    key_primary = ""

                variable_combinations.append(
                    self._variable_term_key(term, key_edge, key_secondary, key_primary)
                )
                grid_variable_term_combinations.append(
                    CouplingVariableTerm(
                        (g_primary, g_secondary, e),
                        key_edge,
                        key_primary,
                        key_secondary,
                        term,
                    )
                )

        # Store values in self
        self.variable_combinations: List[str] = variable_combinations
        self._grid_variable_term_combinations = grid_variable_term_combinations

    @pp.time_logger(sections=module_sections)
    def update_dof_count(self) -> None:
        """Update the count of degrees of freedom related to a GridBucket.

        The method loops thruogh the defined combinations of grids (standard or mortar)
        and variables, and updates the number of fine-scale degree of freedom for this
        combination. The system size will be updated if the grid has changed or
        (perhaps less realistically) a variable has had its number of dofs per grid
        quantity changed.

        The method will not identify any new variables, for this, the preferred approach
        is to define a new assembler object.

        """
        # Loop over identified grid-varibale combinations
        for key, index in self._dof_manager.block_dof.items():
            # Grid quantity (grid or interface), and variable
            grid, variable = key
            # Get data dictionary - this is slightly different for grid and interface
            if isinstance(grid, pp.Grid):
                d = self.gb.node_props(grid)
            else:  # This is an edge
                d = self.gb.edge_props(grid)
                # Also fetch mortar grid
                grid = d["mortar_grid"]

            # Dofs related to cell
            dof: Dict[str, int] = d[pp.PRIMARY_VARIABLES][variable]
            num_dofs: int = grid.num_cells * dof.get("cells", 0)  # type: ignore

            if isinstance(grid, pp.Grid):
                # Add dofs on faces and nodes, but not on interfaces
                num_dofs += grid.num_faces * dof.get(
                    "faces", 0
                ) + grid.num_nodes * dof.get("nodes", 0)

            # Update local counting
            self._dof_manager.full_dof[index] = num_dofs

    @pp.time_logger(sections=module_sections)
    def _initialize_matrix_rhs(
        self, sps_matrix: Type[csc_or_csr_matrix]
    ) -> Tuple[Dict[str, csc_or_csr_matrix], Dict[str, np.ndarray]]:
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
                individual blocks in the matrix.

        Returns:
            dict: Global system matrices, on block form (one per node/edge per
                variable). There is one item per term (e.g. diffusion/advection)
                per variable.
            dict: Right hand sides. Similar to the system matrix.

        """
        # We will have one discretization matrix per variable
        matrix_dict: Dict[str, Union[csc_or_csr_matrix, np.ndarray]] = {}
        rhs_dict: Dict[str, np.ndarray] = {}

        num_blocks = len(self._dof_manager.full_dof)

        # Uniquify list of variable combinations. Then iterate over all variable
        # combinations and initialize matrices of the right size
        for var in list(set(self.variable_combinations)):

            # Generate a block matrix
            matrix_dict[var] = np.empty((num_blocks, num_blocks), dtype=object)
            rhs_dict[var] = np.empty(num_blocks, dtype=object)

            # Loop over all blocks, initialize the diagonal block.
            # We could also initialize off-diagonal blocks, however, this turned
            # out to be computationally expensive.
            for di in np.arange(num_blocks):
                # Initilize the block diagonal parts, this is useful for the bmat done
                # at the end of assemble_matrix_rhs to know the correct shape of the
                # full matrix
                matrix_dict[var][di, di] = sps_matrix(
                    (self._dof_manager.full_dof[di], self._dof_manager.full_dof[di])
                )
                rhs_dict[var][di] = np.zeros(self._dof_manager.full_dof[di])

        return matrix_dict, rhs_dict

    @staticmethod
    @pp.time_logger(sections=module_sections)
    def _assign_matrix_vector(
        dof: np.ndarray, sps_matrix: Type[csc_or_csr_matrix], create_matrix: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """ Assign a block matrix and vector with specified number of dofs per block"""
        num_blocks = dof.size

        if create_matrix:
            matrix = np.empty((num_blocks, num_blocks), dtype=object)
        rhs = np.empty(num_blocks, dtype=object)

        for ri in range(num_blocks):
            rhs[ri] = np.zeros(dof[ri])
            if create_matrix:
                for ci in range(num_blocks):
                    matrix[ri, ci] = sps_matrix((dof[ri], dof[ci]))

        if create_matrix:
            return matrix, rhs
        else:
            return rhs

    @pp.time_logger(sections=module_sections)
    def assemble_operator(self, keyword: str, operator_name: str) -> sps.spmatrix:
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

        @pp.time_logger(sections=module_sections)
        def _get_operator(data, _keyword, _operator_name):
            loc_disc = data[pp.DISCRETIZATION_MATRICES].get(_keyword, None)
            if loc_disc is None:  # Return if keyword is not found
                return None
            loc_op = loc_disc.get(_operator_name, None)
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

    @pp.time_logger(sections=module_sections)
    def assemble_parameter(self, keyword: str, parameter_name: str) -> np.ndarray:
        """
        Assemble a global parameter from the local parameters defined on
        the nodes or edges of a grid bucket. The global parameter is a nd-vector
        of the stacked local parameters.

        Parameters:
            keyword (string): Keyword to access the dictionary
                d[pp.PARAMETERS][keyword] for which the parameters are stored.
            parameter_name (string): keyword of the parameter. Will access
                d[pp.DISCRETIZATION_MATRICES][keyword][parameter].

        Returns:
            Operator (np.ndarray): Global parameter.

        """
        parameter = []
        for _, d in self.gb:
            parameter.append(d[pp.PARAMETERS][keyword][parameter_name])
        return np.hstack(parameter)

    @pp.time_logger(sections=module_sections)
    def _local_variables(self, d: Dict) -> Dict[str, Dict[str, int]]:
        """Find variables defined in a data dictionary, and do intersection
        with defined active variables.

        If no active variables are specified, returned all declared variables.

        Parameters:
            d (dict): Data dictionary defined on a GridBucket node or edge

        Returns:
            dict: With variable names and information (#dofs of various kinds), as
                specified by user, but possibly restricted to the active variables

        """

        # Active variables
        return d.get(pp.PRIMARY_VARIABLES, None)

    @pp.time_logger(sections=module_sections)
    def distribute_variable(
        self, values: np.ndarray, variable_names: Optional[List[str]] = None
    ) -> None:
        """Distribute a vector to the nodes and edges in the GridBucket.

        The intended use is to split a multi-physics solution vector into its
        component parts.

        Parameters:
            values (np.array): Vector to be split. It is assumed that it corresponds
                to the ordering implied in block_dof and full_dof, e.g. that it is
                the solution of a linear system assembled with the assembler.
            variable_names (list of str, optional): Names of the variable to be
                distributed. If not provided, all variables found in block_dof
                will be distributed

        """
        self._dof_manager.distribute_variable(values, variable_names)

    @pp.time_logger(sections=module_sections)
    def num_dof(self) -> int:
        """Get total number of unknowns of the identified variables.

        Returns:
            int: Number of unknowns. Size of solution vector.
        """
        return self._dof_manager.num_dofs()

    @pp.time_logger(sections=module_sections)
    def variables_of_grid(
        self, g: Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]
    ) -> List[str]:
        """Get all variables defined for a given grid or edge.

        Args:
            g (Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]): Target grid, or an edge

        Returns:
            List[str]: List of all variables known for this entity.

        """
        return [key[1] for key in self._dof_manager.block_dof.keys() if key[0] == g]

    @pp.time_logger(sections=module_sections)
    def __str__(self) -> str:
        names = [key[1] for key in self._dof_manager.block_dof.keys()]
        unique_vars = list(set(names))
        s = (
            f"Assembler object on a GridBucket with {self.gb.num_graph_nodes()} "
            f"subdomains and {self.gb.num_graph_edges()} interfaces.\n"
            f"Total number of degrees of freedom: {self.num_dof()}\n"
            "Total number of subdomain and interface variables:"
            f"{len(self._dof_manager.block_dof)}\n"
            f"Variable names: {unique_vars}"
        )

        return s

    @pp.time_logger(sections=module_sections)
    def __repr__(self) -> str:
        s = (
            f"Assembler objcet with in total {self.num_dof()} dofs"
            f" on {len(self._dof_manager.block_dof)} subdomain and interface variables.\n"
            f"Maximum grid dimension: {self.gb.dim_max()}.\n"
            f"Minimum grid dimension: {self.gb.dim_min()}.\n"
        )
        for dim in range(self.gb.dim_max(), self.gb.dim_min() - 1, -1):
            s += f"In dimension {dim}: {len(self.gb.grids_of_dimension(dim))} grids.\n"
            unique_vars = {
                key[1]
                for key in self._dof_manager.block_dof.keys()
                if not isinstance(key[0], tuple) and key[0].dim == dim
            }
            s += f"All variables present in dimension {dim}: {unique_vars}\n"

            # Also check if some subdomains of this dimension have a subset of the
            # variables defined on the totality of the subdomains

            # List of found special (subset) variable combinations
            found_special_var_combination: List[Set[str]] = []
            # Loop over all grids of this dimension
            for g in self.gb.grids_of_dimension(dim):
                # All variables on this subdomain
                var = set(self.variables_of_grid(g))
                # Check if this is a subset of the full variable list on this dimension
                if var.issubset(unique_vars):
                    # We will only report each subset variable definition once.
                    # If this subset hasn't already been reported, report it.
                    already_reported = np.any(
                        [var == spec for spec in found_special_var_combination]
                    )
                    if not already_reported:
                        found_special_var_combination.append(var)
                        s += (
                            f"Variable subset on at least one subdomain in "
                            f"dimension {dim}: {var}\n"
                        )

        for dim in range(self.gb.dim_max(), self.gb.dim_min(), -1):
            unique_vars = {
                var
                for g in self.gb.grids_of_dimension(
                    dim
                )  # For each grid of dimension dim
                for e, _ in self.gb.edges_of_node(g)  # for each edge of that grid
                if self.gb.nodes_of_edge(e)[1]
                == g  # such that the edge neighbors a lower-dimensional grid
                for var in self.variables_of_grid(e)  # get all variables on that edge
            }

            s += (
                f"All variables present on edges between dimensions {dim} and {dim-1}: "
                f"{unique_vars}\n"
            )

            # Also check if some subdomains of this dimension have a subset of the
            # variables defined on the totality of the subdomains

            # List of found special (subset) variable combinations
            found_special_var_combination = []
            for g in self.gb.grids_of_dimension(dim):
                for e, _ in self.gb.edges_of_node(g):
                    var = set(self.variables_of_grid(e))
                    # Check if this is a subset of the full variable list on this dimension
                    if var.issubset(unique_vars):
                        # We will only report each subset variable definition once.
                        # If this subset hasn't already been reported, report it.
                        already_reported = np.any(
                            [var == spec for spec in found_special_var_combination]
                        )
                        if not already_reported:
                            found_special_var_combination.append(var)
                            s += (
                                f"Variable subset on at least one interface between "
                                f"dimension {dim} and {dim-1}: {var}\n"
                            )

        return s
