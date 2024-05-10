"""Contains the EquationSystem, managing variables and equations for a system modelled
using the AD framework.

"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Sequence, Union, overload

import numpy as np
import scipy.sparse as sps
from typing_extensions import TypeAlias

import porepy as pp

from . import _ad_utils
from .operators import MixedDimensionalVariable, Operator, Variable

__all__ = ["EquationSystem"]


# For Python3.8, a direct definition of type aliases with list is apparently not posible
# (DomainList = Union[list[pp.Grid], list[pp.MortarGrid]]]), the same applies to dict
# and presumably tuple etc. As a temporary solution, we use a TypeAlias together with a
# string representation of the type. This can be replaced with the more straightforward
# definition when we drop support for Python3.8.
DomainList: TypeAlias = "Union[list[pp.Grid], list[pp.MortarGrid]]"
"""A union type representing a list of grids or mortar grids.
This is *not* a list of GridLike, as that would allow a list of mixed grids and
mortar grids."""

VariableList: TypeAlias = (
    "Union[list[str], list[Variable], list[MixedDimensionalVariable]]"
)
"""A union type representing variables

Variables are defined through either
    - names (:class:`str`),
    - multiple :class:`~porepy.numerics.ad.operators.Variable` or
    - :class:`~porepy.numerics.ad.operators.MixedDimensionalVariable`.

This type is accepted as input to various methods and parsed to a list of
:class:`~porepy.numerics.ad.operators.Variable` using
:meth:`~porepy.numerics.ad.equation_system.EquationSystem._parse_variable_list`.

"""

EquationList: TypeAlias = "Union[list[str], list[Operator]]"
"""A union type representing equations through either names (:class:`str`), or
:class:`~porepy.numerics.ad.operators.Operator`.

This type is accepted as input to various methods and parsed to a list of
:class:`~porepy.numerics.ad.operators.Operator` using
:meth:`~porepy.numerics.ad.equation_system.EquationSystem._parse_equations`.

"""

EquationRestriction: TypeAlias = "dict[Union[str, Operator], DomainList]"
"""A dictionary mapping equations to a list of domains on which the equation should be
applied.

The keys of the dictionary can be either the name of the equation, or the equation
itself represented as an :class:`~porepy.numerics.ad.operators.Operator`. The values of
the dictionary are DomainList, i.e., a list of grids or mortar grids.

This type is accepted as input to various methods and parsed to an index set
representing a restricted image of the equation by
:meth:`~porepy.numerics.ad.equation_system.EquationSystem._parse_equations`.

"""
# IMPLEMENTATION NOTE: EK could not find an elegant way to represent all types of
# equation input in a single type. The problem is that, while strings and Operators
# are naturally wrapped in lists, even if there is only one item, restrictions of
# equations are most naturally represented as a dictionary. This means iteration over
# equations and restrictions must be handled separately, as is now done in
#  _parse_equations(). To avoid this, we could have introduced
#
#   EquationType = Union[str, Operator, dict[Union[str, Operator], DomainList]]
#
# and allowed for list[EquationType] as input to various methods. This does however
# require passing a list of dictionaries to _parse_equations(), which was very
# awkward from the user side when EK tried it. The current solution seems like a fair
# compromise, and it has the positive side of being explicit on the difference between
# equations and restrictions of equations, but it does not feel like a fully
# satisfactory solution.

GridEntity = Literal["cells", "faces", "nodes"]
"""A union type representing a grid entity, either a cell, face or node.
This is used to define the domain of a variable or an equation,
i.e. whether it is defined on cells, faces or nodes.
"""


class EquationSystem:
    """Represents an equation system, modelled by AD variables and equations in AD form.

    This class provides functionalities to create and manage variables, as well as
    managing equations on the form of :class:`~porepy.numerics.ad.operators.Operator`.
    It further provides functions to assemble subsystems and using subsets of equations
    and variables.

    Note:
        As of now, the system matrix (Jacobian) is assembled with respect to ALL
        variables and then the columns belonging to the requested subset of variables
        and grids are sliced out and returned. This will be optimized with minor changes
        to the AD operator class and its recursive forward AD mode in the future.

        Currently, this class optimizes the block structure of the Jacobian only
        regarding the subdomains and interfaces. A more localized optimization (e.g.
        cell-wise for equations without spatial differential operators) is not
        performed.

    """

    admissible_dof_types: tuple[
        Literal["cells"], Literal["faces"], Literal["nodes"]
    ] = ("cells", "faces", "nodes")
    """A set denoting admissible types of local DOFs for variables.

    - nodes: DOFs per grid node.
    - cells: DOFs per grid cell.
    - faces: DOFS per grid face.

    """

    def __init__(self, mdg: pp.MixedDimensionalGrid) -> None:
        ### PUBLIC
        self.mdg: pp.MixedDimensionalGrid = mdg
        """Mixed-dimensional domain passed at instantiation."""

        self.assembled_equation_indices: dict[str, np.ndarray] = dict()
        """Contains the row indices in the last assembled (sub-) system for a given
        equation name (key).

        This dictionary changes with every call to any assemble-method, provided the
        method is invoked to assemble *both* Jacobian matrix *and* the residual vector
        (argument ``evaluate_jacobian=True``) If only the residual vector is assembled,
        the indices is not updated.
        """

        ### PRIVATE

        self._equations: dict[str, Operator] = dict()
        """Contains references to equations in AD operator form for a given name (key).

        Private to avoid users setting equations directly and circumventing the current
        set-method which includes information about the image space.

        """

        self._equation_image_space_composition: dict[
            str, dict[pp.GridLike, np.ndarray]
        ] = dict()
        """Definition of image space for all equations.

        Contains for every equation name (key) a dictionary, which provides again for
        every involved grid (key) the indices of equations expressed through the
        equation operator. The ordering of the items in the grid-array dictionaries is
        consistent with the remaining PorePy framework. The ordering is local to the
        equation, so it can be used to slice an eqution prior to concatenation of
        equations into a global matrix.

        """

        self._equation_image_size_info: dict[str, dict[GridEntity, int]] = dict()
        """Contains for every equation name (key) the number of equations per grid
        entity.

        """

        self._variables: dict[int, Variable] = dict()
        """Dictionary mapping variable IDs to the atomic variables created and managed
        by this instance.

        Variables contained here are ordered chronologically in terms of
        instantiation. It does not reflect the order of DOFs, which is to some degree
        optimized.

        A Variable is uniquely identified by its name and domain, stored as attributes
        of the Variable object.

        Implementation-wise it is uniquely identified by its ID.

        """

        self._Schur_complement: Optional[tuple] = None
        """Contains block matrices and the rhs of the last assembled Schur complement.

        """

        self._variable_numbers: dict[int, int] = dict()
        """A Map between a variable's ID and its index in the system vector.

        This is an optimized structure, meaning the order of entries is created in
        :meth:`_cluster_dofs_gridwise`.

        """

        self._variable_num_dofs: np.ndarray = np.array([], dtype=int)
        """Array containing the number of DOFS per block number.

        The block number corresponds to this array's indexation, see also
        attr:`_variable_numbers`.

        """

        self._variable_dof_type: dict[int, dict[GridEntity, int]] = dict()
        """Dictionary mapping from variable IDs to the type of DOFs per variable.

        The type is given as a dictionary with keys 'cells', 'faces' or 'nodes',
        and integer values denoting the number of DOFs per grid entity.

        """

    def SubSystem(
        self,
        equation_names: Optional[EquationList] = None,
        variable_names: Optional[VariableList] = None,
    ) -> EquationSystem:
        """Creates an :class:`EquationSystem` for a given subset of equations and
        variables.

        Currently only subsystems containing *whole* equations and variables in the
        mixed-dimensional sense can be created. Restrictions of equations to subdomains
        is not supported.

        Parameters:
            equation_names: Names of equations for the new subsystem. If None, all
                equations known to the :class:`EquationSystem` are used.
            variable_names: Names of known variables for the new subsystem. If None, all
                variables known to the :class:`EquationSystem` are used.

        Returns:
            A new instance of :class:`EquationSystem`. The subsystem equations and
            variables are ordered as imposed by this systems's order.

        Raises:
            ValueError: if passed names are not among created variables and set
                equations.

        """
        # Parsing of input arguments.
        equations = list(self._parse_equations(equation_names).keys())
        variables = self._parse_variable_type(variable_names)

        # Check that the requested equations and variables are known to the system.
        known_equations = set(self._equations.keys())
        unknown_equations = set(equations).difference(known_equations)
        if len(unknown_equations) > 0:
            raise ValueError(f"Unknown variable(s) {unknown_equations}.")
        unknown_variables = set(variables).difference(self.variables)
        if len(unknown_variables) > 0:
            raise ValueError(f"Unknown variable(s) {unknown_variables}.")

        # Create the new subsystem.
        new_equation_system = EquationSystem(self.mdg)

        # IMPLEMENTATION NOTE: This method imitates the variable creation and equation
        # setting procedures by calling private methods and accessing private
        # attributes. This should be acceptable since this is a factory method.

        # Loop over known variables to preserve DOF order.
        for variable in self.variables:
            if variable in variables:
                # Update variables in subsystem.
                new_equation_system._variables[variable.id] = variable

                # Update variable numbers in subsystem.
                new_equation_system._variable_dof_type[variable.id] = (
                    self._variable_dof_type[variable.id]
                )

                # Create dofs in subsystem.
                new_equation_system._append_dofs(variable)

        new_equation_system._cluster_dofs_gridwise()
        # Loop over known equations to preserve row order.
        for name in known_equations:
            if name in equations:
                equation = self._equations[name]
                image_info = self._equation_image_size_info[name]
                image_composition = self._equation_image_space_composition[name]
                # et the information produced in set_equations directly.
                new_equation_system._equation_image_space_composition.update(
                    {name: image_composition}
                )
                new_equation_system._equation_image_size_info.update({name: image_info})
                new_equation_system._equations.update({name: equation})

        return new_equation_system

    @property
    def equations(self) -> dict[str, Operator]:
        """Dictionary containing names of operators (keys) and operators (values), which
        have been set as equations in this system.

        """
        return self._equations

    @property
    def variables(self) -> list[Variable]:
        """List containing all :class:`~porepy.numerics.ad.Variable`s known to this
        system.

        """
        return [var for var in self._variables.values()]

    @property
    def variable_domains(self) -> list[pp.GridLike]:
        """List containing all domains where at least one variable is defined."""
        domains = set()
        for var in self.variables:
            domains.add(var.domain)
        return list(domains)

    ### Variable management ------------------------------------------------------------

    def md_variable(
        self, name: str, grids: Optional[DomainList] = None
    ) -> MixedDimensionalVariable:
        """Create a mixed-dimensional variable for a given name-domain list combination.

        Parameters:
            name (str): Name of the mixed-dimensional variable.
            grids (optional): List of grids where the variable is defined. If None
                (default), all grids where the variable is defined are used.

        Returns:
            A mixed-dimensional variable.

        Raises:
            ValueError: If variables name exist on both grids and interfaces and domain
                type is not specified (grids is None).

        """
        if grids is None:
            variables = [var for var in self.variables if var.name == name]
            # We don't allow combinations of variables with different domain types
            # in a md variable.
            heterogeneous_domain = False
            if isinstance(variables[0].domain, pp.Grid):
                heterogeneous_domain = any(
                    [isinstance(var.domain, pp.MortarGrid) for var in variables]
                )
            elif isinstance(variables[0].domain, pp.MortarGrid):
                heterogeneous_domain = any(
                    [isinstance(var.domain, pp.Grid) for var in variables]
                )
            else:
                raise ValueError("Unknown domain type for variable")
            if heterogeneous_domain:
                raise ValueError(
                    f"Variable {name} is defined on multiple domain types."
                )
        else:
            variables = [
                var
                for var in self.variables
                if var.name == name and var.domain in grids
            ]
        return MixedDimensionalVariable(variables)

    def create_variables(
        self,
        name: str,
        dof_info: Optional[dict[GridEntity, int]] = None,
        subdomains: Optional[list[pp.Grid]] = None,
        interfaces: Optional[list[pp.MortarGrid]] = None,
        tags: Optional[dict[str, Any]] = None,
    ) -> MixedDimensionalVariable:
        """Creates new variables according to specifications.

        This method does not assign any values to the variable. This has to be done in a
        subsequent step (e.g. using :meth:`set_variable_values`).

        Examples:
            An example on how to define a pressure variable with cell-wise one DOF
            (default) on **all** subdomains and **no** interfaces would be

            .. code:: Python

                p = ad_system.create_variables('pressure', subdomains=mdg.subdomains())

        Parameters:
            name: Name of the variable.
            dof_info: Dictionary containing information about number of DOFs per
                admissible type. Defaults to ``{'cells':1}``.
            subdomains (optional): List of subdomains on which the variable is defined.
                If None, then it will not be defined on any subdomain.
            interfaces (optional): list of interfaces on which the variable is defined.
                If None, then it will not be defined on any interface.
            tags (optional): dictionary containing tags for the variables. The tags are
                assigned to all variables created by this method and can be updated
                using :meth:`update_variable_tags`.

        Returns:
            A mixed-dimensional variable with above specifications.

        Raises:
            ValueError: If non-admissible DOF types are used as local DOFs.
            ValueError: If one attempts to create a variable not defined on any grid,
                or both on interfaces and subdomains.
            KeyError: If a variable with given name is already defined.

        """
        # Set default value for dof_info. This is a mutable object, so we need to
        # create a new one each time and not set the default in the signature.
        if dof_info is None:
            dof_info = {"cells": 1}

        # Sanity check for admissible DOF types.
        requested_type = set(dof_info.keys())
        if not requested_type.issubset(set(self.admissible_dof_types)):
            non_admissible = requested_type.difference(self.admissible_dof_types)
            raise ValueError(f"Non-admissible DOF types {non_admissible} requested.")

        # Container for all grid variables.
        variables = []

        # Merge subdomains and interfaces into a single list.
        grids: Sequence[pp.GridLike]
        if subdomains is not None and interfaces is None:
            grids = subdomains
        elif subdomains is None and interfaces is not None:
            grids = interfaces
        elif subdomains is None and interfaces is None:
            raise ValueError(
                "Cannot create variable not defined on any subdomain or interface."
            )
        else:
            raise ValueError(
                "Cannot create variable both on interfaces and subdomains."
            )

        # Check if a md variable was already defined under that name on any of grids.
        for var in self.variables:
            if var.name == name and var.domain in grids:
                raise KeyError(f"Variable {name} already defined on {var.domain}.")

        for grid in grids:
            if subdomains:
                assert isinstance(grid, pp.Grid)  # mypy
                data = self.mdg.subdomain_data(grid)

                # Register boundary grid data for the subdomain if applicable.
                if (bg := self.mdg.subdomain_to_boundary_grid(grid)) is not None:
                    bg_data = self.mdg.boundary_grid_data(bg)
                    for key in [pp.TIME_STEP_SOLUTIONS, pp.ITERATE_SOLUTIONS]:
                        if key not in data:
                            bg_data[key] = {}
            else:
                assert isinstance(grid, pp.MortarGrid)  # mypy
                data = self.mdg.interface_data(grid)

            for key in [pp.TIME_STEP_SOLUTIONS, pp.ITERATE_SOLUTIONS]:
                if key not in data:
                    data[key] = {}
                if name not in data[key]:
                    data[key][name] = {}

            # Create grid variable.
            new_variable = Variable(name, dof_info, domain=grid, tags=tags)

            # Store it in the system
            variables.append(new_variable)
            self._variables[new_variable.id] = new_variable

            # Append the new DOFs to the global system.
            self._variable_dof_type[new_variable.id] = dof_info
            self._append_dofs(new_variable)

        # New optimized order
        self._cluster_dofs_gridwise()
        # Create an md variable that wraps all the individual variables created on
        # individual grids.
        merged_variable = MixedDimensionalVariable(variables)

        return merged_variable

    def update_variable_tags(
        self,
        tags: dict[str, Any],
        variables: Optional[VariableList] = None,
    ) -> None:
        """Assigns tags to variables.

        Parameters:
            tag_name: Tag dictionary (tag-value pairs). This will be assigned to all
                variables in the list.
            variables: List of variables to which the tag should be assigned. None is
                interpreted as all variables. If a mixed-dimensional variable is passed,
                the tags will be assigned to its sub-variables (living on individual
                grids).

        """
        assert isinstance(variables, list)

        variables = self._parse_variable_type(variables)
        for var in variables:
            var.tags.update(tags)

    def get_variables(
        self,
        variables: Optional[VariableList] = None,
        grids: Optional[list[pp.GridLike]] = None,
        tag_name: Optional[str] = None,
        tag_value: Optional[Any] = None,
    ) -> list[Variable]:
        """Filter variables based on grid, tag name and tag value.

        Particular usage: calling without arguments will return all variables in the
        system.

        Parameters:
            variables: List of variables to filter. If None, all variables in the system
                are included. Variables can be given as a list of variables, mixed-
                dimensional variables, or variable names (strings).
            grids: List of grids to filter on. If None, all grids are included.
            tag_name: Name of the tag to filter on. If None, no filtering on tags.
            tag_value: Value of the tag to filter on. If None, no filtering on tag
                values. If tag_name is not None, but tag_value is None, all variables
                with the given tag_name are returned regardless of value.

        Returns:
            List of filtered variables.

        """
        # Shortcut for efficiency.
        # The same behavior is achieved without this, but it is slower.
        if (
            variables is None
            and grids is None
            and tag_name is None
            and tag_value is None
        ):
            return self.variables

        # If no variables or grids are given, use full sets.
        if variables is None:
            variables = self.variables
        if grids is None:
            # Note: This gives all grids known to variables, not all grids in the
            # md grid. The result of the filtering will be the same, though.
            grids = self.variable_domains

        filtered_variables = []
        variables = self._parse_variable_type(variables)
        for var in variables:
            if var.domain in grids:
                # Add variable if tag_name is not specified or if the variable has the
                # tag and the tag value matches the requested value.
                if tag_name is None:
                    filtered_variables.append(var)
                elif tag_name in var.tags:
                    if tag_value is None or var.tags[tag_name] == tag_value:
                        filtered_variables.append(var)

        return filtered_variables

    def get_variable_values(
        self,
        variables: Optional[VariableList] = None,
        time_step_index: Optional[int] = None,
        iterate_index: Optional[int] = None,
    ) -> np.ndarray:
        """Assembles an array containing values for the passed variable-like argument.

        The gathered values will be the variable values corresponding to the storage
        index specified by the user. The global order is preserved and independent of
        the order of the argument.

        Parameters:
            variables (optional): VariableType input for which the values are
                requested. If None (default), the global vector of unknowns is returned.
            time_step_index: Specified by user if they want to gather variable values
                from a specific time-step. Value 0 provides the most recent time-step. A
                value of 1 will give the values of one time-step back in time.
            iterate_index: Specified by user if they want to gather a specific set of
                iterate values. Similar to ``time_step_index``, value 0 is the
                default value and gives the most recent iterate.

        Returns:
            The respective (sub) vector in numerical format, size anywhere between 0 and
                :meth:`num_dofs`.

        Raises:
            ValueError: If unknown VariableType arguments are passed.

        """
        variables = self._parse_variable_type(variables)
        var_ids = [var.id for var in variables]
        # Storage for atomic blocks of the sub vector (identified by name-grid pairs).
        values = []

        # Loop over all blocks and process those requested.
        # This ensures uniqueness and correct order.
        for id_ in self._variable_numbers:
            if id_ in var_ids:
                variable = self._variables[id_]

                val = pp.get_solution_values(
                    variable.name,
                    self._get_data(variable.domain),
                    time_step_index=time_step_index,
                    iterate_index=iterate_index,
                )
                # NOTE get_solution_values already returns a copy
                values.append(val)

        # If there are matching blocks, concatenate and return.
        if values:
            return np.concatenate(values)
        # Else return an empty vector.
        else:
            return np.array([])

    def set_variable_values(
        self,
        values: np.ndarray,
        variables: Optional[VariableList] = None,
        time_step_index: Optional[int] = None,
        iterate_index: Optional[int] = None,
        additive: bool = False,
    ) -> None:
        """Sets values for a (sub) vector of the global vector of unknowns.

        The order of values is assumed to fit the global order.

        Note:
            The vector is assumed to be of proper size and will be dissected according
            to the global order, starting with the index 0.
            Mismatches of is-size and should-be-size according to the subspace specified
            by ``variables`` will raise respective errors by numpy.

        See also:
            :meth:`~porepy.numerics.ad._ad_utils.set_solution_values`.

        Parameters:
            values: Vector of size corresponding to number of DOFs of the specified
                variables.
            variables (optional): VariableType input for which the values are
                requested. If None (default), the global vector of unknowns will be
                set.
            time_step_index: Index of previous time step for which the values are
                intended.
            iterate_index: Iterate index for current time step for which the values are
                intended.
            additive (optional): Flag to write values additively. To be used in
                iterative procedures.

        Raises:
            ValueError: If unknown VariableType arguments are passed.

        """

        # Start of dissection.
        dof_start = 0
        dof_end = 0
        variables = self._parse_variable_type(variables)
        var_ids = [var.id for var in variables]

        for id_, variable_number in self._variable_numbers.items():
            if id_ in var_ids:
                # 1. Slice the vector to local size
                # This will raise errors if indexation is out of range.
                num_dofs = int(self._variable_num_dofs[variable_number])
                # Extract local vector.
                # This will raise errors if indexation is out of range.
                dof_end = dof_start + num_dofs
                # Extract local vector.
                # This will raise errors if indexation is out of range.
                local_vec = values[dof_start:dof_end]

                # 2.  Use the AD utilities to set the values
                variable = self._variables[id_]
                pp.set_solution_values(
                    variable.name,
                    local_vec,
                    self._get_data(grid=variable.domain),
                    time_step_index=time_step_index,
                    iterate_index=iterate_index,
                    additive=additive,
                )

                # 3. Move dissection forward.
                dof_start = dof_end

        # Last sanity check if the vector was properly sized, or if it was too large.
        # This imposes a theoretically unnecessary restriction on the input argument
        # since we only require a vector of at least this size.
        assert dof_end == values.size

    def shift_time_step_values(
        self,
        variables: Optional[VariableList] = None,
    ) -> None:
        """Method for shifting stored time step values in data sub-dictionary.

        For details of the value shifting see the method :meth:`_shift_variable_values`.

        Parameters:
            variables (optional): VariableType input for which the values are
                requested. If None (default), the global vector of unknowns will
                be shifted.

        """
        self._shift_variable_values(
            location=pp.TIME_STEP_SOLUTIONS, variables=variables
        )

    def shift_iterate_values(
        self,
        variables: Optional[VariableList] = None,
    ) -> None:
        """Method for shifting stored iterate values in data sub-dictionary.

        For details of the value shifting see the method :meth:`_shift_variable_values`.

        Parameters:
            variables (optional): VariableType input for which the values are
                requested. If None (default), the global vector of unknowns will
                be shifted.

        """
        self._shift_variable_values(location=pp.ITERATE_SOLUTIONS, variables=variables)

    def _shift_variable_values(
        self,
        location: str,
        variables: Optional[VariableList] = None,
    ) -> None:
        """Method for shifting values in data dictionary.

        Time step and iterate values are stored with storage indices as keys in
        the data dictionary for the subdomain or interface in question. For each
        time-step/iteration, these values are shifted such that the most recent
        variable value later can be placed at index 0. The previous
        time-step/iterate values have their index incremented by one. Values
        of key 0 is moved to key 1, values of key 1 is moved to key 2, and so
        on. The value at the highest key is discarded.

        Parameters:
            location: Should be ``pp.TIME_STEP_SOLUTIONS`` or ``pp.ITERATE_SOLUTIONS``
                depending on which one of solutions/iterates that are to be shifted.
            variables (optional): VariableType input for which the values are
                requested. If None (default), the global vector of unknowns will
                be shifted.

        Raises:
            ValueError: If unknown VariableType arguments are passed.

        """
        # Looping through the variables and shifting the values
        variables = self._parse_variable_type(variables)
        for variable in variables:
            name = variable.name
            grid = variable.domain
            data = self._get_data(grid=grid)

            # Shift old values as requested.
            num_stored = len(data[location][name])
            for i in range(num_stored - 1, 0, -1):
                data[location][name][i] = data[location][name][i - 1].copy()

    def _get_data(
        self,
        grid: pp.GridLike,
    ) -> dict:
        """Method for gathering data dictionary for a given grid.

        Parameters:
            grid: Subdomain/interface whose data dictionary the user is interested in.

        Returns:
            Data dictionary corresponding to ``grid``.

        """
        if isinstance(grid, pp.Grid):
            data = self.mdg.subdomain_data(sd=grid)
        elif isinstance(grid, pp.MortarGrid):
            data = self.mdg.interface_data(intf=grid)
        return data

    ### DOF management -----------------------------------------------------------------

    def _append_dofs(self, variable: pp.ad.Variable) -> None:
        """Appends DOFs for a newly created variable at the end of the current order.

        Optimization of variable order is done afterwards.

        Must only be called by :meth:`create_variables`.

        Parameters:
            variable: The newly created variable

        """
        # number of totally created dof blocks so far
        last_variable_number: int = len(self._variable_numbers)

        # Sanity check that no previous data is overwritten. This should not happen,
        # if class not used in hacky way.
        assert variable.id not in self._variable_numbers

        # Count number of dofs for this variable on this grid and store it.
        # The number of dofs for each dof type defaults to zero.

        local_dofs = self._variable_dof_type[variable.id]
        # Both subdomains and interfaces have cell variables.
        num_dofs = variable.domain.num_cells * local_dofs.get("cells", 0)

        # For subdomains, but not interfaces, we also need to account for faces and
        # nodes.
        if isinstance(variable.domain, pp.Grid):
            num_dofs += variable.domain.num_faces * local_dofs.get(
                "faces", 0
            ) + variable.domain.num_nodes * local_dofs.get("nodes", 0)

        # Update the global dofs and block numbers
        self._variable_numbers.update({variable.id: last_variable_number})
        self._variable_num_dofs = np.concatenate(
            [self._variable_num_dofs, np.array([num_dofs], dtype=int)]
        )

    def _cluster_dofs_gridwise(self) -> None:
        """Re-arranges the DOFs grid-wise s.t. we obtain grid-blocks in the column sense
        and reduce the matrix bandwidth.

        The aim is to impose a more block-diagonal-like structure on the Jacobian where
        blocks in the column sense represent single grids in the following order:

        1. For each grid in ``mdg.subdomains``
            1. For each variable defined on that grid
        2. For each grid in ``mdg.interfaces``
            1. For each variable defined on that mortar grid

        The order of variables per grid is given by the order of variable creation.
        This method is called after each creation of variables and respective DOFs.

        """
        # Data stracture for the new order of dofs.
        new_variable_counter: int = 0
        new_variable_numbers: dict[int, int] = dict()
        new_block_dofs: list[int] = list()

        # 1. Per subdomain, order variables
        for grid in self.mdg.subdomains():
            for id_, variable in self._variables.items():
                if variable.domain == grid:
                    local_dofs = self._variable_num_dofs[self._variable_numbers[id_]]
                    new_block_dofs.append(local_dofs)
                    new_variable_numbers.update({id_: new_variable_counter})
                    new_variable_counter += 1

        # 2. Per interface, order variables
        for intf in self.mdg.interfaces():
            for id_, variable in self._variables.items():
                if variable.domain == intf:
                    local_dofs = self._variable_num_dofs[self._variable_numbers[id_]]
                    new_block_dofs.append(local_dofs)
                    new_variable_numbers.update({id_: new_variable_counter})
                    new_variable_counter += 1

        # Replace old block order
        self._variable_num_dofs = np.array(new_block_dofs, dtype=int)
        self._variable_numbers = new_variable_numbers

    def _parse_variable_type(self, variables: Optional[VariableList]) -> list[Variable]:
        """Parse the input argument for the variable type.

        This method is used to parse the input argument for the variable type in
        several exposed methods, allowing the user to specify a single variable or a
        list of variables more flexibly.

        There is no filtering of the variables, for instance:

            - No assumptions should be made on the order of the parsed variables.
            - The variable list is not uniquified; if the same variable is passed twice
              (say, as a Variable and by its string), it will duplicated in the list of
              parsed variables.

        Parameters:
            variables: The input argument for the variable type.
                The following interpretation rules are applied:
                    - If None, return all variables.
                    - If a list of variables, return same.
                    - If a list of strings, return all variables with those names.
                    - If mixed-dimensional variable, return sub-variables.

        Returns:
            List of Variables.

        """
        if variables is None:
            return self.variables
        parsed_variables = []
        assert isinstance(variables, list)
        for variable in variables:
            if isinstance(variable, MixedDimensionalVariable):
                parsed_variables += [var for var in variable.sub_vars]
            elif isinstance(variable, Variable):
                parsed_variables.append(variable)
            elif isinstance(variable, str):
                # Use _variables to avoid recursion (get_variables() calls this method)
                vars = [var for var in self._variables.values() if var.name == variable]
                parsed_variables += vars
            else:
                raise ValueError(
                    "Variable type must be a string or a Variable, not {}".format(
                        type(variable)
                    )
                )
        return parsed_variables

    def _gridbased_variable_complement(self, variables: VariableList) -> list[Variable]:
        """Finds the grid-based complement of a variable-like structure.

        The grid-based complement consists of all variables known to this
        EquationSystem, but which are not in the passed list ``variables``.

        TODO: Revisit. This method is not used anywhere, and I am not sure it is
        correct/does what it is supposed to do.
        """

        # strings and md variables represent always a whole in the variable sense. Hence,
        # the complement is empty
        if isinstance(variables, (str, MixedDimensionalVariable)):
            # TODO: Can we drop this, or is it possible that a single variable has made
            # it into this subroutine?
            return list()

        # non sequential var-like structure
        else:
            grid_variables = list()
            for variable in variables:
                # same processing as above, only grid variables are of interest
                if isinstance(variable, Variable):
                    md_variable = self.md_variable(variable.name)
                    grid_variables += [
                        var
                        for var in md_variable.sub_vars
                        if var.domain != variable.domain
                    ]
            # return a unique collection
            return list(set(grid_variables))

    def num_dofs(self) -> int:
        """Returns the total number of dofs managed by this system."""
        return int(sum(self._variable_num_dofs))  # cast numpy.int64 into Python int

    def projection_to(self, variables: Optional[VariableList] = None) -> sps.csr_matrix:
        """Create a projection matrix from the global vector of unknowns to a specified
        subspace.

        The transpose of the returned matrix can be used to slice respective columns out
        of the global Jacobian.

        The projection preserves the global order defined by the system, i.e. it
        includes no permutation.

        Parameters:
            variables (optional): VariableType input for which the subspace is
                requested. If no subspace is specified using ``variables``,
                a null-space projection is returned.

        Returns:
            a sparse projection matrix of shape ``(M, num_dofs)``, where
            ``0 <= M <= num_dofs``.

        """
        # current number of total dofs
        num_dofs = self.num_dofs()
        if variables:
            # Array for the indices associated with argument.
            # The sort is needed so as not to permute the columns of the projection.
            indices = np.sort(self.dofs_of(variables))
            # case where no dofs where found for the VariableType input
            if len(indices) == 0:
                return sps.csr_matrix((0, num_dofs))
            else:
                subspace_size = indices.size
                return sps.coo_matrix(
                    (np.ones(subspace_size), (np.arange(subspace_size), indices)),
                    shape=(subspace_size, num_dofs),
                ).tocsr()
        # Case where the subspace is null, i.e. no variables specified
        else:
            return sps.csr_matrix((0, num_dofs))

    def dofs_of(self, variables: VariableList) -> np.ndarray:
        """Get the indices in the global vector of unknowns belonging to the variables.

        Parameters:
            variables: VariableType input for which the indices are requested.

        Returns:
            An array of indices/ DOFs corresponding to ``variables``.
            Note that the order of indices corresponds to the order in ``variables``.

        Raises:
            ValueError: If an unknown  variable is passed as argument.

        """
        variables = self._parse_variable_type(variables)
        global_variable_dofs = np.hstack((0, np.cumsum(self._variable_num_dofs)))

        indices: list[np.ndarray] = []

        for var in variables:
            if var.id in self._variable_numbers:
                variable_number = self._variable_numbers[var.id]
                var_indices = np.arange(
                    global_variable_dofs[variable_number],
                    global_variable_dofs[variable_number + 1],
                    dtype=int,
                )
                indices.append(var_indices)
            else:
                raise ValueError(
                    f"Variable {var.name} with ID {var.id} not registered among DOFS"
                    + f" of equation system {self}."
                )

        # Concatenate indices, if any
        if len(indices) > 0:
            all_indices = np.concatenate(indices, dtype=int)
        else:
            all_indices = np.array([], dtype=int)

        return all_indices

    def identify_dof(self, dof: int) -> Variable:
        """Identifies the variable to which a specific DOF index belongs.

        The intended use is to help identify entries in the global vector or the column
        of the Jacobian.

        Parameters:
            dof: a single index in the global vector.

        Returns: the identified Variable object.

        Raises:
            KeyError: if the dof is out of range (larger than ``num_dofs`` or smaller
                than 0).

        """
        num_dofs = self.num_dofs()
        if not (0 <= dof < num_dofs):  # indices go from 0 to num_dofs - 1
            raise KeyError("Dof index out of range.")

        global_variable_dofs = np.hstack((0, np.cumsum(self._variable_num_dofs)))
        # Find the variable number belonging to this index
        variable_number = np.argmax(global_variable_dofs > dof) - 1
        # Get the variable key from _variable_numbers
        # find the ID belonging to the dof
        id_ = [
            id_ for id_, num in self._variable_numbers.items() if num == variable_number
        ]
        # sanity check that only 1 ID was found
        assert len(id_) == 1, "Failed to find unique ID corresponding to `dof`."
        # find variable with the ID
        variable = [var for _id, var in self._variables.items() if _id == id_[0]]
        assert len(variable) == 1, "Failed to find Variable corresponding to `dof`."
        return variable[0]

    ### Equation management -------------------------------------------------------------------

    def set_equation(
        self,
        equation: Operator,
        grids: DomainList,
        equations_per_grid_entity: dict[GridEntity, int],
    ) -> None:
        """Sets an equation using the passed operator and uses its name as an identifier.

        If an equation already exists under that name, it is overwritten.

        Information about the image space must be provided for now, such that grid-wise
        row slicing is possible. This will hopefully be provided automatically in the
        future.

        Note:
            Regarding the number of equations, this method assumes that the AD framework
            assembles row blocks per grid in subdomains, then per grid in interfaces,
            for each operator representing an equation. This is assumed to be the way
            PorePy AD works.

        Parameters:
            equation: An equation in AD operator form, assuming the right-hand side is
                zero and this instance represents the left-hand side.
            grids: A list of subdomain *or* interface grids on which the equation is
                defined.
            equations_per_grid_entity: a dictionary describing how many equations
                ``equation_operator`` provides. This is a temporary work-around until
                operators are able to provide information on their image space.
                The dictionary must contain the number of equations per grid entity
                (cells, faces, nodes) for the operator.

        Raises:
            ValueError: If the equation operator has a name already assigned to a
                previously set equation.
            ValueError: If the equation is defined on both subdomains and interfaces.
            AssertionError: If the equation is defined on an unknown grid.
            ValueError: If indicated number of equations does not match the actual
                number as per evaluation of operator.

        """
        # The grid list is changed in place, so we need to make a copy
        grids = grids[:]
        # The function loops over all grids the operator is defined on and calculate the
        # number of equations per grid quantity (cell, face, node). This information
        # is then stored together with the equation itself.
        image_info: dict[pp.GridLike, np.ndarray] = dict()
        total_num_equ = 0

        # The domain of this equation is the set of grids on which it is defined
        name = equation.name
        if name in self._equations:
            raise ValueError(
                "The name of the equation operator is already used by another equation:"
                f"\n{self._equations[name]}"
                "\n\nMake sure your equations are uniquely named."
            )

        # If no grids are specified, there is nothing to do
        if not grids:
            self._equation_image_space_composition.update({name: image_info})
            # Information on the size of the equation, in terms of the grids it is defined
            # on.
            self._equation_image_size_info.update({name: equations_per_grid_entity})
            # Store the equation itself.
            self._equations.update({name: equation})
            return

        # We require that equations are defined either on a set of subdomains, or a set
        # of interfaces. The combination of the two is mathematically possible, provided
        # a sufficiently general notation is used, but the chances of this being
        # misused is considered high compared to the benefits of allowing such combined
        # domains, and we therefore disallow it.

        all_subdomains = all([isinstance(g, pp.Grid) for g in grids])
        all_interfaces = all([isinstance(g, pp.MortarGrid) for g in grids])

        # Allow for no subdomains or interfaces here (case < 1). This is relevant for
        # equations stated for general md problems, but on domains that happened not to
        # have, e.g., fractures.
        if not all_interfaces + all_subdomains <= 1:
            raise AssertionError(
                "An equation should not be defined on both subdomains and interfaces."
            )

        # We loop over the subdomains and interfaces in that order to assert a correct
        # indexation according to the global order (for grid in sds, for grid in intfs).
        # The user does not have to care about the order in grids.
        for sd in self.mdg.subdomains():
            if sd in grids:
                # Equations on subdomains can be defined on any grid quantity.
                num_equ_per_grid = int(
                    sd.num_cells * equations_per_grid_entity.get("cells", 0)
                    + sd.num_nodes * equations_per_grid_entity.get("nodes", 0)
                    + sd.num_faces * equations_per_grid_entity.get("faces", 0)
                )
                # Row indices for this grid, cast to integers.
                block_idx = np.arange(num_equ_per_grid, dtype=int) + total_num_equ
                # Cumulate total number of equations.
                total_num_equ += num_equ_per_grid
                # Store block idx per grid.
                image_info.update({sd: block_idx})
                # Remove the subdomain from the domain list.
                # Ignore mypy error here, since we know that sd is in grids.
                grids.remove(sd)  # type: ignore

        for intf in self.mdg.interfaces():
            if intf in grids:
                # Equations on interfaces can only be defined on cells.
                num_equ_per_grid = int(
                    intf.num_cells * equations_per_grid_entity.get("cells", 0)
                )
                # Row indices for this grid, cast to integers.
                block_idx = np.arange(num_equ_per_grid, dtype=int) + total_num_equ
                # Cumulate total number of equations.
                total_num_equ += num_equ_per_grid
                # Store block idx per grid
                image_info.update({intf: block_idx})
                # Remove the grid from the domain list
                # Ignore mypy error here, since we know that intf is in grids.
                grids.remove(intf)  # type: ignore

        # Assert the equation is not defined on an unknown domain.
        assert len(grids) == 0

        # If all good, we store the information:
        # The rows (referring to a global indexation) that this equation provides.
        self._equation_image_space_composition.update({name: image_info})
        # Information on the size of the equation, in terms of the grids it is defined
        # on.
        self._equation_image_size_info.update({name: equations_per_grid_entity})
        # Store the equation itself.
        self._equations.update({name: equation})

    def remove_equation(self, name: str) -> Operator | None:
        """Removes a previously set equation and all related information.

        Returns:
            A reference to the equation in operator form or None, if the equation is
            unknown.

        Raises:
            ValueError: If an unknown equation is attempted removed.

        """
        if name in self._equations:
            # Remove the equation from the storage
            equ = self._equations.pop(name)
            # Remove the image space information.
            # Note that there is no need to modify the numbering of the other equations,
            # since this is a local (to the equation) numbering.
            del self._equation_image_space_composition[name]
            return equ
        else:
            raise ValueError(f"Cannot remove unknown equation {name}")

    def update_variable_num_dofs(self) -> None:
        """Update the count of degrees of freedom related to a MixedDimensionalGrid.

        The method loops through the variables and updates the number of fine-scale
        degree of freedom. The system size will be updated if the grid has changed or
        (perhaps less realistically) a variable has had its number of dofs per grid
        quantity changed.

        NOTE: This method is experimental and should be used with caution. After this
        method has been called, other attributes of the class that depend on the number
        of dofs (such as _equation_image_space_composition) will be outdated and should
        be used with care.

        """
        for id_, var in self._variables.items():
            # Grid quantity (grid or interface), and variable
            grid = var.domain

            dof = self._variable_dof_type[id_]
            num_dofs: int = grid.num_cells * dof.get("cells", 0)  # type: ignore

            if isinstance(grid, pp.Grid):
                # Add dofs on faces and nodes, but not on interfaces
                num_dofs += grid.num_faces * dof.get(
                    "faces", 0
                ) + grid.num_nodes * dof.get("nodes", 0)

            # Update local counting
            self._variable_num_dofs[self._variable_numbers[id_]] = num_dofs

    ### System assembly and discretization ----------------------------------------------------

    @staticmethod
    def _recursive_discretization_search(operator: Operator, discr: list) -> list:
        """Recursive search in the tree of this operator to identify all discretizations
        represented in the operator.

        Parameters:
            operator: top level operator to be searched.
            discr: list storing found discretizations

        """
        if len(operator.children) > 0:
            # Go further in recursion
            for child in operator.children:
                discr += EquationSystem._recursive_discretization_search(child, list())

        if isinstance(operator, _ad_utils.MergedOperator):
            # We have reached the bottom; this is a discretization (example: mpfa.flux)
            discr.append(operator)

        return discr

    def _parse_equations(
        self, equations: Optional[EquationList | EquationRestriction] = None
    ) -> dict[str, None | np.ndarray]:
        """Helper method to parse equations into a properly ordered structure.

        The equations will be ordered according to the order in self._equations (which
        is the order in which they were added to the equation system manager and which
        alsois fixed since iteration of dictionaries is so).

        Parameters:
            equations: A list of equations or a dictionary of equation restrictions.

        Returns:
            A dictionary with the index set of the restricted equations (referring to
            equation rows) as values. If no restriction is given, the value is None.

        """
        # The default return value is all equations with no grid restrictions.
        if equations is None:
            return dict((name, None) for name in self._equations)

        # We need to parse the input.
        # Storage for requested blocks, unique information per equation name.
        requested_row_blocks = dict()
        # Storage for restricted equations.
        restricted_equations = dict()

        # Get the row indices (in the global system) associated with this equation.
        # If the equation is restricted (the user has provided a dictionary with
        # grids on which the equation should be evaluated), the variable blocks
        # will contain only the row indices associated with the restricted grids.

        for equation in equations:
            # Store restrictions, using different storage for restricted and
            # unrestricted equations.
            if isinstance(equations, dict):
                block = self._parse_single_equation({equation: equations[equation]})
                # A dictionary means the equation is restricted to a subset of grids.
                restricted_equations.update(block)
            else:
                # This equation is not restricted to a subset of grids.
                block = self._parse_single_equation(equation)
                requested_row_blocks.update(block)

        # Update the requested blocks with the restricted to overwrite the indices if
        # an equation was passed in both restricted and unrestricted structure.
        requested_row_blocks.update(restricted_equations)

        # Build the restricted set of equations, using the order in self._equations.
        # The latter is critical for ensuring determinism of the system.
        ordered_blocks = dict()
        for equation in self._equations:
            # By now, all equations are contained in requested_row_blocks.
            if equation in requested_row_blocks:
                ordered_blocks.update({equation: requested_row_blocks[equation]})

        return ordered_blocks

    def _parse_single_equation(
        self, equation: str | Operator | EquationRestriction
    ) -> dict[str, None | np.ndarray]:
        """Helper method to identify possible restrictions of a single equation.

        Parameters:
            equation: Equation to be parsed.

        Returns:
            A dictionary with the name of the equation as key and the corresponding
            restricted indices as values. If no restriction is given, the value is None.

        Raises:
            ValueError: If an unknown equation name is requested.
            ValueError: If an unknown operator is requested.
            ValueError: If an equation is requested restricted to a grid on which it is
                not defined.
            TypeError: If the input is not an equation.

        """
        # If the equation is a dictionary, the dictionary values are grids (subdomains
        # or interfaces) that defines restrictions of the equation; these must be
        # identified. If the equation is not a dictionary, there will be restriction.

        # Equation represented by string - return the corresponding equation.
        if isinstance(equation, str):
            if equation not in self._equations:
                raise ValueError(f"Unknown equation name {equation}.")
            return {equation: None}

        # Equation represented by Operator. Return the
        elif isinstance(equation, Operator):
            if equation.name not in self._equations:
                raise ValueError(f"Unknown equation operator {equation}.")
            # No restriction.
            return {equation.name: None}

        # Equations represented by dict with restriction to grids: get target row
        # indices.
        elif isinstance(equation, dict):
            block: dict[str, None | np.ndarray] = dict()
            for equ, grids in equation.items():
                # equ is an identifier of the equation (either a string or an operator)
                # grids is a list of grids (subdomains or interfaces) that defines
                # a restriction of the equation domain.

                # Translate equ into a name (string).
                if isinstance(equ, Operator):
                    name = equ.name
                    if name not in self._equations:
                        raise ValueError(f"Unknown equation name {equation}.")
                elif isinstance(equ, str):
                    name = equ
                    if name not in self._equations:
                        raise ValueError(f"Unknown equation operator {equation}.")
                else:
                    raise TypeError(
                        f"Item ({type(equ)}, {type(grids)}) not parsable as equation."
                    )

                # Get the indices associated with this equation.
                img_info = self._equation_image_space_composition[name]

                # Check if the user requests a properly defined subsets of the grids
                # associated with this equation.
                unknown_grids = set(grids).difference(set(img_info.keys()))
                if len(unknown_grids) > 0:
                    # Getting an error here means the user has requested a grid that is
                    # not associated with this equation. This is not a meaningful
                    # operation.
                    raise ValueError(
                        f"Equation {name} not defined on grids {unknown_grids}"
                    )

                # The indices (row indices in the global system) associated with this
                # equation and the grids requested by the user.
                block_idx: list[np.ndarray] = list()

                # Loop over image space information to ensure correct order.
                # Note that looping over the grids risks that the order does not
                # correspond to the order in the equation was defined. This will surely
                # lead to trouble down the line.
                for grid in img_info:
                    if grid in grids:
                        block_idx.append(img_info[grid])

                if len(block_idx) > 0:
                    # If indices not empty, concatenate and return.
                    block.update({name: np.concatenate(block_idx, dtype=int)})
                else:
                    # If indices empty, return empty array.
                    block.update({name: np.array([], dtype=int)})
            return block
        else:
            # Getting an error here means the user has passed a type that is not
            # an equation.
            raise TypeError(f"Type {type(equation)} not parsable as an equation.")

    def _gridbased_equation_complement(
        self, equations: dict[str, None | np.ndarray]
    ) -> dict[str, None | np.ndarray]:
        """Takes the information from equation parsing and finds for each equation
        (identified by its name string) the indices which were excluded in the
        grid-sense.

        Parameters:
            equations: Dictionary with equation names as keys and indices as values.
                The indices are the indices of the rows in the global system that
                were included in the last parsing of the equations.

        Returns:
            A dictionary with the name of the equation as key and the grid-complement
            as values. If the complement is empty, the value is None.

        """
        complement: dict[str, None | np.ndarray] = dict()
        for name, idx in equations.items():
            # If indices were filtered based on grids, we find the complementing
            # indices.
            # If idx is None, this means no filtering was done.
            if idx is not None:
                # Get the indices associated with this equation.
                img_info = self._equation_image_space_composition[name]

                # Ensure ordering and uniqueness of equation indexation.
                img_values: list[np.ndarray] = list(img_info.values())
                all_idx = np.unique(np.hstack(img_values))

                # Complementing indices are found by deleting the filtered indices.
                complement_idx = np.delete(all_idx, idx)
                complement.update({name: complement_idx})

            # If there was no grid-based row filtering, the complement is empty.
            else:
                complement.update({name: None})
        return complement

    def discretize(
        self, equations: Optional[EquationList | EquationRestriction] = None
    ) -> None:
        """Find and loop over all discretizations in the equation operators, extract
        unique references and discretize.

        This is more efficient than discretizing on the Operator level, since
        discretizations which occur more than once in a set of equations will be
        identified and only discretized once.

        Parameters:
            equations (optional): A subset of equations. If not provided (None), all
                known equations will be discretized.

        """
        equation_names = list(self._parse_equations(equations).keys())

        # List containing all discretizations
        discr: list = []
        # TODO: the search can be done once (in some kind of initialization). Revisit
        # this during update of the Ad machinery.
        for name in equation_names:
            # this raises a key error if a given equation name is unknown
            eqn = self._equations[name]
            # This will expand the list discr with new discretizations.
            # The list may contain duplicates.
            discr += self._recursive_discretization_search(eqn, list())

        # Uniquify to save computational time, then discretize.
        unique_discr = _ad_utils.uniquify_discretization_list(discr)
        _ad_utils.discretize_from_list(unique_discr, self.mdg)

    @overload
    def assemble(
        self,
        evaluate_jacobian: Literal[True] = True,
        equations: Optional[EquationList | EquationRestriction] = None,
        variables: Optional[VariableList] = None,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]: ...

    @overload
    def assemble(
        self,
        evaluate_jacobian: Literal[False],
        equations: Optional[EquationList | EquationRestriction] = None,
        variables: Optional[VariableList] = None,
        state: Optional[np.ndarray] = None,
    ) -> np.ndarray: ...

    def assemble(
        self,
        evaluate_jacobian: bool = True,
        equations: Optional[EquationList | EquationRestriction] = None,
        variables: Optional[VariableList] = None,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray] | np.ndarray:
        """Assemble Jacobian matrix and residual vector using a specified subset of
        equations, variables and grids.

        The method is intended for use in splitting algorithms. Matrix blocks not
        included will simply be sliced out.

        Note:
            The ordering of columns in the returned system are defined by the global DOF
            order. The row blocks are in the same order as equations were added to this
            system. If an equation is defined on multiple grids, the respective
            row-block is internally ordered as given by the mixed-dimensional grid (for
            sd in subdomains, for intf in interfaces).

            The columns of the subsystem are assumed to be properly defined by
            ``variables``, otherwise a matrix of shape ``(M,)`` is returned. This
            happens if grid variables are passed which are unknown to this
            :class:`EquationSystem`.

        Parameters:
            evaluate_jacobian: Whether to evaluate and return the Jacobian matrix.
                Defaults to True.
            equations (optional): a subset of equations to which the subsystem should be
                restricted. If not provided (None), all equations known to this
                :class:`EquationSystem` will be included.

                The user can specify grids per equation (name) to which the subsystem
                should be restricted in the row-sense. Grids not belonging to the domain
                of an equation will raise an error.

            variables (optional): VariableType input specifying the subspace in
                column-sense. If not provided (None), all variables will be included.
            state (optional): State vector to assemble from. By default, the
                ``pp.ITERATE_SOLUTIONS`` or ``pp.TIME_STEP_SOLUTIONS`` are used, in that
                order.

        Returns:
            Tuple with two elements

                spmatrix: (Part of the) Jacobian matrix corresponding to the targeted
                variable state, for the specified equations and variables.
                ndarray: Residual vector corresponding to the targeted variable state,
                for the specified equations. Scaled with -1 (moved to rhs).

            or, if ``evaluate_jacobian`` is False,

                ndarray: Residual vector corresponding to the targeted variable state,
                for the specified equations. Scaled with -1 (moved to rhs).

        """
        if variables is None:
            variables = self.variables

        # equ_blocks is a dictionary with equation names as keys and the corresponding
        # row indices of the equations. If the user has requested that equations are
        # restricted to a subset of grids, the row indices are restricted accordingly.
        # If no such request has been made, the value is None.
        equ_blocks: dict[str, np.ndarray | None] = self._parse_equations(equations)

        # Data structures for building matrix and residual vector
        mat: list[sps.spmatrix] = []
        rhs: list[np.ndarray] = []

        # Keep track of DOFs for each equation/block
        ind_start = 0

        # Store the indices of the assembled equations only if the Jacobian is
        # requested.
        if evaluate_jacobian:
            self.assembled_equation_indices = dict()

        # Iterate over equations, assemble.
        # Also keep track of the row indices of each equation, and store it in
        # assembled_equation_indices.
        for equ_name, rows in equ_blocks.items():
            # This will raise a key error if the equation name is unknown.
            eq = self._equations[equ_name]

            if not evaluate_jacobian:
                # Evaluate the residual vector only. Enforce that the result is a numpy
                # array.
                val = np.asarray(eq.value(self, state))
                if rows is not None:
                    rhs.append(val[rows])
                else:
                    rhs.append(val)
                # Go to the next equation
                continue

            ad = eq.value_and_jacobian(self, state)

            # If restriction to grid-related row blocks was made,
            # perform row slicing based on information we have obtained from parsing.
            if rows is not None:
                mat.append(ad.jac.tocsr()[rows])
                rhs.append(ad.val[rows])
                block_length = len(rhs[-1])
            # If no grid-related row restriction was made, append the whole thing.
            else:
                mat.append(ad.jac)
                rhs.append(ad.val)
                block_length = len(ad.val)

            # Create indices range and shift to correct position.
            block_indices = np.arange(block_length) + ind_start
            # Extract last index and add 1 to get the starting point for next block of
            # indices.

            self.assembled_equation_indices.update({equ_name: block_indices})

            if block_length > 0:
                ind_start = block_indices[-1] + 1
        # Concatenate results equation-wise.
        if len(rhs) > 0:
            if evaluate_jacobian:
                A = sps.vstack(mat, format="csr")
            rhs_cat = np.concatenate(rhs)
        else:
            # Special case if the restriction produced an empty system.
            A = sps.csr_matrix((0, self.num_dofs()))
            rhs_cat = np.empty(0)

        if not evaluate_jacobian:
            return -rhs_cat

        # Slice out the columns belonging to the requested subsets of variables and
        # grid-related column blocks by using the transposed projection to respective
        # subspace.
        # Multiply rhs by -1 to move to the rhs.
        column_projection = self.projection_to(variables).transpose()
        return A * column_projection, -rhs_cat

    def assemble_schur_complement_system(
        self,
        primary_equations: EquationList | EquationRestriction,
        primary_variables: VariableList,
        inverter: Optional[Callable[[sps.spmatrix], sps.spmatrix]] = None,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        r"""Assemble Jacobian matrix and residual vector using a Schur complement
        elimination of the variables and equations not to be included.

        The specified equations and variables will define blocks of the linearized
        system as

        .. math::
            \left [ \begin{matrix} A_{pp} & A_{ps} \\ A_{sp} & A_{ss} \end{matrix} \right]
            \left [ \begin{matrix} x_p \\ x_s \end{matrix}\right]
            = \left [ \begin{matrix} b_p \\ b_s \end{matrix}\right]


        where subscripts p and s define primary and secondary blocks.
        The Schur complement system is then given by

        .. math::

            \left( A_{pp} - A_{ps} * A_{ss}^{-1} * A_{sp}\right) * x_p
            = b_p - A_{ps} * A_{ss} * b_s

        The Schur complement is well-defined only if the inverse of :math:`A_{ss}`
        exists, and the efficiency of the approach assumes that an efficient inverter
        for :math:`A_{ss}` can be found.
        **The user must ensure both requirements are fulfilled.**

        Note:
            The optional arguments defining the secondary block, and the flag
            ``excl_loc_prim_to_sec`` are meant for nested Schur-complements and
            splitting solvers. This is an advanced usage and requires the user to be
            careful, since the resulting blocks :math:`A_{pp}` and :math:`A_{ss}` might
            end up to be not square. This will result in errors.

        Examples:
            The default inverter can be defined by

            .. code-block:: python

                import scipy.sparse as sps
                inverter = lambda A: sps.csr_matrix(sps.linalg.inv(A.A))

            It is costly in terms of computational time and memory, though.

            TODO: We should rather use the block inverter in pp.matrix_operations. This
            will require some work on ensuring the system is block-diagonal.

        Parameters:
            primary_equations: a subset of equations specifying the primary subspace in
                row-sense.
            primary_variables: VariableType input specifying the primary subspace in
                column-sense.
            inverter (optional): callable object to compute the inverse of the matrix
                :math:`A_{ss}`. By default, the scipy direct sparse inverter is used.
            state (optional): see :meth:`assemble`. Defaults to None.

        Returns:
            Tuple containing

                sps.spmatrix: Jacobian matrix representing the Schur complement with
                respect to the targeted state.
                np.ndarray: Residual vector for the Schur complement with respect to the
                targeted state. Scaled with -1 (moved to rhs).

        Raises:
            AssertionError:

                - If the primary block would have 0 rows or columns.
                - If the secondary block would have 0 rows or columns.
                - If the secondary block is not square.

            ValueError: If primary and secondary columns overlap.

        """
        if inverter is None:
            inverter = lambda A: sps.csr_matrix(np.linalg.inv(A.A))

        # Find the rows of the primary block. This can include both equations defined
        # on their full image, and equations specified on a subset of grids.
        # The variable primary_rows will contain the indices in the global system
        # corresponding to the primary block.
        primary_rows = self._parse_equations(primary_equations)
        # Find indices of equations involved in the primary block, but on grids that
        # were filtered out. These will be added to the secondary block.
        excluded_primary_rows = self._gridbased_equation_complement(primary_rows)

        # Names of equations that form the primary block.
        primary_equation_names = list(primary_rows.keys())

        # Get the primary variables, represented as Ad variables.
        active_variables = self._parse_variable_type(primary_variables)

        # Projection of variables to the set of primary blocks.
        primary_projection = self.projection_to(active_variables)

        # Assert non-emptiness of primary block.
        assert len(primary_rows) > 0
        assert primary_projection.shape[0] > 0

        # Equations that are not part of the primary block. These will form parts of the
        # secondary block, as will the equations that are defined on grids that were
        # excluded.
        secondary_equation_names: list[str] = list(
            set(self._equations.keys()).difference(set(primary_rows.keys()))
        )
        secondary_variables = list(set(self.variables).difference(active_variables))
        secondary_projection = self.projection_to(secondary_variables)

        # Assert non-emptiness of secondary block. We do not check the length of
        # sequandary_equation_names, since this can empty if the secondary block is
        # defined by a subset of grids.
        assert secondary_projection.shape[0] > 0

        # Storage of primary and secondary row blocks.
        A_sec: list[sps.csr_matrix] = list()
        b_sec: list[np.ndarray] = list()
        A_prim: list[sps.csr_matrix] = list()
        b_prim: list[np.ndarray] = list()

        # Keep track of indices or primary block.
        ind_start = 0
        assembled_equation_indices = dict()

        # We loop over stored equations to ensure the correct order but process only
        # primary equations.
        # Excluded local primary blocks are stored as top rows in the secondary block.
        for name in self._equations:
            if name in primary_equation_names:
                A_temp, b_temp = self.assemble(equations=[name], state=state)
                idx_p = primary_rows[name]
                # Check if a grid filter was applied for that equation
                if idx_p is not None:
                    # Append the respective rows.
                    A_prim.append(A_temp[idx_p])
                    b_prim.append(b_temp[idx_p])
                    # If requested, the excluded primary rows are appended as secondary.
                    idx_excl_p = excluded_primary_rows[name]
                    A_sec.append(A_temp[idx_excl_p])
                    b_sec.append(b_temp[idx_excl_p])
                else:
                    # If no filter was applied, the whole row block is appended.
                    A_prim.append(A_temp)
                    b_prim.append(b_temp)

                # Track indices of block rows. Only primary equations are included.
                row_idx = np.arange(b_prim[-1].size, dtype=int)
                indices = row_idx + ind_start
                ind_start += row_idx.size
                assembled_equation_indices.update({name: indices})

        # store the assembled row indices for the primary block only (Schur)
        self.assembled_equation_indices = assembled_equation_indices

        # We loop again over stored equation to ensure a correct order
        # but process only secondary equations.
        for name in self._equations:
            # Secondary equations (those not explicitly given as being primary) are
            # assembled wholesale to the secondary block.
            if name in secondary_equation_names:
                A_temp, b_temp = self.assemble(equations=[name], state=state)
                A_sec.append(A_temp)
                b_sec.append(b_temp)

        # stack the results
        A_p = sps.vstack(A_prim, format="csr")
        b_p = np.concatenate(b_prim)
        A_s = sps.vstack(A_sec, format="csr")
        b_s = np.concatenate(b_sec)

        # turn the projections into prolongations
        primary_projection = primary_projection.transpose()
        secondary_projection = secondary_projection.transpose()

        # Matrices involved in the Schur complements
        A_pp = A_p * primary_projection
        A_ps = A_p * secondary_projection
        A_sp = A_s * primary_projection
        A_ss = A_s * secondary_projection

        # Last sanity check, if A_ss is square.
        assert A_ss.shape[0] == A_ss.shape[1]

        # Compute the inverse of A_ss using the passed inverter.
        inv_A_ss = inverter(A_ss)

        S = A_pp - A_ps * inv_A_ss * A_sp
        rhs_S = b_p - A_ps * inv_A_ss * b_s

        # Store information necessary for expanding the Schur complement later.
        self._Schur_complement = (
            inv_A_ss,
            b_s,
            A_sp,
            primary_projection,
            secondary_projection,
        )

        return S, rhs_S

    def expand_schur_complement_solution(
        self, reduced_solution: np.ndarray
    ) -> np.ndarray:
        r"""Expands the solution of the *last assembled* Schur complement system to the
        whole solution.

        With ``reduced_solution`` as :math:`x_p` from

        .. math::
            \left [ \begin{matrix} A_{pp} & A_{ps} \\ A_{sp} & A_{ss} \end{matrix} \right]
            \left [ \begin{matrix} x_p \\ x_s \end{matrix}\right]
            = \left [ \begin{matrix} b_p \\ b_s \end{matrix}\right],

        the method returns the whole vector :math:`[x_p, x_s]`, where

        .. math::
            x_s = A_{ss}^{-1} * (b_s - A_{sp} * x_p).

        Note:
            Independent of how the primary and secondary blocks were chosen, this method
            always returns a vector of size ``num_dofs``.
            Especially when the primary and secondary variables did not constitute the
            whole vector of unknowns, the result is still of size ``num_dofs``.
            The entries corresponding to the excluded grid variables are zero.

        Parameters:
            reduced_solution: Solution to the linear system returned by
                :meth:`assemble_schur_complement_system`.

        Returns:
            The expanded Schur solution in global size.

        Raises:
            ValueError: If the Schur complement system was not assembled before.

        """
        if self._Schur_complement is None:
            raise ValueError("Schur complement system was not assembled before.")

        # Get data stored from last constructed Schur complement.
        inv_A_ss, b_s, A_sp, prolong_p, prolong_s = self._Schur_complement

        # Calculate the complement solution.
        x_s = inv_A_ss * (b_s - A_sp * reduced_solution)

        # Prolong primary and secondary block to global-sized arrays
        X = prolong_p * reduced_solution + prolong_s * x_s
        return X

    ### Special methods ----------------------------------------------------------------

    def __repr__(self) -> str:
        s = (
            "EquationSystem for mixed-dimensional grid with "
            f"{self.mdg.num_subdomains()} subdomains "
            f"and {self.mdg.num_interfaces()}"
            " interfaces.\n"
        )
        # Sort variables alphabetically, not case-sensitive
        all_variables = set([var.name for var in self.variables])
        s += "Variables present on at least one grid or interface:\n\t"
        s += ", ".join(all_variables) + "\n"

        if self._equations is not None:
            eq_names = [name for name in self._equations]
            s += f"In total {len(self._equations)} equations, with names: \n\t"
            s += ", ".join(eq_names)

        return s

    def __str__(self) -> str:
        s = (
            "EquationSystem for mixed-dimensional grid with "
            f"{self.mdg.num_subdomains()} subdomains "
            f"and {self.mdg.num_interfaces()}"
            " interfaces.\n"
        )

        all_variables: set[str] = set([var.name for var in self.variables])
        variable_grid: dict[str, list[pp.GridLike]] = {var: [] for var in all_variables}
        for var in self.variables:
            variable_grid[var.name].append(var.domain)

        s += (
            f"There are in total {len(all_variables)} variables,"
            + " distributed as follows:\n"
        )

        # Sort variables alphabetically, not case-sensitive
        for var_name, grids in variable_grid.items():
            s += "\t" + f"{var_name} is present on"
            if isinstance(grids[0], pp.Grid):
                assert all([isinstance(g, pp.Grid) for g in grids])
                sorted_grids = self.mdg.sort_subdomains(grids)  # type: ignore
                s += " subdomains with id: " + ", ".join(
                    [str(g.id) for g in sorted_grids]
                )
            else:
                assert all([isinstance(g, pp.MortarGrid) for g in grids])
                sorted_grids = self.mdg.sort_interfaces(grids)  # type: ignore
                s += " interfaces with id: " + ", ".join(
                    [str(g.id) for g in sorted_grids]
                )

        s += "\n"
        if self._equations is not None:
            eq_names = [name for name in self._equations]
            s += f"In total {len(self._equations)} equations, with names: \n\t"
            s += "\n\t".join(eq_names) + "\n"

        return s
