"""Contains the EquationSystem, managing variables and equations for a system modelled
using the AD framework.

"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from . import _ad_utils
from .operators import MixedDimensionalVariable, Operator, Variable

__all__ = ["EquationSystem"]

GridLike = Union[pp.Grid, pp.MortarGrid]
"""A union type representing a domain either by a grid or mortar grid.
FIXME: Rename to Domain? Or GridLikeList/GridList below?"""

DomainList = Union[list[pp.Grid], list[pp.MortarGrid]]
"""A union type representing a list of grids or mortar grids.
This is *not* a list of GridLike, as that would allow a list of mixed grids and
mortar grids."""

VariableList = Union[list[str], list[Variable], list[MixedDimensionalVariable]]
"""A union type representing variables through either names (:class:`str`), multiple
:class:`~porepy.numerics.ad.operators.Variable` or
:class:`~porepy.numerics.ad.operators.MixedDimensionalVariable`.

This type is accepted as input to various methods and parsed to a list of
:class:`~porepy.numerics.ad.operators.Variable` using
:meth:`~porepy.numerics.ad.equation_system.EquationSystem._parse_variable_list`.

"""

EquationList = Union[list[str], list[Operator]]
"""A union type representing equations through either names (:class:`str`), or
:class:`~porepy.numerics.ad.operators.Operator`.

This type is accepted as input to various methods and parsed to a list of
:class:`~porepy.numerics.ad.operators.Operator` using
:meth:`~porepy.numerics.ad.equation_system.EquationSystem._parse_equations`.

"""

EquationRestriction = dict[Union[str, Operator], DomainList]
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

    This class provides functionalities to create and manage variables,
    as well as managing equations in AD operator form.

    It further provides functions to assemble subsystems and using subsets of equations and
    variables.

    Notes:
        As of now, the system matrix (Jacobian) is assembled with respect to ALL variables
        and then the columns belonging to the requested subset of variables and grids are
        sliced out and returned. This will be optimized with minor changes to the AD operator
        class and its recursive forward AD mode in the future.

        Currently, this class optimizes the block structure of the Jacobian only regarding the
        subdomains and interfaces. A more localized optimization
        (e.g. cell-wise for equations without spatial differential operators) is not performed.

    Parameters:
        mdg: mixed-dimensional grid representing the whole computational domain.

    """

    admissible_dof_types: tuple[
        Literal["cells"], Literal["faces"], Literal["nodes"]
    ] = ("cells", "faces", "nodes")
    """A set denoting admissible types of local DOFs for variables.

    - nodes: DOFs per grid node
    - cells: DOFs per grid cell (center)
    - faces: DOFS per grid face (center)

    """

    def __init__(self, mdg: pp.MixedDimensionalGrid) -> None:

        ### PUBLIC
        self.mdg: pp.MixedDimensionalGrid = mdg
        """Mixed-dimensional domain passed at instantiation."""

        self.assembled_equation_indices: dict[str, np.ndarray] = dict()
        """Contains the row indices in the last assembled (sub-) system for a given equation
        name (key). This dictionary changes with every call to any assemble-method.
        """

        ### PRIVATE

        self._equations: dict[str, Operator] = dict()
        """Contains references to equations in AD operator form for a given name (key).
        Private to avoid having people setting equations directly and circumventing the current
        set-method which includes information about the image space.

        """

        self._equation_image_space_composition: dict[
            str, dict[GridLike, np.ndarray]
        ] = dict()
        """Contains for every equation name (key) a dictionary, which provides again for
        every involved grid (key) the indices of equations expressed through the
        equation operator. The ordering of the items in the grid-array dictionaries is
        consistent with the remaining PorePy framework. The ordering is local to the
        equation, so it can be used to slice an eqution prior to concatenation of
        equations into a global matrix.

        """

        self._equation_image_size_info: dict[str, dict[GridEntity, int]] = dict()
        """Contains for every equation name (key) the argument the number of equations
        per grid entity.

        """

        self._variables: list[Variable] = list()
        """Contains references to grid Variables. A Variable is uniquely identified by its
        name and domain, stored as attributes of the Variable object.

        """

        self._Schur_complement: Optional[tuple] = None
        """Contains block matrices and the split rhs of the last assembled Schur complement,
        such that the expansion can be made. FIXME: What does this line mean?

        """

        self._variable_numbers: dict[Variable, int] = dict()
        """Dictionary containing the index of the variable in the system vector of the last
        assembled system.

        """

        self._variable_num_dofs: np.ndarray = np.array([], dtype=int)
        """Array containing the number of DOFS per block number. The block number corresponds
        to this array's indexation, see also _variable_numbers.

        """

        self._variable_dof_type: dict[Variable, dict[str, int]] = dict()

    def SubSystem(
        self,
        equation_names: Optional[EquationList] = None,
        variable_names: Optional[VariableList] = None,
    ) -> EquationSystem:
        """Creates a ``EquationSystem`` for a given subset of equations and variables.

        Currently only subsystems containing *whole* equations and variables in the
        mixed-dimensional sense can be created. Restrictions of equations to subdomains
        is not supported.

        Parameters:
            equation_names: Names of equation for the new subsystem. If None, all
                equations known to the EquationSystem is used.
            variable_names: Names of known variables for the new subsystem. If None, all
                variables known to the EquationSystem is used.

        Returns:
            A new instance of ``EquationSystem``. The subsystem equations and variables
                are ordered as imposed by this systems's order.

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
        unknown_variables = set(variables).difference(self._variables)
        if len(unknown_variables) > 0:
            raise ValueError(f"Unknown variable(s) {unknown_variables}.")

        # Create the new subsystem.
        new_manager = EquationSystem(self.mdg)

        # IMPLEMENTATION NOTE: This method imitates the variable creation and equation
        # setting procedures by calling private methods and accessing private
        # attributes. This should be acceptable since this is a factory method.

        # Loop over known variables to preserve DOF order.
        for variable in self._variables:
            if variable in variables:
                # Update variables in subsystem.
                new_manager._variables.append(variable)

                # Update variable numbers in subsystem.
                new_manager._variable_dof_type[variable] = self._variable_dof_type[
                    variable
                ]

                # Create dofs in subsystem.
                new_manager._append_dofs(variable)

        # Loop over known equations to preserve row order.
        for name in known_equations:
            if name in equations:
                equation = self._equations[name]
                image_info = self._equation_image_size_info[name]
                image_composition = self._equation_image_space_composition[name]
                # et the information produced in set_equations directly.
                new_manager._equation_image_space_composition.update(
                    {name: image_composition}
                )
                new_manager._equation_image_size_info.update({name: image_info})
                new_manager._equations.update({name: equation})

        return new_manager

    @property
    def equations(self) -> dict[str, Operator]:
        """Dictionary containing names of operators (keys) and operators (values), which
        have been set as equations in this system.

        """
        return self._equations

    @property
    def variables(self) -> list[Variable]:
        """List containing all Variables known to this system."""
        return self._variables

    @property
    def variable_domains(self) -> DomainList:
        """List containing all domains where at least one variable is defined."""
        # EK: There is a nuance here, from the docstring, I expected something like
        # return self.mdg.subdomains() + self.mdg.interfaces(). I'm fine with having a
        # method that considers only GridLike where the variables are defined, but I
        # think we should reconsider the name.
        # IS: Fair. Done. You might want an equiation_domain method.
        domains = set()
        for var in self._variables:
            domains.add(var.domain)
        return list(domains)

    ### Variable management ------------------------------------------------------------

    def md_variable(
        self, name: str, grids: Optional[DomainList] = None
    ) -> MixedDimensionalVariable:
        """Create a mixed-dimensional variable for a given name-domain list combination.

        Parameters:
            name (str): Name of the mixed-dimensional variable.
            grids (optional): List of grids where the variable is defined. If None (default),
                all grids where the variable is defined are used.

        Returns:
            MixedDimensionalVariable: The mixed-dimensional variable.

        Raises:
            ValueError if variables called name exist on both grids and interfaces and
                domain type is not specified (grids is None).
        """
        if grids is None:
            variables = [var for var in self._variables if var.name == name]
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
                for var in self._variables
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
        subsequent step (using e.g. :meth:`set_var_values`).

        Notes:
            This method provides support for creating variables on **all** subdomains
            or interfaces, without having to pass them all as arguments.
            If the argument ``subdomains`` is an empty list, the method will use all
            subdomains found in the mixed-dimensional grid.
            If the argument ``interfaces`` is an empty list, the method will use all
            interfaces found in the mixed-dimensional list.

        Examples:
            An example on how to define a pressure variable with cell-wise one DOF
            (default) on **all** subdomains and **no** interfaces would be

            >>> p = ad_system.create_variables('pressure', subdomains=mdg.subdomains())

        Parameters:
            name: used here as an identifier. Can be used to associate the variable with
                some physical quantity like ``'pressure'``.
            dof_info: dictionary containing information about number of DOFs per
                admissible type. Defaults to ``{'cells':1}``.
            subdomains (optional): list of subdomains on which the variable is defined.
                If None, then it will not be defined on any subdomain.
            interfaces (optional): list of interfaces on which the variable is defined.
                If None, then it will not be defined on any interface.
            tags (optional): dictionary containing tags for the variables. The tags are
                assigned to all variables created by this method and can be updated
                using :meth:`update_variable_tags`.

        Returns:
            a mixed-dimensional variable with above specifications.

        Raises:
            ValueError: if non-admissible DOF types are used as local DOFs.
            ValueError: if one attempts to create a variable not defined on any grid.
            KeyError: if a variable with given name is already defined.

        """
        # Set default value for dof_info. This is a mutable object, so we need to
        # create a new one each time and not set the default in the signature.
        if dof_info is None:
            dof_info = {"cells": 1}

        # sanity check for admissible DOF types
        requested_type = set(dof_info.keys())
        if not requested_type.issubset(set(self.admissible_dof_types)):
            non_admissible = requested_type.difference(self.admissible_dof_types)
            raise ValueError(f"Non-admissible DOF types {non_admissible} requested.")

        # sanity check if variable is defined anywhere
        if subdomains is None and interfaces is None:
            raise ValueError(
                "Cannot create variable not defined on any subdomain or interface."
            )

        # check if a md variable was already defined under that name
        if name in self._variables:
            raise KeyError(f"Variable with name '{name}' already defined.")

        # container for all grid variables
        variables = list()

        # Merge subdomains and interfaces into a single list
        assert subdomains is not None or interfaces is not None  # for mypy
        grids: DomainList = subdomains if subdomains else interfaces
        if grids:

            for grid in grids:
                # check if the grid is known to the system

                if subdomains:
                    assert isinstance(grid, pp.Grid)  # mypy
                    data = self.mdg.subdomain_data(grid)
                else:
                    assert isinstance(grid, pp.MortarGrid)  # mypy
                    data = self.mdg.interface_data(grid)

                # prepare data dictionary if this was not done already
                if pp.STATE not in data:
                    data[pp.STATE] = dict()
                if pp.ITERATE not in data[pp.STATE]:
                    data[pp.STATE][pp.ITERATE] = dict()

                # create grid variable
                new_variable = Variable(name, dof_info, domain=grid, tags=tags)

                # Store it in the system
                variables.append(new_variable)
                self._variables.append(new_variable)

                # append the new DOFs to the global system
                self._variable_dof_type[new_variable] = dof_info
                self._append_dofs(new_variable)

        # Create an md variable that wrapps all the individual variables created on
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
                interpreted as all variables.

                NOTE: If a mixed-dimensional variable is passed, the tags will be
                assigned to its sub-variables (living on individual grids).

        """
        assert isinstance(variables, list)

        variables = self._parse_variable_type(variables)
        for var in variables:
            var.tags.update(tags)

    def get_variables(
        self,
        variables: Optional[list[Variable]] = None,
        grids: Optional[list[pp.Grid]] = None,
        tag_name: Optional[str] = None,
        tag_value: Optional[Any] = None,
    ) -> list[Variable]:
        """Filter variables based on grid, tag name and tag value.

        Particular usage: calling without arguments will return all variables in the system.
        FIXME: Rename to filter_variables? Remove variables property? The property is covered
        by this method, which might be renamed to variables if property is removed.

        EK: I kept the name. It seems fair to me to keep the property, it is a simple code.

        Parameters:
            variables: list of variables to filter. If None, all variables in the system
                are included.
            grids: list of grids to filter on. If None, all grids are included.
            tag_name: name of the tag to filter on. If None, no filtering on tags.
            tag_value: value of the tag to filter on. If None, no filtering on tag values.
                If tag_name is not None, but tag_value is None, all variables with the given
                tag_name are returned regardless of value.

        Returns:
            list of filtered variables.

        """
        # Shortcut for efficiency.
        # The same behavior is achieved without this, but it is slower.
        if (
            variables is None
            and grids is None
            and tag_name is None
            and tag_value is None
        ):
            return self._variables

        # If no variables or grids are given, use full sets.
        if variables is None:
            variables = self._variables
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
        self, variables: Optional[VariableList] = None, from_iterate: bool = False
    ) -> np.ndarray:
        """Assembles an array containing values for the passed variable-like argument.

        The global order is preserved and independent of the order of the argument.

        Notes:
            The resulting array is of any size between 0 and ``num_dofs``.

        Parameters:
            variables (optional): VariableType input for which the values are
                requested. If None (default), the global vector of unknowns is returned.
            from_iterate (optional): flag to return values stored as ITERATE,
                instead of STATE (default).

        Returns:
            the respective (sub) vector in numerical format.

        Raises:
            KeyError: if no values are stored for the VariableType input.
            ValueError: if unknown VariableType arguments are passed.

        """
        variables = self._parse_variable_type(variables)
        # storage for atomic blocks of the sub vector (identified by name-grid pairs)
        values = list()

        # loop over all blocks and process those requested
        # this ensures uniqueness and correct order
        for variable in self._variable_numbers:
            if variable in variables:
                name = variable.name
                grid = variable.domain
                if isinstance(grid, pp.Grid):
                    data = self.mdg.subdomain_data(grid)
                elif isinstance(grid, pp.MortarGrid):
                    data = self.mdg.interface_data(grid)
                # extract a copy of requested values
                try:
                    if from_iterate:
                        values.append(data[pp.STATE][pp.ITERATE][name].copy())
                    else:
                        values.append(data[pp.STATE][name].copy())
                except KeyError:
                    raise KeyError(
                        f"No values stored for variable {name}, "
                        f"from_iterate={from_iterate}"
                        f"\non grid {grid}."
                    )

        # if there are matching blocks, concatenate and return
        if values:
            return np.concatenate(values)
        # else return an empty vector
        else:
            return np.array([])

    def set_variable_values(
        self,
        values: np.ndarray,
        variables: Optional[VariableList] = None,
        to_state: bool = False,
        to_iterate: bool = False,
        additive: bool = False,
    ) -> None:
        """Sets values for a (sub) vector of the global vector of unknowns specified by
        ``variables``.

        The order of values is assumed to fit the global order.

        Notes:
            The vector is assumed to be of proper size and will be dissected according
            to the global order, starting with the index 0.
            Mismatches of is-size and should-be-size according to the subspace specified
            by ``variables`` will raise respective errors by numpy.

        Parameters:
            values: vector of corresponding size.
            variables (optional): VariableType input for which the values are
                 requested. If None (default), the global vector of unknowns will be
                 set.
            to_state (optional): Flag to write values to STATE.
            to_iterate (optional): Flag to write values to ITERATE.

            additive (optional): flag to write values *additively* to ITERATE or STATE.
                To be used in iterative procedures.

        Raises:
            ValueError: if unknown VariableType arguments are passed.

        """
        # Start of dissection.
        dof_start = 0
        dof_end = 0
        variables = self._parse_variable_type(variables)
        for variable, variable_number in self._variable_numbers.items():
            if variable in variables:
                name = variable.name
                grid = variable.domain
                num_dofs = int(self._variable_num_dofs[variable_number])
                dof_end = dof_start + num_dofs
                # Extract local vector.
                # This will raise errors if indexation is out of range.
                local_vec = values[dof_start:dof_end]
                # Fetch the storage from the relevant dicitonary in the
                # MixedDimensionalGrid.
                if isinstance(grid, pp.Grid):
                    data = self.mdg.subdomain_data(grid)
                elif isinstance(grid, pp.MortarGrid):
                    data = self.mdg.interface_data(grid)

                # Data dict will have pp.STATE and pp.ITERATE entries already created
                # during create_variables. If an error is returned here, a variable has
                # been created in a non-standard way.
                # Store new values as requested.
                if additive:
                    if to_iterate:
                        # No need for a copy here, since we are adding to an existing
                        # array.
                        data[pp.STATE][pp.ITERATE][name] += local_vec
                    if to_state:
                        data[pp.STATE][name] += local_vec
                else:
                    if to_iterate:
                        # The copy is critcial here.
                        data[pp.STATE][pp.ITERATE][name] = local_vec.copy()
                    if to_state:
                        data[pp.STATE][name] = local_vec.copy()

                # Move dissection forward.
                dof_start = dof_end

        # Last sanity check if the vector was properly sized, or if it was too large.
        # This imposes a theoretically unnecessary restriction on the input argument
        # since we only require a vector of at least this size.
        assert dof_end == values.size

    ### DOF management -----------------------------------------------------------------

    def _append_dofs(self, variable: pp.ad.Variable) -> None:
        """Appends DOFs for a newly created variable.

        Must only be called by :meth:`create_variables`.

        This method defines a preliminary global order of dofs:

        Parameters:
            variable: The newly created variable

        """
        # number of totally created dof blocks so far
        last_variable_number: int = len(self._variable_numbers)

        # Sanity check that no previous data is overwritten. This should not happen,
        # if class not used in hacky way.
        assert variable not in self._variable_numbers

        # Count number of dofs for this variable on this grid and store it.
        # The number of dofs for each dof type defaults to zero.

        local_dofs = self._variable_dof_type[variable]
        # Both subdomains and interfaces have cell variables.
        num_dofs = variable.domain.num_cells * local_dofs.get("cells", 0)

        # For subdomains, but not interfaces, we also need to account for faces and
        # nodes.
        if isinstance(variable.domain, pp.Grid):
            num_dofs += variable.domain.num_faces * local_dofs.get(
                "faces", 0
            ) + variable.domain.num_nodes * local_dofs.get("nodes", 0)

        # Update the global dofs and block numbers
        self._variable_numbers.update({variable: last_variable_number})
        self._variable_num_dofs = np.concatenate(
            [self._variable_num_dofs, np.array([num_dofs], dtype=int)]
        )

        # first optimization of Jacobian structure
        self._cluster_dofs_gridwise()

    def _cluster_dofs_gridwise(self) -> None:
        """Re-arranges the DOFs grid-wise s.t. we obtain grid-blocks in the column sense
        and reduce the matrix bandwidth.

        The aim is to impose a more block-diagonal-like structure on the Jacobian where
        blocks in the column sense represent single grids in the following order:

        Notes:
            Off-diagonal blocks will still be present if subdomain-interface variables are
            defined.

        1. For each grid in ``mdg.subdomains``
            1. For each variable defined on that grid
        2. For each grid in ``mdg.interfaces``
            1. For each variable defined on that mortar grid

        The order of variables per grid is given by the order of variable creation
        (stored as order of keys in ``self.variables``).

        This method is called after each creation of variables and respective DOFs.
        TODO: Revisit. I think I have broken it by looping over _variables instead of
        subdomains and interfaces.
        """
        # Data stracture for the new order of dofs.
        new_variable_counter: int = 0
        new_variable_numbers: dict[Variable, int] = dict()
        new_block_dofs: list[int] = list()

        for variable in self._variables:
            # If this variable-grid combination is present, add it to the new
            # order of dofs.
            if variable in self._variable_numbers:
                # Extract created number of dofs
                local_dofs: int = self._variable_num_dofs[
                    self._variable_numbers[variable]
                ]

                # Store new block number and dofs in new order.
                new_block_dofs.append(local_dofs)
                new_variable_numbers.update({variable: new_variable_counter})
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
            variables (dict or list): The input argument for the variable type.
                The following interpretation rules are applied:
                    - If None, return all variables
                    - If a list of variables, return same
                    - If a list of strings, return all variables with those names
                    - If mixed-dimensional variable, return sub-variables

        Returns:
            list: A list of Variables.

        """
        if variables is None:
            return self.get_variables()
        parsed_variables = []
        for variable in variables:
            if isinstance(variable, MixedDimensionalVariable):
                parsed_variables += variable.sub_vars
            elif isinstance(variable, Variable):
                parsed_variables.append(variable)
            elif isinstance(variable, str):
                # Use _variables to avoid recursion (get_variables() calls this method)
                vars = [var for var in self._variables if var.name == variable]
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
                    md_variable = self._variables[variable.name]
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
        """Get the indices in the global vector of unknowns belonging to the variable(s).

        Parameters:
            variables: VariableType input for which the indices are requested.

        Returns:
            an order-preserving array of indices of DOFs belonging to the VariableType input.

        Raises:
            ValueError: if unknown VariableType arguments are passed.

        """
        variables = self._parse_variable_type(variables)
        global_variable_dofs = np.hstack((0, np.cumsum(self._variable_num_dofs)))

        # Storage of indices per requested variable.
        indices = list()
        for variable in variables:
            var_number = self._variable_numbers[variable]
            var_indices = np.arange(
                global_variable_dofs[var_number],
                global_variable_dofs[var_number + 1],
                dtype=int,
            )
            indices.append(var_indices)

        # Concatenate indices, if any
        if indices:
            return np.concatenate(indices, dtype=int)
        else:
            return np.array([], dtype=int)

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
        variable = [
            var for var, num in self._variable_numbers.items() if num == variable_number
        ][0]
        return variable

    ### Equation management -------------------------------------------------------------------

    def set_equation(
        self,
        equation: Operator,
        grids: DomainList,
        equations_per_grid_entity: dict[GridEntity, int],
    ) -> None:
        """Sets an equation using the passed operator **and uses its name as an identifier**.

        If an equation already exists under that name, it is overwritten.

        Information about the image space must be provided for now, such that grid-wise
        row slicing is possible. This will hopefully be provided automatically in the
        future.

        Notes:
            Regarding the number of equations, this method assumes that the AD framework
            assembles row blocks per grid in subdomains, then per grid in interfaces,
            for each operator representing an equation. This is assumed to be the way
            PorePy AD works.

        Parameters:
            equation: An equation in AD operator form, assuming the right-hand side is
                zero and this instance represents the left-hand side.
                **The equation must be ready for evaluation,
                i.e. all involved variables must have values set.**
                An equation can be defined on sets of subdomains or interfaces, but
                not on a combination.
            grids: A list of grids on which the equation is defined.
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
        grids = list(grids)
        # The function loops over all grids the operator is defined on and calculate the
        # number of equations per grid quantity (cell, face, node). This information
        # is then stored together with the equation itself.
        image_info: dict[GridLike, np.ndarray] = dict()
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
        # FIXME: Is it allowed to have no domain or should we change to == 1?
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
                grids.remove(sd)

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
                grids.remove(intf)

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

    ### System assembly and discretization ----------------------------------------------------

    @staticmethod
    def _recursive_discretization_search(operator: Operator, discr: list) -> list:
        """Recursive search in the tree of this operator to identify all discretizations
        represented in the operator.

        Parameters:
            operator: top level operator to be searched.
            discr: list storing found discretizations

        """
        if len(operator.tree.children) > 0:
            # Go further in recursion
            for child in operator.tree.children:
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
                # If indices not empty, concatenate and return
                if block_idx:
                    block.update({name: np.concatenate(block_idx, dtype=int)})
                # indices should by logic always be found, if not alert the user.
                else:
                    # TODO: Should this not be permissible, say, due to a filtering of
                    # the grids? However, it may lead to errors downstream.
                    raise TypeError(
                        f"Equation-like item ({type(equ)}, {type(grids)}) yielded no rows."
                    )
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

        Args:
            equations: dictionary with equation names as keys and the corresponding

            TODO!!!

        """
        complement: dict[str, None | np.ndarray] = dict()
        for name, idx in equations.items():
            # If indices where filtered based on grids, we find the complementing
            # indices.
            # If idx is None, this means no filtering was done.
            if idx:
                # Get the indices associated with this equation.
                img_info = self._equation_image_space_composition[name]

                # Assure ordering and uniqueness whole equation indexation
                all_idx = np.unique(np.hstack(img_info.values()))

                # Complementing indices are found by deleting the filtered indices
                complement_idx = np.delete(all_idx, idx)
                complement.update({name: complement_idx})

            # If there was no grid-based row filtering, the complement is empty.
            else:
                complement.update({name: None})
        return complement

    def discretize(self, equations: Optional[EquationLike] = None) -> None:
        """Find and loop over all discretizations in the equation operators, extract
        unique references and discretize.

        This is more efficient than discretizing on the Operator level, since
        discretizations which occur more than once in a set of equations will be
        identified and only discretized once.

        Parameters:
            equations (optional): a subset of equations.
                If not provided (None), all known equations will be discretized.

        """
        equation_names = list(self._parse_equations(equations).keys())

        # List containing all discretizations
        discr: list = []
        # TODO: the search can be done once (in some kind of initialization)
        for name in equation_names:
            # this raises a key error if a given equation name is unknown
            eqn = self._equations[name]
            # This will expand the list discr with new discretizations.
            # The list may contain duplicates.
            discr += self._recursive_discretization_search(eqn, list())

        # Uniquify to save computational time, then discretize.
        unique_discr = _ad_utils.uniquify_discretization_list(discr)
        _ad_utils.discretize_from_list(unique_discr, self.mdg)

    def assemble(
        self,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector of the whole system.

        This is a shallow wrapper of :meth:`assemble_subsystem`, where the subsystem is
        the complete set of equations, variables and grids.

        Parameters:
            state (optional): see :meth:`assemble_subsystem`. Defaults to None.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the targeted state.
                The ordering of the equations (rows) is determined by the order the
                equations were added. The DOFs (columns) are ordered according the
                global order.
            np.ndarray: Residual vector corresponding to the targeted state,
                scaled with -1 (moved to rhs).

        """
        return self.assemble_subsystem(state=state)

    def assemble_subsystem(
        self,
        equations: Optional[EquationList | EquationRestriction] = None,
        variables: Optional[VariableList] = None,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector using a specified subset of
        equations, variables and grids.

        The method is intended for use in splitting algorithms. Matrix blocks not
        included will simply be sliced out.

        Notes:
            The ordering of columns in the returned system are defined by the global
            DOF. The row blocks are in the same order as equations were added to this
            system. If an equation is defined on multiple grids, the respective
            row-block is internally ordered as given by the mixed-dimensional grid
            (for sd in subdomains, for intf in interfaces).

            The columns of hthe subsystem are assumed to be properly defined by
            ``variables``, otherwise a matrix of shape ``(M,)`` is returned. This
            happens if grid variables
            are passed which are unknown to this EquationSystem.

        Parameters:
            equations (optional): a subset of equations to which the subsystem should be
                restricted. If not provided (None), all equations known to this
                equation system manager will be included.

                The user can specify grids per equation (name) to which the subsystem should be
                restricted in the row-sense. Grids not belonging to the domain of an equation
                will raise an error.

            variables (optional): VariableType input specifying the subspace in column-sense.
                If not provided (None), all variables will be included.
            state (optional): State vector to assemble from. By default, the stored ITERATE or
                STATE are used, in that order.

        Returns:
            spmatrix: (Part of the) Jacobian matrix corresponding to the targeted variable
                state, for the specified equations and variables.
            ndarray: Residual vector corresponding to the targeted variable state,
                for the specified equations. Scaled with -1 (moved to rhs).

        """
        if variables is None:
            variables = self._variables

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
        self.assembled_equation_indices = dict()

        # Iterate over equations, assemble.
        # Also keep track of the row indices of each equation, and store it in
        # assembled_equation_indices.
        for equ_name, rows in equ_blocks.items():
            # This will raise a key error if the equation name is unknown.
            eq = self._equations[equ_name]
            ad = eq.evaluate(self, state)

            # If restriction to grid-related row blocks was made,
            # perform row slicing based on information we have obtained from parsing.
            if rows:
                mat.append(ad.jac[rows])
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
            ind_start = block_indices[-1] + 1
            self.assembled_equation_indices.update({equ_name: block_indices})

        # Concatenate results equation-wise
        if len(mat) > 0:
            A = sps.vstack(mat, format="csr")
            rhs_cat = np.concatenate(rhs)
        else:
            # Special case if the restriction produced an empty system.
            A = sps.csr_matrix((0, self.num_dofs()))
            rhs_cat = np.empty(0)

        # Slice out the columns belonging to the requested subsets of variables and
        # grid-related column blocks by using the transposed projection to respective
        # subspace.
        # Multiply rhs by -1 to move to the rhs.
        column_projection = self.projection_to(variables).transpose()
        return A * column_projection, -rhs_cat

    def assemble_schur_complement_system(
        self,
        primary_equations: EquationLike,
        primary_variables: VariableList,
        secondary_equations: Optional[EquationLike] = None,
        secondary_variables: Optional[VariableList] = None,
        excl_loc_prim_to_sec: bool = False,
        inverter: Callable[[sps.spmatrix], sps.spmatrix] = sps.linalg.inv,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector using a Schur complement
        elimination of the variables and equations not to be included.

        The specified equations and variables will define blocks of the linearized
        system as

            ``[A_pp, A_ps  [x_p   = [b_p``
            `` A_sp, A_ss]  x_s]     b_s]``

        where subscripts p and s define primary and secondary blocks.
        The Schur complement system is then given by

            ``(A_pp - A_ps * inv(A_ss) * A_sp) * x_p = b_p - A_ps * inv(A_ss) * b_s``

        The Schur complement is well-defined only if the inverse of A_ss exists,
        and the efficiency of the approach assumes that an efficient inverter for
        A_ss can be found.
        **The user must ensure both requirements are fulfilled.**

        Notes:
            The optional arguments defining the secondary block, and the flag
            ``excl_loc_prim_to_sec`` are meant for nested Schur-complements and
            splitting solvers. This is an advanced usage and requires the user to be
            careful, since the resulting blocks ``A_pp`` and ``A_ss`` might end up
            to be not square. This will result in errors.

        Examples:
            The default inverter can be defined by

            >>> import scipy.sparse as sps
            >>> inverter = lambda A: sps.csr_matrix(sps.linalg.inv(A.A))

            It is costly in terms of computational time and memory though.

            TODO: We should rather use the block inverter in pp.matrix_operations. This
            will require some work on ensuring the system is block-diagonal.

        Parameters:
            primary_equations: a subset of equations specifying the primary subspace in
                row-sense.
            primary_variables: VariableType input specifying the primary subspace in
                column-sense.
            secondary_equations: a subset of equations specifying the secondary subspace in
                row-sense.
                By default, the complement of the primary rows is used.
            secondary_variables: VariableType input specifying the secondary subspace in
                column-sense.
                By default, the complement of the primary columns is used.
            excl_loc_prim_to_sec (optional): If True, primary local blocks which are excluded
                by the variable- and equation structure, are added to the secondary block.

                I.e. if a variable ``p`` is defined on two grids ``sd1, sd2``, and the user
                defines the primary (column) block to be only given by ``p`` on ``sd1``,
                then the (column) block corresponding to ``p`` on ``sd2`` will be added to the
                secondary block.
                Analogously for equations (row blocks), which are defined on multiple grids.
                The flag acts in both column and row sense.

                If False (default), they will not be included and the union of primary and
                secondary blocks will **not** constitute the whole system.
            inverter (optional): callable object to compute the inverse of the matrix A_ss.
                By default, the scipy direct sparse inverter is used.
            state (optional): see :meth:`assemble_subsystem`. Defaults to None.

        Returns:
            sps.spmatrix: Jacobian matrix representing the Schur complement with respect to
                the targeted state.
            np.ndarray: Residual vector for the Schur complement with respect to the targeted
                state. Scaled with -1 (moved to rhs).

        Raises:
            AssertionError:
                - if the primary block would have 0 rows or columns
                - if the secondary block would have 0 rows or columns
                - if the secondary block is not square
            ValueError: if primary and secondary columns overlap
        """
        # Find the rows of the primary block. This can include both equations defined
        # on their full image, and equations specified on a subset of grids.
        # The variable primary_rows will contain the indices in the global system
        # corresponding to the primary block.
        primary_rows = self._parse_equations(primary_equations)
        # Find indices of equations involved in the primary block, but on grids that
        # were filtered out.
        excl_prim_rows = self._gridbased_equation_complement(primary_rows)

        # Names of equations that form the primary block.
        prim_equ_names = list(primary_rows.keys())

        # Projection of variables to the set of primary blocks.
        # TODO: If the user has specified a subset of grids, the variables must be
        # adjusted accordingly. EK thinks this is not implemented, but it should be
        # straightforward if we implement a method variables_on_subset_of_grids().
        # UPDATE: Something like this is done below (see if on excl_loc_prim_to_sec).
        # It seems this will do the trick, but it should be included in tests.
        primary_projection = self.projection_to(primary_variables)
        num_dofs = primary_projection.shape[1]

        # Assert non-emptiness of primary block.
        assert len(primary_rows) > 0
        assert primary_projection.shape[0] > 0

        # Finding secondary column indices and respective projection.
        if secondary_variables:
            # Default projection to secondaries.
            # TODO: This must likely be supplemented with primary variables defined on
            # excluded grids, see comment regarding primary variables.
            secondary_projection = self.projection_to(secondary_variables)
            # Assert primary and secondary columns do not overlap
            common_column_indices: np.ndarray = np.intersect1d(
                primary_projection.indices, secondary_projection.indices
            )
            if common_column_indices.size > 0:
                raise ValueError("Primary and secondary columns overlap.")

            # find indices of excluded primary columns and change the secondary projection
            if excl_loc_prim_to_sec:
                # finding grid variables, who are primary in terms of name, but excluded by the
                # filter the VariableType structure imposes
                # TODO: The workings of the below function is unclear to EK.
                # TODO: We need to revisit this whole if-block after having decided how
                # variables should be represented.
                excluded_grid_variables = self._gridbased_variable_complement(
                    primary_variables
                )
                excl_projection = self.projection_to(excluded_grid_variables)
                # take the indices of the excluded local prims and all secs
                idx_s = np.unique(
                    np.hstack([excl_projection.indices, secondary_projection.indices])
                )
                shape = (idx_s.size, num_dofs)
                # re-compute the secondary projection including new indices
                secondary_projection = sps.coo_matrix(
                    (np.ones(shape[0]), (np.arange(shape[0]), idx_s)),
                    shape=shape,
                ).tocsr()
        else:
            # we use the complement of the indices in the primary projection
            pass

        # finding secondary row indices
        secondary_rows: dict[str, None | np.ndarray] | None
        sec_equ_names: list[str]
        shape = (
            primary_projection.shape[1] - primary_projection.shape[0],
            num_dofs,
        )
        if excl_loc_prim_to_sec:

            # remove indices found in primary projection
            # csr sparse projections have only one entry per column
            idx_s = np.delete(
                np.arange(shape[1], dtype=int), primary_projection.indices
            )
            assert len(idx_s) == shape[0]
            # TODO EK: Why is the projection defined above overwritten here?
            secondary_projection = sps.coo_matrix(
                (np.ones(shape[0]), (np.arange(shape[0]), idx_s)),
                shape=shape,
            ).tocsr()
        else:
            # finding grid vars, who are primary in terms of name, but excluded by the
            # filter the VariableType structure imposes
            excluded_grid_variables = self._gridbased_variable_complement(
                primary_variables
            )
            excl_projection = self.projection_to(excluded_grid_variables)
            # take the indices of the excluded local prims and included prims
            idx_excl = np.unique(
                np.hstack([excl_projection.indices, primary_projection.indices])
            )
            # the secondary indices are computed by the complement of above
            # FIXME: Define shape
            idx_s = np.delete(np.arange(shape[1], dtype=int), idx_excl)
            shape = (idx_s.size, num_dofs)
            secondary_projection = sps.coo_matrix(
                (np.ones(shape[0]), (np.arange(shape[0]), idx_s)),
                shape=shape,
            ).tocsr()
        if secondary_equations:
            secondary_rows = self._parse_equations(secondary_equations)
            sec_equ_names = list(secondary_rows.keys())
        else:
            secondary_rows = None
            sec_equ_names = list(
                set(self._equations.keys()).difference(set(primary_rows.keys()))
            )

        # check the primary and secondary system are not overlapping in terms of equations
        if len(set(prim_equ_names).intersection(set(sec_equ_names))) > 0:
            raise ValueError("Primary and secondary rows overlap.")
        # assert non-emptiness of secondary block
        assert secondary_projection.shape[0] > 0
        assert len(sec_equ_names) > 0

        # storage of primary and secondary row blocks
        A_sec: list[sps.csr_matrix] = list()
        b_sec: list[np.ndarray] = list()
        A_prim: list[sps.csr_matrix] = list()
        b_prim: list[np.ndarray] = list()
        # keep track of indices or primary block
        ind_start = 0
        assembled_equation_indices = dict()

        # we loop over stored equations to ensure the correct order
        # but process only primary equations
        # excluded local primary blocks are stored as top rows in the secondary block
        for name in self._equations:
            if name in prim_equ_names:
                A_temp, b_temp = self.assemble_subsystem([name], state=state)
                idx_p = primary_rows[name]
                # check if a grid filter was applied for that equation
                if idx_p:
                    # append the respective rows
                    A_prim.append(A_temp[idx_p])
                    b_prim.append(b_temp[idx_p])
                    # if requested, the excluded primary rows are appended as secondary
                    if excl_loc_prim_to_sec:
                        idx_excl_p = excl_prim_rows[name]
                        A_sec.append(A_temp[idx_excl_p])
                        b_sec.append(b_temp[idx_excl_p])
                # if no filter was applied, the whole row block is appended
                else:
                    A_prim.append(A_temp)
                    b_prim.append(b_temp)

                # track indices of block rows
                row_idx = np.arange(b_prim[-1].size, dtype=int)
                indices = row_idx + ind_start
                ind_start += row_idx.size
                assembled_equation_indices.update({name: indices})

        # store the assembled row indices for the primary block only (Schur)
        self.assembled_equation_indices = assembled_equation_indices

        # we loop again over stored equation to ensure a correct order
        # but process only secondary equations
        for name in self._equations:
            if name in sec_equ_names:
                A_temp, b_temp = self.assemble_subsystem([name], state=state)
                # if secondary equations were defined, we check if we have to restrict
                # them
                if secondary_rows:
                    idx_s = secondary_rows[name]
                    # slice or no slice
                    if idx_s:
                        A_sec.append(A_temp[idx_s])
                        b_sec.append(b_temp[idx_s])
                    else:
                        A_sec.append(A_temp)
                        b_sec.append(b_temp)
                # no slicing of secondary equations at all
                else:
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

        # last sanity check, if A_ss is square
        assert A_ss.shape[0] == A_ss.shape[1]

        # compute the inverse of A_ss using the passed inverter
        inv_A_ss = inverter(A_ss)

        S = A_pp - A_ps * inv_A_ss * A_sp
        rhs_S = b_p - A_ps * inv_A_ss * b_s

        # storing necessary information for Schur complement expansion
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
        """Expands the solution of the **last assembled** Schur complement system to the
        whole solution.

        I.e it takes x_p from

            [A_pp, A_ps  [x_p   = [b_p
             A_sp, A_ss]  x_s]     b_s]

        and returns the whole [x_p, x_s] where

            x_s = inv(A_ss) * (b_s - A_sp * x_p)

        Notes:
            Independent of how the primary and secondary blocks were chosen, this method always
            returns a vector of size ``num_dofs``.
            Especially when the primary and secondary variables did not constitute the whole
            vector of unknowns, the result is still of size ``num_dofs``.
            The entries corresponding to the excluded grid variables are zero.

        Parameters:
            reduced_solution: Solution to the linear system returned by
                :meth:`assemble_schur_complement_system`

        Returns: the expanded Schur solution in global size.

        Raises:
            AssertionError: if the Schur complement system was not assembled before.

        """
        assert self._Schur_complement is not None
        # get data stored from last complement
        inv_A_ss, b_s, A_sp, prolong_p, prolong_s = self._Schur_complement
        # calculate the complement solution
        x_s = inv_A_ss * (b_s - A_sp * reduced_solution)
        # prolong primary and secondary block to global-sized arrays
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

        all_variables = set([var.name for var in self.variables])
        variable_grid = {var: [] for var in all_variables}
        for var in self.variables:
            variable_grid[var.name].append(var.domain)

        s += (
            f"There are in total {len(all_variables)} variables,"
            + " distributed as follows:\n"
        )

        # Sort variables alphabetically, not case-sensitive
        for var, grids in variable_grid:
            s += "\t" + f"{var} is present on"
            if isinstance(grids[0], pp.Grid):
                sorted_grids = self.mdg.sort_subdomains(grids)
                s += " subdomains with id: " + ", ".join(
                    [str(g.id) for g in sorted_grids]
                )
            else:
                sorted_grids = self.mdg.sort_interfaces(grids)
                s += " interfaces with id: " + ", ".join(
                    [str(g.id) for g in sorted_grids]
                )

        s += "\n"
        if self._equations is not None:
            eq_names = [name for name in self._equations]
            s += f"In total {len(self._equations)} equations, with names: \n\t"
            s += "\n\t".join(eq_names) + "\n"

        return s
