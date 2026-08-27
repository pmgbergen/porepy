"""Contains the EquationSystem, managing variables and equations for a system modelled
using the AD framework.

"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal, Optional, Sequence, Union, overload
from warnings import warn

import numpy as np
import scipy.sparse as sps
from typing_extensions import TypeAlias

import porepy as pp

from . import _ad_parser
from .grid_entity import GridEntities, GridEntity
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

EquationList: TypeAlias = (
    "Union[list[str], list[Operator], list[pp.ad.EquationOnDomain]]"
)
"""A union type representing equations through:
- their names (:class:`str`)
- operators (:class:`~porepy.numerics.ad.operators.Operator`)
- atomic identifiers (:class:`~porepy.numerics.ad.indexers.EquationOnDomain`), each
    defining a single equation on a single domain, similarly to atomic variables
    (:class:`~porepy.numerics.ad.operators.Variable`).

This type is accepted as input to various methods and parsed to a list of
:class:`~porepy.numerics.ad.indexers.EquationOnDomain` using
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


class EquationSystem:
    """Represents an equation system, modelled by AD variables and equations in AD form.

    This class provides functionalities to create and manage variables, as well as
    managing equations on the form of :class:`~porepy.numerics.ad.operators.Operator`.
    It further provides functions to assemble subsystems and using subsets of equations
    and variables.

    The information about how the equations and variables of a discretized problem are
    arranged in a single contiguous vector is stored in :attr:`equation_indexer` and
    :attr:`variable_indexer`.

    Note:
        As of now, the system matrix (Jacobian) is assembled with respect to ALL
        variables and then the columns belonging to the requested subset of variables
        and grids are sliced out and returned. This will be optimized with minor changes
        to the AD operator class and its recursive forward AD mode in the future.

    """

    def __init__(self, mdg: pp.MixedDimensionalGrid) -> None:
        ### PUBLIC
        self.mdg: pp.MixedDimensionalGrid = mdg
        """Mixed-dimensional domain passed at instantiation."""

        ### PRIVATE

        self._equations: dict[str, Operator] = dict()
        """Contains references to equations in AD operator form for a given name (key).

        Private to avoid users setting equations directly and circumventing the current
        set-method which includes information about the image space.

        """

        self._variables: dict[int, Variable] = dict()
        """Dictionary mapping variable IDs to the atomic variables created and managed
        by this instance.

        Variables contained here are ordered chronologically in terms of
        instantiation. The order in the default :attr:`variable_indexer` is different.

        A Variable is uniquely identified by its name and domain, stored as attributes
        of the Variable object.

        Implementation-wise it is uniquely identified by its ID.

        """

        self._equation_indexer: pp.ad.EquationSystemIndexer | None = None
        """Indexer defining the ordering of the equations (rows) when multiple equation
        values are packed in a single vector.

        """
        self._variable_indexer: pp.ad.VariableIndexer | None = None
        """Indexer defining the ordering of the variables (columns) when multiple
        variable values are packed in a single vector.

        """
        self._ad_parser = _ad_parser.AdParser(self.mdg)

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
        # Parse and validate input arguments. Will raise if unknown equations or
        # variables are requested.
        equations = {eq.name for eq in self._parse_equations(equation_names)}
        variables = self._parse_variable_type(variable_names, ordered=True)

        # Create the new equation system.
        new_equation_system = EquationSystem(self.mdg)

        # IMPLEMENTATION NOTE: This method imitates the variable creation and equation
        # setting procedures by calling private methods and accessing private
        # attributes. This should be acceptable since this is a factory method.

        # Loop over new system's variables, old DOF ordering is preserved.
        for variable in variables:
            # Update variables.
            new_equation_system._variables[variable.id] = variable

        # Loop over known equations to preserve row order.
        for name in self._equations.keys():
            if name in equations:
                equation = self._equations[name]
                new_equation_system._equations[name] = equation

        return new_equation_system

    @property
    def equations(self) -> dict[str, Operator]:
        """Dictionary containing names of operators (keys) and operators (values), which
        have been set as equations in this EquationSystem.

        """
        return self._equations

    @property
    def equation_image_space_composition(
        self,
    ) -> dict[str, dict[pp.GridLike, np.ndarray]]:
        """Dictionary containing image space composition, including subdomains
        and their block indices in the global system, for every equation
        set in this EquationSystem.

        Note: It does not include equations with empty domains.

        """
        warn(
            "equation_system.equation_image_space_composition is deprecated and will be"
            " removed. See equation_system.equation_indexer, or "
            "equation_system.construct_assembled_matrix_indexers.",
            category=FutureWarning,
            stacklevel=2,
        )
        return self.equation_indexer.equation_image_space_composition

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

    ### Indexers -----------------------------------------------------------------------

    @property
    def variable_indexer(self) -> pp.ad.VariableIndexer:
        """Indexer defining the ordering of the variables (columns) when multiple
        variable values are packed in a single vector.

        Initialized lazily, upon first request. Changes in equation system may
        invalidate the indexer, leading to its recomputation.

        """
        if self._variable_indexer is None:
            self._variable_indexer = self._construct_variable_indexer()
        return self._variable_indexer

    @property
    def equation_indexer(self) -> pp.ad.EquationSystemIndexer:
        """Indexer defining the ordering of the equations (rows) when multiple
        equation values are packed in a single vector.

        Initialized lazily, upon first request. Changes in equation system may
        invalidate the indexer, leading to its recomputation.

        The row blocks are in the same order as equations were added to this
        EquationSystem. If an equation is defined on multiple grids, the respective
        row-block is internally ordered as given by the mixed-dimensional grid
        (for sd in subdomains, for intf in interfaces).

        """
        if self._equation_indexer is None:
            self._equation_indexer = self.construct_equation_indexer()
        return self._equation_indexer

    def _construct_variable_indexer(self) -> pp.ad.VariableIndexer:
        """Construct a variable indexer for all the variables registered in this
        equation system.

        The ordering of registered variables is determined by
        :func:`cluster_dofs_gridwise`.

        Returns:
            The indexer.

        """
        indices: dict[pp.ad.Variable, np.ndarray] = {}
        offset = 0

        ordered_variables = cluster_dofs_gridwise(self.variables)

        for var in ordered_variables:
            dofs_per_grid = var.size
            dofs = np.arange(dofs_per_grid) + offset
            indices[var] = dofs
            offset += len(dofs)
        return pp.ad.VariableIndexer(indices=indices)

    def construct_equation_indexer(self) -> pp.ad.EquationSystemIndexer:
        """Construct an equation indexer for all the registered equations.

        Equation ordering follows registration order. Equations not defined anywhere
        are excluded.

        Returns:
            The indexer.

        """
        # Result dictionary.
        indices: dict[str, dict[pp.GridLike, np.ndarray]] = {}

        # self.equations defines the desired order of equations.
        for name, equation in self.equations.items():
            dofs_on_domains: dict[pp.GridLike, np.ndarray] = {}

            # Offset applies to domains within the same equation.
            offset = 0

            for domain in equation.domains:
                dofs_per_grid = pp.ad.OperatorSpace.from_domains(
                    (domain,), equation.target.dof_info
                ).num_dofs()

                assert dofs_per_grid > 0, (
                    f"Equation {name} has no DOFs on domain {domain}."
                )
                dofs = np.arange(dofs_per_grid) + offset
                dofs_on_domains[domain] = dofs
                offset += dofs_per_grid

            # Filter out equations not defined anywhere.
            if len(dofs_on_domains) > 0:
                indices[name] = dofs_on_domains

        return pp.ad.EquationSystemIndexer(equation_image_space_composition=indices)

    ### Variable management ------------------------------------------------------------

    def md_variable(
        self, name: str, domains: Optional[DomainList] = None
    ) -> MixedDimensionalVariable:
        """Create a mixed-dimensional variable for a given name-domain list combination.

        Parameters:
            name (str): Name of the mixed-dimensional variable.
            domains (optional): List of grids where the variable is defined. If None
                (default), all grids where the variable is defined are used.

        Returns:
            A mixed-dimensional variable.

        Raises:
            ValueError: If variables name exist on both grids and interfaces and domain
                type is not specified (domains is None).

        """
        if domains is None:
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
                if var.name == name and var.domain in domains
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

                p = equation_system.create_variables('pressure',
                                                     subdomains=mdg.subdomains())

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
        # Sanity check for admissible DOF types. A dof_info of None defaults to one DOF
        # per cell (see Variable.__init__), which is always admissible.
        if dof_info is not None:
            requested_type = set(dof_info.keys())
            if not requested_type.issubset(set(GridEntity)):
                non_admissible = requested_type.difference(set(GridEntity))
                raise ValueError(
                    f"Non-admissible DOF types {non_admissible} requested."
                )
        grid_entities = (
            None if dof_info is None else GridEntities.from_mapping(dof_info)
        )

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
            new_variable = Variable(name, grid_entities, domain=grid, tags=tags)

            # Store it in the system
            variables.append(new_variable)
            self._variables[new_variable.id] = new_variable

        # Create an md variable that wraps all the individual variables created on
        # individual grids.
        merged_variable = MixedDimensionalVariable(variables)

        # Invalidating variable indexer forces to recompute it next time it is accessed.
        self._variable_indexer = None

        return merged_variable

    def remove_variables(self, variables: VariableList) -> None:
        """Removes variables from the system.
        The variables are removed from the system and the DOFs are reordered.

        Parameters:
            variables: List of variables to remove. Variables can be given as a list of
                variables, mixed-dimensional variables, or variable names (strings).

        Raises:
            ValueError: If a variable is not known to the system.

        """
        variables = self._parse_variable_type(variables, ordered=False)
        for var in variables:
            if var.id not in self._variables:
                raise ValueError(
                    f"Variable {var.name} (ID: {var.id}) not known to the system."
                )
            # Remove the variable from the system.
            del self._variables[var.id]

        # Invalidating variable indexer forces to recompute it next time it is accessed.
        self._variable_indexer = None

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
        variables = self._parse_variable_type(variables, ordered=False)
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
        EquationSystem.

        Parameters:
            variables: List of variables to filter. If None, all variables in the
                EquationSystem are included. Variables can be given as a list of
                variables, mixed- dimensional variables, or variable names (strings).
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
        # Using the ordering of the input list.
        variables = self._parse_variable_type(variables, ordered=False)
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
        reference: bool = False,
    ) -> np.ndarray:
        """Assembles an array containing values for the passed variable-like argument.

        The gathered values will be the variable values corresponding to the storage
        index specified by the user. The global order is preserved and independent of
        the order of the argument.

        See also:
            :meth:`~porepy.numerics.ad.ad_utils.get_solution_values`.

        Parameters:
            variables: ``default=None``

                VariableType input for which the values are requested.
                If None (default), the global vector of unknowns is returned.
            time_step_index: Time step index for which the values should be fetched.
            iterate_index: Iterate index for which the values should be fetched.

        Raises:
            ValueError: If unknown VariableType arguments are passed.

        Returns:
            The respective (sub) vector in numerical format, size anywhere between 0 and
            :meth:`num_dofs`.

        """
        # Normalize the variable input. Using equation_system's ordering of variables.
        variables = self._parse_variable_type(variables, ordered=True)

        # Storage for atomic blocks of the sub vector (identified by name-grid pairs).
        values = []

        for variable in variables:
            val = pp.get_solution_values(
                variable.name,
                self._get_data(variable.domain),
                time_step_index=time_step_index,
                iterate_index=iterate_index,
                reference=reference,
            )
            # NOTE get_solution_values already returns a copy
            values.append(val)

        # If there are matching blocks, concatenate and return.
        # Else return an empty vector.
        return np.concatenate(values) if values else np.empty(0, dtype=float)

    def set_variable_values(
        self,
        values: np.ndarray,
        variables: Optional[VariableList] = None,
        time_step_index: Optional[int] = None,
        iterate_index: Optional[int] = None,
        additive: bool = False,
        reference: bool = False,
    ) -> None:
        """Sets values for a (sub) vector of the global vector of unknowns.

        The order of values is assumed to fit the global order.

        Note:
            The vector is assumed to be of proper size and will be dissected according
            to the global order, starting with the index 0.
            Mismatches of is-size and should-be-size according to the subspace specified
            by ``variables`` will raise respective errors by numpy.

        See also:
            :meth:`~porepy.numerics.ad.ad_utils.set_solution_values`.

        Parameters:
            values: Vector of size corresponding to number of DOFs of the specified
                variables.
            variables: ``default=None``

                VariableType input for which the values are prescribed.
                If None (default), the global vector of unknowns will be set.
            time_step_index: Time step index for which the values are intended.
            iterate_index: Iterate index for which the values are intended.
            additive: ``default=False``

                Flag to write values additively. To be used in iterative procedures.

        Raises:
            ValueError: If unknown VariableType arguments are passed.

        """

        # Start of dissection.
        dof_start = 0
        dof_end = 0

        # Normalize the variable input. Using equation_system's ordering of variables.
        variables = self._parse_variable_type(variables, ordered=True)

        for variable in variables:
            # 1. Slice the vector to local size
            num_dofs = variable.size
            dof_end = dof_start + num_dofs
            # Extract local vector. This will return a smaller-than-requested array if
            # indexation is out of range.
            local_vec = values[dof_start:dof_end]

            # 2. Use the AD utilities to set the values
            pp.set_solution_values(
                variable.name,
                local_vec,
                self._get_data(grid=variable.domain),
                time_step_index=time_step_index,
                iterate_index=iterate_index,
                additive=additive,
                reference=reference,
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
        max_index: Optional[int] = None,
    ) -> None:
        """Method for shifting stored time step values in data sub-dictionary.

        For details of the value shifting see the method
        :func:`~porepy.numerics.ad.ad_utils.shift_solution_values`.

        Parameters:
            variables: ``default=None``

                VariableType input for which the values should be shifted in time.
                If None, all variables created by this EquationSystem will be shifted.
            max_index: ``default=None``

                A positive integer, capping the range of the shift operation to
                ``i -> max_index``.
                If called repeatedly with ``None``, the depth in time keeps increasing.

        """
        for var in self._parse_variable_type(variables, ordered=False):
            pp.shift_solution_values(
                var.name,
                self._get_data(var.domain),
                pp.TIME_STEP_SOLUTIONS,
                max_index,
            )

    def shift_iterate_values(
        self,
        variables: Optional[VariableList] = None,
        max_index: Optional[int] = None,
    ) -> None:
        """Analogous to :meth:`shift_time_step_values`, but for iterates of the current
        (unknown) time step."""
        for var in self._parse_variable_type(variables, ordered=False):
            pp.shift_solution_values(
                var.name,
                self._get_data(var.domain),
                pp.ITERATE_SOLUTIONS,
                max_index,
            )

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

    def _parse_variable_type(
        self, variables: Optional[VariableList], ordered: bool = False
    ) -> list[Variable]:
        """Parse the input argument for the variable type.

        This method is used to parse the input argument for the variable type in
        several exposed methods, allowing the user to specify a single variable or a
        list of variables more flexibly.

        Parameters:
            variables: The input argument for the variable type.
                The following interpretation rules are applied:
                    - If None, return all variables.
                    - If a list of variables, return same.
                    - If a list of strings, return all variables with those names.
                    - If mixed-dimensional variable, return sub-variables.
            ordered: If False (default), respects the input ordering. Otherwise, orders
                the returned variables according to the :attr:`variable_indexer`.

        Raises:
            ValueError: if passed variables are duplicated.
            ValueError: if any of the passed variables is not registered in this
                equation system.

        Returns:
            List of Variables.

        """
        if variables is None:
            return list(self.variable_indexer.indices.keys())

        parsed_variables: list[Variable] = []
        assert isinstance(variables, list)
        for variable in variables:
            if isinstance(variable, MixedDimensionalVariable):
                parsed_variables.extend(variable.sub_vars)
            elif isinstance(variable, Variable):
                parsed_variables.append(variable)
            elif isinstance(variable, str):
                vars = [var for var in self._variables.values() if var.name == variable]
                parsed_variables.extend(vars)
            else:
                raise ValueError(
                    "Variable type must be a string or a Variable, not {}".format(
                        type(variable)
                    )
                )

        # Validate that variables are registered.
        for variable in parsed_variables:
            if variable.id not in self._variables:
                raise ValueError(
                    f"Variable {variable} is not registered in this equation system."
                )

        # Validate that variables are unique.
        parsed_variables_lookup = set(parsed_variables)
        if len(parsed_variables) != len(parsed_variables_lookup):
            raise ValueError(f"Passed variables are duplicated: {parsed_variables}.")

        if not ordered:
            return parsed_variables

        # Order variables according to the indexer.
        parsed_variables_ordered: list[Variable] = []
        for variable in self.variable_indexer.indices:
            if variable in parsed_variables_lookup:
                parsed_variables_ordered.append(variable)
        return parsed_variables_ordered

    def num_dofs(self) -> int:
        """Returns the total number of dofs managed by this system."""
        return self.variable_indexer.size

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
            variables = self._parse_variable_type(variables, ordered=True)
            # Array for the indices associated with argument.
            # The ordering is preserved in variable_indexer.
            indices = self.variable_indexer.projection_indices(operators=variables)
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
        # Respect the ordering of the input list of variables.
        variables = self._parse_variable_type(variables, ordered=False)
        unknown_variables = set(variables).difference(self.variable_indexer.indices)
        if unknown_variables:
            raise ValueError(
                "Variables not registered by this equation system: "
                f"{unknown_variables}."
            )
        dofs = [self.variable_indexer.indices[var] for var in variables]
        return np.concatenate(dofs) if len(dofs) > 0 else np.empty(0, dtype=np.int64)

    ### Equation management ------------------------------------------------------------

    def set_equation(
        self,
        equation: Operator,
        equations_per_grid_entity: Optional[dict[GridEntity, int]] = None,
    ) -> None:
        """Sets an equation using the passed operator and uses its name as an
        identifier.

        If an equation already exists under that name, it is overwritten.

        Note:
            Regarding the number of equations, this method assumes that the AD framework
            assembles row blocks per grid in subdomains, then per grid in interfaces,
            for each operator representing an equation. This is assumed to be the way
            PorePy AD works.

        Parameters:
            equation: An equation in AD operator form, assuming the right-hand side is
                zero and this instance represents the left-hand side.
            equations_per_grid_entity: a dictionary describing how many equations
                ``equation_operator`` provides, i.e. the number of equations per grid
                entity (cells, faces, nodes) for the operator. If None, this is inferred
                from the equation operator's own ``target.dof_info``. Providing it
                explicitly is optional and kept for backwards compatibility and as an
                extra safety net.

        Raises:
            ValueError: If the equation operator has a name already assigned to a
                previously set equation.
            ValueError: If the equation is defined on both subdomains and interfaces.
            AssertionError: If the equation is defined on an unknown grid.
            AssertionError: If ``equations_per_grid_entity`` is given explicitly and
                does not match the equation operator's own ``target.dof_info``.
            ValueError: If indicated number of equations does not match the actual
                number as per evaluation of operator.

        """
        # The domain of this equation is the set of grids on which it is defined
        name = equation.name
        if name in self._equations:
            raise ValueError(
                "The name of the equation operator is already used by another equation:"
                f"\n{self._equations[name]}"
                "\n\nMake sure your equations are uniquely named."
            )

        # If no grids are specified, there is nothing to do. Note: equation.target is
        # then a scalar/unclear/waived space with (necessarily) empty dof_info, so
        # equations_per_grid_entity cannot be validated against it in this case.
        grids = equation.target.grids
        if len(grids) == 0:
            # Store the equation itself.
            self._equations.update({name: equation})
            return

        # If provided, check that the number of equations per grid entity is consistent
        # with the equation operator's own target.dof_info.
        if equations_per_grid_entity is not None:
            if dict(equation.target.dof_info) != dict(equations_per_grid_entity):
                s = (
                    f"equations_per_grid_entity {equations_per_grid_entity} does not "
                    f"match the equation operator's own target.dof_info "
                    f" {equation.target.dof_info} for equation {name}."
                )

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

        # Assert the equation is not defined on an unknown domain.
        known_domains = set(self.mdg.subdomains()) | set(self.mdg.interfaces())
        unknown_domains = set(grids).difference(known_domains)
        assert not unknown_domains, (
            f"Equation defined on unknown domains: {unknown_domains}"
        )

        # If all good, store the equation itself.
        self._equations.update({name: equation})

        # Invalidating equation indexer forces to recompute it next time it is accessed.
        self._equation_indexer = None

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
            # Invalidating equation indexer.
            self._equation_indexer = None
            return equ
        else:
            raise ValueError(f"Cannot remove unknown equation {name}")

    def update_equation(
        self,
        equation_name: str,
        new_equation: Operator,
        equations_per_grid_entity: Optional[dict[GridEntity, int]] = None,
    ) -> None:
        """Updates an existing equation with a new equation operator.

        This method removes the existing equation and sets a new equation under the same
        name as the old equation.

        Parameters:
            equation_name: Name of the equation to be updated.
            new_equation: New equation in AD form.
            equations_per_grid_entity: a dictionary describing how many equations
                ``equation_operator`` provides. This is a temporary work-around until
                operators are able to provide information on their image space. The
                dictionary must contain the number of equations per grid entity (cells,
                faces, nodes) for the operator. The default value is None, and in that
                case, the equations_per_grid_entity of the previous equation are used.

        """
        if equations_per_grid_entity is None:
            equations_per_grid_entity = dict(
                self._equations[equation_name].target.dof_info
            )

        self.remove_equation(equation_name)
        new_equation.set_name(equation_name)
        self.set_equation(
            equation=new_equation,
            equations_per_grid_entity=equations_per_grid_entity,
        )

    def reset_variable_equation_indices(self) -> None:
        """Inform the equation system that the domain of definition of variables and/or
        equations has been changed externally, and it needs to recompute the data
        arrangement in a contiguous vector.

        Known use case is the dynamic fracture propagation.

        """
        # Invalidate indexers to force their recomputation next time they are accessed.
        self._variable_indexer = None
        self._equation_indexer = None

    ### System assembly and discretization ---------------------------------------------

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

        if isinstance(operator, pp.ad.MergedOperator):
            # We have reached the bottom; this is a discretization (example: mpfa.flux)
            discr.append(operator)

        return discr

    def _validate_equation_name(self, equation: str | pp.ad.Operator) -> pp.ad.Operator:
        """Ensures that the equation is registered.

        Parameters:
            equation: Either equation name or its operator.

        Raises:
            ValueError: If the equation is not registered.

        Returns:
            The corresponding equation operator.

        """

        if isinstance(equation, pp.ad.Operator):
            equation = equation.name

        equation_or_none = self._equations.get(equation, None)
        if equation_or_none is None:
            raise ValueError(
                f"Requested equation with name '{equation}' is not registered "
                "in this equation system."
            )
        return equation_or_none

    def _validate_equation_restriction(
        self, equation: str | pp.ad.Operator, domains: DomainList
    ) -> pp.ad.Operator:
        """Validate an equation and the domains to which it is restricted.

        Raises:
            ValueError: If the equation is not registered or is restricted to a domain
                on which it is not defined.

        Returns:
            The registered equation operator.

        """
        equation = self._validate_equation_name(equation)
        invalid_domains = set(domains).difference(equation.domains)
        if invalid_domains:
            raise ValueError(
                f"Domains {invalid_domains} are not part of equation '{equation.name}'."
            )
        return equation

    def _parse_equations(
        self,
        equations: Optional[EquationList | EquationRestriction] = None,
        ordered: bool = True,
    ) -> list[pp.ad.EquationOnDomain]:
        """Helper method to parse equations into a properly ordered structure.

        The domains of the resulting equations will be ordered according to the MDG, the
        input ordering of the domains will be ignored.

        Equations not defined anywhere are filtered out.

        Parameters:
            equations: A list of equations or a dictionary of equation restrictions.
            ordered: If True (default), the equations will be ordered according to
            :attr:`equation_indexer`. Otherwise, the input ordering will be preserved.

        Raises:
            ValueError: If the requested equation is not registered.
            ValueError: If an equation is restricted to a domain on which it is not
                defined.

        Returns:
            A list of atomic equation-domain identifiers.

        """
        equations_on_domains: list[pp.ad.EquationOnDomain] = []

        if equations is None:
            # Requested are all the registered equations.
            equations = list(self.equations.keys())

        if isinstance(equations, list):
            # Equation names are restricted, domains are not restricted (None).
            for eq in equations:
                if isinstance(eq, pp.ad.EquationOnDomain):
                    # This assumes that eq.domains are already sorted.
                    domains: DomainList = [eq.domain]  # type: ignore[assignment]
                    self._validate_equation_restriction(eq.name, domains=domains)
                    equations_on_domains.append(eq)
                    continue

                eq = self._validate_equation_name(eq)
                for domain in eq.domains:
                    # This assumes that eq.domains are already sorted.
                    equations_on_domains.append(
                        pp.ad.EquationOnDomain(name=eq.name, domain=domain)
                    )

        elif isinstance(equations, dict):
            # Equation names and domains are restricted.

            for eq, domains in equations.items():
                eq = self._validate_equation_restriction(eq, domains)
                # Order domains based on the MDG and append.
                domain_indices = self.mdg.argsort_grids(domains)
                domains = [domains[i] for i in domain_indices]
                for domain in domains:
                    equations_on_domains.append(
                        pp.ad.EquationOnDomain(name=eq.name, domain=domain)
                    )

        if not ordered:
            return equations_on_domains

        # Order according to equation_indexer.
        equation_order = {eq: i for i, eq in enumerate(self.equation_indexer.indices)}
        return list(sorted(equations_on_domains, key=lambda eq: equation_order[eq]))

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
        equation_names = [
            eq.name for eq in self._parse_equations(equations, ordered=False)
        ]
        # This keeps only unique names and preserves their original (input) order.
        equation_names = list(dict.fromkeys(equation_names))

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
        unique_discr = pp.ad.uniquify_discretization_list(discr)
        pp.ad.discretize_from_list(unique_discr, self.mdg)
        # Reduce the memory footprint of discretization matrices.
        pp.matrix_operations.prune_discretization_matrices(self.mdg)

    @overload
    def assemble(
        self,
        evaluate_jacobian: Literal[True] = True,
        equations: Optional[EquationList | EquationRestriction] = None,
        variables: Optional[VariableList] = None,
        state: Optional[np.ndarray] = None,
    ) -> pp.solvers.LinearSystem: ...

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
    ) -> pp.solvers.LinearSystem | np.ndarray:
        """Assemble Jacobian matrix and residual vector using a specified subset of
        equations, variables and grids.

        The ordering of rows and columns in the returned LinearSystem are defined
        by the equation system's :attr:`equation_indexer` and
        :attr:`variable_indexer`, respectively.

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

        Raises:
            ValueError: if the format of `equations` or `variables` is incorrect, or
                they are not registered by this equation system.

        Returns:
            A linear system containing (requested part of) the Jacobian matrix and
                residual vector. The residual is scaled with -1 (moved to the right-hand
                side).

            or, if ``evaluate_jacobian`` is False,

            ndarray: Residual vector corresponding to the targeted variable state,
                for the specified equations. Scaled with -1 (moved to rhs).

        """
        # Standardize and validate input, order it according to the equation_indexer.
        equations_on_domains = self._parse_equations(equations=equations, ordered=True)

        # Distinguish equations where restriction is required by comparing requested
        # domains with domains of definition of the equation.
        equation_name_to_domains: dict[str, list[pp.GridLike]] = {}
        for eq in equations_on_domains:
            equation_name_to_domains.setdefault(eq.name, []).append(eq.domain)

        # Assembling the list of equations and their requested restriction (np.ndarray),
        # or no restriction (None).
        equations_rows: dict[pp.ad.Operator, np.ndarray | None] = {}
        equation_image_space = self.equation_indexer.equation_image_space_composition
        for eq_name, domains in equation_name_to_domains.items():
            # eq_names are validated in _parse_equations, they must be registered.
            equation = self._equations[eq_name]
            if domains == equation.domains:
                # Requested domains match the definition. No restriction is needed.
                equations_rows[equation] = None
            else:
                # Concatenate restriction dofs.
                domains_to_dofs = equation_image_space[eq_name]
                dofs = [domains_to_dofs[domain] for domain in domains]
                equations_rows[equation] = (
                    np.concatenate(dofs) if len(dofs) else np.empty(0, dtype=int)
                )

        # Data structures for building matrix and residual vector
        mat: list[sps.spmatrix] = []
        rhs: list[np.ndarray] = []

        # Ignore impenetrable mypy error here, the overloaded signatures are correctly
        # defined.
        values = self.evaluate(  # type: ignore[call-overload]
            list(equations_rows.keys()),
            derivative=evaluate_jacobian,
            state=state,
        )

        for row, value in zip(equations_rows.values(), values):
            # Extract residual vector and possibly Jacobian matrix.
            rhs_value = value.val if evaluate_jacobian else value
            jac = value.jac if evaluate_jacobian else None
            if row is not None:
                # If restriction to grid-related row blocks was made, perform row
                # slicing based on information we have obtained from parsing.
                rhs.append(rhs_value[row])
                if evaluate_jacobian:
                    assert jac is not None  # mypy
                    mat.append(jac[row])
            else:
                # If no grid-related row restriction was made, append the whole thing.
                rhs.append(rhs_value)
                if evaluate_jacobian:
                    mat.append(jac)

        # Concatenate results equation-wise.
        if len(rhs) > 0:
            if evaluate_jacobian:
                A = sps.vstack(mat, format="csr")
            rhs_cat = np.concatenate(rhs)
        else:
            # Special case if the restriction produced an empty system.
            A = sps.csr_matrix((0, self.num_dofs()))
            rhs_cat = np.empty(0, dtype=float)

        if not evaluate_jacobian:
            return -rhs_cat

        equation_indexer, variable_indexer = self._construct_assembled_matrix_indexers(
            equations=equations, variables=variables
        )

        # Slice out the columns belonging to the requested subsets of variables and
        # grid-related column blocks by using the transposed projection to respective
        # subspace.
        if variables is not None:
            # Respect the ordering of the input list of variables.
            variables_ = self._parse_variable_type(variables=variables, ordered=True)
            col_proj = [self.variable_indexer.indices[var] for var in variables_]
            column_projection = (
                np.concatenate(col_proj)
                if len(col_proj) > 0
                else np.empty(0, dtype=int)
            )
            A = A[:, column_projection]

        # Multiply rhs by -1 to move to the rhs.
        return pp.solvers.LinearSystem(
            matrix=A,
            rhs=-rhs_cat,
            equation_indexer=equation_indexer,
            variable_indexer=variable_indexer,
        )

    def _construct_assembled_matrix_indexers(
        self,
        equations: Optional[EquationList | EquationRestriction] = None,
        variables: Optional[VariableList] = None,
    ) -> tuple[pp.ad.EquationIndexer, pp.ad.VariableIndexer]:
        """Generate indexers for the linear system produced by :meth:`assemble`.

        Parameters:
            equations: Equation restriction passed to :meth:`assemble`.
            variables: Variable restriction passed to :meth:`assemble`.

        Returns:
            `equation_indexer` and `variable_indexer`. If no restriction is requested,
            returns equation system's own indexers.

        """
        if variables is not None:
            # Respect the ordering of the input list of variables.
            variable_indexer = self.variable_indexer.construct_restricted_indexer(
                self._parse_variable_type(variables=variables, ordered=True)
            )
        else:
            variable_indexer = self.variable_indexer

        if equations is not None:
            equation_indexer = self.equation_indexer.construct_restricted_indexer(
                self._parse_equations(equations=equations, ordered=True)
            )
        else:
            equation_indexer = self.equation_indexer

        return equation_indexer, variable_indexer

    ### Evaluate Ad operators ----------------------------------------------------------

    # IMPLEMENTATION NOTE: The following overloads was what turned out to be necessary
    # to get the typing right, or keep mypy silent. EK cannot see a principled reason
    # why exactly these signatures had to be represented, but has reached the point of
    # exhaustion, so this is what we have. For future reference, the following link
    # might be useful to tackle similar issues:
    #   https://github.com/python/typing/discussions/1326

    @overload
    def evaluate(
        self,
        operator: pp.ad.Operator,
    ) -> pp.number | np.ndarray | sps.spmatrix: ...

    @overload
    def evaluate(
        self,
        operator: list[pp.ad.Operator],
    ) -> list[pp.number | np.ndarray | sps.spmatrix]: ...

    @overload
    def evaluate(
        self,
        operator: pp.ad.Operator,
        derivative: None,
        state: np.ndarray | None,
    ) -> pp.number | np.ndarray | sps.spmatrix: ...

    @overload
    def evaluate(
        self,
        operator: list[pp.ad.Operator],
        derivative: None,
        state: np.ndarray | None,
    ) -> list[pp.number | np.ndarray | sps.spmatrix]: ...

    @overload
    def evaluate(
        self,
        operator: pp.ad.Operator,
        derivative: Literal[False] = False,
        state: Optional[np.ndarray] = None,
    ) -> pp.number | np.ndarray | sps.spmatrix: ...

    @overload
    def evaluate(
        self,
        operator: list[pp.ad.Operator],
        derivative: Literal[False] = False,
        state: Optional[np.ndarray] = None,
    ) -> list[pp.number | np.ndarray | sps.spmatrix]: ...

    @overload
    def evaluate(
        self,
        operator: pp.ad.Operator,
        derivative: Literal[True],
        state: np.ndarray | None,
    ) -> pp.ad.AdArray: ...

    @overload
    def evaluate(
        self,
        operator: list[pp.ad.Operator],
        derivative: Literal[True],
        state: np.ndarray | None,
    ) -> list[pp.ad.AdArray]: ...

    def evaluate(
        self,
        operator: pp.ad.Operator | list[pp.ad.Operator],
        derivative: Optional[bool] = False,
        state: Optional[np.ndarray] = None,
    ) -> (
        pp.number
        | np.ndarray
        | sps.spmatrix
        | pp.ad.AdArray
        | list[pp.number | np.ndarray | sps.spmatrix]
        | list[pp.ad.AdArray]
    ):
        """Evaluate an operator on the current state.

        Parameters:
            operator: Operator to evaluate.
            derivative: Whether to evaluate the derivative of the operator. Defaults to
                False.
            state: State vector to evaluate the operator on. By default, the current
                state is used.

        Returns:
            The operator evaluated on the current state. If the operator is a list, a
            list of evaluations is returned. If the derivative is requested, the
            evaluation is returned as an AdArray.

        """
        # EK: Ignore a typing error regarding 'no overload variant of "evaluate" matches
        # the argument types' since the overloads are correctly defined. I have no idea
        # why this error occurs.
        return self._ad_parser.evaluate(  # type: ignore[call-overload]
            operator, self, derivative, state
        )

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


def cluster_dofs_gridwise(variables: list[pp.ad.Variable]) -> list[pp.ad.Variable]:
    """Re-arranges the DOFs grid-wise s.t. we obtain grid-blocks in the column sense
    and reduce the matrix bandwidth.

    The aim is to impose a more block-diagonal-like structure on the Jacobian where
    blocks in the column sense represent single grids in the following order:

    1. For each grid in ``mdg.subdomains``
        1. For each variable defined on that grid
    2. For each grid in ``mdg.interfaces``
        1. For each variable defined on that mortar grid

    The order of variables per grid is given by the grid id.

    """
    mapping_grid_to_domains = defaultdict(lambda: [])
    for variable in variables:
        mapping_grid_to_domains[variable.domain].append(variable)

    known_grids = list(
        sorted(
            mapping_grid_to_domains.keys(),
            key=lambda grid: (
                isinstance(grid, pp.MortarGrid),
                -grid.dim,
                grid.id,
            ),
        )
    )

    ordered_variables = []

    for grid in known_grids:
        ordered_variables.extend(mapping_grid_to_domains[grid])

    return ordered_variables
