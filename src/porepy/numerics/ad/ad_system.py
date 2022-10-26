"""Contains the AD system manager, managing variables and equations for a system modelled
using the AD framework.

"""

from __future__ import annotations

import itertools
from enum import Enum, EnumMeta
from typing import Any, Callable, Literal, Optional, Union, overload

import numpy as np
import scipy.sparse as sps

import porepy as pp

from . import _ad_utils
from .operators import MixedDimensionalVariable, Operator, Variable

__all__ = ["ADSystem"]

GridLike = Union[pp.Grid, pp.MortarGrid]
"""A union type representing a domain either by a grid or mortar grids."""

VarLike = Union[
    str,
    list[str],
    Variable,
    list[Variable],
    MixedDimensionalVariable,
    list[MixedDimensionalVariable],
    Enum,
    list[Enum],
    list[Union[str, Variable, MixedDimensionalVariable, Enum]],
]
"""A union type representing variables either by one or multiple names (:class:`str`),
one or multiple :class:`~porepy.numerics.ad.operators.Variable`
one or multiple :class:`~porepy.numerics.ad.operators.MixedDimensionalVariable`
or one or multiple categories of variables (:class:`~enum.Enum`).

During parsing, grid variables are prioritized, i.e. a grid variable restricts its respective
mixed-dimensional variable, as well as categories.

Examples:
    If ``T`` and ``T_g`` are a mixed-dimensional variable and a grid variable respectively,
    representing the temperature, the variable-like structure

    >>> [T, T_g]

    Represents the temperature variable on a single grid.

    If additionally ``h`` and ``T`` are categorized as ``energy``, the structure

    >>> [energy, T_g]

    will represent the mixed-dimensional variable ``h`` and the grid variable ``T_g``.

    Additionally, note that if ``T`` is named ``'temperature'``
    (consequently ``T_g`` will have the same name), then the following variable-like structures
    are equivalent

    >>> [T, T_g]
    >>> ['temperature', T_g]

    Also equivalent are

    >>> [T]
    >>> ['temperature']

    In this case, the user can omit the list and pass only

    >>> T
    >>> 'temperature'

    For consistency reasons, the order within sequential variable-like structures does not
    influence the result when assembling subsystems or requesting variable values,
    since the AD framework always imposes its internally defined order of dofs and columns,
    which by logic of linear algebra must be the same.
    No restrictions are imposed on the user on how to assemble variable-like structure.

"""

EquationLike = Union[
    str,
    list[str],
    Operator,
    list[Operator],
    dict[str, list[GridLike]],
    dict[Operator, list[GridLike]],
    list[Union[str, dict[str, list[GridLike]], dict[Operator, list[GridLike]]]],
]
"""A union type representing equations either by one or multiple names (:class:`str`),
one or multiple :class:`~porepy.numerics.ad.operators.Operator`,
or a dictionary containing equation domains (:data:`GridLike`) per equation (name or Operator).

If an equation is defined on multiple grids, the dictionary can be used to represent a
restriction of that equation on respective grids.

During parsing, restrictions to grids defined by dictionaries will be prioritized.

Examples:
    An equation given by the operator ``e1``, defined on subdomains ``sd1, f1``, representing
    matrix and fracture respectively, has two row blocks. The equation-like structure

    >>> [e1, {e1: sd1}]

    will result only in the row block belonging to ``sd1``.

    Similar to variable-like structures, note that if the operator ``e1`` is named
    ``'mass_balance'`` for example, the following is equivalent

    >>> [e1, {e1: sd1}]
    >>> ['mass_balance', {e1: sd1}]
    >>> ['mass_balance', {'mass_balance': sd1}]
    >>> [e1, {'mass_balance': sd1}]
    >>> {e1: sd1}
    >>> {'mass_balance': sd1}

    Additionally, if only the assembly of one equation is requested by the user, the list can
    be omitted, i.e. the following equation-like structures are equivalent

    >>> e1
    >>> 'mass_balance'
    >>> [e1]
    >>> ['mass_balance']

    If the equation is defined on an additional fracture ``f2``, the following structure
    represents the equation only on the fractures

    >>> {e1: [f1, f2]}

    i.e. it will consist of two row blocks belonging to respective fractures.

    If multiple restrictions are passed for a single equation, the last one will be used,
    i.e.

    >>> [{e1: [f1, f2]}, {e1: [sd1]}]

    will result in

    >>> {e1: [sd1]}

    For consistency reasons, the ordering inside dictionary values, or general sequential
    equation-like structures with multiple equations, does not influence the resulting order of
    blocks since the AD framework always imposes its internally defined order of rows.
    No restrictions are imposed on the user on how to assemble the equation-like structure.

"""


class ADSystem:
    """Represents a physical system, modelled by AD variables and equations in AD form.

    This class provides functionalities to create and manage variables,
    as well as managing equations in AD operator form.

    It further provides functions to assemble subsystems, using subsets of equations and
    variables.

    Notes:
        As of now, the system matrix (Jacobian) is assembled with respect to ALL variables
        and then the columns belonging to the requested subset of variables and grids are
        sliced out and returned. This will be optimized with minor changes to the AD operator
        class and its recursive forward AD mode in the future.

        Currently, this class optimizes the block structure of the Jacobian only regarding the
        subdomains and interfaces. A more localized optimization
        (e.g. cell-wise for equations without spatial differential operators) is not performed.

    Examples:
        An example of how to instantiate an AD system with *primary* and *secondary* variables
        would be

        >>> from enum import Enum
        >>> import porepy as pp
        >>> var_categories = Enum('var_categories, ['primary', 'secondary'])
        >>> mdg = ...  # some mixed-dim grid
        >>> ad_sys = pp.ad.ADSystem(mdg, var_categories)
        >>> p = ad_sys.create_variable('pressure', category=var_categories.primary)
        >>> rho = ad_sys.create_variable('density', category=var_categories.secondary)
        >>> primary_projection = ad_sys.projection_to(var_category.primary)

    Parameters:
        mdg: mixed-dimensional grid representing the whole computational domain.
        var_categories (optional): an :class:`~enum.EnumMeta` object containing categories for
            variables, defined by the user. They can later be used to assign categories to
            created variables and assemble respective subsystems using the enumerated object.

    """

    admissible_dof_types: tuple[
        Literal["cells"], Literal["faces"], Literal["nodes"]
    ] = ("cells", "faces", "nodes")
    """A set denoting admissible types of local DOFs for variables.

    - nodes: DOFs per node, which constitute the grid
    - cells: DOFs per cell (center), which are defined by nodes
    - faces: DOFS per face, which form the (polygonal) boundary of cells

    """

    def __init__(
        self, mdg: pp.MixedDimensionalGrid, var_categories: Optional[EnumMeta] = None
    ) -> None:

        ### PUBLIC

        self.mdg: pp.MixedDimensionalGrid = mdg
        """Mixed-dimensional domain passed at instantiation."""

        self.var_categories: Optional[EnumMeta] = var_categories
        """Enumeration object containing the variable categories passed at instantiation."""

        self.variables: dict[str, MixedDimensionalVariable] = dict()
        """Contains references to (global) MergedVariables for a given name (key)."""

        self.assembled_equation_indices: dict[str, np.ndarray] = dict()
        """Contains the row indices in the last assembled (sub-) system for a given equation
        name (key). This dictionary changes with every call to any assemble-method.

        """

        ### PRIVATE

        self._equations: dict[str, Operator] = dict()
        """Contains references to equations in AD operator form for a given name (key).
        Private to avoid having people setting equations directly and circumventing the current
        set-method which included information about the image space.

        """

        self._equ_image_space_composition: dict[
            str, dict[GridLike, np.ndarray]
        ] = dict()
        """Contains for every equation name (key) a dictionary, which provides again for every
        involved grid (key) the indices of equations expressed through the equation operator.
        The ordering of the items in the grid-array dictionaries is consistent with the
        remaining PorePy framework.

        """

        self._grid_variables: dict[GridLike, dict[str, Variable]] = dict()
        """Contains references to grid Variables and their names for a given grid (key).
        The reference is stored as another dict, which returns the variable for a given name
        (key).

        """

        self._Schur_complement: Optional[tuple] = None
        """Contains block matrices and the split rhs of the last assembled Schur complement,
        such that the expansion can be made.

        """

        self._vars_of_category: dict[Any, list[str]] = dict()
        """Contains the names of variables assigned to a specific category (key)."""

        self._block_numbers: dict[tuple[str, GridLike], int] = dict()
        """Dictionary containing the block number for a given combination of grid/ mortar grid
        and variable name (key).

        """

        self._block_dofs: np.ndarray = np.array([], dtype=int)
        """Array containing the number of DOFS per block number. The block number corresponds
        to this array's indexation.

        """

    ### Variable management -------------------------------------------------------------------

    def create_variable(
        self,
        name: str,
        dof_info: dict[str, int] = {"cells": 1},
        subdomains: Optional[list[pp.Grid]] = None,
        interfaces: Optional[list[pp.MortarGrid]] = None,
        category: Optional[Enum] = None,
    ) -> MixedDimensionalVariable:
        """Creates a new variable according to specifications.

        This method does now assign any values to the variable. This has to be done in a
        subsequent step (using e.g. :meth:`set_var_values`).

        If variables are categorized using enums, each variable must be assigned to a category.
        A variable cannot be in multiple categories, but only one.

        Notes:
            This methods provides support for creating variables on **all** subdomains or
            interfaces, without having to pass them all as arguments.
            If the argument ``subdomains`` is an empty list, the method will use all
            subdomains found in the mixed-dimensional grid.
            If the argument ``interfaces`` is an empty list, the method will use all
            interfaces found in the mixed-dimensional list.

        Examples:
            An example on how to define a pressure variable with cell-wise one DOF (default)on
            **all** subdomains and **no** interfaces would be

            >>> p = ad_system.create_variable('pressure', subdomains=[])

        Parameters:
            name: used here as an identifier. Can be used to associate the variable with some
                physical quantity like ``'pressure'``.
            dof_info: dictionary containing information about number of DOFs per admissible
                type (see :data:`admissible_dof_types`). Defaults to ``{'cells':1}``.
            subdomains (optional): list of subdomains on which the variable is defined.
                If None, then it will not be defined on any subdomain.
            interfaces (optional): list of interfaces on which the variable is defined.
                If None, then it will not be defined on any interface.
            category (optional): assigns a category to the variable. Must be an :class:`Enum`
                contained in :class:`EnumMeta` passed at instantiation.

        Returns:
            a mixed-dimensional variable with above specifications.

        Raises:
            ValueError: if non-admissible DOF types are used as local DOFs.
            ValueError: if one attempts to create a variable not defined on any grid.
            ValueError: if passed category is not in enumeration object passed at
                instantiation.
            KeyError: if a variable with given name is already defined.

        """
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
        if name in self.variables.keys():
            raise KeyError(f"Variable with name '{name}' already defined.")

        # if an empty list was received, we use ALL subdomains
        if isinstance(subdomains, list) and len(subdomains) == 0:
            subdomains = [sg for sg in self.mdg.subdomains()]
        # if an empty list was received, we use ALL interfaces
        if isinstance(interfaces, list) and len(interfaces) == 0:
            interfaces = [intf for intf in self.mdg.interfaces()]

        # container for all grid variables
        variables = list()
        var_cat: Any
        # sanity check for passed category, allow only pre-defined categories
        if self.var_categories:
            if category is None:
                raise ValueError(
                    "Category not assigned. "
                    "Use one of the categories passed at instantiation."
                )
            elif category not in self.var_categories:
                raise ValueError(f"Unknown variable category {category}.")
            else:
                var_cat = category
        # if no categories given, we use PorePy's default.
        else:
            var_cat = pp.PRIMARY_VARIABLES

        if subdomains:
            for sd in subdomains:
                data = self.mdg.subdomain_data(sd)

                # prepare data dictionary if this was not done already
                if var_cat not in data:
                    data[var_cat] = dict()
                if pp.STATE not in data:
                    data[pp.STATE] = dict()
                if pp.ITERATE not in data[pp.STATE]:
                    data[pp.STATE][pp.ITERATE] = dict()

                data[var_cat].update({name: dof_info})

                # create grid variable
                new_var = Variable(name, dof_info, domain=sd)
                if sd not in self._grid_variables.keys():
                    self._grid_variables.update({sd: dict()})
                if name not in self._grid_variables[sd]:
                    self._grid_variables[sd].update({name: new_var})
                variables.append(new_var)

        if interfaces:
            for intf in interfaces:
                data = self.mdg.interface_data(intf)

                if (
                    intf.codim == 2
                ):  # no variables in points TODO check if this up-to-date
                    continue
                else:
                    # prepare data dictionary if this was not done already
                    if var_cat not in data:
                        data[var_cat] = dict()
                    if pp.STATE not in data:
                        data[pp.STATE] = dict()
                    if pp.ITERATE not in data[pp.STATE]:
                        data[pp.STATE][pp.ITERATE] = dict()

                    # store DOF information about variable
                    data[var_cat].update({name: dof_info})

                    # create mortar grid variable
                    new_var = Variable(name, dof_info, domain=intf)
                    if intf not in self._grid_variables.keys():
                        self._grid_variables.update({intf: dict()})
                    if name not in self._grid_variables[intf]:
                        self._grid_variables[intf].update({name: new_var})
                    variables.append(new_var)

        # create and store the md variable
        merged_var = MixedDimensionalVariable(variables)
        self.variables.update({name: merged_var})
        # store categorization
        if var_cat not in self._vars_of_category:
            self._vars_of_category.update({var_cat: list()})
        self._vars_of_category[var_cat].append(name)

        # append the new DOFs to the global system
        self._append_dofs(name, var_cat)

        return merged_var

    def get_var_names(
        self,
        categories: Optional[Enum | list[Enum]] = None,
    ) -> tuple[str, ...]:
        """Get all (unique) variable names defined so far.

        If specific categories are requested and the categories do not correspond to the ones
        passed at instantiation, an empty list is returned for unknown categories.

        Parameters:
            category (optional): filter names by one or multiple categories.

        Returns: a tuple of names, optionally corresponding to the passed category.

        """
        # if no categories were assigned, we use PorePy's default storage key
        # and ignore any category argument
        if self.var_categories is None:
            return tuple(self._vars_of_category.get(pp.PRIMARY_VARIABLES, list()))
        # if categories where assigned, we try to get the respective ones
        # unknown categories return an empty list
        else:
            var_names = list()
            if categories is None:
                for vars in self._vars_of_category.values():
                    var_names += vars
            else:
                if isinstance(categories, Enum):
                    categories = [categories]  # type: ignore
                for cat in categories:
                    var_names += self._vars_of_category.get(cat, list())
            return tuple(var_names)

    def get_variable_values(
        self, variables: Optional[VarLike] = None, from_iterate: bool = False
    ) -> np.ndarray:
        """Assembles an array containing values for the passed var-like argument.

        The global order is preserved and independent of the order of the argument.

        Notes:
            The resulting array is of any size between 0 and ``num_dofs``.

        Parameters:
            variables (optional): variable-like input for which the values are requested.
                If None (default), the global vector of unknowns is returned.
            from_iterate (optional): flag to return values stored as ITERATE,
                instead of STATE (default).

        Returns: the respective (sub) vector in numerical format.

        Raises:
            KeyError: if no values are stored for the var-like input.
            ValueError: if unknown var-like arguments are passed.

        """
        # storage for atomic blocks of the sub vector (identified by name-grid pairs)
        values = list()
        # assemble requested name-grid pairs
        requested_blocks = self._parse_varlike(variables)

        # loop over all blocks and process those requested
        # this ensures uniqueness and correct order
        for block in self._block_numbers.keys():
            if block in requested_blocks:
                name, grid = block
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
                        f"No values stored for variable {name}, from_iterate={from_iterate}"
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
        variables: Optional[VarLike] = None,
        to_iterate: bool = False,
        additive: bool = False,
    ) -> None:
        """Sets values for a (sub) vector of the global vector of unknowns specified by
        ``variables``.

        The order of values is assumed to fit the global order.

        Notes:
            The vector is assumed to be of proper size and will be dissected according to the
            global order, starting with the index 0.
            Mismatches of is-size and should-be-size according to the subspace specified by
            ``variables`` will raise respective errors by numpy.

        Parameters:
            values: vector of corresponding size.
            variables (optional): variable-like input for which the values are requested.
                If None (default), the global vector of unknowns will be set.
            to_iterate (optional): flag to write values to ITERATE, instead of STATE (default).
            additive (optional): flag to write values *additively* to ITERATE or STATE.
                To be used in iterative procedures.

        Raises:
            ValueError: if unknown var-like arguments are passed.

        """
        # assemble requested name-grid pairs
        requested_blocks = self._parse_varlike(variables)
        # start of dissection
        block_start = 0
        block_end = 0

        for block, block_number in self._block_numbers.items():
            if block in requested_blocks:
                name, grid = block
                block_length = int(self._block_dofs[block_number])
                block_end = block_start + block_length
                # extract local vector
                # this will raise errors if indexation out of range
                local_vec = values[block_start:block_end]
                # get storage
                if isinstance(grid, pp.Grid):
                    data = self.mdg.subdomain_data(grid)
                elif isinstance(grid, pp.MortarGrid):
                    data = self.mdg.interface_data(grid)
                # data dict should have pp.STATE and pp.ITERATE entries already created during
                # create_variable

                # store new values as requested
                if additive:
                    if to_iterate:
                        data[pp.STATE][pp.ITERATE][name] = (
                            data[pp.STATE][pp.ITERATE][name] + local_vec
                        )
                    else:
                        data[pp.STATE][name] = data[pp.STATE][name] + local_vec
                else:
                    if to_iterate:
                        data[pp.STATE][pp.ITERATE][name] = local_vec.copy()
                    else:
                        data[pp.STATE][name] = local_vec.copy()

                # move dissection forward
                block_start = block_end

        # last sanity check if the vector was properly sized, or if it was too large.
        # This imposes a theoretically unnecessary restriction on the input argument
        # since we only require a vector of at least this size.
        # Do we care if there are more values than necessary? TODO
        assert block_end == values.size

    ### DOF management ------------------------------------------------------------------------

    def _append_dofs(self, var_name: str, var_cat: Any) -> None:
        """Appends DOFs for a newly created variable.

        Must only be called by :meth:`create_variable`.

        This method defines a preliminary global order of dofs:

            For each variable
                append dofs on subdomains where variable is defined (order given by mdg)
                append dofs on interfaces where variable is defined (order given by mdg)

        Parameters:
            var_name: name of the newly created variable
            var_cat: category of the variable under which the DOF information is stored

        """
        # number of totally created dof blocks so far
        last_block_number: int = len(self._block_numbers)
        # new blocks
        new_block_numbers: dict[tuple[str, GridLike], int] = dict()
        # new dofs per block
        new_block_dofs_: list[int] = list()

        # add dofs found on subdomains for this variable name
        for sd, data in self.mdg.subdomains(return_data=True):
            if var_cat in data:
                if var_name in data[var_cat]:
                    # last sanity check that no previous data is overwritten
                    # should not happen if class not used in hacky way
                    assert (var_name, sd) not in self._block_numbers.keys()

                    # assign a new block number and increase the counter
                    new_block_numbers[(var_name, sd)] = last_block_number
                    last_block_number += 1

                    # count number of dofs for this variable on this grid and store it.
                    # the number of dofs for each dof type defaults to zero.
                    local_dofs = data[var_cat][var_name]
                    num_local_dofs = (
                        sd.num_cells * local_dofs.get("cells", 0)
                        + sd.num_faces * local_dofs.get("faces", 0)
                        + sd.num_nodes * local_dofs.get("nodes", 0)
                    )
                    new_block_dofs_.append(num_local_dofs)
        # add dofs found on interfaces for this variable
        for intf, data in self.mdg.interfaces(return_data=True):
            if var_cat in data:
                if var_name in data[var_cat]:
                    # last sanity check that no previous data is overwritten
                    # should not happen if class not used in hacky way
                    assert (var_name, intf) not in self._block_numbers.keys()

                    # assign a new block number and increase the counter
                    new_block_numbers[(var_name, intf)] = last_block_number
                    last_block_number += 1

                    # count number of dofs for this variable on this grid and store it.
                    # tonly cell-wise dofs are allowed on interfaces
                    local_dofs = data[var_cat][var_name]
                    total_local_dofs = intf.num_cells * local_dofs.get("cells", 0)
                    new_block_dofs_.append(total_local_dofs)

        # converting block dofs to array
        new_block_dofs = np.array(new_block_dofs_, dtype=int)
        # update the global dofs so far with the new blocks
        self._block_numbers.update(new_block_numbers)
        self._block_dofs = np.concatenate([self._block_dofs, new_block_dofs])

        # first optimization of Jacobian structure
        self._cluster_dofs_gridwise()

    def _cluster_dofs_gridwise(self) -> None:
        """Re-arranges the DOFs grid-wise s.t. we obtain grid-blocks in the column sense and
        reduce the matrix bandwidth.

        The aim is to impose a more block-diagonal-like structure on the Jacobian where blocks
        in the column sense represent single grids in the following order:

        Notes:
            Off-diagonal blocks will still be present if subdomain-interface fluxes are present
            in the md-sense.

        1. For each grid in ``mdg.subdomains``
            1. For each var defined on that grid
        2. For each grid in ``mdg.interfaces``
            1. For each var defined on that mortar grid

        The order of variables per grid is given by the order of variable creation
        (stored as order of keys in ``self.variables``).

        This method is called after each creation of variables and respective DOFs.

        """
        new_block_counter: int = 0
        new_block_numbers: dict[tuple[str, GridLike], int] = dict()
        new_block_dofs: list[int] = list()
        block_pair: tuple[GridLike, str]  # appeasing mypy

        # first set of diagonal blocks, per subdomain
        for sd in self.mdg.subdomains():
            # sub-loop of variables per subdomain
            for var_name in self.variables:
                block_pair = (var_name, sd)
                if block_pair in self._block_numbers.keys():
                    # extract created number of dofs
                    local_dofs = self._block_dofs[self._block_numbers[block_pair]]
                    # store new block number and dofs in new order
                    new_block_dofs.append(local_dofs)
                    new_block_numbers.update({block_pair: new_block_counter})
                    new_block_counter += 1
        # second set of diagonal blocks, per interface
        for intf in self.mdg.interfaces():
            # sub-loop of variables per interface
            for var_name in self.variables:
                block_pair = (var_name, intf)
                if block_pair in self._block_numbers.keys():
                    # extract created number of dofs
                    local_dofs = self._block_dofs[self._block_numbers[block_pair]]
                    # store new block number and dofs in new order
                    new_block_dofs.append(local_dofs)
                    new_block_numbers.update({block_pair: new_block_counter})
                    new_block_counter += 1
        # replace old block order
        self._block_dofs = np.array(new_block_dofs, dtype=int)
        self._block_numbers = new_block_numbers

    def _parse_varlike(
        self,
        variables: Optional[VarLike] = None,
    ) -> list[tuple[str, GridLike]]:
        """Helper method to create name-grid pairs for VarLike input.

        Raises a type error if the input or part of the input is not VarLike.

        Notes:
            This method, and the sub-routine _parse_single_varlike, are crucial in terms of
            performance.

        """
        # the default return value is all blocks
        if variables is None:
            return list(self._block_numbers.keys())
        # else we parse the input
        else:
            # parsing of non-sequential VarLikes
            if isinstance(variables, (str, Variable, MixedDimensionalVariable, Enum)):
                return self._parse_single_varlike(variables)
            # try to iterate over sequential VarLike
            else:
                # storage for all requested blocks, possible not unique and un-restricted in
                # terms of grids
                requested_blocks = list()
                # storage for all grid-restricted vars to eliminate the rest
                grid_restricted_vars = set()
                restricted_grids: dict[str, set] = dict()

                for variable in variables:
                    # same parsing as for non_sequentials
                    requested_blocks += self._parse_single_varlike(variable)
                    # filtering grid restrictions
                    if isinstance(variable, Variable):
                        grid_restricted_vars.add(variable.name)
                        if variable.name in restricted_grids:
                            restricted_grids[variable.name].add(variable.domain)
                        else:
                            restricted_grids[variable.name] = set([variable.domain])
                # make results unique
                requested_blocks = set(requested_blocks)
                # processing grid restrictions
                for name in grid_restricted_vars:
                    all_grids = set(self.variables[name].domain)
                    filtered_grids = all_grids.difference(restricted_grids[name])
                    for f_name, f_grid in itertools.product([name], filtered_grids):
                        if (f_name, f_grid) in requested_blocks:
                            requested_blocks.remove((f_name, f_grid))
                # iterate over available blocks and check if they are requested
                # this ensures uniqueness again and correct order
                return [
                    block for block in self._block_numbers if block in requested_blocks
                ]

    def _parse_single_varlike(
        self, variable: str | Variable | MixedDimensionalVariable | Enum
    ) -> list[tuple[str, GridLike]]:
        """Helper of helper :) Parses non-sequential VarLikes."""
        # variable represented as a string: include all associated grids
        if isinstance(variable, str):
            if variable not in self.variables.keys():
                raise ValueError(f"Unknown variable name {variable}.")
            return [block for block in self._block_numbers if block[0] == variable]
        # variable represented as grid variable: return local block
        elif isinstance(variable, Variable):
            if (variable.name, variable.domain) in self._block_numbers:
                return [(variable.name, variable.domain)]
            else:
                raise ValueError(f"Unknown grid variable {variable}.")
        # variable represented as md-var: include all associated grids
        elif isinstance(variable, MixedDimensionalVariable):
            if variable not in self.variables.values():
                raise ValueError(f"Unknown mixed-dimensional variable {variable}.")
            return [block for block in self._block_numbers if block[0] == variable.name]
        # if a category is passed, with return all blocks associated with respective vars
        elif isinstance(variable, Enum):
            if variable not in self.var_categories:
                raise ValueError(f"Unknown category {variable}.")
            else:
                var_names = self._vars_of_category[variable]
                return [block for block in self._block_numbers if block[0] in var_names]
        else:
            raise TypeError(f"Type {type(variable)} not parsable as variable-like.")

    def _gridbased_var_complement(self, variables: VarLike) -> list[Variable]:
        """Finds the grid-based complement of a variable-like structure.

        The grid-based complement consists of all those grid variables, which are not
        inside ``variables``, but their respective variable names appear in the structure.
        """
        # strings, md variables and categories represent always a whole in the variable sense
        # and the complement is empty
        if isinstance(variables, (str, MixedDimensionalVariable, Enum)):
            return list()
        # grid variables are part of a md variable, the complement are the remaining grid vars
        elif isinstance(variables, Variable):
            md_v = self.variables[variables.name]
            return [var for var in md_v.sub_vars if var.domain != variables.domain]
        # non sequential var-like structure
        else:
            grid_vars = list()
            for variable in variables:
                # same processing as above, only grid variables are of interest
                if isinstance(variable, Variable):
                    md_v = self.variables[variable.name]
                    grid_vars += [
                        v for v in md_v.sub_vars if v.domain != variable.domain
                    ]
            # return a unique collection
            return list(set(grid_vars))

    def num_dofs(self) -> int:
        """Returns the total number of dofs managed by this system."""
        return int(sum(self._block_dofs))  # cast numpy.int64 into Python int

    def projection_to(self, variables: Optional[VarLike] = None) -> sps.csr_matrix:
        """Create a projection matrix from the global vector of unknowns to a specified
        subspace.

        The subspace can be specified by variable categories, variable names and variables
        (see :data:`VarLike`).

        The transpose of the returned matrix can be used to slice respective columns out of the
        global Jacobian.

        The projection preserves the global order defined by the system, i.e. it includes no
        permutation.

        If no subspace is specified using ``variables``, a null-space projection is returned.

        Parameters:
            variables (optional): variable-like input for which the subspace is requested.

        Returns:
            a sparse projection matrix of shape ``(M,num_dofs)``, where ``0<=M<=num_dofs``.

        """

        # current number of total dofs
        num_dofs = self.num_dofs()
        if variables:
            # Array for the indices associated with argument
            indices = self.dofs_of(variables)
            # case where no dofs where found for the var-like input
            if len(indices) == 0:
                return sps.csr_matrix((0, num_dofs))
            else:
                subspace_size = indices.size
                return sps.coo_matrix(
                    (np.ones(subspace_size), (np.arange(subspace_size), indices)),
                    shape=(subspace_size, num_dofs),
                ).tocsr()
        # case where the subspace is null, i.e. no vars specified
        else:
            return sps.csr_matrix((0, num_dofs))

    def dofs_of(self, variables: VarLike) -> np.ndarray:
        """Get the indices in the global vector of unknowns belonging to the variable(s).

        The global order of indices is preserved.

        Parameters:
            variables: variable-like input for which the indices are requested.

        Returns:
            an order-preserving array of indices of DOFs belonging to the var-like input.

        Raises:
            ValueError: if unknown var-like arguments are passed.

        """
        # global block indices
        global_block_dofs = np.hstack((0, np.cumsum(self._block_dofs)))
        # parsing of requested blocks
        requested_blocks = self._parse_varlike(variables)
        # storage of indices per requested block
        indices = list()
        for block in requested_blocks:
            block_number = self._block_numbers[block]
            block_indices = np.arange(
                global_block_dofs[block_number],
                global_block_dofs[block_number + 1],
                dtype=int,
            )
            indices.append(block_indices)
        # concatenate indices, if any
        if indices:
            return np.concatenate(indices, dtype=int)
        else:
            return np.array([], dtype=int)

    @overload
    def identify_dof(
        self, dof: int, return_var: Literal[False] = False
    ) -> tuple[str, GridLike]:
        ...

    @overload
    def identify_dof(self, dof: int, return_var: Literal[True]) -> Variable:
        ...

    def identify_dof(
        self, dof: int, return_var: bool = False
    ) -> tuple[str, GridLike] | Variable:
        """Identifies the block to which a specific DOF index belongs.

        The block is represented either by a name-grid pair or the respective variable.

        The intended use is to help identify entries in the global vector or the column of the
        Jacobian, which do not behave as expected.

        Parameters:
            dof: a single index in the global vector.
            return_var (optional): if True, returns the variable object instead of the
                name-grid combination representing the dof.

        Returns: a 2-tuple containing variable name and a grid, or the respective variable
            itself.

        Raises:
            KeyError: if the dof is out of range (larger then ``num_dofs`` or smaller than 0).

        """
        num_dofs = self.num_dofs()
        if not (0 <= dof < num_dofs):  # indices go from 0 to num_dofs - 1
            raise KeyError("Dof index out of range.")

        # global block indices
        global_block_dofs = np.hstack((0, np.cumsum(self._block_dofs)))
        # Find the block number belonging to this index
        target_block_number = np.argmax(global_block_dofs > dof) - 1
        # find block belonging to the number, first and logically only occurrence
        for block_pair, block_number in self._block_numbers.items():
            if block_number == target_block_number:
                if return_var:
                    # note the grid variables are stored per grid-name not name-grid
                    return self._grid_variables[block_pair[1]][block_pair[0]]
                else:
                    return block_pair
        # if search was not successful, something went terribly wrong
        # should never happen, but if it does, notify the user
        raise RuntimeError("Someone messed with the global block indexation...")

    ### Equation management -------------------------------------------------------------------

    def set_equation(
        self,
        equation: Operator,
        num_equ_per_dof: dict[GridLike, dict[str, int]],
    ) -> None:
        """Sets an equation using the passed operator **and uses its name as an identifier**.

        If an equation already exists under that name, it is overwritten.

        Information about the image space must be provided for now, such that grid-wise row
        slicing is possible. This will hopefully be provided automatically in the future.

        Notes:
            Regarding the number of equations, this method assumes that the AD framework
            assembles row blocks per grid in subdomains, then per grid in interfaces, for each
            operator representing an equation. This is assumed to be the way PorePy AD works.

        Parameters:
            equation: An equation in AD operator form, assuming the right-hand side is
                zero and this instance represents the left-hand side.
                **The equation must be ready for evaluation,
                i.e. all involved variables must have values set.**
            num_equ_per_dof: a dictionary describing how many equations ``equation_operator``
                provides. This is a temporary work-around until operators are able to provide
                information on their image space.
                The dictionary must contain the number of equations per admissible dof type,
                (see :data:`porepy.DofManager.admissible_dof_types`),
                for each grid the operator was defined on.

        Raises:
            ValueError: if the equation operator has a name already assigned to a previously
                set equation
            AssertionError: if the equation is defined on an unknown grid.
            ValueError: if indicated number of equations does not match the actual number as
                per evaluation of operator.

        """
        image_info: dict[GridLike, np.ndarray] = dict()
        total_num_equ = 0
        valid_grids = [sd for sd in self.mdg.subdomains()]
        valid_intf = [intf for intf in self.mdg.interfaces()]
        equation_domain = list(num_equ_per_dof.keys())
        name = equation.name
        if name in self._equations.keys():
            raise ValueError(
                "The name of the equation operator is already used by another equation:\n"
                f"{self._equations[name]}"
                "\n\nMake sure your equations are uniquely named."
            )

        # we loop over the valid grids and interfaces in that order to assert a correct
        # indexation according to the global order (for grid in sds, for grid in intfs)
        # the user does not have to care about the order in num_equ_per_dof
        for grid in valid_grids:
            if grid in equation_domain:
                dof_info = num_equ_per_dof[grid]
                # equations on subdomains can be defined on any dof type
                num_equ_per_grid = int(
                    grid.num_cells * dof_info.get("cells", 0)
                    + grid.num_nodes * dof_info.get("nodes", 0)
                    + grid.num_faces * dof_info.get("faces", 0)
                )
                # row indices for this grid, cast to integers
                block_idx = np.arange(num_equ_per_grid, dtype=int) + total_num_equ
                # cumulate total number of equations
                total_num_equ += num_equ_per_grid
                # store block idx per grid
                image_info.update({grid: block_idx})
                # remove the grid from the domain list
                equation_domain.remove(grid)

        for grid in valid_intf:
            if grid in equation_domain:
                dof_info = num_equ_per_dof[grid]
                # equations on interfaces can only be defined on cells
                num_equ_per_grid = int(grid.num_cells * dof_info.get("cells", 0))
                # row indices for this grid, cast to integers
                block_idx = np.arange(num_equ_per_grid, dtype=int) + total_num_equ
                # cumulate total number of equations
                total_num_equ += num_equ_per_grid
                # store block idx per grid
                image_info.update({grid: block_idx})
                # remove the grid from the domain list
                equation_domain.remove(grid)

        # assert the equation is not defined on an unknown domain
        assert len(equation_domain) == 0

        # perform a validity check of the input
        equ_ad = equation.evaluate(self.dof_manager)
        is_num_equ = len(equ_ad.val)
        if total_num_equ != is_num_equ:
            raise ValueError(
                f"Passed 'equation_operator' has {is_num_equ} equations,"
                f" opposing indicated number of {total_num_equ}."
            )

        # if all good, we assume we can proceed
        self._equ_image_space_composition.update({name: image_info})
        self._equations.update({name: equation})

    def get_equation(self, name: str) -> Operator:
        """
        Returns: a reference to a previously passed equation in operator form.

        Raises:
            KeyError: if ``name`` does not correspond to any known equation.

        """
        return self._equations[name]

    def remove_equation(self, name: str) -> Operator | None:
        """Removes a previously set equation and all related information.

        Returns:
            a reference to the equation in operator form or None, if the equation is unknown.

        """
        if name in self._equations.keys():
            equ = self._equations.pop(name)
            del self._equ_image_space_composition[name]
            return equ
        else:
            return None  # appeasing mypy

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
                discr += ADSystem._recursive_discretization_search(child, list())

        if isinstance(operator, _ad_utils.MergedOperator):
            # We have reached the bottom; this is a discretization (example: mpfa.flux)
            discr.append(operator)

        return discr

    def _parse_equation_like(
        self, equations: Optional[EquationLike] = None
    ) -> dict[str, None | np.ndarray]:
        """Helper method to parse equation-like inputs into a properly ordered structure.

        Raises a type error if the input or part of the input is not equation-like.
        If an equation is requested for a grid on which it is not defined, a value error will
        be raised.

        """
        # the default return value is all equations with no grid restrictions
        if equations is None:
            return dict((name, None) for name in self._equations)
        # else we parse the input
        else:
            # parsing non-sequential arguments
            if isinstance(equations, (str, Operator, dict)):
                return self._parse_single_equation_like(equations)
            # try to iterate over sequential equation-likes
            else:
                # storage for requested blocks, unique information per equation name
                requested_row_blocks = dict()
                # storage for restricted equations
                restricted_equations = dict()

                for equation in equations:
                    block = self._parse_single_equation_like(equation)
                    # store restrictions
                    if isinstance(equation, dict):
                        restricted_equations.update(block)
                    else:
                        requested_row_blocks.update(block)
                # update the requested blocks with the restricted to overwrite the indices if
                # an equation was passed in both restricted and unrestricted structure
                requested_row_blocks.update(restricted_equations)
                # ensure order
                ordered_blocks = dict()
                for equation in self._equations:
                    if equation in requested_row_blocks:
                        ordered_blocks.update(
                            {equation: requested_row_blocks[equation]}
                        )
                return ordered_blocks

    def _parse_single_equation_like(
        self, equation: str | Operator | dict[str | Operator, list[GridLike]]
    ) -> dict[str, None | list[GridLike]]:
        """Another helper's helper, this time for row blocks.
        Parses non-sequential equation-likes."""
        # equation represented by string: No row-slicing
        if isinstance(equation, str):
            if equation not in self._equations:
                raise ValueError(f"Unkown equation name {equation}.")
            return {equation: None}
        # equation represented by Operator: No row-slicing
        elif isinstance(equation, Operator):
            if equation.name not in self._equations:
                raise ValueError(f"Unkown equation operator {equation}.")
            return {equation.name: None}
        # equations represented by dict with restriction to grids: get target row indices
        elif isinstance(equation, dict):
            block = dict()
            for equ, grids in equation.items():
                if isinstance(equ, Operator):
                    name = equ.name
                    if name not in self._equations:
                        raise ValueError(f"Unkown equation name {equation}.")
                elif isinstance(equ, str):
                    name = equ
                    if name not in self._equations:
                        raise ValueError(f"Unkown equation operator {equation}.")
                else:
                    raise TypeError(
                        f"Item ({type(equ)}, {type(grids)}) not parsable as equation-like."
                    )

                img_info = self._equ_image_space_composition[name]
                # check if the user requests a properly defined subsystem
                unknown_grids = set(grids).difference(set(img_info.keys()))
                if len(unknown_grids) > 0:
                    raise ValueError(
                        f"Equation {name} not defined on grids {unknown_grids}"
                    )
                block_idx = list()
                # loop over image space information to ensure correct order
                for grid in img_info:
                    if grid in grids:
                        block_idx.append(img_info[grid])
                # if indices not empty, concatenate and return
                if block_idx:
                    block.update({name: np.concatenate(block_idx, dtype=int)})
                # indices should by logic always be found, if not alert the user.
                else:
                    raise TypeError(
                        f"Equation-like item ({type(equ)}, {type(grids)}) yielded no rows."
                    )
            return block
        else:
            raise TypeError(f"Type {type(equation)} not parsable as equation-like.")

    def _gridbased_equation_complement(
        self, equations: dict[str, None | np.ndarray]
    ) -> dict[str, None | np.ndarray]:
        """Takes the information from equation parsing and finds for each equation the indices
        which were excluded in the grid-sense.
        """
        complement = dict()
        for name, idx in equations.items():
            # if indices where filtered based on grids, we find the complementing indices
            if idx:
                img_info = self._equ_image_space_composition[name]
                # assure ordering and uniqueness whole equation indexation
                all_idx = np.unique(np.hstack(img_info.values()))
                # complementing indices are found by deleting the filtered indices
                complement_idx = np.delete(all_idx, idx)
                complement.update({name: complement_idx})
            # if there was no grid-based row filtering, the complement is empty
            else:
                complement.update({name: None})

    def discretize(self, equations: Optional[str | list[str]] = None) -> None:
        """Find and loop over all discretizations in the equation operators, extract unique
        references and discretize.

        This is more efficient than discretizing on the Operator level, since
        discretizations which occur more than once in a set of equations will be
        identified and only discretized once.

        Parameters:
            equations (optional): name(s) of equation(s) to be discretized.
                If not given, all known equations will be searched and discretized.

        Raises:
            KeyError: if a given equation name is not known.

        """
        # TODO the search can be done once (in some kind of initialization) and must not be
        # done always (performance)
        if equations is None:
            equations = list(self._equations.keys())  # type: ignore
        elif isinstance(equations, str):
            equations = [equations]  # type: ignore

        # List containing all discretizations
        discr: list = []
        for eqn_name in equations:
            # this raises a key error if a given equation name is unknown
            eqn = self._equations[eqn_name]
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

        This is a shallow wrapper of :meth:`assemble_subsystem`, where the subsystem is the
        complete set of equations, variables and grids.

        Parameters:
            state (optional): see :meth:`assemble_subsystem`. Defaults to None.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the targeted state.
                The ordering of the equations (rows) is determined by the order the equations
                were added. The DOFs (columns) are ordered according the global order.
            np.ndarray: Residual vector corresponding to the targeted state,
                scaled with -1 (moved to rhs).

        """
        return self.assemble_subsystem(state=state)

    def assemble_subsystem(
        self,
        equations: Optional[EquationLike] = None,
        variables: Optional[VarLike] = None,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector using a specified subset of
        equations, variables and grids.

        The method is intended for use in splitting algorithms. Matrix blocks not
        included will simply be sliced out.

        Notes:
            The ordering of columns in the returned system are defined by the global DOF.
            The row blocks are of the same order as equations were added to this system.
            If an equation is defined on multiple grids, the respective row-block is internally
            ordered as given by the mixed-dimensional grid
            (for sd in subdomains, for intf in interfaces).

            The columns of the subsystem are assumed to be properly defined by ``variables``,
            otherwise a matrix of shape ``(M,)`` is returned. This happens if grid variables
            are passed which are unknown to this AD system.

        Parameters:
            equations (optional): a subset of equation to which the subsystem should be
                restricted.
                If not provided (None), all equations known to this manager will be included.

                The user can specify grids per equation (name) to which the subsystem should be
                restricted in the row-sense. Grids not belonging to the domain of an equation
                will raise an error.
            variables (optional): variable-like input specifying the subspace in column-sense.
                If not provided (None), all variables will be included.
            state (optional): State vector to assemble from. By default the stored ITERATE or
                STATE are used, in that order.

        Returns:
            spmatrix: (Part of the) Jacobian matrix corresponding to the targeted variable
                state, for the specified equations and variables.
            ndarray: Residual vector corresponding to the targeted variable state,
                for the specified equations. Scaled with -1 (moved to rhs).

        """
        equ_blocks = self._parse_equation_like(equations)

        # Data structures for building matrix and residual vector
        mat: list[sps.spmatrix] = []
        rhs: list[np.ndarray] = []

        # Keep track of DOFs for each equation/block
        ind_start = 0
        self.assembled_equation_indices = dict()

        # Iterate over equations, assemble.
        for equ_name, rows in equ_blocks.items():
            # this will raise a key error if the equation name is unknown
            eq = self._equations[equ_name]
            ad = eq.evaluate(self.dof_manager, state)

            # if restriction to grid-related row blocks was made,
            # perform row slicing based on information we have obtained from parsing
            if rows:
                mat.append(ad.jac[rows])
                rhs.append(ad.val[rows])
                block_length = len(rhs[-1])
            # if no grid-related row restriction was made, append the whole thing
            else:
                mat.append(ad.jac)
                rhs.append(ad.val)
                block_length = len(ad.val)

            # create indices range and shift to correct position
            block_indices = np.arange(block_length) + ind_start
            # extract last index as starting point for next block of indices
            ind_start = block_indices[-1] + 1
            self.assembled_equation_indices.update({equ_name: block_indices})

        # Concatenate results equation-wise
        if len(mat) > 0:
            A = sps.vstack(mat, format="csr")
            rhs_cat = np.concatenate(rhs)
        else:
            # Special case if the restriction produced an empty system.
            A = sps.csr_matrix((0, 0))
            rhs_cat = np.empty(0)

        # slice out the columns belonging to the requested subsets of variables and
        # grid-related column blocks by using the transposed projection to respective subspace
        # Multiply rhs by -1 to move to the rhs
        column_projection = self.projection_to(variables).transpose()
        return A * column_projection, -rhs_cat

    def assemble_schur_complement_system(
        self,
        primary_equations: EquationLike,
        primary_variables: VarLike,
        secondary_equations: Optional[EquationLike] = None,
        secondary_variables: Optional[VarLike] = None,
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
            ``excl_loc_prim_to_sec`` are meant for nested Schur-complements and splitting
            solvers. This is an advanced usage and requires the user to be sensitive about what
            he is doing, since the resulting blocks ``A_pp`` and ``A_ss`` might end up
            to be not square. This will result in errors.

        Examples:
            The default inverter can be defined by

            >>> import scipy.sparse as sps
            >>> inverter = lambda A: sps.csr_matrix(sps.linalg.inv(A.A))

            It is costly in terms of computational time and memory though.

        Parameters:
            primary_equations: a subset of equations specifying the primary subspace in
                row-sense.
            primary_variables: variable-like input specifying the primary subspace in
                column-sense.
            secondary_equations: a subset of equations specifying the secondary subspace in
                row-sense.
                By default, the complement of the primary rows is used.
            secondary_variables: variable-like input specifying the secondary subspace in
                column-sense.
                By default, the complement of the primary columns is used.
            excl_loc_prim_to_sec (optional): If True, primary local blocks which are excluded
                by the variable- and equation-like structure, are added to the secondary block.

                I.e. if a variable ``p`` is defined on two grids ``sd1, sd2``, ad the user
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

        primary_rows = self._parse_equation_like(primary_equations)
        excl_prim_rows = self._gridbased_equation_complement(primary_rows)
        prim_equ_names = list(primary_rows.keys())
        primary_projection = self.projection_to(primary_variables)
        num_dofs = primary_projection.shape[1]

        # assert non-emptiness of primary block
        assert len(primary_rows) > 0
        assert primary_projection.shape[0] > 0

        # finding secondary column indices and respective projection
        if secondary_variables:
            # default projection to secondaries
            secondary_projection = self.projection_to(secondary_variables)
            # assert primary and secondary columns do not overlap
            common_column_indices = np.intersect1d(
                primary_projection.indices, secondary_projection.indices
            )
            if common_column_indices.size > 0:
                raise ValueError("Primary and secondary columns overlap.")

            # find indices of excluded primary columns and change the secondary projection
            if excl_loc_prim_to_sec:
                # finding grid vars, who are primary in terms of name, but excluded by the
                # filter the varlike structure imposes
                excluded_grid_vars = self._gridbased_var_complement(primary_variables)
                excl_projection = self.projection_to(excluded_grid_vars)
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
            if excl_loc_prim_to_sec:
                shape = (
                    primary_projection.shape[1] - primary_projection.shape[0],
                    num_dofs,
                )
                # remove indices found in primary projection
                # csr sparse projections have only one entry per column
                idx_s = np.delete(
                    np.arange(shape[1], dtype=int), primary_projection.indices
                )
                assert len(idx_s) == shape[0]
                secondary_projection = sps.coo_matrix(
                    (np.ones(shape[0]), (np.arange(shape[0]), idx_s)),
                    shape=shape,
                ).tocsr()
            else:
                # finding grid vars, who are primary in terms of name, but excluded by the
                # filter the varlike structure imposes
                excluded_grid_vars = self._gridbased_var_complement(primary_variables)
                excl_projection = self.projection_to(excluded_grid_vars)
                # take the indices of the excluded local prims and included prims
                idx_excl = np.unique(
                    np.hstack([excl_projection.indices, primary_projection.indices])
                )
                # the secondary indices are computed by the complement of above
                idx_s = np.delete(np.arange(shape[1], dtype=int), idx_excl)
                shape = (idx_s.size, num_dofs)
                secondary_projection = sps.coo_matrix(
                    (np.ones(shape[0]), (np.arange(shape[0]), idx_s)),
                    shape=shape,
                ).tocsr()

        # finding secondary row indices
        secondary_rows: dict[str, None | np.ndarray] | None
        sec_equ_names: list[str]
        if secondary_equations:
            secondary_rows = self._parse_equation_like(secondary_equations)
            sec_equ_names = list(secondary_rows.keys())
        else:
            secondary_rows = None
            sec_equ_names = list(
                set(self._equations.keys).difference(set(primary_rows.keys()))
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
        for name in self._equations.keys():
            if name in prim_equ_names:
                A_temp, b_temp = self.assemble_subsystem(name, state=state)
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
        for name in self._equations.keys():
            if name in sec_equ_names:
                A_temp, b_temp = self.assemble_subsystem(name, state=state)
                # if secondary equations were defined, we check if we have to restrict them
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
        # prolong primary and secondary block to global-sized arrays using the transpose
        # of the projection
        X = prolong_p * reduced_solution + prolong_s * x_s
        return X

    def create_subsystem_manager(self, equations: Union[str, list[str]]) -> ADSystem:
        """Creates an ``ADSystemManager`` for a given subset of equations.

        Parameters:
            equations: equations to be assigned to the new manager.
                Must be known to this manager.

        Returns:
            a new instance of ``ADSystemManager``. The subsystem equations are ordered as
            imposed by this manager's order.

        """
        # creating a new manager and adding the requested equations
        new_manger = ADSystem(self.mdg)

        if isinstance(equations, str):
            equations = [equations]  # type: ignore

        for name in equations:
            image_info = self._equ_image_space_composition[
                name
            ]  # TODO fix this, incorrect
            new_manger.set_equation(
                name, self._equations[name], num_equ_per_dof=image_info
            )

        return new_manger

    ### special methods -----------------------------------------------------------------------

    def __repr__(self) -> str:
        s = (
            "AD System manager for mixed-dimensional grid with "
            f"{self.mdg.num_subdomains()} subdomains "
            f"and {self.mdg.num_interfaces()}"
            " interfaces.\n"
        )
        # Sort variables alphabetically, not case-sensitive
        all_vars = self.get_var_names()
        s += "Variables present on at least one grid or interface:\n\t"
        s += ", ".join(all_vars) + "\n"

        if self._equations is not None:
            eq_names = [name for name in self._equations]
            s += f"In total {len(self._equations)} equations, with names: \n\t"
            s += ", ".join(eq_names)

        return s

    def __str__(self) -> str:
        s = (
            "AD System manager for mixed-dimensional grid with "
            f"{self.mdg.num_subdomains()} subdomains "
            f"and {self.mdg.num_interfaces()}"
            " interfaces.\n"
        )

        all_vars = self.get_var_names()
        var_grid = [
            (sub_var.name, sub_var.domain)
            for var in self.variables.values()
            for sub_var in var.sub_vars
        ]
        # make combinations unique
        var_grid = list(set(var_grid))

        s += f"There are in total {len(all_vars)} variables, distributed as follows:\n"

        # Sort variables alphabetically, not case-sensitive
        for var, grid in var_grid:
            s += "\t" + f"{var} is present on"
            s += " subdomain" if isinstance(grid, pp.Grid) else " interface(s)"
            s += "\n"

        s += "\n"
        if self._equations is not None:
            eq_names = [name for name in self._equations]
            s += f"In total {len(self._equations)} equations, with names: \n\t"
            s += "\n\t".join(eq_names) + "\n"

        return s
