"""Contains the AD system manager, managing variables and equations for a system modelled
using the AD framework.

"""

from __future__ import annotations

import itertools
from enum import Enum, EnumMeta
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from . import _ad_utils

__all__ = ["ADSystem"]

GridLike = Union[pp.Grid, pp.MortarGrid]


class ADSystem:
    """Represents a physical system, modelled by AD variables and equations in AD form.

    This class provides functionalities to create and manage primary and secondary variables,
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
        >>> primary_projection = ad_sys.projection_to(categories=[var_category.primary])

    Parameters:
        mdg: mixed-dimensional grid representing the whole computational domain.
        var_categories (optional): an :class:`Enum` object containing categories for variables,
            defined by the user/modeler. They can later be used to assign categories to created
            variables and assemble respective subsystems using the enumerated object.

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

        self.dof_manager: pp.DofManager = pp.DofManager(mdg)
        """DofManager created using the passed grid."""

        self.variables: dict[str, pp.ad.MixedDimensionalVariable] = dict()
        """Contains references to (global) MergedVariables for a given name (key)."""

        self.grid_variables: dict[GridLike, dict[str, pp.ad.Variable]] = dict()
        """Contains references to local Variables and their names for a given grid (key).
        The reference is stored as another dict, which returns the variable for a given name
        (key).

        """

        self.assembled_equation_indices: dict[str, np.ndarray] = dict()
        """Contains the row indices in the last assembled (sub-) system for a given equation
        name (key). This dictionary changes with every call to any assemble-method.

        """

        ### PRIVATE

        self._equations: dict[str, pp.ad.Operator] = dict()
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

        self._Schur_complement: Optional[tuple] = None
        """Contains block matrices and the split rhs of the last assembled Schur complement,
        such that the expansion can be made.

        """

        self._vars_per_category: dict[Union[str, Enum], list[str]] = dict()
        """Contains the names of variables assigned to a specific category (key)."""

        self._block_numbers: dict[tuple[GridLike, str], int] = dict()
        """Dictionary containing the block number for a given combination of grid/ mortar grid
        and variable name (key).

        """

        self._block_dofs: np.ndarray = np.array([], dtype=int)
        """Array containing the number of DOFS per block number. The block number corresponds
        to this array's indexation.

        """

    def _filter(
        self,
        category: Optional[Union[Enum, list[Enum]]] = None,
        variable: Optional[Union[str, list[str]]] = None,
        grids: Optional[Union[GridLike, list[GridLike]]] = None,
    ) -> itertools.product:
        """Helper method to filter grid-name pairs."""
        # reformat non-sequential arguments
        if grids is None:
            grids = list(set([block_pair[0] for block_pair in self._block_numbers]))
        elif isinstance(grids, (pp.Grid, pp.MortarGrid)):
            grids = [grids]  # type: ignore

        if variable is None:
            variable = list(set([block_pair[1] for block_pair in self._block_numbers]))
        elif isinstance(variable, str):
            variable = [variable]  # type: ignore

        if category:
            assert self.var_categories is not None
            if isinstance(category, Enum):
                category = [category]  # type: ignore
            cat_filter = list()
            for cat in category:
                if cat in self.var_categories:
                    cat_filter += self._vars_per_category[cat]
                else:
                    raise ValueError(f"Unknown category {cat}.")
            # filter variables by category
            variable = [var for var in variable if var in cat_filter]

        return itertools.product(grids, variable)

    ### Variable management -------------------------------------------------------------------

    def create_variable(
        self,
        name: str,
        dof_info: dict[str, int] = {"cells": 1},
        subdomains: Optional[list[pp.Grid]] = None,
        interfaces: Optional[list[pp.MortarGrid]] = None,
        category: Optional[Enum] = None,
    ) -> pp.ad.MixedDimensionalVariable:
        """Creates a new variable according to specifications.

        This method does now assign any values to the variable. This has to be done in a
        subsequent step (using e.g. :meth:`set_var_values`).

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
            if category not in self.var_categories:
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
                new_var = pp.ad.Variable(name, dof_info, domain=sd)
                if sd not in self.grid_variables.keys():
                    self.grid_variables.update({sd: dict()})
                if name not in self.grid_variables[sd]:
                    self.grid_variables[sd].update({name: new_var})
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
                    new_var = pp.ad.Variable(name, dof_info, domain=intf)
                    if intf not in self.grid_variables.keys():
                        self.grid_variables.update({intf: dict()})
                    if name not in self.grid_variables[intf]:
                        self.grid_variables[intf].update({name: new_var})
                    variables.append(new_var)

        # create and store the md variable
        merged_var = pp.ad.MixedDimensionalVariable(variables)
        self.variables.update({name: merged_var})
        # store categorization
        if var_cat not in self._vars_per_category:
            self._vars_per_category.update({var_cat: list()})
        self._vars_per_category[var_cat].append(name)

        # append the new DOFs to the global system
        self._append_dofs(name, var_cat)

        return merged_var

    def get_var_names(
        self,
        category: Optional[Union[Enum, list[Enum]]] = None,
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
            return tuple(self._vars_per_category[pp.PRIMARY_VARIABLES])
        # if categories where assigned, we try to get the respective ones
        # unknown categories return an empty list
        else:
            var_names = list()
            if category is None:
                for vars in self._vars_per_category.values():
                    var_names += vars
                # assert uniqueness of names
                var_names = list(set(var_names))
            else:
                if isinstance(category, Enum):
                    category = [category]  # type: ignore
                for cat in category:
                    var_names += self._vars_per_category.get(cat, list())
            return tuple(var_names)

    def get_variable_vector(
        self,
        variable: str,
        from_iterate: bool = True
    ) -> np.ndarray:
        """A wrapper to return the sub vector belonging to a specified variable on all grids.

        For more details see :meth:`get_sub_vector`.

        Parameters:
            variable: name of the variable for which the values are requested.
            from_iterate: flag to return values stored as ITERATE, instead of STATE (default).

        Returns: the sub vector belonging to the specified variable in numerical format.

        """
        return self.get_sub_vector(variable=variable, from_iterate=from_iterate)

    def set_variable_vector(
        self,
        values: np.ndarray,
        variable: str,
        to_iterate: bool = False,
        additive: bool = False,
    ) -> None:
        """A wrapper to set the values for the specified variable on all grids.

        For more details see :meth:`set_sub_vector`.

        Parameters:
            values: vector of corresponding size.
            variable: name of the variable for which the values should be stored.
            to_iterate (optional): flag to write values to ITERATE, instead of STATE (default).
            additive (optional): flag to write values *additively* to ITERATE or STATE.
                To be used in iterative procedures.

        """
        self.set_sub_vector(
            values, variable=variable, to_iterate=to_iterate, additive=additive
        )

    def get_sub_vector(
        self,
        category: Optional[Union[Enum, list[Enum]]] = None,
        variable: Optional[Union[str, list[str]]] = None,
        grids: Optional[Union[GridLike, list[GridLike]]] = None,
        from_iterate: bool = False,
    ) -> np.ndarray:
        """Assembles a sub vector of the global vector of unknowns containing values for passed
        categories, variables names and grids.

        The user can rely on the order corresponding to the global order as used for the
        respective subsystem.

        The filter hierarchy corresponds to the hierarchy defined in :meth:`project_to`.

        Notes:
            An undefined combination will return an empty vector e.g., a variable which is not
            in any given category and not defined on any given grid.

        Parameters:
            category (optional): one or multiple categories of variables.
                Defaults to all categories
            variable (optional): names of variables to be contained in the sub vector.
                Defaults to all variables.
            grids (optional): grids to which the sub vector should be restricted.
                Defaults to all grids.
            from_iterate: flag to return values stored as ITERATE, instead of STATE (default).

        Returns: the respective sub vector in numerical format.

        Raises:
            AssertionError: if a category is passed despite the the system not having
                categories.
            ValueError: if an unknown category is passed.
            ValueError: if no values are stored for a grid-name combination

        """
        # storage for atomic blocks of the sub vector (identified by grid-name pairs)
        values = list()
        # assemble requested grid-variable pairs
        requested_blocks = self._filter(category, variable, grids)

        # loop over all atomic blocks and process those passing the filter
        for grid, var in self._block_numbers.keys():
            if (grid, var) in requested_blocks:
                if isinstance(grid, pp.Grid):
                    data = self.mdg.subdomain_data(grid)
                elif isinstance(grid, pp.MortarGrid):
                    data = self.mdg.interface_data(grid)
                # extract a copy of requested values
                try:
                    if from_iterate:
                        values.append(data[pp.STATE][pp.ITERATE][var].copy())
                    else:
                        values.append(data[pp.STATE][var].copy())
                except KeyError:
                    raise ValueError(
                        f"No values stored for variable {var}, from_iterate={from_iterate}"
                        f"\non grid {grid}."
                    )

        # if there are atomic blocks, concatenate and return
        if values:
            return np.concatenate(values)
        # else return an empty vector
        else:
            return np.array([])

    def set_sub_vector(
        self,
        values: np.ndarray,
        category: Optional[Union[Enum, list[Enum]]] = None,
        variable: Optional[Union[str, list[str]]] = None,
        grids: Optional[Union[GridLike, list[GridLike]]] = None,
        to_iterate: bool = False,
        additive: bool = False,
    ) -> None:
        """Sets values for a sub vector of the global vector of unknowns specified by
        categories, variable names and grids.

        The order of values is assumed to fit the respective subsystem. The size as well.

        The filter hierarchy corresponds to the hierarchy defined in :meth:`project_to`.

        Notes:
            The vector is assumed to be of proper size and will be dissected according to the
            global order, starting with the index 0.
            Mismatches of is-size and should-be-size according to specified categories, names
            and grids will raise a value error.

        Parameters:
            values: vector of corresponding size.
            category (optional): categories of variables to which the values should be
                distributed
            variable (optional): names of variables to which the values should be distributed.
            grids (optional): grids for which the values are assigned.
            to_iterate (optional): flag to write values to ITERATE, instead of STATE (default).
            additive (optional): flag to write values *additively* to ITERATE or STATE.
                To be used in iterative procedures.

        """
        # assemble requested grid-variable pairs
        requested_blocks = self._filter(category, variable, grids)
        # start of dissection
        block_start = 0
        block_end = 0

        for block, block_number in self._block_numbers.items():
            if block in requested_blocks:
                grid, var = block
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
                        data[pp.STATE][pp.ITERATE][var] = (
                            data[pp.STATE][pp.ITERATE][var] + local_vec
                        )
                    else:
                        data[pp.STATE][var] = data[pp.STATE][var] + local_vec
                else:
                    if to_iterate:
                        data[pp.STATE][pp.ITERATE][var] = local_vec.copy()
                    else:
                        data[pp.STATE][var] = local_vec.copy()

                # move dissection forward
                block_start = block_end

        # last sanity check if the vector was properly sized, or if it was too large.
        # This imposes a theoretically unnecessary restriction on the input argument
        # since we only require a vector of at least this size.
        # Do we care if there are more values than necessary? TODO
        assert block_end == values.size

    def get_global_vector(self, from_iterate: bool = False) -> np.ndarray:
        """A wrapper to return the vector of unknowns containing all variables on all grids.

        For more details see :meth:`get_sub_vector`.

        Parameters:
            from_iterate: flag to return values stored as ITERATE, instead of STATE (default).

        Returns: the global vector of unknowns in numerical format.

        """
        variables = list(self.variables.keys())
        return self.get_sub_vector(variable=variables, from_iterate=from_iterate)

    def set_global_vector(
        self,
        values: np.ndarray,
        to_iterate: bool = False,
        additive: bool = False
    ) -> None:
        """A wrapper to set the global vector of unknowns containing all variables on all
        grids.

        For more details see :meth:`set_sub_vector`.

        Parameters:
            values: vector of size ``num_dofs``.
            to_iterate (optional): flag to write values to ITERATE, instead of STATE (default).
            additive (optional): flag to write values *additively* to ITERATE or STATE.
                To be used in iterative procedures.

        """
        variables = list(self.variables.keys())
        self.set_sub_vector(
            values, variable=variables, to_iterate=to_iterate, additive=additive
        )

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
        new_block_numbers: dict[tuple[GridLike, str], int] = dict()
        # new dofs per block
        new_block_dofs_: list[int] = list()

        # add dofs found on subdomains for this variable name
        for sd, data in self.mdg.subdomains(return_data=True):
            if var_cat in data:
                if var_name in data[var_cat]:
                    # last sanity check that no previous data is overwritten
                    # should not happen if class not used in hacky way
                    assert (sd, var_name) not in self._block_numbers.keys()

                    # assign a new block number and increase the counter
                    new_block_numbers[(sd, var_name)] = last_block_number
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
                    assert (intf, var_name) not in self._block_numbers.keys()

                    # assign a new block number and increase the counter
                    new_block_numbers[(intf, var_name)] = last_block_number
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
        new_block_numbers: dict[tuple[GridLike, str], int] = dict()
        new_block_dofs: list[int] = list()
        block_pair: tuple[GridLike, str]  # appeasing mypy

        # first set of diagonal blocks, per subdomain
        for sd in self.mdg.subdomains():
            # sub-loop of variables per subdomain
            for var_name in self.variables:
                block_pair = (sd, var_name)
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
                block_pair = (intf, var_name)
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

    def num_dofs(self) -> int:
        """Returns the total number of dofs managed by this system."""
        return int(sum(self._block_dofs))  # cast numpy.int64 into Python int

    def projection_to(
        self,
        category: Optional[Union[Enum, list[Enum]]] = None,
        variable: Optional[Union[str, list[str]]] = None,
        grids: Optional[Union[GridLike, list[GridLike]]] = None,
    ) -> sps.csr_matrix:
        """Create a projection matrix from the global vector of unknowns to a specified
        subspace.

        The subspace can be specified by variable categories, variable names and grids.

        The filtering hierarchy is:

        1. If no category is passed, all known variable names are used.
           Otherwise only names associated with passed categories are used.
        2. If no variable names are passed, all names passing the category filter are used.
           Otherwise only names inside categories matching the passed names are used.
           If a name is not associated with any passed category, it is ignored.
        3. The grid filter is applied to all variables passing the category and name filter,
           narrowing the subspace further down to specific domains (grids or mortar grids).

        The transpose of the returned matrix can be used to slice respective columns out of the
        global Jacobian.

        The projection preserves the global order defined by the system, i.e. it includes no
        permutation.

        Notes:
            If any combination of category, name and grid does not match any variable in this
            system, the respective partial projection projects into the null space!
            I.e. if a complete mismatch of combinations is passed
            the resulting projection will be of shape ``(,num_dofs)``,
            where ``num_dofs`` is given by :meth:`num_dofs`.

        Parameters:
            category (optional): one or multiple categories defined during instantiation.
            variable (optional): names of variables to be projected on.
            grids (optional): grids or mortar grids to which the projection should be
                restricted.

        Returns:
            a sparse projection matrix of shape ``(M,num_dofs)``, where ``0<=M<=num_dofs``.

        Raises:
            AssertionError: if a category is passed despite the the system not having
                categories.
            ValueError: if an unknown category is passed.

        """

        # current number of total dofs
        num_dofs = self.num_dofs()
        # Array for the dofs associated with each argument combination
        inds = []
        # assemble requested grid-variable pairs
        requested_blocks = self._filter(category, variable, grids)

        # loop over all blocks and process those passing the filter
        for grid, var in self._block_numbers.keys():
            if (grid, var) in requested_blocks:
                inds.append(self.dofs_of(var, grid))

        if len(inds) == 0:
            # Special case if no indices were returned
            return sps.csr_matrix((0, num_dofs))

        # Create projection matrix. Uniquify indices here, both to sort (will preserve
        # the ordering of the unknowns given by the DofManager) and remove duplicates
        # (in case variables were specified more than once).
        local_dofs = np.unique(np.hstack(inds))
        num_local_dofs = local_dofs.size

        return sps.coo_matrix(
            (np.ones(num_local_dofs), (np.arange(num_local_dofs), local_dofs)),
            shape=(num_local_dofs, num_dofs),
        ).tocsr()

    def dofs_of(
        self,
        variable: str,
        grids: Optional[Union[GridLike, list[GridLike]]] = None,
    ) -> np.ndarray:
        """Get the indices in the global vector of unknowns belonging to the variable.

        For variables defined on multiple grids in the mixed-dimensional sense, an additional
        grid filter can be passed to return indices belonging only to the respective grids.

        The global order of indices is preserved.

        Parameters:
            variable: name of the variable for which the indices should be returned
            grids: one or multiple grids on which the variable is defined.

        Returns:
            an order-preserving array of indices of DOFs for this combination of variable name
            and domain(s).

        Raises:
            KeyError: if an undefined combination of grid and variable is passed.

        """
        # global block indices
        global_block_dofs = np.hstack((0, np.cumsum(self._block_dofs)))
        ## first and most simple case, single grid for a given variable:
        if isinstance(grids, (pp.Grid, pp.MortarGrid)):
            # this will raise a key error if the combination is unknown
            block_number = self._block_numbers[(grids, variable)]
            return np.arange(
                global_block_dofs[block_number],
                global_block_dofs[block_number + 1],
                dtype=int,
            )
        else:
            # all grids the variable is defined, preserving the global order
            var_grids = [
                block[0] for block in self._block_numbers if block[1] == variable
            ]
            grid_indices = list()
            ## second case, indices for all grids the variable is defined on (no grid filter)
            if grids is None:
                # append indices per grid
                for grid in var_grids:
                    # this will raise a key error if the combination is unknown
                    block_number = self._block_numbers[(grid, variable)]
                    grid_indices.append(
                        np.arange(
                            global_block_dofs[block_number],
                            global_block_dofs[block_number + 1],
                            dtype=int,
                        )
                    )
            ## third case, indices for requested grids
            else:
                # append indices per grid, if they pass the filter
                # loop over all domains of the variable to preserve the global order
                for grid in var_grids:
                    if grid not in grids:
                        continue
                    # this will raise a key error if the combination is unknown
                    block_number = self._block_numbers[(grid, variable)]
                    grid_indices.append(
                        np.arange(
                            global_block_dofs[block_number],
                            global_block_dofs[block_number + 1],
                            dtype=int,
                        )
                    )
            # concatenate the indices on multiple grids
            return np.concatenate(grid_indices, dtype=int)

    def block_of_dof(self, dof: int) -> tuple[GridLike, str]:
        """Identifies the grid and variable to which a specific DOF index belongs.

        The intended use is to help identify entries in the global vector or the column of the
        Jacobian, which do not behave as expected.

        Parameters:
            dof: a single index in the global vector.

        Returns: a tuple of grid and variable name associated with the passed dof.

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
                return block_pair
        # if search was not successful, something went terribly wrong
        # should never happen, but if it does, notify the user
        raise RuntimeError("Someone messed with the global block indexation...")

    ### Equation management -------------------------------------------------------------------

    def set_equation(
        self,
        name: str,
        equation: pp.ad.Operator,
        num_equ_per_dof: dict[GridLike, dict[str, int]],
    ) -> None:
        """Sets an equation and assigns the given name.

        If an equation already exists under that name, it is overwritten.

        Information about the image space must be provided for now, such that grid-wise row
        slicing is possible. This will hopefully be provided automatically in the future. TODO

        Parameters:
            name: given name for this equation. Used as identifier and key.
            equation: An equation in AD operator form, assuming the right-hand side is
                zero and this instance represents the left-hand side.
                The equation mus be ready for evaluation! i.e. all involved variables must have
                values set.
            num_equ_per_dof: a dictionary describing how many equations ``equation_operator``
                provides. This is a temporary work-around until operators are able to provide
                information on their image space.
                The dictionary must contain the number of equations per admissible dof type,
                (see :data:`porepy.DofManager.admissible_dof_types`),
                for each grid the operator was defined on.

                The order of the items in ``num_equ_per_dof`` must be consistent with the
                general order of grids, as used by the dof manager and implemented in various
                discretizations used inside ``equation_operator``.

        Raises:
            AssertionError: if the equation is defined on an unknown grid.
            ValueError: if indicated number of equations does not match the actual number as
                per evaluation of operator.

        """
        image_info: dict[GridLike, np.ndarray] = dict()
        total_num_equ = 0
        valid_grids = [sd for sd in self.mdg.subdomains()]
        valid_intf = [intf for intf in self.mdg.interfaces()]

        for grid, dof_info in num_equ_per_dof.items():
            # calculate number of equations per grid
            if isinstance(grid, pp.Grid):
                assert grid in valid_grids
                num_equ_per_grid = (
                    grid.num_cells * dof_info.get("cells", 0)
                    + grid.num_nodes * dof_info.get("nodes", 0)
                    + grid.num_faces * dof_info.get("faces", 0)
                )
            # mortar grids have only cell-wise dofs
            elif isinstance(grid, pp.MortarGrid):
                assert grid in valid_intf
                num_equ_per_grid = grid.num_cells * dof_info.get("cells", 0)
            # should not happen, alert anyways
            else:
                raise RuntimeError("Grid checking failed for some reason...")
            # create operator-specific block indices with shift regarding previous blocks
            block_idx = np.arange(num_equ_per_grid) + total_num_equ
            # cumulate total number of equations
            total_num_equ += num_equ_per_grid
            # store block idx per grid
            image_info.update({grid: block_idx})

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

    def get_equation(self, name: str) -> pp.ad.Operator:
        """
        Returns: a reference to a previously passed equation in operator form.

        Raises:
            KeyError: if ``name`` does not correspond to any known equation.

        """
        return self._equations[name]

    def remove_equation(self, name: str) -> pp.ad.Operator | None:
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

    def discretize(self, equations: Optional[list[str]] = None) -> None:
        """Find and loop over all discretizations in the equation operators, extract unique
        references and discretize.

        This is more efficient than discretizing on the Operator level, since
        discretizations which occur more than once in a set of equations will be
        identified and only discretized once.

        Parameters:
            equations (optional): name of equations to be discretized.
                If not given, all known equations will be searched and discretized.

        Raises:
            KeyError: if a given equation name is not known.

        """
        # TODO the search can be done once (in some kind of initialization) and must not be
        # done always (performance)
        if equations is None:
            equations = list(self._equations.keys())  # type: ignore

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

    @staticmethod
    def _recursive_discretization_search(operator: pp.ad.Operator, discr: list) -> list:
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

    def assemble(
        self,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector of the whole system.

        This is a shallow wrapper of :meth:`assemble_subsystem`, where the subsystem is the
        complete set of equations, primary and secondary variables, and grids.

        Parameters:
            state (optional): see :meth:`assemble_subsystem`. Defaults to None.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the targeted state.
                The ordering of the equations (rows) is determined by the order the equations
                were added. The DOFs (columns) are ordered as imposed by the DofManager.
            np.ndarray: Residual vector corresponding to the targeted state,
                scaled with -1 (moved to rhs).

        """
        return self.assemble_subsystem(state=state)

    def assemble_subsystem(
        self,
        equations: Optional[Union[str, list[str]]] = None,
        variables: Optional[Union[str, list[str]]] = None,
        grid_rows: Optional[Union[GridLike, list[GridLike]]] = None,
        grid_columns: Optional[Union[GridLike, list[GridLike]]] = None,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector using a specified subset of
        equations and variables.

        The method is intended for use in splitting algorithms. Matrix blocks not
        included will simply be sliced out.

        Notes:
            The ordering of columns in the returned system are defined by the global DOF order
            provided by the DofManager. The rows are of the same order as equations were added
            to this system.

            If a combination of variables and grids is chosen, s.t. the variables were **not**
            defined on those grids, the resulting matrix is a zero matrix which **does not**
            appear as a block in the complete assembly using :meth:`assemble`,
            because the methods assemble by default only columns belonging to well-defined dofs
            , i.e. dofs for which the the grid-var combo was properly defined.
            The matrix is of shape ``num_equations x 0``. (see project_to of DofManager)
            This means that this method can create (possibly empty ) blocks which are **not**
            in the global system. TODO ask IS and EK if an error should be raised instead.
            This is connected with the TODO in project_to of the DofManager
            (0 projection or error or projection with adequate size but still 0)

        Parameters:
            equations (optional): a subset of equation names to which the subsystem should be
                restricted.
                If not provided (None), all equations known to this manager will be included.
            variables (optional): names of variables to which the subsystem should be
                restricted. If not provided (None), all variables will be included.
            grid_rows (optional): grids or mortar grids which should be kept in the row-wise
                sense, i.e. subsystems on specified domains.
                If not provided (None), all involved grids and mortar grids will be included.
            grid_columns (optional): grids or mortar grids for which the column-wise
                contribution to each equation should be kept. This is a narrowing down of the
                restriction imposed by ``variables``.
                If not provided (None), all grids and mortar grids will be included.
            state (optional): State vector to assemble from. By default the stored ITERATE or
                STATE are used, in that order.

        Returns:
            spmatrix: (Part of the) Jacobian matrix corresponding to the targeted variable
                state, for the specified equations and columns.
            ndarray: Residual vector corresponding to the targeted variable state,
                for the specified equations. Scaled with -1 (moved to rhs).

        """
        # reformat non-sequential arguments
        # if no restriction, use all variables and grids
        if variables is None:
            variables = list(self.variables.keys())  # type: ignore
        elif isinstance(variables, str):
            variables = [variables]  # type: ignore
        if equations is None:
            equations = list(self._equations.keys())  # type: ignore
        elif isinstance(equations, str):
            equations = [equations]  # type: ignore
        if isinstance(grid_rows, (pp.Grid, pp.MortarGrid)):
            grid_rows = [grid_rows]  # type: ignore
        if grid_columns is None:
            grid_columns = list(
                set([key[0] for key in self.dof_manager.block_dof])
            )  # type: ignore
        elif isinstance(grid_columns, (pp.Grid, pp.MortarGrid)):
            grid_columns = [grid_columns]  # type: ignore

        # TODO think about argument validation, i.e. are the variables, equations and grids
        # known to this system and dof manager.
        # This will help users during debugging, otherwise just some key errors will be raised
        # at some points.

        # Data structures for building matrix and residual vector
        mat: list[sps.spmatrix] = []
        rhs: list[np.ndarray] = []

        # Keep track of DOFs for each equation/block
        ind_start = 0
        self.assembled_equation_indices = dict()

        # Iterate over equations, assemble.
        for equ_name in equations:
            eq = self._equations[equ_name]
            ad = eq.evaluate(self.dof_manager, state)
            # if restriction to grid-related row blocks was made,
            # perform row slicing based on information we have on the image
            if grid_rows:
                # store row blocks for an equation related to a specific grid
                equ_mat = list()
                equ_rhs = list()
                # if this equation is defined on one of the restricted grids,
                # slice out respective rows in an order-preserving way
                # the order is stored in the image space composition
                img_info = self._equ_image_space_composition[equ_name]
                for grid, block_idx in img_info.items():
                    if grid in grid_rows:
                        equ_mat.append(ad.jac[block_idx])
                        equ_rhs.append(ad.val[block_idx])
                # stack the sliced out blocks vertically and append as equation block to the
                # resulting subsystem
                mat.append(sps.vstack(equ_mat, format="csr"))
                rhs.append(np.concatenate(equ_rhs))
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
        # grid-related column blocks by using transposed projection for the global dof vector
        # Multiply rhs by -1 to move to the rhs
        column_projection = self.dof_manager.projection_to(
            variables, grid_columns
        ).transpose()
        return A * column_projection, -rhs_cat

    def assemble_schur_complement_system(
        self,
        primary_equations: Union[str, list[str]],
        primary_variables: Union[str, list[str]],
        inverter: Callable[[sps.spmatrix], sps.spmatrix] = sps.linalg.inv,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector using a Schur complement
        elimination of the variables and equations not to be included.

        The specified equations and variables will define a reordering of the linearized
        system into

            [A_pp, A_ps  [x_p   = [b_p
             A_sp, A_ss]  x_s]     b_s]

        Where subscripts p and s define primary and secondary quantities. The Schur
        complement system is then given by

            (A_pp - A_ps * inv(A_ss) * A_sp) * x_p = b_p - A_ps * inv(A_ss) * b_s.

        The Schur complement is well-defined only if the inverse of A_ss exists,
        and the efficiency of the approach assumes that an efficient inverter for
        A_ss can be found. The user must ensure both requirements are fulfilled.
        The simplest option (default) is a lambda function of the form:

        .. code:: Python

            inverter = lambda A: sps.csr_matrix(sps.linalg.inv(A.A))

        (used by default) but depending on A (size and sparsity pattern),
        this can be costly in terms of computational time and memory.

        The method can be used e.g. for splitting between primary and secondary variables,
        where the latter can be efficiently eliminated (for instance, they contain no
        spatial derivatives).

        Parameters:
            primary_equations: equations to be assembled, representing the row-block A_pp.
                Should have length > 0.
            primary_variables: names of variables representing the columns of A_pp.
                Should have length > 0.
            inverter (optional): callable object to compute the inverse of the matrix A_ss.
                If not given (None), the scipy sparse inverter is used.
            state (optional): see :meth:`assemble_subsystem`. Defaults to None.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the current variable state,
                as found in grid dictionaries for the specified equations and variables.
            np.ndarray: Residual vector corresponding to the current variable state,
                as found in grid dictionaries, for the specified equations and variables.
                Scaled with -1 (moved to rhs).

        """
        # reformatting non-sequential arguments
        if isinstance(primary_variables, str):
            primary_variables = [primary_variables]  # type: ignore
        if isinstance(primary_equations, str):
            primary_equations = [primary_equations]  # type: ignore
        # ensuring the Schur complement is not the whole matrix
        if len(primary_variables) == 0:
            raise ValueError("Must make Schur complement with at least one variable")
        if len(primary_equations) == 0:
            raise ValueError("Must make Schur complement with at least one equation")

        # Get lists of all variables and equations, and find the secondary items
        # by a set difference
        all_variables = self.dof_manager.get_variables()
        all_eq_names = list(self._equations.keys())
        secondary_equations = list(set(all_eq_names).difference(set(primary_equations)))
        secondary_variables = list(
            set(all_variables).difference(set(primary_variables))
        )

        # First assemble the primary and secondary equations for all variables and all grids
        # save the indices and shift the indices for the second assembly accordingly
        # Keep track of DOFs for each equation/block
        ind_start = 0
        assembled_equation_indices = dict()
        A_p, b_p = self.assemble_subsystem(
            equations=primary_equations, variables=all_variables, state=state
        )
        for equ in primary_equations:
            # get assembled subsystem indexing, accumulate length of the primary block
            block_indices = self.assembled_equation_indices[equ]
            ind_start += len(block_indices)
            # store the primary block indices as given
            assembled_equation_indices.update({equ: block_indices})

        A_s, b_s = self.assemble_subsystem(
            equations=secondary_equations, variables=all_variables, state=state
        )
        # shift the secondary equation blocks
        for equ in secondary_equations:
            # get the indexing of the secondary subsystem and shift them by the length of the
            # blocks before
            block_indices = self.assembled_equation_indices[equ] + ind_start
            assembled_equation_indices.update({equ: block_indices})

        # store the indices for the Schur complement assembly
        self.assembled_equation_indices = assembled_equation_indices

        # Projection matrices to reduce matrices to the relevant columns
        # case where no further restriction on grids is made, projection is simple to compute
        proj_primary = self.dof_manager.projection_to(primary_variables).transpose()
        proj_secondary = self.dof_manager.projection_to(secondary_variables).transpose()

        # Matrices involved in the Schur complements
        A_pp = A_p * proj_primary
        A_ps = A_p * proj_secondary
        A_sp = A_s * proj_primary
        A_ss = A_s * proj_secondary

        # Explicitly compute the inverse of the secondary block.
        # Depending on the matrix, and the inverter, this can take a long time.
        # this should raise an error if the secondary block is not invertible
        inv_A_ss = inverter(A_ss)

        S = A_pp - A_ps * inv_A_ss * A_sp
        rhs_S = b_p - A_ps * inv_A_ss * b_s

        # storing necessary information for Schur complement expansion
        self._Schur_complement = (inv_A_ss, b_s, A_sp, proj_primary, proj_secondary)

        return S, rhs_S

    def expand_schur_complement_solution(
        self, reduced_solution: np.ndarray
    ) -> np.ndarray:
        """Expands the solution of the **last assembled** Schur complement system to the
        global solution.

        I.e it takes x_p from

            [A_pp, A_ps  [x_p   = [b_p
             A_sp, A_ss]  x_s]     b_s]

        and returns the whole [x_p, x_s] where

            x_s = inv(A_ss) * (b_s - A_sp * x_p)

        Notes:
            This method works with any vector of fitting size, i.e. be aware of what you do.

        Parameters:
            reduced_solution: Solution to the linear system returned by
                :meth:`assemble_schur_complement_system`

        Returns:
            the complete solution, where ``reduced_solution`` constitutes the first part of the
            vector.

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

    def create_subsystem_manager(
        self, equations: Union[str, list[str]]
    ) -> ADSystem:
        """Creates an ``ADSystemManager`` for a given subset of equations.

        Parameters:
            equations: equations to be assigned to the new manager.
                Must be known to this manager.

        Returns:
            a new instance of ``ADSystemManager``. The subsystem equations are ordered as
            imposed by this manager's order.

        """
        # creating a new manager and adding the requested equations
        new_manger = ADSystem(self.dof_manager)

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
