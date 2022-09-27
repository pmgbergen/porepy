"""Contains the AD system manager, managing variables and equations for a system modelled
using the AD framework.

"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import porepy as pp
import scipy.sparse as sps

from . import _ad_utils

__all__ = ["ADSystemManager"]

GridLike = Union[pp.Grid, pp.MortarGrid]


class ADSystemManager:
    """Represents a system of equations, modelled by AD variables and equations in AD form.

    This class provides functionalities to create and manage primary and secondary variables,
    as well as managing equations in AD operator form.

    It further provides functions to assemble subsystems, using subsets of equations and
    primary variables.
    Secondary variables are not considered for creating e.g., the Jacobian and only their
    values are used (similar to parameters).

    Notes:
        As of now, the system matrix (Jacobian) is assembled with respect to ALL variables
        (including secondary ones) and then the columns belonging to primary variables are
        sliced out and returned. This can be optimized with minor changes to the AD operator
        class and its recursive forward AD mode.

        This class also lacks optimization methods reducing the band width of the resulting
        Jacobian. This will be added in the future.

    Parameters:
        dof_manager: The DOF manager, providing read-write functionalities to data dictionaries
            and the mixed-dimensional grid itself.

    """

    def __init__(self, dof_manager: pp.DofManager) -> None:

        ### PUBLIC

        self.dof_manager: pp.DofManager = dof_manager
        """DofManager passed at instantiation."""

        self.variables: Dict[str, pp.ad.MergedVariable] = dict()
        """Contains references to (global) MergedVariables for a given name (key)."""

        self.grid_variables: Dict[GridLike, Dict[str, pp.ad.Variable]] = dict()
        """Contains references to local Variables and their names for a given grid (key).
        The reference is stored as another dict, which returns the variable for a given name
        (key).

        """

        self.equations: Dict[str, pp.ad.Operator] = dict()
        """Contains references to equations in AD operator form for a given name (key)."""

        self.assembled_equation_indices: Dict[str, np.ndarray] = dict()
        """Contains the row indices in the last assembled (sub-) system for a given equation
        name (key).

        """

    ### Variable management -------------------------------------------------------------------

    def create_variable(
        self,
        name: str,
        primary: bool,
        dof_info: dict[str, int] = {"cells": 1},
        subdomains: Union[None, list[pp.Grid]] = list(),
        interfaces: Optional[list[pp.MortarGrid]] = None,
    ) -> pp.ad.MergedVariable:
        """Creates a new variable according to specifications.

        Parameters:
            name: used here as an identifier. Can be used to associate the variable with some
                physical quantity.
            primary: indicator if primary variable. Otherwise it is a secondary variable and
                will not contribute to the global DOFs. **This is a coming feature. Currently
                all variables are treated as primary in the system.**
            dof_info: dictionary containing information about number of DOFs per admissible
                type (see :data:`admissible_dof_types`). Defaults to ``{'cells':1}``.
            subdomains: List of subdomains on which the variable is defined. If None, then it
                will not be defined on any subdomain. If the list is empty, it will be defined
                on **all** subdomains. Defaults to empty list.
            interfaces: List of interfaces on which the variable is defined. If None, then it
                will not be defined on any interface. Here an empty list is equally treated as
                None. Defaults to none.

        Returns:
            a merged variable with above specifications.

        Raises:
            ValueError: If non-admissible DOF types are used as local DOFs.
            ValueError: If one attempts to create a variable not defined on any grid.
            KeyError: If a variable with given name is already defined.

        """
        # sanity check for admissible DOF types
        requested_type = set(dof_info.keys())
        if not requested_type.issubset(self.dof_manager.admissible_dof_types):
            non_admissible = requested_type.difference(
                self.dof_manager.admissible_dof_types
            )
            raise ValueError(f"Non-admissible DOF types {non_admissible} requested.")
        # sanity check if variable is defined anywhere
        if subdomains is None and interfaces is None:
            raise ValueError(
                "Cannot create variable not defined on any grid or interface."
            )
        # check if variable was already defined
        if name in self.variables.keys():
            raise KeyError(f"Variable with name '{name}' already created.")

        # if an empty list was received, we use ALL subdomains
        if isinstance(subdomains, list) and len(subdomains) == 0:
            subdomains = [sg for sg in self.dof_manager.mdg.subdomains()]

        variables = list()

        # TODO activate when feature becomes available
        # if primary:
        #     variable_category = pp.PRIMARY_VARIABLES
        # else:
        #     variable_category = pp.SECONDARY_VARIABLES
        variable_category = pp.PRIMARY_VARIABLES

        if isinstance(subdomains, list):
            for sd in subdomains:
                data = self.dof_manager.mdg.subdomain_data(sd)

                # prepare data dictionary if this was not done already
                if variable_category not in data:
                    data[variable_category] = dict()
                if pp.STATE not in data:
                    data[pp.STATE] = dict()
                if pp.ITERATE not in data[pp.STATE]:
                    data[pp.STATE][pp.ITERATE] = dict()

                data[variable_category].update({name: dof_info})

                # create grid-specific variable
                new_var = pp.ad.Variable(name, dof_info, subdomains=[sd])
                if not sd in self.grid_variables.keys():
                    self.grid_variables.update({sd: dict()})
                if not name in self.grid_variables[sd].keys():
                    self.grid_variables[sd].update({name: new_var})
                variables.append(new_var)

        if isinstance(interfaces, list):
            for intf in interfaces:
                data = self.dof_manager.mdg.interface_data(intf)

                if (
                    intf.codim == 2
                ):  # no variables in points TODO check if this up-to-date
                    continue
                else:
                    # prepare data dictionary if this was not done already
                    if variable_category not in data:
                        data[variable_category] = dict()
                    if pp.STATE not in data:
                        data[pp.STATE] = dict()
                    if pp.ITERATE not in data[pp.STATE]:
                        data[pp.STATE][pp.ITERATE] = dict()

                    # store DOF information about variable
                    data[variable_category].update({name: dof_info})

                    # create mortar grid variable
                    new_var = pp.ad.Variable(
                        name, dof_info, interfaces=[intf], num_cells=intf.num_cells
                    )
                    if not intf in self.grid_variables.keys():
                        self.grid_variables.update({intf: dict()})
                    if not name in self.grid_variables[intf].keys():
                        self.grid_variables[intf].update({name: new_var})
                    variables.append(new_var)

        # create and store the merged variable
        merged_var = pp.ad.MergedVariable(variables)
        self.variables.update({name: merged_var})

        # append the new DOFs to the global system if a primary variable has been created
        if primary:
            self.dof_manager.append_dofs([name])

        return merged_var

    def set_var_values(
        self, var_name: str, values: np.ndarray, copy_to_state: bool = False
    ) -> None:
        """Sets values for a given variable name in the grid data dictionaries.

        It is assumed the variable (name) is known to this instance.
        This is a shallow wrapper for respective functionalities of the DOF manager.
        The values are set for the ITERATE, additionally to the STATE if flagged.

        Parameters:
            var_name: name of the :class:`~porepy.ad.MergedVariable` for which the STATE should
                be set.
            values: respective values. It is assumed the exactly as many values are provided
                as can fit in the global DOF vector with the variable's respective DOF indexes.
            copy_to_state: copies the values additionally to the STATE.

        Raises:
            KeyError: If the variable name is not known to this instance.

        """
        if var_name not in self.variables.keys():
            raise KeyError(f"Unknown variable '{var_name}'.")

        variable = [var_name]

        # inserting the values in a global-sized zero vector.
        X = np.zeros(self.dof_manager.num_dofs())
        dof = self.dof_manager.dof_var(variable)
        X[dof] = values
        # setting ITERATE and optionally STATE
        self.dof_manager.distribute_variable(X, variables=variable, to_iterate=True)
        if copy_to_state:
            self.dof_manager.distribute_variable(X, variables=variable)

    def get_var_values(self, var_name: str, from_iterate: bool = True) -> np.ndarray:
        """Gets all values of variable ``var_name`` in a local vector by slicing respective
        indices out of the global vector.

        Preserves the order induced by the global DOF vector.
        This is a shallow wrapper for respective functionalities of the DOF manager.

        Parameters:
            var_name: name of the variable for which the value vector is requested.
            from_iterate: flag to get values from the ITERATE instead of STATE.
                Defaults to True.

        Returns:
            value vector. It's size depends on the number of local DOFs of this variable and on
            how many grids it is defined.

        Raises:
            KeyError: If the variable name is not known to this instance.

        """
        if var_name not in self.variables.keys():
            raise KeyError(f"Unknown variable '{var_name}'.")
        # extrating the values from the global vector
        dof = self.dof_manager.dof_var([var_name])
        X = self.dof_manager.assemble_variable(
            variables=[var_name], from_iterate=from_iterate
        )
        return X[dof]

    def get_var_names_from(
        self,
        variables: Sequence[Union[pp.ad.Variable, pp.ad.MergedVariable]],
    ) -> tuple[pp.ad.Variable]:
        """Extracts the (unique) names of passed variables.

        Merged variables are unravelled into single-domain variables.

        Parameters:
            variables: variables to be unravelled.

        Returns:
            a list of unique names of variables present in the passed sequence of AD variables.

        """

        # A set keeps the names unique
        var_names = set()

        # Loop over all variables, add the name of the variable itself, or its sub-variables
        for var in variables:
            # Unravels each variable inside a MergedVariable
            # TODO is this really necessary? Can a merged var contain different sd vars?
            if isinstance(var, pp.ad.MergedVariable):
                for sv in var.sub_vars:
                    var_names.add(sv.name)
            elif isinstance(var, pp.ad.Variable):
                var_names.add(var.name)
            else:
                raise TypeError(
                    f"Encountered unknown variable type '{type(var)}' in argument list."
                )

        return tuple(var_names)

    ### Equation management -------------------------------------------------------------------

    def set_equation(self, name: str, equation_operator: pp.ad.Operator) -> None:
        """Sets an equation and assigns the given name.

        If an equation already exists under that name, it is overwritten.

        Parameters:
            name: given name for this equation. Used as identifier and key.
            equation_operator: An equation in AD operator form, assuming the right-hand side is
                zero and this instance represents the left-hand side.

        """
        self.equations.update({name: equation_operator})

    ### System assembly and discretization ----------------------------------------------------

    def discretize(self) -> None:
        """Loop over all discretizations in self.equations, find all unique discretizations
        and discretize.

        This is more efficient than discretizing on the Operator level, since
        discretizations which occur more than once in a set of equations will be
        identified and only discretized once.

        """
        # Somehow loop over all equations, discretize identified objects
        # (but should also be able to do re-discretization based on
        # dependency graph etc.).

        # List of discretizations, build up by iterations over all equations
        discr: list = []
        for eqn in self.equations.values():
            # This will expand the list discr with new discretizations.
            # The list may contain duplicates.
            discr = eqn._identify_subtree_discretizations(discr)

        # Uniquify to save computational time, then discretize.
        unique_discr = _ad_utils.uniquify_discretization_list(discr)
        _ad_utils.discretize_from_list(unique_discr, self.dof_manager.mdg)

    def assemble(
        self,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector.

        If ``state`` is given, this method will use the data stored in grid dictionaries.
        It will use ITERATE. Alternatively it will use STATE (as per AD operator standard).
        Derivatives for secondary variables are not included in the final Jacobian matrix.

        Parameters:
            state (optional): State vector to assemble from. Defaults to None.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the targeted state.
                The ordering of the equations (rows) is determined by the order the equations
                were added. The DOFs (columns) are ordered as imposed by the DofManager.
            np.ndarray: Residual vector corresponding to the targeted state,
                scaled with -1 (moved to rhs).

        """
        ## Data structures for building the Jacobian matrix and residual vector
        # list of row blocks
        mat = list()
        # list of corresponding row blocks of the rhs
        rhs: list[np.ndarray] = []

        # Keep track of DOFs for each equation/block
        ind_start = 0
        self.assembled_equation_indices = dict()

        # Iterate over equations, assemble.
        for name, eq in self.equations.items():
            ad = eq.evaluate(self.dof_manager, state)
            # Append matrix and rhs
            mat.append(ad.jac)
            # Multiply by -1 to move to the rhs
            rhs.append(-ad.val)

            # create indices range and shift to correct position
            block_indices = np.arange(len(ad.val)) + ind_start
            # extract last index as starting point for next block of indices
            ind_start = block_indices[-1]
            self.assembled_equation_indices.update({name: block_indices})

        # The system assembled in the for-loop above contains derivatives for both
        # primary and secondary variables, where the primary is understood as the
        # complement of the secondary ones. Columns relating to secondary variables
        # should therefore be removed. Construct a projection matrix onto the set
        # of primary variables and right multiply the Jacobian matrix.
        primary_variables = self.dof_manager.get_variables(True, False)
        projection = self.dof_manager.projection_to(primary_variables).transpose()

        # Concatenate matrix and remove columns of secondary variables
        A = sps.bmat([[m] for m in mat], format="csr") * projection

        return A, np.concatenate(rhs)

    def assemble_subsystem(
        self,
        eq_names: Optional[Sequence[str]] = None,
        variables: Optional[
            Sequence[Union[pp.ad.Variable, pp.ad.MergedVariable]]
        ] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector using a specified subset of
        equations and variables.

        The method is intended for use in splitting algorithms. Matrix blocks not
        included will simply be sliced out.

        Notes:
            The ordering of columns in the returned system are defined by the global DOF order
            provided by the DofManager. The rows are of the same order as equations were added
            to this system.

        Parameters:
            eq_names (optional): a subset of known equation names to be assembled.
                If not provided (None), all equations known to this manager will be included.
                Defaults to None.
            variables (optional): Variables for which the columns should to be returned.
                If not provided (None), all columns will be returned.

        Returns:
            spmatrix: (Part of the) Jacobian matrix corresponding to the current variable
                state, as found in grid dictionaries, for the specified equations and columns.
            ndarray: Residual vector corresponding to the current variable state,
                as found in the grid dictionaries, for the specified equations.
                Scaled with -1 (moved to rhs).

        """
        if variables:
            var_names = self.get_var_names_from(variables)
        else:
            var_names = self.dof_manager.get_variables(True, True)

        if eq_names is None:
            eq_names = list(self.equations.keys())

        # Data structures for building matrix and residual vector
        mat: list[sps.spmatrix] = []
        rhs: list[np.ndarray] = []

        # Projection to the subset of active variables
        projection = self.dof_manager.projection_to(var_names).transpose()

        # Keep track of DOFs for each equation/block
        ind_start = 0
        self.assembled_equation_indices = dict()

        # Iterate over equations, assemble.
        for name in eq_names:
            eq = self.equations[name]
            ad = eq.evaluate(self.dof_manager)
            # append matrices and rhs
            mat.append(ad.jac)
            # Multiply by -1 to move to the rhs
            rhs.append(-ad.val)

            # create indices range and shift to correct position
            block_indices = np.arange(len(ad.val)) + ind_start
            # extract last index as starting point for next block of indices
            ind_start = block_indices[-1]
            self.assembled_equation_indices.update({name: block_indices})

        # Concatenate results.
        if len(mat) > 0:
            A = sps.bmat([[m] for m in mat], format="csr")
            rhs_cat = np.concatenate(rhs)
        else:
            # Special case if the restriction produced an empty system.
            A = sps.csr_matrix((0, 0))
            rhs_cat = np.empty(0)

        # slice out the columns belonging to the requested variables and return the results
        return A * projection, rhs_cat

    def assemble_schur_complement_system(
        self,
        primary_equations: Sequence[str],
        primary_variables: Sequence[Union[pp.ad.Variable, pp.ad.MergedVariable]],
        inverter: Callable[[sps.spmatrix], sps.spmatrix],
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector using a Schur complement
        elimination of the variables and equations not to be included.

        The specified equations and variables will define a reordering of the linearized
        system into

            [A_pp, A_ps  [x_p   = [b_p
             A_sp, A_ss]  x_s]     b_s]

        Where subscripts p and s define primary and secondary quantities. The Schur
        complement system is then given by

            (A_pp - A_ps * inv(A_ss) * A_sp) * x_p = b_p - A_ps * inv(A_pp) * b_s.

        The Schur complement is well-defined only if the inverse of A_ss exists,
        and the efficiency of the approach assumes that an efficient inverter for
        A_ss can be found. The user must ensure both requirements are fulfilled.
        The simplest option is a lambda function on the form:

        .. code:: Python

            inverter = lambda A: sps.csr_matrix(np.linalg.inv(A.A))

        but depending on A (size and sparsity pattern), this can be costly in terms of
        computational time and memory.

        The method can be used e.g. for splitting between primary and secondary variables,
        where the latter can be efficiently eliminated (for instance, they contain no
        spatial derivatives).

        Parameters:
            primary_equations: equations to be assembled, representing the row-block A_pp.
                Should have length > 0.
            primary_variables: Variables representing the columns of A_pp.
                Should have length > 0.
            inverter (Callable): Method to compute the inverse of the matrix A_ss.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the current variable state,
                as found in grid dictionaries for the specified equations and variables.
            np.ndarray: Residual vector corresponding to the current variable state,
                as found in grid dictionaries, for the specified equations and variables.
                Scaled with -1 (moved to rhs).

        """
        if len(primary_equations) == 0:
            raise ValueError("Must make Schur complement with at least one equation")
        if len(primary_variables) == 0:
            raise ValueError("Must make Schur complement with at least one variable")

        # Unravel any merged variables
        primary_var_names = self.get_var_names_from(primary_variables)

        # Get lists of all variables and equations, and find the secondary items
        # by a set difference
        all_eq_names = list(self.equations.keys())
        all_var_names = self.dof_manager.get_variables(True, True)

        secondary_equations = list(set(all_eq_names).difference(set(primary_equations)))
        secondary_var_names = list(
            set(all_var_names).difference(set(primary_var_names))
        )

        # First assemble the primary and secondary equations for all variables
        # Note the reverse order here: Assemble the primary variables last so that
        # the attribute assembled_equation_indices contains the right ones.
        A_s, b_s = self.assemble_subsystem(secondary_equations, all_var_names)
        A_p, b_p = self.assemble_subsystem(primary_equations, all_var_names)

        # Projection matrices to reduce matrices to the relevant columns
        proj_primary = self.dof_manager.projection_to(primary_var_names).transpose()
        proj_secondary = self.dof_manager.projection_to(secondary_var_names).transpose()

        # Matrices involved in the Schur complements
        A_pp = A_p * proj_primary
        A_ps = A_p * proj_secondary
        A_sp = A_s * proj_primary
        A_ss = A_s * proj_secondary

        # Explicitly compute the inverse.
        # Depending on the matrix, and the inverter, this can take a long time.
        inv_A_ss = inverter(A_ss)

        S = A_pp - A_ps * inv_A_ss * A_sp
        bs = b_p - A_ps * inv_A_ss * b_s

        return S, bs

    def create_subsystem_manager(self, eq_names: Sequence[str]) -> ADSystemManager:
        """Creates an ``ADSystemManager`` for a given subset of equations.

        Parameters:
            eq_names: equations to be assigned to the new manager.
                Must be known to this manager.

        Returns:
            a new instance of ``ADSystemManager``. The subsystem equations are ordered as
            imposed by this manager's order.

        """
        # creating a new manager and adding the requested equations
        new_manger = ADSystemManager(self.dof_manager)
        for name in eq_names:
            new_manger.set_equation(name, self.equations[name])

        return new_manger

    ### special methods -----------------------------------------------------------------------

    def __repr__(self) -> str:
        s = (
            "AD System manager for mixed-dimensional grid with "
            f"{self.dof_manager.mdg.num_subdomains()} subdomains "
            f"and {self.dof_manager.mdg.num_interfaces()}"
            " interfaces.\n"
        )
        # Sort variables alphabetically, not case-sensitive
        all_vars = self.dof_manager.get_variables(True, True)
        secondary_vars = self.dof_manager.get_variables(False, True)
        s += "Variables present on at least one grid or interface:\n\t"
        s += ", ".join(all_vars) + "\n"

        if self.equations is not None:
            eq_names = [name for name in self.equations]
            s += f"In total {len(self.equations)} equations, with names: \n\t"
            s += ", ".join(eq_names)

        if len(secondary_vars) > 0:
            s += "\n"
            s += f"In total {len(secondary_vars)} secondary variables."

        return s

    def __str__(self) -> str:
        s = (
            "AD System manager for mixed-dimensional grid with "
            f"{self.dof_manager.mdg.num_subdomains()} subdomains "
            f"and {self.dof_manager.mdg.num_interfaces()}"
            " interfaces.\n"
        )

        all_vars = self.dof_manager.get_variables(True, True)
        var_grid = [
            (sub_var.name, sub_var.grid)
            for var in self.variables.values()
            for sub_var in var.sub_vars
        ]
        # make combinations unique
        var_grid = set(var_grid)

        s += f"There are in total {len(all_vars)} variables, distributed as follows:\n"

        # Sort variables alphabetically, not case-sensitive
        for var, grid in var_grid:
            s += "\t" + f"{var} is present on"
            s += " subdomain" if isinstance(grid, pp.Grid) else " interface(s)"
            s += "\n"

        s += "\n"
        if self.equations is not None:
            eq_names = [name for name in self.equations]
            s += f"In total {len(self.equations)} equations, with names: \n\t"
            s += "\n\t".join(eq_names) + "\n"

        return s
