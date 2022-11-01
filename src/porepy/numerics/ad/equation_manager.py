""" Main content:
EquationManager: representation of a set of equations on Ad form.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from . import _ad_utils, operators

__all__ = ["EquationManager"]

GridLike = Union[pp.Grid, pp.MortarGrid]


class EquationManager:
    """Representation of a set of equations specified on Ad form.

    The equations are tied to a specific MixedDimensionalGrid, with variables fixed in a
    corresponding DofManager.

    Attributes:
        mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid on which this EquationManager
            operates.
        dof_manager (pp.DofManager): Degree of freedom manager used for this
            EquationManager.
        equations (List of Ad Operators): Equations assigned to this EquationManager.
            can be expanded by direct addition to the list.
        variables (Dict): Mapping from subdomains or grid tuples (interfaces) to Ad
            variables. These are set at initialization from the MixedDimensionalGrid, and
            should not be changed later.
        secondary_variables (List of Ad Variables): List of variables that are secondary,
            that is, their derivatives will not be included in the Jacobian matrix.
            Variables will be represented on atomic form, that is, merged variables are
            unravelled. Secondary variables act as a filter during assembly, that is,
            they do not impact the ordering or treatment of variables.
        row_block_indices_last_assembled (np.ndarray): Row indices for the start of blocks
            corresponding to different equations in the last assembled system. The last item
            in the array is the total number of rows, so that row indices for block i can
            be recovered by np.arange(row_bl..[i], row_bl..[i+1]). The user must relate the
            indices to equations (either in self.equations or the equation list given to the
            relevant assembly method). This information is intended for diagnostic usage.

    """

    def __init__(
        self,
        mdg: pp.MixedDimensionalGrid,
        dof_manager: pp.DofManager,
        equations: Optional[Dict[str, "pp.ad.Operator"]] = None,
        secondary_variables: Optional[Sequence["pp.ad.Variable"]] = None,
    ) -> None:
        """Initialize the EquationManager.

        Parameters:
            mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid for this EquationManager.
            dof_manager (pp.DofManager): Degree of freedom manager.
            equations (List, Optional): List of equations. Defaults to empty list.
            secondary_variables (List of Ad Variable or MergedVariable): Variables
                to be considered secondary for this EquationManager.

        """
        DeprecationWarning("The EquationManager will be replaced by SystemManager.")
        self.mdg = mdg

        # Inform mypy about variables, and then set them by a dedicated method.
        self.variables: Dict[GridLike, Dict[str, "pp.ad.Variable"]]
        self._set_variables(mdg)

        if equations is None:
            self.equations: Dict[str, pp.ad.Operator] = {}
        else:
            self.equations = equations

        self.dof_manager: pp.DofManager = dof_manager

        # Define secondary variables.
        # Note that secondary variables will be present in self.variables; the exclusion
        # of secondary variables happens in assembly methods.

        # This gets a bit technical: Since every EquationManager makes its own set of
        # variables with unique Ids, we should make sure the secondary variables are
        # taken with respect to the variable set of this EquationManager. This should be
        # okay for standard usage (although the user may abuse it, and suffer for doing so).
        # However, when the EquationManager is formed by extracting a subsystem,
        # the set of secondary variables are defined from the original EquationManager,
        # and this can create all sort of problems. Therefore, translate the secondary
        # variables into variables identified with this EquationManager.
        # IMPLEMENTATION NOTE: The code below has not been thoroughly tested on
        # mixed-dimensional subdomains.

        if secondary_variables is None:
            secondary_variables = []

        # Make a list of combinations of grids and names, this should form a unique
        # classification of variables
        secondary_grid_name = [
            (var._g, var._name) for var in self._variables_as_list(secondary_variables)
        ]
        # Do the same for all variables, but make this a map to the atomic (non-merged)
        # variable.
        primary_grid_name = {
            (var._g, var._name): var for var in self._variables_as_list()
        }

        # Data structure for secondary variables
        sec_var = []
        # Loop over all variables known to this EquationManager. If its grid-name
        # combination is identical to that of a secondary variable, we store it as a
        # secondary variable, with the representation known to this EquationManager
        for v in primary_grid_name:
            if v in secondary_grid_name:
                sec_var.append(primary_grid_name[v])

        self.secondary_variables = sec_var

        # Start index for blocks corresponding to rows of the different equations.
        # Defaults to None, will be overwritten by assembly methods.
        self.row_block_indices_last_assembled: Optional[np.ndarray] = None

    def _set_variables(self, mdg: pp.MixedDimensionalGrid):
        # Define variables as specified in the MixedDimensionalGrid
        variables: Dict[
            Union[pp.Grid, pp.MortarGrid], dict[str, operators.Variable]
        ] = dict()
        for sd, sd_data in mdg.subdomains(return_data=True):
            variables[sd] = {}
            for var, info in sd_data[pp.PRIMARY_VARIABLES].items():
                variables[sd][var] = operators.Variable(var, info, subdomains=[sd])

        for intf, intf_data in mdg.interfaces(return_data=True):
            variables[intf] = {}
            num_cells = intf.num_cells
            for var, info in intf_data[pp.PRIMARY_VARIABLES].items():
                variables[intf][var] = operators.Variable(var, info, interfaces=[intf])

        self.variables = variables

    def merge_variables(
        self, grid_var: Sequence[Tuple[GridLike, str]]
    ) -> "pp.ad.MixedDimensionalVariable":
        """Concatenate a variable defined over several subdomains or interfaces.



        The merged variable can be used to define mathematical operations on multiple
        subdomains simultaneously (provided it is combined with other operators defined on
        the same subdomains).

        NOTE: Merged variables are assigned unique ids (see documentation of
        Variable and MergedVariable), thus two MergedVariables will have different
        ids even if they represent the same combination of subdomains and variables.
        This does not impact the parsing of the variables into numerical values.

        Args:
            grid_var: Tuple containing first a (Mortar)grid representing the subdomain
                or interface and second the name of the variable.

        Returns:
            pp.ad.MergedVariable: Joint representation of the variable on the specified
                subdomains.

        """
        return pp.ad.MixedDimensionalVariable([self.variables[g][v] for g, v in grid_var])

    def variable(self, grid_like: GridLike, variable: str) -> "pp.ad.Variable":
        """Get a variable for a specified grid or interface between grids, that is
        a mortar grid.

        Subsequent calls of this method with the same grid and variable will return
        references to the same variable.

        Returns:
            pp.ad.Variable: Ad representation of a variable.

        """
        return self.variables[grid_like][variable]

    def variable_state(
        self, grid_var: List[Tuple[pp.Grid, str]], state: np.ndarray
    ) -> List[np.ndarray]:
        # This should likely be placed somewhere else
        values: List[np.ndarray] = []
        for item in grid_var:
            ind: np.ndarray = self.dof_manager.grid_and_variable_to_dofs(*item)
            values.append(state[ind])

        return values

    def assemble(
        self,
        state: Optional[np.ndarray] = None,
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector with respect to the current
        state represented in self.mdg.

        Derivatives for secondary variables are not included in the Jacobian matrix.

        Parameters:
            state (np.ndarray, optional): State vector to assemble from. If not provided,
                the default behavior of pp.ad.Expression.to_ad() will be followed.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the current variable state,
                as found in self.mdg. The ordering of the equations is determined by
                the ordering in self.equations (for rows) and self.dof_manager (for
                columns).
            np.ndarray: Residual vector corresponding to the current variable state,
                as found in self.mdg. Scaled with -1 (moved to rhs).

        """
        # Data structures for building the Jacobian matrix and residual vector
        mat: List[sps.spmatrix] = []
        rhs: List[np.ndarray] = []

        # Keep track of first row index for each equation/block
        ind_start: List[int] = [0]

        # Iterate over equations, assemble.
        for eq in self.equations.values():
            ad = eq.evaluate(self.dof_manager, state)
            # Append matrix and rhs
            mat.append(ad.jac)
            # Multiply by -1 to move to the rhs
            rhs.append(-ad.val)
            ind_start.append(ind_start[-1] + ad.val.size)

        # The system assembled in the for-loop above contains derivatives for both
        # primary and secondary variables, where the primary is understood as the
        # complement of the secondary ones. Columns relating to secondary variables
        # should therefore be removed. Construct a projection matrix onto the set
        # of primary variables and right multiply the Jacobian matrix.

        # Define primary variables as the complement of the secondary ones
        # This operation we do on atomic variables (not merged), or else there may
        # be problems for
        primary_variables = self._variable_set_complement(
            self._variables_as_list(self.secondary_variables)
        )
        proj = self._column_projection(primary_variables)

        # Concatenate matrix and remove columns of secondary variables
        A = sps.bmat([[m] for m in mat], format="csr") * proj

        # The right hand side vector. This should have contributions form both primary
        # and secondary variables, thus no need to modify it before concatenation.
        rhs_cat = np.hstack([vec for vec in rhs])

        # Store information on start of each block
        self.row_block_indices_last_assembled = np.array(ind_start)

        return A, rhs_cat

    def assemble_subsystem(
        self,
        eq_names: Optional[Sequence[str]] = None,
        variables: Optional[
            Sequence[Union["pp.ad.Variable", "pp.ad.MixedDimensionalVariable"]]
        ] = None,
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector using a specified subset of
        equations and variables.

        The method is intended for use in splitting algorithms. Matrix blocks not
        included will simply be ignored.

        Parameters:
            eq_names (Sequence of str, optional): Equations to be assembled, specified
                as keys in self.equations. If not provided (None), all equations known to
                this EquationManager will be included.
            variables (Sequence of Variables, optional): Variables to be assembled.
                If not provided (None), all variables known to this EquationManager will be
                included. If a secondary variable is specified, this will be included in
                the returned system.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the current variable state,
                as found in self.mdg, for the specified equations and variables.
            np.ndarray: Residual vector corresponding to the current variable state,
                as found in self.mdg, for the specified equations and variables.
                Scaled with -1 (moved to rhs).

            NOTE: The ordering of columns in the system are defined by the order of the
                variables specified in DofManager. For the rows, no corresponding global
                ordering of equations exists, and the rows will therefore be organized
                by the ordering in the parameter eq_names.

        """
        variables = self._variables_as_list(variables)

        if eq_names is None:
            eq_names = list(self.equations.keys())

        # Data structures for building matrix and residual vector
        mat: List[sps.spmatrix] = []
        rhs: List[np.ndarray] = []

        # Projection to the subset of active variables
        projection = self._column_projection(variables)

        ind_start = [0]

        # Iterate over equations, assemble.
        for name in eq_names:
            eq = self.equations[name]
            ad = eq.evaluate(self.dof_manager)

            # ad contains derivatives with respect to all variables, while
            # we need a subset. Project the columns to get the right size.
            mat.append(ad.jac * projection)

            # The residuals can be stored without reordering.
            # Multiply by -1 to move to the rhs
            rhs.append(-ad.val)

            ind_start.append(ind_start[-1] + ad.val.size)

        # Concatenate results.
        if len(mat) > 0:
            A = sps.bmat([[m] for m in mat], format="csr")
            rhs_cat = np.hstack([vec for vec in rhs])
        else:
            # Special case if the restriction produced an empty system.
            A = sps.csr_matrix((0, 0))
            rhs_cat = np.empty(0)

        # Store information on start of each block
        self.row_block_indices_last_assembled = np.array(ind_start)

        return A, rhs_cat

    def assemble_schur_complement_system(
        self,
        primary_equations: Sequence[str],
        primary_variables: Sequence[Union["pp.ad.Variable", "pp.ad.MixedDimensionalVariable"]],
        inverter: Callable[[sps.spmatrix], sps.spmatrix],
    ) -> Tuple[sps.spmatrix, np.ndarray]:
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

            inverter = lambda A: sps.csr_matrix(np.linalg.inv(A.A))

        but depending on A (size and sparsity pattern), this can be costly in terms of
        computational time and memory.

        The method can be used e.g. for splitting between primary and secondary variables,
        where the latter can be efficiently eliminated (for instance, they contain no
        spatial derivatives).

        Parameters:
            primary_equations (Sequence of str): Equations to be assembled, specified
                as keys in self.equations. Should have length > 0.
            primary_variables (Sequence of Variables): Variables to be assembled. Should have
                length > 0.
            inverter (Callable): Method to compute the inverse of the matrix A_ss.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the current variable state,
                as found in self.mdg, for the specified equations and variables.
            np.ndarray: Residual vector corresponding to the current variable state,
                as found in self.mdg, for the specified equations and variables.
                Scaled with -1 (moved to rhs).

        """
        if len(primary_equations) == 0:
            raise ValueError("Must take Schur complement with at least one equation")
        if len(primary_variables) == 0:
            raise ValueError("Must take Schur complement with at least one variable")

        # Unravel any merged variables
        primary_variables = self._variables_as_list(primary_variables)

        # Get lists of all variables and equations, and find the secondary items
        # by a set difference
        all_eq_names = list(self.equations.keys())
        all_variables = self._variables_as_list()

        secondary_equations = list(set(all_eq_names).difference(set(primary_equations)))
        secondary_variables = list(
            set(all_variables).difference(set(primary_variables))
        )

        # First assemble the primary and secondary equations for all variables
        # Note the reverse order here: Assemble the primary variables last so that
        # the attribute row_block_indices_last_assembled is set correctly.
        A_s, b_s = self.assemble_subsystem(secondary_equations, all_variables)
        A_p, b_p = self.assemble_subsystem(primary_equations, all_variables)

        # Projection matrices to reduce matrices to the relevant columns
        proj_primary = self._column_projection(primary_variables)
        proj_secondary = self._column_projection(secondary_variables)

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

    def subsystem_equation_manager(
        self,
        eq_names: Sequence[str],
        variables: Sequence[Union["pp.ad.Variable", "pp.ad.MixedDimensionalVariable"]],
    ) -> "EquationManager":
        """Extract an EquationManager for a subset of variables and equations.
        In effect, this produce a nonlinear subsystem.

        Parameters:
            eq_names (Sequence of str): Equations assigned to the new EquationManager,
                specified as keys in self.equations.
            variables (Sequence of Variables): Variables for which the new EquationManager is
                defined.

        Returns:
            EquationManager: System of nonlinear equations. The ordering of the
                equations in the subsystem will be the same as in the original
                EquationManager (i.e. self),  disregarding equations not included in the
                subset. Variables that were excluded are added to the set of
                secondary_variables in the new EquationManager.

        """
        secondary_variables = self._variable_set_complement(variables)

        sub_eqs = {name: self.equations[name] for name in eq_names}
        return EquationManager(
            mdg=self.mdg,
            equations=sub_eqs,
            dof_manager=self.dof_manager,
            secondary_variables=secondary_variables,
        )

    def discretize(self, mdg: pp.MixedDimensionalGrid) -> None:
        """Loop over all discretizations in self.equations, find all unique discretizations
        and discretize.

        This is more efficient than discretizing on the Operator level, since
        discretizations which occur more than once in a set of equations will be
        identified and only discretized once.

        Parameters:
            mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid from which parameters etc.
                will be taken and where discretization matrices will be stored.

        """
        # Somehow loop over all equations, discretize identified objects
        # (but should also be able to do rediscretization based on
        # dependency graph etc.).

        # List of discretizations, build up by iterations over all equations
        discr: List = []
        for eqn in self.equations.values():
            # This will expand the list discr with new discretizations.
            # The list may contain duplicates.
            discr = eqn._identify_subtree_discretizations(discr)

        # Uniquify to save computational time, then discretize.
        unique_discr = _ad_utils.uniquify_discretization_list(discr)
        _ad_utils.discretize_from_list(unique_discr, mdg)

    def _column_projection(self, variables: Sequence["pp.ad.Variable"]) -> sps.spmatrix:
        """Create a projection matrix from the full variable set to a subset.

        Parameters:
            variables (Sequence of pp.ad.Variable): Variables to be preserved in the
                projection. Should be atomic variables, *not* merged ones. The
                projection will preserve the ordering of the included variables, as
                defined in self.dof_manager.block_dof.

        Returns:
            sps.spmatrix: Projection matrix to be right multiplied with a Jacobian matrix to
                remove columns corresponding to variables not included.

        """
        num_global_dofs = self.dof_manager.full_dof.sum()

        # Array for the dofs associated with each grid-variable combination
        inds = []

        # Loop over variables, find dofs
        for v in variables:
            inds.append(self.dof_manager.grid_and_variable_to_dofs(v._g, v._name))

        if len(inds) == 0:
            # Special case if no indices were returned
            return sps.csr_matrix((num_global_dofs, 0))

        # Create projection matrix. Uniquify indices here, both to sort (will preserve
        # the ordering of the unknowns given by the DofManager) and remove duplicates
        # (in case variables were specified more than once).
        local_dofs = np.unique(np.hstack([i for i in inds]))
        num_local_dofs = local_dofs.size

        return sps.coo_matrix(
            (np.ones(num_local_dofs), (local_dofs, np.arange(num_local_dofs))),
            shape=(num_global_dofs, num_local_dofs),
        ).tocsr()

    def _variable_set_complement(
        self,
        variables: Sequence[Union["pp.ad.Variable", "pp.ad.MixedDimensionalVariable"]] = None,
    ) -> List["pp.ad.Variable"]:
        """
        Take the complement of a set of variables, with respect to the full set of
        variables. The variables are returned as atomic (merged variables are
        unravelled as part of the process).

        Parameters:
            variables (Sequence of pp.ad.Variable or pp.ad.MergedVariable, optional):
                Variables for which the complement should be taken. If not provided,
                all variables known to this EquationManager will be added, thus an
                empty list will be returned.

        Returns:
            List of pp.ad.Variable: Variables known to this EquationManager which were
            not present in the input list.

        """

        # Unravel any merged variable
        variables = self._variables_as_list(variables)
        # Get list of all variables
        all_variables = self._variables_as_list()

        # Do the complement
        other_variables = list(set(all_variables).difference(set(variables)))
        return other_variables

    def _variables_as_list(
        self,
        variables: Optional[
            Sequence[Union["pp.ad.Variable", "pp.ad.MixedDimensionalVariable"]]
        ] = None,
    ) -> List["pp.ad.Variable"]:
        """Unravel a list of variables into atomic (non-merged) variables.

        This is a bit cumbersome, since the variables are stored as a
        mapping from individual GridLike to an innermapping between variable
        names and actual variables. To top it off, this variable can be merged,
        and we need the atomic variables.

        Parameters:
            variables (Sequence of pp.ad.Variable or pp.ad.MergedVariable, optional):
                Variables to be unravelled. If not provided, all variables known to this
                EquationManager will be considered.

        Returns:
            List of pp.ad.Variable: Atomic form of all variables in the input list.

        """
        if variables is None:
            # First get a list of all variables (single or merged)
            tmp_vars = []
            # Loop over GridType-variable Dict
            for v_dict in self.variables.values():
                # loop over name-variable dict
                for v in v_dict.values():
                    # Append variable
                    tmp_vars.append(v)
        else:
            tmp_vars = list(variables)

        # List of all variables.
        var = []

        # Loop over all variables, add the variable itself, or its subvariables
        # (if it is Merged)
        for v in tmp_vars:
            if isinstance(v, pp.ad.MixedDimensionalVariable):
                for sv in v.sub_vars:
                    var.append(sv)
            elif isinstance(v, (pp.ad.Variable, operators.Variable)):
                var.append(v)
            else:
                raise ValueError("Encountered unknown type in variable list")

        return var

    def __repr__(self) -> str:
        s = (
            "Equation manager for mixed-dimensional grid with "
            f"{self.mdg.num_subdomains()} subdomains and {self.mdg.num_interfaces()}"
            " interfaces.\n"
        )

        var = []
        for sd in self.mdg.subdomains():
            for v in self.variables[sd]:
                var.append(v)

        for intf in self.mdg.interfaces():
            for v in self.variables[intf]:
                var.append(v)

        # Sort variables alphabetically, not case-sensitive
        unique_vars = sorted(list(set(var)), key=str.casefold)
        s += "Variables present on at least one grid or interface:\n\t"
        s += ", ".join(unique_vars) + "\n"

        if self.equations is not None:
            eq_names = [name for name in self.equations]
            s += f"In total {len(self.equations)} equations, with names: \n\t"
            s += ", ".join(eq_names)

        if len(self.secondary_variables) > 0:
            s += "\n"
            s += f"In total {len(self.secondary_variables)} secondary variables."

        return s

    def __str__(self) -> str:
        s = (
            "Equation manager for mixed-dimensional grid with "
            f"{self.mdg.num_subdomains()} subdomains and {self.mdg.num_interfaces()}"
            " interfaces.\n\n"
        )

        var: Dict = {}
        for sd in self.mdg.subdomains():
            for v in self.variables[sd]:
                if v not in var:
                    var[v] = []
                var[v].append(sd)

        for intf in self.mdg.interfaces():
            for v in self.variables[intf]:
                if v not in var:
                    var[v] = []
                var[v].append(intf)

        s += (
            f"There are in total {len(var)} variables, distributed as follows "
            "(sorted alphabetically):\n"
        )

        # Sort variables alphabetically, not case-sensitive
        for v in sorted(var, key=str.casefold):
            grids = var[v]
            s += "\t" + f"{v} is present on {len(grids)}"
            s += " subdomain(s)" if isinstance(grids[0], pp.Grid) else " interface(s)"
            s += "\n"

        if len(self.secondary_variables) > 0:
            s += "\n"
            # Leave a hint that any merged secondary variables have been split into subparts
            s += (
                f"In total {len(self.secondary_variables)} secondary variables"
                "(having split merged variables).\n"
                "Listing secondary variables:\n"
            )
            # Make a list of
            sec_names = [v._name for v in self.secondary_variables]
            for key, val in Counter(sec_names).items():
                s += "\t" + f"{key} occurs on {val} subdomains or interfaces" + "\n"

        s += "\n"
        if self.equations is not None:
            eq_names = [name for name in self.equations]
            s += f"In total {len(self.equations)} equations, with names: \n\t"
            s += "\n\t".join(eq_names) + "\n"

        return s

    def name_and_assign_equations(
        self, equation_dictionary: Dict[str, pp.ad.Operator]
    ) -> None:
        """Convenience method for assigning and naming equations.

        Parameters:
            equation_dictionary (Dict): Dictionary containing name: equation pairs.


        """
        for name, eq in equation_dictionary.items():
            eq.set_name(name)
            self.equations.update({name: eq})
