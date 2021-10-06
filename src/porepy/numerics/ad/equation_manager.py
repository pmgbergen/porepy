""" Main content:
EquationManager: representation of a set of equations on Ad form.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from . import _ad_utils, operators

__all__ = ["EquationManager"]

GridLike = Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]


class EquationManager:
    """Representation of a set of equations specified on Ad form.

    The equations are tied to a specific GridBucket, with variables fixed in a
    corresponding DofManager.

    Attributes:
        gb (pp.GridBucket): Mixed-dimensional grid on which this EquationManager
            operates.
        dof_manager (pp.DofManager): Degree of freedom manager used for this
            EquationManager.
        equations (List of Expressions): Equations assigned to this EquationManager.
            can be expanded by direct addition to the list.
        variables (Dict): Mapping from grids or grid tuples (interfaces) to Ad
            variables. These are set at initialization from the GridBucket, and should
            not be changed later.
        secondary_variables (List of Ad Variables): List of variables that are secondary,
            that is, their derivatives will not be included in the Jacobian matrix.
            Variables will be represented on atomic form, that is, merged variables are
            unravelled. Secondary variables act as a filter during assembly, that is,
            they do not impact the ordering or treatment of variables.

    """

    def __init__(
        self,
        gb: pp.GridBucket,
        dof_manager: pp.DofManager,
        equations: Optional[Dict[str, "pp.ad.Operator"]] = None,
        secondary_variables: Optional[Sequence["pp.ad.Variable"]] = None,
    ) -> None:
        """Initialize the EquationManager.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid for this EquationManager.
            dof_manager (pp.DofManager): Degree of freedom manager.
            equations (List, Optional): List of equations. Defaults to empty list.
            secondary_variables (List of Ad Variable or MergedVariable): Variables
                to be considered secondary for this EquationManager.

        """
        self.gb = gb

        # Inform mypy about variables, and then set them by a dedicated method.
        self.variables: Dict[GridLike, Dict[str, "pp.ad.Variable"]]
        self._set_variables(gb)

        if equations is None:
            self.equations: Dict[str, pp.ad.Operator] = {}
        else:
            self.equations = equations

        self.dof_manager: pp.DofManager = dof_manager

        if secondary_variables is None:
            secondary_variables = []

        # Unravel any MergedVariable and store as a list.
        # Note that secondary variables will be present in self.variables; the exclusion
        # of secondary variables happens in assembly methods.
        self.secondary_variables: List[pp.ad.Variable] = self._variables_as_list(
            secondary_variables
        )

    def _set_variables(self, gb):
        # Define variables as specified in the GridBucket
        variables = {}
        for g, d in gb:
            variables[g] = {}
            for var, info in d[pp.PRIMARY_VARIABLES].items():
                variables[g][var] = operators.Variable(var, info, grids=[g])

        for e, d in gb.edges():
            variables[e] = {}
            num_cells = d["mortar_grid"].num_cells
            for var, info in d[pp.PRIMARY_VARIABLES].items():
                variables[e][var] = operators.Variable(
                    var, info, edges=[e], num_cells=num_cells
                )

        self.variables = variables

    def merge_variables(
        self, grid_var: Sequence[Tuple[GridLike, str]]
    ) -> "pp.ad.MergedVariable":
        """Concatenate a variable defined over several grids or interfaces between grids,
        that is a mortar grid.

        The merged variable can be used to define mathematical operations on multiple
        grids simultaneously (provided it is combined with other operators defined on
        the same grids).

        NOTE: Merged variables are assigned unique ids (see documentation of
        Variable and MergedVariable), thus two MergedVariables will have different
        ids even if they represent the same combination of grids and variables.
        This does not impact the parsing of the variables into numerical values.

        Returns:
            pp.ad.MergedVariable: Joint representation of the variable on the specified
                grids.

        """
        return pp.ad.MergedVariable([self.variables[g][v] for g, v in grid_var])

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
            ind: np.ndarray = self.dof_manager.dof_ind(*item)
            values.append(state[ind])

        return values

    def assemble(
        self,
        state: Optional[np.ndarray] = None,
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector with respect to the current
        state represented in self.gb.

        Derivatives for secondary variables are not included in the Jacobian matrix.

        Parameters:
            state (np.ndarray, optional): State vector to assemble from. If not provided,
                the default behavior of pp.ad.Expression.to_ad() will be followed.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the current variable state,
                as found in self.gb. The ordering of the equations is determined by
                the ordering in self.equations (for rows) and self.dof_manager (for
                columns).
            np.ndarray: Residual vector corresponding to the current variable state,
                as found in self.gb. Scaled with -1 (moved to rhs).

        """
        # Data structures for building the Jacobian matrix and residual vector
        mat: List[sps.spmatrix] = []
        b: List[np.ndarray] = []

        # Iterate over equations, assemble.
        for eq in self.equations.values():
            ad = eq.evaluate(self.dof_manager, state)
            # Append matrix and rhs
            mat.append(ad.jac)
            # Multiply by -1 to move to the rhs
            b.append(-ad.val)

        # Define secondary variables as the complement of the primary ones
        # This operation we do on atomic variables (not merged)
        primary_variables = self._variable_set_complement(
            self._variables_as_list(self.secondary_variables)
        )
        proj = self._column_projection(primary_variables)

        A = sps.bmat([[m] for m in mat]).tocsr() * proj
        rhs = np.hstack([vec for vec in b])

        return A, rhs

    def assemble_subsystem(
        self,
        eq_names: Optional[Sequence[str]] = None,
        variables: Optional[
            Sequence[Union["pp.ad.Variable", "pp.ad.MergedVariable"]]
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
                If not provided (None), all variabels known to this EquationManager will be
                included.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the current variable state,
                as found in self.gb, for the specified equations and variables.
            np.ndarray: Residual vector corresponding to the current variable state,
                as found in self.gb, for the specified equations and variables.
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
        b: List[np.ndarray] = []

        # Projection to the subset of active variables
        projection = self._column_projection(variables)

        # Iterate over equations, assemble.
        for name in eq_names:
            eq = self.equations[name]
            ad = eq.evaluate(self.dof_manager)

            # ad contains derivatives with respect to all variables, while
            # we need a subset. Project the columns to get the right size.
            mat.append(ad.jac * projection)

            # The residuals can be stored without reordering.
            # Multiply by -1 to move to the rhs
            b.append(-ad.val)

        # Concatenate results. Return
        if len(mat) > 0:
            A = sps.bmat([[m] for m in mat]).tocsr()
            rhs = np.hstack([vec for vec in b])
        else:
            A = sps.csr_matrix((0, 0))
            rhs = np.empty(0)
        return A, rhs

    def assemble_schur_complement_system(
        self,
        primary_equations: Sequence[str],
        primary_variables: Sequence[Union["pp.ad.Variable", "pp.ad.MergedVariable"]],
        inverter: Callable[[sps.spmatrix], sps.spmatrix],
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        """Assemble Jacobian matrix and residual vector using a Schur complement
        elimination of the variables and equations not to be included.

        The specified equations and variables will define a reordering of the linearized
        system into

            [J_pp, J_ps  [x_p   = [b_p
             J_sp, J_ss]  x_s]     b_s]

        Where subscripts p and s define primary and secondary quantities. The Schur
        complement system is then given by

            (J_pp - J_ps * inv(J_ss) * J_sp) * x_p = b_p - J_ps * inv(J_pp) * b_s.

        The Schur complement is well defined only if the inverse of J_ss exists,
        and the efficiency of the approach assumes that an efficient inverter for
        J_ss can be found. The user must ensure both requirements are fulfilled.
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
            inverter (Callable): Method to compute the inverse of the matrix J_ss.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the current variable state,
                as found in self.gb, for the specified equations and variables.
            np.ndarray: Residual vector corresponding to the current variable state,
                as found in self.gb, for the specified equations and variables.
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
        A_p, b_p = self.assemble_subsystem(primary_equations, all_variables)
        A_s, b_s = self.assemble_subsystem(secondary_equations, all_variables)

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

    def extract_subsystem(
        self,
        eq_names: Sequence[str],
        variables: Sequence[Union["pp.ad.Variable", "pp.ad.MergedVariable"]],
    ) -> "EquationManager":
        """Extract an EquationManager for a subset of variables and equations.
        In effect, this produce a nonlinear subsystem.

        Parameters:
            eq_names (Sequence of str): Equations assigned to the new EquationManager, specified
                as keys in self.equations.
            variables (Sequence of Variables): Variables for which the new EquationManager is defined.

        Returns:
            EquationManager: System of nonlinear equations. The ordering of the
                equations in the subsystem will be the same as in the original
                set (disregarding equations not included in the subset). Variables
                that were excluded are added to the set of secondary_variables in
                the new EquationManager.

        """
        secondary_variables = self._variable_set_complement(variables)

        sub_eqs = {name: self.equations[name] for name in eq_names}
        return EquationManager(
            gb=self.gb,
            equations=sub_eqs,
            dof_manager=self.dof_manager,
            secondary_variables=secondary_variables,
        )

    def discretize(self, gb: pp.GridBucket) -> None:
        """Loop over all discretizations in self.equations, find all unique discretizations
        and discretize.

        This is more effecient than discretizing on the Operator level, since
        discretizations which occur more than once in a set of equations will be
        identified and only discretized once.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid from which parameters etc. will
                be taken and where discretization matrices will be stored.

        """
        # Somehow loop over all equations, discretize identified objects
        # (but should also be able to do rediscretization based on
        # dependency graph etc).

        # List of discretizations, build up by iterations over all equations
        discr: List = []
        for eqn in self.equations.values():
            # This will expand the list discr with new discretizations.
            # The list may contain duplicates.
            discr = eqn._identify_subtree_discretizations(discr)

        # Uniquify to save computational time, then discretize.
        unique_discr = _ad_utils.uniquify_discretization_list(discr)
        _ad_utils.discretize_from_list(unique_discr, gb)

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
            inds.append(self.dof_manager.dof_ind(v._g, v._name))

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
        variables: Sequence[Union["pp.ad.Variable", "pp.ad.MergedVariable"]] = None,
    ) -> List["pp.ad.Variable"]:
        # Take the complement of a set of variables, with respect to the full set of
        # variables. The variables are returned as atomic (merged variables are
        # unravelled as part of the process).

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
            Sequence[Union["pp.ad.Variable", "pp.ad.MergedVariable"]]
        ] = None,
    ) -> List["pp.ad.Variable"]:
        # Get a list of all variables known to this EquationManager
        # This is a bit cumbersome, since the variables are stored as a
        # mapping from individual GridLike to an innermapping between variable
        # names and actual variables. To top it off, this variable can be merged,
        # and we need the atomic variables.

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
            if isinstance(v, (pp.ad.MergedVariable, operators.MergedVariable)):
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
            f"{self.gb.num_graph_nodes()} grids and {self.gb.num_graph_edges()}"
            " interfaces.\n"
        )

        var = []
        for g, _ in self.gb:
            for v in self.variables[g]:
                var.append(v)

        unique_vars = list(set(var))
        s += "Variables present on at least one grid or interface:\n\t"
        s += ", ".join(unique_vars) + "\n"

        if self.equations is not None:
            eq_names = [eq._name for eq in self.equations.values()]
            s += f"In total {len(self.equations)} equations, with names: \n\t"
            s += ", ".join(eq_names)

        if len(self.secondary_variables) > 0:
            s += "\n"
            s += f"In total {len(self.secondary_variables)} secondary variables."

        return s
