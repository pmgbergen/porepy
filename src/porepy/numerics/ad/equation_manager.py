"""
* Resue assembly when relevant (if no operator that maps to a specific block has been changed)
* Concatenate equations with the same sequence of operators
  - Should use the same discretization object
  - divergence operators on different grids considered the same
* Concatenated variables will share ad derivatives. However, it should be possible to combine
  subsets of variables with other variables (outside the set) to assemble different terms
*
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from . import _ad_utils, operators

__all__ = ["EquationManager"]

GridLike = Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]


class EquationManager:
    """Representation of a set of equations specified on Ad form.

    The equations are tied to a specific GridBucket, with variables fixed in a
    corresponding DofManager. Both these are set on initialization, and should
    not be modified later.

    Central methods are:
        discretize(): Discretize all operators identified in the set equations.
        assemble_matrix_rhs(): Provide a Jacobian matrix and residual for the
            current state in the GridBucket.

    TODO: Add functionality to derive subset of equations, fit for splitting
    algorithms.

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

    """

    def __init__(
        self,
        gb: pp.GridBucket,
        dof_manager: pp.DofManager,
        equations: Optional[List] = None,
    ) -> None:
        """Initialize the EquationManager.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid for this EquationManager.
            dof_manager (pp.DofManager): Degree of freedom manager.
            equations (List, Optional): List of equations. Defaults to empty list.

        """
        self.gb = gb

        # Inform mypy about variables, and then set them by a dedicated method.
        self.variables: Dict[GridLike, Dict[str, "pp.ad.Variable"]]
        self._set_variables(gb)

        if equations is None:
            self.equations: List = []
        else:
            self.equations = equations

        self.dof_manager: pp.DofManager = dof_manager

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

    def assemble_matrix_rhs(
        self,
        state: Optional[np.ndarray] = None,
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        """Assemble residual vector and Jacobian matrix with respect to the current
        state represented in self.gb.

        As an experimental feature, subset of variables and equations can also be
        assembled. This functionality may be moved somewhere else in the future.

        Parameters:
            state (np.ndarray, optional): State vector to assemble from. If not provided,
                the default behavior of pp.ad.Expression.to_ad() will be followed.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the current variable state,
                as found in self.gb.
            np.ndarray: Residual vector corresponding to the current variable state,
                as found in self.gb.

        """
        mat: List[sps.spmatrix] = []
        b: List[np.ndarray] = []

        for eq in self.equations:
            ad = eq.evaluate(self.dof_manager, state)

            # EK: Comment out this part for now; we may need something like this
            # when we get around to implementing subsystems.
            # The columns of the Jacobian has the size of the local variables.
            # Map these to the global ones
            # local_dofs = eq.local_dofs(true_ad_variables=variables)
            # if variables is not None:
            #    local_dofs = self.dof_manager.transform_dofs(local_dofs, var=names)

            # num_local_dofs = local_dofs.size
            # projection = sps.coo_matrix(
            #    (np.ones(num_local_dofs), (np.arange(num_local_dofs), local_dofs)),
            #    shape=(num_local_dofs, num_global_dofs),
            # )
            # mat.append(ad.jac * projection)
            mat.append(ad.jac)
            # Concatenate the residuals
            # Multiply by -1 to move to the rhs
            b.append(-ad.val)

        A = sps.bmat([[m] for m in mat]).tocsr()
        rhs = np.hstack([vec for vec in b])
        return A, rhs

    def discretize(self, gb: pp.GridBucket) -> None:
        """Loop over all discretizations in self.equations, find all unique discretizations
        and discretize.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid from which parameters etc. will
                be taken.

        """
        # Somehow loop over all equations, discretize identified objects
        # (but should also be able to do rediscretization based on
        # dependency graph etc).

        # List of discretizations, build up by iterations over all equations
        discr: List = []
        for eqn in self.equations:
            # This will expand the list discr with new discretizations.
            # The list may contain duplicates.
            discr = eqn._identify_subtree_discretizations(discr)

        # Uniquify to save computational time, then discretize.
        unique_discr = _ad_utils.uniquify_discretization_list(discr)
        _ad_utils.discretize_from_list(unique_discr, gb)

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
            eq_names = [eq.name for eq in self.equations]
            s += f"In total {len(self.equations)} equations, with names: \n\t"
            s += ", ".join(eq_names)

        return s
