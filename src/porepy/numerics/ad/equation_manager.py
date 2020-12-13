"""
* Resue assembly when relevant (if no operator that maps to a specific block has been changed)
* Concatenate equations with the same sequence of operators
  - Should use the same discretization object
  - divergence operators on different grids considered the same
* Concatenated variables will share ad derivatives. However, it should be possible to combine
  subsets of variables with other variables (outside the set) to assemble different terms
*
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps

from .forward_mode import initAdArrays
from . import operators

import porepy as pp

__all__ = ["Equation", "EquationManager"]

grid_like_type = Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]


class Equation:
    """Ad representation of an equation.

    Conceptually, an Equation is an Operator tree that has been equated to zero.

    The equation has a fixed set of variables, identified from the operator tree.

    The residual and Jacobian matrix of an Equation can be evaluated via the function
    to_ad().

    Attributes:
        operator (Operator): Top operator in the operator tree.
        dof_manager (pp.DofManager): Degree of freedom manager associated with the
            mixed-dimensional GridBucket with which this equation is associated. Used
            to map between local (to the equation) and global variables.
        name (str): Name identifier of this variable.

    """

    def __init__(
        self, operator: operators.Operator, dof_manager: pp.DofManager, name: str = None
    ):
        """Define an Equation.

        Parameters:
            operator (pp.ad.Operator): Top-level operator of the Operator tree that will
                be equated to zero.
            dof_manager (pp.DofManager): Degree of freedom manager associated with the
                mixed-dimensional GridBucket with which this equation is associated.
            name (str): Name of the Eqution.

        """
        # The only non-trival operation in __init__ is the identification of variables.
        # Besides, some bookkeeping is necessary.

        # Black sometimes formats long equations with parantheses in a way that is
        # interpreted as a tuple by Python. Sigh.
        if (
            isinstance(operator, tuple)
            and len(operator) == 1
            and isinstance(operator[0], operators.Operator)
        ):
            operator = operator[0]

        self._operator = operator

        self.name = name

        variable_dofs, variable_ids = self._identify_variables(dof_manager)

        self._variable_dofs = variable_dofs
        self._variable_ids = variable_ids

        # Storage for matrices; will likely be removed (moved to the individual
        # operators).
        self._stored_matrices = {}

    def local_dofs(self) -> np.ndarray:
        dofs = np.hstack([d for d in self._variable_dofs])
        return dofs

    def __repr__(self) -> str:
        return f"Equation named {self.name}"

    def _find_subtree_variables(self, op: operators.Operator):
        """Method to recursively look for Variables (or MergedVariables) in an
        operator tree.
        """
        # The variables should be located at leaves in the tree. Traverse the tree
        # recursively, look for varibales, and then gather the results.

        if isinstance(op, operators.Variable) or isinstance(op, pp.ad.Variable):
            # We are at the bottom of the a branch of the tree, return the operator
            return op
        else:
            # We need to look deeper in the tree.
            # Look for variables among the children
            sub_variables = [
                self._find_subtree_variables(child) for child in op.tree.children
            ]
            # Some work is needed to parse the information
            var_list = []
            for var in sub_variables:
                if isinstance(var, operators.Variable) or isinstance(
                    var, pp.ad.Variable
                ):
                    # Effectively, this node is one step from the leaf
                    var_list.append(var)
                elif isinstance(var, list):
                    # We are further up in the tree.
                    for sub_var in var:
                        if isinstance(sub_var, operators.Variable) or isinstance(
                            sub_var, pp.ad.Variable
                        ):
                            var_list.append(sub_var)
            return var_list

    def _identify_variables(self, dof_manager):
        # NOTES TO SELF:
        # state: state vector for all unknowns. Should be possible to pick this
        # from pp.STATE or pp.ITERATE

        # 1. Get all variables present in this equation.
        # The variable finder is implemented in a special function, aimed at recursion
        # through the operator tree.
        # Uniquify by making this a set, and then sort on variable id
        variables = sorted(
            list(set(self._find_subtree_variables(self._operator))),
            key=lambda var: var.id,
        )

        # 2. Get a mapping between variables (*not* only MergedVariables) and their
        # indices according to the DofManager. This is needed to access the state of
        # a variable when parsing the equation to Ad format.

        # For each variable, get the global index
        inds = []
        variable_ids = []
        for variable in variables:
            # Indices (in DofManager sense) of this variable. Will be built gradually
            # for MergedVariables, in one go for plain Variables.
            ind_var = []

            if isinstance(variable, pp.ad.MergedVariable):
                # Loop over all subvariables for the merged variable
                for i, sub_var in enumerate(variable.sub_vars):
                    # Store dofs
                    ind_var.append(dof_manager.dof_ind(sub_var.g, sub_var._name))
                    if i == 0:
                        # Store id of variable, but only for the first one; we will
                        # concatenate the arrays in ind_var into one array
                        # Q: Why not use the id of variable here?
                        variable_ids.append(sub_var.id)
            else:
                # This is a variable that lives on a single grid
                ind_var.append(dof_manager.dof_ind(variable.g, variable._name))
                variable_ids.append(variable.id)

            # Gather all indices for this variable
            inds.append(np.hstack([i for i in ind_var]))

        return inds, variable_ids

    def to_ad(self, gb: pp.GridBucket, state: np.ndarray):
        """Evaluate the residual and Jacobian matrix for a given state.

        Parameters:
            gb (pp.GridBucket): GridBucket used to represent the problem. Will be used
                to parse the operators that combine to form this Equation..
            state (np.ndarray): State vector for which the residual and its derivative
                should be formed.

        Returns:
            An Ad-array representation of the residual and Jacbobian.

        """
        # Parsing in two stages: First make an Ad-representation of the variable state
        # (this must be done jointly for all variables of the Equation to get all
        # derivatives represented). Then parse the equation by traversing its
        # tree-representation, and parse and combine individual operators.

        # Initialize variables
        ad_vars = initAdArrays([state[ind] for ind in self._variable_dofs])
        self._ad = {var_id: ad for (var_id, ad) in zip(self._variable_ids, ad_vars)}

        # Parse operators. This is left to a separate function to facilitate the
        # necessary recursion for complex operators.
        eq = self._parse_operator(self._operator, gb)

        return eq

    def _parse_operator(self, op: operators.Operator, gb):
        """TODO: Currently, there is no prioritization between the operations; for
        some reason, things just work. We may need to make an ordering in which the
        operations should be carried out. It seems that the strategy of putting on
        hold until all children are processed works, but there likely are cases where
        this is not the case.
        """

        # The parsing strategy depends on the operator at hand:
        # 1) If the operator is a Variable, it will be represented according to its
        #    state.
        # 2) If the operator is a leaf in the tree-representation of the equation,
        #    parsing is left to the operator itself.
        # 3) If the operator is formed by combining other operators lower in the tree,
        #    parsing is handled by first evaluating the children (leads to recursion)
        #    and then perform the operation on the result.

        # Check for case 1 or 2
        if isinstance(op, pp.ad.Variable) or isinstance(op, operators.Variable):
            # Case 1: Variable
            # How to access the array of (Ad representation of) states depends on wether
            # this is a single or combined variable; see self.__init__, definition of
            # self._variable_ids.
            if isinstance(op, pp.ad.MergedVariable) or isinstance(
                op, operators.MergedVariable
            ):
                return self._ad[op.sub_vars[0].id]
            else:
                return self._ad[op.id]
        elif op.is_leaf():
            # Case 2
            return op.parse(gb)

        # This is not an atomic operator. First parse its children, then combine them
        tree = op.tree
        results = [self._parse_operator(child, gb) for child in tree.children]

        # Combine the results
        if tree.op == operators.Operation.add:
            # To add we need two objects
            assert len(results) == 2
            return results[0] + results[1]

        elif tree.op == operators.Operation.sub:
            # To subtract we need two objects
            assert len(results) == 2
            return results[0] - results[1]

        elif tree.op == operators.Operation.mul:
            # To multiply we need two objects
            assert len(results) == 2
            return results[0] * results[1]

        elif tree.op == operators.Operation.evaluate:
            # This is a function, which should have at least one argument
            assert len(results) > 1
            return results[0].func(results[1:])

        elif tree.op == operators.Operation.div:
            return results[0] / results[1]

        else:
            raise ValueError("Should not happen")


class EquationManager:
    # The usage of this class is not yet clear, and will likely undergo substantial
    # enhancements and changes.
    def __init__(
        self,
        gb: pp.GridBucket,
        dof_manager: pp.DofManager,
        equations: Optional[List[Equation]] = None,
    ) -> None:
        self.gb = gb
        self._set_variables(gb)

        if equations is None:
            self._equations = []
        else:
            self._equations = equations
            # Separate a dof-manager from assembler?
        self.dof_manager = dof_manager

    def _set_variables(self, gb):
        # Define variables as specified in the GridBucket
        variables = {}
        for g, d in gb:
            variables[g] = {}
            for var, info in d[pp.PRIMARY_VARIABLES].items():
                variables[g][var] = operators.Variable(var, info, g)

        for e, d in gb.edges():
            variables[e] = {}
            num_cells = d["mortar_grid"].num_cells
            for var, info in d[pp.PRIMARY_VARIABLES].items():
                variables[e][var] = operators.Variable(var, info, e, num_cells)

        self.variables = variables
        # Define discretizations

    def merge_variables(self, grid_var: List[Tuple[grid_like_type, str]]):
        return pp.ad.MergedVariable([self.variables[g][v] for g, v in grid_var])

    def variable_state(
        self, grid_var: List[Tuple[pp.Grid, str]], state: np.ndarray
    ) -> List[np.ndarray]:
        # This should likely be placed somewhere else
        values: List[np.ndarray] = []
        for item in grid_var:
            ind: np.ndarray = self.dof_manager.dof_ind(*item)
            values.append(state[ind])

        return values

    def assemble_matrix_rhs(self, state):
        mat: List[sps.spmatrix] = []
        b: List[np.ndarray] = []

        num_global_dofs = self.dof_manager.full_dof.sum()

        for eq in self._equations:
            ad = eq.to_ad(self.gb, state)

            # The columns of the Jacobian has the size of the local variables.
            # Map these to the global ones
            local_dofs = eq.local_dofs()
            num_local_dofs = local_dofs.size
            projection = sps.coo_matrix(
                (np.ones(num_local_dofs), (np.arange(num_local_dofs), local_dofs)),
                shape=(num_local_dofs, num_global_dofs),
            )

            mat.append(ad.jac * projection)

            # Concatenate the residuals
            # Multiply by -1 to move to the rhs
            b.append(-ad.val)

        A = sps.bmat([[m] for m in mat]).tocsr()
        rhs = np.hstack([vec for vec in b])
        return A, rhs

    def discretize(self):
        # Somehow loop over all equations, discretize identified objects
        # (but should also be able to do rediscretization based on
        # dependency graph etc).
        pass
