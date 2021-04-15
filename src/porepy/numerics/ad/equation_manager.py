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

from . import operators
from .discretizations import _MergedOperator
from .forward_mode import Ad_array, initAdArrays
from .local_forward_mode import Local_Ad_array

__all__ = ["Expression", "EquationManager"]

grid_like_type = Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]


class Expression:
    """Ad representation of an expression which can be evaluated (translated to
    numerical values).

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
        self,
        operator: operators.Operator,
        dof_manager: pp.DofManager,
        name: str = "",
        grid_order: Optional[Sequence[Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]]] = None,
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
        self._dof_manager = dof_manager

        self.name = name
        self.grid_order = grid_order

        # Identify all variables in the Operator tree. This will include real variables,
        # and representation of previous time steps and iterations.
        (
            variable_dofs,
            variable_ids,
            is_prev_time,
            is_prev_iter,
        ) = self._identify_variables(dof_manager)

        # Split variable dof indices and ids into groups of current variables (those
        # of the current iteration step), and those from the previous time steps and
        # iterations.
        current_indices = []
        current_ids = []
        prev_indices = []
        prev_ids = []
        prev_iter_indices = []
        prev_iter_ids = []
        for ind, var_id, is_prev, is_prev_it in zip(
            variable_dofs, variable_ids, is_prev_time, is_prev_iter
        ):
            if is_prev:
                prev_indices.append(ind)
                prev_ids.append(var_id)
            elif is_prev_it:
                prev_iter_indices.append(ind)
                prev_iter_ids.append(var_id)
            else:
                current_indices.append(ind)
                current_ids.append(var_id)

        # Save information.
        self._variable_dofs = current_indices
        self._variable_ids = current_ids
        self._prev_time_dofs = prev_indices
        self._prev_time_ids = prev_ids
        self._prev_iter_dofs = prev_iter_indices
        self._prev_iter_ids = prev_iter_ids

        self._identify_discretizations()

    # TODO why is this called local? and not global?
    def local_dofs(self, true_ad_variables: Optional[list] = None) -> np.ndarray:
        if true_ad_variables is None:
            dofs = np.hstack([d for d in self._variable_dofs])
        else:
            true_ad_variable_ids = [v.id for v in true_ad_variables]
            assert all([i in self._variable_ids for i in true_ad_variable_ids])
            ad_variable_local_ids = [
                self._variable_ids.index(i) for i in true_ad_variable_ids
            ]
            ad_variable_dofs = [self._variable_dofs[i] for i in ad_variable_local_ids]
            dofs = np.hstack([d for d in ad_variable_dofs])
        return dofs

    def __repr__(self) -> str:
        return f"Equation named {self.name}"

    def _find_subtree_variables(
        self, op: operators.Operator
    ) -> List[operators.Variable]:
        """Method to recursively look for Variables (or MergedVariables) in an
        operator tree.
        """
        # The variables should be located at leaves in the tree. Traverse the tree
        # recursively, look for variables, and then gather the results.

        if isinstance(op, operators.Variable) or isinstance(op, pp.ad.Variable):
            # We are at the bottom of the a branch of the tree, return the operator
            return [op]
        else:
            # We need to look deeper in the tree.
            # Look for variables among the children
            sub_variables = [
                self._find_subtree_variables(child) for child in op.tree.children
            ]
            # Some work is needed to parse the information
            var_list: List[operators.Variable] = []
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

    def _identify_variables(self, dof_manager, var: Optional[list] = None):
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
        prev_time = []
        prev_iter = []
        for variable in variables:
            # Indices (in DofManager sense) of this variable. Will be built gradually
            # for MergedVariables, in one go for plain Variables.
            ind_var = []
            prev_time.append(variable.prev_time)
            prev_iter.append(variable.prev_iter)

            if isinstance(variable, (pp.ad.MergedVariable, operators.MergedVariable)):
                # Loop over all subvariables for the merged variable
                for i, sub_var in enumerate(variable.sub_vars):
                    # Store dofs
                    ind_var.append(dof_manager.dof_ind(sub_var.g, sub_var._name))
                    if i == 0:
                        # Store id of variable, but only for the first one; we will
                        # concatenate the arrays in ind_var into one array
                        variable_ids.append(variable.id)
            else:
                # This is a variable that lives on a single grid
                ind_var.append(dof_manager.dof_ind(variable.g, variable._name))
                variable_ids.append(variable.id)

            # Gather all indices for this variable
            inds.append(np.hstack([i for i in ind_var]))

        return inds, variable_ids, prev_time, prev_iter

    def _identify_subtree_discretizations(self, op, discr):

        if len(op.tree.children) > 0:
            for child in op.tree.children:
                discr += self._identify_subtree_discretizations(child, [])

        if isinstance(op, _MergedOperator):
            discr.append(op)

        return discr

    def _identify_discretizations(self):
        all_discr = self._identify_subtree_discretizations(self._operator, [])
        self.discretizations = _uniquify_discretization_list(all_discr)

    def discretize(self, gb: pp.GridBucket) -> None:
        _discretize_from_list(self.discretizations, gb)

    def to_ad(
        self,
        gb: pp.GridBucket,
        state: Optional[np.ndarray] = None,
        active_variables: Optional[list] = None,
    ):
        """Evaluate the residual and Jacobian matrix for a given state.

        Parameters:
            gb (pp.GridBucket): GridBucket used to represent the problem. Will be used
                to parse the operators that combine to form this Equation..
            state (np.ndarray, optional): State vector for which the residual and its
                derivatives should be formed. If not provided, the state will be pulled from
                the previous iterate (if this exists), or alternatively from the state
                at the previous time step.

        Returns:
            An Ad-array representation of the residual and Jacbobian.

        """
        # Parsing in two stages: First make an Ad-representation of the variable state
        # (this must be done jointly for all variables of the Equation to get all
        # derivatives represented). Then parse the equation by traversing its
        # tree-representation, and parse and combine individual operators.

        # Initialize variables
        prev_vals = np.zeros(self._dof_manager.num_dofs())

        populate_state = state is None
        if populate_state:
            state = np.zeros(self._dof_manager.num_dofs())

        assert state is not None
        for (g, var) in self._dof_manager.block_dof:
            ind = self._dof_manager.dof_ind(g, var)
            if isinstance(g, tuple):
                prev_vals[ind] = gb.edge_props(g, pp.STATE)[var]
            else:
                prev_vals[ind] = gb.node_props(g, pp.STATE)[var]

            if populate_state:
                if isinstance(g, tuple):
                    try:
                        state[ind] = gb.edge_props(g, pp.STATE)[pp.ITERATE][var]
                    except KeyError:
                        prev_vals[ind] = gb.edge_props(g, pp.STATE)[var]
                else:
                    try:
                        state[ind] = gb.node_props(g, pp.STATE)[pp.ITERATE][var]
                    except KeyError:
                        state[ind] = gb.node_props(g, pp.STATE)[var]

        # Initialize Ad variables with the current iterates
        if active_variables is None:
            ad_vars = initAdArrays([state[ind] for ind in self._variable_dofs])
            self._ad = {var_id: ad for (var_id, ad) in zip(self._variable_ids, ad_vars)}
        else:
            active_variable_ids = [v.id for v in active_variables]

            ad_variable_ids = list(
                set(self._variable_ids).intersection(active_variable_ids)
            )
            assert all([i in self._variable_ids for i in active_variable_ids])
            ad_variable_local_ids = [
                self._variable_ids.index(i) for i in active_variable_ids
            ]
            ad_variable_dofs = [self._variable_dofs[i] for i in ad_variable_local_ids]
            ad_vars = initAdArrays([state[ind] for ind in ad_variable_dofs])
            self._ad = {var_id: ad for (var_id, ad) in zip(ad_variable_ids, ad_vars)}

        # Also make mappings from the previous iteration.
        if active_variables is None:
            prev_iter_vals_list = [state[ind] for ind in self._prev_iter_dofs]
            self._prev_iter_vals = {
                var_id: val
                for (var_id, val) in zip(self._prev_iter_ids, prev_iter_vals_list)
            }
        else:
            # FIXME: This needs explanations
            prev_iter_vals_list = [state[ind] for ind in self._prev_iter_dofs]
            non_ad_variable_ids = list(set(self._variable_ids) - set(ad_variable_ids))
            non_ad_variable_local_ids = [
                self._variable_ids.index(i) for i in non_ad_variable_ids
            ]
            non_ad_variable_dofs = [
                self._variable_dofs[i] for i in non_ad_variable_local_ids
            ]
            non_ad_vals_list = [state[ind] for ind in non_ad_variable_dofs]
            self._prev_iter_vals = {
                var_id: val
                for (var_id, val) in zip(
                    self._prev_iter_ids + non_ad_variable_ids,
                    prev_iter_vals_list + non_ad_vals_list,
                )
            }

        # Also make mappings from the previous time step.
        prev_vals_list = [prev_vals[ind] for ind in self._prev_time_dofs]
        self._prev_vals = {
            var_id: val for (var_id, val) in zip(self._prev_time_ids, prev_vals_list)
        }

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
            # TODO no differecen between merged or no merged variables!?
            if isinstance(op, pp.ad.MergedVariable) or isinstance(
                op, operators.MergedVariable
            ):
                if op.prev_time:
                    return self._prev_vals[op.id]
                elif op.prev_iter:
                    return self._prev_iter_vals[op.id]
                else:
                    return self._ad[op.id]
            else:
                if op.prev_time:
                    return self._prev_vals[op.id]
                elif op.prev_iter or not (
                    op.id in self._ad
                ):  # TODO make it more explicit that op corresponds to a non_ad_variable?
                    # e.g. by op.id in non_ad_variable_ids.
                    return self._prev_iter_vals[op.id]
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

            # Convert any vectors that mascarade as a nx1 (1xn) scipy matrix
            self._ravel_scipy_matrix(results)

            return results[0] + results[1]

        elif tree.op == operators.Operation.sub:
            # To subtract we need two objects
            assert len(results) == 2

            # Convert any vectors that mascarade as a nx1 (1xn) scipy matrix
            self._ravel_scipy_matrix(results)

            return results[0] - results[1]

        elif tree.op == operators.Operation.mul:
            # To multiply we need two objects
            assert len(results) == 2
            return results[0] * results[1]

        elif tree.op == operators.Operation.evaluate:
            # This is a function, which should have at least one argument
            assert len(results) > 1
            return results[0].func(*results[1:])

        elif tree.op == operators.Operation.localeval:
            # This is a local function, which should have at least one argument
            assert len(results) > 1
            if all([isinstance(r, Ad_array) for r in results[1:]]):
                # TODO: return results[0].func(*makeLocalAd(results[1:]))
                if len(results) > 2:
                    # evaluation of local functions only supported for single
                    # argument functions.
                    raise RuntimeError("Not implemented.")
                else:
                    argval = results[1].val
                    argjac = results[1].jac.diagonal()
                    arg = Local_Ad_array(argval, argjac)
                    return results[0].func(arg)
            else:
                return results[0].func(*results[1:])

        elif tree.op == operators.Operation.apply:
            assert len(results) > 1
            return results[0].apply(*results[1:])

        elif tree.op == operators.Operation.div:
            return results[0] / results[1]

        else:
            raise ValueError("Should not happen")

    def _ravel_scipy_matrix(self, results):
        # In some cases, parsing may leave what is essentially an array, but with the
        # format of a scipy matrix. This must be converted to a numpy array before
        # moving on.
        # Note: It is not clear that this conversion is meaningful in all cases, so be
        # cautious with adding this extra parsing to more operations.
        for i, res in enumerate(results):
            if isinstance(res, sps.spmatrix):
                assert res.shape[0] == 1 or res.shape[1] == 1
                results[i] = res.toarray().ravel()


class EquationManager:
    # The usage of this class is not yet clear, and will likely undergo substantial
    # enhancements and changes.
    def __init__(
        self,
        gb: pp.GridBucket,
        dof_manager: pp.DofManager,
        equations: Optional[List[Expression]] = None,
    ) -> None:
        self.gb = gb

        # Inform mypy about variables, and then set them by a dedicated method.
        self.variables: Dict[grid_like_type, Dict[str, "pp.ad.Variable"]]
        self._set_variables(gb)

        if equations is None:
            self.equations: List[Expression] = []
        else:
            self.equations = equations

        self.dof_manager: pp.DofManager = dof_manager

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

    def merge_variables(
        self, grid_var: Sequence[Tuple[grid_like_type, str]]
    ) -> "pp.ad.MergedVariable":
        return pp.ad.MergedVariable([self.variables[g][v] for g, v in grid_var])

    def variable(self, grid_like: grid_like_type, variable: str) -> "pp.ad.Variable":
        # Method to access the variabe list; to syntax similar to merge_variables
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
        equations: Optional[List[str]] = None,
        ad_var: Optional[List[Union["pp.ad.Variable", "pp.ad.MergedVariable"]]] = None,
        state: Optional[np.ndarray] = None,
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        """Assemble residual vector and Jacobian matrix with respect to the current
        state represented in self.gb.

        As an experimental feature, subset of variables and equations can also be
        assembled. This functionality may be moved somewhere else in the future.

        Parameters:
            equations (list of str, optional): Name of equations to be assembled.
                Defaults to assembly of all equations.
            ad_var (Ad variable, optional): Variables to be assembled. Defaults to all
                variables present known to the EquationManager.
            state (np.ndarray, optional): State vector to assemble from. If not provided,
                the default behavior of pp.ad.Expression.to_ad() will be followed.

        """
        mat: List[sps.spmatrix] = []
        b: List[np.ndarray] = []

        # Make sure the variables are uniquely sorted
        if ad_var is None:
            num_global_dofs = self.dof_manager.num_dofs()
            variables = None
        else:
            variables = sorted(list(set(ad_var)), key=lambda v: v.id)
            names: List[str] = [v._name for v in variables]
            num_global_dofs = self.dof_manager.num_dofs(var=names)

        for eq in self.equations:

            # Neglect equation if not explicilty asked for.
            if equations is not None and not (eq.name in equations):
                continue

            ad = eq.to_ad(self.gb, state, active_variables=variables)
            # The columns of the Jacobian has the size of the local variables.
            # Map these to the global ones
            local_dofs = eq.local_dofs(true_ad_variables=variables)
            if variables is not None:
                local_dofs = self.dof_manager.transform_dofs(local_dofs, var=names)

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

    def discretize(self, gb):
        # Somehow loop over all equations, discretize identified objects
        # (but should also be able to do rediscretization based on
        # dependency graph etc).
        discr = []

        for eqn in self.equations:
            discr = eqn._identify_subtree_discretizations(eqn._operator, discr)

        unique_discr = _uniquify_discretization_list(discr)
        _discretize_from_list(unique_discr, gb)

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


def _uniquify_discretization_list(all_discr):
    unique_discr_grids: Dict[
        Union["pp.Discretization", "pp.AbstractInterfaceLaw"], List
    ] = {}

    cls_obj_map = {}

    cls_key_covered = []

    for discr in all_discr:
        cls = discr.discr.__class__
        param_keyword = discr.keyword

        key = (cls, param_keyword)

        if key in cls_key_covered:
            d = cls_obj_map[cls]
            for g in discr.grids:
                if g not in unique_discr_grids[d]:
                    unique_discr_grids[d].append(g)
        else:
            cls_obj_map[cls] = discr.discr
            cls_key_covered.append(key)
            unique_discr_grids[discr.discr] = discr.grids

    return unique_discr_grids


def _discretize_from_list(discretizations, gb):
    for discr in discretizations:
        # Discr has type _MergedOperator
        for g in discretizations[discr]:
            if isinstance(g, tuple):
                data = gb.edge_props(g)
                g_primary, g_secondary = g
                d_primary = gb.node_props(g_primary)
                d_secondary = gb.node_props(g_secondary)
                discr.discretize(g_primary, g_secondary, d_primary, d_secondary, data)
            else:
                data = gb.node_props(g)
                try:
                    discr.discretize(g, data)
                except NotImplementedError:
                    # This will likely be GradP and other Biot discretizations
                    pass
