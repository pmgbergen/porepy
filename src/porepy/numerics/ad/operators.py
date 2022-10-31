""" Implementation of wrappers for Ad representations of several operators.
"""
from __future__ import annotations

import copy
import numbers
from enum import Enum, EnumMeta
from itertools import count
from typing import Any, Optional, Sequence, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sps

import porepy as pp

from . import _ad_utils
from .forward_mode import Ad_array, initAdArrays

__all__ = [
    "Operator",
    "Matrix",
    "Array",
    "TimeDependentArray",
    "Scalar",
    "Variable",
    "MixedDimensionalVariable",
]

GridLike = Union[pp.Grid, pp.MortarGrid]


def _get_shape(mat):
    """Get shape of a numpy.ndarray or the Jacobian of Ad_array"""
    if isinstance(mat, Ad_array):
        return mat.jac.shape
    else:
        return mat.shape


class Operator:
    """Superclass for all Ad operators.

    Objects of this class are not meant to be initiated directly, rather the various
    subclasses should be used. Instances of this class will still be created when
    subclasses are combined by operations.

    """

    Operations: EnumMeta = Enum("Operations", [
        "void", "add", "sub", "mul", "div", "evaluate", "approximate"
    ])
    """Object representing all supported operations by the operator class.

    Used to construct the operator tree and identify operations.

    """

    def __init__(
        self,
        name: Optional[str] = None,
        subdomains: Optional[list[pp.Grid]] = None,
        interfaces: Optional[list[pp.MortarGrid]] = None,
        tree: Optional[Tree] = None,
    ) -> None:
        self.interfaces: list[pp.MortarGrid] = [] if interfaces is None else interfaces
        """List of interfaces on which the operator is defined.
        Will be empty for operators not associated with specific interfaces.

        """

        self.subdomains: list[pp.Grid] = [] if subdomains is None else subdomains
        """List of subdomains on which the operator is defined.
        Will be empty for operators not associated with specific subdomains.

        """
        ### PRIVATE

        self._name = name if name is not None else ""

        self._set_tree(tree)

    @property
    def name(self) -> str:
        """The name given to this variable."""
        return self._name

    def _set_tree(self, tree=None):
        if tree is None:
            self.tree = Tree(Operation.void)
        else:
            self.tree = tree

    def _set_subdomains_or_interfaces(
        self,
        subdomains: Optional[list[pp.Grid]] = None,
        interfaces: Optional[list[pp.MortarGrid]] = None,
    ) -> None:
        """For operators which are defined for either subdomains or interfaces but not both.

        Check that exactly one of subdomains and interfaces is given and assign to the
        operator. The unspecified grid-like type will also be set as an attribute, i.e.
        either op.subdomains or op.interfaces is an empty list, while the other is a
        list with len>0.

        Parameters:
            subdomains (optional list of subdomains): The subdomain list.
            interfaces (optional list of tuples of subdomains): The interface list.

        """
        if subdomains is None:
            subdomains = []
        if interfaces is None:
            interfaces = []

        if len(subdomains) > 0 and len(interfaces) > 0:
            raise ValueError("Operator defined on both subdomains and interfaces.")

        self.subdomains = subdomains
        self.interfaces = interfaces

    def _find_subtree_variables(self) -> list[pp.ad.Variable]:
        """Method to recursively look for Variables (or MixedDimensionalVariables) in an
        operator tree.
        """
        # The variables should be located at leaves in the tree. Traverse the tree
        # recursively, look for variables, and then gather the results.

        if isinstance(self, Variable) or isinstance(self, pp.ad.Variable):
            # We are at the bottom of a branch of the tree, return the operator
            return [self]
        else:
            # We need to look deeper in the tree.
            # Look for variables among the children
            sub_variables = []
            # When using nested pp.ad.Functions, some of the children may be Ad_arrays
            # (forward mode), rather than Operators. For the former, don't look for
            # children - they have none.
            for child in self.tree.children:
                if isinstance(child, pp.ad.Operator):
                    sub_variables += child._find_subtree_variables()

            # Some work is needed to parse the information
            var_list: list[Variable] = []
            for var in sub_variables:
                if isinstance(var, Variable) or isinstance(var, pp.ad.Variable):
                    # Effectively, this node is one step from the leaf
                    var_list.append(var)
                elif isinstance(var, list):
                    # We are further up in the tree.
                    for sub_var in var:
                        if isinstance(sub_var, Variable) or isinstance(
                            sub_var, pp.ad.Variable
                        ):
                            var_list.append(sub_var)
            return var_list

    def _identify_variables(
        self, dof_manager: pp.DofManager, var: Optional[list] = None
    ):
        """Identify all variables in this operator."""
        # 1. Get all variables present in this operator.
        # The variable finder is implemented in a special function, aimed at recursion
        # through the operator tree.
        # Uniquify by making this a set, and then sort on variable id
        variables = sorted(
            list(set(self._find_subtree_variables())),
            key=lambda var: var.id,
        )

        # 2. Get a mapping between variables (*not* only MixedDimensionalVariables) and their
        # indices according to the DofManager. This is needed to access the state of
        # a variable when parsing the operator to numerical values using forward Ad.

        # For each variable, get the global index
        inds = []
        variable_ids = []
        prev_time = []
        prev_iter = []
        for variable in variables:
            # Indices (in DofManager sense) of this variable. Will be built gradually
            # for MixedDimensionalVariables, in one go for plain Variables.
            ind_var = []
            prev_time.append(variable.prev_time)
            prev_iter.append(variable.prev_iter)

            if isinstance(
                variable, (pp.ad.MixedDimensionalVariable, MixedDimensionalVariable)
            ):  # Is this equivalent to the test in previous function?
                # Loop over all subvariables for the merged variable
                for i, sub_var in enumerate(variable.sub_vars):
                    # Store dofs
                    ind_var.append(
                        dof_manager.grid_and_variable_to_dofs(sub_var._g, sub_var._name)
                    )
                    if i == 0:
                        # Store id of variable, but only for the first one; we will
                        # concatenate the arrays in ind_var into one array
                        variable_ids.append(variable.id)

                if len(variable.sub_vars) == 0:
                    # For empty lists of subvariables, we still need to assign an id
                    # to the variable.
                    variable_ids.append(variable.id)
            else:
                # This is a variable that lives on a single grid
                ind_var.append(
                    dof_manager.grid_and_variable_to_dofs(variable._g, variable._name)
                )
                variable_ids.append(variable.id)

            # Gather all indices for this variable
            if len(ind_var) > 0:
                inds.append(np.hstack([i for i in ind_var]))
            else:
                inds.append(np.array([], dtype=int))

        return inds, variable_ids, prev_time, prev_iter

    def _identify_subtree_discretizations(self, discr: list) -> list:
        """Recursive search in the tree of this operator to identify all discretizations
        represented in the operator.
        """
        if len(self.tree.children) > 0:
            # Go further in recursion
            for child in self.tree.children:
                discr += child._identify_subtree_discretizations([])

        if isinstance(self, _ad_utils.MergedOperator):
            # We have reached the bottom; this is a discretization (example: mpfa.flux)
            discr.append(self)

        return discr

    def _identify_discretizations(
        self,
    ) -> dict[_ad_utils.MergedOperator, GridLike]:
        """Perform a recursive search to find all discretizations present in the
        operator tree. Uniquify the list to avoid double computations.

        """
        all_discr = self._identify_subtree_discretizations([])
        return _ad_utils.uniquify_discretization_list(all_discr)

    def discretize(self, mdg: pp.MixedDimensionalGrid) -> None:
        """Perform discretization operation on all discretizations identified in
        the tree of this operator, using data from mdg.

        IMPLEMENTATION NOTE: The discretizations was identified at initialization of
        Expression - it is now done here to accommodate updates (?) and

        """
        unique_discretizations: dict[
            _ad_utils.MergedOperator, GridLike
        ] = self._identify_discretizations()
        _ad_utils.discretize_from_list(unique_discretizations, mdg)

    def is_leaf(self) -> bool:
        """Check if this operator is a leaf in the tree-representation of an object.

        Returns:
            bool: True if the operator has no children. Note that this implies that the
                method parse() is expected to be implemented.
        """
        return len(self.tree.children) == 0

    def set_name(self, name: str) -> None:
        self._name = name

    def previous_timestep(self) -> pp.ad.Operator:
        """Return an operator that represents the value of this operator at the previous
        timestep.

        The operator tree at the previous time step is created as a shallow copy, and will
        thus be identical to the original operator, except that all time dependent operators
        are evaluated at the previous time step.

        Returns:
            A copy of self, with all time dependent operators evaluated at the previous
                time step.

        """
        # Create a copy of the operator tree evaluated at a previous time step. This is done
        # by traversing the underlying graph, and set all time-dependent objects to be
        # evaluated at the previous time step.

        def _traverse_tree(op: Operator) -> Operator:
            """Helper function which traverses an operator tree by recursion."""

            children = op.tree.children

            if len(children) == 0:
                # We are on an atomic operator. If this is a time-dependent operator,
                # set it to be evaluated at the previous time step. If not, leave the
                # operator as it is.
                if isinstance(op, (Variable, MixedDimensionalVariable, TimeDependentArray)):
                    # IMPLEMENTATION NOTE: We need to use a copy, not a deep copy here. A
                    # deep copy would change the underlying grids and mortar grids. For
                    # variables (atomic and merged) this would render a KeyValue if the
                    # grid is used to fetch the relevant degree of freedom in the global
                    # ordering, as is done by the DofManager.
                    # If other time-dependent other operators are added, they will have to
                    # override this previous_timestep method.
                    return copy.copy(op).previous_timestep()

                else:
                    # No need to use a copy here.
                    # This also means that operators that are not time dependent need not
                    # override this previous_timestep method.
                    return op
            else:
                # Recursively iterate over the subtree, get the children, evaluated at the
                # previous time when relevant, and add it to the new list.
                new_children: list[Operator] = list()
                for ci, child in enumerate(children):
                    # Recursive call to fix the subtree.
                    new_children.append(_traverse_tree(child))

                # We would like to return a new operator which represents the same
                # calculation as op, though with a different set of children. We cannot use
                # copy.copy (shallow copy), since this will identify the lists of children
                # in the old and new operator. Also, we cannot do a deep copy, since this
                # will copy grids in individual subdomains - see implementation not in the
                # above treatment of Variables.
                # The solution is to make a new Tree with the same operation as the old
                # operator, but with the new list of children.
                new_tree = Tree(op.tree.op, children=new_children)

                # Use the same lists of subdomains and interfaces as in the old operator,
                # with empty lists if these are not present.
                subdomains = getattr(op, "subdomains", [])
                interfaces = getattr(op, "interfaces", [])

                # Create new operator from the tree.
                new_op = Operator(
                    name=op._name,
                    subdomains=subdomains,
                    interfaces=interfaces,
                    tree=new_tree,
                )
                return new_op

        # Get a copy of the operator with all time-dependent quantities evaluated at the
        # previous time step.
        prev_time = _traverse_tree(self)

        return prev_time

    def parse(self, mdg: pp.MixedDimensionalGrid) -> Any:
        """Translate the operator into a numerical expression.
        Subclasses that represent atomic operators (leaves in a tree-representation of
        an operator) should override this method to return e.g. a number, an array or a
        matrix.
        This method should not be called on operators that are formed as combinations
        of atomic operators; such operators should be evaluated by the method evaluate().
        """
        raise NotImplementedError("This type of operator cannot be parsed right away")

    def _parse_operator(self, op: Operator, mdg: pp.MixedDimensionalGrid):
        """TODO: Currently, there is no prioritization between the operations; for
        some reason, things just work. We may need to make an ordering in which the
        operations should be carried out. It seems that the strategy of putting on
        hold until all children are processed works, but there likely are cases where
        this is not the case.
        """

        # The parsing strategy depends on the operator at hand:
        # 1) If the operator is a Variable, it will be represented according to its
        #    state.
        # 2) If the operator is a leaf in the tree-representation of the operator,
        #    parsing is left to the operator itself.
        # 3) If the operator is formed by combining other operators lower in the tree,
        #    parsing is handled by first evaluating the children (leads to recursion)
        #    and then perform the operation on the result.

        # Check for case 1 or 2
        if isinstance(op, pp.ad.Variable) or isinstance(op, Variable):
            # Case 1: Variable

            # How to access the array of (Ad representation of) states depends on weather
            # this is a single or combined variable; see self.__init__, definition of
            # self._variable_ids.
            # TODO no different between merged or no merged variables!?
            if isinstance(op, pp.ad.MixedDimensionalVariable) or isinstance(op, MixedDimensionalVariable):
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
        elif isinstance(op, pp.ad.Ad_array):
            # When using nested operator functions, op can be an already evaluated term.
            # Just return it.
            return op

        elif op.is_leaf():
            # Case 2
            return op.parse(mdg)  # type:ignore

        # This is not an atomic operator. First parse its children, then combine them
        tree = op.tree
        results = [self._parse_operator(child, mdg) for child in tree.children]

        # Combine the results
        if tree.op == Operation.add:
            # To add we need two objects
            assert len(results) == 2

            # Convert any vectors that mascarade as a nx1 (1xn) scipy matrix
            self._ravel_scipy_matrix(results)

            if isinstance(results[0], np.ndarray):
                # With the implementation of Ad arrays, addition does not
                # commute for combinations with numpy arrays. Switch the order
                # of results, and everything works.
                results = results[::-1]
            try:
                return results[0] + results[1]
            except ValueError as exc:
                msg = self._get_error_message("adding", tree, results)
                raise ValueError(msg) from exc

        elif tree.op == Operation.sub:
            # To subtract we need two objects
            assert len(results) == 2

            # Convert any vectors that mascarade as a nx1 (1xn) scipy matrix
            self._ravel_scipy_matrix(results)

            factor = 1

            if isinstance(results[0], np.ndarray):
                # With the implementation of Ad arrays, subtraction does not
                # commute for combinations with numpy arrays. Switch the order
                # of results, and everything works.
                results = results[::-1]
                factor = -1

            try:
                return factor * (results[0] - results[1])
            except ValueError as exc:
                msg = self._get_error_message("subtracting", tree, results)
                raise ValueError(msg) from exc

        elif tree.op == Operation.mul:
            # To multiply we need two objects
            assert len(results) == 2

            if isinstance(results[0], np.ndarray) and isinstance(
                results[1], (pp.ad.Ad_array, pp.ad.forward_mode.Ad_array)
            ):
                # In the implementation of multiplication between an Ad_array and a
                # numpy array (in the forward mode Ad), a * b and b * a do not
                # commute. Flip the order of the results to get the expected behavior.
                results = results[::-1]
            try:
                return results[0] * results[1]
            except ValueError as exc:
                if isinstance(
                    results[0], (pp.ad.Ad_array, pp.ad.forward_mode.Ad_array)
                ) and isinstance(results[1], np.ndarray):
                    # Special error message here, since the information provided by
                    # the standard method looks like a contradiction.
                    # Move this to a helper method if similar cases arise for other
                    # operations.
                    msg_0 = tree.children[0]._parse_readable()
                    msg_1 = tree.children[1]._parse_readable()
                    nl = "\n"
                    msg = (
                        "Error when right multiplying \n"
                        + f"  {msg_0}"
                        + nl
                        + "with"
                        + nl
                        + f"  numpy array {msg_1}"
                        + nl
                        + f"Size of arrays: {results[0].val.size} and {results[1].size}"
                        + nl
                        + "Did you forget some parentheses?"
                    )

                else:
                    msg = self._get_error_message("multiplying", tree, results)
                raise ValueError(msg) from exc

        elif tree.op == Operation.div:
            # Some care is needed here, to account for cases where item in the results
            # array is a numpy array
            if isinstance(results[0], pp.ad.Ad_array):
                # If the first item is an Ad array, the implementation of the forward
                # mode should take care of everything.
                return results[0] / results[1]
            elif isinstance(results[0], (np.ndarray, sps.spmatrix)):
                # if the first array is a numpy array or sparse matrix,
                # then numpy's implementation of division will be invoked.
                if isinstance(results[1], (np.ndarray, numbers.Real)):
                    # Both items are numpy arrays or scalars, everything is fine.
                    return results[0] / results[1]
                elif isinstance(results[1], pp.ad.Ad_array):
                    # Numpy cannot deal with division with an Ad_array. Instead, multiply
                    # with the inverse of results[1] (this is equivalent, and makes
                    # numpy happy). The return from numpy will be a new array (data type
                    # object) with the actual Ad_array as the first item. Exactly why
                    # numpy functions in this way is not clear to EK.
                    return (results[0] * results[1] ** -1)[0]
                else:
                    # Not sure what this will cover. We have to wait for it to happen.
                    raise NotImplementedError(
                        "Encountered a case not covered when dividing Ad objects"
                    )
            elif isinstance(results[0], numbers.Real):
                # if the dividend is a number, the divisor has to be an Ad_array,
                # otherwise the overloaded division wouldn't have been invoked
                # We use the same strategy as in above case where the divisor is an Ad_array
                if isinstance(results[1], pp.ad.Ad_array):
                    # See remarks by EK in case ndarray / Ad_array
                    return (results[0] * results[1] ** -1)[0]
                else:
                    # In case above argument, that the divisor can only be an Ad_array,
                    # is wrong
                    raise NotImplementedError(
                        "Encountered a case not covered when dividing Ad objects"
                    )
            else:
                # This case could include results[0] being a float, or different numbers,
                # which again should be easy to cover.
                raise NotImplementedError(
                    "Encountered a case not covered when dividing Ad objects"
                )

        elif tree.op == Operation.evaluate:
            # This is a function, which should have at least one argument
            assert len(results) > 1
            func_op = results[0]

            # if the callable can be fed with Ad_arrays, do it
            if func_op.ad_compatible:
                return func_op.func(*results[1:])
            else:
                # This should be a Function with approximated Jacobian and value.
                try:
                    val = func_op.get_values(*results[1:])
                    jac = func_op.get_jacobian(*results[1:])
                except Exception as exc:
                    # TODO specify what can go wrong here (Exception type)
                    msg = "Ad parsing: Error evaluating operator function:\n"
                    msg += func_op._parse_readable()
                    raise ValueError(msg) from exc
                return Ad_array(val, jac)

        else:
            raise ValueError("Should not happen")

    def _get_error_message(self, operation: str, tree, results: list) -> str:
        # Helper function to format error message
        msg_0 = tree.children[0]._parse_readable()
        msg_1 = tree.children[1]._parse_readable()

        nl = "\n"
        msg = (
            f"Ad parsing: Error when {operation}\n"
            + "  "
            + msg_0
            + nl
            + "with"
            + nl
            + "  "
            + msg_1
            + nl
        )

        msg += (
            f"Matrix sizes are {_get_shape(results[0])} and "
            f"{_get_shape(results[1])}"
        )
        return msg

    def _parse_readable(self) -> str:
        """
        Make a human-readable error message related to a parsing error.
        NOTE: The exact formatting should be considered work in progress,
        in particular when it comes to function evaluation.
        """

        # There are three cases to consider: Either the operator is a leaf,
        # it is a composite operator with a name, or it is a general composite
        # operator.
        if self.is_leaf():
            # Leafs are represented by their strings.
            return str(self)
        elif self._name is not None:
            # Composite operators that have been given a name (possibly
            # with a goal of simple identification of an error)
            return self._name

        # General operator. Split into its parts by recursion.
        tree = self.tree

        child_str = [child._parse_readable() for child in tree.children]

        is_func = False
        operator_str = None

        # readable representations of known operations
        if tree.op == Operation.add:
            operator_str = "+"
        elif tree.op == Operation.sub:
            operator_str = "-"
        elif tree.op == Operation.mul:
            operator_str = "*"
        elif tree.op == Operation.div:
            operator_str = "/"
        # function evaluations have their own readable representation
        elif tree.op == Operation.evaluate:
            is_func = True
        # for unknown operations, 'operator_str' remains None

        # error message for function evaluations
        if is_func:
            msg = f"{child_str[0]}("
            msg += ", ".join([f"{child}" for child in child_str[1:]])
            msg += ")"
            return msg
        # if operation is unknown, a new error will be raised to raise awareness
        elif operator_str is None:
            msg = "UNKNOWN parsing of operation on: "
            msg += ", ".join([f"{child}" for child in child_str])
            raise NotImplementedError(msg)
        # error message for known Operations
        else:
            return f"({child_str[0]} {operator_str} {child_str[1]})"

    def _ravel_scipy_matrix(self, results):
        # In some cases, parsing may leave what is essentially an array, but with the
        # format of a scipy matrix. This must be converted to a numpy array before
        # moving on.
        # Note: It is not clear that this conversion is meaningful in all cases, so be
        # cautious with adding this extra parsing to more operations.
        for i, res in enumerate(results):
            if isinstance(res, sps.spmatrix):
                assert res.shape[0] <= 1 or res.shape[1] <= 1
                results[i] = res.toarray().ravel()

    def __repr__(self) -> str:
        if self._name is None or len(self._name) == 0:
            s = "Operator with no name"
        else:
            s = f"Operator named {self._name}"
        s += f" formed by {self.tree.op} with {len(self.tree.children)} children."
        return s

    def __str__(self) -> str:
        return self._name if self._name is not None else ""

    def viz(self):
        """Give a visualization of the operator tree that has this operator at the top."""
        G = nx.Graph()

        def parse_subgraph(node: Operator):
            G.add_node(node)
            if len(node.tree.children) == 0:
                return
            operation = node.tree.op
            G.add_node(operation)
            G.add_edge(node, operation)
            for child in node.tree.children:
                parse_subgraph(child)
                G.add_edge(child, operation)

        parse_subgraph(self)
        nx.draw(G, with_labels=True)
        plt.show()

    def is_leaf(self) -> bool:
        """Check if this operator is a leaf in the tree-representation of an object.

        Returns:
            bool: True if the operator has no children. Note that this implies that the
                method parse() is expected to be implemented.
        """
        return len(self.tree.children) == 0

    def _set_tree(self, tree=None):
        """Helper method to set the tree, alternatively instantiate a tree using the void
        operation.

        """
        if tree is None:
            self.tree = Tree(Operator.Operations.void)
        else:
            self.tree = tree

    def set_name(self, name: str) -> None:
        self._name = name

    ### Operator discretization ---------------------------------------------------------------
    # TODO this is specific to discretizations and should not be done here
    # let the AD system do this by calling respective util methods

    def discretize(self, mdg: pp.MixedDimensionalGrid) -> None:
        """Perform discretization operation on all discretizations identified in
        the tree of this operator, using data from mdg.

        IMPLEMENTATION NOTE: The discretizations was identified at initialization of
        Expression - it is now done here to accommodate updates (?) and

        """
        unique_discretizations: dict[
            _ad_utils.MergedOperator, GridLike
        ] = self._identify_discretizations()
        _ad_utils.discretize_from_list(unique_discretizations, mdg)

    def _identify_discretizations(
        self,
    ) -> dict['_ad_utils.MergedOperator', GridLike]:
        """Perform a recursive search to find all discretizations present in the
        operator tree. Uniquify the list to avoid double computations.

        """
        all_discr = self._identify_subtree_discretizations([])
        return _ad_utils.uniquify_discretization_list(all_discr)

    def _identify_subtree_discretizations(self, discr: list) -> list:
        """Recursive search in the tree of this operator to identify all discretizations
        represented in the operator.
        """
        if len(self.tree.children) > 0:
            # Go further in recursion
            for child in self.tree.children:
                discr += child._identify_subtree_discretizations([])

        if isinstance(self, _ad_utils.MergedOperator):
            # We have reached the bottom; this is a discretization (example: mpfa.flux)
            discr.append(self)

        return discr


    ### Operator parsing ----------------------------------------------------------------------

    def evaluate(
        self,
        system_manager: pp.ad.SystemManager,
        state: Optional[np.ndarray] = None,
    ):  # TODO ensure the operator always returns an AD array
        """Evaluate the residual and Jacobian matrix for a given state.

        Parameters:
            dof_manager (pp.DofManager): used to represent the problem. Will be used
                to parse the sub-operators that combine to form this operator.
            state (optional): State vector for which the residual and its
                derivatives should be formed. If not provided, the state will be pulled from
                the previous iterate (if this exists), or alternatively from the state
                at the previous time step.

        Returns:
            An Ad-array representation of the residual and Jacobian. Note that the Jacobian
            matrix need not be invertible, or ever square; this depends on the operator.

        """
        # Get the mixed-dimensional grid used for the dof-manager.
        mdg = system_manager.mdg

        # Identify all variables in the Operator tree. This will include real variables,
        # and representation of previous time steps and iterations.
        (
            variable_dofs,
            variable_ids,
            is_prev_time,
            is_prev_iter,
        ) = self._identify_variables(system_manager)

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
        # IMPLEMENTATION NOTE: Storage in a separate data class could have
        # been a more elegant option.
        self._variable_dofs = current_indices
        self._variable_ids = current_ids
        self._prev_time_dofs = prev_indices
        self._prev_time_ids = prev_ids
        self._prev_iter_dofs = prev_iter_indices
        self._prev_iter_ids = prev_iter_ids

        # Parsing in two stages: First make a forward Ad-representation of the variable
        # state (this must be done jointly for all variables of the operator to get all
        # derivatives represented). Then parse the operator by traversing its
        # tree-representation, and parse and combine individual operators.

        populate_state = state is None

        prev_vals = system_manager.get_variable_values(from_iterate=False)

        if populate_state:
            state = system_manager.get_variable_values(from_iterate=True)

        # Initialize Ad variables with the current iterates

        # The size of the Jacobian matrix will always be set according to the
        # variables found by the DofManager in the MixedDimensionalGrid.

        # NOTE: This implies that to derive a subsystem from the Jacobian
        # matrix of this Expression will require restricting the columns of
        # this matrix.

        # First generate an Ad array (ready for forward Ad) for the full set.
        ad_vars = initAdArrays([state])[0]

        # Next, the Ad array must be split into variables of the right size
        # (splitting impacts values and number of rows in the Jacobian, but
        # the Jacobian columns must stay the same to preserve all cross couplings
        # in the derivatives).

        # Dictionary which maps from Ad variable ids to Ad_array.
        self._ad: dict[int, Ad_array] = {}

        # Loop over all variables, restrict to an Ad array corresponding to
        # this variable.
        for (var_id, dof) in zip(self._variable_ids, self._variable_dofs):
            ncol = state.size
            nrow = np.unique(dof).size
            # Restriction matrix from full state (in Forward Ad) to the specific
            # variable.
            R = sps.coo_matrix(
                (np.ones(nrow), (np.arange(nrow), dof)), shape=(nrow, ncol)
            ).tocsr()
            self._ad[var_id] = R * ad_vars

        # Also make mappings from the previous iteration.
        # This is simpler, since it is only a matter of getting the residual vector
        # correctly (not Jacobian matrix).

        prev_iter_vals_list = [state[ind] for ind in self._prev_iter_dofs]
        self._prev_iter_vals = {
            var_id: val
            for (var_id, val) in zip(self._prev_iter_ids, prev_iter_vals_list)
        }

        # Also make mappings from the previous time step.
        prev_vals_list = [prev_vals[ind] for ind in self._prev_time_dofs]
        self._prev_vals = {
            var_id: val for (var_id, val) in zip(self._prev_time_ids, prev_vals_list)
        }

        # Parse operators. This is left to a separate function to facilitate the
        # necessary recursion for complex operators.
        eq = self._parse_operator(self, mdg)

        return eq

    def parse(self, mdg: pp.MixedDimensionalGrid) -> Any:
        """Translate the operator into a numerical expression.

        Subclasses that represent atomic operators (leaves in a tree-representation of
        an operator) should override this method to return e.g. a number, an array or a
        matrix.
        This method should not be called on operators that are formed as combinations
        of atomic operators; such operators should be evaluated by the method evaluate().

        """
        raise NotImplementedError("This type of operator cannot be parsed right away")

    def _identify_variables(self, system_manager: pp.ad.SystemManager, var: Optional[list] = None):
        """Identify all variables in this operator."""
        # 1. Get all variables present in this operator.
        # The variable finder is implemented in a special function, aimed at recursion
        # through the operator tree.
        # Uniquify by making this a set, and then sort on variable id
        variables = sorted(
            list(set(self._find_subtree_variables())),
            key=lambda var: var.id,
        )

        # 2. Get a mapping between variables (*not* only MixedDimensionalVariables) and their
        # indices according to the DofManager. This is needed to access the state of
        # a variable when parsing the operator to numerical values using forward Ad.

        # For each variable, get the global index
        inds = []
        variable_ids = []
        prev_time = []
        prev_iter = []
        for variable in variables:
            # Indices (in DofManager sense) of this variable. Will be built gradually
            # for MixedDimensionalVariables, in one go for plain Variables.
            ind_var = []
            prev_time.append(variable.prev_time)
            prev_iter.append(variable.prev_iter)

            if isinstance(variable, MixedDimensionalVariable):
                # Is this equivalent to the test in previous function?
                # Loop over all subvariables for the merged variable
                for i, sub_var in enumerate(variable.sub_vars):
                    # Store dofs
                    ind_var.append(
                        system_manager.dofs_of(sub_var)
                    )
                    if i == 0:
                        # Store id of variable, but only for the first one; we will
                        # concatenate the arrays in ind_var into one array
                        variable_ids.append(variable.id)

                if len(variable.sub_vars) == 0:
                    # For empty lists of subvariables, we still need to assign an id
                    # to the variable.
                    variable_ids.append(variable.id)
            else:
                # This is a variable that lives on a single grid
                ind_var.append(
                    system_manager.dofs_of(variable)
                )
                variable_ids.append(variable.id)

            # Gather all indices for this variable
            if len(ind_var) > 0:
                inds.append(np.hstack([i for i in ind_var]))
            else:
                inds.append(np.array([], dtype=int))

        return inds, variable_ids, prev_time, prev_iter

    def _find_subtree_variables(self) -> Sequence[Variable]:
        """Method to recursively look for Variables (or MixedDimensionalVariables) in an
        operator tree.
        """
        # The variables should be located at leaves in the tree. Traverse the tree
        # recursively, look for variables, and then gather the results.

        if isinstance(self, Variable):
            # We are at the bottom of a branch of the tree, return the operator
            return [self]
        else:
            # We need to look deeper in the tree.
            # Look for variables among the children
            sub_variables = []
            # When using nested pp.ad.Functions, some of the children may be Ad_arrays
            # (forward mode), rather than Operators. For the former, don't look for
            # children - they have none.
            for child in self.tree.children:
                if isinstance(child, Operator):
                    sub_variables += child._find_subtree_variables()

            # Some work is needed to parse the information
            var_list: list[Variable] = []
            for var in sub_variables:
                if isinstance(var, Variable):
                    # Effectively, this node is one step from the leaf
                    var_list.append(var)
                elif isinstance(var, list):
                    # We are further up in the tree.
                    for sub_var in var:
                        if isinstance(sub_var, Variable):
                            var_list.append(sub_var)
            return var_list

    # TODO define clear return type
    def _parse_operator(self, op: Operator, mdg: pp.MixedDimensionalGrid):
        """Parses combined operators recursively into a numeric representation.

        Parses first recursively all children in the operator tree.
        Then returns the resulting AD arrays and combines them according to Forward Mode.
        The parsing of operation follows Python's Operator precedence.

        """

        # Case 1: If the operator is a Variable, it will be represented according to its state.
        if isinstance(op, Variable):
            # Case 1: Variable

            # How to access the array of (Ad representation of) states depends on weather
            # this is a single or combined variable; see self.__init__, definition of
            # self._variable_ids.
            # TODO no different between merged or no merged variables!?
            if isinstance(op, MixedDimensionalVariable):
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
        # Case 2: If the operator is already an AD array, return it
        elif isinstance(op, Ad_array):
            # When using nested operator functions, op can be an already evaluated term.
            # Just return it.
            return op
        # Case 3: If the operator is a leaf in the tree-representation of the operator,
        # parsing is left to the operator itself.
        elif op.is_leaf():
            # Case 2
            return op.parse(mdg)  # type:ignore

        # Case 4:
        # This is not an atomic operator. First parse its children (recursively),
        # then combine the results using the respective operation
        tree = op.tree
        results = [self._parse_operator(child, mdg) for child in tree.children]

        # Combine the results
        if tree.op == Operator.Operations.add:
            # To add we need two objects
            assert len(results) == 2

            # Convert any vectors that mascarade as a nx1 (1xn) scipy matrix
            self._ravel_scipy_matrix(results)

            if isinstance(results[0], np.ndarray):
                # With the implementation of Ad arrays, addition does not
                # commute for combinations with numpy arrays. Switch the order
                # of results, and everything works.
                results = results[::-1]
            try:
                return results[0] + results[1]
            except ValueError as exc:
                msg = self._get_error_message("adding", tree, results)
                raise ValueError(msg) from exc

        elif tree.op == Operator.Operations.sub:
            # To subtract we need two objects
            assert len(results) == 2

            # Convert any vectors that mascaraed as a nx1 (1xn) scipy matrix
            self._ravel_scipy_matrix(results)

            factor = 1

            if isinstance(results[0], np.ndarray):
                # With the implementation of Ad arrays, subtraction does not
                # commute for combinations with numpy arrays. Switch the order
                # of results, and everything works.
                results = results[::-1]
                factor = -1

            try:
                return factor * (results[0] - results[1])
            except ValueError as exc:
                msg = self._get_error_message("subtracting", tree, results)
                raise ValueError(msg) from exc

        elif tree.op == Operator.Operations.mul:
            # To multiply we need two objects
            assert len(results) == 2

            if isinstance(results[0], np.ndarray) and isinstance(results[1], Ad_array):
                # In the implementation of multiplication between an Ad_array and a
                # numpy array (in the forward mode Ad), a * b and b * a do not
                # commute. Flip the order of the results to get the expected behavior.
                results = results[::-1]
            try:
                return results[0] * results[1]
            except ValueError as exc:
                if isinstance(results[0], Ad_array) and isinstance(results[1], np.ndarray):
                    # Special error message here, since the information provided by
                    # the standard method looks like a contradiction.
                    # Move this to a helper method if similar cases arise for other
                    # operations.
                    msg_0 = tree.children[0]._parse_readable()
                    msg_1 = tree.children[1]._parse_readable()
                    nl = "\n"
                    msg = (
                        "Error when right multiplying \n"
                        + f"  {msg_0}"
                        + nl
                        + "with"
                        + nl
                        + f"  numpy array {msg_1}"
                        + nl
                        + f"Size of arrays: {results[0].val.size} and {results[1].size}"
                        + nl
                        + "Did you forget some parentheses?"
                    )

                else:
                    msg = self._get_error_message("multiplying", tree, results)
                raise ValueError(msg) from exc

        elif tree.op == Operator.Operations.div:
            # Some care is needed here, to account for cases where item in the results
            # array is a numpy array
            if isinstance(results[0], Ad_array):
                # If the first item is an Ad array, the implementation of the forward
                # mode should take care of everything.
                return results[0] / results[1]
            elif isinstance(results[0], (np.ndarray, sps.spmatrix)):
                # if the first array is a numpy array or sparse matrix,
                # then numpy's implementation of division will be invoked.
                if isinstance(results[1], (np.ndarray, numbers.Real)):
                    # Both items are numpy arrays or scalars, everything is fine.
                    return results[0] / results[1]
                elif isinstance(results[1], Ad_array):
                    # Numpy cannot deal with division with an Ad_array. Instead, multiply
                    # with the inverse of results[1] (this is equivalent, and makes
                    # numpy happy). The return from numpy will be a new array (data type
                    # object) with the actual Ad_array as the first item. Exactly why
                    # numpy functions in this way is not clear to EK.
                    return (results[0] * results[1] ** -1)[0]
                else:
                    # Not sure what this will cover. We have to wait for it to happen.
                    raise NotImplementedError(
                        "Encountered a case not covered when dividing Ad objects"
                    )
            elif isinstance(results[0], numbers.Real):
                # if the dividend is a number, the divisor has to be an Ad_array,
                # otherwise the overloaded division wouldn't have been invoked
                # We use the same strategy as in above case where the divisor is an Ad_array
                if isinstance(results[1], Ad_array):
                    # See remarks by EK in case ndarray / Ad_array
                    return (results[0] * results[1] ** -1)[0]
                elif isinstance(results[1], numbers.Real): # trivial case
                    return results[0] / results[1]
                elif isinstance(results[1], np.ndarray):
                    # element-wise division for numpy vectors
                    if len(results[1].shape) == 1:
                        return results[0] / results[1]
                    else:  # if it is a matrix, the inversion should be treated differently
                        return NotImplementedError(
                            "Encountered a case not covered when dividing Ad objects"
                        )
                else:
                    # In case above argument, that the divisor can only be an Ad_array,
                    # is wrong
                    raise NotImplementedError(
                        "Encountered a case not covered when dividing Ad objects"
                    )
            else:
                # This case could include results[0] being a float, or different numbers,
                # which again should be easy to cover.
                raise NotImplementedError(
                    "Encountered a case not covered when dividing Ad objects"
                )

        # function which can handle AD Arrays
        elif tree.op == Operator.Operations.evaluate:
            # This is a function, which should have at least one argument
            assert len(results) > 1
            func_op = results[0]
            # feed the callable directly with AD arrays
            try:
                return func_op.func(*results[1:])
            except Exception as exc:
                msg = "Ad parsing: Error evaluating operator function:\n"
                msg += func_op._parse_readable()
                raise ValueError(msg) from exc

        # functions which need some kind of other approximation
        elif tree.op == Operator.Operations.approximate:
            # This is a function, which should have at least one argument
            assert len(results) > 1
            func_op = results[0]
            try:
                val = func_op.get_values(*results[1:])
                jac = func_op.get_jacobian(*results[1:])
            except Exception as exc:
                msg = "Ad parsing: Error using approximate operator function:\n"
                msg += func_op._parse_readable()
                raise ValueError(msg) from exc
            return Ad_array(val, jac)

        # Unknown operation encountered, should not happen if AD framework used properly
        else:
            raise ValueError(f"Operator tree contains unknown operation {tree.op}")

    def _get_error_message(self, operation: str, tree, results: list) -> str:
        # Helper function to format error message
        msg_0 = tree.children[0]._parse_readable()
        msg_1 = tree.children[1]._parse_readable()

        msg = (
            f"Ad parsing: Error when {operation}\n"
            + "  "
            + msg_0
            + "\nwith\n"
            + "  "
            + msg_1
            + "\n"
        )

        msg += (
            f"Matrix sizes are {_get_shape(results[0])} and "
            f"{_get_shape(results[1])}"
        )
        return msg

    def _parse_readable(self) -> str:
        """Make a human-readable error message related to a parsing error."""

        # There are three cases to consider: Either the operator is a leaf,
        # it is a composite operator with a name, or it is a general composite
        # operator.
        if self.is_leaf():
            # Leafs are represented by their strings.
            return str(self)
        elif self._name is not None:
            # Composite operators that have been given a name (possibly
            # with a goal of simple identification of an error)
            return self._name

        # General operator. Split into its parts by recursion.
        tree = self.tree

        child_str = [child._parse_readable() for child in tree.children]

        is_func = False
        operator_str = None

        # readable representations of known operations
        if tree.op == Operator.Operations.add:
            operator_str = "+"
        elif tree.op == Operator.Operations.sub:
            operator_str = "-"
        elif tree.op == Operator.Operations.mul:
            operator_str = "*"
        elif tree.op == Operator.Operations.div:
            operator_str = "/"
        # function evaluations have their own readable representation
        elif tree.op in (Operator.Operations.evaluate, Operator.Operations.approximate):
            is_func = True
        # for unknown operations, 'operator_str' remains None

        # error message for function evaluations
        if is_func:
            msg = f"{child_str[0]}("
            msg += ", ".join([f"{child}" for child in child_str[1:]])
            msg += ")"
            return msg
        # if operation is unknown, a new error will be raised to raise awareness
        elif operator_str is None:
            msg = "UNKNOWN parsing of operation on: "
            msg += ", ".join([f"{child}" for child in child_str])
            raise NotImplementedError(msg)
        # error message for known Operations
        else:
            return f"({child_str[0]} {operator_str} {child_str[1]})"

    def _ravel_scipy_matrix(self, results):
        # In some cases, parsing may leave what is essentially an array, but with the
        # format of a scipy matrix. This must be converted to a numpy array before
        # moving on.
        # Note: It is not clear that this conversion is meaningful in all cases, so be
        # cautious with adding this extra parsing to more operations.
        for i, res in enumerate(results):
            if isinstance(res, sps.spmatrix):
                assert res.shape[0] <= 1 or res.shape[1] <= 1
                results[i] = res.toarray().ravel()

    ### Special methods -----------------------------------------------------------------------

    def __str__(self) -> str:
        return self._name if self._name is not None else ""

    def __repr__(self) -> str:
        if self._name is None or len(self._name) == 0:
            s = "Operator with no name"
        else:
            s = f"Operator '{self._name}'"
        s += f" formed by {self.tree.op} with {len(self.tree.children)} children."
        return s

    def __mul__(self, other):
        children = self._parse_other(other)
        return Operator(
            tree=Tree(Operator.Operations.mul, children), name="* operator"
        )

    def __truediv__(self, other):
        children = self._parse_other(other)
        return Operator(tree=Tree(Operator.Operations.div, children), name="/ operator")

    def __add__(self, other):
        children = self._parse_other(other)
        return Operator(tree=Tree(Operator.Operations.add, children), name="+ operator")

    def __sub__(self, other):
        children = self._parse_other(other)
        return Operator(tree=Tree(Operator.Operations.sub, children), name="- operator")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        # consider the expression a-b. right-subtraction means self == b
        children = self._parse_other(other)
        # we need to change the order here since a-b != b-a
        children = [children[1], children[0]]
        return Operator(tree=Tree(Operator.Operations.sub, children), name="- operator")

    def _parse_other(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return [self, Scalar(other)]
        elif isinstance(other, np.ndarray):
            return [self, Array(other)]
        elif isinstance(other, sps.spmatrix):
            return [self, Matrix(other)]
        elif isinstance(other, Operator):
            return [self, other]
        elif isinstance(other, Ad_array):
            # This may happen when using nested pp.ad.Function.
            return [self, other]
        else:
            raise ValueError(f"Cannot parse {other} as an AD operator")


class Matrix(Operator):
    """Ad representation of a sparse matrix.

    For dense matrices, use an Array instead.

    This is a shallow wrapper around the real matrix; it is needed to combine the matrix
    with other types of Ad objects.

    Attributes:
        shape (Tuple of ints): Shape of the wrapped matrix.

    """

    def __init__(self, mat: sps.spmatrix, name: Optional[str] = None) -> None:
        """Construct an Ad representation of a matrix.

        Parameters:
            mat (sps.spmatrix): Sparse matrix to be represented.

        """
        super().__init__(name=name)
        self._mat = mat
        self.shape = mat.shape

    def __repr__(self) -> str:
        return f"Matrix with shape {self._mat.shape} and {self._mat.data.size} elements"

    def __str__(self) -> str:
        s = "Matrix "
        if self._name is not None:
            s += self._name
        return s

    def parse(self, mdg) -> sps.spmatrix:
        """Convert the Ad matrix into an actual matrix.

        Parameters:
            mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid. Not used, but it is
                needed as input to be compatible with parse methods for other operators.

        Returns:
            sps.spmatrix: The wrapped matrix.

        """
        return self._mat

    def transpose(self) -> "Matrix":
        return Matrix(self._mat.transpose())


class Array(Operator):
    """Ad representation of a numpy array.

    For sparse matrices, use a Matrix instead.

    This is a shallow wrapper around the real array; it is needed to combine the array
    with other types of Ad objects.

    See also TimeDependentArray.

    """

    def __init__(self, values: np.ndarray, name: Optional[str] = None) -> None:
        """Construct an Ad representation of a numpy array.

        Parameters:
            values (np.ndarray): Numpy array to be represented.

        """
        super().__init__(name=name)
        self._values = values

    def __repr__(self) -> str:
        return f"Wrapped numpy array of size {self._values.size}"

    def __str__(self) -> str:
        s = "Array"
        if self._name is not None:
            s += f"({self._name})"
        return s

    def parse(self, mdg: pp.MixedDimensionalGrid) -> np.ndarray:
        """Convert the Ad Array into an actual array.

        Parameters:
            mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid. Not used, but it is
                needed as input to be compatible with parse methods for other operators.

        Returns:
            np.ndarray: The wrapped array.

        """
        return self._values


class TimeDependentArray(Array):
    """An Ad-wrapper around a time-dependent numpy array.

    The array is tied to a MixedDimensionalGrid, and is distributed among the data
    dictionaries associated with subdomains and interfaces. The array values are stored
    in data[pp.STATE][pp.ITERATE][self._name] for the current time and
    data[pp.STATE][self._name] for the previous time.

    The array can be differentiated in time using pp.ad.dt().

    The intended use is to represent time-varying quantities in equations, e.g., source
    terms. Future use will also include numerical values of boundary conditions,
    however, this is pending an update to the model classes.

    Attributes:
        prev_time (boolean): If True, the array will be evaluated using
            data[pp.STATE] (data being the data dictionaries for subdomains and
            interfaces), if False, data[pp.STATE][pp.ITERATE] is used.

    """

    def __init__(
        self,
        name: str,
        subdomains: Optional[list[pp.Grid]] = None,
        interfaces: Optional[list[pp.MortarGrid]] = None,
        previous_timestep: bool = False,
    ):
        """Initialize a TimeDependentArray.

        Args:
            name: Name of the variable. Should correspond to items in data[pp.STATE].
            subdomains: Subdomains on which the array is defined. Defaults to None.
            interfaces: Interfaces on which the array is defined. Defaults to None.

            Exactly one of subdomains and interfaces must be non-empty.

            previous_timestep: Flag indicating if the array should be evaluated at the
                previous time step.

        Raises:
            ValueError: If either none of, or both of, subdomains and interfaces are
                empty.

        """

        self._name: str = name

        if subdomains is None:
            subdomains = []
        if interfaces is None:
            interfaces = []

        if len(interfaces) == 0 and len(subdomains) == 0:
            raise ValueError(
                "A time dependent array must be associated with"
                " interfaces or subdomains."
            )
        if len(interfaces) > 0 and len(subdomains) > 0:
            raise ValueError(
                "A time dependent array must be associated with either"
                " interfaces or subdomains, not both."
            )

        # Store subdomains and interfaces.
        self.subdomains = subdomains
        self.interfaces = interfaces

        self._grids: Sequence[GridLike]
        self._is_interface_array: bool

        # Shorthand access to subdomain or interface grids:
        if len(interfaces) == 0:
            # Appease mypy
            assert all([isinstance(sd, pp.Grid) for sd in subdomains])
            self._grids = subdomains
            self._is_interface_array = False
        else:
            assert all([isinstance(intf, pp.MortarGrid) for intf in interfaces])
            self._grids = interfaces
            self._is_interface_array = True

        self.prev_time: bool = previous_timestep

        self._set_tree()

    def previous_timestep(self) -> TimeDependentArray:
        """Return a representation of this variable on the previous time step.

        Returns:
            This array represented at the previous time step.

        """
        if self._is_interface_array:
            return TimeDependentArray(
                self._name, interfaces=self.interfaces, previous_timestep=True
            )
        else:
            return TimeDependentArray(
                self._name, subdomains=self.subdomains, previous_timestep=True
            )

    def parse(self, mdg: pp.MixedDimensionalGrid) -> np.ndarray:
        """Convert this array into numerical values.

        The numerical values will be picked from the representation of the array in
        data[pp.STATE][pp.ITERATE] (where data is the data dictionary of the subdomains
        or interfaces of this Array), or, if self.prev_time = True, from data[pp.STATE].

        Args:
            mdg: Mixed-dimensional grid.

        Returns:
            A numpy ndarray containing the numerical values of this array.

        """
        vals = []
        for g in self._grids:
            if self._is_interface_array:
                # Appease mypy
                assert isinstance(g, pp.MortarGrid)
                data = mdg.interface_data(g)
            else:
                assert isinstance(g, pp.Grid)
                data = mdg.subdomain_data(g)

            if self.prev_time:
                vals.append(data[pp.STATE][self._name])
            else:
                vals.append(data[pp.STATE][pp.ITERATE][self._name])

        return np.hstack((vals))

    def __repr__(self) -> str:
        s = f"Wrapped time-dependent array with name {self._name}.\n"

        if self._is_interface_array:
            s += f"Defined on {len(self._grids)} interfaces.\n"
        else:
            s += f"Defined on {len(self._grids)} subdomains.\n"

        if self.prev_time:
            s += "Evaluated at the previous time step."
        return s


class Scalar(Operator):
    """Ad representation of a scalar.

    This is a shallow wrapper around the real scalar; it may be useful to combine
    the scalar with other types of Ad objects.

    NOTE: Since this is a wrapper around a Python immutable, copying a Scalar will
    effectively create a deep copy, i.e., changes in the value of one Scalar will not
    be reflected in the other. This is in contrast to the behavior of the other
    Ad objects.

    """

    def __init__(self, value: float, name: Optional[str] = None) -> None:
        """Construct an Ad representation of a float.

        Parameters:
            value (float): Number to be represented

        """
        super().__init__(name=name)
        self._value = value

    def __repr__(self) -> str:
        return f"Wrapped scalar with value {self._value}"

    def __str__(self) -> str:
        s = "Scalar"
        if self._name is not None:
            s += f"({self._name})"
        return s

    def parse(self, mdg: pp.MixedDimensionalGrid) -> float:
        """Convert the Ad Scalar into an actual number.

        Parameters:
            mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid. Not used, but it is
                needed as input to be compatible with parse methods for other operators.

        Returns:
            float: The wrapped number.

        """
        return self._value


class Variable(Operator):
    """Ad representation of a variable defined **either** on a single Grid **or** MortarGrid.

    Conversion of the variables into numerical value should be done with respect to the
    state of an array; see :meth:`Operator.evaluate`.
    Therefore, the class does not implement :meth:`Operator.parse`.

    Variables should not be created directly in a model. Use an AD System class instead.

    Parameters:
        name : given name. Used to together with the domain to identify the variable
        ndof: number of dofs per dof type (cells, faces, nodes).
        domain: domain of definition, i.e. either a grid or mortar grid representing a
            subdomain or interface.
        previous_timestep: flag indicating that the variable represents the state at the
            previous time step.
        previous_iteration: flag indicating that the variable represents the state at the
            previous iteration.

    """

    # Identifiers for variables. This will assign a unique id to all instances of this
    # class. This is used when operators are parsed to the forward Ad format. The
    # value of the id has no particular meaning.
    _ids = count(0)

    def __init__(
        self,
        name: str,
        ndof: dict[str, int],
        domain: GridLike,
        previous_timestep: bool = False,
        previous_iteration: bool = False,
    ) -> None:
        super().__init__(name=name)

        ### PUBLIC

        self.prev_time: bool = previous_timestep
        """Flag indicating if the variable represents the state at the previous time step."""

        self.prev_iter: bool = previous_iteration
        """Flag indicating if the variable represents the state at the previous iteration."""

        self.id = next(Variable._ids)
        """ID counter. Used to identify variables during operator parsing."""

        # overwrite properties set by the parent constructor
        if isinstance(domain, pp.MortarGrid):
            self.interfaces = [domain]
        elif isinstance(domain, pp.Grid):
            self.subdomains = [domain]
        else:
            raise ValueError("Variable domain must be either a grid or mortar grid.")

        ### PRIVATE
        # domain
        self._g: GridLike = domain
        # dofs per
        self._cells: int = ndof.get("cells", 0)
        self._faces: int = ndof.get("faces", 0)
        self._nodes: int = ndof.get("nodes", 0)

    @property
    def domain(self) -> GridLike:
        """The grid or mortar grid on which this variable is defined."""
        return self._g

    def size(self) -> int:
        """Returns the total number of dofs this variable has."""
        # TODO access to num_cells etc will throw an error if compute_geometry was not called
        if isinstance(self.domain, pp.MortarGrid):
            # This is a mortar grid. Assume that there are only cell dofs
            return self._g.num_cells * self._cells
        else:
            # We now know the domain is a grid by logic, make an assertion to appease mypy
            assert isinstance(self.domain, pp.Grid)
            return (
                self._g.num_cells * self._cells
                + self._g.num_faces * self._faces
                + self._g.num_nodes * self._nodes
            )

    def set_name(self, name: str) -> None:
        """
        Raises:
            RuntimeError: Variables must not be re-named once defined,
                since the name is used as an identifier.

        """
        raise RuntimeError("Cannot rename operators representing a variable.")


    def previous_timestep(self) -> Variable:
        """Return a representation of this variable on the previous time step.

        Returns:
            Variable: A representation of this variable, with self.prev_time=True.

        """
        ndof = {"cells": self._cells, "faces": self._faces, "nodes": self._nodes}
        return Variable(self.name, ndof, self.domain, previous_timestep=True)

    def previous_iteration(self) -> Variable:
        """Return a representation of this variable on the previous time iteration.

        Returns:
            Variable: A representation of this variable, with self.prev_iter=True.

        """
        ndof = {"cells": self._cells, "faces": self._faces, "nodes": self._nodes}
        return Variable(self.name, ndof, self.domain, previous_iteration=True) 

    def __repr__(self) -> str:
        s = (
            f"Variable {self.name}, id: {self.id}\n"
            f"Degrees of freedom: cells ({self._cells}), faces ({self._faces}), "
            f"nodes ({self._nodes})\n"
        )
        if self.prev_iter:
            s += "Evaluated at the previous iteration.\n"
        elif self.prev_time:
            s += "Evaluated at the previous time step.\n"

        return s


class MixedDimensionalVariable(Variable):
    """Ad representation of a collection of variables that individually live on separate
    subdomains or interfaces, but which it is useful to treat jointly.

    Conversion of the variables into numerical value should be done with respect to the
    state of an array; see :meth:`Operator.evaluate`.
    Therefore, the class does not implement :meth:`Operator.parse`.

    Parameters:
        variables: Variables to be merged. Must all have the same name.

    """

    def __init__(self, variables: list[Variable]) -> None:
        """Create a merged representation of variables."""

        ### PUBLIC

        self.sub_vars = variables
        """List of sub variables passed at instantiation, which are defined on a separate
        domain each.

        """

        self.id = next(Variable._ids)
        """ID counter. Used to identify variables during operator parsing."""

        self.prev_time: bool = False
        """Flag indicating if the variable represents the state at the previous time step."""

        self.prev_iter: bool = False
        """Flag indicating if the variable represents the state at the previous iteration."""

        ### PRIVATE

        # Flag to identify variables merged over no subdomains. This requires special treatment
        # in various parts of the code.
        # A use case is variables that are only defined on subdomains of codimension >= 1
        # (e.g., contact traction variable), assigned to a problem where the grid happened
        # not to have any fractures.
        self._no_variables = len(variables) == 0

        # Take the name from the first variable.
        if self._no_variables:
            self._name = "no_sub_variables"
        else:
            self._name = variables[0].name
            # Check that all variables have the same name.
            # We may release this in the future, but for now, we make it a requirement
            all_names = set(var.name for var in variables)
            assert len(all_names) <= 1

        # must be done since super not called here in init
        self._set_tree()

    @property
    def domain(self) -> list[GridLike]:
        """A tuple of all domains on which the atomic sub-variables are defined."""
        return [var.domain for var in self.sub_vars]

    def size(self) -> int:
        """Returns the total size of the merged variable by summing the sizes of sub-variables.
        """
        return sum([v.size() for v in self.sub_vars])

    def previous_timestep(self) -> MixedDimensionalVariable:
        """Return a representation of this merged variable on the previous time step.

        Returns:
            Variable: A representation of this variable, with self.prev_time=True.

        """
        new_subs = [var.previous_timestep() for var in self.sub_vars]
        new_var = MixedDimensionalVariable(new_subs)
        new_var.prev_time = True
        return new_var

    def previous_iteration(self) -> MixedDimensionalVariable:
        """Return a representation of this merged variable on the previous iteration.

        Returns:
            Variable: A representation of this variable, with self.prev_iter=True.

        """
        new_subs = [var.previous_iteration() for var in self.sub_vars]
        new_var = MixedDimensionalVariable(new_subs)
        new_var.prev_iter = True
        return new_var

    def copy(self) -> "MixedDimensionalVariable":
        # A shallow copy should be sufficient here; the attributes are not expected to
        # change.
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        if self._no_variables:
            return (
                "Mixed-dimensional variable defined on an empty list of "
                "subdomains or interfaces."
            )

        s = "Mixed-dimensional"
        s += (
            f" variable with name {self.name}, id {self.id}\n"
            f"Composed of {len(self.sub_vars)} variables\n"
            f"Total size: {self.size()}\n"
        )
        if self.prev_iter:
            s += "Evaluated at the previous iteration.\n"
        elif self.prev_time:
            s += "Evaluated at the previous time step.\n"

        return s


class Tree:
    """Simple implementation of a Tree class. Used to represent combinations of
    Ad operators.
    """

    # https://stackoverflow.com/questions/2358045/how-can-i-implement-a-tree-in-python
    def __init__(
        self,
        operation: Operation,
        children: Optional[Sequence[Union[Operator, Ad_array]]] = None,
    ):

        self.op = operation

        self.children: list[Union[Operator, Ad_array]] = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def add_child(self, node: Union[Operator, Ad_array]) -> None:
        #        assert isinstance(node, (Operator, "pp.ad.Operator"))
        self.children.append(node)
