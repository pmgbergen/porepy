""" Implementation of wrappers for Ad representations of several operators.
"""

from __future__ import annotations

import copy
from enum import Enum
from functools import reduce
from itertools import count
from typing import Any, Literal, Optional, Sequence, Union, overload

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.utils.porepy_types import GridLike, GridLikeSequence

from . import _ad_utils
from .forward_mode import AdArray, initAdArrays

__all__ = [
    "Operator",
    "SparseArray",
    "DenseArray",
    "TimeDependentDenseArray",
    "Scalar",
    "Variable",
    "MixedDimensionalVariable",
    "sum_operator_list",
]


def _get_shape(mat):
    """Get shape of a numpy.ndarray or the Jacobian of AdArray"""
    if isinstance(mat, AdArray):
        return mat.jac.shape
    else:
        return mat.shape


class Operator:
    """Parent class for all AD operators.

    Objects of this class are not meant to be initiated directly, rather the various
    subclasses should be used. Instances of this class will still be created when
    subclasses are combined by Operator.Operations.

    Contains a tree structure of child operators for the recursive forward evaluation.

    Provides overload functions for basic arithmetic operations.

    Parameters:
        name: Name of this operator. Used for string representations
        subdomains (optional): List of subdomains on which the operator is defined.
            Will be empty for operators not associated with any subdomains.
            Defaults to None (converted to empty list).
        interfaces (optional): List of interfaces in the mixed-dimensional grid on which
            the operator is defined. Will be empty for operators not associated with any
            interface. Defaults to None (converted to empty list).
        operation (optional): Arithmetic or other operation represented by this
            operator. Defaults to void operation.
        children (optional): List of children, other AD operators. Defaults to empty
            list.

    """

    class Operations(Enum):
        """Object representing all supported operations by the operator class.

        Used to construct the operator tree and identify Operator.Operations.

        """

        void = "void"
        add = "add"
        sub = "sub"
        mul = "mul"
        rmul = "rmul"
        matmul = "matmul"
        rmatmul = "rmatmul"
        div = "div"
        rdiv = "rdiv"
        evaluate = "evaluate"
        approximate = "approximate"
        secondary = "secondary"
        pow = "pow"
        rpow = "rpow"

    def __init__(
        self,
        name: Optional[str] = None,
        domains: Optional[GridLikeSequence] = None,
        operation: Optional[Operator.Operations] = None,
        children: Optional[Sequence[Operator]] = None,
    ) -> None:
        if domains is None:
            domains = []
        self._domains: GridLikeSequence = domains
        self._domain_type: Literal["subdomains", "interfaces", "boundary grids"]
        if all([isinstance(d, pp.Grid) for d in domains]):
            self._domain_type = "subdomains"
        elif all([isinstance(d, pp.MortarGrid) for d in domains]):
            self._domain_type = "interfaces"
        elif all([isinstance(d, pp.BoundaryGrid) for d in domains]):
            self._domain_type = "boundary grids"
        else:
            raise ValueError(
                "An operator must be associated with either"
                " interfaces, subdomains or boundary grids."
            )

        self.children: Sequence[Operator]
        """List of children, other AD operators.

        Will be empty if the operator is a leaf.
        """

        self.operation: Operator.Operations
        """Arithmetic or other operation represented by this operator.

        Will be void if the operator is a leaf.
        """

        self._initialize_children(operation=operation, children=children)

        ### PRIVATE
        self._name = name if name is not None else ""

    @property
    def interfaces(self):
        """List of interfaces on which the operator is defined, passed at instantiation.

        Will be empty for operators not associated with specific interfaces.

        """
        return self._domains if self._domain_type == "interfaces" else []

    @property
    def subdomains(self):
        """List of subdomains on which the operator is defined, passed at instantiation.

        Will be empty for operators not associated with specific subdomains.

        """
        return self._domains if self._domain_type == "subdomains" else []

    @property
    def domain_type(self) -> Literal["subdomains", "interfaces", "boundary grids"]:
        """Type of domains where the operator is defined."""
        return self._domain_type

    @property
    def domains(self) -> GridLikeSequence:
        """List of domains where the operator is defined."""
        return self._domains

    @property
    def name(self) -> str:
        """The name given to this variable."""
        return self._name

    def _initialize_children(
        self,
        operation: Optional[Operator.Operations] = None,
        children: Optional[Sequence[Operator]] = None,
    ):
        """This is a part of initialization which can be called separately since some
        subclasses do not call super().__init__()

        """
        self.children = [] if children is None else children
        self.operation = Operator.Operations.void if operation is None else operation

    def is_leaf(self) -> bool:
        """Check if this operator is a leaf in the tree-representation of an expression.

        Note that this implies that the method ``parse()`` is expected to be implemented.

        Returns:
            True if the operator has no children.

        """
        return len(self.children) == 0

    def set_name(self, name: str) -> None:
        """Reset this object's name originally passed at instantiation.

        Parameters:
            name: the new name to be assigned.

        """
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

            children = op.children

            if len(children) == 0:
                # We are on an atomic operator. If this is a time-dependent operator,
                # set it to be evaluated at the previous time step. If not, leave the
                # operator as it is.
                if isinstance(
                    op, (Variable, MixedDimensionalVariable, TimeDependentDenseArray)
                ):
                    # Use the previous_timestep() method of the operator to get the
                    # operator evaluated at the previous time step. This in effect
                    # creates a copy of the operator.
                    # If other time-dependent other operators are added, they will have
                    # to override this previous_timestep method.
                    return op.previous_timestep()

                else:
                    # No need to use a copy here.
                    # This also means that operators that are not time dependent need not
                    # override this previous_timestep method.
                    return op
            # Secondary expressions/operators have children, but also a prev time step
            elif isinstance(op, pp.composite.SecondaryOperator):
                return op.previous_timestep()
            else:
                # Recursively iterate over the subtree, get the children, evaluated at the
                # previous time when relevant, and add it to the new list.
                new_children: list[Operator] = list()
                for ci, child in enumerate(children):
                    # Recursive call to fix the subtree.
                    new_children.append(_traverse_tree(child))

                # Use the same lists of domains as in the old operator.
                domains = op.domains

                # Create new operator from the tree.
                new_op = Operator(
                    name=op.name,
                    domains=domains,
                    operation=op.operation,
                    children=new_children,
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
        of atomic operators; such operators should be evaluated by the method :meth:`evaluate`.

        Parameters:
            mdg: Mixed-dimensional grid on which this operator is to be parsed.

        Returns:
            A numerical format representing this operator;s values on given domain.

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
        #    stored state.
        # 2) If the operator is a leaf in the tree-representation of the operator,
        #    parsing is left to the operator itself.
        # 3) If the operator is formed by combining other operators lower in the tree,
        #    parsing is handled by first evaluating the children (leads to recursion)
        #    and then perform the operation on the result.

        # Check for case 1 or 2
        if isinstance(op, pp.ad.Variable) or isinstance(op, Variable):
            # Case 1: Variable

            # How to access the array of (Ad representation of) states depends on
            # whether this is a single or combined variable; see self.__init__,
            # definition of self._variable_ids.
            # TODO: no difference between merged or no mixed-dimensional variables!?
            if isinstance(op, pp.ad.MixedDimensionalVariable) or isinstance(
                op, MixedDimensionalVariable
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
        elif isinstance(op, pp.ad.AdArray):
            # When using nested operator functions, op can be an already evaluated term.
            # Just return it.
            return op

        elif op.is_leaf():
            # Case 2
            return op.parse(mdg)  # type:ignore

        # This is not an atomic operator. First parse its children, then combine them
        results = [self._parse_operator(child, mdg) for child in op.children]

        # Combine the results
        operation = op.operation
        if operation == Operator.Operations.add:
            # To add we need two objects
            assert len(results) == 2

            if isinstance(results[0], np.ndarray):
                # We should not do numpy_array + Ad_array, since numpy will interpret
                # this in a strange way. Instead switch the order of the operands and
                # everything will be fine.
                results = results[::-1]
            try:
                # An error here would typically be a dimension mismatch between the
                # involved operators.
                return results[0] + results[1]
            except ValueError as exc:
                msg = self._get_error_message("adding", op.children, results)
                raise ValueError(msg) from exc

        elif operation == Operator.Operations.sub:
            # To subtract we need two objects
            assert len(results) == 2

            # We need a minor trick to take care of numpy arrays.
            factor = 1.0
            if isinstance(results[0], np.ndarray):
                # We should not do numpy_array - Ad_array, since numpy will interpret
                # this in a strange way. Instead switch the order of the operands, and
                # switch the sign of factor to compensate.
                results = results[::-1]
                factor = -1.0
            try:
                # An error here would typically be a dimension mismatch between the
                # involved operators.
                return factor * (results[0] - results[1])
            except ValueError as exc:
                msg = self._get_error_message("subtracting", op.children, results)
                raise ValueError(msg) from exc

        elif operation == Operator.Operations.mul:
            # To multiply we need two objects
            assert len(results) == 2

            if isinstance(results[0], np.ndarray) and isinstance(
                results[1], (pp.ad.AdArray, pp.ad.forward_mode.AdArray)
            ):
                # In the implementation of multiplication between an AdArray and a
                # numpy array (in the forward mode Ad), a * b and b * a do not
                # commute. Flip the order of the results to get the expected behavior.
                # This is permissible, since the elementwise product commutes.
                results = results[::-1]
            try:
                # An error here would typically be a dimension mismatch between the
                # involved operators.
                return results[0] * results[1]
            except ValueError as exc:
                msg = self._get_error_message("multiplying", op.children, results)
                raise ValueError(msg) from exc

        elif operation == Operator.Operations.div:
            # Some care is needed here, to account for cases where item in the results
            # array is a numpy array
            try:
                if isinstance(results[0], np.ndarray) and isinstance(
                    results[1], (pp.ad.AdArray, pp.ad.forward_mode.AdArray)
                ):
                    # If numpy's __truediv__ method is called here, the result will be
                    # strange because of how numpy works. Instead we directly invoke the
                    # right-truedivide method in the AdArary.
                    return results[1].__rtruediv__(results[0])
                else:
                    return results[0] / results[1]
            except ValueError as exc:
                msg = self._get_error_message("dividing", op.children, results)
                raise ValueError(msg) from exc

        elif operation == Operator.Operations.pow:
            try:
                if isinstance(results[0], np.ndarray) and isinstance(
                    results[1], (pp.ad.AdArray, pp.ad.forward_mode.AdArray)
                ):
                    # If numpy's __pow__ method is called here, the result will be
                    # strange because of how numpy works. Instead we directly invoke the
                    # right-power method in the AdArary.
                    return results[1].__rpow__(results[0])
                else:
                    return results[0] ** results[1]
            except ValueError as exc:
                msg = self._get_error_message(
                    "raising to a power", op.children, results
                )
                raise ValueError(msg) from exc

        elif operation == Operator.Operations.matmul:
            try:
                if isinstance(results[0], np.ndarray) and isinstance(
                    results[1], (pp.ad.AdArray, pp.ad.forward_mode.AdArray)
                ):
                    # Again, we do not want to call numpy's matmul method, but instead
                    # directly invoke AdArarray's right matmul.
                    return results[1].__rmatmul__(results[0])
                # elif isinstance(results[1], np.ndarray) and isinstance(
                #     results[0], (pp.ad.AdArray, pp.ad.forward_mode.AdArray)
                # ):
                #     # Again, we do not want to call numpy's matmul method, but instead
                #     # directly invoke AdArarray's right matmul.
                #     return results[0].__rmatmul__(results[1])
                else:
                    return results[0] @ results[1]
            except ValueError as exc:
                msg = self._get_error_message(
                    "matrix multiplying", op.children, results
                )
                raise ValueError(msg) from exc

        elif operation == Operator.Operations.secondary:
            try:
                return op.func(*results)
            except (ValueError, AssertionError) as exc:
                msg = (
                    f"Failed to evaluate secondary operator"
                    + f" {op.name}{[c.name for c in op.children]}:\n"
                )
                raise ValueError(msg) from exc

        elif operation == Operator.Operations.evaluate:
            # This is a function, which should have at least one argument
            assert len(results) > 1
            func_op = results[0]

            # if the callable can be fed with AdArrays, do it
            if func_op.ad_compatible:
                return func_op.func(*results[1:])
            else:
                # This should be a Function with approximated Jacobian and value.
                if all(isinstance(r, np.ndarray) for r in results[1:]):
                    try:
                        val = func_op.get_values(*results[1:])
                    except Exception as exc:
                        # TODO specify what can go wrong here (Exception type)
                        msg = "Ad parsing: Error evaluating operator function:\n"
                        msg += func_op._parse_readable()
                    return val
                else:
                    try:
                        val = func_op.get_values(*results[1:])
                        jac = func_op.get_jacobian(*results[1:])
                    except Exception as exc:
                        # TODO specify what can go wrong here (Exception type)
                        msg = "Ad parsing: Error evaluating operator function:\n"
                        msg += func_op._parse_readable()
                    return AdArray(val, jac)

        else:
            raise ValueError(f"Encountered unknown operation {operation}")

    def _get_error_message(
        self, operation: str, children: Sequence[Operator], results: list
    ) -> str:
        # Helper function to format error message
        msg_0 = children[0]._parse_readable()
        msg_1 = children[1]._parse_readable()

        nl = "\n"
        msg = f"Ad parsing: Error when {operation}\n\n"
        # First give name information. If the expression under evaluation is c = a + b,
        # the below code refers to c as the intended result, and a and b as the first
        # and second argument, respectively.
        msg += "Information on names given to the operators involved: \n"
        if len(self.name) > 0:
            msg += f"Name of the intended result: {self.name}\n"
        else:
            msg += "The intended result is not named\n"
        if len(children[0].name) > 0:
            msg += f"Name of the first argument: {children[0].name}\n"
        else:
            msg += "The first argument is not named\n"
        if len(children[1].name) > 0:
            msg += f"Name of the second argument: {children[1].name}\n"
        else:
            msg += "The second argument is not named\n"
        msg += nl

        # Information on how the terms a and b are defined
        msg += "The first argument represents the expression:\n " + msg_0 + nl + nl
        msg += "The second argument represents the expression:\n " + msg_1 + nl

        # Finally some information on sizes
        if isinstance(results[0], sps.spmatrix):
            msg += f"First argument is a sparse matrix of size {results[0].shape}\n"
        elif isinstance(results[0], pp.ad.AdArray):
            msg += (
                f"First argument is an AdArray of size {results[0].val.size} "
                f" and Jacobian of shape  {results[0].jac.shape} \n"
            )
        elif isinstance(results[0], np.ndarray):
            msg += f"First argument is a numpy array of size {results[0].size}\n"

        if isinstance(results[1], sps.spmatrix):
            msg += f"Second argument is a sparse matrix of size {results[1].shape}\n"
        elif isinstance(results[1], pp.ad.AdArray):
            msg += (
                f"Second argument is an AdArray of size {results[1].val.size} "
                f" and Jacobian of shape  {results[1].jac.shape} \n"
            )
        elif isinstance(results[1], np.ndarray):
            msg += f"Second argument is a numpy array of size {results[1].size}\n"

        msg += nl
        msg += "Note that a size mismatch may be caused by an error in the definition\n"
        msg += "of the intended result, or in the definition of one of the arguments."
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
        child_str = [child._parse_readable() for child in self.children]

        is_func = False
        operator_str = None

        # readable representations of known operations
        op = self.operation
        if op == Operator.Operations.add:
            operator_str = "+"
        elif op == Operator.Operations.sub:
            operator_str = "-"
        elif op == Operator.Operations.mul:
            operator_str = "*"
        elif op == Operator.Operations.matmul:
            operator_str = "@"
        elif op == Operator.Operations.div:
            operator_str = "/"
        elif op == Operator.Operations.pow:
            operator_str = "**"

        # function evaluations have their own readable representation
        elif op == Operator.Operations.evaluate:
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

    def viz(self):
        """Draws a visualization of the operator tree that has this operator as its root."""
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.Graph()

        def parse_subgraph(node: Operator):
            G.add_node(node)
            if len(node.children) == 0:
                return
            operation = node.operation
            G.add_node(operation)
            G.add_edge(node, operation)
            for child in node.children:
                parse_subgraph(child)
                G.add_edge(child, operation)

        parse_subgraph(self)
        nx.draw(G, with_labels=True)
        plt.show()

    ### Operator discretization --------------------------------------------------------
    # TODO this is specific to discretizations and should not be done here
    # let the EquationSystem do this by calling respective util methods

    def discretize(self, mdg: pp.MixedDimensionalGrid) -> None:
        """Perform discretization operation on all discretizations identified in
        the tree of this operator, using data from mdg.

        IMPLEMENTATION NOTE: The discretizations was identified at initialization of
        Expression - it is now done here to accommodate updates (?) and

        """
        unique_discretizations: dict[pp.discretization_type, list[GridLike]] = (
            self._identify_discretizations()
        )
        _ad_utils.discretize_from_list(unique_discretizations, mdg)

    def _identify_discretizations(
        self,
    ) -> dict[pp.discretization_type, list[GridLike]]:
        """Perform a recursive search to find all discretizations present in the
        operator tree. Uniquify the list to avoid double computations.

        """
        all_discr = self._identify_subtree_discretizations([])
        return _ad_utils.uniquify_discretization_list(all_discr)

    def _identify_subtree_discretizations(self, discr: list) -> list:
        """Recursive search in the tree of this operator to identify all discretizations
        represented in the operator.
        """
        if len(self.children) > 0:
            # Go further in recursion
            for child in self.children:
                discr += child._identify_subtree_discretizations([])

        if isinstance(self, _ad_utils.MergedOperator):
            # We have reached the bottom; this is a discretization (example: mpfa.flux)
            discr.append(self)

        return discr

    ### Operator parsing ---------------------------------------------------------------

    def value(
        self, system_manager: pp.ad.EquationSystem, state: Optional[np.ndarray] = None
    ) -> pp.number | np.ndarray | sps.spmatrix:
        """Evaluate the residual for a given solution.

        Parameters:
            system_manager: Used to represent the problem. Will be used to parse the
                sub-operators that combine to form this operator.
            state (optional): Solution vector for which the residual and its derivatives
                should be formed. If not provided, the solution will be pulled from the
                previous iterate (if this exists), or alternatively from the solution at
                the previous time step.

        Returns:
            A representation of the residual in form of a number, numpy array or sparse
            matrix.

        """
        return self._evaluate(system_manager, state=state, evaluate_jacobian=False)

    def value_and_jacobian(
        self, system_manager: pp.ad.EquationSystem, state: Optional[np.ndarray] = None
    ) -> AdArray:
        """Evaluate the residual and Jacobian matrix for a given solution.

        Parameters:
            system_manager: Used to represent the problem. Will be used to parse the
                sub-operators that combine to form this operator.
            state (optional): Solution vector for which the residual and its derivatives
                should be formed. If not provided, the solution will be pulled from the
                previous iterate (if this exists), or alternatively from the solution at
                the previous time step.

        Returns:
            A representation of the residual and Jacobian in form of an AD Array.
            Note that the Jacobian matrix need not be invertible, or even square;
            this depends on the operator.

        """
        ad = self._evaluate(system_manager, state=state, evaluate_jacobian=True)

        # Casting the result to AdArray or raising an error.
        # It's better to set pp.number here, but isinstance requires a tuple, not Union.
        # This should be reconsidered when pp.number is replaced with numbers.Real
        if isinstance(ad, (int, float)):
            # AdArray requires 1D numpy array as value, not a scalar.
            ad = np.array([ad])

        if isinstance(ad, np.ndarray) and len(ad.shape) == 1:
            return AdArray(ad, sps.csr_matrix((ad.shape[0], system_manager.num_dofs())))
        elif isinstance(ad, (sps.spmatrix, np.ndarray)):
            # this case coverse both, dense and sparse matrices returned from
            # discretizations f.e.
            raise NotImplementedError(
                f"The Jacobian of {type(ad)} is not implemented because it is "
                "multidimensional"
            )
        else:
            return ad

    def evaluate(
        self,
        system_manager: pp.ad.EquationSystem,
        state: Optional[np.ndarray] = None,
    ):
        raise ValueError(
            "`evaluate` is deprecated. Use `value` or `value_and_jacobian` instead."
        )

    def _evaluate(
        self,
        system_manager: pp.ad.EquationSystem,
        state: Optional[np.ndarray] = None,
        evaluate_jacobian: bool = True,
    ) -> pp.number | np.ndarray | sps.spmatrix | AdArray:
        """Evaluate the residual and Jacobian matrix for a given solution.

        Parameters:
            system_manager: Used to represent the problem. Will be used to parse the
                sub-operators that combine to form this operator.
            state (optional): Solution vector for which the residual and its derivatives
                should be formed. If not provided, the solution will be pulled from the
                previous iterate (if this exists), or alternatively from the solution at
                the previous time step.

        Returns:
            A representation of the residual and Jacobian in form of an AD Array.
            Note that the Jacobian matrix need not be invertible, or even square; this
            depends on the operator.

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

        prev_vals = system_manager.get_variable_values(time_step_index=0)
        prev_iter_vals = system_manager.get_variable_values(iterate_index=0)

        if state is None:
            state = system_manager.get_variable_values(iterate_index=0)

        # Initialize Ad variables with the current iterates

        # The size of the Jacobian matrix will always be set according to the
        # variables found by the EquationSystem.

        # NOTE: This implies that to derive a subsystem from the Jacobian
        # matrix of this Operator will require restricting the columns of
        # this matrix.

        # First generate an Ad array (ready for forward Ad) for the full set.
        # If the Jacobian is not requested, this step is skipped.
        vars: AdArray | np.ndarray
        if evaluate_jacobian:
            vars = initAdArrays([state])[0]
        else:
            vars = state

        # Next, the Ad array must be split into variables of the right size
        # (splitting impacts values and number of rows in the Jacobian, but
        # the Jacobian columns must stay the same to preserve all cross couplings
        # in the derivatives).

        # Dictionary which maps from Ad variable ids to AdArray.
        self._ad: dict[int, AdArray] = {}

        # Loop over all variables, restrict to an Ad array corresponding to
        # this variable.
        for var_id, dof in zip(self._variable_ids, self._variable_dofs):
            ncol = state.size
            nrow = np.unique(dof).size
            # Restriction matrix from full state (in Forward Ad) to the specific
            # variable.
            R = sps.coo_matrix(
                (np.ones(nrow), (np.arange(nrow), dof)), shape=(nrow, ncol)
            ).tocsr()
            self._ad[var_id] = R @ vars

        # Also make mappings from the previous iteration.
        # This is simpler, since it is only a matter of getting the residual vector
        # correctly (not Jacobian matrix).

        prev_iter_vals_list = [prev_iter_vals[ind] for ind in self._prev_iter_dofs]
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

    def _identify_variables(
        self,
        system_manager: pp.ad.EquationSystem,
        var: Optional[list] = None,
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

        # 2. Get a mapping between variables (*not* only MixedDimensionalVariables) and
        # their indices according to the EquationSystem. This is needed to access the
        # state of a variable when parsing the operator to numerical values using
        # forward Ad.

        # For each variable, get the global index
        inds = []
        variable_ids = []
        prev_time = []
        prev_iter = []
        for variable in variables:
            # Indices (in EquationSystem sense) of this variable. Will be built
            # gradually for MixedDimensionalVariables, in one go for plain Variables.
            ind_var = []
            prev_time.append(variable.prev_time)
            prev_iter.append(variable.prev_iter)

            if isinstance(variable, MixedDimensionalVariable):
                # Is this equivalent to the test in previous function?
                # Loop over all subvariables for the mixed-dimensional variable
                for i, sub_var in enumerate(variable.sub_vars):
                    if sub_var.prev_time or sub_var.prev_iter:
                        # If this is a variable representing a previous time step or
                        # iteration, we need to use the original variable to get hold of
                        # the correct dof indices, since this is the variable that was
                        # created by the EquationSystem. However, we will tie the
                        # indices to the id of this variable, since this is the one that
                        # will be used for lookup later on.
                        sub_var_known_to_eq_system: Variable = sub_var.original_variable
                    else:
                        sub_var_known_to_eq_system = sub_var

                    # Get the index of this sub variable in the global numbering of the
                    # EquationSystem. If an error message is raised that the variable is
                    # not present in the EquationSystem, it is likely that this operator
                    # contains a variable that is not known to the EquationSystem (it
                    # has not passed through EquationSystem.create_variable()).
                    ind_var.append(system_manager.dofs_of([sub_var_known_to_eq_system]))
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
                if variable.prev_iter or variable.prev_time:
                    # If this is a variable representing a previous time step or
                    # iteration, we need to use the original variable to get hold of
                    # the correct dof indices, since this is the variable that was
                    # created by the EquationSystem. However, we will tie the
                    # indices to the id of this variable, since this is the one that
                    # will be used for lookup later on.
                    variable_known_to_eq_system = variable.original_variable
                else:
                    variable_known_to_eq_system = variable

                ind_var.append(system_manager.dofs_of([variable_known_to_eq_system]))
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
            sub_variables: list[Variable] = []
            # When using nested pp.ad.Functions, some of the children may be AdArrays
            # (forward mode), rather than Operators. For the former, don't look for
            # children - they have none.
            for child in self.children:
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

    ### Special methods ----------------------------------------------------------------

    def __str__(self) -> str:
        return self._name if self._name is not None else ""

    def __repr__(self) -> str:
        if self._name is None or len(self._name) == 0:
            s = "Operator with no name"
        else:
            s = f"Operator '{self._name}'"
        s += f" formed by {self.operation} with {len(self.children)} children."
        return s

    def __neg__(self) -> Operator:
        """Unary minus operation.

        Returns:
            Operator: The negative of the operator.

        """
        return pp.ad.Scalar(-1) * self

    def __add__(self, other: Operator) -> Operator:
        """Add two operators.

        Parameters:
            other: The operator to add to self.

        Returns:
            The sum of self and other.

        """
        children = self._parse_other(other)
        return Operator(
            children=children, operation=Operator.Operations.add, name="+ operator"
        )

    def __radd__(self, other: Operator) -> Operator:
        """Add two operators.

        This is the reverse addition operator, i.e., it is called when self is on the
        right hand side of the addition operator.

        Parameters:
            other: The operator to add to self.

        Returns:
            The sum of self and other.

        """
        return self.__add__(other)

    def __sub__(self, other: Operator) -> Operator:
        """Subtract two operators.

        Parameters:
            other: The operator to subtract from self.

        Returns:
            The difference of self and other.

        """
        children = self._parse_other(other)
        return Operator(
            children=children, operation=Operator.Operations.sub, name="- operator"
        )

    def __rsub__(self, other: Operator) -> Operator:
        """Subtract two operators.

        Parameters:
            other: An operator which should be subtracted by self.

        Returns:
            The difference of other and self.

        """
        # consider the expression a-b. right-subtraction means self == b
        children = self._parse_other(other)
        # we need to change the order here since a-b != b-a
        children = [children[1], children[0]]
        return Operator(
            children=children, operation=Operator.Operations.sub, name="- operator"
        )

    def __mul__(self, other: Operator) -> Operator:
        """Elementwise multiplication of two operators.

        Parameters:
            other: The operator to multiply with self.

        Returns:
            The elementwise product of self and other.

        """
        children = self._parse_other(other)
        return Operator(
            children=children, operation=Operator.Operations.mul, name="* operator"
        )

    def __rmul__(self, other: Operator) -> Operator:
        """Elementwise multiplication of two operators.

        This is the reverse multiplication operator, i.e., it is called when self is on
        the right hand side of the multiplication operator.

        Parameters:
            other: The operator to multiply with self.

        Returns:
            The elementwise product of self and other.

        """
        children = self._parse_other(other)
        return Operator(
            children=children,
            operation=Operator.Operations.rmul,
            name="right * operator",
        )

    def __truediv__(self, other: Operator) -> Operator:
        """Elementwise division of two operators.

        Parameters:
            other: The operator to divide self with.

        Returns:
            The elementwise division of self and other.

        """
        children = self._parse_other(other)
        return Operator(
            children=children, operation=Operator.Operations.div, name="/ operator"
        )

    def __rtruediv__(self, other: Operator) -> Operator:
        """Elementwise division of two operators.

        This is the reverse division operator, i.e., it is called when self is on
        the right hand side of the division operator.

        Parameters:
            other: The operator to be divided by self.

        Returns:
            The elementwise division of other and self.

        """
        children = self._parse_other(other)
        return Operator(
            children=children,
            operation=Operator.Operations.rdiv,
            name="right / operator",
        )

    def __pow__(self, other: Operator) -> Operator:
        """Elementwise exponentiation of two operators.

        Parameters:
            other: The operator to exponentiate self with.

        Raises:
            ValueError: If self is a SparseArray and other is a Scalar or a DenseArray.

        Returns:
            The elementwise exponentiation of self and other.

        """
        if isinstance(self, pp.ad.SparseArray) and isinstance(other, pp.ad.Scalar):
            # Special case: Scipy sparse matrices only accepts integers as exponents,
            # but we cannot know if the exponent is an integer or not, so we need to
            # disallow this case. Implementation detail: It turns out that if the scalar
            # can be represented as an integer (say, it is 2.0), Scipy may or may not do
            # the cast and go on with the calculation. It semes the behavior depends on
            # the Python and Scipy installation (potentially on which operating system
            # is used). Thus in this case, we cannot rely on the external library
            # (SciPy) to give a consistent treatment of this operation, and instead
            # raise an error here. This breaks with the general philosophy of ad
            # Operators, that when combining two externally provided objects (Scalars,
            # DenseArray, SparseArray), the external library should be responsible for
            # the calculation, but this seems like the least bad option.
            raise ValueError("Cannot take SparseArray to the power of a Scalar.")
        elif isinstance(self, pp.ad.SparseArray) and isinstance(
            other, pp.ad.DenseArray
        ):
            # When parsing this case, one of the operators (likely the numpy array) will
            # apply broadcasting, to produce a list of sparse matrices containing the
            # matrix raised to the power of of the array elements (provided these are
            # integers, the same problem as above applies). We explicitly disallow this.
            raise ValueError("Cannot take SparseArray to the power of an DenseArray.")

        children = self._parse_other(other)
        return Operator(
            children=children, operation=Operator.Operations.pow, name="** operator"
        )

    def __rpow__(self, other: Operator) -> Operator:
        """Elementwise exponentiation of two operators.

        This is the reverse exponentiation operator, i.e., it is called when self is on
        the right hand side of the exponentiation operator.

        Parameters:
            other: The operator that should be raised to the power of self.

        Returns:
            The elementwise exponentiation of other and self.

        """
        children = self._parse_other(other)
        return Operator(
            children=children,
            operation=Operator.Operations.rpow,
            name="reverse ** operator",
        )

    def __matmul__(self, other: Operator) -> Operator:
        """Matrix multiplication of two operators.

        Parameters:
            other: The operator to right-multiply with self.

        Returns:
            The matrix product of self and other.

        """
        children = self._parse_other(other)
        return Operator(
            children=children, operation=Operator.Operations.matmul, name="@ operator"
        )

    def __rmatmul__(self, other):
        """Matrix multiplication of two operators.

        This is the reverse matrix multiplication operator, i.e., it is called when self
        is on the right hand side of the matrix multiplication operator.

        Parameters:
            other: The operator to left-multiply with self.

        Returns:
            The matrix product of other and self.

        """
        children = self._parse_other(other)
        return Operator(
            children=children,
            operation=Operator.Operations.rmatmul,
            name="reverse @ operator",
        )

    def _parse_other(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return [self, Scalar(other)]
        elif isinstance(other, np.ndarray):
            return [self, DenseArray(other)]
        elif isinstance(other, sps.spmatrix):
            return [self, SparseArray(other)]
        elif isinstance(other, Operator):
            return [self, other]
        elif isinstance(other, AdArray):
            # This may happen when using nested pp.ad.Function.
            return [self, other]
        else:
            raise ValueError(f"Cannot parse {other} as an AD operator")


class SparseArray(Operator):
    """Ad representation of a sparse matrix.

    For dense matrices, use :class:`DenseArray` instead.

    This is a shallow wrapper around the real matrix; it is needed to combine the matrix
    with other types of Ad objects.

    Parameters:
        mat: Sparse matrix to be wrapped as an AD operator.
        name: Name of this operator

    """

    def __init__(self, mat: sps.spmatrix, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self._mat = mat
        # Force the data to be float, so that we limit the number of combinations of
        # data types that we need to consider in parsing.
        self._mat.data = self._mat.data.astype(float)
        self._shape = mat.shape
        """Shape of the wrapped matrix."""

    def __repr__(self) -> str:
        return f"Matrix with shape {self._mat.shape} and {self._mat.data.size} elements"

    def __str__(self) -> str:
        s = "Matrix "
        if self._name is not None:
            s += self._name
        return s

    def __neg__(self) -> SparseArray:
        """We override :meth:`Operator.__neg__` to prevent constructing a composite
        operator from just a sparse array.

        Returns:
            Operator: The negative of the operator.

        """
        new_name = None if self.name is None else f"minus {self.name}"
        return SparseArray(mat=-self._mat, name=new_name)

    def parse(self, mdg: pp.MixedDimensionalGrid) -> sps.spmatrix:
        """See :meth:`Operator.parse`.

        Returns:
            The wrapped matrix.

        """
        return self._mat

    def transpose(self) -> "SparseArray":
        """Returns an AD operator representing the transposed matrix."""
        return SparseArray(self._mat.transpose())

    @property
    def T(self) -> "SparseArray":
        """Shorthand for transpose."""
        return self.transpose()

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the wrapped matrix."""
        return self._shape


class DenseArray(Operator):
    """AD representation of a constant numpy array.

    For sparse matrices, use :class:`SparseArray` instead.
    For time-dependent arrays see :class:`TimeDependentArray`.

    This is a shallow wrapper around the real array; it is needed to combine the array
    with other types of AD operators.

    Parameters:
        values: Numpy array to be represented.

    """

    def __init__(self, values: np.ndarray, name: Optional[str] = None) -> None:
        """Construct an Ad representation of a numpy array.

        Parameters:
            values: Numpy array to be represented.

        """
        super().__init__(name=name)
        # Force the data to be float, so that we limit the number of combinations of
        # data types that we need to consider in parsing.
        self._values = values.astype(float, copy=False)

    def __repr__(self) -> str:
        return f"Wrapped numpy array of size {self._values.size}."

    def __str__(self) -> str:
        s = "Array"
        if self._name is not None:
            s += f"({self._name})"
        return s

    def __neg__(self) -> DenseArray:
        """We override :meth:`Operator.__neg__` to prevent constructing a composite
        operator from just an array.

        Returns:
            Operator: The negative of the operator.

        """
        new_name = None if self.name is None else f"minus {self.name}"
        return DenseArray(values=-self._values, name=new_name)

    @property
    def size(self) -> int:
        """Number of elements in the wrapped array."""
        return self._values.size

    def parse(self, mdg: pp.MixedDimensionalGrid) -> np.ndarray:
        """See :meth:`Operator.parse`.

        Returns:
            The wrapped array.

        """
        return self._values


class TimeDependentDenseArray(Operator):
    """An Ad-wrapper around a time-dependent numpy array.

    The array is tied to a MixedDimensionalGrid, and is distributed among the data
    dictionaries associated with subdomains and interfaces.
    The array values are stored
    in ``data[pp.ITERATE_SOLUTIONS][self._name][0]`` for the current time and
    ``data[pp.TIME_STEP_SOLUTIONS][self._name][0]`` for the previous time.

    The array can be differentiated in time using ``pp.ad.dt()``.

    The intended use is to represent time-varying quantities in equations, e.g., source
    terms. Future use will also include numerical values of boundary conditions,
    however, this is pending an update to the model classes.

    Parameters:
        name: Name of the variable. Should correspond to items in
            ``data[pp.TIME_STEP_SOLUTIONS]``.
        subdomains: Subdomains on which the array is defined. Defaults to None.
        interfaces: Interfaces on which the array is defined. Defaults to None.
            Exactly one of subdomains and interfaces must be non-empty.
        previous_timestep: Flag indicating if the array should be evaluated at the
            previous time step.

    Attributes:
        previous_timestep: If True, the array will be evaluated using
            ``data[pp.TIME_STEP_SOLUTIONS]`` (data being the data dictionaries for
            subdomains and interfaces), if False, ``data[pp.ITERATE_SOLUTIONS]`` is used.

    Raises:
        ValueError: If either none of, or both of, subdomains and interfaces are empty.

    """

    def __init__(
        self,
        name: str,
        domains: GridLikeSequence,
        previous_timestep: bool = False,
    ):
        self.prev_time: bool = previous_timestep
        """If True, the array will be evaluated using ``data[pp.TIME_STEP_SOLUTIONS]``
        (data being the data dictionaries for subdomains and interfaces).

        If False, ``data[pp.ITERATE_SOLUTIONS]`` is used.

        """

        super().__init__(name=name, domains=domains)

    def previous_timestep(self) -> TimeDependentDenseArray:
        """
        Returns:
            This array represented at the previous time step.

        """
        return TimeDependentDenseArray(
            name=self._name, domains=self._domains, previous_timestep=True
        )

    def parse(self, mdg: pp.MixedDimensionalGrid) -> np.ndarray:
        """Convert this array into numerical values.

        The numerical values will be picked from the representation of the array in
        ``data[pp.ITERATE_SOLUTIONS]`` (where data is the data dictionary of the
        subdomains
        or interfaces of this Array), or, if ``self.prev_time = True``,
        from ``data[pp.TIME_STEP_SOLUTIONS]``.

        Parameters:
            mdg: Mixed-dimensional grid.

        Returns:
            A numpy ndarray containing the numerical values of this array.

        """
        vals = []
        for g in self._domains:
            if self._domain_type == "subdomains":
                assert isinstance(g, pp.Grid)
                data = mdg.subdomain_data(g)
            elif self._domain_type == "interfaces":
                assert isinstance(g, pp.MortarGrid)
                data = mdg.interface_data(g)
            elif self._domain_type == "boundary grids":
                assert isinstance(g, pp.BoundaryGrid)
                data = mdg.boundary_grid_data(g)
            else:
                raise ValueError(f"Unknown grid type: {self._domain_type}.")
            if self.prev_time:
                vals.append(
                    pp.get_solution_values(
                        name=self._name, data=data, time_step_index=0
                    )
                )
            else:
                vals.append(
                    pp.get_solution_values(name=self._name, data=data, iterate_index=0)
                )

        if len(vals) > 0:
            # Normal case: concatenate the values from all grids
            return np.hstack((vals))
        else:
            # Special case: No grids. Return an empty array.
            return np.empty(0, dtype=float)

    def __repr__(self) -> str:
        return (
            f"Wrapped time-dependent array with name {self._name}.\n"
            f"Defined on {len(self._domains)} {self._domain_type}.\n"
        )


class Scalar(Operator):
    """Ad representation of a scalar.

    This is a shallow wrapper around a real scalar. It may be useful to combine
    the scalar with other types of Ad objects.

    NOTE: Since this is a wrapper around a Python immutable, copying a Scalar will
    effectively create a deep copy, i.e., changes in the value of one Scalar will not
    be reflected in the other. This is in contrast to the behavior of the other
    Ad objects.

    """

    def __init__(self, value: float, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        # Force the data to be float, so that we limit the number of combinations of
        # data types that we need to consider in parsing.
        self._value = float(value)

    def __repr__(self) -> str:
        return f"Wrapped scalar with value {self._value}"

    def __str__(self) -> str:
        s = "Scalar"
        if self._name is not None:
            s += f"({self._name})"
        return s

    def __neg__(self) -> Scalar:
        """We override :meth:`Operator.__neg__` to prevent constructing a composite
        operator from just a scalar.

        Returns:
            Operator: The negative of the operator.

        """
        new_name = None if self.name is None else f"minus {self.name}"
        return Scalar(value=-self._value, name=new_name)

    def parse(self, mdg: pp.MixedDimensionalGrid) -> float:
        """See :meth:`Operator.parse`.

        Returns:
            The wrapped number.

        """
        return self._value

    def set_value(self, value: float) -> None:
        """Set the value of this scalar.

        Usage includes changing the value of the scalar, as needed when using dynamic
        time stepping.

        Parameters:
            value: The new value.

        """
        self._value = value


class Variable(Operator):
    """AD operator representing a variable defined on a single grid or mortar grid.

    For combinations of variables on different subdomains, see :class:`MergedVariable`.

    Conversion of the variable into numerical value should be done with respect to the
    state of an array; see :meth:`Operator.evaluate`. Therefore, the variable does not
    implement the method :meth:`Operator.parse`.

    A variable is associated with either a grid or an interface. Therefore it is assumed
    that either ``subdomains`` or ``interfaces`` is passed as an argument.

    Parameters:
        name: Variable name.
        ndof: Number of dofs per grid element.
            Valid keys are ``cells``, ``faces`` and ``nodes``.
        subdomains (length=1): List containing a single grid.
        interfaces (length=1): List containing a single mortar grid.
        num_cells: Number of cells in the grid.
            Only relevant if this is an interface variable.

    """

    # Identifiers for variables. This will assign a unique id to all instances of this
    # class. This is used when operators are parsed to the forward Ad format. The
    # value of the id has no particular meaning.
    _ids = count(0)

    def __init__(
        self,
        name: str,
        ndof: dict[Literal["cells", "faces", "nodes"], int],
        domain: GridLike,
        tags: Optional[dict[str, Any]] = None,
        previous_timestep: bool = False,
        previous_iteration: bool = False,
    ) -> None:
        # Block a mypy warning here: Domain is known to be GridLike (grid, mortar grid,
        # or boundary grid), thus the below wrapping in a list gives a list of GridLike,
        # but the super constructor expects a sequence of grids, sequence or mortar
        # grids etc. Mypy makes a difference, but the additional entropy needed to
        # circumvent the warning is not worth it.
        super().__init__(name=name, domains=[domain])  # type: ignore [arg-type]

        ### PUBLIC

        self.prev_time: bool = previous_timestep
        """Flag indicating if the variable represents the state at the previous time
        step.

        """

        self.prev_iter: bool = previous_iteration
        """Flag indicating if the variable represents the state at the previous
        iteration.

        """

        self.id: int = next(Variable._ids)
        """ID counter. Used to identify variables during operator parsing."""

        self.original_variable: Variable
        """The original variable, if this variable is a copy of another variable.

        This attribute is used by the methods :meth:`Variable.previous_timestep` and
        :meth:`Variable.previous_iteration` to keep a link to the original variable.
        """

        if self._domain_type == "boundary grids":
            raise NotImplementedError("Variables on boundaries are not supported.")

        ### PRIVATE
        # domain
        self._g: GridLike = domain
        # dofs per
        self._cells: int = ndof.get("cells", 0)
        self._faces: int = ndof.get("faces", 0)
        self._nodes: int = ndof.get("nodes", 0)

        # tag
        self._tags: dict[str, Any] = tags if tags is not None else {}

    @property
    def domain(self) -> GridLike:
        """The grid or mortar grid on which this variable is defined."""
        return self._g

    @property
    def tags(self) -> dict[str, Any]:
        """A dictionary of tags associated with this variable."""
        return self._tags

    @property
    def size(self) -> int:
        """Returns the total number of dofs this variable has."""
        if isinstance(self.domain, pp.MortarGrid):
            # This is a mortar grid. Assume that there are only cell dofs
            return self.domain.num_cells * self._cells
        if isinstance(self.domain, pp.Grid):
            return (
                self.domain.num_cells * self._cells
                + self.domain.num_faces * self._faces
                + self.domain.num_nodes * self._nodes
            )
        raise ValueError()

    def set_name(self, name: str) -> None:
        """
        Raises:
            RuntimeError: Variables must not be re-named once defined,
                since the name is used as an identifier.

        """
        raise RuntimeError("Cannot rename operators representing a variable.")

    def previous_timestep(self) -> Variable:
        """Return a representation of this variable on the previous time step.

        Raises:
            ValueError:
                If the variable is a representation of the previous iteration,
                previously set by :meth:`~previous_iteration`.

            NotImplementedError:
                If the variable is already a representation of the previous time step.
                Currently, we support creating only one previous time step.

        Returns:
            A representation of this variable at the previous time step,
            with its ``prev_time`` attribute set to ``True``.

        """
        if self.prev_time:
            raise NotImplementedError(
                "Currently, it is not supported to create a variable that represents "
                "more than one time step behind."
            )

        if self.prev_iter:
            raise ValueError(
                "Cannot create a variable both on the previous time step and "
                "previous iteration."
            )

        ndof: dict[Literal["cells", "faces", "nodes"], int] = {
            "cells": self._cells,
            "faces": self._faces,
            "nodes": self._nodes,
        }
        new_var = Variable(self.name, ndof, self.domain, previous_timestep=True)
        # Assign self as the original variable.
        new_var.original_variable = self
        return new_var

    def previous_iteration(self) -> Variable:
        """Return a representation of this mixed-dimensional variable on the previous
        iteration.

        Raises:
            ValueError:
                If the variable is a representation of the previous time step,
                previously set by :meth:`~previous_timestep`.

            NotImplementedError:
                If the variable is already a representation of the previous time
                iteration. Currently, we support creating only one previous iteration.

        Returns:
            A representation of this variable on the previous time iteration,
            with its ``prev_iter`` attribute set to ``True``.

        """
        if self.prev_time:
            raise ValueError(
                "Cannot create a variable both on the previous time step and "
                "previous iteration."
            )
        if self.prev_iter:
            raise NotImplementedError(
                "Currently, it is not supported to create a variable that represents "
                "more than one iteration behind."
            )

        ndof: dict[Literal["cells", "faces", "nodes"], int] = {
            "cells": self._cells,
            "faces": self._faces,
            "nodes": self._nodes,
        }
        new_var = Variable(self.name, ndof, self.domain, previous_iteration=True)
        # Assign self as the original variable.
        new_var.original_variable = self
        return new_var

    def __repr__(self) -> str:
        s = f"Variable {self.name} with id {self.id}"
        if isinstance(self.domain, pp.MortarGrid):
            s += f" on interface {self.domain.id}\n"
        else:
            s += f" on grid {self.domain.id}\n"
        s += (
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
    subdomains or interfaces, but treated jointly in the mixed-dimensional sense.

    Conversion of the variables into numerical value should be done with respect to the
    state of an array; see :meth:`Operator.evaluate`. Therefore, the MergedVariable does
    not implement the method :meth:`Operator.parse`.

    Parameters:
        variables: List of variables to be merged. Should all have the same name.

    """

    def __init__(self, variables: list[Variable]) -> None:
        ### PUBLIC

        self.sub_vars = variables
        """List of sub-variables passed at instantiation, each defined on a separate
        domain.

        """

        self.id = next(Variable._ids)
        """ID counter. Used to identify variables during operator parsing."""

        self.prev_time: bool = False
        """Flag indicating if the variable represents the state at the previous time
        step.

        """

        self.prev_iter: bool = False
        """Flag indicating if the variable represents the state at the previous
        iteration.

        """

        self.original_variable: MixedDimensionalVariable
        """The original variable, if this variable is a copy of another variable.

        This attribute is used by the methods :meth:`Variable.previous_timestep` and
        :meth:`Variable.previous_iteration` to keep a link to the original variable.

        """

        ### PRIVATE

        # Flag to identify variables merged over no subdomains. This requires special
        # treatment in various parts of the code. A use case is variables that are only
        # defined on subdomains of codimension >= 1 (e.g., contact traction variable),
        # assigned to a problem where the grid happened not to have any fractures.
        self._no_variables = len(variables) == 0

        # It should be defined in the parent class, but we do not call super().__init__
        # Mypy complains that we do not know that all variables have the same type of
        # domain. While formally correct, this should be picked up in other places so we
        # ignore the warning here.
        self._domains = [
            var.domains[0] for var in variables  # type: ignore[assignment]
        ]
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
        # Yura: Is it only the problem of type checking that makes us inherit from
        # Variable?
        self._initialize_children()
        self.copy_common_sub_tags()

    def copy_common_sub_tags(self) -> None:
        """Copy any shared tags from the sub-variables to this variable.

        Only tags with identical values are copied. Thus, the md variable can "trust"
        that its tags are consistent with all sub-variables.

        """
        self._tags = {}
        # If there are no sub variables, there is nothing to do.
        if self._no_variables:
            return
        # Initialize with tags from the first sub-variable.
        common_tags = set(self.sub_vars[0].tags.keys())
        # Loop over all other sub-variables, take the intersection with the existing set
        # (common_tags) and update the set.
        for var in self.sub_vars[1:]:
            common_tags.intersection_update(set(var.tags.keys()))
        # Now, common_tags contains all tags that are shared by all sub-variables.
        for key in common_tags:
            # Find the tag values for the common tags. If the tag value is unique,
            # assign it to the md variable.
            values = set(var.tags[key] for var in self.sub_vars)
            if len(values) == 1:
                self.tags[key] = values.pop()

    @property
    def domain(self) -> list[GridLike]:  # type: ignore[override]
        """A tuple of all domains on which the atomic sub-variables are defined."""
        domains = [var.domain for var in self.sub_vars]
        # Verify that all domains of of the same type
        assert all(isinstance(d, pp.Grid) for d in domains) or all(
            isinstance(d, pp.MortarGrid) for d in domains
        )
        return domains

    @property
    def size(self) -> int:
        """Returns the total size of the mixed-dimensional variable
        by summing the sizes of sub-variables."""
        return sum([v.size for v in self.sub_vars])

    def previous_timestep(self) -> MixedDimensionalVariable:
        """Return a representation of this mixed-dimensional variable on the previous
        time step.

        Raises:
            ValueError:
                If the variable is a representation of the previous iteration,
                previously set by :meth:`~previous_iteration`.

            NotImplementedError:
                If the variable is already a representation of the previous time step.
                Currently, we support creating only one previous time step.

        Returns:
            A representation of this merged variable on the previous time
            iteration, with its ``prev_iter`` attribute set to ``True``.

        """

        new_subs = [var.previous_timestep() for var in self.sub_vars]
        new_var = MixedDimensionalVariable(new_subs)
        new_var.prev_time = True
        # Assign self as the original variable.
        new_var.original_variable = self
        return new_var

    def previous_iteration(self) -> MixedDimensionalVariable:
        """Return a representation of this mixed-dimensional variable on the previous
        iteration.

        Raises:
            ValueError:
                If the variable is a representation of the previous time step,
                previously set by :meth:`~previous_timestep`.

            NotImplementedError:
                If the variable is already a representation of the previous time
                iteration. Currently, we support creating only one previous iteration.

        Returns:
            A representation of this merged variable on the previous
            iteration, with its ``prev_iter`` attribute set to ``True``

        """
        new_subs = [var.previous_iteration() for var in self.sub_vars]
        new_var = MixedDimensionalVariable(new_subs)
        new_var.prev_iter = True
        # Assign self as the original variable.
        new_var.original_variable = self
        return new_var

    def copy(self) -> "MixedDimensionalVariable":
        """Copy the mixed-dimensional variable.

        Returns:
            A shallow copy should be sufficient here; the attributes are not expected to
            change.

        """
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
            f"Total size: {self.size}\n"
        )
        if self.prev_iter:
            s += "Evaluated at the previous iteration.\n"
        elif self.prev_time:
            s += "Evaluated at the previous time step.\n"

        return s


@overload
def _ad_wrapper(
    vals: Union[pp.number, np.ndarray],
    as_array: Literal[False],
    size: Optional[int] = None,
    name: Optional[str] = None,
) -> SparseArray:
    # See md_grid for explanation of overloading and type hints.
    ...


@overload
def _ad_wrapper(
    vals: Union[pp.number, np.ndarray],
    as_array: Literal[True],
    size: Optional[int] = None,
    name: Optional[str] = None,
) -> DenseArray: ...


def _ad_wrapper(
    vals: Union[pp.number, np.ndarray],
    as_array: bool,
    size: Optional[int] = None,
    name: Optional[str] = None,
) -> DenseArray | pp.ad.SparseArray:
    """Create ad array or diagonal matrix.

    Utility method.

    Parameters:
        vals: Values to be wrapped. Floats are broadcast to an np array.
        array: Whether to return a matrix or vector.
        size: Size of the array or matrix. If not set, the size is inferred from vals.
        name: Name of ad object.

    Returns:
        Values wrapped as an Ad object.

    """
    if type(vals) is not np.ndarray:
        assert size is not None, "Size must be set if vals is not an array"
        value_array: np.ndarray = vals * np.ones(size)
    else:
        value_array = vals

    if as_array:
        return pp.ad.DenseArray(value_array, name)
    else:
        if size is None:
            size = value_array.size
        matrix = sps.diags(vals, shape=(size, size))
        return pp.ad.SparseArray(matrix, name)


def wrap_as_dense_ad_array(
    vals: pp.number | np.ndarray,
    size: Optional[int] = None,
    name: Optional[str] = None,
) -> DenseArray:
    """Wrap a number or array as ad array.

    Parameters:
        vals: Values to be wrapped. Floats are broadcast to an np array.
        size: Size of the array. If not set, the size is inferred from vals.
        name: Name of ad object.

    Returns:
        Values wrapped as an ad Array.

    """
    return _ad_wrapper(vals, True, size=size, name=name)


def wrap_as_sparse_ad_array(
    vals: Union[pp.number, np.ndarray],
    size: Optional[int] = None,
    name: Optional[str] = None,
) -> SparseArray:
    """Wrap a number or array as ad matrix.

    Parameters:
        vals: Values to be wrapped. Floats are broadcast to an np array.
        size: Size of the array. If not set, the size is inferred from vals.
        name: Name of ad object.

    Returns:
        Values wrapped as an ad Matrix.

    """
    return _ad_wrapper(vals, False, size=size, name=name)


def sum_operator_list(
    operators: list[Operator],
    name: Optional[str] = None,
) -> Operator:
    """Sum a list of operators.

    Parameters:
        operators: List of operators to be summed.
        name: Name of the resulting operator.

    Returns:
        Operator that is the sum of the input operators.

    """
    result = reduce(lambda a, b: a + b, operators)

    if name is not None:
        result.set_name(name)

    return result
