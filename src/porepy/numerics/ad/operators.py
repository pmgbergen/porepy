"""Implementation of wrappers for Ad representations of several operators."""

from __future__ import annotations

import copy
from collections import deque
from enum import Enum
from functools import reduce
from hashlib import sha256
from itertools import count
from typing import Any, Callable, Literal, Optional, Sequence, TypeVar, Union, overload
from warnings import warn

import networkx as nx
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.utils.porepy_types import GridLike, GridLikeSequence

from . import _ad_utils
from .forward_mode import AdArray

__all__ = [
    "Operator",
    "TimeDependentOperator",
    "IterativeOperator",
    "SparseArray",
    "DenseArray",
    "TimeDependentDenseArray",
    "Scalar",
    "Variable",
    "MixedDimensionalVariable",
    "Projection",
    "ProjectionList",
    "sum_operator_list",
    "sum_projection_list",
]


def _get_previous_time_or_iterate(
    op: Operator, prev_time: bool = True, steps: int = 1
) -> Operator:
    """Helper function which traverses an operator's tree recursively to get a
    copy of it and it's children, representing ``op`` at a previous time or
    iteration.

    Parameters:
        op: Some operator whose tree should be traversed.
        prev_time: ``default=True``

            If True, it calls :meth:`Operator.previous_timestep`, otherwise it calls
            :meth:`Operator.previous_iteration`.

            This is the only difference in the recursion and we can avoid duplicate
            code.
        steps: ``default=1``

            Number of steps backwards in time or iterate sense.

    Returns:
        A copy of the operator and its children, representing the previous time or
        iteration.

    """

    # The recursion reached an atomic operator, which has some time- or
    # iterate-dependent behaviour
    if isinstance(op, TimeDependentOperator) and prev_time:
        return op.previous_timestep(steps=steps)
    elif isinstance(op, IterativeOperator) and not prev_time:
        return op.previous_iteration(steps=steps)
    # NOTE The previous_iteration of a time-dependent operator will return the operator
    # itself. Vice-versa, the previous_timestep of an Iterative operator will return
    # itself. Holds only if the operator is original (no previous_* operation performed)

    # The recursion reached an operator without children and without time- or iterate-
    # dependent behaviour
    elif op.is_leaf():
        return op
    # Else we are in the middle of the operator tree and need to go deeper, creating
    # copies along.
    else:
        # Create new operator from the tree, with the only difference being the new
        # children, for which the recursion is invoked
        # NOTE copy takes care of references to original_operator and func
        new_op = copy.copy(op)
        new_op.children = [
            _get_previous_time_or_iterate(child, prev_time=prev_time, steps=steps)
            for child in op.children
        ]
        return new_op


class Operations(Enum):
    """Object representing all supported operations by the operator class.

    Used to construct the operator tree and identify Operations.

    """

    # NOTE: The string values of the operations are used in the construction of hash
    # keys for compound operators. If adding new operations, these must be assigned
    # unique string values.

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
    pow = "pow"
    rpow = "rpow"

    @classmethod
    def to_symbol(cls, value):
        symbols = {
            cls.add: "+",
            cls.sub: "-",
            cls.mul: "*",
            cls.rmul: "*",
            cls.matmul: "@",
            cls.rmatmul: "@",
            cls.div: "/",
            cls.rdiv: "/",
            cls.pow: "**",
            cls.rpow: "**",
            cls.evaluate: "evaluate",
            cls.approximate: "approximate",
            cls.void: "void",
        }
        return symbols.get(value, "unknown")

    @classmethod
    def to_str(cls, value):
        strings = {
            cls.add: "adding",
            cls.sub: "subtracting",
            cls.mul: "multiplying",
            cls.rmul: "multiplying",
            cls.matmul: "matrix multiplying",
            cls.rmatmul: "matrix multiplying",
            cls.div: "dividing",
            cls.rdiv: "dividing",
            cls.pow: "raising to the power of",
            cls.rpow: "raising to the power of",
            cls.evaluate: "evaluating",
            cls.approximate: "approximating",
            cls.void: "void",
        }
        return strings.get(value, "unknown")


class Operator:
    """Parent class for all AD operators.

    Objects of this class are not meant to be initiated directly, rather the various
    subclasses should be used. Instances of this class will still be created when
    subclasses are combined by Operations.

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

    def __init__(
        self,
        name: Optional[str] = None,
        domains: Optional[GridLikeSequence] = None,
        operation: Optional[Operations] = None,
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

        self.func: Callable[..., float | np.ndarray | AdArray]
        """Functional representation of this operator.

        As of now, only instances of
        :class:`~porepy.numerics.ad.operator_functions.AbstractFunction` have a
        functional representation, whereas basic arithmetics are implemented by
        arithmetic overloads in this class.

        Note:
            This declaration avoids operator functions creating operators with
            themselves as the first child to provide access to
            :meth:`~porepy.numerics.ad.operator_functions.AbstractFunction.func`,
            and hence artificially bloating the operator tree.

        Note:
            For future development:

            Functional representation can be used for an optimized representation
            (keyword numba compilation).

        """

        self.children: Sequence[Operator]
        """List of children, other AD operators.

        Will be empty if the operator is a leaf.
        """

        self.operation: Operations
        """Arithmetic or other operation represented by this operator.

        Will be void if the operator is a leaf.
        """
        ### PRIVATE
        self._name = name if name is not None else ""

        self._initialize_children(operation=operation, children=children)
        self._cached_key: Optional[str] = None

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
        operation: Optional[Operations] = None,
        children: Optional[Sequence[Operator]] = None,
    ):
        """This is a part of initialization which can be called separately since some
        subclasses do not call super().__init__()

        """
        self.children = [] if children is None else children
        self.operation = Operations.void if operation is None else operation

    def as_graph(
        self, depth_first: bool = True
    ) -> tuple[nx.DiGraph, dict[Operator, int]]:
        """Return the operator tree as a directed graph using networkx.

        Since networkx uses the object hash as the node id, and operators can share the
        same hash, we cannot use the operator themselves as nodes. Instead, the nodes in
        the graph are integers. To map the nodes back to the operators, a dictionary is
        returned. For a given node, the actual operator is stored as an attribute, with
        key "obj".

        The graph will actually be that of a tree with self as root. The parameter
        depth_first determines if the graph is constructed, and node ids assigned, by
        depth-first or breadth-first traversal.
        EK comment: At the moment, it is not clear whether this has any practical impact
        on the graph properties, e.g. for traversal.

        The edges of the graph are directed from parent to child. The attribute
        "operand_id" is used to store the index of the child in the parent's children
        list. This identifies the left and right operand, e.g. in a multiplication
        operation.

        Parameters:
            depth_first: If True, the graph is traversed depth-first, otherwise
                breadth-first.

        Returns:
            A tuple with the graph and a dictionary mapping the nodes to the operators.

        """
        # EK note: We will probably take this method into use in the near future, right
        # now it is not used, but kept for future applications.

        # Counter for node ids.
        idx = count(0)

        graph = nx.DiGraph()
        # We will loop over all successors of this node and add them to the graph. Use
        # a deque to keep track of the nodes discovered but not yet added to the graph.
        queue = deque([self])
        # Add the root node.
        id = next(idx)
        graph.add_node(id, obj=self)

        # Mapping from operator to node id. Needed to go from an operator to a node in
        # the graph.
        node_map: dict[Operator, int] = {}
        node_map[self] = id

        while queue:
            # Depth-first traversal is implemented by popping from the right (last in,
            # first out).
            if depth_first:
                parent = queue.pop()
            else:
                parent = queue.popleft()

            for counter, child in enumerate(parent.children):
                id = next(idx)
                # Add the child to the graph, store the mapping between the node id
                # and the operator, and make the edge between the parent and the child.
                graph.add_node(id, obj=child)
                node_map[child] = id
                graph.add_edge(node_map[parent], node_map[child], operand_id=counter)
                # Always append to the right.
                queue.append(child)

        return graph, node_map

    def is_leaf(self) -> bool:
        """Check if this operator is a leaf in the tree-representation of an expression.

        Note that this implies that the method ``parse()`` is expected to be
        implemented.

        Returns:
            True if the operator has no children.

        """
        return len(self.children) == 0

    @property
    def is_current_iterate(self) -> bool:
        """Returns True if this AD-operator represents its designated term at the
        current time and iterate index.

        Note:
            This flag is used in time step and iterate notions of
            :class:`TimeDependentOperator` and :class:`IterativeOperator`.

        """
        # NOTE we use the existence of the original operator (not the index)
        # because this works for both previous time and iteration.
        if hasattr(self, "original_operator"):
            return False
        else:
            return True

    def set_name(self, name: str) -> None:
        """Reset this object's name originally passed at instantiation.

        Parameters:
            name: the new name to be assigned.

        """
        self._name = name

    def previous_timestep(self, steps: int = 1) -> pp.ad.Operator:
        """Base method to trigger a recursion over the operator tree and create a
        shallow copy of this operator, where child operators with time-dependent
        behaviour are pushed backwards in time.

        For more information, see :class:`TimeDependentOperator`.

        """
        return _get_previous_time_or_iterate(self, prev_time=True, steps=steps)

    def previous_iteration(self, steps: int = 1) -> pp.ad.Operator:
        """Base method to trigger a recursion over the operator tree and create a
        shallow copy of this operator, where child operators with iterative
        behaviour are pushed backwards in the iterative sense.

        For more information, see :class:`IterativeOperator`.

        """
        return _get_previous_time_or_iterate(self, prev_time=False, steps=steps)

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

        DEPRECATED: This method is deprecated. Use the `evaluate` method of
        EquationSystem.

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
        msg = "This method is deprecated. Use the `evaluate` method of EquationSystem."
        warn(msg, DeprecationWarning)

        return self._evaluate(system_manager, state=state, evaluate_jacobian=False)

    def value_and_jacobian(
        self, system_manager: pp.ad.EquationSystem, state: Optional[np.ndarray] = None
    ) -> AdArray:
        """Evaluate the residual and Jacobian matrix for a given solution.

        DEPRECATED: This method is deprecated. Use the `evaluate` method of
        EquationSystem.

        Parameters:
            system_manager: Used to represent the problem. Will be used to parse the
                sub-operators that combine to form this operator.
            state: Solution vector for which the residual and its derivatives should be
                formed. If not provided, the solution will be pulled from the previous
                iterate (if this exists), or alternatively from the solution at the
                previous time step.

        Returns:
            A representation of the residual and Jacobian in form of an AD Array. Note
                that the Jacobian matrix need not be invertible, or even square; this
                depends on the operator.

        """
        msg = "This method is deprecated. Use the `evaluate` method of EquationSystem."
        warn(msg, DeprecationWarning)
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

        See also:
            EquationSystem, method evaluate.

        """

        # If state is not specified, use values at current time, current iterate
        if state is None:
            state = system_manager.get_variable_values(iterate_index=0)

        # Use methods in the EquationSystem to evaluate the operator. This inversion of
        # roles (self.value) reflects a gradual shift
        if evaluate_jacobian:
            return system_manager.evaluate(self, derivative=True, state=state)
        else:
            return system_manager.evaluate(self, derivative=False, state=state)

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
        return Operator(children=children, operation=Operations.add, name="+ operator")

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
        return Operator(children=children, operation=Operations.sub, name="- operator")

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
        return Operator(children=children, operation=Operations.sub, name="- operator")

    def __mul__(self, other: Operator) -> Operator:
        """Elementwise multiplication of two operators.

        Parameters:
            other: The operator to multiply with self.

        Returns:
            The elementwise product of self and other.

        """
        children = self._parse_other(other)
        return Operator(children=children, operation=Operations.mul, name="* operator")

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
            operation=Operations.rmul,
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
        return Operator(children=children, operation=Operations.div, name="/ operator")

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
            operation=Operations.rdiv,
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
        return Operator(children=children, operation=Operations.pow, name="** operator")

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
            operation=Operations.rpow,
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
            children=children, operation=Operations.matmul, name="@ operator"
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
            operation=Operations.rmatmul,
            name="reverse @ operator",
        )

    def __hash__(self):
        return hash(self._key())

    def _key(self) -> str:
        """String representation for hashing.

        Provides a representation of the operator tree which grows from the current
        operator meant for hashing. Different AD objects that represent the identical
        trees defined on the same domains must have identical keys. Otherwise, the keys
        must be different.

        All the leave operators (which are the subclasses) must override this method.
        Calling super is not expected.

        Returns:
            The key string.

        Raises:
            ValueError: If this method is called from the leave AD object.

        """
        if self._cached_key is None:
            if self.operation == Operations.void or len(self.children) == 0:
                raise ValueError("Base class operator must represent an operation.")
            tmp = [self.operation.value] + [child._key() for child in self.children]
            self._cached_key = " ".join(tmp)
        return self._cached_key

    def _parse_other(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return [self, Scalar(other)]
        elif isinstance(other, np.ndarray):
            return [self, DenseArray(other)]
        elif isinstance(other, sps.spmatrix):
            return [self, SparseArray(other)]
        elif isinstance(other, AdArray):
            # This may happen when using nested pp.ad.Function.
            return [self, other]
        elif isinstance(other, pp.ad.AbstractFunction):
            # Need to put this here, because overload of AbstractFunction is not
            # applied if AbstractFunction is right operand.
            pp.ad.operator_functions._raise_no_arithmetics_with_functions_error()
        elif isinstance(other, Operator):
            # Put Operator at end, because Seconary and Abstract are also operators
            return [self, other]
        else:
            raise ValueError(f"Cannot parse {other} as an AD operator")


class TimeDependentOperator(Operator):
    """Intermediate parent class for operator classes, which can have a time-dependent
    representation.

    Implements the notion of time step indices, as well as a method to create a
    representation of an operator instance at a previous time.

    Operators created via constructor always start at the current time.

    """

    def __init__(
        self,
        name: str | None = None,
        domains: Optional[pp.GridLikeSequence] = None,
        operation: Optional[Operations] = None,
        children: Optional[Sequence[Operator]] = None,
    ) -> None:
        super().__init__(
            name=name, domains=domains, operation=operation, children=children
        )

        self.original_operator: Operator
        """Reference to the operator representing this operator at the current time amd
        iterate.

        This attribute is only available in operators representing previous time steps.

        """

        self._time_step_index: int = -1
        """Time step index, starting with 0 (current time) and increasing for previous
        time steps."""

    @property
    def is_previous_time(self) -> bool:
        """True, if the operator represents a previous time-step."""
        return True if self._time_step_index >= 0 else False

    @property
    def time_step_index(self) -> int | None:
        """Returns the time step index this instance represents.

        - None indicates the current time (unknown value)
        - 0 indicates this is an operator at the first previous time step
        - 1 at the time step before
        - ...

        """
        if self._time_step_index < 0:
            return None
        else:
            return self._time_step_index

    def previous_timestep(
        self: _TimeDependentOperator, steps: int = 1
    ) -> _TimeDependentOperator:
        """Returns a copy of the time-dependent operator with an advanced time-step
        index.

        Time-dependent operators do not invoke the recursion (like the base class),
        but represent a leaf in the recursion tree.

        Note:
            You cannot create operators at the previous time step from operators which
            are at some previous iterate. Use the :attr:`original_operator` instead.

        Parameters:
            steps: ``default=1``

                Number of steps backwards in time.

        Raises:
            ValueError: If this instance represents an operator at a previous iterate.
            AssertionError: If ``steps`` is not strictly positive.

        """
        if isinstance(self, IterativeOperator):
            if self.is_previous_iterate:
                raise ValueError(
                    "Cannot create an operator representing a previous time step,"
                    + " if it already represents a previous iterate."
                )

        assert steps > 0, "Number of steps backwards must be strictly positive."
        # TODO copy or deepcopy? Is this enough for every operator class?
        op = copy.copy(self)

        # NOTE Use private time step index, because it is always an integer
        # The public time step index is NONE for current time
        # (which translates to -1 for the private index)
        op._time_step_index = self._time_step_index + int(steps)

        # keeping track to the very first one
        if self.is_current_iterate:
            op.original_operator = self
        else:
            op.original_operator = self.original_operator

        return op


_TimeDependentOperator = TypeVar("_TimeDependentOperator", bound=TimeDependentOperator)


class IterativeOperator(Operator):
    """Intermediate parent class for operator classes, which can have multiple
    representations in the iterative sense.

    Implements the notion of iterate indices, as well as a method to create a
    representation of an operator instance at a iterate time.

    Operators created via constructor always start at the current iterate.

    Note:
        Operators which represents some previous iterate represent also
        always the current time.

    """

    def __init__(
        self,
        name: str | None = None,
        domains: Optional[pp.GridLikeSequence] = None,
        operation: Optional[Operations] = None,
        children: Optional[Sequence[Operator]] = None,
    ) -> None:
        super().__init__(
            name=name, domains=domains, operation=operation, children=children
        )

        self.original_operator: Operator
        """Reference to the operator representing this operator at the current time amd
        iterate.

        This attribute is only available in operators representing previous time steps.

        """

        self._iterate_index: int = -1
        """Iterate index, starting with 0 (current iterate at current time) and
        increasing for previous iterates."""

    @property
    def is_previous_iterate(self) -> bool:
        """True, if the operator represents a previous iterate."""
        return True if self._iterate_index >= 0 else False

    @property
    def iterate_index(self) -> int | None:
        """Returns the iterate index this instance represents, at the current time.

        - None indicates this instance is at a previous time
        - 0 represents the most recently computed iterate.
        - 1 represents the iterate before that
        - ...

        Note:
            Operators at current time (unknown value) also have the index 0, since those
            values are used to linearize the system and construct the Jacobian.

        """
        # Operators at previous time have no iterate indices
        if isinstance(self, TimeDependentOperator):
            if self.is_previous_time:
                return None

        # operators representing at current time use the values stored at index 0
        # in that case the private index is -1
        if self._iterate_index < 0:
            return 0
        # return respective index
        else:
            return self._iterate_index

    def previous_iteration(
        self: _IterativeOperator, steps: int = 1
    ) -> _IterativeOperator:
        """Returns a copy of the iterative operator with an advanced iterate index.

        Iterative operators do not invoke the recursion (like the base class),
        but represent a leaf in the recursion tree.

        Note:
            You cannot create operators at the previous iterates from operators which
            are at some previous time step. Use the :attr:`original_operator` instead.

        Parameters:
            steps: ``default=1``

                Number of steps backwards in the iterate sense.

        Raises:
            ValueError: If this instance represents an operator at a previous time step.
            AssertionError: If ``steps`` is not strictly positive.

        """
        if isinstance(self, TimeDependentOperator):
            if self.is_previous_time:
                raise ValueError(
                    "Cannot create an operator representing a previous iterate,"
                    + " if it already represents a previous time step."
                )
        assert steps > 0, "Number of steps backwards must be strictly positive."
        # See TODO in TimeDependentOperator.previous_timestep
        op = copy.copy(self)
        op._iterate_index = self._iterate_index + int(steps)

        # keeping track to the very first one
        if self.is_current_iterate:
            op.original_operator = self
        else:
            op.original_operator = self.original_operator

        return op


_IterativeOperator = TypeVar("_IterativeOperator", bound=IterativeOperator)


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
        self._mat = mat
        # Force the data to be float, so that we limit the number of combinations of
        # data types that we need to consider in parsing.
        self._mat.data = self._mat.data.astype(float)
        self._shape = mat.shape
        """Shape of the wrapped matrix."""

        # TODO: Make readonly, see https://github.com/pmgbergen/porepy/issues/1214
        self._hash_value: str = self._compute_spmatrix_hash(mat)
        """String to uniquly identify the contents of the matrix."""

        super().__init__(name=name)

    def _key(self) -> str:
        if self._cached_key is None:
            self._cached_key = f"(sparse_array, hash={self._hash_value})"
        return self._cached_key

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

    def transpose(self) -> SparseArray:
        """Returns an AD operator representing the transposed matrix."""
        return SparseArray(self._mat.transpose())

    @property
    def T(self) -> SparseArray:
        """Shorthand for transpose."""
        return self.transpose()

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the wrapped matrix."""
        return self._shape

    @staticmethod
    def _compute_spmatrix_hash(mat: sps.spmatrix) -> str:
        """Utility function that handles all the sparse formats to compute the hash.

        The hash will match for two sparse arrays with the identical: sparse formats,
        shapes, data arrays, rows and columns arrays. The identical matrices in the
        different formats will have different hashes. Such behavior is expected, since
        two different formats are likely used for a good reason.

        The rows and columns are considered to avoid the collision between, e.g., the
        following rows in the csr format: `[1, 0, 1, 0]` vs `[0, 1, 0, 1]`.

        The sparse format is included explicitly to resolve the following collision:
        `hash(mat.tocsr()) == hash(mat.T.tocsc())` - we do not want them to be equal,
        but the data, indices and indptr will be identical for them.

        """
        # Proper handling of all the formats like csr, coo, etc..
        if isinstance(
            mat,
            (
                sps.csr_matrix,
                sps.csr_array,
                sps.csc_matrix,
                sps.csc_array,
                sps.bsr_matrix,
                sps.bsr_array,
            ),
        ):
            properties = [mat.data, mat.indices, mat.indptr]
        elif isinstance(mat, (sps.coo_matrix, sps.coo_array)):
            properties = [mat.data, mat.row, mat.col]
        elif isinstance(mat, (sps.dia_matrix, sps.dia_array)):
            properties = [mat.data, mat.offsets]
        else:
            raise NotImplementedError("Hashing not provided for the format", type(mat))

        # Concatenating the hashes of the data, rows and columns arrays.
        data_hash = "".join(
            [sha256(array, usedforsecurity=False).hexdigest() for array in properties]
        )
        # Adding the information about the matrix format and shape. `mat.format` is not
        # used because it does not distinguish between, e.g., csr_matrix and csr_array.
        return f"{type(mat).__name__}_{mat.shape}_{data_hash}"


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
        # Force the data to be float, so that we limit the number of combinations of
        # data types that we need to consider in parsing.
        self._values = values.astype(float, copy=False)

        # TODO: Make readonly, see https://github.com/pmgbergen/porepy/issues/1214
        self._hash_value: str = sha256(
            self._values,
            usedforsecurity=False,  # type: ignore[arg-type]
        ).hexdigest()
        """String to uniquly identify the array."""
        super().__init__(name=name)

    def _key(self) -> str:
        if self._cached_key is None:
            self._cached_key = f"(dense_array, hash={self._hash_value})"
        return self._cached_key

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


class TimeDependentDenseArray(TimeDependentOperator):
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
    ):
        super().__init__(name=name, domains=domains)

    def _key(self) -> str:
        if self._cached_key is None:
            domain_ids = [domain.id for domain in self.domains]
            self._cached_key = (
                f"(time_dependent_dense_array, name={self.name}, domains={domain_ids})"
            )
        return self._cached_key

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
        if self.is_previous_time:
            index_kwarg = {"time_step_index": self.time_step_index}
        else:
            index_kwarg = {"iterate_index": 0}

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

            vals.append(
                pp.get_solution_values(name=self._name, data=data, **index_kwarg)
            )

        if len(vals) > 0:
            # Normal case: concatenate the values from all grids
            return np.hstack((vals))
        else:
            # Special case: No grids. Return an empty array.
            return np.empty(0, dtype=float)

    def __repr__(self) -> str:
        msg = (
            f"Wrapped time-dependent array with name {self._name}.\n"
            f"Defined on {len(self._domains)} {self._domain_type}.\n"
        )
        if self.is_previous_time:
            msg += f"Evaluated at the previous time step {self.time_step_index}.\n"
        return msg


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
        # Force the data to be float, so that we limit the number of combinations of
        # data types that we need to consider in parsing.
        self._value = float(value)
        # Call the super constructor after setting the value.
        super().__init__(name=name)

    def _key(self) -> str:
        if self._cached_key is None:
            self._cached_key = f"(scalar, {self._value})"
        return self._cached_key

    def __repr__(self) -> str:
        # Normally, we will return a string with the value. However, for debugging of
        # initialization, where self._value is not yet defined, we need a reasonable
        # fallback (we can do without, but that results in annoying error messages while
        # debugging).
        try:
            value = f"{self._value}"
            return f"Wrapped scalar with value {value}"
        except AttributeError:
            return "Wrapped scalar with no value"

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


class Variable(TimeDependentOperator, IterativeOperator):
    """AD operator representing a variable defined on a single grid or mortar grid.

    For combinations of variables on different subdomains, see
    :class:`MixedDimensionalVariable`.

    A variable is associated with either a grid or an interface. Therefore it is assumed
    that either ``subdomains`` or ``interfaces`` is passed as an argument.

    Also, a variable is associated with a specific time and iterate index. :meth:`parse`
    will return the values at respective index on its :meth:`domain`.

    Important:
        Each atomic variable (a variable on a single grid) has a :attr:`id`, unique
        among created variables. This ID is used to map the DOFs in the global system
        and hence critical.

        As of now, variable instances representing the same quantity at different
        time and iterate steps have the same ID.

        This might with future development (e.g. adaptive mesh refinement).

    Parameters:
        name: Variable name.
        ndof: Number of dofs per grid element.
            Valid keys are ``cells``, ``faces`` and ``nodes``.
        domain: A subdomain or interface on which the variable is defined.
        tags: A dictionary of tags.

    Raises:
        NotImplementedError: If ``domain`` is not a grid or mortar grid. Variables are
            not supported on boundaries.

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
    ) -> None:
        # Variables are not supported on the boundary.
        if not isinstance(domain, (pp.Grid, pp.MortarGrid)):
            raise NotImplementedError(
                "Variables only supported on domains of type 'Grid' or 'MortarGrid'."
            )

        self._id: int = next(Variable._ids)
        """See :meth:`id`."""
        self._g: GridLike = domain
        """See :meth:`domain`"""

        # Block a mypy warning here: Domain is known to be GridLike (grid, mortar grid,
        # or boundary grid), thus the below wrapping in a list gives a list of GridLike,
        # but the super constructor expects a sequence of grids, sequence or mortar
        # grids etc. Mypy makes a difference, but the additional entropy needed to
        # circumvent the warning is not worth it.
        super().__init__(name=name, domains=[domain])  # type: ignore [arg-type]

        # dofs per
        self._cells: int = ndof.get("cells", 0)
        self._faces: int = ndof.get("faces", 0)
        self._nodes: int = ndof.get("nodes", 0)

        # tag
        self._tags: dict[str, Any] = tags if tags is not None else {}

    def _key(self):
        if self._cached_key is None:
            self._cached_key = f"(var, name={self.name}, domain={str(self.domain.id)})"
        return self._cached_key

    @property
    def id(self) -> int:
        """Returns an integer unique among variables used for identification.
        Assigned during instantiation.

        The id of a variable is common for all instances of the variable, regardless of
        whether it represents the present state, the previous iteration, or the previous
        time step.

        While a specific variable can be identified in terms of its id, it is often
        advisable to rather use its name and domain, preferrably using relevant
        functionality in
        :class:`~porepy.numerics.ad.equation_system.EquationSystem`.

        """
        return self._id

    @property
    def domain(self) -> GridLike:
        """The grid or mortar grid on which this variable is defined.

        Note:
            Not to be confused with :meth:`domains`, which has the grid in a sequence
            of length 1.

            This is for inheritance reasons, since :class:`Variable` inherits from
            :class:`Operator`.

            TODO: Clean up.


        """
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

    def parse(self, mdg: pp.MixedDimensionalGrid) -> Any:
        """Returns the values stored for this variable at its time step or iterate
        index."""

        # By logic in the constructor, it can only be a subdomain or interface
        if isinstance(self._g, pp.Grid):
            data = mdg.subdomain_data(self._g)
        elif isinstance(self._g, pp.MortarGrid):
            data = mdg.interface_data(self._g)

        # We can safely use both indices as arguments, without checking prev time,
        # because iterate index is None if prev time, and vice versa
        return pp.get_solution_values(
            self.name,
            data,
            iterate_index=self.iterate_index,
            time_step_index=self.time_step_index,
        )

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
        if self.is_previous_iterate:
            s += f"Evaluated at the previous iteration {self.iterate_index}.\n"
        elif self.is_previous_time:
            s += f"Evaluated at the previous time step {self.time_step_index}.\n"

        return s


class MixedDimensionalVariable(Variable):
    """Ad representation of a collection of variables that individually live on separate
    subdomains or interfaces, but represent the same quantity and are treated jointly in
    the mixed-dimensional sense.

    Note:
        As of now, the wrapped fixed-dimensional variables must fulfill the following
        assumptions:

        1. They have the same name
        2. They are at the same time step and iterate.
        3. They are defined in different grids (no overlaps).

    Parameters:
        variables: List of variables to be merged.

    Raises:
        AssertionError: If one of the above assumptions is violated.

    """

    def _key(self) -> str:
        # The MixedDimensionalVariable is not stored in the equation system but
        # constructed every time when needed. Thus, its id is updated and cannot be
        # relied on. Instead, we rely on the name, which is guaranteed to be unique.
        if self._cached_key is None:
            domain_ids = [domain.id for domain in self.domains]
            self._cached_key = f"(mdvar, name={self.name}, domains={domain_ids})"
        return self._cached_key

    def __init__(self, variables: list[Variable]) -> None:
        # IMPLEMENTATION NOTE VL
        # I guess the original idea was to have the md-variable as a leaf in an operator
        # tree, hence bypassing the super().__init__ and not having children.
        # The code would be clearer if the md-variable is **not** a leaf, with no
        # impairment to the efficiency of the parsing.
        # Then we could call super() here and would have no need to essentially mimic
        # its functionality here (along with the functionality of time-dependent and
        # iterate operators), no duplicate code and less margin for errors.
        # Unclear is however, how much of the remaining code must change, because
        # the md-variable would not have an ID anymore, only the atomic variables.
        # Also, there would be no attribute sub_vars, but the regular children, and the
        # class would need custom implementations for is_previous_time/iterate and
        # is_current_iterate, because these flags are useful on md-level as well.
        # My guess, it's not much because EquationSystem operatores solely on atomic
        # variables and their dofs, and changes are restricted to there (and tests)

        time_indices = []
        iter_indices = []
        current_iter = []
        names = []
        domains = []

        for var in variables:
            time_indices.append(var.time_step_index)
            iter_indices.append(var.iterate_index)
            current_iter.append(var.is_current_iterate)
            names.append(var.name)
            domains.append(var.domain)

        # check assumptions
        if len(variables) > 0:
            assert len(set(time_indices)) == 1, (
                "Cannot create md-variable from variables at different time steps."
            )
            # NOTE both must be unique for all sub-variables, to avoid md-variables
            # having sub-variables at different iterate states.
            # Both current value, and most recent previous iterate have iterate index 0,
            # hence the need to check the size of the current_iter set.
            assert len(set(iter_indices)) == 1 and len(set(current_iter)) == 1, (
                "Cannot create md-variable from variables at different iterates."
            )
            assert len(set(names)) == 1, (
                "Cannot create md-variable from variables with different names."
            )
            assert len(set(domains)) == len(domains), (
                "Cannot create md-variable from variables with overlapping domains."
            )
        # Default values for empty md variable
        else:
            time_indices = [-1]
            iter_indices = [None]
            names = ["empty_md_variable"]
            current_iter = [True]

        # NOTE everything below here is redundent with a proper super() call
        # See top comment in constructor

        ### PRIVATE
        self._id = next(Variable._ids)
        # NOTE private time step index is -1 if public time step index of atomic
        # variables is None (current time)
        self._time_step_index = -1 if time_indices[0] is None else time_indices[0]

        # If current time and iterate
        if current_iter[0]:
            # NOTE need to catch current-iter in if-else, otherwise the md-variable at
            # current time (to be solved for) would get the private index 0, instead of
            # -1, because atomic vars at current time and iter have public iter index 0
            self._iterate_index = -1  # current time and iter
        else:
            # can be None if variables at previous time. Set iterate index to default
            # value.
            self._iterate_index = -1 if iter_indices[0] is None else iter_indices[0]

        self._name = names[0]

        # Mypy complains that we do not know that all variables have the same type of
        # domain. While formally correct, this should be picked up in other places so we
        # ignore the warning here.
        self._domains = domains  # type: ignore[assignment]

        # If someone attempts to create a prev time or iter md-variable using
        # atomic variables at prev time and iter, we have a missing reference to the
        # operator at current time and iter. Need ro reverse-engineer that, for
        # is_current_iterate to work on the md-variable-level
        if self.is_previous_iterate or self.is_previous_time:
            # Mypy complains because of the typing of original_operator
            original_mdg = MixedDimensionalVariable(
                [var.original_operator for var in variables]  # type:ignore[misc]
            )
            original_mdg._id = self._id
            self.original_operator = original_mdg

        ### PUBLIC

        self.sub_vars = variables
        """List of sub-variables passed at instantiation, each defined on a separate
        domain.

        """

        self._initialize_children()
        self.copy_common_sub_tags()
        self._cached_key: Optional[str] = None

    def __repr__(self) -> str:
        if len(self.sub_vars) == 0:
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
        if self.is_previous_iterate:
            s += f"Evaluated at the previous iteration {self.iterate_index}.\n"
        elif self.is_previous_time:
            s += f"Evaluated at the previous time step {self.time_step_index}.\n"

        return s

    def copy_common_sub_tags(self) -> None:
        """Copy any shared tags from the sub-variables to this variable.

        Only tags with identical values are copied. Thus, the md variable can "trust"
        that its tags are consistent with all sub-variables.

        """
        self._tags = {}
        # If there are no sub variables, there is nothing to do.
        if len(self.sub_vars) == 0:
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
        return [var.domain for var in self.sub_vars]

    @property
    def size(self) -> int:
        """Returns the total size of the mixed-dimensional variable
        by summing the sizes of sub-variables."""
        return sum([v.size for v in self.sub_vars])

    def parse(self, mdg: pp.MixedDimensionalGrid) -> Any:
        """Returns a sequence of values stored for each variable in :attr:`sub_vars`."""
        raise TypeError(
            "Md-variables parsed on a md-grid without the equation system."
            + " Use ``value(equation_system)`` instead."
        )

    def previous_timestep(self, steps: int = 1) -> MixedDimensionalVariable:
        """Mixed-dimensional variables have sub-variables which also need to be
        obtained at the previous time step."""

        op = super().previous_timestep(steps=steps)
        op.sub_vars = [var.previous_timestep(steps=steps) for var in self.sub_vars]
        return op

    def previous_iteration(self, steps: int = 1) -> MixedDimensionalVariable:
        """Mixed-dimensional variables have sub-variables which also need to be
        obtained at the previous iteration."""
        op = super().previous_iteration(steps=steps)
        op.sub_vars = [var.previous_iteration(steps=steps) for var in self.sub_vars]
        return op


class Projection(Operator):
    """Wrapper class for Ad representations of projection operators."""

    def __init__(
        self,
        domain_indices: np.ndarray,
        range_indices: np.ndarray,
        domain_size: int,
        range_size: int,
        name: Optional[str] = None,
    ):
        """Construct a projection operator.

        Parameters:
            domain_indices: Indices of the domain space.
            range_indices: Indices of the range space.
            domain_size: Size of the domain space.
            range_size: Size of the range space.
            name: Name of the operator. Default is None.

        """
        self._slicer = pp.matrix_operations.MatrixSlicer(
            domain_indices=domain_indices,
            range_indices=range_indices,
            range_size=range_size,
            domain_size=domain_size,
        )
        super().__init__(name=name)

    def transpose(self) -> None:
        """Return the transpose of the operator."""

        return Projection(
            domain_indices=self._slicer._range_indices,
            range_indices=self._slicer._domain_indices,
            range_size=self._slicer._domain_size,
            domain_size=self._slicer._range_size,
            name=self.name,
        )

    def __repr__(self) -> str:
        s = "Projcetion operator"
        if self._name is not None and len(self._name) > 0:
            s += f" named {self._name}"
        s += ".\n"
        s += f"The projection maps from {self._slicer._domain_size} to "
        s += f"{self._slicer._range_indices.size} dimensions.\n"
        s += f"The projection maps {self._slicer._domain_indices.size} elements.\n"
        if self._slicer._is_transposed:
            s += "The operator is transposed."
        return s

    def _key(self) -> str:
        if self._cached_key is None:
            s = f"(prolongation, range_indices={self._slicer._range_indices})"
            s += f", domain_size={self._slicer._domain_indices}"
            s += f", range_size={self._slicer._range_size}"
            if self._slicer._is_transposed:
                s += ", transposed"
            self._cached_key = s
        return self._cached_key

    def __getattr__(self, name: str) -> Projection:
        if name == "T":
            return self.transpose()
        else:
            raise AttributeError(f"Prolongation has no attribute {name}")

    def is_transposed(self) -> bool:
        return self._slicer._is_transposed

    def parse(self, mdg: pp.MixedDimensionalGrid) -> pp.matrix_operations.MatrixSlicer:
        """Convert the Ad expression into a projection operator.

        Parameters:
            mdg: Not used, but needed for compatibility with the general parsing method
                for Operators.

        Returns:
            Projection operator.

        """
        return self._slicer


class ProjectionList(Operator):
    """Wrapper class for a list of projection operators that are to be summed.

    Objects of this class will usually be created by invoking the method
    `pp.ad.sum_projection_list`. Though it is possible to create ProjectionList objects
    directly, *this is not recommended*. Should you choose to do so, be very careful
    with the input, and verify that the AdParser treats the object correctly.

    Motivation:
        The MatrixSlicer objects that underly the Projection objects cannot be combined
        into a single object, the way one can combine projections represented as sparse
        matrices. Thus, expressions of the type ``(P1 + P2) @ x``, where P1 and P2 are
        projections are not directly permissible. Still, it is useful to be allowed to
        combine projections this way.

        The ProjectionList, together with its treatment in the AdParser's
        _evaluate_single() method, offers a workaround that allows for summing (but
        *only* summing) projection operators in the manner described in the previous
        paragraph.

    """

    def __init__(self, operators: list[Projection], name: Optional[str] = None):
        super().__init__(name=name, children=operators)

    def _key(self) -> str:
        if self._cached_key is None:
            self._cached_key = f"(slicing_operator_list, operators={self.children})"
        return self._cached_key

    def __repr__(self) -> str:
        return f"Slicing operator list with {len(self.children)} operators."

    def parse(
        self, mdg: pp.MixedDimensionalGrid
    ) -> list[pp.matrix_operations.MatrixSlicer]:
        """Parse the list items."""
        return [op.parse(mdg) for op in self.children]

    def __getitem__(self, key: int) -> Projection:
        """Enable indexing of the list. This is not needed in operational mode, but is
        useful for testing and development."""
        return self.children[key]


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


def sum_projection_list(
    # This cannot be list[Projection], since list items can be multiplications.
    operators: list[Operator],
    name: Optional[str] = None,
) -> Operator:
    """Sum a list of projection operators.

    This method should only be called if the input list to be summed consists
    exclusively of Projection objects, or of products of precisely two Projection
    objects that have been (matrix) multiplied. For different use cases (such as
    suming Projection objects multiplied with other types of operators), the standard
    sum_operator_list method should be used instead.

    Parameters:
        operators: List of projection operators to be summed. name: Name of the
        resulting operator.

    Raises:
        ValueError: If not one of the following two cases is met:
            1. operators is a list of one or more Projection objects.
            2. operators is a list of one or more products of precisely two Projection
                objects that have been (matrix) multiplied.

    Returns:
        Operator that is the sum of the input operators.

    """
    # First check if this is a sum of atomic projection operators.
    is_projection = [isinstance(op, Projection) for op in operators]
    if any(is_projection):
        if not all(is_projection):
            # This is a mix of slicing and non-slicing operators. This is not allowed.
            raise ValueError("Cannot sum slicing and non-slicing operators.")
        if len(operators) == 1:
            # If there is only one operator, there is no need to put it in a list.
            result = operators[0]
        else:
            # We need the list.
            result = ProjectionList(operators, name)
    else:
        # This else covers the case of one or more products of precisily two projections
        # that have been multiplied. While more cases in principle could be covered,
        # this is the only one that is currently relevant.
        #
        # NOTE TO FUTURE SELF: If it at some point becomes tempting to add more cases,
        # consider if this is really the right approach to take, or if the approach to
        # constructing operator trees should be revisited.
        new_operators = []
        for op in operators:
            # Check that this is a case we can handle.
            if not op.operation == Operations.matmul:
                raise ValueError(f"Operator {op} is not a valid matmul operation.")
            if not isinstance(op.children[0], Projection) or not isinstance(
                op.children[1], Projection
            ):
                raise ValueError(
                    f"Operator {op} does not have valid Projection children."
                )

            # The trick here is to do a local parsing of the two Projections to fetch
            # their underlying MatrixSlicer objects. Then we multiply them and set the
            # combined slicer to the second child (because this is what works together
            # with the way matrix MatrixSlicer objects are matrix multiplied).
            child_1 = op.children[1]
            slicer_0 = op.children[0].parse(None)
            slicer_1 = child_1.parse(None)
            prod = slicer_0 @ slicer_1
            child_1._slicer = prod
            # Child is now a representation of the combined projection.
            new_operators.append(child_1)

        if len(new_operators) == 1:
            # No need to wrap in a list if there is only one operator.
            result = new_operators[0]
        else:
            # We do need the list.
            result = ProjectionList(new_operators, name)

    return result
