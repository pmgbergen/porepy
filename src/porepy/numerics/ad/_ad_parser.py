"""This is a helper module for parsing and evaluating operators in automatic
differentiation.

Regarding testing, there are at the moment no tests explicitly for this module. However,
the functionality is thoroughly tested through the test suit for the models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sps

import porepy as pp
from typing import overload, Literal
from .operators import _Operations


class AdParser:
    """Helper class used for parsing and evaluating operators in automatic
    differentiation.

    This class is not meant to be accessed directly (for instance in a model), but
    rather invoked through the evaluation of an operator in an equation system.

    """

    def __init__(self, mdg: pp.MixedDimensionalGrid) -> None:
        self._mdg = mdg
        """MixedDimensionalGrid on which the operators are defined."""

        self._cache: dict[pp.ad.Operator, Any] = {}
        """Cache for parsed operators. This is used to avoid re-parsing the same
        operator multiple times. The cache is cleared after each evaluation.

        Efficient use of caching has turned out to be difficult to achieve, and the
        cache is at the moment used sparingly. This will be revisited in the future.
        """

    # Since the methods value and value_and_jacobian accept both single operators and
    # lists of operators, we need to overload them to get proper type checking.
    @overload
    def value(
        self, op: pp.ad.Operator, eq_sys: pp.ad.EquationSystem, state: np.ndarray | None
    ) -> np.ndarray: ...

    @overload
    def value(
        self,
        op: list[pp.ad.Operator],
        eq_sys: pp.ad.EquationSystem,
        state: np.ndarray | None,
    ) -> list[np.ndarray]: ...

    def value(
        self,
        op: pp.ad.Operator | list[pp.ad.Operator],
        eq_sys: pp.ad.EquationSystem,
        state: np.ndarray | None,
    ) -> np.ndarray | list[np.ndarray]:
        """Get the value (but not the Jacobian) of the operator op.

        Parameters:
            op: The operator, or list of operators, to evaluate.
            eq_sys: The EquationSystem wherein the system state is defined.
            state: The state of the system. If not provided, the state is taken from the
                variable values provided by eq_sys.

        Returns:
            The value of the operator op, or a list of values if op is a list.

        """
        return self._evaluate(op, derivative=False, eq_sys=eq_sys, state=state)

    @overload
    def value_and_jacobian(
        self, op: pp.ad.Operator, eq_sys: pp.ad.EquationSystem, state: np.ndarray | None
    ) -> pp.ad.AdArray: ...

    @overload
    def value_and_jacobian(
        self,
        op: list[pp.ad.Operator],
        eq_sys: pp.ad.EquationSystem,
        state: np.ndarray | None,
    ) -> list[pp.ad.AdArray]: ...

    def value_and_jacobian(
        self,
        op: pp.ad.Operator | list[pp.ad.Operator],
        eq_sys: pp.ad.EquationSystem,
        state: np.ndarray | None,
    ) -> pp.ad.AdArray | list[pp.ad.AdArray]:
        """Get the value and Jacobian of the operator op.

        Parameters:
            op: The operator, or list of operators, to evaluate.
            eq_sys: The EquationSystem wherein the system state is defined.
            state: The state of the system. If not provided, the state is taken from the
                variable values provided by eq_sys.

        Returns:
            An ArArray representation of the operator op, or a list of AdArrays if op is
                a list.

        """

        return self._evaluate(op, derivative=True, eq_sys=eq_sys, state=state)

    def clear_cache(self) -> None:
        """Clear the cache of parsed operators."""
        self._cache = {}

    @overload
    def _evaluate(
        self,
        op: pp.ad.Operator,
        derivative: Literal[True],
        eq_sys: pp.ad.EquationSystem,
        state: np.ndarray | None,
    ) -> pp.ad.AdArray: ...

    @overload
    def _evaluate(
        self,
        op: pp.ad.Operator,
        derivative: Literal[False],
        eq_sys: pp.ad.EquationSystem,
        state: np.ndarray | None,
    ) -> np.ndarray: ...

    @overload
    def _evaluate(
        self,
        op: list[pp.ad.Operator],
        derivative: Literal[True],
        eq_sys: pp.ad.EquationSystem,
        state: np.ndarray | None,
    ) -> list[pp.ad.AdArray]: ...

    @overload
    def _evaluate(
        self,
        op: list[pp.ad.Operator],
        derivative: Literal[False],
        eq_sys: pp.ad.EquationSystem,
        state: np.ndarray | None,
    ) -> list[np.ndarray]: ...

    def _evaluate(
        self,
        op: pp.ad.Operator | list[pp.ad.Operator],
        derivative: bool,
        eq_sys: pp.ad.EquationSystem,
        state: np.ndarray | None,
    ) -> np.ndarray | pp.ad.AdArray | list[np.ndarray] | list[pp.ad.AdArray]:
        """Evaluate the operator x and its derivative if requested.

        A forward mode automatic differentiation is used to evaluate the operator op.

        Parameters:
            op: The operator, or list of operators, to evaluate.
            derivative: If True, the value and the derivative of the operator is
                returned. If False, only the value of the operator is returned.
            eq_sys: The EquationSystem wherein the system state is defined.
            state: The state of the system. If not provided, the state is taken from the
                variable values provided by eq_sys.

        Returns:
            The value, or value and Jacobian combined in an AdArray, of the operator op,
                or a list of values, or AdArrays if op is a list.

        """

        # Get the state of the system, if not provided.
        if state is None:
            state = eq_sys.get_variable_values(iterate_index=0)

        # Create an AdArray representation of the state, if the derivative is requested.
        # If not, the state is used as is (as a numpy array).
        ad_base = pp.ad.initAdArrays([state])[0] if derivative else state

        if isinstance(op, list):
            result = [self._evaluate_single(o, ad_base, eq_sys) for o in op]
        else:
            result = self._evaluate_single(op, ad_base, eq_sys)

        # Clear the cache after each evaluation. For the moment, this seems like the
        # safest option, although it should be possible to safely cache some results
        # also between evaluations.
        self.clear_cache()
        return result

    def _evaluate_single(
        self,
        op: pp.ad.Operator,
        ad_base: np.ndarray | pp.ad.AdArray,
        eq_sys: pp.EquationSystem,
    ) -> np.ndarray | pp.ad.AdArray:
        """Evaluate a single operator.

        Parameters:
            op: The operator to evaluate.
            ad_base: The base for the automatic differentiation. This should be an
                AdArray if the derivative is requested, and a numpy array if not.
            eq_sys: The EquationSystem wherein the system state is defined.

        Returns:
            A numpy array or an AdArray representation of the operator op, depending on
                whether the derivative is requested.

        """
        # If the operator is in the cache, return the cached value.
        if op in self._cache:
            cached = self._cache[op]
            return cached

        # The operator is not in the cache. Parse the operator by recursion:
        # 1. If the operator is a leaf (has no children), parse the leaf.
        # 2. If the operator is a composite operator, parse the children and combine
        #    them according to the operator.

        if op.is_leaf():
            if isinstance(op, pp.ad.MixedDimensionalVariable):
                if op.is_previous_iterate or op.is_previous_time:
                    # Empty vector like the global vector of unknowns for prev time/iter
                    # insert the values at the right dofs and slice.
                    vals = np.empty_like(
                        ad_base.val if isinstance(ad_base, pp.ad.AdArray) else ad_base
                    )
                    # List of indices for sub variables.
                    dofs = []
                    for sub_var in op.sub_vars:
                        sub_dofs = eq_sys.dofs_of([sub_var])
                        vals[sub_dofs] = sub_var.parse(eq_sys.mdg)
                        dofs.append(sub_dofs)

                    return vals[np.hstack(dofs, dtype=int)] if dofs else np.array([])
                else:
                    # Fetch the values from the state vector.
                    return ad_base[eq_sys.dofs_of([op])]

            # Atomic variables.
            elif isinstance(op, pp.ad.Variable):
                # If a variable represents a previous iteration or time, parse values.
                if op.is_previous_iterate or op.is_previous_time:
                    return op.parse(eq_sys.mdg)
                # Otherwise use the current time and iteration values.
                else:
                    return ad_base[eq_sys.dofs_of([op])]
            # All other leafs like discretizations or some wrapped data.
            else:
                # Mypy complains because the return type of parse is Any.
                res = op.parse(eq_sys.mdg)  # type:ignore
                # Profiling indicated that in this case, caching actually pays off, so
                # we keep it.
                self._cache[op] = res
                return res

        # This is not a leaf, but a composite operator. Parse the children and combine
        # them according to the operator.
        child_values = [
            self._evaluate_single(child, ad_base, eq_sys) for child in op.children
        ]

        # Get the operation represented by op.
        operation = op.operation
        match operation:
            case _Operations.add | _Operations.sub:
                # Addition and subtraction can be handled rather straightforwardly,
                # though with some tweaks for subtraction.

                assert len(child_values) == 2  # These operations are binary

                # Take note of whether the operands are flipped.
                flipped = False
                if isinstance(child_values[0], np.ndarray):
                    # We should not do numpy_array {+,-,*} Ad_array, since numpy will
                    # interpret this in a strange way. Instead switch the order of the
                    # operands. For subtraction, we need to negate the result below.
                    child_values = child_values[::-1]
                    flipped = True
                try:
                    symbol = _Operations.to_symbol(operation)
                    res = eval(f"child_values[0] {symbol} child_values[1]")

                except ValueError as exc:
                    msg = self._get_error_message(
                        _Operations.to_str(operation), op, child_values
                    )
                    raise ValueError(msg) from exc

                # Implementation note: If we in the future want to cache the result
                # based on some logic (e.g. type of operation or size of the arrays
                # involved), we should do it here.
                if operation == _Operations.sub and flipped:
                    # We need to negate the result if we subtract two numpy arrays
                    # and switched the order of the operands in the above if.
                    return -res
                else:
                    return res

            case (
                _Operations.mul | _Operations.div | _Operations.pow | _Operations.matmul
            ):
                # Multiplication, division, power and matrix multiplication can in most
                # cases be handled in the same way. However, some special cases need to
                # be handled separately, see below.

                # Division, power, and matrix multiplication are binary operations
                assert len(child_values) == 2

                if isinstance(child_values[0], np.ndarray) and isinstance(
                    child_values[1], (pp.ad.AdArray, pp.ad.forward_mode.AdArray)
                ):
                    if operation == _Operations.mul:
                        # In the implementation of multiplication between an AdArray and
                        # a numpy array (in the forward mode Ad), a * b and b * a do not
                        # commute. Flip the order of the results to get the expected
                        # behavior. This is permissible, since the elementwise product
                        # commutes.
                        child_values = child_values[::-1]

                    # If the first operand is a numpy array and the second is an
                    # AdArray, numpy will make the operation in a strange way, using its
                    # broadcasting logic. Instead enforce the operation to be done by
                    # the AdArray, using the relevant right-operation method.
                    try:
                        if operation == _Operations.div:
                            return child_values[1].__rtruediv__(child_values[0])
                        elif operation == _Operations.pow:
                            return child_values[1].__rpow__(child_values[0])
                        elif operation == _Operations.matmul:
                            return child_values[1].__rmatmul__(child_values[0])
                        # NOTE: Operations.mul will pass through this if-else and be
                        # evaluated together with the other standard cases in the next
                        # try-except.

                    except ValueError as exc:
                        msg = self._get_error_message(
                            _Operations.to_str(operation), op, child_values
                        )
                        raise ValueError(msg) from exc
                try:
                    symbol = _Operations.to_symbol(operation)
                    res = eval(f"child_values[0] {symbol} child_values[1]")
                    return res
                except ValueError as exc:
                    msg = self._get_error_message(
                        _Operations.to_str(operation), op, child_values
                    )
                    raise ValueError(msg) from exc

            case _Operations.evaluate:
                # Operator functions should have at least 1 child (themselves).
                assert len(child_values) >= 1, (
                    "Operator functions must have at least 1 child."
                )
                assert hasattr(op, "func"), (
                    f"Operators with operation {operation} must have a functional"
                    + " representation `func` implemented as a callable member."
                )
                try:
                    res = op.func(*child_values)
                    return res
                except Exception as exc:
                    msg = "Error while parsing operator function:\n"
                    msg += self._parse_readable(op)
                    raise ValueError(msg) from exc

            case _:
                raise ValueError(f"Encountered unknown operation {operation}")

    def _get_error_message(
        self, operation: str, op: pp.ad.Operator, results: list
    ) -> str:
        """Get a human-readable error message related to a parsing error.

        Parameters:
            operation: The operation that failed.
            op: The operator that failed.
            results: The results of the parsing of the operator.

        Returns:
            A human-readable error message.

        """

        # Helper function to format error message
        msg_0 = self._parse_readable(op.children[0])
        msg_1 = self._parse_readable(op.children[1])

        nl = "\n"
        msg = f"Ad parsing: Error when {operation}\n\n"
        # First give name information. If the expression under evaluation is c = a + b,
        # the below code refers to c as the intended result, and a and b as the first
        # and second argument, respectively.
        msg += "Information on names given to the operators involved: \n"
        if len(op.name) > 0:
            msg += f"Name of the intended result: {op.name}\n"
        else:
            msg += "The intended result is not named\n"
        if len(op.children[0].name) > 0:
            msg += f"Name of the first argument: {op.children[0].name}\n"
        else:
            msg += "The first argument is not named\n"
        if len(op.children[1].name) > 0:
            msg += f"Name of the second argument: {op.children[1].name}\n"
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

    def _parse_readable(self, op: pp.ad.Operator) -> str:
        """Make a human-readable error message related to an operator.

        Parameters:
            op: The operator to parse.

        Returns:
            A human-readable error message.

        """

        # There are three cases to consider: Either the operator is a leaf, it is a
        # composite operator with a name, or it is a general composite operator.
        if op.is_leaf():
            # Leafs are represented by their strings.
            return str(self)
        elif op._name is not None:
            # Composite operators that have been given a name (possibly with a goal of
            # simple identification of an error).
            return op._name

        # General operator. Split into its parts by recursion.
        child_str = [self._parse_readable(child) for child in op.children]

        if op.operation == _Operations.evaluate:
            # Function evaluations have their own readable representation.
            msg = f"{child_str[0]}("
            msg += ", ".join([f"{child}" for child in child_str[1:]])
            msg += ")"
            return msg

        # String representation of the operator.
        operator_str = _Operations.to_symbol(op.operation)
        # If operation is unknown, a new error will be raised to raise awareness.
        if operator_str == "unknown":
            msg = "UNKNOWN parsing of operation on: "
            msg += ", ".join([f"{child}" for child in child_str])
            raise NotImplementedError(msg)
        # Error message for known Operations.
        else:
            return f"({child_str[0]} {operator_str} {child_str[1]})"
