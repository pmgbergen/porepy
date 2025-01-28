from __future__ import annotations

import numpy as np
import porepy as pp
import heapq
import networkx as nx

import scipy.sparse as sps


class AdParser:
    
    def __init__(self, mdg: pp.MixedDimensionalGrid) -> None:
        self._mdg = mdg
        self._cache = {}
        self._alt_cache = {}

    def value(self, x: pp.ad.Operator | list[pp.ad.Operator], eq_sys: pp.ad.EquationSystem, state: np.ndarray | None) -> np.ndarray:
        return self._evaluate(x, derivative=False, eq_sys=eq_sys, state=state)

    def value_and_jacobian(self, x: pp.ad.Operator | list[pp.ad.Operator], eq_sys: pp.ad.EquationSystem, state: np.ndarray) -> pp.ad.AdArray:
        return self._evaluate(x, derivative=True, eq_sys=eq_sys, state=state)

    def clear_cache(self):
        self._cache = {}

    # @profile
    def _evaluate(self, x: pp.ad.Operator, derivative: bool, eq_sys: pp.ad.EquationSystem, state: np.ndarray | None) -> np.ndarray | pp.ad.AdArray:
        """Evaluate the operator x and its derivative if requested.

        A forward mode automatic differentiation is used to evaluate the operator x.

        State is normally provided by a call of the form
            state = eq_sys.get_variable_values(iterate_index=0)
        where eq_sys is an instance of a pp.ad.EquationSystem.

        """

        if state is None:
            state = eq_sys.get_variable_values(iterate_index=0)

        if not isinstance(x, list):
            x = [x]
        
        # What will happen here if state is None? On the other hand, it should be
        # possible to evaluate the operator without a state, if the operator does not
        # depend on the state (e.g. it is a numpy array).
        ad_base = pp.ad.initAdArrays([state])[0] if derivative else state

        results = []
        for op in x:
            results.append(self._alt_eval(op, ad_base, eq_sys))

        return results

    def _alt_eval(self, op, ad_base, eq_sys):
        if op in self._alt_cache:
            cached = self._alt_cache[op]
            return cached

        if op.is_leaf():
            if isinstance(op, pp.ad.MixedDimensionalVariable):
                if op.is_previous_iterate or op.is_previous_time:
                    # Empty vector like the global vector of unknowns for prev time/iter
                    # insert the values at the right dofs and slice
                    vals = np.empty_like(
                        ad_base.val if isinstance(ad_base, pp.ad.AdArray) else ad_base
                    )
                    # list of indices for sub variables
                    dofs = []
                    for sub_var in op.sub_vars:
                        sub_dofs = eq_sys.dofs_of([sub_var])
                        vals[sub_dofs] = sub_var.parse(eq_sys.mdg)
                        dofs.append(sub_dofs)

                    res = vals[np.hstack(dofs, dtype=int)] if dofs else np.array([])
                    # self._alt_cache[op] = res
                    return res
                # Like for atomic variables, ad_base contains current time and iter
                else:
                    res = ad_base[eq_sys.dofs_of([op])]
                    # self._alt_cache[op] = res
                    return res                    
            # Case 2.b) atomic variables
            elif isinstance(op, pp.ad.Variable):
                # If a variable represents a previous iteration or time, parse values.
                if op.is_previous_iterate or op.is_previous_time:
                    res= op.parse(eq_sys.mdg)
                    # self._alt_cache[op] = res
                    return res                    
                # Otherwise use the current time and iteration values.
                else:
                    res= ad_base[eq_sys.dofs_of([op])]
                    # self._alt_cache[op] = res
                    return res                    
            # Case 2.c) All other leafs like discretizations or some wrapped data
            else:
                # Mypy complains because the return type of parse is Any.
                res= op.parse(eq_sys.mdg)  # type:ignore
                self._alt_cache[op] = res
                return res                
        
        child_values = [self._alt_eval(child, ad_base, eq_sys) for child in op.children]
    
        # Get the operation represented by op.
        operation = op.operation

        # TODO: Since the operation is brought into use outside of the operator class,
        # it should probably be promoted to an independent class.
        if operation == pp.ad.Operator.Operations.add:
            # To add we need two objects
            assert len(child_values) == 2

            if isinstance(child_values[0], np.ndarray):
                # We should not do numpy_array + Ad_array, since numpy will interpret
                # this in a strange way. Instead switch the order of the operands and
                # everything will be fine.
                child_values = child_values[::-1]
            try:
                # An error here would typically be a dimension mismatch between the
                # involved operators.
                res = child_values[0] + child_values[1]
                self._alt_cache[op] = res
                return res                
            except ValueError as exc:
                msg = self._get_error_message("adding", op.children, child_values)
                raise ValueError(msg) from exc

        elif operation == pp.ad.Operator.Operations.sub:
            # To subtract we need two objects
            assert len(child_values) == 2

            # We need a minor trick to take care of numpy arrays.
            factor = 1.0
            if isinstance(child_values[0], np.ndarray):
                # We should not do numpy_array - Ad_array, since numpy will interpret
                # this in a strange way. Instead switch the order of the operands, and
                # switch the sign of factor to compensate.
                child_values = child_values[::-1]
                factor = -1.0
            try:
                # An error here would typically be a dimension mismatch between the
                # involved operators.
                res = factor * (child_values[0] - child_values[1])
                # self._alt_cache[op] = res
                return res                
            except ValueError as exc:
                msg = self._get_error_message("subtracting", op.children, child_values)
                raise ValueError(msg) from exc

        elif operation == pp.ad.Operator.Operations.mul:
            # To multiply we need two objects
            assert len(child_values) == 2

            if isinstance(child_values[0], np.ndarray) and isinstance(
                child_values[1], (pp.ad.AdArray, pp.ad.forward_mode.AdArray)
            ):
                # In the implementation of multiplication between an AdArray and a
                # numpy array (in the forward mode Ad), a * b and b * a do not
                # commute. Flip the order of the results to get the expected behavior.
                # This is permissible, since the elementwise product commutes.
                child_values = child_values[::-1]
            try:
                # An error here would typically be a dimension mismatch between the
                # involved operators.
                res = child_values[0] * child_values[1]
                # self._alt_cache[op] = res
                return res                
            except ValueError as exc:
                msg = self._get_error_message("multiplying", op.children, child_values)
                raise ValueError(msg) from exc

        elif operation == pp.ad.Operator.Operations.div:
            # Some care is needed here, to account for cases where item in the results
            # array is a numpy array
            try:
                if isinstance(child_values[0], np.ndarray) and isinstance(
                    child_values[1], (pp.ad.AdArray, pp.ad.forward_mode.AdArray)
                ):
                    # If numpy's __truediv__ method is called here, the result will be
                    # strange because of how numpy works. Instead we directly invoke the
                    # right-truedivide method in the AdArary.
                    res = child_values[1].__rtruediv__(child_values[0])
                    # self._alt_cache[op] = res
                    return res                    
                else:
                    res = child_values[0] / child_values[1]
                    # self._alt_cache[op] = res
                    return res                    
            except ValueError as exc:
                msg = self._get_error_message("dividing", op.children, child_values)
                raise ValueError(msg) from exc

        elif operation == pp.ad.Operator.Operations.pow:
            try:
                if isinstance(child_values[0], np.ndarray) and isinstance(
                    child_values[1], (pp.ad.AdArray, pp.ad.forward_mode.AdArray)
                ):
                    # If numpy's __pow__ method is called here, the result will be
                    # strange because of how numpy works. Instead we directly invoke the
                    # right-power method in the AdArary.
                    res = child_values[1].__rpow__(child_values[0])
                    # self._alt_cache[op] = res
                    return res                    
                else:
                    res= child_values[0] ** child_values[1]
                    # self._alt_cache[op] = res
                    return res                    
            except ValueError as exc:
                msg = self._get_error_message(
                    "raising to a power", op.children, child_values
                )
                raise ValueError(msg) from exc

        elif operation == pp.ad.Operator.Operations.matmul:
            try:
                if isinstance(child_values[0], np.ndarray) and isinstance(
                    child_values[1], (pp.ad.AdArray, pp.ad.forward_mode.AdArray)
                ):
                    # Again, we do not want to call numpy's matmul method, but instead
                    # directly invoke AdArarray's right matmul.
                    res = child_values[1].__rmatmul__(child_values[0])
                    # self._alt_cache[op] = res
                    return res                    
                # elif isinstance(results[1], np.ndarray) and isinstance(
                #     results[0], (pp.ad.AdArray, pp.ad.forward_mode.AdArray)
                # ):
                #     # Again, we do not want to call numpy's matmul method, but instead
                #     # directly invoke AdArarray's right matmul.
                #     return results[0].__rmatmul__(results[1])
                else:
                    res = child_values[0] @ child_values[1]
                    # self._alt_cache[op] = res
                    return res                    
            except ValueError as exc:
                msg = self._get_error_message(
                    "matrix multiplying", op.children, child_values
                )
                raise ValueError(msg) from exc

        elif operation == pp.ad.Operator.Operations.evaluate:
            # Operator functions should have at least 1 child (themselves)
            assert len(child_values) >= 1, "Operator functions must have at least 1 child."
            assert hasattr(op, "func"), (
                f"Operators with operation {operation} must have a functional"
                + f" representation `func` implemented as a callable member."
            )

            try:
                res = op.func(*child_values)
                # self._alt_cache[op] = res
                return res                
            except Exception as exc:
                # TODO specify what can go wrong here (Exception type)
                msg = "Error while parsing operator function:\n"
                msg += op._parse_readable()
                raise ValueError(msg) from exc

        else:
            raise ValueError(f"Encountered unknown operation {operation}")


    def _get_error_message(
        self, operation: str, children: list[pp.ad.Operator], results: list
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
