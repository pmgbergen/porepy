from __future__ import annotations

import numpy as np
import porepy as pp
from queue import PriorityQueue
import networkx as nx

import scipy.sparse as sps


class AdParser:
    
    def __init__(self, mdg: pp.MixedDimensionalGrid) -> None:
        self._mdg = mdg
        self._cache = {}

    def value(self, x: pp.ad.Operator, state: np.ndarray | None) -> np.ndarray:
        return self._evaluate(x, derivative=False, state=state)

    def value_and_jacobian(self, x: pp.ad.Operator, state: np.ndarray) -> pp.ad.AdArray:
        return self._evaluate(x, derivative=True, state=state)

    def clear_cache(self):
        self._cache = {}

    def _evaluate(self, x: pp.ad.Operator, derivative: bool, state: np.ndarray | None, eq_sys: pp.ad.EquationSystem) -> np.ndarray | pp.ad.AdArray:
        """Evaluate the operator x and its derivative if requested.

        A forward mode automatic differentiation is used to evaluate the operator x.

        State is normally provided by a call of the form
            state = eq_sys.get_variable_values(iterate_index=0)
        where eq_sys is an instance of a pp.ad.EquationSystem.

        """

        if state is None and derivative:
            raise ValueError("State must be provided when computing the derivative")
        
        ad_base = pp.ad.initAdArrays([state])[0] if derivative else state

        # Keep track of the latest value of the evalutation. This will eventually store
        # the value of the full tree.
        current_val = None

        # Convert the operator to a queue of operations
        queue = self._graph_to_queue(x)

        for item in queue:
            
            if item in self._cache:
                # The value of this item has already been computed
                current_val = self._cache[item]
                continue

            elif item.is_leaf():
                current_val = self._parse_leaf(item, ad_base, eq_sys)
                self._cache[item] = current_val
            elif item.operation is not pp.ad.Operator.Operations.void:
                # This is the result of an operation.
                current_val = self._parse_operation(item, eq_sys)
                self._cache[item] = current_val
            else:
                # Who knows what this is?
                raise ValueError(f"Unknown item {item}")
            
        # We are done. Return.
        return current_val

    def _parse_leaf(self, op: pp.ad.Operator, ad_base, eq_sys) -> np.ndarray | pp.ad.AdArray:
        """Parse a leaf node in the graph."""
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

                return vals[np.hstack(dofs, dtype=int)] if dofs else np.array([])
            # Like for atomic variables, ad_base contains current time and iter
            else:
                return ad_base[eq_sys.dofs_of([op])]
        # Case 2.b) atomic variables
        elif isinstance(op, pp.ad.Variable):
            # If a variable represents a previous iteration or time, parse values.
            if op.is_previous_iterate or op.is_previous_time:
                return op.parse(eq_sys.mdg)
            # Otherwise use the current time and iteration values.
            else:
                return ad_base[eq_sys.dofs_of([op])]
        # Case 2.c) All other leafs like discretizations or some wrapped data
        else:
            # Mypy complains because the return type of parse is Any.
            return op.parse(eq_sys.mdg)  # type:ignore
        
    def _parse_operation(self, op: pp.ad.Operator, eq_sys: pp.ad.EquationSystem) -> np.ndarray | pp.ad.AdArray:

        # Get the children of the operator
        children = [op.nx_graph.nodes[child] for child in op.nx_graph.successors(op)]

        # Get the operand_id for each child and sort children based on this id
        sorted_children = sorted(children, key=lambda child: op.nx_graph.get_edge_data(op, child)['operand_id'])
        
        # Get evaluation of the sorted children. If we get a key error here, something
        # is wrong with the order of evaluation.
        child_values = [self._cache[child] for child in sorted_children]

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
                return child_values[0] + child_values[1]
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
                return factor * (child_values[0] - child_values[1])
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
                return child_values[0] * child_values[1]
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
                    return child_values[1].__rtruediv__(child_values[0])
                else:
                    return child_values[0] / child_values[1]
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
                    return child_values[1].__rpow__(child_values[0])
                else:
                    return child_values[0] ** child_values[1]
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
                    return child_values[1].__rmatmul__(child_values[0])
                # elif isinstance(results[1], np.ndarray) and isinstance(
                #     results[0], (pp.ad.AdArray, pp.ad.forward_mode.AdArray)
                # ):
                #     # Again, we do not want to call numpy's matmul method, but instead
                #     # directly invoke AdArarray's right matmul.
                #     return results[0].__rmatmul__(results[1])
                else:
                    return child_values[0] @ child_values[1]
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
                return op.func(*child_values)
            except Exception as exc:
                # TODO specify what can go wrong here (Exception type)
                msg = "Error while parsing operator function:\n"
                msg += op._parse_readable()
                raise ValueError(msg) from exc

        else:
            raise ValueError(f"Encountered unknown operation {operation}")
        
    def _graph_to_queue(self, x: pp.ad.Operator) -> PriorityQueue:

        tree = nx.bfs_tree(x.nx_graph, source=x)
        nodes_bfs = list(tree.nodes)
        nodes_bfs.reverse()

        # Now, we can iterate over the nodes in the tree in reverse order, and compute the
        # height of each node. 
        height = {}
        for node in nodes_bfs:
            if tree.out_degree(node) == 0:
                height[node] = 0
            else:
                # The height of a node is the maximum height of its children + 1. This will
                # ensure that, when traversing the tree in the order of increasing height,
                # we will always have visited all the children of a node before visiting the
                # node itself.
                # NOTE: Since the nodes are visited in reverse BFS order, we can be sure that
                # all the children of a node have already been visited.
                height[node] = max([height[child] for child in tree.successors(node)]) + 1
            
        # Now, we can use a priority queue to sort the nodes by height. It could be that
        # this is not necessary, and that we can just iterate over the nodes in the reverse
        # BFS order, but EK is not sure.
        queue = PriorityQueue(maxsize=len(height))
        for node, priority in height.items():
            queue.put((priority, node))        

        return queue

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
