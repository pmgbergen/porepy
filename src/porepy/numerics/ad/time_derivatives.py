"""
"""
from __future__ import annotations
import copy
import porepy as pp


__all__ = ["dt"]


def dt(op: pp.ad.Operator, time_step: pp.ad.Scalar) -> pp.ad.Operator:
    """Approximate the time-derivative of an operator tree.

    The operator tree at the previous time step is created as a shalllow copy, and will
    thus be identical to the original operator, except that all time dependent operators
    are evaluated at the previous time step.

    If the time-dependent quantity q is already evaluated at the previous time step, its
    derivative will be defined as (q(time=n-1) - q(time=n-1)) / dt = 0.

    Args:
        op: Operator tree to be time-differentiated.
        dt: Size of time step.

    Returns:
        A first-order approximation of the time derivative of op.

    """
    # Create a copy of the operator tree evaluated at a previous time step. This is done
    # by traversing the underlying graph, and set all time-dependent objects to be
    # evaluated at the previous time step.

    def _traverse_tree(op: pp.ad.Operator) -> pp.ad.Operator:
        """Helper function which traverses an operator tree by recursion."""

        children = op.tree.children

        if len(children) == 0:
            # We are on an atomic operator. If this is a time-dependent operator,
            # set it to be evaluated at the previous time step. If not, leave the
            # operator as it is.
            if isinstance(
                op, (pp.ad.Variable, pp.ad.MergedVariable, pp.ad.TimeDependentArray)
            ):
                # IMPLEMENTATION NOTE: We need to use a copy, not a deep copy here. A
                # deep copy would change the underlying grids and mortar grids. For
                # variables (atomic and merged) this would render a KeyValue if the
                # grid is used to fetch the relevant degree of freedom in the global
                # ordering, as is done by the DofManager.
                return copy.copy(op).previous_timestep()

            else:
                return copy.copy(op)
        else:
            # Recursively iterate over the subtree, get the children, evaluated at the
            # previous time when relevant, and add it to the new list.
            new_children = []
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
            new_tree = pp.ad.operators.Tree(op.tree.op, children=new_children)

            # Use the same lists of subdomains and interfaces as in the old operator,
            # with empty lists if these are not present.
            subdomains = getattr(op, "subdomains", [])
            interfaces = getattr(op, "interfaces", [])

            # Create new operator from the tree.
            new_op = pp.ad.Operator(
                name=op._name,
                subdomains=subdomains,
                interfaces=interfaces,
                tree=new_tree,
            )
            return new_op

    # Get a copy of the operator with all time-dependent quantities evaluated at the
    # previous time step.
    prev_time = _traverse_tree(op)

    # Return the first-order approximation of the time derivative.
    return (op - prev_time) / time_step
