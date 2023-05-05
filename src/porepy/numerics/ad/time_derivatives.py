"""The module contains functionality for time-differentiation of operator trees.

The module contains the following functions:
    dt: Take the time derivative of an operator tree (first-order approximation).
    time_increment: Find the time increment of an operator tree.

"""
from __future__ import annotations

import porepy as pp

__all__ = ["dt", "time_increment"]


def dt(op: pp.ad.Operator, time_step: pp.ad.Scalar) -> pp.ad.Operator:
    """Approximate the time-derivative of an operator tree.

    This time derivative is obtained by taking a first-order finite difference.

    The operator tree at the previous time step is created as a shallow copy, and will
    thus be identical to the original operator, except that all time dependent operators
    are evaluated at the previous time step.

    If the time-dependent quantity q is already evaluated at the previous time step, its
    derivative will be defined as (q(time=n-1) - q(time=n-1)) / dt = 0.

    Parameters:
        op: Operator tree to be differentiated.
        time_step: Size of time step.

    Returns:
        A first-order approximation of the time derivative of op.

    """
    # Return the first-order approximation of the time derivative.
    return time_increment(op) / time_step


def time_increment(op: pp.ad.Operator) -> pp.ad.Operator:
    """Find the time increment of an operator tree.

    Parameters:
        op: Operator tree for which we need the time increment.

    Returns:
        The difference of the operator tree now and at the previous time step.

    """
    return op - op.previous_timestep()
