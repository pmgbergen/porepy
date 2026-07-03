"""This module contains mixin classes that extend the behavior of an operator.

The mixins will modify how an operator is evaluated:
  - TimeDependentOperator: The operator represents a previous time step, and its value
    will be fetched from the time step which it represents, see class documentation for
    details.
  - IterativeOperator: The operator represents a previous iterate, and its value will be
    fetched from the iterate which it represents, see class documentation for details.
  - ReferenceOperator: The operator represents a reference value and will evaluate this
    reference.

When combining the mixins, their behavior is prioritized as follows:
  - Calling the method reference() on an operator which is a ReferenceOperator will
    return the operator in a reference state, independent of the original state (default
    state, previous iterate, previous timestep) of the operator.
  - Taking the previous time step or previous iterate of an operator which is a
    ReferenceOperator will return the reference operator itself. Taking the reference is
    a one-way operation.
  - Taking the previous time step of an operator at a previous iterate will raise an
    error.
  - Taking the previous iterate of an operator at a previous time step will raise an
    error.


"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    import porepy as pp
    from porepy.numerics.ad.operators import (
        Operations,
        Operator,
        OperatorSpace,
        _IterativeOperator,
        _ReferenceOperator,
        _TimeDependentOperator,
    )


class TimeDependentOperator:
    """Mixin class for Operator classes, which can have a time-dependent
    representation.

    Implements the notion of time step indices, as well as a method to create a
    representation of an operator instance at a previous time.

    Operators created via constructor always start at the current time. To create an
    operator representing a previous time step, use the :meth:`previous_timestep`
    method.

    """

    is_current_iterate: bool
    """True, if the operator represents the current iterate."""
    _cached_key: Optional[str]

    def __init__(
        self,
        name: str | None = None,
        operation: Optional[Operations] = None,
        children: Optional[Sequence[Operator]] = None,
        *,
        source: Optional[OperatorSpace],
        target: Optional[OperatorSpace],
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            name=name,
            operation=operation,
            children=children,
            source=source,
            target=target,
        )

        self.original_operator: Operator
        """Reference to the operator representing this operator at the current time and
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

        - None indicates the current time (unknown value) or that the operator
          represents a reference value.
        - 0 indicates this is an operator at the first previous time step (the last
          accepted time step).
        - 1 at the time step before that
        - ...

        """
        if isinstance(self, ReferenceOperator):
            if self.is_reference:
                return None

        if self._time_step_index < 0:
            return None
        else:
            return self._time_step_index

    def previous_timestep(
        self: _TimeDependentOperator, steps: int = 1
    ) -> _TimeDependentOperator:
        """Returns a copy of the time-dependent operator with an advanced time-step
        index.

        Note:
            You cannot create operators at the previous time step from operators which
            are at some previous iterate. Use the :attr:`original_operator` instead.

        Parameters:
            steps: ``default=1``

                Number of steps backwards in time. If steps=0, a copy of the operator at
                the current time is represented.

        Raises:
            ValueError: If this instance represents an operator at a previous iterate.
            ValueError: If ``steps`` is negative.

        """
        if isinstance(self, ReferenceOperator) and self.is_reference:
            return self
        if isinstance(self, IterativeOperator) and self.is_previous_iterate:
            raise ValueError(
                "Cannot create an operator representing a previous time step,"
                + " if it already represents a previous iterate."
            )

        if steps < 0:
            raise ValueError("Number of steps backwards must be non-negative.")
        # TODO copy or deepcopy? Is this enough for every operator class?
        op = copy.copy(self)
        # Delete the cached key, so that this must be regenerated for the new operator,
        # which is different from the original one.
        op._cached_key = None

        # NOTE Use private time step index, because it is always an integer.
        # The public time step index is NONE for current time (which translates to -1
        # for the private index).
        op._time_step_index = self._time_step_index + int(steps)

        # Keeping track of the original operator.
        if self.is_current_iterate:
            op.original_operator = self  # type: ignore[assignment]
        else:
            op.original_operator = self.original_operator

        return op


class IterativeOperator:
    """Mixin class for Operator classes, which can have multiple representations in the
    iterative sense.

    Implements the notion of iterate indices, as well as a method to create a
    representation of an operator instance at a iterate time.

    Operators created via constructor always start at the current iterate.

    Note:
        Operators which represents some previous iterate represent also always the
        current time.

    """

    _cached_key: Optional[str]
    """Cached key for the operator, used for hashing."""
    is_current_iterate: bool
    """True, if the operator represents the current iterate."""

    def __init__(
        self,
        name: str | None = None,
        operation: Optional[Operations] = None,
        children: Optional[Sequence[Operator]] = None,
        source: Optional[OperatorSpace] = None,
        target: Optional[OperatorSpace] = None,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            name=name,
            operation=operation,
            children=children,
            source=source,
            target=target,
        )

        self.original_operator: Operator
        """Reference to the operator representing this operator at the current time and
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

        - None indicates this instance is at a previous time or reference.
        - 0 represents the most recently computed iterate (e.g. the last accepted
          iterate of a non-linear solver).
        - 1 represents the iterate before that
        - ...

        Note:
            Operators representing the current iterate (that is, an operator on which
            the method previous_iteration has not been called, hence it represents the
            next iterate to be computed in an iterative scheme) will also have the index
            0, since it will also evaluate to the last accepted iterate. The difference
            is that the current iterate will have a non-zero Jacobian, while the
            previous iterate will not.

        """
        # Operators at previous time have no iterate indices.
        if isinstance(self, TimeDependentOperator):
            if self.is_previous_time:
                return None

        if isinstance(self, ReferenceOperator):
            if self.is_reference:
                return None

        # Operators representing at current time use the values stored at index 0
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

        Note:
            Calling this method with ``steps=1`` on an operator representing the current
            iterate (the next iterate to be computed in an iterative scheme) will return
            an operator representing the last accepted iterate. When evaluated, the
            current and last accepted iterate will return the same values, but only the
            former will have a non-zero Jacobian. To get the difference between the
            current and second to last accepted iterate, call this method with
            ``steps=2``.

        Example:
            To obtain the Newton update of an operator
            ```
            op = ...
            newton_update = op - op.previous_iteration(steps=2)
            newton_update_value = newton_update.value(equation_system)
            ```

        Note:
            You cannot create operators at the previous iterates from operators which
            are at some previous time step. Use the :attr:`original_operator` instead.

        Parameters:
            steps: ``default=1``

                Number of steps backwards in the iterate sense. If ``steps`` is 0, a
                copy of the operator at the same iteration index is returned.

        Raises:
            ValueError: If this instance represents an operator at a previous time step.
            ValueError: If ``steps`` is negative.

        """
        if isinstance(self, ReferenceOperator) and self.is_reference:
            return self
        if isinstance(self, TimeDependentOperator) and self.is_previous_time:
            raise ValueError(
                "Cannot create an operator representing a previous iterate,"
                + " if it already represents a previous time step."
            )
        if steps < 0:
            raise ValueError("Number of steps backwards must be non-negative.")
        # See TODO in TimeDependentOperator.previous_timestep
        op = copy.copy(self)
        # Delete the cached key, so that this must be regenerated for the new operator,
        # which is different from the original one.
        op._cached_key = None
        op._iterate_index = self._iterate_index + int(steps)

        # keeping track to the very first one
        if self.is_current_iterate:
            op.original_operator = self  # type: ignore[assignment]
        else:
            op.original_operator = self.original_operator

        return op


class ReferenceOperator:
    """Mixin class for Operator classes, which can have a reference representation."""

    _cached_key: Optional[str]
    """Cached key for the operator, used for hashing."""

    def __init__(
        self,
        name: str | None = None,
        operation: Optional[Operations] = None,
        children: Optional[Sequence[Operator]] = None,
        source: Optional[OperatorSpace] = None,
        target: Optional[OperatorSpace] = None,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            name=name,
            operation=operation,
            children=children,
            source=source,
            target=target,
        )

        self.original_operator: Operator
        """The original operator representing this operator at the current time and
        iterate.

        """
        self._is_reference: bool = False
        """True if this operator represents a reference value."""

    @property
    def is_reference(self) -> bool:
        """True, if the operator represents a reference."""
        return self._is_reference

    def reference(self: _ReferenceOperator) -> _ReferenceOperator:
        """Returns a copy of the reference operator with an advanced time-step
        index.

        Reference operators do not invoke the recursion (like the base class),
        but represent a leaf in the recursion tree.

        """
        # Currently, only "non-fixed" operators can be evaluated at reference.
        if isinstance(self, ReferenceOperator) and self.is_reference:
            return self
        # TODO copy or deepcopy? Is this enough for every operator class?
        op = copy.copy(self)
        # Delete the cached key, so that this must be regenerated for the new operator,
        # which is different from the original one.
        op._cached_key = None
        op._is_reference = True

        if (not hasattr(self, "original_operator")) or self.original_operator is None:
            op.original_operator = self  # type: ignore[assignment]
        else:
            op.original_operator = self.original_operator

        return op


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
    # Keep reference operators as they are.
    if isinstance(op, ReferenceOperator) and op.is_reference:
        return op
    elif steps == 0:
        return op
    # The recursion reached an atomic operator, which has some time- or
    # iterate-dependent behaviour.
    elif isinstance(op, TimeDependentOperator) and prev_time:
        return op.previous_timestep(steps=steps)
    elif isinstance(op, IterativeOperator) and not prev_time:
        return op.previous_iteration(steps=steps)
    # NOTE The previous_iteration of a time-dependent operator will return the operator
    # itself. Vice-versa, the previous_timestep of an Iterative operator will return
    # itself. Holds only if the operator is original (no previous_* operation performed)

    # The recursion reached an operator without children and without time- or iterate-
    # dependent behaviour.
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


def _get_reference(op: Operator) -> Operator:
    """Helper function for providing correct AD structure for reference operators.

    The reference is taken according to the following prioritized rules:
        1. If the operator has a reference, we return it.
        2. If the operator is a leaf, we return the operator itself.
        3. Else, we copy the operator tree that has the operator as root, but with the
           reference behaviour in all the children.

    Returns:
        A reference operator according to the above rules.

    """
    if isinstance(op, ReferenceOperator):
        return op.reference()
    elif op.is_leaf():
        return op
    else:
        new_op = copy.copy(op)
        new_op.children = [_get_reference(child) for child in op.children]
        return new_op
