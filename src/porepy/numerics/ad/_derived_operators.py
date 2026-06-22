"""This module contains mixin classes that extend the behavior of an operator.

The mixins will modify how an operator is evaluated:
  - TimeDependentOperator: The operator represents a previous time step, and its
    value will be fetched from the time step which it represents, see class
    documentation for details.
  - IterativeOperator: The operator represents a previous iterate, and its value
    will be fetched from the iterate which it represents, see class documentation for
    details.
  - ReferenceOperator: The operator represents a reference value and will evaluate this
    reference.


"""

import copy


class TimeDependentOperator:
    """Mixin class for operator classes, which can have a time-dependent
    representation.

    Implements the notion of time step indices, as well as a method to create a
    representation of an operator instance at a previous time.

    Operators created via constructor always start at the current time. To create an
    operator representing a previous time step, use the :meth:`previous_timestep`
    method.

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

        - None indicates the current time (unknown value)
        - 0 indicates this is an operator at the first previous time step
        - 1 at the time step before
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

        Time-dependent operators do not invoke the recursion (like the base class), but
        represent a leaf in the recursion tree.

        Note:
            You cannot create operators at the previous time step from operators which
            are at some previous iterate. Use the :attr:`original_operator` instead.

        Parameters:
            steps: ``default=1``

                Number of steps backwards in time. If steps=0, the current time is
                represented.

        Raises:
            ValueError: If this instance represents an operator at a previous iterate.
            ValueError: If ``steps`` is not non-negative.

        """
        if isinstance(self, IterativeOperator):
            if self.is_previous_iterate:
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
        # The public time step index is NONE for current time
        # (which translates to -1 for the private index)
        op._time_step_index = self._time_step_index + int(steps)

        # Keeping track of the original operator.
        if self.is_current_iterate:
            op.original_operator = self
        else:
            op.original_operator = self.original_operator

        return op


class IterativeOperator:
    """Mixin class for operator classes, which can have multiple representations in the
    iterative sense.

    Implements the notion of iterate indices, as well as a method to create a
    representation of an operator instance at a iterate time.

    Operators created via constructor always start at the current iterate.

    Note:
        Operators which represents some previous iterate represent also always the
        current time.

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
        - 0 represents the most recently computed iterate.
        - 1 represents the iterate before that
        - ...

        Note:
            Operators at current time (unknown value) also have the index 0, since those
            values are used to linearize the system and construct the Jacobian.

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

        Iterative operators do not invoke the recursion (like the base class),
        but represent a leaf in the recursion tree.

        Note:
            You cannot create operators at the previous iterates from operators which
            are at some previous time step. Use the :attr:`original_operator` instead.

        Parameters:
            steps: ``default=1``

                Number of steps backwards in the iterate sense. If ``steps`` is 0, the
                current iterate is returned.

        Raises:
            ValueError: If this instance represents an operator at a previous time step.
            ValueError: If ``steps`` is not non-negative.

        """
        if isinstance(self, TimeDependentOperator):
            if self.is_previous_time:
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
            op.original_operator = self
        else:
            op.original_operator = self.original_operator

        return op


class ReferenceOperator:
    """Mixin class for operator classes, which can have a reference representation."""

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
        if isinstance(self, TimeDependentOperator) and self.is_previous_time:
            return self
        if isinstance(self, IterativeOperator) and self.is_previous_iterate:
            return self

        # TODO copy or deepcopy? Is this enough for every operator class?
        op = copy.copy(self)
        # Delete the cached key, so that this must be regenerated for the new operator,
        # which is different from the original one.
        op._cached_key = None
        op._is_reference = True

        if (not hasattr(self, "original_operator")) or self.original_operator is None:
            op.original_operator = self
        else:
            op.original_operator = self.original_operator

        return op
