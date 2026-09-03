"""The module contains the Operations enum, which is used to identify the operation
performed by an operator and to validate the source/target spaces of the operands.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from porepy.numerics.ad.operator_space import DomainType, OperatorSpace

if TYPE_CHECKING:
    from .operators import Operator


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

    def infer_source_target(
        self, left: Operator, right: Operator
    ) -> tuple[OperatorSpace, OperatorSpace]:
        """Validate operand spaces and infer the source/target of the result.

        For matrix multiplication (``matmul``), the target of the *right* operand must
        equal the source of the *left* operand (i.e.
        ``target(right) == source(left)`` for ``left @ right``). The result's source is
        ``right.source`` and the result's target is ``left.target``.

        For elementwise operations (``add``, ``sub``, ``mul``, ``div``, ``pow``), both
        operands must have the same source *and* the same target when both are
        specified.

        The scalar space (:meth:`OperatorSpace.scalar`) is compatible with any space, so
        operations with a :class:`Scalar` operator are always valid and the result
        inherits the non-scalar space.

        Validation is skipped whenever either operand's space is ``None``, so operators
        that carry no space information are fully supported.

        Parameters:
            left: The left operand.
            right: The right-hand-side operand.

        Raises:
            ValueError: If the operands have specified spaces that are incompatible.

        Returns:
            A 2-tuple ``(source, target)`` where ``source`` is the inferred
            :class:`OperatorSpace` for the source and ``target`` is the inferred
            :class:`OperatorSpace` for the target.

        """
        left_is_scalar = left.source.domain_type == DomainType.scalar
        right_is_scalar = right.source.domain_type == DomainType.scalar

        if self == Operations.matmul:
            # left @ right: target(right) must equal source(left)
            return self._process_matmul(left, right, right_is_scalar)
        elif self == Operations.rmatmul:
            # right @ left (dispatched as left.__rmatmul__(right)):
            # target(left) must equal source(right)
            return self._process_matmul(right, left, left_is_scalar)
        else:
            # Elementwise operations
            if left_is_scalar and right_is_scalar:
                # Both operands are numerically scalar (broadcastable), but either
                # may still carry a non-scalar, domain-bearing space (see the
                # docstring note on `Scalar` above), e.g. a material property
                # `Scalar` constructed with `domains=subdomains`. When that is the
                # case, the result should inherit that domain information rather
                # than collapsing to the plain scalar space, so that the domain
                # provenance survives arithmetic between domain-bearing scalars.
                left_has_domain = left.source.domain_type != DomainType.scalar
                right_has_domain = right.source.domain_type != DomainType.scalar
                if left_has_domain and right_has_domain:
                    return (
                        self._pick_source(left.source, right.source),
                        self._pick_target(left.target, right.target),
                    )
                elif left_has_domain:
                    return left.source, left.target
                elif right_has_domain:
                    return right.source, right.target
                return OperatorSpace.scalar(), OperatorSpace.scalar()
            elif left_is_scalar:
                return right.source, right.target
            elif right_is_scalar:
                return left.source, left.target
            else:
                # We need compatibility between the targets (since this is where the
                # quantity of interest lives), but the sources can be different.
                if not self._spaces_compatible(left.target, right.target):
                    s = (
                        "Incompatible operator targets:"
                        f" {left.target} vs {right.target}."
                    )
                    raise ValueError(s)
                return (
                    self._pick_source(left.source, right.source),
                    self._pick_target(left.target, right.target),
                )

    def _process_matmul(
        self, first, second, second_is_scalar: bool
    ) -> tuple[OperatorSpace, OperatorSpace]:
        # left @ right: target(right) must equal source(left)
        if first.source.domain_type == DomainType.unclear:
            raise ValueError(
                f"Cannot matrix multiply with {first!r} as the left operand: "
                "its source is unclear."
            )
        if not second_is_scalar and not self._spaces_compatible(
            first.source, second.target
        ):
            raise ValueError(
                f"Incompatible matrix multiplication: the target of {second!r} "
                f"({second.target}) does not match the source of {first!r} "
                f"({first.source})."
            )
        return second.source, first.target

    def _can_broadcast(self, space: OperatorSpace) -> bool:
        """Return True if space represents exactly one DOF per grid entity.

        Such a space numerically broadcasts against any other space defined on the same
        grids *and the same grid entity* (e.g. cells). This mirrors the broadcast of
        :class:`Scalar` operators, but applies to any operator whose computed space
        happens to carry a single DOF per entity.

        Important cases are pure variables and constitutive relations that are defined
        per grid entity (e.g. per cell) and hence have one DOF per entity.

        """
        return space.dof_info.is_unit_on_single_entity()

    def _spaces_compatible(self, a: OperatorSpace, b: OperatorSpace) -> bool:
        """Return True if a and b represent compatible operator spaces."""
        if a == b:
            # Spaces are equal. Fine.
            return True
        if len(a.grids) == 0 and len(b.grids) == 0:
            # None of the spaces carry any grids, so they are considered compatible
            # (even if they have different domain types).
            return True
        if (
            a.domain_type == b.domain_type
            and a.grids == b.grids
            and a.dof_info.present_entities == b.dof_info.present_entities
            and len(a.dof_info.present_entities) == 1
        ):
            # Spaces are not equal, but they are defined on the same grids, same entity
            # type, and one of them is a cellwise scalar, hence we can broadcast the
            # result to the other space.
            if self._can_broadcast(a) or self._can_broadcast(b):
                return True
        return False

    def _pick_target(self, a: OperatorSpace, b: OperatorSpace) -> OperatorSpace:
        """Pick the target space.

        It assumed that the caller has already verified that the two spaces are
        compatible.
        """
        # When one operand is a cellwise-scalar broadcast the result should carry the
        # other operand's (non-broadcast) space, since that is where the actual degrees
        # of freedom of the result live.
        if self._can_broadcast(a) and not self._can_broadcast(b):
            return b
        if self._can_broadcast(b) and not self._can_broadcast(a):
            return a
        return a

    def _pick_source(self, a: OperatorSpace, b: OperatorSpace) -> OperatorSpace:
        """Pick the source space.

        It assumed that the caller has already verified that the two spaces are
        compatible.
        """
        if a.domain_type == DomainType.unclear or b.domain_type == DomainType.unclear:
            return OperatorSpace.unclear()
        if a != b:
            if len(a.grids) == 0 and len(b.grids) == 0:
                # Both spaces carry no actual dofs, so their exact domain type is
                # immaterial; arbitrarily keep the left operand's space.
                return a
            if (
                a.domain_type == b.domain_type
                and a.grids == b.grids
                and a.dof_info.present_entities == b.dof_info.present_entities
                and len(a.dof_info.present_entities) == 1
            ):
                # Same grids/domain type/entity key, differing only in the per-entity
                # DOF count (e.g. one side is a cellwise-scalar broadcast). Keep the
                # non-broadcast space when possible.
                return self._pick_target(a, b)
            # If we reach this point, the spaces are incompatible, but the caller has
            # already verified that they are compatible, so we return unclear.
            return OperatorSpace.unclear()
        return a
