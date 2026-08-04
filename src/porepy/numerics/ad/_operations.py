from __future__ import annotations
from enum import Enum


from typing import TYPE_CHECKING
from porepy.numerics.ad.operator_space import OperatorSpace, DomainType

if TYPE_CHECKING:
    from porepy.numerics.ad.operator import Operator


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
        self,
        left: Operator,
        right: Operator | int | float,
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
        inherits the non-scalar space. Plain Python scalars (``int``, ``float``) are
        treated as the scalar space.

        Validation is skipped whenever either operand's space is ``None``, so operators
        that carry no space information are fully supported.

        Parameters:
            left: The left operand. right: The right-hand-side operand.

        Returns:
            A 2-tuple ``(source, target)`` where ``source`` is the inferred
            :class:`OperatorSpace` for the source and ``target`` is the inferred
            :class:`OperatorSpace` for the target.

        Raises:
            ValueError: If both operands have specified spaces that are incompatible.

        """
        # A Scalar operator always parses to a plain Python float (see
        # `Scalar.parse`), regardless of whether it was constructed with a
        # `domains` argument. Such domain-bearing scalars carry a non-scalar
        # `OperatorSpace` (see `Scalar.__init__`) purely for provenance/error
        # message purposes, but are numerically compatible with (broadcastable
        # against) any other operand, exactly like a "true" scalar space.
        left_is_scalar = left.source.domain_type == DomainType.scalar
        right_is_scalar = right.source.domain_type == DomainType.scalar

        if self == Operations.matmul:
            # left @ right: target(right) must equal source(left)
            if left.source.domain_type == DomainType.unclear:
                raise ValueError(
                    f"Cannot matrix multiply with {left!r} as the left operand: "
                    "its source is unclear."
                )
            if not right_is_scalar and not self._spaces_compatible(
                left.source, right.target
            ):
                raise ValueError(
                    f"Incompatible matrix multiplication: the target of {right!r} "
                    f"({right.target}) does not match the source of {left!r} "
                    f"({left.source})."
                )
            return right.source, left.target
        elif self == Operations.rmatmul:
            # right @ left (dispatched as left.__rmatmul__(right)):
            # target(left) must equal source(right)
            if left.source.domain_type == DomainType.unclear:
                raise ValueError(
                    f"Cannot matrix multiply with {left!r} as the right operand: "
                    "its source is unclear."
                )
            if not right_is_scalar and not self._spaces_compatible(
                right.source, left.target
            ):
                raise ValueError(
                    f"Incompatible matrix multiplication: the target of {left!r} "
                    f"({left.target}) does not match the source of {right!r} "
                    f"({right.source})."
                )
            return left.source, right.target
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
                    raise ValueError(
                        f"Incompatible operator targets: {left.target} vs {right.target}."
                    )
                return (
                    self._pick_source(left.source, right.source),
                    self._pick_target(left.target, right.target),
                )

    def _is_cellwise_scalar(self, space: OperatorSpace) -> bool:
        """Return True if space represents exactly one DOF per grid entity.

        Such a space numerically broadcasts against any other space defined on the
        same grids *and the same grid entity* (e.g. cells). This mirrors the
        broadcast already granted to :class:`Scalar` operators, but applies to any
        operator whose computed space happens to carry a single DOF per entity. This
        does *not* make e.g. a cell-based space broadcastable against a face-based
        space: the grid entity keys must still match (see ``_spaces_compatible``).

        """
        return len(space.dof_info) == 1 and set(space.dof_info.values()) == {1}

    def _spaces_compatible(self, a: OperatorSpace, b: OperatorSpace) -> bool:
        """Return True if a and b represent the same operator space.

        Two spaces defined on an empty set of grids are considered compatible
        regardless of their exact domain type, since they both carry zero actual
        degrees of freedom (e.g. a discretization defined on an empty list of
        interfaces, as can happen for well couplings in a model without wells,
        must be compatible with a genuinely scalar operator).

        """
        if a == b:
            return True
        if len(a.grids) == 0 and len(b.grids) == 0:
            return True
        if (
            a.domain_type == b.domain_type
            and a.grids == b.grids
            and set(a.dof_info.keys()) == set(b.dof_info.keys())
            and len(a.dof_info) == 1
        ):
            if self._is_cellwise_scalar(a) or self._is_cellwise_scalar(b):
                return True
        return False

    def _is_vacuous(self, space: OperatorSpace) -> bool:
        """Return True if the space carries no grids, and hence no actual dofs.

        This includes the scalar space, but also spaces with a non-scalar
        domain_type that happen to be defined on an empty grid list.

        """
        return len(space.grids) == 0

    def _pick_target(self, a: OperatorSpace, b: OperatorSpace) -> OperatorSpace:
        """Return the known space when one side is unspecified.

        When one operand is a cellwise-scalar broadcast (see
        ``_is_cellwise_scalar``), the result should carry the *other*
        operand's (non-broadcast) space, since that is where the actual
        degrees of freedom of the result live.

        """
        if self._is_cellwise_scalar(a) and not self._is_cellwise_scalar(b):
            return b
        if self._is_cellwise_scalar(b) and not self._is_cellwise_scalar(a):
            return a
        return a

    def _pick_source(self, a: OperatorSpace, b: OperatorSpace) -> OperatorSpace:
        if a.domain_type == DomainType.unclear or b.domain_type == DomainType.unclear:
            return OperatorSpace.unclear()
        if a != b:
            if self._is_vacuous(a) and self._is_vacuous(b):
                # Both spaces carry no actual dofs, so their exact domain type is
                # immaterial; arbitrarily keep the left operand's space.
                return a
            if (
                a.domain_type == b.domain_type
                and a.grids == b.grids
                and set(a.dof_info.keys()) == set(b.dof_info.keys())
                and len(a.dof_info) == 1
            ):
                # Same grids/domain type/entity key, differing only in the
                # per-entity DOF count (e.g. one side is a cellwise-scalar
                # broadcast): resolve the same way as the target, keeping the
                # non-broadcast space when possible.
                return self._pick_target(a, b)
            return OperatorSpace.unclear()
        return a
