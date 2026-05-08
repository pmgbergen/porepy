from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Sequence
from enum import Enum

import porepy as pp

if TYPE_CHECKING:
    from porepy.utils.porepy_types import GridLike, GridLikeSequence
    from ._grid_entity import GridEntity


__all__ = [
    "DomainType",
    "OperatorSpace",
]


class DomainType(Enum):
    """Type of a function space domain or range.

    Describes whether the grids associated with an :class:`OperatorSpace`.
    """

    subdomains = "subdomains"
    interfaces = "interfaces"
    boundary_grids = "boundary_grids"
    scalar = "scalar"
    unclear = "unclear"
    """Used for composits formed by operators with different domains."""


@dataclasses.dataclass(eq=False)
class OperatorSpace:
    """Represents the mathematical domain or range of an AD operator.

    An ``OperatorSpace`` is characterized by:

    - A :class:`DomainType` indicating the kind of grids.
    - A tuple of grids over which the space is defined.
    - A ``dof_info`` dictionary mapping each :class:`~porepy.numerics.ad.GridEntity`
      to the number of degrees of freedom *per grid entity*.  For example,
      ``{GridEntity.cells: 1}`` means one DOF per cell.

    Use the class methods :meth:`scalar` and :meth:`from_domains` to construct
    instances instead of calling the constructor directly.

    """

    domain_type: DomainType
    """The type of the space (subdomains, interfaces, boundary_grids, or scalar)."""

    grids: tuple[pp.Grid | pp.MortarGrid | pp.BoundaryGrid, ...]
    """Grids that define the space."""

    dof_info: dict[GridEntity, int]
    """Number of DOFs per grid entity for each entity type present in the space."""

    def __post_init__(self) -> None:
        self.grids = tuple(self.grids)

        if self.domain_type in (DomainType.scalar, DomainType.unclear):
            if self.grids:
                raise ValueError(
                    f"{self.domain_type.value.capitalize()} spaces cannot have grids."
                )
            if self.dof_info:
                s = (
                    f"{self.domain_type.value.capitalize()} spaces cannot have"
                    " dof_info."
                )
                raise ValueError(s)
            self.dof_info = {}
            return

        if not self.grids:
            raise ValueError(
                f"{self.domain_type.value.capitalize()} spaces must define grids."
            )
        if not self.dof_info:
            raise ValueError(
                f"{self.domain_type.value.capitalize()} spaces must define dof_info."
            )
        self.dof_info = dict(self.dof_info)

        if self.domain_type == DomainType.subdomains and not all(
            isinstance(g, pp.Grid) for g in self.grids
        ):
            raise ValueError("Subdomain spaces must be defined on pp.Grid objects.")
        if self.domain_type == DomainType.interfaces and not all(
            isinstance(g, pp.MortarGrid) for g in self.grids
        ):
            raise ValueError(
                "Interface spaces must be defined on pp.MortarGrid objects."
            )
        if self.domain_type == DomainType.boundary_grids and not all(
            isinstance(g, pp.BoundaryGrid) for g in self.grids
        ):
            raise ValueError(
                "Boundary-grid spaces must be defined on pp.BoundaryGrid objects."
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OperatorSpace):
            return NotImplemented
        return (
            self.domain_type == other.domain_type
            and self.grids == other.grids
            and self.dof_info == other.dof_info
        )

    def __hash__(self) -> int:
        return hash((self.domain_type, self.grids, frozenset(self.dof_info.items())))

    @classmethod
    def scalar(cls) -> OperatorSpace:
        """Return the trivial (scalar / zero-dimensional) operator space."""
        return cls(DomainType.scalar, (), {})

    @classmethod
    def unclear(cls) -> OperatorSpace:
        """Return a sentinel space for operators with no clear mathematical domain."""
        return cls(DomainType.unclear, (), {})

    @classmethod
    def from_domains(
        cls,
        domains: Sequence[pp.Grid | pp.MortarGrid | pp.BoundaryGrid],
        dof_info: dict[GridEntity, int],
    ) -> OperatorSpace:
        """Construct an :class:`OperatorSpace` from a sequence of grids.

        Parameters:
            domains: Sequence of grid objects.  All grids must be of the same
                type (all :class:`~porepy.Grid`, all :class:`~porepy.MortarGrid`,
                or all :class:`~porepy.BoundaryGrid`).
            dof_info: Mapping from :class:`~porepy.numerics.ad.GridEntity` to
                the number of DOFs per entity.

        Returns:
            A new :class:`OperatorSpace`.

        Raises:
            ValueError: If ``domains`` contains a mix of grid types.

        """
        if len(domains) == 0:
            return cls.scalar()
        grids = tuple(domains)
        if all(isinstance(g, pp.Grid) for g in grids):
            domain_type = DomainType.subdomains
        elif all(isinstance(g, pp.MortarGrid) for g in grids):
            domain_type = DomainType.interfaces
        elif all(isinstance(g, pp.BoundaryGrid) for g in grids):
            domain_type = DomainType.boundary_grids
        else:
            raise ValueError(
                "All grids in `domains` must have the same type (pp.Grid, "
                "pp.MortarGrid, or pp.BoundaryGrid)."
            )
        return cls(domain_type, grids, dict(dof_info))
