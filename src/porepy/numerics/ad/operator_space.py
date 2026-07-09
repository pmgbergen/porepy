from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Sequence
from enum import Enum

import porepy as pp

from ._grid_entity import GridEntity

if TYPE_CHECKING:
    from porepy.utils.porepy_types import GridLike, GridLikeSequence


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

        if not self.dof_info:
            raise ValueError(
                f"{self.domain_type.value.capitalize()} spaces must define dof_info."
            )
        self.dof_info = dict(self.dof_info)

        # Note: self.grids may legitimately be empty here, e.g. for grid operators
        # (Trace, Divergence, SubdomainProjections, MortarProjections, etc.) that are
        # constructed on an empty list of subdomains/interfaces/boundary grids, but
        # whose domain_type is nonetheless known from the context of construction.
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

    def num_dofs(self) -> int:
        """Total number of degrees of freedom represented by this space.

        Computed as the sum, over all grids in :attr:`grids`, of the number of
        grid entities of each type (cells, faces nodes) times the number of DOFs per
        entity of that type.

        This is used by the shape-consistency check in
        :class:`~porepy.numerics.ad.operators.SparseArray`/
        :class:`~porepy.numerics.ad.operators.DenseArray` to catch
        constructor calls where the supplied space does not actually match the
        shape of the wrapped array/matrix.

        Returns:
            The total number of DOFs.

        Raises:
            ValueError: If this space is :attr:`DomainType.scalar` or
                :attr:`DomainType.unclear`.

        """
        if self.domain_type in (DomainType.scalar, DomainType.unclear):
            raise ValueError(
                f"{self.domain_type.value.capitalize()} spaces have no "
                "grid-based DOF count."
            )
        total = 0
        for grid in self.grids:
            for entity, num_per_entity in self.dof_info.items():
                if entity == GridEntity.cells:
                    total += num_per_entity * grid.num_cells
                elif entity == GridEntity.faces:
                    if isinstance(grid, pp.Grid):
                        total += num_per_entity * grid.num_faces
                    elif num_per_entity:
                        raise ValueError(
                            f"{type(grid).__name__} has no faces, but dof_info "
                            f"specifies {num_per_entity} DOFs per face."
                        )
                elif entity == GridEntity.nodes:
                    if isinstance(grid, (pp.Grid, pp.MortarGrid)):
                        total += num_per_entity * grid.num_nodes
                    elif num_per_entity:
                        raise ValueError(
                            f"{type(grid).__name__} has no nodes, but dof_info "
                            f"specifies {num_per_entity} DOFs per node."
                        )
                elif entity == GridEntity.void:
                    continue
                else:
                    raise ValueError(f"Unknown grid entity {entity}.")
        return total

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
        domains: Sequence[pp.Grid | pp.MortarGrid | pp.BoundaryGrid] | None,
        dof_info: dict[GridEntity, int],
        domain_type: DomainType | None = None,
    ) -> OperatorSpace | None:
        """Construct an :class:`OperatorSpace` from a sequence of grids.

        Parameters:
            domains: Sequence of grid objects.  All grids must be of the same
                type (all :class:`~porepy.Grid`, all :class:`~porepy.MortarGrid`,
                or all :class:`~porepy.BoundaryGrid`).
            dof_info: Mapping from :class:`~porepy.numerics.ad.GridEntity` to
                the number of DOFs per entity.
            domain_type: If given, the returned space is forced to have this
                domain type, and an empty ``domains`` sequence will *not* be
                interpreted as a scalar space. This is needed by grid operators
                (e.g. :class:`~porepy.numerics.ad.grid_operators.Trace`) whose
                domain type is known from context even if they happen to be
                constructed on an empty list of grids.

        Returns:
            A new :class:`OperatorSpace`.

        Raises:
            ValueError: If ``domains`` contains a mix of grid types.

        """
        if domains is None:
            return None
        if domain_type is not None:
            return cls(domain_type, tuple(domains), dict(dof_info))
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
