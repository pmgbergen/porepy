"""GridEntity enum for identifying grid entities (cells, faces, nodes), and
GridEntities, an immutable value object representing DOF counts per grid entity.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Iterator, Mapping, Union

__all__ = ["GridEntity", "GridEntities"]


class GridEntity(enum.Enum):
    """Enumeration of grid entities (cells, faces, nodes).

    This enum is used to identify what kind of grid entity a variable or
    equation is defined on.

    Members:
        cells: Degrees of freedom located at cell centres.
        faces: Degrees of freedom located at cell faces.
        nodes: Degrees of freedom located at grid nodes (vertices).
    """

    cells = "cells"
    faces = "faces"
    nodes = "nodes"


@dataclasses.dataclass(frozen=True, eq=False)
class GridEntities:
    """Number of degrees of freedom (DOFs) per grid entity.

    An immutable value object that behaves like a read-only
    ``Mapping[GridEntity, int]`` (supporting :meth:`get`, :meth:`items`, :meth:`keys`,
    :meth:`values`, ``len()``, iteration, ``in`` and ``[]``), while also giving named
    attribute access (``.cells``, ``.faces``, ``.nodes``). Only entities with a
    nonzero count are considered "present" by the Mapping-like interface. This matches
    the convention previously used for ``dict[GridEntity, int]``-typed ``dof_info``,
    where an entity simply wasn't a dict key if its count was 0: an absent entity and
    an explicit zero count are treated identically here too.

    Equality (and thus set/dict-key usage) is also supported against a plain
    ``Mapping[GridEntity, int]`` (e.g. a ``dict`` literal).

    Use :meth:`from_mapping` to construct an instance from a plain
    ``dict[GridEntity, int]`` (or an existing ``GridEntities``, returned unchanged).

    """

    cells: int = 0
    """Number of DOFs per cell."""

    faces: int = 0
    """Number of DOFs per face."""

    nodes: int = 0
    """Number of DOFs per node."""

    def __post_init__(self) -> None:
        for entity in GridEntity:
            value = getattr(self, entity.value)
            if value < 0:
                raise ValueError(
                    f"Number of DOFs per {entity.value} must be non-negative, got "
                    f"{value}."
                )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GridEntities):
            return (self.cells, self.faces, self.nodes) == (
                other.cells,
                other.faces,
                other.nodes,
            )
        if isinstance(other, Mapping):
            return self == GridEntities.from_mapping(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.cells, self.faces, self.nodes))

    @classmethod
    def from_mapping(
        cls, dof_info: Union[GridEntities, Mapping[GridEntity, int]]
    ) -> GridEntities:
        """Normalize a plain mapping or an existing ``GridEntities`` into a
        ``GridEntities``.

        Parameters:
            dof_info: Either an existing ``GridEntities`` (returned unchanged), or a
                mapping from :class:`GridEntity` to the number of DOFs per that entity
                type. Entity types not present in the mapping default to 0.

        Raises:
            TypeError: If ``dof_info`` contains a key that is not a :class:`GridEntity`
                (e.g. a plain string).

        Returns:
            A ``GridEntities`` instance.

        """
        if isinstance(dof_info, GridEntities):
            return dof_info
        kwargs = {}
        for entity, count in dof_info.items():
            if not isinstance(entity, GridEntity):
                raise TypeError(
                    f"Non-admissible DOF type key {entity!r} in dof_info; expected a "
                    "GridEntity member (e.g. GridEntity.cells), not "
                    f"{type(entity).__name__}."
                )
            kwargs[entity.value] = count
        return cls(**kwargs)

    def get(self, entity: GridEntity, default: int = 0) -> int:
        """Dict-like ``.get()``: the DOF count for ``entity``, or ``default`` if that
        entity is not present (i.e. its count is 0)."""
        value = getattr(self, entity.value)
        return value if value != 0 else default

    def items(self) -> Iterator[tuple[GridEntity, int]]:
        """Dict-like ``.items()``, yielding only entities with a nonzero count."""
        for entity in GridEntity:
            value = getattr(self, entity.value)
            if value != 0:
                yield entity, value

    def keys(self) -> Iterator[GridEntity]:
        """Dict-like ``.keys()``, yielding only entities with a nonzero count."""
        for entity, _ in self.items():
            yield entity

    def values(self) -> Iterator[int]:
        """Dict-like ``.values()``, yielding only nonzero counts."""
        for _, value in self.items():
            yield value

    def __len__(self) -> int:
        """Number of entities with a nonzero count."""
        return sum(1 for _ in self.items())

    def __iter__(self) -> Iterator[GridEntity]:
        return self.keys()

    def __contains__(self, entity: object) -> bool:
        if not isinstance(entity, GridEntity):
            return False
        return getattr(self, entity.value) != 0

    def __getitem__(self, entity: GridEntity) -> int:
        value = getattr(self, entity.value)
        if value == 0:
            raise KeyError(entity)
        return value
