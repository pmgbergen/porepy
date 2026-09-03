"""GridEntity enum for identifying grid entities (cells, faces, nodes), and
GridEntities, an immutable value object representing DOF counts per grid entity.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Mapping, Union

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


@dataclasses.dataclass(frozen=True)
class GridEntities:
    """Number of degrees of freedom (DOFs) per grid entity.

    An immutable, hashable value object with one field per :class:`GridEntity`
    member, accessed by name (``.cells``, ``.faces``, ``.nodes``).

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

    def __bool__(self) -> bool:
        """True if DOFs are located on at least one grid entity."""
        return bool(self.cells or self.faces or self.nodes)

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
            ValueError: If ``dof_info`` contains a key that is not a
                :class:`GridEntity` (e.g. a plain string).

        Returns:
            A ``GridEntities`` instance.

        """
        if isinstance(dof_info, GridEntities):
            return dof_info
        kwargs = {}
        for entity, count in dof_info.items():
            if not isinstance(entity, GridEntity):
                raise ValueError(
                    f"Non-admissible DOF type {entity!r} in dof_info; expected a "
                    "GridEntity member (e.g. GridEntity.cells), not "
                    f"{type(entity).__name__}."
                )
            kwargs[entity.value] = count
        return cls(**kwargs)

    @property
    def present_entities(self) -> frozenset[GridEntity]:
        """The grid entities that carry a nonzero number of DOFs."""
        return frozenset(
            entity for entity in GridEntity if getattr(self, entity.value) != 0
        )

    def is_unit_on_single_entity(self) -> bool:
        """True if DOFs are located on exactly one grid entity, with one DOF per
        entity.

        Such a DOF distribution describes a quantity that numerically broadcasts
        against any other quantity defined on the same grids and the same entity.

        """
        counts = (self.cells, self.faces, self.nodes)
        return counts.count(1) == 1 and counts.count(0) == 2
