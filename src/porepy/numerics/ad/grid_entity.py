"""GridEntity enum for identifying grid entities (cells, faces, nodes)."""

from __future__ import annotations

import enum

__all__ = ["GridEntity"]


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
