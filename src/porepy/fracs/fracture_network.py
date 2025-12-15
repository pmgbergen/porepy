from __future__ import annotations

from abc import ABC, abstractmethod
import warnings
from typing import Optional, Union, cast, TYPE_CHECKING, Literal
import gmsh
import porepy as pp
import numpy as np
from pathlib import Path

# Custom typings
FractureList = Optional[
    list[pp.LineFracture] | list[pp.PlaneFracture | pp.EllipticFracture]
]


class FractureNetwork(ABC):
    """Abstract base class for fracture networks."""

    def __init__(
        self,
        nd: Literal[2, 3],
        fractures: Optional[FractureList] = None,
        domain: Optional[pp.Domain] = None,
        tol: float = 1e-8,
    ) -> None:
        self.nd = nd
        """Number of spatial dimensions (2 or 3)."""

        self.fractures = []
        """List of fractures forming the network."""
        if fractures is not None:
            for f in fractures:
                self.fractures.append(f)

        self.domain: Optional[pp.Domain] = domain
        """Domain specification for the fracture network."""

        self._tol = tol
        """Tolerance for geometric computations."""

    @abstractmethod
    def domain_to_gmsh(self) -> None:
        """Define the domain in gmsh."""
        pass

    @abstractmethod
    def fractures_to_gmsh(self) -> None:
        """Define the fractures in gmsh."""
        pass

    @abstractmethod
    def mesh(
        self,
        mesh_args: dict[str, float],
        file_name: Optional[Path] = None,
        constraints: Optional[np.ndarray] = None,
        dfn: bool = False,
        tags_to_transfer: Optional[list[str]] = None,
        write_geo: bool = True,
        finalize_gmsh: bool = True,
        clear_gmsh: bool = False,
        **kwargs,
    ) -> pp.MixedDimensionalGrid:
        pass

    def _set_background_mesh_field(self, gmsh_fields: list[int]) -> None:
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", gmsh_fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        # The background mesh incorporates all mesh size specifications. We turn off
        # other mesh size specifications.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
