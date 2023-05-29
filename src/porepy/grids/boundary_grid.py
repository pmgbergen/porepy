"""This module contains a subclass of the standard grid class which represents the
boundary of a domain in the form of a grid. The intention is to use this class to set
boundary conditions in cases where constitutive laws are defined on the boundary.

"""
from __future__ import annotations

from itertools import count
from typing import Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp


class BoundaryGrid:
    """A grid representing the boundary of a domain.

    The BoundaryGrid is generated from a parent grid, and represents those faces in the
    parent which are on the boundary of the computational domain (i.e. those with tag
    'domain_boundary_faces' set to True).

    The boundary grid is intended for computing boundary conditions, e.g. when
    constitutive laws must be evaluated on the boundary.

    Note:
        The BoundaryGrid class is an experimental feature which is subject to major
        changes, hereunder deletion.

    """

    _counter = count(0)
    """Counter of instantiated grids. See :meth:`__new__` and :meth:`id`."""
    __id: int
    """Name-mangled reference to assigned ID."""

    def __new__(cls, *args, **kwargs) -> BoundaryGrid:
        """Make object and set ID by forwarding :attr:`_counter`."""

        obj = object.__new__(cls)
        obj.__id = next(cls._counter)
        return obj

    def __init__(self, g: pp.Grid, name: Optional[str] = None) -> None:
        if name is None:
            name = "boundary_grid"
        self.name: str = name
        """Name of the grid. """

        self._parent = g
        """Parent grid from which the boundary grid is constructed."""

        if g.dim == 0:
            raise ValueError("Boundary grids are not supported for 0d grids.")
        self.dim = g.dim - 1
        """Dimension of the boundary grid."""

        self.num_cells: int
        """Number of cells in the boundary grid.

        Will correspond to the number of faces on the domain boundary in the parent
        grid. Initialized in :meth:`~compute_geometry`.

        """
        self._projections: sps.csr_matrix
        """Projection matrix from the parent grid to the boundary grid.

        Initialized in :meth:`~compute_geometry`.

        """
        self.cell_centers: np.ndarray
        """Cell centers of the boundary grid.

        Subset of the face centers of the parent grid. Initialized in
        :meth:`~compute_geometry`.

        """

    def compute_geometry(self) -> None:
        """Compute geometric quantities for the boundary grid.

        Currently, there are no heavy computations here. However, the parent grid can
        be modified during its construction, and the boundary grid must reflect these
        changes. Thus, it is required to call this method after calling
        :meth:`Grid.compute_geometry`. Usually, this is done by the MDG.

        """
        parent_boundary = self._parent.tags["domain_boundary_faces"]

        self.num_cells = int(np.sum(parent_boundary))
        self.cell_centers = self._parent.face_centers[:, parent_boundary]

        sz = self.num_cells
        self._projections = sps.coo_matrix(
            (np.ones(sz), (np.arange(sz), np.where(parent_boundary)[0])),
            shape=(self.num_cells, self._parent.num_faces),
        ).tocsr()

    @property
    def projection(self) -> sps.spmatrix:
        """Projection matrix from the parent grid to the boundary grid.

        The projection matrix is a sparse matrix,
        with  ``shape=(num_cells, num_faces_parent)``,
        which maps face-wise values on the parent grid to the boundary grid.

        """
        return self._projections

    @property
    def id(self):
        """BoundaryGrid ID.

        The returned attribute must not be changed.
        This may severely compromise other parts of the code,
        such as sorting in md grids.

        The attribute is set in :meth:`__new__`.
        This avoids calls to the super constructor in child classes.

        """
        return self.__id

    def __repr__(self) -> str:
        """Get a string representation of the boundary grid.

        Returns:
            A string representation of the boundary grid.

        """
        geometry_computed = hasattr(self, "num_cells")
        if geometry_computed:
            return (
                f"Boundary grid of dimension {self.dim} "
                f"containing {self.num_cells} cells.\n"
                f"ID of parent grid: {self._parent.id}.\n"
                "Dimension of the projection from the parent grid: "
                f"{self._projections.shape}."
            )

        return (
            f"Boundary grid of dimension {self.dim}.\n"
            f"ID of parent grid: {self._parent.id}.\n"
            "Geometry has not been computed yet."
        )

    def __str__(self) -> str:
        return self.__repr__()
