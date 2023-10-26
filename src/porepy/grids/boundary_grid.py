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
from porepy.numerics.linalg.matrix_operations import sparse_kronecker_product


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

        self.num_cells: int = np.sum(
            self._parent.tags["domain_boundary_faces"], dtype=int
        )
        """Number of cells in the boundary grid.

        Corresponds to the number of faces on the domain boundary in the parent grid.

        """
        self._projections: sps.csr_matrix
        """Projection matrix from the parent grid to the boundary grid.

        Initialized in :meth:`~set_projections`.

        """
        self.cell_centers: np.ndarray
        """Cell centers of the boundary grid.

        Subset of the face centers of the parent grid. Initialized in
        :meth:`~compute_geometry`.

        """

        self.cell_volumes: np.ndarray
        """Volumes of cells of the boundary grid. Remember that boundary grid cells are
        faces of the parent grid. Thus, it stores areas of boundary faces.

        Initialized in :meth:`~compute_geometry`.

        """

    def compute_geometry(self) -> None:
        """Compute the geometry of the boundary grid.

        Compute the cell centers and volumes of the boundary grid. By default, the
        boundary grid cell information is constructed from the domain boundary faces of
        theparent grid.

        """
        parent_boundary = self._parent.tags["domain_boundary_faces"]
        self.cell_centers = self._parent.face_centers[:, parent_boundary]
        self.cell_volumes = self._parent.face_areas[parent_boundary]

    def set_projections(self) -> None:
        """Set projections from the parent grid and set the corresponding attributes.

        The parent grid can be modified during its construction, and the boundary grid
        must reflect these changes. It is required to call this method after having
        split the fracture faces in order to finish the initialization of
        the boundary grid.

        """
        sz = self.num_cells
        parent_boundary = self._parent.tags["domain_boundary_faces"]
        if not sz == np.sum(parent_boundary):
            raise NotImplementedError(
                "The number of boundary cells does not match the number of boundary "
                "faces in the parent grid, as is assumed in this implementation."
            )
        self._projections = sps.coo_matrix(
            (np.ones(sz), (np.arange(sz), np.where(parent_boundary)[0])),
            shape=(self.num_cells, self._parent.num_faces),
        ).tocsr()

    def projection(self, nd: int = 1) -> sps.spmatrix:
        """Projection matrix from the parent grid to the boundary grid.

        The projection matrix is a sparse matrix, with  ``shape=(num_cells * nd,
        num_faces_parent * nd)``, which maps face-wise values on the parent grid to the
        boundary grid.

        Parameters:
            nd: ``default=1``. Spatial dimension of the projected quantity. Defaults to
                1 (mapping for scalar quantities). Higher integer values for projection
                of vector-valued quantities.

        """
        return sparse_kronecker_product(matrix=self._projections, nd=nd)

    @property
    def parent(self) -> pp.Grid:
        """Subdomain associated with this boundary grid."""
        return self._parent

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
