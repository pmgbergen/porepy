from typing import Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp


class BoundaryGrid(pp.Grid):
    """A grid representing the boundary of a domain.

    The BoundaryGrid is generated from a parent grid, and represents those faces in the
    parent which are on the boundary of the computational domain (i.e. those with tag
    'domain_boundary_faces' set to True).

    The boundary grid is intended as a holder of boundary conditions only.

    Todo:
        The BoundaryGrid is not a full grid, and should not be used as such. The current
        inheritance from Grid is a hack to get the BoundaryGrid to work with methods
        that expect a Grid.
        The natural solution is to make an abc superclass, say, GridBase, which is
        inherited by both Grid and BoundaryGrid, and then use GridBase in type hints
        when either a Grid or a BoundaryGrid can be used.

    Note:
        Boundary grids have a id counter, which is shared with the parent class Grid.
        This may confuse expectations of consecutive numbering of grids, if generation
        of standard grids is interleaved with generation of boundary grids.

    """

    def __init__(self, g: pp.Grid, name: Optional[str] = None) -> None:
        parent_boundary = g.tags["domain_boundary_faces"]

        self.num_cells: int = np.sum(parent_boundary)
        """Number of cells in the boundary grid. Will correspond to the number of
        faces on the domain boundary in the parent grid.

        """

        if name is None:
            name = "boundary_grid"
        self.name: str = name
        """Name of the grid. """

        self._parent = g
        """Parent grid from which the boundary grid is constructed."""

        self.cell_centers = g.face_centers[:, parent_boundary]
        """Cell centers of the boundary grid.

        Subset of the face centers of the parent grid.

        """

        if g.dim == 0:
            raise ValueError("Boundary grids are not supported for 0d grids.")
        self.dim = g.dim - 1
        """Dimension of the boundary grid."""

        sz = self.num_cells
        """Number of cells in the boundary grid."""

        self._proj = sps.coo_matrix(
            (np.ones(sz), (np.arange(sz), np.where(parent_boundary)[0])),
            shape=(self.num_cells, g.num_faces),
        ).tocsr()

    @property
    def projection(self):
        """Projection matrix from the parent grid to the boundary grid.

        The projection matrix is a sparse matrix,
        with  ``shape=(num_cells, num_faces_parent)``,
        which maps face-wise values on the parent grid to the boundary grid.

        """
        return self._proj

    def __repr__(self) -> str:
        """Get a string representation of the boundary grid.

        Returns:
            A string representation of the boundary grid.

        """
        s = f"Boundary grid of dimension {self.dim}\n"
        s += f"ID of parent grid: {self._parent.id}\n"
        s += f"{self.num_cells} cells.\n"
        s += f"Dimension of the projection from the parent grid: {self._proj.shape}"
        return s

    def __str__(self) -> str:
        """Get a string representation of the boundary grid.

        Returns:
            A string representation of the boundary grid.

        """
        s = f"Boundary grid of dimension {self.dim}\n"
        s += f"ID of parent grid: {self._parent.id}\n"
        s += f"{self.num_cells} cells."
        return s
