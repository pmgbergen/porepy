import porepy as pp
import numpy as np
import scipy.sparse as sps


class BoundaryGrid(pp.Grid):
    """A grid representing the boundary of a domain.

    The BoundaryGrid is generated from a parent grid, and represents those faces in the
    parent which are on the boundary of the computational domain (i.e. those with tag
    'domain_boundary_faces' set to True).

    The boundary grid is intended as a holder of boundary conditions only.

    TODO: The BoundaryGrid is not a full grid, and should not be used as such. The
    current inheritance from Grid is a hack to get the BoundaryGrid to work with methods
    that expect a Grid. The natural solution is to make an abc superclass, say,
    GridBase, which is inherited by both Grid and BoundaryGrid, and then use GridBase
    in type hints when either a Grid or a BoundaryGrid can be used.

    """

    def __init__(self, g: pp.Grid, name: str) -> None:
        parent_boundary = g.tags["domain_boundary_faces"]

        self.num_cells: int = np.sum(parent_boundary)
        """Number of cells in the boundary grid. Will correspond to the number of
        faces on the domain boundary in the parent grid.

        """

        self._parent = g

        self.cell_centers = g.face_centers[:, parent_boundary]

        sz = self.num_cells
        self._proj = sps.coo_matrix(
            (np.ones(sz), (np.arange(sz), np.where(parent_boundary)[0])),
            shape=(self.num_cells, g.num_faces),
        ).tocsr()

    @property
    def projection(self):
        return self._proj
