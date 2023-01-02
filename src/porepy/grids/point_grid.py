"""Module for creating point grids."""

from typing import Optional

import numpy as np
import scipy.sparse as sps

from porepy.grids.grid import Grid


class PointGrid(Grid):
    """Representation of a 0D grids.

    Parameters:
        pt: ``shape=(3,)``

           Point which represents the grid.
        name: ``default=None``

           Name of grid, passed to super constructor.

    """

    def __init__(self, pt: np.ndarray, name: Optional[str] = None) -> None:
        # check input
        if np.asarray(pt).shape[0] != 3:
            raise ValueError("PointGrid: points must be given in 3 dimensions")

        pt = np.atleast_2d(pt)
        if pt.shape[0] == 1:  # point is given as 1d array
            if pt.shape[1] != 3:
                raise ValueError(
                    "PointGrid: 1d point arrays only allowed for single points"
                )
            pt = pt.T

        name = "PointGrid" if name is None else name

        face_nodes = sps.identity(0, int, "csr")
        cell_faces = sps.csr_matrix((0, pt.shape[1]), dtype=int)

        nodes = np.zeros((3, 0))
        self.cell_centers: np.ndarray = pt

        super().__init__(0, nodes, face_nodes, cell_faces, name)
