import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.grids.grid import Grid

module_sections = ["grids", "gridding"]


class PointGrid(Grid):
    @pp.time_logger(sections=module_sections)
    def __init__(self, pt: np.ndarray, name: str = None) -> None:
        """
        Constructor for 0D grid

        Parameters
            pt (np.array): Point which represent the grid.
            name (str): Name of grid, passed to super constructor.

        """

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
        self.cell_centers = pt

        super(PointGrid, self).__init__(0, nodes, face_nodes, cell_faces, name)
