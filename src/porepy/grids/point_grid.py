import numpy as np
import scipy.sparse as sps

from porepy.grids.grid import Grid


class PointGrid(Grid):

    # ------------------------------------------------------------------------------#

    def __init__(self, pt, name=None):
        """
        Constructor for 0D grid

        Parameters
            pt (np.array): Point which represent the grid
            name (str): Name of grid, passed to super constructor
        """

        name = "PointGrid" if name is None else name

        face_nodes = sps.identity(1, np.int, "csr")
        cell_faces = sps.identity(1, np.int, "csr")
        pt = np.asarray(pt).reshape((3, 1))

        super(PointGrid, self).__init__(0, pt, face_nodes, cell_faces, name)


# ------------------------------------------------------------------------------#
