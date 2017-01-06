"""
Module for partitioning of grids based on various methods.

Intended support is by Cartesian indexing, and METIS-based.

"""

import pymetis
import numpy as np

from core.grids import structured


def partition_metis(g, num_part):
    """
    Partition a grid using metis.

    This function requires that pymetis is installed, as can be done by

        pip install pymetis

    This will install metis itself in addition to the python bindings. There
    are other python bindings for metis as well, but pymetis has done the job
    until now.

    Parameters:
        g: core.grids.grid: To be partitioned. Only the cell_face attribute is
            used
        num_part (int): Number of partitions.

    Returns:
        np.array (size:g.num_cells): Partition vector, one number in
            [0, num_part) for each cell.

    """

    # Create a copy of the cell-face relation, so that we can modify it at will
    cell_face = g.cell_faces.copy()

    # Direction of normal vector does not matter here, only 0s and 1s
    cell_faces.data = np.abs(cell_faces.data)

    # Find connection between cells via the cell-face map
    c2c = cell_faces.transpose() * cell_faces
    # Only care about absolute values
    c2c.data = np.clip(c2c.data, 0, 1)

    # Convert the cells into the format required by pymetis
    adjacency_list = [c2c.getrow(i).indices for i in range(c2c.shape[0])]
    # Call pymetis
    part = pymetis.part_graph(10, adjacency=adjacency_list)

    # The meaning of the first number returned by pymetis is not clear (poor
    # documentation), only return the partitioning.
    return np.array(part[1])

