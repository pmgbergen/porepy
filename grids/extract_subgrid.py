"""
Extract a part of a grid, with
"""
import numpy as np
import scipy.sparse as sps

from core.grids.grid import Grid


def extract_subgrid(g, c):
    """
    Extract a subset a subgrid based on cell indices.abs

    For simplicity the cell indices will be sorted before the subgrid is
    extracted.

    If the parent grid has geometry attributes (cell centers etc.) these are
    copied to the child.

    No checks are done on whether the cells form a connected area. The method
    should work in theory for non-connected cells, the user will then have to
    decide what to do with the resulting grid. This option has however not been
    tested.

    Parameters:
        g (core.grids.Grid): Grid object, parent
        c (np.array, dtype=int): Indices of cells to be extracted

    Returns:
        core.grids.Grid: Extracted subgrid
        np.ndarray, dtype=int: Index of the extracted faces, ordered so that
            element i is the global index of face i in the subgrid.
        np.ndarray, dtype=int: Index of the extracted nodes, ordered so that
            element i is the global index of node i in the subgrid.
    """
    c.sort()

    # Local cell-face and face-node maps.
    cf_sub, unique_faces = __extract_submatrix(g.cell_faces, c)
    fn_sub, unique_nodes = __extract_submatrix(g.face_nodes, unique_faces)

    # Append information on subgrid extraction to the new grid's history
    name = list(g.name)
    name.append('Extract subgrid')

    # Construct new grid.
    h = Grid(g.dim, g.nodes[:, unique_nodes], fn_sub, cf_sub, name)

    # Copy geometric information if any
    if hasattr(g, 'cell_centers'):
        h.cell_centers = g.cell_centers[:, c]
    if hasattr(g, 'cell_volumes'):
        h.cell_volumes = g.cell_volumes[c]
    if hasattr(g, 'face_centers'):
        h.face_centers = g.face_centers[:, unique_faces]
    if hasattr(g, 'face_normals'):
        h.face_normals = g.face_normals[:, unique_faces]
    if hasattr(g, 'face_areas'):
        h.face_areas = g.face_areas[unique_faces]

    return h, unique_faces, unique_nodes

def __extract_submatrix(mat, ind):
    """ From a matrix, extract the column specified by ind. All zero columns
    are stripped from the sub-matrix. Mappings from global to local row numbers
    are also returned.
    """
    sub_mat = mat[:, ind]
    cols = sub_mat.indptr
    rows = sub_mat.indices
    data = sub_mat.data
    unique_rows, rows_sub = np.unique(sub_mat.indices,
                                      return_inverse=True)
    return sps.csc_matrix((data, rows_sub, cols)), unique_rows

