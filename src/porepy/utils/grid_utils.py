"""Module contains various utility functions for working with grids.
"""
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.utils.array_operations import sparse_array_to_row_col_data


def switch_sign_if_inwards_normal(
    g: pp.Grid, nd: int, faces: np.ndarray
) -> sps.spmatrix:
    """Construct a matrix that changes sign of quantities on faces with a
    normal that points into the grid.

    Parameters:
        g (pp.Grid): Grid.
        nd (int): Number of quantities per face; this will for instance be the
            number of components in a face-vector.
        faces (np.array-like of ints): Index for which faces to be considered. Should only
            contain boundary faces.

    Returns:
        sps.dia_matrix: Diagonal matrix which switches the sign of faces if the
            normal vector of the face points into the grid g. Faces not considered
            will have a 0 diagonal term. If nd > 1, the first nd rows are associated
            with the first face, then nd elements of the second face etc.

    """

    faces = np.asarray(faces)

    # Find out whether the boundary faces have outwards pointing normal vectors
    # Negative sign implies that the normal vector points inwards.
    sgn, _ = g.signs_and_cells_of_boundary_faces(faces)

    # Create vector with the sign in the places of faces under consideration,
    # zeros otherwise
    sgn_mat = np.zeros(g.num_faces)
    sgn_mat[faces] = sgn
    # Duplicate the numbers, the operator is intended for vector quantities
    sgn_mat = np.tile(sgn_mat, (nd, 1)).ravel(order="F")

    # Create the diagonal matrix.
    return sps.dia_matrix((sgn_mat, 0), shape=(sgn_mat.size, sgn_mat.size))


def star_shape_cell_centers(g: "pp.Grid", as_nan: bool = False) -> np.ndarray:
    """
    For a given grid compute the star shape center for each cell.
    The algorithm computes the half space intersections of the spaces defined
    by the cell faces and the face normals by using the method half_space_interior_point.
    half_space_pt,
    of the spaces defined by the cell faces and the face normals.
    This is a wrapper method that operates on a grid.

    Parameters
    ----------
    g: pp.Grid
        the grid
    as_nan: bool, optional
        Decide whether to return nan as the new center for cells which are not
         star-shaped. Otherwise, an exception is raised (default behaviour).

    Returns
    -------
    np.ndarray
        The new cell centers.

    """

    # no need for 1d or 0d grids
    if g.dim < 2:
        return g.cell_centers

    # retrieve the faces and nodes
    faces, _, sgn = sparse_array_to_row_col_data(g.cell_faces)
    nodes, _, _ = sparse_array_to_row_col_data(g.face_nodes)

    # Shift the nodes close to the origin to avoid numerical problems when coordinates are
    # too big
    xn = g.nodes.copy()
    xn_shift = np.average(xn, axis=1)
    xn -= np.tile(xn_shift, (xn.shape[1], 1)).T

    # compute the star shape cell centers by constructing the half spaces of each cell
    # given by its faces and related normals
    cell_centers = np.zeros((3, g.num_cells))
    for c in np.arange(g.num_cells):
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
        faces_loc = faces[loc]
        loc_n = g.face_nodes.indptr[faces_loc]
        # make the normals coherent
        normal = np.multiply(
            sgn[loc], np.divide(g.face_normals[:, faces_loc], g.face_areas[faces_loc])
        )

        x0, x1 = xn[:, nodes[loc_n]], xn[:, nodes[loc_n + 1]]
        coords = np.concatenate((x0, x1), axis=1)
        # compute a point in the half space intersection of all cell faces
        try:
            cell_centers[:, c] = pp.half_space.half_space_interior_point(
                normal, (x1 + x0) / 2.0, coords
            )
        except ValueError:
            # the cell is not star-shaped
            if as_nan:
                cell_centers[:, c] = np.array([np.nan, np.nan, np.nan])
            else:
                raise ValueError(
                    "Cell not star-shaped impossible to compute the center."
                )

    # shift back the computed cell centers and return them
    return cell_centers + np.tile(xn_shift, (g.num_cells, 1)).T
