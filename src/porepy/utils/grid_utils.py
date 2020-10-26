import numpy as np
import scipy.sparse as sps


def switch_sign_if_inwards_normal(g, nd, faces):
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
    sgn = g.sign_of_faces(faces)

    # Create vector with the sign in the places of faces under consideration,
    # zeros otherwise
    sgn_mat = np.zeros(g.num_faces)
    sgn_mat[faces] = sgn
    # Duplicate the numbers, the operator is intended for vector quantities
    sgn_mat = np.tile(sgn_mat, (nd, 1)).ravel(order="F")

    # Create the diagonal matrix.
    return sps.dia_matrix((sgn_mat, 0), shape=(sgn_mat.size, sgn_mat.size))
