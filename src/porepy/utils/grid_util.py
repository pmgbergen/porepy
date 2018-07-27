import numpy as np
import scipy.sparse as sps


def sign_of_boundary_faces(g):
    """
    returns the sign of boundary faces as defined by g.cell_faces. 
    Parameters:
    g: (Grid Object)
    faces: (ndarray) indices of faces that you want to know the sign for. The 
           faces must be boundary faces.

    Returns:
    sgn: (ndarray) the sign of the faces
    """
    faces = g.get_all_boundary_faces()

    IA = np.argsort(faces)
    IC = np.argsort(IA)

    fi, _, sgn = sps.find(g.cell_faces[faces[IA], :])
    assert fi.size == faces.size, "sign of internal faces does not make sense"
    I = np.argsort(fi)
    sgn = sgn[I]
    sgn = sgn[IC]
    return sgn
