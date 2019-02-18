""" This is a resurrection of a small part of the old fracture_deformation
model. It contains a single function that should really be placed somewhere els.


"""

import numpy as np
import scipy.sparse as sps


def sign_of_faces(g, faces):
    """
    returns the sign of faces as defined by g.cell_faces. 
    Parameters:
    g: (Grid Object)
    faces: (ndarray) indices of faces that you want to know the sign for. The 
           faces must be boundary faces.
    Returns:
    sgn: (ndarray) the sign of the faces
    """

    IA = np.argsort(faces)
    IC = np.argsort(IA)

    fi, _, sgn = sps.find(g.cell_faces[faces[IA], :])
    assert fi.size == faces.size, "sign of internal faces does not make sense"
    I = np.argsort(fi)
    sgn = sgn[I]
    sgn = sgn[IC]
    return sgn
