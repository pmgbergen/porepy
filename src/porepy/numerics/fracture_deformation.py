""" This is a resurrection of a small part of the old fracture_deformation
model. It contains a single function that should really be placed somewhere els.


"""

import numpy as np
import scipy.sparse as sps


def sign_of_faces(g, faces):
    """ Get the direction of the normal vector (inward or outwards from a cell)
    of faces. Only boundary faces are permissible.

    Parameters:
        g: (Grid Object)
        faces: (ndarray) indices of faces that you want to know the sign for. The
            faces must be boundary faces.

    Returns:
        (ndarray) the sign of the faces

    Raises:
        ValueError if a target face is internal.

    """

    IA = np.argsort(faces)
    IC = np.argsort(IA)

    fi, _, sgn = sps.find(g.cell_faces[faces[IA], :])
    if fi.size != faces.size:
        raise ValueError("sign of internal faces does not make sense")

    I = np.argsort(fi)
    sgn = sgn[I]
    sgn = sgn[IC]
    return sgn
