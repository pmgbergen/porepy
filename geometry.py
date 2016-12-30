import numpy as np

#------------------------------------------------------------------------------#

def project_plane( pts, normal = None ):
    """ Project the points on a plane using local coordinates.

    Parameters:
    pts: np.ndarray, 3xn, the points.
    normal: (optional) the normal of the plane, otherwise three points are
    required.

    Returns:
    pts: np.array, 2xn, projected points on the plane in the local coordinates.

    """

    if normal is None: normal = compute_normal( pts )
    else:              normal = normal / np.linalg.norm( normal )
    T = np.identity(3) - np.outer( normal, normal )
    pts = np.array( [ np.dot( T, p ) for p in pts.T ] )
    index = np.where( np.sum( np.abs( pts ), axis = 0 ) != 0 )[0]
    return pts[:,index]

#------------------------------------------------------------------------------#

def compute_normal( pts ):
    """ Compute the normal of a set of points.

    The algorithm assume that the points lie on a plane.
    Three points are required.

    Parameters:
    pts: np.ndarray, 3xn, the points.

    Returns:
    normal: np.array, 1x3, the normal.

    """

    assert( pts.shape[1] > 2 )
    normal = np.cross( pts[:,0] - pts[:,1], pts[:,0] - pts[:,2] )
    return normal / np.linalg.norm( normal )

#------------------------------------------------------------------------------#
