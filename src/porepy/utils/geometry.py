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

    if normal is None:
    	normal = compute_normal( pts )
    else:
    	normal = normal / np.linalg.norm( normal )

    # Projection matrix onto tangential plane
    T = np.identity(3) - np.outer( normal, normal )
    # Projected points
    pts = np.array( [ np.dot( T, p ) for p in pts.T ] )
    # Disregard points on the origin??
    index = np.where( np.sum( np.abs( pts ), axis = 0 ) != 0 )[0]
    return pts[:,index]

#------------------------------------------------------------------------------#

def compute_normal( pts ):
    """ Compute the normal of a set of points.

    The algorithm computes the normal of the plane defined by the first three
    points in the set. 
    TODO: Introduce optional check that all points lay in the same plane
    (should be separate subroutine).

    Parameters:
    pts: np.ndarray, 3xn, the points. Need n > 2.

    Returns:
    normal: np.array, 1x3, the normal.

    """

    assert( pts.shape[1] > 2 )
    normal = np.cross( pts[:,0] - pts[:,1], pts[:,0] - pts[:,2] )
    return normal / np.linalg.norm( normal )

#------------------------------------------------------------------------------#
