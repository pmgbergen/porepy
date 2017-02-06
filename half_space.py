import numpy as np


def half_space_int(n,x0, pts):
    """
    Find the points that lie in the intersection of half spaces (3D)

    Parameters
    ----------
    n : ndarray
        This is the normal vectors of the half planes. The normal 
        vectors is assumed to point out of the half spaces.
    x0 : ndarray
        Point on the boundary of the half-spaces. Half space i is given
        by all points satisfying (x - x0[:,i])*n[:,i]<=0
    pts : ndarray
        The points to be tested if they are in the intersection of all
        half-spaces or not.

    Returns
    -------
    out : ndarray
        A logical array with length equal number of pts.

        out[i] is True if pts[:,i] is in all half-spaces


    Examples
    --------
    >>> import numpy as np
    >>> n = np.array([[0,1],[1,0],[0,0]])
    >>> x0 = np.array([[0,-1],[0,0],[0,0]])
    >>> pts = np.array([[-1,-1,4],[2,-2,-2],[0,0,0]])
    >>> half_space_int(n,x0,pts)
    array([False,  True, False], dtype=bool)
    """
    assert n.shape[0] == 3, ' only 3D supported'
    assert x0.shape[0] == 3, ' only 3D supported'
    assert pts.shape[0] == 3, ' only 3D supported'
    assert n.shape[1] == x0.shape[1], 'ther must be the same number of normal vectors as points'
    n_pts   = pts.shape[1]
    in_hull = np.zeros(n_pts)
    x0      = np.repeat(x0[:,:,np.newaxis], n_pts,axis=2)
    n       = np.repeat( n[:,:,np.newaxis], n_pts,axis=2)
    for i in range(x0.shape[1]):
        in_hull += np.sum((pts - x0[:,i,:])*n[:,i,:],axis=0)<=0
        
    return in_hull==x0.shape[1]
