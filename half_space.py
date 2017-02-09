import numpy as np
import scipy.optimize as opt


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

#------------------------------------------------------------------------------#

def half_space_pt(n, x0, pts, recompute=True):
    """
    Find an interior point for the halfspaces.

    Parameters
    ----------
    n : ndarray
        This is the normal vectors of the half planes. The normal
        vectors is assumed to coherently for all the half spaces
        (inward or outward).
    x0 : ndarray
        Point on the boundary of the half-spaces. Half space i is given
        by all points satisfying (x - x0[:,i])*n[:,i]<=0
    pts : ndarray
        Points which defines a bounds for the algorithm.
    recompute: bool
        If the algorithm fails try again with flipped normals.

    Returns
    -------
    out: array
        Interior point of the halfspaces.

    We use linear programming to find one interior point for the half spaces.
    Assume, n halfspaces defined by: aj*x1+bj*x2+cj*x3+dj<=0, j=1..n.
    Perform the following linear program:
    max(x5) aj*x1+bj*x2+cj*x3+dj*x4+x5<=0, j=1..n

    Then, if [x1,x2,x3,x4,x5] is an optimal solution with x4>0 and x5>0 we get:
    aj*(x1/x4)+bj*(x2/x4)+cj*(x3/x4)+dj<=(-x5/x4) j=1..n and (-x5/x4)<0,
    and conclude that the point [x1/x4,x2/x4,x3/x4] is in the interior of all
    the halfspaces. Since x5 is optimal, this point is "way in" the interior
    (good for precision errors).
    http://www.qhull.org/html/qhalf.htm#notes

    """
    dim = (1,n.shape[1])
    c = np.array([0,0,0,0,-1])
    A_ub = np.concatenate( (n, [np.sum(-n*x0, axis=0)], np.ones(dim)) ).T
    bounds = ( (np.amin(pts[0,:]), np.amax(pts[0,:]) ),
               (np.amin(pts[1,:]), np.amax(pts[1,:]) ),
               (np.amin(pts[2,:]), np.amax(pts[2,:]) ),
               (None, None), (None, None) )
    res = opt.linprog(c, A_ub, np.zeros(dim).T, bounds=bounds)
    if recompute and (res.status != 0 or res.x[3] <=0 or res.x[4] <= 0):
        return half_space_pt(-n, x0, pts, False)
    else:
        assert res.status == 0 and res.x[3] > 0 and res.x[4] > 0
        return np.array(res.x[0:3]) / res.x[3]

#------------------------------------------------------------------------------#
