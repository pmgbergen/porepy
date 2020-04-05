import numpy as np
import scipy.sparse as sps


def half_space_int(n, x0, pts):
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
    >>> n = np.array([[0, 1], [1, 0], [0, 0]])
    >>> x0 = np.array([[0, -1], [0, 0], [0, 0]])
    >>> pts = np.array([[-1 ,-1 ,4], [2, -2, -2], [0, 0, 0]])
    >>> half_space_int(n,x0,pts)
    array([False,  True, False], dtype=bool)
    """
    assert n.shape[0] == 3, " only 3D supported"
    assert x0.shape[0] == 3, " only 3D supported"
    assert pts.shape[0] == 3, " only 3D supported"
    assert (
        n.shape[1] == x0.shape[1]
    ), "there must be the same number of normal vectors as points"

    n_pts = pts.shape[1]
    in_hull = np.zeros(n_pts)
    x0 = np.repeat(x0[:, :, np.newaxis], n_pts, axis=2)
    n = np.repeat(n[:, :, np.newaxis], n_pts, axis=2)
    for i in range(x0.shape[1]):
        in_hull += np.sum((pts - x0[:, i, :]) * n[:, i, :], axis=0) <= 0

    return in_hull == x0.shape[1]


# ------------------------------------------------------------------------------#


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
    import scipy.optimize as opt

    dim = (1, n.shape[1])
    c = np.array([0, 0, 0, 0, -1])
    A_ub = np.concatenate((n, [np.sum(-n * x0, axis=0)], np.ones(dim))).T
    b_ub = np.zeros(dim).T
    b_min, b_max = np.amin(pts, axis=1), np.amax(pts, axis=1)
    bounds = (
        (b_min[0], b_max[0]),
        (b_min[1], b_max[1]),
        (b_min[2], b_max[2]),
        (0, None),
        (0, None),
    )
    res = opt.linprog(c, A_ub, b_ub, bounds=bounds)

    if recompute and (not res.success or np.all(np.isclose(res.x[3:], 0))):
        return half_space_pt(-n, x0, pts, False)

    if res.success and not np.all(np.isclose(res.x[3:], 0)):
        return np.array(res.x[:3]) / res.x[3]
    else:
        raise ValueError("Half space intersection empty")


# ------------------------------------------------------------------------------#


def star_shape_cell_centers(g, as_nan=False):
    """
    For a given grid compute the star shape center for each cell.
    The algorithm computes the half space intersections, by using the above method half_space_pt,
    of the spaces defined by the cell faces and the face normals.
    This is a wrapper method that operate on a grid.

    Parameters
    ----------
    g: the grid
    as_nan: (default False) in the case some cells are not star-shaped return nan as new center.
        Otherwise an exception is raised (default behaviour).
    """

    # no need for 1d or 0d grids
    if g.dim < 2:
        return g.cell_centers

    # retrieve the faces and nodes
    faces, _, sgn = sps.find(g.cell_faces)
    nodes, _, _ = sps.find(g.face_nodes)

    # shift the nodes close to the origin, to avoid numerical problems when coordinates are
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
            cell_centers[:, c] = half_space_pt(normal, (x1 + x0) / 2.0, coords)
        except ValueError:
            # the cell is not star-shaped
            if as_nan:
                cell_centers[:, c] = np.array([np.nan, np.nan, np.nan])
            else:
                raise ValueError(
                    "Cell not star-shaped impossible to compute the centre"
                )

    # shift back the computed cell centers and return them
    return cell_centers + np.tile(xn_shift, (g.num_cells, 1)).T


# ------------------------------------------------------------------------------#
