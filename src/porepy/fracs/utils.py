""" Frontend utility functions related to fractures and their meshing.

"""
import numpy as np
import logging

import porepy as pp


# Module level logger
logger = logging.getLogger(__name__)


def fracture_length_2d(pts, edges):
    """ Find the length of 2D fracture traces.

    Parameters:
        pts (np.ndarray, 2 x n_pts): Coordinates of start and endpoints of
            fractures.
        edges (np.ndarary, 2 x n_fracs): Indices of start and endpoint of
            fractures, referring to columns in pts.

    Returns:
        np.ndarray, length n_fracs: Length of each fracture.

    """
    start = pts[:, edges[0]]
    end = pts[:, edges[1]]

    length = np.sqrt(np.sum(np.power(end - start, 2), axis=0))
    return length


def uniquify_points(pts, edges, tol):
    """ Uniquify a set of points by merging almost coinciding coordinates.

    Also update fractures, and remove edges that consist of a single point
    (either after the points were merged, or because the input was a point
    edge).

    Parameters:
        pts (np.ndarary, n_dim x n_pts): Coordinates of start and endpoints of
            the fractures.
        edges (np.ndarray, n x n_fracs): Indices of start and endpoint of
            fractures, referring to columns in pts. Should contain at least two
            rows; additional rows representing fracture tags are also accepted.
        tol (double): Tolerance used for merging points.

    Returns:
        np.ndarray (n_dim x n_pts_unique): Unique point array.
        np.ndarray (2 x n_fracs_update): Updated start and endpoints of
            fractures.
        np.ndarray: Index (referring to input) of fractures deleted as they
            effectively contained a single coordinate.

    """

    # uniquify points based on coordinates
    p_unique, _, o2n = pp.utils.setmembership.unique_columns_tol(pts, tol=tol)
    # update edges
    e_unique_p = np.vstack((o2n[edges[:2]], edges[2:]))

    # Find edges that start and end in the same point, and delete them
    point_edge = np.where(np.diff(e_unique_p[:2], axis=0)[0] == 0)[0].ravel()
    e_unique = np.delete(e_unique_p, point_edge, axis=1)

    return p_unique, e_unique, point_edge


def snap_fracture_set_2d(pts, edges, snap_tol, termination_tol=1e-2, max_iter=100):
    """ Snap vertexes of a set of fracture lines embedded in 2D, so that small
    distances between lines and vertexes are removed.

    This is intended as a utility function to preprocess a fracture network
    before meshing. The function may change both connectivity and orientation
    of individual fractures in the network. Specifically, fractures that
    almost form a T-intersection (or L), may be connected, while
    X-intersections with very short ends may be truncated to T-intersections.

    The modification snaps vertexes to the closest point on the adjacent line.
    This will in general change the orientation of the fracture with the
    snapped vertex. The alternative, to prolong the fracture along its existing
    orientation, may result in very long fractures for almost intersecting
    lines. Depending on how the fractures are ordered, the same point may
    need to be snapped to a segment several times in an iterative process.

    The algorithm is *not* deterministic, in the sense that if the ordering of
    the fractures is permuted, the snapped fracture network will be slightly
    different.

    Parameters:
        pts (np.array, 2 x n_pts): Array of start and endpoints for fractures.
        edges (np.ndarray, n x n_fracs): First row contains index of start
            of all fractures, referring to columns in pts. Second contains
            index of endpoints.
        snap_tol (double): Snapping tolerance. Distances below this will be
            snapped.
        termination_tol (double): Minimum point movement needed for the
            iterations to continue.
        max_iter (int, optional): Maximum number of iterations. Defaults to
            100.

    Returns:
        np.array (2 x n_pts): Copy of the point array, with modified point
            coordinates.
        boolean: True if the iterations converged within allowed number of
            iterations.

    """
    pts_orig = pts.copy()
    counter = 0
    pn = 0 * pts
    while counter < max_iter:
        pn = pp.cg.snap_points_to_segments(pts, edges, tol=snap_tol)
        diff = np.max(np.abs(pn - pts))
        logger.debug("Iteration " + str(counter) + ", max difference" + str(diff))
        pts = pn
        if diff < termination_tol:
            break
        counter += 1

    if counter < max_iter:
        logger.info("Fracture snapping converged after " + str(counter) + " iterations")
        logger.info("Maximum modification " + str(np.max(np.abs(pts - pts_orig))))
        return pts, True
    else:
        logger.warning("Fracture snapping failed to converge")
        logger.warning("Residual: " + str(diff))
        return pts, False
