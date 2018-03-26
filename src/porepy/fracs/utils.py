""" Frontend utility functions related to fractures and their meshing.

"""
import numpy as np

import porepy as pp


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
