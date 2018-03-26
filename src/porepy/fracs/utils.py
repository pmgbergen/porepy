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
