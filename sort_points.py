import numpy as np
from compgeom.basics import project_plane

#------------------------------------------------------------------------------#

def sort_point_pairs(lines, check_circular=True):
    """ Sort pairs of numbers to form a chain.

    The target application is to sort lines, defined by their
    start end endpoints, so that they form a continuous polyline.

    The algorithm is brute-force, using a double for-loop. This can
    surely be imporved.

    Parameters:
    lines: np.ndarray, 2xn, the line pairs.
    check_circular: Verify that the sorted polyline form a circle.
                    Defaluts to true.

    Returns:
    sorted_lines: np.ndarray, 2xn, sorted line pairs.

    """

    num_lines = lines.shape[1]
    sorted_lines = -np.ones( (2, num_lines), dtype = lines.dtype )

    # Start with the first line in input
    sorted_lines[:, 0] = lines[:, 0]

    # The starting point for the next line
    prev = sorted_lines[1, 0]

    # Keep track of which lines have been found, which are still candidates
    found = np.zeros(num_lines, dtype='bool')
    found[0] = True

    # The sorting algorithm: Loop over all places in sorted_line to be filled,
    # for each of these, loop over all members in lines, check if the line is still
    # a candidate, and if one of its points equals the current starting point.
    # More efficient and more elegant approaches can surely be found, but this
    # will do for now.
    for i in range(1, num_lines):  # The first line has already been found
        for j in range(0, num_lines):
            if not found[j] and lines[0, j] == prev:
                sorted_lines[:, i] = lines[:, j]
                found[j] = True
                prev = lines[1, i]
                break
            elif not found[j] and lines[1, j] == prev:
                sorted_lines[:, i] = lines[::-1, j]
                found[j] = True
                prev = lines[1, i]
                break
    # By now, we should have used all lines
    assert(np.all(found))
    if check_circular:
        assert sorted_lines[0, 0] == sorted_lines[1, -1]
    return sorted_lines

#------------------------------------------------------------------------------#

def sort_point_plane( pts, centre, normal = None ):
    """ Sort the points which lie on a plane.

    The algorithm assumes a star-shaped disposition of the points with respect
    the centre.

    Parameters:
    pts: np.ndarray, 3xn, the points.
    centre: np.ndarray, 3x1, the face centre.
    normal: (optional) the normal of the plane, otherwise three points are
    required.

    Returns:
    map_pts: np.array, 1xn, sorted point ids.

    """
    delta = project_plane( np.array( [ p - centre for p in pts.T ] ).T, normal )
    delta = np.divide( delta, np.linalg.norm( delta ) )
    return np.argsort( np.arctan2( *delta.T ) )

#------------------------------------------------------------------------------#
