""" Functions to sort points and edges belonging to geometric objects.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import porepy as pp


def sort_point_pairs(
    lines: np.ndarray,
    check_circular: bool = True,
    is_circular: Optional[bool] = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort pairs of numbers to form a chain.

    The target application is to sort lines, defined by their
    start end endpoints, so that they form a continuous polyline.

    The algorithm is brute-force, using a double for-loop. This can
    surely be improved.

    Args:
        lines (np.ndarray, 2xn): the line pairs. If lines has more than 2 rows, we assume
            that the points are stored in the first two rows.
        check_circular (bool): Verify that the sorted polyline form a circle.
        is_circular (bool): if the lines form a closed set. Default is True.

    Returns:
        np.ndarray, 2xn: sorted line pairs. If lines had more than 2 rows,
            the extra are sorted accordingly.
        np.ndarray, n: Sorted column indices, so that
            sorted_lines = lines[:, sort_ind], modulo flipping of rows in individual
            columns

    """

    num_lines = lines.shape[1]
    sorted_lines = -np.ones(lines.shape, dtype=lines.dtype)

    # Keep track of which lines have been found, which are still candidates
    found = np.zeros(num_lines, dtype=bool)

    # Initialize array of sorting indices
    sort_ind = np.zeros(num_lines, dtype=int)

    # In the case of non-circular ordering ensure to start from the correct one
    if not is_circular:
        # The first segment must contain one of the endpoints, identified by a single
        # occurrence in line
        values = lines.ravel()
        count = np.bincount(values)
        one_occurrence = np.where(count == 1)[0]
        hit = np.where(
            np.logical_or(
                np.isin(lines[0], one_occurrence), np.isin(lines[1], one_occurrence)
            )
        )[0][0]
        sorted_lines[:, 0] = lines[:, hit]
        # The end of the first segment must also occur somewhere else.
        if np.count_nonzero(lines == sorted_lines[0, 0]) > 1:
            sorted_lines[:2, 0] = np.flip(sorted_lines[:2, 0], 0)
        found[hit] = True
        sort_ind[0] = hit

        # No check for circularity here
        check_circular = False
    else:
        # Start with the first line in input
        sorted_lines[:, 0] = lines[:, 0]
        found[0] = True
    # The starting point for the next line
    prev = sorted_lines[1, 0]

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
                prev = lines[1, j]
                sort_ind[i] = j

                break
            elif not found[j] and lines[1, j] == prev:
                sorted_lines[:, i] = lines[:, j]
                sorted_lines[:2, i] = np.flip(sorted_lines[:2, i], 0)
                found[j] = True
                prev = lines[0, j]
                sort_ind[i] = j
                break
    # By now, we should have used all lines
    assert np.all(found)
    if check_circular:
        assert sorted_lines[0, 0] == sorted_lines[1, -1]
    return sorted_lines, sort_ind


def sort_multiple_point_pairs(lines: np.ndarray) -> np.ndarray:
    """Function to sort multiple pairs of points to form circular chains.

    The routine contains essentially the same functionality as sort_point_pairs,
    but stripped down to the special case of circular chains. Differently to
    sort_point_pairs, this variant sorts an arbitrary amount of independent
    point pairs. The chains are restricted by the assumption that each contains
    equally many line segments. Finally, this routine uses numba.

    Parameters:
        lines (np.ndarray): Array of size 2 * num_chains x num_lines_per_chain,
            containing node indices. For each pair of two rows, each column
            represents a line segment connectng the two nodes in the two entries
            of this column.

    Returns:
        np.ndarray: Sorted version of lines, where for each chain, the collection
            of l

    Raises:
        ImportError ifine segments has been potentially flipped and sorted.
    """

    try:
        import numba
    except ImportError:
        raise ImportError("Numba not available on the system")

    @numba.njit("i4[:,:](i4[:,:])", cache=True)
    def _function_to_compile(lines):
        """
        Copy of pp.utils.sort_points.sort_point_pairs. This version is extended
        to multiple chains. Each chain is implicitly assumed to be circular.
        """

        # Retrieve number of chains and lines per chain from the shape.
        # Implicitly expect that all chains have the same length
        num_chains, chain_length = lines.shape
        # Since for each chain lines includes two rows, divide by two
        num_chains = int(num_chains / 2)

        # Initialize array of sorted lines to be the final output
        sorted_lines = np.zeros((2 * num_chains, chain_length), dtype=np.int32)
        # Fix the first line segment for each chain and identify
        # it as in place regarding the sorting.
        sorted_lines[:, 0] = lines[:, 0]
        # Keep track of which lines have been fixed and which are still candidates
        found = np.zeros(chain_length, dtype=np.int32)
        found[0] = 1

        # Loop over chains and consider each chain separately.
        for c in range(num_chains):
            # Initialize found making any line segment aside of the first a candidate
            found[1:] = 0

            # Define the end point of the previous and starting point for the next
            # line segment
            prev = sorted_lines[2 * c + 1, 0]

            # The sorting algorithm: Loop over all positions in the chain to be set next.
            # Find the right candidate to be moved to this position and possibly flipped
            # if needed. A candidate is identified as fitting if it contains one point
            # equal to the current starting point. This algorithm uses a double loop,
            # which is the most naive approach. However, assume chain_length is in
            # general small.
            for i in range(1, chain_length):  # The first line has already been found
                for j in range(
                    1, chain_length
                ):  # The first line has already been found
                    # A candidate line segment with matching start and end point
                    # in the first component of the point pair.
                    if np.abs(found[j]) < 1e-6 and lines[2 * c, j] == prev:
                        # Copy the segment to the right place
                        sorted_lines[2 * c : 2 * c + 2, i] = lines[2 * c : 2 * c + 2, j]
                        # Mark as used
                        found[j] = 1
                        # Define the starting point for the next line segment
                        prev = lines[2 * c + 1, j]
                        break
                    # A candidate line segment with matching start and end point
                    # in the second component of the point pair.
                    elif np.abs(found[j]) < 1e-6 and lines[2 * c + 1, j] == prev:
                        # Flip and copy the segment to the right place
                        sorted_lines[2 * c, i] = lines[2 * c + 1, j]
                        sorted_lines[2 * c + 1, i] = lines[2 * c, j]
                        # Mark as used
                        found[j] = 1
                        # Define the starting point for the next line segment
                        prev = lines[2 * c, j]
                        break

        # Return the sorted lines defining chains.
        return sorted_lines

    # Run numba compiled function
    return _function_to_compile(lines)


def sort_point_plane(
    pts: np.ndarray,
    centre: np.ndarray,
    normal: Optional[np.ndarray] = None,
    tol: float = 1e-5,
) -> np.ndarray:
    """Sort the points which lie on a plane.

    The algorithm assumes a star-shaped disposition of the points with respect
    the centre.

    Args:
        pts: np.ndarray, 3xn, the points.
        centre: np.ndarray, 3x1, the face centre.
        normal: (optional) the normal of the plane, otherwise three points are
            required.
        tol:
            Absolute tolerance used to identify active (non-constant) dimensions.

    Returns:
        map_pts: np.array, 1xn, sorted point ids.

    """
    centre = centre.reshape((-1, 1))
    R = pp.map_geometry.project_plane_matrix(pts, normal)
    # project points and center,  project to plane
    delta = np.dot(R, pts - centre)

    # Find active dimension in the projected system
    check = np.sum(np.abs(delta), axis=1)
    check /= np.sum(check)
    # Dimensions where not all coordinates are equal
    active_dim = np.logical_not(np.isclose(check, 0, atol=tol, rtol=0))

    return np.argsort(np.arctan2(*delta[active_dim]))


def sort_triangle_edges(t: np.ndarray) -> np.ndarray:
    """Sort a set of triangles so that no edges occur twice with the same ordering.

    For a planar triangulation, this will end up with all the triangles being
    ordered CW or CCW. In cases where the triangulated surface(s) do not share
    a common plane, methods based on geometry are at best cumbersome. This
    approach should work also in those cases.

    Args:
        t (np.ndarray, 3 x n_tri): Triangulation to have vertexes ordered.

    Returns:
        np.ndarray, 3 x n_tri: With the vertexes ordered.

    Example:
        >>> t = np.array([[0, 1, 2], [1, 2, 3]]).T
        >>> sort_triangle_edges(t)
        np.array([[0, 2], [1, 1], [2, 3]])

    """

    # Helper method to remove pairs from the queue if they already exist,
    # add them if not
    def update_queue(pair_0, pair_1):
        if pair_0 in queue:
            queue.remove(pair_0)
        elif (pair_0[1], pair_0[0]) in queue:
            queue.remove((pair_0[1], pair_0[0]))
        else:
            queue.append(pair_0)
        if pair_1 in queue:
            queue.remove(pair_1)
        elif (pair_1[1], pair_1[0]) in queue:
            queue.remove((pair_1[1], pair_1[0]))
        else:
            queue.append(pair_1)

    nt = t.shape[1]

    # Add all edges of the first triangle to the queue
    queue = [(t[0, 0], t[1, 0]), (t[1, 0], t[2, 0]), (t[2, 0], t[0, 0])]

    # For safeguarding, we count the number of iterations. Maximum number
    # is if all triangles are isolated.
    max_iter = nt * 3
    num_iter = 0

    # Bookkeeping of already processed triangles. Not sure if this is needed.
    is_ordered = np.zeros(nt, dtype=bool)
    is_ordered[0] = 1

    while len(queue) > 0:
        # Pick an edge to be processed
        q = queue.pop(0)

        # Find the other occurrence of this edge
        hit_new = np.logical_and.reduce(
            (
                np.logical_not(is_ordered),
                np.any(t == q[0], axis=0),
                np.any(t == q[1], axis=0),
            )
        )
        hit_old = np.logical_and.reduce(
            (is_ordered, np.any(t == q[0], axis=0), np.any(t == q[1], axis=0))
        )
        ind_old = np.where(hit_old > 0)[0]
        ind_new = np.where(hit_new > 0)[0]

        # Check if the edge occurred at all among the non-processed triangles
        if ind_new.size == 0:
            continue
        # It should at most occur once among non-processed triangles
        elif ind_new.size > 1:
            raise ValueError("Edges should only occur twice")

        # Find the triangle to be processed
        ti_new = ind_new[0]
        ti_old = ind_old[0]
        # Find row index of the first and second item of the pair q
        hit_new_0 = np.where(t[:, ti_new] == q[0])[0][0]
        hit_new_1 = np.where(t[:, ti_new] == q[1])[0][0]
        # Find row index of the first and second item of the pair q
        hit_old_0 = np.where(t[:, ti_old] == q[0])[0][0]
        hit_old_1 = np.where(t[:, ti_old] == q[1])[0][0]

        if hit_old_0 < hit_old_1 or (hit_old_0 == 2 and hit_old_1 == 0):
            # q0 comes before q1 in the already sorted column.
            if hit_new_1 - hit_new_0 == 1 or (hit_new_0 == 2 and hit_new_1 == 0):
                t[hit_new_0, ti_new] = q[1]
                t[hit_new_1, ti_new] = q[0]
            else:
                t[hit_new_0, ti_new] = q[0]
                t[hit_new_1, ti_new] = q[1]
        else:
            # q1 before q0 in the sorted column, reverse in the other
            if hit_new_1 - hit_new_0 or (hit_new_0 == 2 and hit_new_1 == 0):
                t[hit_new_0, ti_new] = q[0]
                t[hit_new_1, ti_new] = q[1]
            else:
                t[hit_new_0, ti_new] = q[1]
                t[hit_new_1, ti_new] = q[0]

        # Find the new pairs to be generated. This must be done in terms of
        # the content in t[:, ti], not the indices represented by hit_0 and _1.
        # The two pairs are formed by row hit_0 and hit_1, both combined with the
        # third element. First, the latter must be identified
        if hit_new_0 + hit_new_1 == 1:
            # Existing pair in rows 0 and 1
            pair_0 = (t[1, ti_new], t[2, ti_new])
            pair_1 = (t[2, ti_new], t[0, ti_new])
        elif hit_new_0 + hit_new_1 == 2:
            # Existing pair in rows 0 and 2
            pair_0 = (t[1, ti_new], t[2, ti_new])
            pair_1 = (t[0, ti_new], t[1, ti_new])
        else:  # sum is 3
            # Existing pair in rows 1 and 2
            pair_0 = (t[0, ti_new], t[1, ti_new])
            pair_1 = (t[2, ti_new], t[0, ti_new])
        # Update the queue, either remove the pairs or add them
        update_queue(pair_0, pair_1)

        # Bookkeeping
        is_ordered[ti_new] = 1

        # Safeguarding
        num_iter += 1
        if num_iter > max_iter:
            raise ValueError("Should not come here")
    return t
