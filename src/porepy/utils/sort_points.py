""" Functions to sort points and edges belonging to geometric objects.
"""
from typing import Optional, Tuple, Union

import numpy as np

import porepy as pp

module_sections = ["utils"]


@pp.time_logger(sections=module_sections)
def sort_point_pairs(
    lines: np.ndarray,
    check_circular: Optional[bool] = True,
    ordering: Optional[bool] = False,
    is_circular: Optional[bool] = True,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sort pairs of numbers to form a chain.

    The target application is to sort lines, defined by their
    start end endpoints, so that they form a continuous polyline.

    The algorithm is brute-force, using a double for-loop. This can
    surely be imporved.

    Parameters:
    lines: np.ndarray, 2xn, the line pairs. If lines has more than 2 rows, we assume
        that the points are stored in the first two rows.
    check_circular: Verify that the sorted polyline form a circle.
                    Defaluts to true.
    ordering: np.array, return in the original order if a line is flipped or not
    is_circular: if the lines form a closed set. Default is True.

    Returns:
    sorted_lines: np.ndarray, 2xn, sorted line pairs. If lines had more than 2 rows,
        the extra are sorted accordingly.
    sort_ind: np.ndarray, n: Sorted column indices, so that
        sorted_lines = lines[:, sort_ind], modulu flipping of rows in individual columns
    is_ordered: np.ndarray (optional): True if the ordering of a segment (first and second
        row in input lines) is kept in the sorted lines. Refers to the original ordering
        of the lines (so lines, not sorted_lines).

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
        # occurence in line
        values = lines.ravel()
        count = np.bincount(values)
        one_occurence = np.where(count == 1)[0]
        hit = np.where(
            np.logical_or(
                np.isin(lines[0], one_occurence), np.isin(lines[1], one_occurence)
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

    # Order of the origin line list, store if they are flipped or not to form the chain
    is_ordered = np.zeros(num_lines, dtype=bool)
    is_ordered[0] = True

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
                is_ordered[j] = True
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
    if ordering:
        return sorted_lines, sort_ind, is_ordered
    return sorted_lines, sort_ind


@pp.time_logger(sections=module_sections)
def sort_point_plane(
    pts: np.ndarray,
    centre: np.ndarray,
    normal: Optional[np.ndarray] = None,
    tol: float = 1e-5,
) -> np.ndarray:
    """Sort the points which lie on a plane.

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


@pp.time_logger(sections=module_sections)
def sort_triangle_edges(t: np.ndarray) -> np.ndarray:
    """Sort a set of triangles so that no edges occur twice with the same ordering.

    For a planar triangulation, this will end up with all the triangles being
    ordered CW or CCW. In cases where the triangulated surface(s) do not share
    a common plane, methods based on geometry are at best cumbersome. This
    approach should work also in those cases.

    Parameters:
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
    @pp.time_logger(sections=module_sections)
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

        # Find the other occurence of this edge
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

        #   pdb.set_trace()
        # Check if the edge occured at all among the non-processed triangles
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
            # Existi_newng pair in rows 0 and 1
            pair_0 = (t[1, ti_new], t[2, ti_new])
            pair_1 = (t[2, ti_new], t[0, ti_new])
        elif hit_new_0 + hit_new_1 == 2:
            # Existi_newng pair in rows 0 and 2
            pair_0 = (t[1, ti_new], t[2, ti_new])
            pair_1 = (t[0, ti_new], t[1, ti_new])
        else:  # sum is 3
            # Existi_newng pair in rows 1 and 2
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
