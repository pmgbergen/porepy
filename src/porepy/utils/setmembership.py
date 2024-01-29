"""
Various functions with set operations.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.spatial import KDTree


def unique_rows(
    data: np.ndarray[Any, np.dtype[np.float64]]
) -> Tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.int64]],
]:
    """
    Function similar to Matlab's unique(...,'rows')

    See also function unique_columns in this module; this is likely slower, but
    is understandable, documented, and has a tolerance option.

    Copied from
    http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array/
    (summary pretty far down on the page)
    Note: I have no idea what happens here

    """
    b = np.ascontiguousarray(data).view(
        np.dtype((np.void, data.dtype.itemsize * data.shape[1]))
    )
    _, ia = np.unique(b, return_index=True)
    _, ic = np.unique(b, return_inverse=True)
    #    return np.unique(b).view(data.dtype).reshape(-1, data.shape[1]), ia, ic
    return data[ia], ia, ic


def ismember_rows(
    a: np.ndarray[Any, np.dtype[np.int64]],
    b: np.ndarray[Any, np.dtype[np.int64]],
    sort: float = True,
) -> Tuple[np.ndarray[Any, np.dtype[np.bool_]], np.ndarray[Any, np.dtype[np.int64]]]:
    """
    Find *columns* of a that are also members of *columns* of b.

    The function mimics Matlab's function ismember(..., 'rows').

    TODO: Rename function, this is confusing!

    Parameters:
        a (np.array): Each column in a will search for an equal in b.
        b (np.array): Array in which we will look for a twin
        sort (boolean, optional): If true, the arrays will be sorted before
            seraching, increasing the chances for a match. Defaults to True.

    Returns:
        np.array (boolean): For each column in a, true if there is a
            corresponding column in b.
        np.array (int): Indexes so that b[:, ind] is also found in a.

    Examples:
        >>> a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
        >>> b = np.array([[3, 1, 3, 5, 3], [3, 3, 2, 1, 2]])
        >>> ismember_rows(a, b)
        (array([ True,  True,  True,  True, False], dtype=bool), [1, 0, 2, 1])

        >>> a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
        >>> b = np.array([[3, 1, 2, 5, 1], [3, 3, 3, 1, 2]])
        >>> ismember_rows(a, b, sort=False)
        (array([ True,  True, False,  True, False], dtype=bool), [1, 0, 1])

    """
    # IMPLEMENTATION NOTE: A serious attempt was made (June 2022) to speed up
    # this calculation by using functionality from scipy.spatial. This did not
    # work: The Scipy functions scaled poorly with the size of arrays a and b,
    # and lead to memory overflow for large arrays.

    # Sort if required, but not if the input is 1d
    if sort and a.ndim > 1:
        sa = np.sort(a, axis=0)
        sb = np.sort(b, axis=0)
    else:
        sa = a
        sb = b

    b = np.atleast_1d(b)
    a = np.atleast_1d(a)
    num_a = a.shape[-1]

    # stack the arrays
    c = np.hstack((sa, sb))
    # Uniquify c. We don't care about the unique array itself, but rather
    # the indices which maps the unique array back to the original c
    if a.ndim > 1:
        _, ind = np.unique(c, axis=1, return_inverse=True)
    else:
        _, ind = np.unique(c, return_inverse=True)

    # Indices in a and b referring to the unique array.
    # Elements in ind_a that are also in ind_b will correspond to rows
    # in a that are also found in b
    ind_a = ind[:num_a]
    ind_b = ind[num_a:]

    # Find common members
    ismem_a = np.isin(ind_a, ind_b)

    # Found this trick on
    # https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
    # See answer by Joe Kington
    sort_ind = np.argsort(ind_b)
    ypos = np.searchsorted(ind_b[sort_ind], ind_a[ismem_a])
    ia = sort_ind[ypos]

    # Done
    return ismem_a, ia


def unique_columns_tol(
    mat: np.ndarray,
    tol: float = 1e-8,
) -> Tuple[
    np.ndarray,
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.int64]],
]:
    """
    For an array, remove columns that are closer than a given tolerance.

    To uniquify a point set, consider using the function uniquify_point_set
    instead.

    Resembles Matlab's uniquetol function, as applied to columns. To rather
    work on rows, use a transpose.

    Parameters:
        mat (np.ndarray, nd x n_pts): Columns to be uniquified.
        tol (double, optional): Tolerance for when columns are considered equal.
            Should be seen in connection with distance between the points in
            the points (due to rounding errors). Defaults to 1e-8.

    Returns:
        np.ndarray: Unique columns.
        new_2_old: Index of which points that are preserved
        old_2_new: Index of the representation of old points in the reduced
            list.

    Example:
        >>> p_un, n2o, o2n = unique_columns(np.array([[1, 0, 1], [1, 0, 1]]))
        >>> p_un
        array([[1, 0], [1, 0]])
        >>> n2o
        array([0, 1])
        >>> o2n
        array([0, 1, 0])

    """
    import numba

    # Treat 1d array as 2d
    mat = np.atleast_2d(mat)

    # Some special cases
    if mat.shape[1] == 0:
        # Empty arrays gets empty return
        return mat, np.array([], dtype=int), np.array([], dtype=int)
    elif mat.shape[1] == 1:
        # Array with a single column needs no processing
        return mat, np.array([0], dtype=int), np.array([0], dtype=int)

    # If the matrix is integers, and the tolerance less than 1/2, we can use
    # numpy's unique function
    if issubclass(mat.dtype.type, np.int_) and tol < 0.5:
        un_ar, new_2_old, old_2_new = np.unique(
            mat, return_index=True, return_inverse=True, axis=1
        )
        return un_ar, new_2_old, old_2_new

    @numba.jit("Tuple((b1[:],i8[:],i8[:]))(f8[:, :],f8)", nopython=True, cache=True)
    def _numba_distance(mat, tol):
        """Helper function for numba acceleration of unique_columns_tol.

        IMPLEMENTATION NOTE: Calling this function many times (it is unclear
        what this really means, but likely >=100s of thousands of times) may
        lead to enhanced memory consumption and significant reductions in
        performance. This could be related to this GH issue

            https://github.com/numba/numba/issues/1361

        However, it is not clear this is really the error. No solution is known
        at the time of writing, the only viable options seem to be algorithmic
        modifications that reduce the number of calls to this function.

        """
        num_cols = mat.shape[0]
        keep = np.zeros(num_cols, dtype=numba.types.bool_)
        keep[0] = True
        keep_counter = 1

        # Map from old points to the unique subspace. Defaults to map to itself.
        old_2_new = np.arange(num_cols)

        # Loop over all points, check if it is already represented in the kept list
        for i in range(1, num_cols):
            d = np.sum((mat[i] - mat[keep]) ** 2, axis=1)
            condition = d < tol**2

            if np.any(condition):
                # We will not keep this point
                old_2_new[i] = np.argmin(d)
            else:
                # We have found a new point
                keep[i] = True
                old_2_new[i] = keep_counter
                keep_counter += 1

        # Finally find which elements we kept
        new_2_old = np.nonzero(keep)[0]

        return keep, new_2_old, old_2_new

    mat_t = np.atleast_2d(mat.T).astype(float)

    # IMPLEMENTATION NOTE: It could pay off to make a pure Python implementation
    # to be used for small arrays, however, attempts on making this work in
    # practice failed.

    keep, new_2_old, old_2_new = _numba_distance(mat_t, tol)

    return mat[:, keep], new_2_old, old_2_new


def uniquify_point_set(
    points: np.ndarray[Any, np.dtype[np.float64]], tol: float = 1e-8
) -> Tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.int64]],
]:
    """Uniquify a set of points so that no two sets of points are closer than a
    distance tol from each other.

    This function partially overlaps the function unique_columns_tol,
    but the latter is more general, as it provides fast treatment of integer
    arrays.

    FIXME: It should be possible to unify the two implementations, however,
    more experience is needed before doing so.

    Parameters:
        points (np.ndarray, nd x n_pts): Columns to be uniquified.
        tol (double, optional): Tolerance for when columns are considered equal.
            Should be seen in connection with distance between the points in
            the point set (due to rounding errors). Defaults to 1e-8.

    Returns:
        np.ndarray: Unique columns.
        new_2_old: Index of which points that are preserved
        old_2_new: Index of the representation of old points in the reduced
            list.

    """
    # The implementation uses Scipy's KDTree implementation to efficiently get
    # the distance between points.
    num_p = points.shape[1]
    # Transpose needed to comply with KDTree.
    tree = KDTree(points.T)

    # Get all pairs of points closer than the tolerance.
    pairs = tree.query_pairs(tol, output_type="ndarray")

    if pairs.size == 0:
        # No points were found, we can return
        return points, np.arange(num_p), np.arange(num_p)

    # Process information to arrive at a unique point set. This is technical,
    # since we need to deal with cases where more than two points coincide.
    # As an example: if the points p1, p2 and p3 coincide, they will be
    # identified either by the pairs {(i1, i2), (i1, i3)}, by
    # {(i1, i2), (i2, i3)}, or by {(i1, i3), (i2, i3)}). To be clear,
    # more than three points can coincide - such configurations will
    # include more point combinations, but will not introduce additional
    # complications.

    # Sort the index pairs of identical points for simpler identification.
    # NOTE: pairs, as returned by KDTree, is a num_pairs x 2 array, thus
    # sorting the pairs should be along axis 1.
    pair_arr = np.sort(pairs, axis=1)
    # Sort the pairs along axis=1. The lexsort will make the sorting first
    # according to pair_arr[:, 0] (the point with the lowest index in each
    # pair), and then according to the second index (pair_arr[:, 1]). The
    # result will be a lexicographically ordered array.
    # Also note the transport back to a 2 x num_pairs array.
    sorted_arr = pair_arr[np.lexsort((pair_arr[:, 1], pair_arr[:, 0]))].T

    # Find points that are both in the first and second row. Referring to the
    # example with three intersecting points, this will identify triplets
    # expressed as pairs {(i1, i2), (i2, i3)}.
    duplicate = np.isin(sorted_arr[0], sorted_arr[1])
    # Array with duplicates of the type {(i1, i2), (i1, i3)} removed.
    reduced_arr = sorted_arr[:, np.logical_not(duplicate)]

    # Also identify points that are not involved in any pairs, these should be
    # included in the unique set. Append these to the point array.
    not_in_pairs = np.setdiff1d(np.arange(points.shape[1]), pair_arr.ravel())
    reduced_arr = np.hstack((reduced_arr, np.tile(not_in_pairs, (2, 1))))

    # The array can still contain pairs of type {(i1, i2), (i1, i3)} and
    # {(i1, i3), (i1, i3)}, again referring to the example with three identical
    # points.
    # These can be identified by a unique on the first row.
    ia = np.unique(reduced_arr[0])

    # Make a mapping from all points to the reduced set.
    ib = np.arange(num_p)
    _, inv_map = np.unique(reduced_arr[0], return_inverse=True)
    ib[reduced_arr[0]] = inv_map
    ib[reduced_arr[1]] = ib[reduced_arr[0]]

    # Uniquify points.
    upoints = points[:, ia]

    # Done.
    return upoints, ia, ib
