"""
Various functions with set operations.
"""
import numpy as np

import porepy as pp

module_sections = ["utils"]


@pp.time_logger(sections=module_sections)
def unique_rows(data):
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


@pp.time_logger(sections=module_sections)
def ismember_rows(a, b, sort=True):
    """
    Find *columns* of a that are also members of *columns* of b.

    The function mimics Matlab's function ismember(..., 'rows').

    If the numpy version is less than 1.13, this function will be slow for
    large arrays.

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

    # Fonud this trick on
    # https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
    # See answer by Joe Kington
    sort_ind = np.argsort(ind_b)
    ypos = np.searchsorted(ind_b[sort_ind], ind_a[ismem_a])
    ia = sort_ind[ypos]

    # Done
    return ismem_a, ia


@pp.time_logger(sections=module_sections)
def unique_columns_tol(mat, tol=1e-8):
    """
    Remove duplicates from a point set, for a given distance traveling.

    Resembles Matlab's uniquetol function, as applied to columns. To rather
    work at rows, use a transpose.


    Parameters:
        mat (np.ndarray, nd x n_pts): Columns to be uniquified
        tol (double, optional): Tolerance for when columns are considered equal.
            Should be seen in connection with distance between the points in
            the points (due to rounding errors). Defaults to 1e-8.

    Returns:
        np.ndarray: Unique columns.
        new_2_old: Index of which points that are preserved
        old_2_new: Index of the representation of old points in the reduced
            list.

    Example (won't work as doctest):
        >>> p_un, n2o, o2n = unique_columns(np.array([[1, 0, 1], [1, 0, 1]]))
        >>> p_un
        array([[1, 0], [1, 0]])
        >>> n2o
        array([0, 1])
        >>> o2n
        array([0, 1, 0])

    """
    # Treat 1d array as 2d
    mat = np.atleast_2d(mat)

    # Special treatment of the case with an empty array
    if mat.shape[1] == 0:
        return mat, np.array([], dtype=int), np.array([], dtype=int)

    # If the matrix is integers, and the tolerance less than 1/2, we can use
    # numpy's unique function
    if issubclass(mat.dtype.type, np.int_) and tol < 0.5:
        un_ar, new_2_old, old_2_new = np.unique(
            mat, return_index=True, return_inverse=True, axis=1
        )
        return un_ar, new_2_old, old_2_new

    num_cols = mat.shape[1]

    # By default, no columns are kept
    keep = np.zeros(num_cols, dtype=bool)

    # We will however keep the first point
    keep[0] = True
    keep_counter = 1

    # Map from old points to the unique subspace. Defaults to map to itself.
    old_2_new = np.arange(num_cols)

    # Matrix of elements to keep. Comparison of new poitns will be run with this
    keep_mat = mat[:, 0].reshape((-1, 1))

    # Loop over all points, check if it is already represented in the kept list
    for i in range(1, num_cols):
        a = mat[:, i].reshape((-1, 1))

        d = np.sum((a - keep_mat) ** 2, axis=0)
        condition = d < tol ** 2
        proximate = np.where(condition)[0]

        if condition[proximate]:  # proximate.size > 0:
            # We will not keep this point
            old_2_new[i] = proximate
        else:
            # We have found a new point
            keep[i] = True
            old_2_new[i] = keep_counter
            keep_counter += 1
            # Update list of elements we keep
            keep_mat = mat[:, keep]

    # Finally find which elements we kept
    new_2_old = np.nonzero(keep)[0]

    return mat[:, keep], new_2_old, old_2_new
