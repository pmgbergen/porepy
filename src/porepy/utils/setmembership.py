# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:16:42 2016

@author: keile
"""
from __future__ import division
import numpy as np


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

    # If numpy >= 1.13 is available, we can utilize functionality in np.unique
    # to speed up the calculation siginficantly.
    # If this is not the case, this will take time.
    np_version = np.version.version.split(".")
    if int(np_version[0]) > 1 or int(np_version[1]) > 12:

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

        # We're done
        return ismem_a, ia

    else:
        # Use straightforward search, based on a for loop. This is slow for
        # large arrays, but as the alternative implementation is opaque, and
        # there has been some doubts on its reliability, this version is kept
        # as a safeguard.
        ismem_a = np.zeros(num_a, dtype=np.bool)
        ind_of_a_in_b = np.empty(0)
        for i in range(num_a):
            if sa.ndim == 1:
                diff = np.abs(sb - sa[i])
            else:
                diff = np.sum(np.abs(sb - sa[:, i].reshape((-1, 1))), axis=0)
            if np.any(diff == 0):
                ismem_a[i] = True
                hit = np.where(diff == 0)[0]
                if hit.size > 1:
                    hit = hit[0]
                ind_of_a_in_b = np.append(ind_of_a_in_b, hit)

        return ismem_a, ind_of_a_in_b.astype("int")


# ---------------------------------------------------------


def unique_columns_tol(mat, tol=1e-8, exponent=2):
    """
    Remove duplicates from a point set, for a given distance traveling.

    Resembles Matlab's uniquetol function, as applied to columns. To rather
    work at rows, use a transpose.


    Parameters:
        mat (np.ndarray, nd x n_pts): Columns to be uniquified
        tol (double, optional): Tolerance for when columns are considered equal.
            Should be seen in connection with distance between the points in
            the points (due to rounding errors). Defaults to 1e-8.
        exponent (double, optional): Exponnet in norm used in distance
            calculation. Defaults to 2.

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
    # the new unique function that ships with numpy 1.13. This comes with a
    # significant speedup, in particular for large arrays (runtime has gone
    # from hours to split-seconds - that is, the alternative implementation
    # below is ineffecient).
    # If the current numpy version is older, an ugly hack is possible: Download
    # the file from the numpy repositories, and place it somewhere in
    # $PYHTONPATH, with the name 'numpy_113_unique'.
    if issubclass(mat.dtype.type, np.integer) and tol < 0.5:
        # Obtain version of numpy that was loaded by the import in this module
        np_version = np.__version__.split(".")
        # If we are on numpy 2, or 1.13 or higher, we're good.
        if int(np_version[0]) > 1 or int(np_version[1]) > 12:
            un_ar, new_2_old, old_2_new = np.unique(
                mat, return_index=True, return_inverse=True, axis=1
            )
            return un_ar, new_2_old, old_2_new
        else:
            try:
                import numpy_113_unique

                un_ar, new_2_old, old_2_new = numpy_113_unique.unique_np1130(
                    mat, return_index=True, return_inverse=True, axis=1
                )
                return un_ar, new_2_old, old_2_new
            except:
                pass

    def dist(p, pset):
        " Helper function to compute distance "
        if p.ndim == 1:
            pt = p.reshape((-1, 1))
        else:
            pt = p

        return np.power(
            np.sum(np.power(np.abs(pt - pset), exponent), axis=0), 1 / exponent
        )

    (nd, l) = mat.shape

    # By default, no columns are kept
    keep = np.zeros(l, dtype=np.bool)

    # We will however keep the first point
    keep[0] = True
    keep_counter = 1

    # Map from old points to the unique subspace. Defaults to map to itself.
    old_2_new = np.arange(l)

    # Loop over all points, check if it is already represented in the kept list
    for i in range(1, l):
        proximate = np.argwhere(dist(mat[:, i], mat[:, keep]) < tol * np.sqrt(nd))

        if proximate.size > 0:
            # We will not keep this point
            old_2_new[i] = proximate[0]
        else:
            # We have found a new point
            keep[i] = True
            old_2_new[i] = keep_counter
            keep_counter += 1
    # Finally find which elements we kept
    new_2_old = np.argwhere(keep).ravel()

    return mat[:, keep], new_2_old, old_2_new
