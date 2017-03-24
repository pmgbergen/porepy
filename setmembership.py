# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:16:42 2016

@author: keile
"""

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
    b = np.ascontiguousarray(data).view(np.dtype((np.void,
                                                  data.dtype.itemsize * data.shape[1])))
    _, ia = np.unique(b, return_index=True)
    _, ic = np.unique(b, return_inverse=True)
#    return np.unique(b).view(data.dtype).reshape(-1, data.shape[1]), ia, ic
    return data[ia], ia, ic


def _asvoid(arr):
    """

    Taken from
    http://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index

    View the array as dtype np.void (bytes)
    This views the last axis of ND-arrays as bytes so you can perform
    comparisons on the entire row.
    http://stackoverflow.com/a/16840350/190597 (Jaime, 2013-05)
    Warning: When using asvoid for comparison, note that float zeros may
    compare UNEQUALLY
    >>> asvoid([-0.]) == asvoid([0.])
    array([False], dtype=bool)
    """
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


def _find_occ(a, b):
    """
    Find index of occurences of a in b.

    The function has only been tested on np.arrays, but it should be fairly
    general (only require iterables?)

    Code snippet found at
    http://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function?rq=1

    """
    # Base search on a dictionary

    bind = {}
    # Invert dictionary to create a map from an item in b to the *first*
    # occurence of the item.
    # NOTE: If we ever need to give the option of returning last index, it
    # should require no more than bypassing the if statement.
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    # Use inverse mapping to obtain
    return [bind.get(itm, None) for itm in a]


def ismember_rows(a, b, sort=True, simple_version=False):
    """
    Find *columns* of a that are also members of *columns* of b.

    The function mimics Matlab's function ismember(..., 'rows').

    TODO: Rename function, this is confusing!

    Parameters:
        a (np.array): Each column in a will search for an equal in b.
        b (np.array): Array in which we will look for a twin
        sort (boolean, optional): If true, the arrays will be sorted before
            seraching, increasing the chances for a match. Defaults to True.
        simple_verion (boolean, optional): Use an alternative implementation
            based on a global for loop. The code is slow for large arrays, but
            easy to understand. Defaults to False.

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

    num_a = a.shape[-1]

    if simple_version:
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

        return ismem_a, ind_of_a_in_b.astype('int')

    else:
        if a.ndim == 1:
            # Special treatment of 1d, vstack of voids (below) ran into trouble
            # here.
            unq, k, count = np.unique(np.hstack((a, b)), return_inverse=True,
                                      return_counts=True)
            _, k_a, count_a = np.unique(a, return_inverse=True,
                                        return_counts=True)
        else:
            # Represent the arrays as voids to facilitate quick comparison
            voida = _asvoid(sa.transpose())
            voidb = _asvoid(sb.transpose())

            # Use unique to count the number of occurences in a
            unq, j, k, count = np.unique(np.vstack((voida, voidb)),
                                         return_index=True,
                                         return_inverse=True,
                                         return_counts=True)
            # Also count the number of occurences in voida
            _, j_a, k_a, count_a = np.unique(voida, return_index=True,
                                             return_inverse=True,
                                             return_counts=True)

        # Index of a and b elements in the combined array
        ind_a = np.arange(num_a)
        ind_b = num_a + np.arange(b.shape[-1])

        # Count number of occurences in combine array, and in a only
        num_occ_a_and_b = count[k[ind_a]]
        num_occ_a = count_a[k_a[ind_a]]

        # Subtraction gives number of a in b
        num_occ_a_in_b = num_occ_a_and_b - num_occ_a
        ismem_a = (num_occ_a_in_b > 0)

        # To get the indices of common elements in a and b, compare the
        # elements in k (pointers to elements in the unique combined arary)
        occ_a = k[ind_a[ismem_a]]
        occ_b = k[ind_b]

        ind_of_a_in_b = _find_occ(occ_a, occ_b)
        # Remove None types when no hit was found
        ind_of_a_in_b = [i for i in ind_of_a_in_b if i is not None]

        return ismem_a, np.array(ind_of_a_in_b, dtype='int')

#---------------------------------------------------------


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

    def dist(p, pset):
        " Helper function to compute distance "
        if p.ndim == 1:
            pt = p.reshape((-1, 1))
        else:
            pt = p

        return np.power(np.sum(np.power(np.abs(pt - pset), exponent),
                               axis=0), 1 / exponent)

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
        proximate = np.argwhere(
            dist(mat[:, i], mat[:, keep]) < tol * np.sqrt(nd))

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
