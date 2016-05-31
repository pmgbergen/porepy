# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:16:42 2016

@author: keile
"""

import numpy as np


def unique_rows(data):
    """
    Function similar to Matlab's unique(...,'rows')

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


def asvoid(arr):
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


def find_occ(a, b):
    """


    Code snippet found at
    http://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function?rq=1

    Parameters
    ----------
    a
    b

    Returns
    -------

    """
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]


def ismember_rows(a, b, sort=True):
    """
    Examples
    >>> a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
    >>> b = np.array([[3, 1, 3, 5, 3], [3, 3, 2, 1, 2]])
    >>> ismember_rows(a, b)
    (array([ True,  True,  True,  True, False], dtype=bool), [1, 0, 2, 1])

    >>> a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
    >>> b = np.array([[3, 1, 2, 5, 1], [3, 3, 3, 1, 2]])
    >>> ismember_rows(a, b, sort=False)
    (array([ True,  True, False,  True, False], dtype=bool), [1, 0, 1])
    """
    if sort:
        sa = np.sort(a, axis=0)
        sb = np.sort(b, axis=0)
    else:
        sa = a
        sb = b

    voida = asvoid(sa.transpose())
    voidb = asvoid(sb.transpose())
    unq, j, k, count = np.unique(np.vstack((voida, voidb)), return_index=True,
                                 return_inverse=True, return_counts=True)

    num_a = a.shape[1]
    ind_a = np.arange(num_a)
    ind_b = num_a + np.arange(b.shape[1])
    num_occ_a = count[k[ind_a]]
    ismem_a = (num_occ_a > 1)
    occ_a = k[ind_a[ismem_a]]
    occ_b = k[ind_b]

    ind_of_a_in_b = find_occ(occ_a, occ_b)
    return ismem_a, ind_of_a_in_b
