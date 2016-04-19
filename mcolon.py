# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 21:02:56 2016

@author: keile
"""

import numpy as np


def mcolon(lo, hi):
    """ Get set of expanded indices
    
    >>> mcolon(np.array([0, 0, 0]), np.array([1, 3, 2]))
    array([0, 1, 0, 1, 2, 3, 0, 1, 2])
    
    >>> mcolon(np.array([0, 1]), np.array([2]))
    array([0, 1, 2, 1, 2])

    >>> mcolon(np.array([0, 1, 1, 1]), np.array([0, 3, 3, 3]))
    array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    """
    if lo.size == 1:
        lo = lo * np.ones(hi.size, dtype='int64')
    if hi.size == 1:
        hi = hi * np.ones(lo.size, dtype='int64')
    if lo.size != hi.size:
        raise ValueError('Low and high should have same number of elements, '
                         'or a single item ')

    i = hi >= lo
    if not any(i):
        return None

    lo = lo[i]
    hi = hi[i]
    d = hi - lo + 1
    n = np.sum(d)

    x = np.ones(n, dtype='int64')
    x[0] = lo[0]
    x[np.cumsum(d[0:-1]).astype('int64')] = lo[1:] - hi[0:-1]
    return np.cumsum(x)
