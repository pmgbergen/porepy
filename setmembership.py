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
