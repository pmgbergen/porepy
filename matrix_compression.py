# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 08:36:47 2016

@author: keile
"""

import numpy as np

def rldecode(A,n):
    """ Decode compressed information. 
        
        The code is heavily inspired by MRST's function with the same name, 
        however, requirements on the shape of functions are probably somewhat
        different.
        
        >>> rlencode(np.array([1, 2, 3]), np.array([2, 3, 1]))
        [1, 1, 2, 2, 2, 3]
        
        >>> rlencode(np.array([0, 2]), np.array([0, 3]))
        [2, 2, 2]
        
        Args:
            A (double, m x k), compressed matrix to be recovered. The 
            compression should be along dimension 1
            n (int): Number of occurences for each element
    """
    r = n > 0         
    i = np.cumsum(np.hstack((np.zeros(1), n[r])), dtype='>i4')
    j = np.zeros(i[-1])
    j[i[1:-1:]] = 1
    B = A[np.cumsum(j, dtype='>i4')]
    return B

def rlencode(A):
    """ Compress matrix by looking for identical columns. """
    comp = A[::, 0:-1] != A[::, 1::]
    i = np.any(comp, axis=0)
    i = np.hstack((np.argwhere(i).ravel(), (A.shape[1]-1)))
    
    num = np.diff(np.hstack((np.array([-1]), i)))
    
    return A[::, i], num
    
    
if __name__ == '__main__':
    A = np.array([[1, 2, 2, 3],[2, 2, 2, 3]])
    rlencode(A)