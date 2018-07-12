""" Functions for compressing matrices to compact format, and recover them.

Acknowledgements:
    The functions are a python translation of the corresponding matlab
    functions found in the Matlab Reservoir Simulation Toolbox (MRST) developed
    by SINTEF ICT, see www.sintef.no/projectweb/mrst/ . 
    
"""

import numpy as np


def rldecode(A, n):
    """ Decode compressed information. 
        
        The code is heavily inspired by MRST's function with the same name, 
        however, requirements on the shape of functions are probably somewhat
        different.
        
        >>> rldecode(np.array([1, 2, 3]), np.array([2, 3, 1]))
        [1, 1, 2, 2, 2, 3]
        
        >>> rldecode(np.array([1, 2]), np.array([1, 3]))
        [1, 2, 2, 2]
        
        Args:
            A (double, m x k), compressed matrix to be recovered. The 
            compression should be along dimension 1
            n (int): Number of occurences for each element
    """
    r = n > 0
    i = np.cumsum(np.hstack((np.zeros(1), n[r])), dtype=">i4")
    j = np.zeros(i[-1])
    j[i[1:-1:]] = 1
    B = A[np.cumsum(j, dtype=">i4")]
    return B


def rlencode(A):
    """ Compress matrix by looking for identical columns. """
    comp = A[::, 0:-1] != A[::, 1::]
    i = np.any(comp, axis=0)
    i = np.hstack((np.argwhere(i).ravel(), (A.shape[1] - 1)))

    num = np.diff(np.hstack((np.array([-1]), i)))

    return A[::, i], num


if __name__ == "__main__":
    import doctest

    doctest.testmod()
