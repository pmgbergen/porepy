""" Efficient numpy.arange for arrays of start and end indices.

Acknowledgements:
    The functions are a python translation of the corresponding matlab
    functions found in the Matlab Reservoir Simulation Toolbox (MRST) developed
    by SINTEF ICT, see www.sintef.no/projectweb/mrst/ .

"""

import numpy as np


def mcolon(lo, hi):
    """ Expansion of np.arange(a, b) for arrays a and b.

    The code is equivalent to the following (less efficient) loop:
    arr = np.empty(0)
    for l, h in zip(lo, hi):
        arr = np.hstack((arr, np.arange(l, h, 1)))

    Parameters:
        lo (np.ndarray, int): Lower bounds of the arrays to be created.
        hi (np.ndarray, int): Upper bounds of the arrays to be created. The
            elements in hi will *not* be included in the resulting array.

        lo and hi should either have 1 or n elements. If their size are both
        larger than one, they should have the same length.

    Examples:
        >>> mcolon(np.array([0, 0, 0]), np.array([2, 4, 3]))
        array([0, 1, 0, 1, 2, 3, 0, 1, 2])

        >>> mcolon(np.array([0, 1]), np.array([2]))
        array([0, 1, 1])

        >>> mcolon(np.array([0, 1, 1, 1]), np.array([1, 3, 3, 3]))
        array([0, 1, 2, 1, 2, 1, 2])

    """
    if lo.size == 1:
        lo = lo * np.ones(hi.size, dtype="int64")
    if hi.size == 1:
        hi = hi * np.ones(lo.size, dtype="int64")
    if lo.size != hi.size:
        raise ValueError(
            "Low and high should have same number of elements, " "or a single item "
        )

    i = hi >= lo + 1
    if not any(i):
        return np.array([], dtype=np.int32)

    lo = lo[i]
    hi = hi[i] - 1
    d = hi - lo + 1
    n = np.sum(d)

    x = np.ones(n, dtype="int64")
    x[0] = lo[0]
    x[np.cumsum(d[0:-1]).astype("int64")] = lo[1:] - hi[0:-1]
    return np.cumsum(x).astype("int64")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
