"""Efficient numpy.arange for arrays of start and end indices.

Acknowledgements:
    The functions are a python translation of the corresponding matlab
    functions found in the Matlab Reservoir Simulation Toolbox (MRST) developed
    by SINTEF ICT, see www.sintef.no/projectweb/mrst/ .

"""

import numpy as np


def mcolon(lo, hi):
    """Expansion of np.arange(a, b) for arrays a and b.

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
    # IMPLEMENTATION NOTE: The below code uses clever tricks to arrive at the correct
    # result (credit for the cleverness goes to SINTEF). The provided comments should
    # explain the logic behind the code to some extent, but if you really want to
    # understand what is going on, it is probably wise to go through the logic by hand
    # for a few examples.

    if lo.size == 1:
        lo = lo * np.ones(hi.size, dtype=np.int32)
    if hi.size == 1:
        hi = hi * np.ones(lo.size, dtype=np.int32)
    if lo.size != hi.size:
        raise ValueError(
            "Low and high should have same number of elements, or a single item "
        )

    # Find the positive differences. If there are none, return an empty array.
    pos_diff = hi >= lo + 1
    if not any(pos_diff):
        return np.array([], dtype=np.int32)

    # We only need the lows and highs where there is a positive difference.
    lo = lo[pos_diff].astype(int)
    # This changes hi from a non-inclusive to an inclusive upper bound of the range.
    hi = (hi[pos_diff] - 1).astype(int)
    # The number of elements to be generated for each interval (each pair of lo and hi).
    # We have to add 1 to include the upper bound.
    num_elements_in_interval = hi - lo + 1
    # Create an array with size equal to the total number of elements to be generated.
    # Initialize the array with ones, so that, if we do a cumulative sum, the indices
    # will be increasing by one. This will work, provided we can also set the elements
    # corresponding to the start of an interval (a lo value in a lo-hi pair) to its
    # correct value.
    tot_num_elements = np.sum(num_elements_in_interval)
    x = np.ones(tot_num_elements, dtype=int)

    # Set the first element to the first lo value.
    x[0] = lo[0]
    # For the elements corresponding to the start of interval 1, 2, ..., we cannot just
    # set the value to lo[1:], since this will be overriden by the cumulative sum.
    # Instead, set the value to lo[1:] - hi[0:-1], that is, to the difference between
    # the end of the previous interval and the start of the current interval. Under a
    # cumulative sum, this will give the correct value for the start of interval i,
    # provided that interval i-1 ends at the correct value, and, since we set the first
    # element to lo[0], with ones up to lo[1], this all works out by induction. Kudos to
    # whomever came up with this!
    x[np.cumsum(num_elements_in_interval[0:-1])] = lo[1:] - hi[0:-1]
    # Finally, a cumulative sum will give the correct values for all elements. Use x to
    # store the result to avoid allocating a new array (this can give substantial
    # savings for large arrays).
    return np.cumsum(x, out=x)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
