""" Utility function for permutation of numbers.
"""


def multinary_permutations(base, length):
    """
    Define a generator over all numbers of a certain length for a number system
    with a specified base.

    For details on the decomposition into an arbitrary base see
        http://math.stackexchange.com/questions/111150/changing-a-number-between-arbitrary-bases

    Note that the generator will loop over base**length combinations.

    Examples:

        Construct the numbers [00] to [11] in binary numbers
        >>> multinary_permutations(2, 2)
        [array([ 0.,  0.]), array([ 1.,  0.]), array([ 0.,  1.]), array([ 1.,  1.])]

        Construct the numbers from 0 to 99 (permuted) in the decimal number
        system.
        >>> it = multinary_permutation(10, 2)

    Parameters:
        base (int): Base of the number system
        length (int): Number of digits in the numbers

    Yields:
        array, size length: Array describing the next number combination.

    """

    # There are in total base ** length numbers to be covered, these need to be
    # rewritten into the base number system
    for iter1 in range(base ** length):

        # Array to store the multi-d index of the current index
        bit_val = [0] * length
        # Number to be decomposed
        v = iter1

        # Loop over all digits, find the expression of v in that system
        for iter2 in range(length):
            bit_val[iter2] = v % base
            v = v // base
        # Yield the next value
        yield bit_val
