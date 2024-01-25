"""Tests of module porepy.utils.permutations."""
import itertools

import numpy as np
import pytest

from porepy.utils import permutations


@pytest.mark.parametrize(
    "base,length",
    [
        (3, 2),
        (4, 3),
        (5, 3),
    ],
)
def test_base_length(base, length):
    lst = []
    for values in itertools.product(*(range(base) for _ in range(length))):
        lst.append([*values])

    # Compare a pre-defined list with the result of multinary_permutations
    # Define a generator, and check that all values produced are contained within lst.
    # Also count the number of iterations
    iter_cnt = 0
    for a in permutations.multinary_permutations(base, length):
        found = False
        for b in lst:
            if np.array_equal(np.array(a), np.array(b)):
                found = True
                break
        assert found
        iter_cnt += 1
    assert iter_cnt == len(lst)
