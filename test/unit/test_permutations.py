"""
Tests of module porepy.utils.permutations.
"""
import unittest
import numpy as np

from porepy.utils import permutations


class TestPermutations(unittest.TestCase):
    def compare_lists(self, base, length, lst):
        # Compare a pre-defined list with the result of multinary_permutations
        # Define a generator, and check that all values produced are contained within lst
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

    def test_length_2(self):
        # Explicitly construct a 2D array of all combination of base 3
        base = 3
        length = 2
        lst = []
        for i in range(base):
            for j in range(base):
                lst.append([i, j])
        self.compare_lists(base, length, lst)

    def test_base_4(self):
        # Check with a manually constructed list of length 3
        base = 4
        length = 3
        lst = []
        for i in range(base):
            for j in range(base):
                for k in range(base):
                    lst.append([i, j, k])
        self.compare_lists(base, length, lst)


if __name__ == "__main__":
    unittest.main()
