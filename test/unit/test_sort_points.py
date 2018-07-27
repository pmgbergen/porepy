import unittest
import numpy as np

from porepy.utils import sort_points


class SortLinePairTest(unittest.TestCase):
    def test_quad(self):
        p = np.array([[1, 2], [5, 1], [2, 7], [7, 5]]).T
        sp = sort_points.sort_point_pairs(p)
        # Use numpy arrays to ease comparison of points
        truth = np.array([[1, 2], [2, 7], [7, 5], [5, 1]]).T
        assert np.allclose(truth, sp)

    if __name__ == "__main__":
        unittest.main()
