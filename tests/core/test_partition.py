import unittest
import numpy as np

from core.grids import structured
from core.grids import partition

class TestCartGrids(unittest.TestCase):

    def test_2d_coarse_dims_specified(self):
        g = structured.CartGrid([4, 10])
        coarse_dims = np.array([2, 3])
        p = partition.partition_structured(g, coarse_dims)

        p_known = np.array([[ 0.,  0.,  1.,  1.],
                            [ 0.,  0.,  1.,  1.],
                            [ 0.,  0.,  1.,  1.],
                            [ 2.,  2.,  3.,  3.],
                            [ 2.,  2.,  3.,  3.],
                            [ 2.,  2.,  3.,  3.],
                            [ 4.,  4.,  5.,  5.],
                            [ 4.,  4.,  5.,  5.],
                            [ 4.,  4.,  5.,  5.],
                            [ 4.,  4.,  5.,  5.]], dtype='int').ravel('C')
        assert np.allclose(p, p_known)

    def test_3d_coarse_dims_specified(self):
        g = structured.CartGrid([4, 4, 4])
        coarse_dims = np.array([2, 2, 2])

        p = partition.partition_structured(g, coarse_dims)
        p_known = np.array([[0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [2, 2, 3, 3],
                            [2, 2, 3, 3],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [2, 2, 3, 3],
                            [2, 2, 3, 3],
                            [4, 4, 5, 5],
                            [4, 4, 5, 5],
                            [6, 6, 7, 7],
                            [6, 6, 7, 7],
                            [4, 4, 5, 5],
                            [4, 4, 5, 5],
                            [6, 6, 7, 7],
                            [6, 6, 7, 7]]).ravel('C')
        assert np.allclose(p, p_known)

    def test_3d_coarse_dims_specified_unequal_size(self):
        g = structured.CartGrid(np.array([6, 5, 4]))
        coarse_dims = np.array([3, 2, 2])

        p = partition.partition_structured(g, coarse_dims)
        # This just happens to be correct
        p_known = np.array([0,  0,  1,  1,  2,  2,  0,  0,  1, 1,  2,
                            2,  3,  3,  4,  4,  5,  5,  3, 3,  4,  4,
                            5,  5,  3,  3,  4,  4,  5,  5, 0,  0,  1,
                            1,  2,  2,  0,  0,  1,  1,  2, 2,  3,  3,
                            4,  4,  5,  5,  3,  3,  4,  4, 5,  5,  3,
                            3,  4,  4,  5,  5,  6,  6,  7, 7,  8,  8,
                            6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11,
                            11,  9,  9, 10, 10, 11, 11,  9, 9, 10, 10,
                            11, 11,  6,  6,  7,  7,  8,  8, 6,  6,  7,
                            7,  8,  8,  9,  9, 10, 10, 11, 11,  9,  9,
                            10, 10, 11, 11,  9,  9, 10, 10, 11, 11])

        assert np.allclose(p, p_known)


    if __name__ == '__main__':
        unittest.main()


class TestCoarseDimensionDeterminer(unittest.TestCase):

    def test_coarse_dimensions_all_fine(self):
        nx = np.array([5, 5, 5])
        target = nx.prod()
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(nx, coarse)

    def test_coarse_dimensions_single_coarse(self):
        nx = np.array([5, 5, 5])
        target = 1
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(coarse, np.ones_like(nx))

    def test_anisotropic_2d(self):
        nx = np.array([10, 4])
        target = 4
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(coarse, np.array([2, 2]))

    def test_round_down(self):
        nx = np.array([10, 10])
        target = 17
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(coarse, np.array([4, 4]))

    def test_round_up_and_down(self):
        nx = np.array([10, 10])
        target = 19
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(np.sort(coarse), np.array([4, 5]))

    def test_bounded_single(self):
        # The will ideally require a 5x4 grid, but the limit on two cells in
        # the  y-direction should redirect to a 10x2.
        nx = np.array([100, 2])
        target = 20
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(np.sort(coarse), np.array([2, 10]))

    def test_two_bounds(self):
        # This will seemingly require 900^1/3 ~ 10 cells in each direction, but
        # the z-dimension has only 1, thus the two other have to do 30 each. y
        # can only do 15, so the loop has to be reset once more to get the
        # final 60.
        nx = np.array([2000, 15, 1])
        target = 900
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(np.sort(coarse), np.array([1, 15, 60]))

    if __name__ == '__main__':
        unittest.main()
