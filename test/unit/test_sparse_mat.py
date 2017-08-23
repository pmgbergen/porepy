import unittest
import scipy.sparse as sps
import numpy as np

from porepy.utils import sparse_mat


class TestZeroColumns(unittest.TestCase):
    def zero_1_column(self):
        A = sps.csc_matrix((np.array([1, 2, 3]), (np.array(
            [0, 2, 1]), np.array([0, 1, 2]))), shape=(3, 3))
        sparse_mat.zero_columns(A, 0)
        assert np.all(A.A == np.array([[0, 0, 0], [0, 0, 3], [0, 2, 0]]))
        assert A.nnz == 3
        assert A.getformat() == 'csc'

    def zero_2_columns(self):
        A = sps.csc_matrix((np.array([1, 2, 3]), (np.array(
            [0, 2, 1]), np.array([0, 1, 2]))), shape=(3, 3))
        sparse_mat.zero_columns(A, np.array([0, 2]))
        assert np.all(A.A == np.array([[0, 0, 0], [0, 0, 0], [0, 2, 0]]))
        assert A.nnz == 3
        assert A.getformat() == 'csc'

    if __name__ == '__main__':
        unittest.main()
