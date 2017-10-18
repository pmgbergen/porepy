import unittest
import scipy.sparse as sps
import numpy as np


from porepy.utils import sparse_mat


class TestSparseMath(unittest.TestCase):
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

    def test_csr_slice(self):
        # Test slicing of csr_matrix
        A = sps.csr_matrix(np.array([[0, 0, 0],
                                     [1, 0, 0],
                                     [0, 0, 3]]))
        cols_0 = sparse_mat.slice_indices(A, np.array([0]))
        cols_2 = sparse_mat.slice_indices(A, np.array([2]))
        cols0_2 = sparse_mat.slice_indices(A, np.array([0, 1, 2]))

        assert cols_0.size == 0
        assert cols_2 == np.array([2])
        assert np.all(cols0_2 == np.array([0, 2]))

    def test_csc_slice(self):
        # Test slicing of csr_matrix
        A = sps.csc_matrix(np.array([[0, 0, 0],
                                     [1, 0, 0],
                                     [0, 0, 3]]))
        rows_0 = sparse_mat.slice_indices(A, np.array([0]))
        rows_2 = sparse_mat.slice_indices(A, np.array([2]))
        rows0_2 = sparse_mat.slice_indices(A, np.array([0, 1, 2]))

        assert rows_0 == np.array([1])
        assert rows_2 == np.array([2])
        assert np.all(rows0_2 == np.array([1, 2]))

    def test_zero_columns(self):
        # Test slicing of csr_matrix
        A = sps.csc_matrix(np.array([[0, 0, 0],
                                     [1, 0, 0],
                                     [0, 0, 3]]))
        A0 = A.copy()
        A2 = A.copy()
        A0_2 = A.copy()
        sparse_mat.zero_columns(A0, np.array([0]))
        sparse_mat.zero_columns(A2, np.array([2]))
        sparse_mat.zero_columns(A0_2, np.array([0, 1, 2]))

        assert np.sum(A0[:, 0] != 0) == 0
        assert np.sum(A2[:, 2] != 0) == 0
        assert np.sum(A0_2[:, 0:3] != 0) == 0

    if __name__ == '__main__':
        unittest.main()
