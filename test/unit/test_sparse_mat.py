import scipy.sparse as sps
import numpy as np
import unittest

from porepy.utils import sparse_mat


class SparseMatTest(unittest.TestCase):

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

    if __name__ == '__main__':
        unittest.main()
