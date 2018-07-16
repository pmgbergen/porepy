import unittest
import scipy.sparse as sps
import numpy as np


from porepy.utils import sparse_mat


class TestSparseMath(unittest.TestCase):
    def zero_1_column(self):
        A = sps.csc_matrix(
            (np.array([1, 2, 3]), (np.array([0, 2, 1]), np.array([0, 1, 2]))),
            shape=(3, 3),
        )
        sparse_mat.zero_columns(A, 0)
        assert np.all(A.A == np.array([[0, 0, 0], [0, 0, 3], [0, 2, 0]]))
        assert A.nnz == 3
        assert A.getformat() == "csc"

    def zero_2_columns(self):
        A = sps.csc_matrix(
            (np.array([1, 2, 3]), (np.array([0, 2, 1]), np.array([0, 1, 2]))),
            shape=(3, 3),
        )
        sparse_mat.zero_columns(A, np.array([0, 2]))
        assert np.all(A.A == np.array([[0, 0, 0], [0, 0, 0], [0, 2, 0]]))
        assert A.nnz == 3
        assert A.getformat() == "csc"

    def test_zero_columns(self):
        # Test slicing of csr_matrix
        A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        A0_t = sps.csc_matrix(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 3]]))
        A2_t = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]))
        A0_2_t = sps.csc_matrix(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
        A0 = A.copy()
        A2 = A.copy()
        A0_2 = A.copy()
        sparse_mat.zero_columns(A0, np.array([0], dtype=int))
        sparse_mat.zero_columns(A2, 2)
        sparse_mat.zero_columns(A0_2, np.array([0, 1, 2]))

        assert np.sum(A0 != A0_t) == 0
        assert np.sum(A2 != A2_t) == 0
        assert np.sum(A0_2 != A0_2_t) == 0

    # ------------------- get slicing indices ------------------
    def test_csr_slice(self):
        # Test slicing of csr_matrix
        A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        cols_0 = sparse_mat.slice_indices(A, np.array([0]))
        cols_1 = sparse_mat.slice_indices(A, 1)
        cols_2 = sparse_mat.slice_indices(A, 2)
        cols_split = sparse_mat.slice_indices(A, np.array([0, 2]))
        cols0_2 = sparse_mat.slice_indices(A, np.array([0, 1, 2]))

        assert cols_0.size == 0
        assert cols_1 == np.array([0])
        assert cols_2 == np.array([2])
        assert np.all(cols_split == np.array([2]))
        assert np.all(cols0_2 == np.array([0, 2]))

    def test_csc_slice(self):
        # Test slicing of csr_matrix
        A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))
        rows_0 = sparse_mat.slice_indices(A, np.array([0], dtype=int))
        rows_1 = sparse_mat.slice_indices(A, 1)
        rows_2 = sparse_mat.slice_indices(A, 2)
        cols_split = sparse_mat.slice_indices(A, np.array([0, 2]))
        rows0_2 = sparse_mat.slice_indices(A, np.array([0, 1, 2]))

        assert rows_0 == np.array([1])
        assert rows_1.size == 0
        assert rows_2 == np.array([2])
        assert np.all(cols_split == np.array([1, 2]))
        assert np.all(rows0_2 == np.array([1, 2]))

    # ------------------ Test sliced_mat() -----------------------
    def test_sliced_mat_columns(self):
        # Test slicing of csr_matrix

        # original matrix
        A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        # expected results
        A0_t = sps.csc_matrix(np.array([[0, 0], [0, 0], [0, 3]]))
        A1_t = sps.csc_matrix(np.array([[0, 0], [1, 0], [0, 0]]))
        A2_t = sps.csc_matrix(np.array([[0], [0], [3]]))
        A3_t = sps.csc_matrix(np.array([[], [], []]))
        A4_t = sps.csc_matrix(np.array([[0, 0], [1, 0], [0, 3]]))

        A5_t = sps.csc_matrix(np.array([[0, 0], [0, 0], [0, 0]]))

        A0 = sparse_mat.slice_mat(A, np.array([1, 2], dtype=int))
        A1 = sparse_mat.slice_mat(A, np.array([0, 1]))
        A2 = sparse_mat.slice_mat(A, 2)
        A3 = sparse_mat.slice_mat(A, np.array([], dtype=np.int))
        A4 = sparse_mat.slice_mat(A, np.array([0, 2], dtype=np.int))
        A5 = sparse_mat.slice_mat(A, np.array([1, 1], dtype=np.int))

        assert np.sum(A0 != A0_t) == 0
        assert np.sum(A1 != A1_t) == 0
        assert np.sum(A2 != A2_t) == 0
        assert np.sum(A3 != A3_t) == 0
        assert np.sum(A4 != A4_t) == 0
        assert np.sum(A5 != A5_t) == 0

    def test_sliced_mat_rows(self):
        # Test slicing of csr_matrix
        A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        A0_t = sps.csr_matrix(np.array([[1, 0, 0], [0, 0, 3]]))
        A1_t = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0]]))
        A2_t = sps.csr_matrix(np.array([[0, 0, 3]]))
        A3_t = sps.csr_matrix(np.atleast_2d(np.array([[], [], []])).T)
        A4_t = sps.csr_matrix(np.array([[0, 0, 0], [0, 0, 3]]))
        A5_t = sps.csr_matrix(np.array([[1, 0, 0], [1, 0, 0]]))

        A0 = sparse_mat.slice_mat(A, np.array([1, 2], dtype=int))
        A1 = sparse_mat.slice_mat(A, np.array([0, 1]))
        A2 = sparse_mat.slice_mat(A, 2)
        A3 = sparse_mat.slice_mat(A, np.array([], dtype=np.int))
        A4 = sparse_mat.slice_mat(A, np.array([0, 2], dtype=np.int))
        A5 = sparse_mat.slice_mat(A, np.array([1, 1], dtype=np.int))

        assert np.sum(A0 != A0_t) == 0
        assert np.sum(A1 != A1_t) == 0
        assert np.sum(A2 != A2_t) == 0
        assert np.sum(A3 != A3_t) == 0
        assert np.sum(A4 != A4_t) == 0
        assert np.sum(A5 != A5_t) == 0

    # ------------------ Test stack_mat() -----------------------
    def test_stack_mat_columns(self):
        # Test slicing of csr_matrix
        A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        B = sps.csc_matrix(np.array([[0, 2], [3, 1], [1, 0]]))

        A_t = sps.csc_matrix(
            np.array([[0, 0, 0, 0, 2], [1, 0, 0, 3, 1], [0, 0, 3, 1, 0]])
        )

        sparse_mat.stack_mat(A, B)

        assert np.sum(A != A_t) == 0

    def test_stack_empty_mat_columns(self):
        # Test slicing of csr_matrix
        A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        B = sps.csc_matrix(np.array([[], [], []]))

        A_t = A.copy()
        sparse_mat.stack_mat(A, B)
        assert np.sum(A != A_t) == 0
        B_t = A.copy()
        sparse_mat.stack_mat(B, A)
        assert np.sum(B != B_t) == 0

    def test_stack_mat_rows(self):
        # Test slicing of csr_matrix
        A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        B = sps.csr_matrix(np.array([[0, 2, 2], [3, 1, 3]]))

        A_t = sps.csr_matrix(
            np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3], [0, 2, 2], [3, 1, 3]])
        )

        sparse_mat.stack_mat(A, B)

        assert np.sum(A != A_t) == 0

    def test_stack_empty_mat_rows(self):
        # Test slicing of csr_matrix
        A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        B = sps.csr_matrix(np.array([[], [], []]).T)

        A_t = A.copy()
        sparse_mat.stack_mat(A, B)
        assert np.sum(A != A_t) == 0
        B_t = A.copy()
        sparse_mat.stack_mat(B, A)
        assert np.sum(B != B_t) == 0

    # ------------------ Test merge_mat() -----------------------

    def test_merge_mat_split_columns(self):
        # Test slicing of csr_matrix
        A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        B = sps.csc_matrix(np.array([[0, 2], [3, 1], [1, 0]]))

        A_t = sps.csc_matrix(np.array([[0, 0, 2], [3, 0, 1], [1, 0, 0]]))

        sparse_mat.merge_matrices(A, B, np.array([0, 2], dtype=np.int))

        assert np.sum(A != A_t) == 0

    def test_merge_mat_columns(self):
        # Test slicing of csr_matrix
        A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        B = sps.csc_matrix(np.array([[0, 2], [3, 1], [1, 0]]))

        A_t = sps.csc_matrix(np.array([[0, 0, 2], [1, 3, 1], [0, 1, 0]]))

        sparse_mat.merge_matrices(A, B, np.array([1, 2], dtype=np.int))

        assert np.sum(A != A_t) == 0

    def test_merge_mat_split_columns_same_pos(self):
        # Test slicing of csr_matrix
        A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        B = sps.csc_matrix(np.array([[0, 2], [3, 1], [1, 0]]))

        A_t = sps.csc_matrix(np.array([[2, 0, 0], [1, 0, 0], [0, 0, 3]]))
        try:
            sparse_mat.merge_matrices(A, B, np.array([0, 0], dtype=np.int))
        except ValueError:
            pass

    def test_merge_mat_split_rows(self):
        # Test slicing of csr_matrix
        A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        B = sps.csr_matrix(np.array([[0, 2, 2], [3, 1, 3]]))

        A_t = sps.csr_matrix(np.array([[0, 2, 2], [1, 0, 0], [3, 1, 3]]))

        sparse_mat.merge_matrices(A, B, np.array([0, 2], dtype=np.int))

        assert np.sum(A != A_t) == 0

    def test_merge_mat_rows(self):
        # Test slicing of csr_matrix
        A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

        B = sps.csr_matrix(np.array([[0, 2, 2], [3, 1, 3]]))

        A_t = sps.csr_matrix(np.array([[0, 2, 2], [3, 1, 3], [0, 0, 3]]))

        sparse_mat.merge_matrices(A, B, np.array([0, 1], dtype=np.int))

        assert np.sum(A != A_t) == 0

    if __name__ == "__main__":
        unittest.main()
