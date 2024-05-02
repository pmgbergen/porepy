"""
Tests for matrix operations for zeroing rows/columns, efficient slicing, stacking,
merging and construction from arrays.
"""

import numpy as np
import pytest
import scipy.sparse as sps

from porepy import matrix_operations
from porepy.applications.test_utils.arrays import compare_matrices

# ------------------ Test zero_columns -----------------------


def test_zero_1_column():
    A = sps.csc_matrix(
        (np.array([1, 2, 3]), (np.array([0, 2, 1]), np.array([0, 1, 2]))),
        shape=(3, 3),
    )
    matrix_operations.zero_columns(A, 0)
    assert np.all(A.toarray() == np.array([[0, 0, 0], [0, 0, 3], [0, 2, 0]]))
    assert A.nnz == 3
    assert A.getformat() == "csc"


def test_zero_2_columns():
    A = sps.csc_matrix(
        (np.array([1, 2, 3]), (np.array([0, 2, 1]), np.array([0, 1, 2]))),
        shape=(3, 3),
    )
    matrix_operations.zero_columns(A, np.array([0, 2]))
    assert np.all(A.toarray() == np.array([[0, 0, 0], [0, 0, 0], [0, 2, 0]]))
    assert A.nnz == 3
    assert A.getformat() == "csc"


def test_zero_columns():
    A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    A0_t = sps.csc_matrix(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 3]]))
    A2_t = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]))
    A0_2_t = sps.csc_matrix(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    A0 = A.copy()
    A2 = A.copy()
    A0_2 = A.copy()
    matrix_operations.zero_columns(A0, np.array([0], dtype=int))
    matrix_operations.zero_columns(A2, 2)
    matrix_operations.zero_columns(A0_2, np.array([0, 1, 2]))

    assert np.sum(A0 != A0_t) == 0
    assert np.sum(A2 != A2_t) == 0
    assert np.sum(A0_2 != A0_2_t) == 0


def test_zero_columns_assert():
    A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))
    with pytest.raises(ValueError):
        # Should be csc_matrix
        matrix_operations.zero_columns(A, 1)


# ------------------ Test zero_rows -----------------------


def test_zero_1_row():
    A = sps.csr_matrix(
        (np.array([1, 2, 3]), (np.array([0, 2, 1]), np.array([0, 1, 2]))),
        shape=(3, 3),
    )
    matrix_operations.zero_rows(A, 0)
    assert np.all(A.toarray() == np.array([[0, 0, 0], [0, 0, 3], [0, 2, 0]]))
    assert A.nnz == 3
    assert A.getformat() == "csr"


def test_zero_2_rows():
    A = sps.csr_matrix(
        (np.array([1, 2, 3]), (np.array([0, 2, 1]), np.array([0, 1, 2]))),
        shape=(3, 3),
    )
    matrix_operations.zero_rows(A, np.array([0, 2]))
    assert np.all(A.toarray() == np.array([[0, 0, 0], [0, 0, 3], [0, 0, 0]]))
    assert A.nnz == 3
    assert A.getformat() == "csr"


def test_zero_rows():
    A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    A0_t = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))
    A2_t = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]))
    A0_2_t = sps.csr_matrix(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    A0 = A.copy()
    A2 = A.copy()
    A0_2 = A.copy()
    matrix_operations.zero_rows(A0, np.array([0], dtype=int))
    matrix_operations.zero_rows(A2, 2)
    matrix_operations.zero_rows(A0_2, np.array([0, 1, 2]))

    assert np.sum(A0 != A0_t) == 0
    assert np.sum(A2 != A2_t) == 0
    assert np.sum(A0_2 != A0_2_t) == 0


def test_zero_rows_assert():
    A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))
    with pytest.raises(ValueError):
        # Should be csr_matrix
        matrix_operations.zero_rows(A, 1)


# ------------------ Test slice_indices -----------------------


def test_csr_slice():
    A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    cols_0 = matrix_operations.slice_indices(A, np.array([0]))
    cols_1 = matrix_operations.slice_indices(A, 1)
    cols_2 = matrix_operations.slice_indices(A, 2)
    cols_split = matrix_operations.slice_indices(A, np.array([0, 2]))
    cols0_2 = matrix_operations.slice_indices(A, np.array([0, 1, 2]))

    assert cols_0.size == 0
    assert cols_1 == np.array([0])
    assert cols_2 == np.array([2])
    assert np.all(cols_split == np.array([2]))
    assert np.all(cols0_2 == np.array([0, 2]))


def test_csc_slice():
    A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))
    rows_0 = matrix_operations.slice_indices(A, np.array([0], dtype=int))
    rows_1 = matrix_operations.slice_indices(A, 1)
    rows_2 = matrix_operations.slice_indices(A, 2)
    cols_split = matrix_operations.slice_indices(A, np.array([0, 2]))
    rows0_2 = matrix_operations.slice_indices(A, np.array([0, 1, 2]))

    assert rows_0 == np.array([1])
    assert rows_1.size == 0
    assert rows_2 == np.array([2])
    assert np.all(cols_split == np.array([1, 2]))
    assert np.all(rows0_2 == np.array([1, 2]))


# ------------------ Test slice_mat -----------------------


def test_sliced_mat_columns():
    # original matrix
    A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    # expected results
    A0_t = sps.csc_matrix(np.array([[0, 0], [0, 0], [0, 3]]))
    A1_t = sps.csc_matrix(np.array([[0, 0], [1, 0], [0, 0]]))
    A2_t = sps.csc_matrix(np.array([[0], [0], [3]]))
    A3_t = sps.csc_matrix(np.array([[], [], []]))
    A4_t = sps.csc_matrix(np.array([[0, 0], [1, 0], [0, 3]]))

    A5_t = sps.csc_matrix(np.array([[0, 0], [0, 0], [0, 0]]))

    A0 = matrix_operations.slice_mat(A, np.array([1, 2], dtype=int))
    A1 = matrix_operations.slice_mat(A, np.array([0, 1]))
    A2 = matrix_operations.slice_mat(A, 2)
    A3 = matrix_operations.slice_mat(A, np.array([], dtype=int))
    A4 = matrix_operations.slice_mat(A, np.array([0, 2], dtype=int))
    A5 = matrix_operations.slice_mat(A, np.array([1, 1], dtype=int))

    assert np.sum(A0 != A0_t) == 0
    assert np.sum(A1 != A1_t) == 0
    assert np.sum(A2 != A2_t) == 0
    assert np.sum(A3 != A3_t) == 0
    assert np.sum(A4 != A4_t) == 0
    assert np.sum(A5 != A5_t) == 0


def test_sliced_mat_rows():
    A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    A0_t = sps.csr_matrix(np.array([[1, 0, 0], [0, 0, 3]]))
    A1_t = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0]]))
    A2_t = sps.csr_matrix(np.array([[0, 0, 3]]))
    A3_t = sps.csr_matrix(np.atleast_2d(np.array([[], [], []])).T)
    A4_t = sps.csr_matrix(np.array([[0, 0, 0], [0, 0, 3]]))
    A5_t = sps.csr_matrix(np.array([[1, 0, 0], [1, 0, 0]]))

    A0 = matrix_operations.slice_mat(A, np.array([1, 2], dtype=int))
    A1 = matrix_operations.slice_mat(A, np.array([0, 1]))
    A2 = matrix_operations.slice_mat(A, 2)
    A3 = matrix_operations.slice_mat(A, np.array([], dtype=int))
    A4 = matrix_operations.slice_mat(A, np.array([0, 2], dtype=int))
    A5 = matrix_operations.slice_mat(A, np.array([1, 1], dtype=int))

    assert np.sum(A0 != A0_t) == 0
    assert np.sum(A1 != A1_t) == 0
    assert np.sum(A2 != A2_t) == 0
    assert np.sum(A3 != A3_t) == 0
    assert np.sum(A4 != A4_t) == 0
    assert np.sum(A5 != A5_t) == 0


# ------------------ Test stack_mat -----------------------


def test_stack_mat_columns():
    A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    B = sps.csc_matrix(np.array([[0, 2], [3, 1], [1, 0]]))

    A_t = sps.csc_matrix(np.array([[0, 0, 0, 0, 2], [1, 0, 0, 3, 1], [0, 0, 3, 1, 0]]))

    matrix_operations.stack_mat(A, B)

    assert compare_matrices(A, A_t)


def test_stack_empty_mat_columns():
    A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    B = sps.csc_matrix(np.array([[], [], []]))

    A_t = A.copy()
    matrix_operations.stack_mat(A, B)
    assert compare_matrices(A, A_t)
    B_t = A.copy()
    matrix_operations.stack_mat(B, A)
    assert compare_matrices(B, B_t)


def test_stack_mat_rows():
    A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    B = sps.csr_matrix(np.array([[0, 2, 2], [3, 1, 3]]))

    A_t = sps.csr_matrix(
        np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3], [0, 2, 2], [3, 1, 3]])
    )

    matrix_operations.stack_mat(A, B)

    assert compare_matrices(A, A_t)


def test_stack_empty_mat_rows():
    A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    B = sps.csr_matrix(np.array([[], [], []]).T)

    A_t = A.copy()
    matrix_operations.stack_mat(A, B)
    assert compare_matrices(A, A_t)
    B_t = A.copy()
    matrix_operations.stack_mat(B, A)
    assert compare_matrices(B, B_t)


# ------------------ Test merge_matrices -----------------------


def test_merge_mat_split_columns():
    A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    B = sps.csc_matrix(np.array([[0, 2], [3, 1], [1, 0]]))

    A_t = sps.csc_matrix(np.array([[0, 0, 2], [3, 0, 1], [1, 0, 0]]))

    matrix_operations.merge_matrices(
        A, B, np.array([0, 2], dtype=int), matrix_format="csc"
    )

    assert compare_matrices(A, A_t)


def test_merge_mat_columns():
    A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    B = sps.csc_matrix(np.array([[0, 2], [3, 1], [1, 0]]))

    A_t = sps.csc_matrix(np.array([[0, 0, 2], [1, 3, 1], [0, 1, 0]]))

    matrix_operations.merge_matrices(
        A, B, np.array([1, 2], dtype=int), matrix_format="csc"
    )

    assert compare_matrices(A, A_t)


def test_merge_mat_split_columns_same_pos():
    A = sps.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    B = sps.csc_matrix(np.array([[0, 2], [3, 1], [1, 0]]))

    with pytest.raises(ValueError):
        matrix_operations.merge_matrices(
            A, B, np.array([0, 0], dtype=int), matrix_format="csc"
        )


def test_merge_mat_split_rows():
    A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    B = sps.csr_matrix(np.array([[0, 2, 2], [3, 1, 3]]))

    A_t = sps.csr_matrix(np.array([[0, 2, 2], [1, 0, 0], [3, 1, 3]]))

    matrix_operations.merge_matrices(
        A, B, np.array([0, 2], dtype=int), matrix_format="csr"
    )

    assert compare_matrices(A, A_t)


def test_merge_mat_rows():
    A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))

    B = sps.csr_matrix(np.array([[0, 2, 2], [3, 1, 3]]))

    A_t = sps.csr_matrix(np.array([[0, 2, 2], [3, 1, 3], [0, 0, 3]]))

    matrix_operations.merge_matrices(
        A, B, np.array([0, 1], dtype=int), matrix_format="csr"
    )

    assert compare_matrices(A, A_t)


# ------------------ Test csr_matrix_from_blocks -----------------------


def test_csr_matrix_from_single_block():
    block_size = 2
    arr = np.arange(block_size**2).reshape((block_size, block_size))

    known = np.array([[0, 1], [2, 3]])
    value = matrix_operations.csr_matrix_from_blocks(arr.ravel("c"), block_size, 1)

    assert np.all(known == value.toarray())

    # Larger block
    block_size = 3
    arr = np.arange(block_size**2).reshape((block_size, block_size))

    known = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    value = matrix_operations.csr_matrix_from_blocks(arr.ravel("c"), block_size, 1)

    assert np.all(known == value.toarray())


def test_csr_matrix_from_two_blocks():
    block_size = 2
    num_blocks = 2
    full_arr = np.arange((num_blocks * block_size) ** 2).reshape(
        (num_blocks * block_size, num_blocks * block_size)
    )

    arr = np.array(
        [full_arr[:block_size, :block_size], full_arr[block_size:, block_size:]]
    )

    known = np.array([[0, 1, 0, 0], [4, 5, 0, 0], [0, 0, 10, 11], [0, 0, 14, 15]])
    value = matrix_operations.csr_matrix_from_blocks(
        arr.ravel("c"), block_size, num_blocks
    )

    assert np.all(known == value.toarray())


def test_csr_matrix_from_array():
    block_size = 2
    num_blocks = 2
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    known = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6], [0, 0, 7, 8]])
    value = matrix_operations.csr_matrix_from_blocks(arr, block_size, num_blocks)

    assert np.all(known == value.toarray())


# ------------------ Test csc_matrix_from_blocks -----------------------


def test_csc_matrix_from_array():
    block_size = 2
    num_blocks = 2
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    known = np.array([[1, 3, 0, 0], [2, 4, 0, 0], [0, 0, 5, 7], [0, 0, 6, 8]])
    value = matrix_operations.csc_matrix_from_blocks(arr, block_size, num_blocks)

    assert np.all(known == value.toarray())


@pytest.mark.parametrize(
    "mat", [np.arange(6).reshape((3, 2)), np.arange(6).reshape((2, 3))]
)
def test_optimized_storage(mat):
    """Check that the optimized sparse storage chooses format according to whether the
    matrix has more columns or rows or opposite.

    Convert input matrix to sparse storage. For the moment, the matrix format is chosen
    according to the number of rows and columns only, thus the lack of true sparsity
    does not matter.
    """
    A = sps.csc_matrix(mat)

    optimized = matrix_operations.optimized_compressed_storage(A)

    if A.shape[0] > A.shape[1]:
        assert optimized.getformat() == "csc"
    else:
        assert optimized.getformat() == "csr"


# ------------------ Test inverting matrices -----------------------


@pytest.fixture(params=["python", "numba"])
def invert_backend(request) -> str:
    # Numba or cython may not be available on the system. For now we consider this okay
    # and skip the test. This behavior may change in the future.
    backend = request.param
    if backend == "numba":
        _ = pytest.importorskip("numba")
    return backend


def test_block_matrix_inverters_full_blocks(invert_backend: str):
    """
    Test inverters for block matrices

    """
    a = np.vstack((np.array([[1, 3], [4, 2]]), np.zeros((3, 2))))
    b = np.vstack((np.zeros((2, 3)), np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]])))
    block_as_csr = sps.csr_matrix(np.hstack((a, b)))
    block_as_csc = sps.csc_matrix(np.hstack((a, b)))

    sz = np.array([2, 3], dtype="i8")
    iblock_ex = np.linalg.inv(block_as_csr.toarray())

    iblock = matrix_operations.invert_diagonal_blocks(
        block_as_csr, sz, method=invert_backend
    )
    assert np.allclose(iblock_ex, iblock.toarray())

    iblock = matrix_operations.invert_diagonal_blocks(
        block_as_csc, sz, method=invert_backend
    )
    assert np.allclose(iblock_ex, iblock.toarray())


def test_block_matrix_invertes_sparse_blocks(invert_backend: str):
    """
    Invert the matrix

    A = [1 2 0 0 0
            3 0 0 0 0
            0 0 3 0 3
            0 0 0 7 0
            0 0 0 1 2]

    Contrary to test_block_matrix_inverters_full_blocks, the blocks of A
    will be sparse. This turned out to give problems
    """

    rows = np.array([0, 0, 1, 2, 2, 3, 4, 4])
    cols = np.array([0, 1, 0, 2, 4, 3, 3, 4])
    data = np.array([1, 2, 3, 3, 3, 7, 1, 2], dtype=np.float64)
    block = sps.coo_matrix((data, (rows, cols))).tocsr()
    sz = np.array([2, 3], dtype="i8")

    iblock = matrix_operations.invert_diagonal_blocks(block, sz, method=invert_backend)
    iblock_ex = np.linalg.inv(block.toarray())

    assert np.allclose(iblock_ex, iblock.toarray())


def test_invert_permutation():
    """
    Test the inverse of a permutation.

    """
    perm = np.random.permutation(np.array([0, 1, 2, 3, 4, 5, 6, 7]))
    inv_perm = matrix_operations.invert_permutation(perm)
    assert np.all(np.isclose(np.argsort(perm), inv_perm))


@pytest.mark.parametrize(
    "sparse_array_type",
    [
        "csr",
        "csc",
    ],
)
def test_sparse_permutation(sparse_array_type):
    """
    Test the permutation of a sparse array. The function checks whether a permuted sparse
    array is the same as its permuted reference dense array. Such permuted reference dense
    array is constructed using numpy slicing.

    """

    # Define data, rows, and columns for the sparse matrix
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    rows = np.array([0, 1, 1, 2, 3, 4, 4, 4, 4, 4])
    cols = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    if sparse_array_type == "csr":
        # Create a CSR matrix
        ref_sparse_array = sps.csr_matrix((data, (rows, cols)), shape=(5, 5))
    else:
        # Create a CSC matrix
        ref_sparse_array = sps.csc_matrix((data, (rows, cols)), shape=(5, 5))

    # reference as a dense array
    ref_dense_array = ref_sparse_array.todense()

    # generate random permutations
    row_perm = np.random.permutation(np.array([0, 1, 2, 3, 4]))
    col_perm = np.random.permutation(np.array([0, 1, 2, 3, 4]))

    # permute the array and generates a new one
    sparse_array = matrix_operations.sparse_permute(
        ref_sparse_array, row_perm, col_perm
    )
    # The nonzero data arrays should not share memory.
    assert not np.shares_memory(ref_sparse_array.data, sparse_array.data)
    # The permuted sparse array represents the same reference dense array
    assert np.all(np.isclose(ref_dense_array[row_perm, :][:, col_perm], sparse_array.A))

    sparse_array = matrix_operations.sparse_permute(
        ref_sparse_array, row_perm, col_perm, True
    )
    # The nonzero data arrays should share memory.
    assert np.shares_memory(ref_sparse_array.data, sparse_array.data)
    # The permuted sparse array represents the same reference dense array
    assert np.all(np.isclose(ref_dense_array[row_perm, :][:, col_perm], sparse_array.A))
