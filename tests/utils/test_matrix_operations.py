"""
Tests for matrix operations for zeroing rows/columns, efficient slicing, stacking,
merging and construction from arrays.
"""

import numpy as np
import pytest
import scipy.sparse as sps
from typing import Literal
import porepy as pp

from porepy import matrix_operations
from porepy.applications.test_utils.arrays import compare_matrices

# ------------------ Test zero_columns -----------------------


@pytest.fixture(scope="module")
def A():
    return sps.csr_matrix(
        np.array(
            [[1, 0, 2, -1], [3, 0, 0, -4], [4, 5, 6, 0], [0, 7, 8, -2], [1, 3, 4, 5]]
        )
    )


@pytest.mark.parametrize("column", [True, False])
@pytest.mark.parametrize("index", [0, np.array([0, 2]), np.array([0])])
def test_zero_column_row(A, column: bool, index: np.ndarray | int):
    """Test that function to zero out columns and rows works as expected.

    Parameters:
        A: sparse matrix to be modified.
        column: if True, zero columns, otherwise rows.
        index: index of the columns or rows to be zeroed.
    """
    A_known = A.toarray()
    if column:
        # If we zero columns, we need to convert to csc format.
        A = A.tocsc()
        matrix_operations.zero_columns(A, index)
        A_known[:, index] = 0
    else:
        matrix_operations.zero_rows(A, index)
        A_known[index, :] = 0

    assert np.allclose(A.toarray(), A_known)


def test_zero_columns_assert():
    A = sps.csr_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 3]]))
    with pytest.raises(ValueError):
        # Should be csc_matrix
        matrix_operations.zero_columns(A, 1)


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


# ------------------ Test slice_sparse_matrix -----------------------


@pytest.mark.parametrize("column", [True, False])
@pytest.mark.parametrize(
    "index",
    [
        2,
        np.array([0, 2]),
        np.array([2]),
        np.array([0, 1, 3]),
        np.array([1, 1]),
        np.array([1, 0, 0, 1, 0], dtype=bool),
    ],
)
def test_sliced_matrix(A, column: bool, index: int | np.ndarray):
    """Test that slicing a sparse matrix works as expected.

    Parameters:
        A: sparse matrix to be sliced.
        column: if True, slice columns, otherwise rows.
        index: index of the columns or rows to be sliced.
    """
    if column:
        # If we slice columns, we need to convert to csc format.
        A = A.tocsc()
        if isinstance(index, np.ndarray) and index.dtype == bool:
            index = index[: A.shape[1]]
        A_known = A[:, index].toarray()
    else:
        A_known = A[index].toarray()

    assert np.allclose(
        matrix_operations.slice_sparse_matrix(A, index).toarray(), A_known
    )


# ------------------ Test ArraySlicer -----------------------


def _get_arrayslicer_target(mat, mode: Literal["float", "dense", "sparse", "ad"]):
    """Get the target matrix for the array slicer test.

    Parameters:
        mat: The matrix to be sliced

    Returns:
        The target quantity to be sliced.

    """
    if mode == "sparse":
        target = mat
    elif mode == "float":
        return 42.0
    else:
        vec = np.array([1, 2, 3, 4, 5])
        if mode == "dense":
            # The dense mode uses a vector. In principle, it is possible also to
            # consider 2d numpy arrays, but the ArraySlicer is not designed to handle
            # this case.
            target = vec
        else:
            target = pp.ad.AdArray(vec, mat)
    return target


@pytest.mark.parametrize("mode", ["dense", "sparse", "ad", "float"])
@pytest.mark.parametrize(
    "domain_inds, range_inds",
    [
        # Extract selected rows, insert at specified rows in the range domain.
        (np.array([3, 1]), np.array([0, 3])),
        # Extract selected columns, force the range to have at least five dimensions.
        (np.array([3, 1]), np.array([0, 4])),
        # Extract selected columns. Both domain and range indices vary non-monotonically.
        (np.array([2, 3, 0]), np.array([3, 0, 1])),
        # Domain indices are not specified. The first three rows will be extracted.
        (None, np.array([0, 1, 3])),
        # Range indices are not specified. The mapped rows will be inserted in the first
        # three rows.
        (np.array([0, 1, 3]), None),
    ],
)
# If None, the range size is given by the range indices. The value 6 will force the
# result to have exactly 6 rows, independent of the range indices.
@pytest.mark.parametrize("range_size", [None, 6])
# The domain size is not used for the ArraySlicer, however, when transpose is True, it
# will take the role of the range_size, see description above.
@pytest.mark.parametrize("domain_size", [None, 6])
@pytest.mark.parametrize("transpose", [True, False])
def test_array_slicer(
    A,
    mode: Literal["dense", "sparse", "ad", "float"],
    domain_inds: np.ndarray | None,
    range_inds: np.ndarray | None,
    range_size: int,
    domain_size: int,
    transpose: bool,
):
    """Test that the ArraySlicer acts as expected.

    The test constructs an ArraySlicer with a given domain and range indices, and
    applies it to a target array. By parametrization, this target is etiher a numpy
    array, a scipy sparse matrix, or an AdArray (that is, the three data types on which
    the ArraySlicer is meant to operate).

    A failure in this test indicates that the ArraySlicer does not behave as expected;
    quite likely, something is wrong with the treatment of indices in the slicer class.

    Parameters:
        A: The matrix to be sliced.
        mode: The mode of the target matrix. Can be 'dense', 'sparse' or 'ad'.
        domain_inds: The indices of the domain of the slicer.
        range_inds: The indices of the range of the slicer.
        range_size: The size of the range of the slicer.
        transpose: If True, transpose the slicer before applying it.

    """
    target = _get_arrayslicer_target(A, mode)
    slicer = matrix_operations.ArraySlicer(
        domain_inds, range_inds, range_size, domain_size
    )

    if transpose:
        # First transpose the slicer.
        slicer = slicer.T
        # Now, the number of rows in the resulting matrix is determined from domain
        # information, since the slicer is transposed.
        if domain_size is not None:
            # If the domain size is given, the number of rows is given by this.
            num_rows = domain_size
        elif domain_inds is not None:
            # If no domain size is given but domain indices are, the output number of
            # rows is the maximum index + 1 (because indices are 0-offset).
            num_rows = domain_inds.max() + 1
        else:
            # If no domain size or indices are given, the output number of rows is the
            # number of range indices.
            num_rows = range_inds.size
    else:
        # Get the number of output rows from the range information. The logic is the
        # same as for the transpose case.
        if range_size is not None:
            num_rows = range_size
        elif range_inds is not None:
            num_rows = range_inds.max() + 1
        else:
            num_rows = domain_inds.size

    if range_inds is None:
        range_inds = np.arange(domain_inds.size)
    if domain_inds is None:
        domain_inds = np.arange(range_inds.size)

    result = slicer @ target

    if mode == "float":
        # A scalar target will effectively be broadcast to a vector. The result is known
        # to be a vector. Create an empty one, then fill in the relevant entries from
        # the target.
        known_scalar = target
        known_array = np.zeros(num_rows)
        if transpose:
            known_array[domain_inds] = known_scalar
        else:
            known_array[range_inds] = known_scalar
        assert isinstance(result, np.ndarray)
        assert result.size == num_rows
        np.testing.assert_allclose(result, known_array)

    elif mode == "dense":
        # The result is known to be a vector. Create an empty one, then fill in the
        # relevant entries from the target.
        A_known = np.zeros(num_rows)
        if transpose:
            # Flip the domain and range indices in the transposed case.
            A_known[domain_inds] = target[range_inds]
        else:
            A_known[range_inds] = target[domain_inds]
        assert np.allclose(result, A_known)

    elif mode == "sparse":
        # The result is known to be a sparse matrix. Create an empty one, having the
        # same number of columns as the target, but with the number of rows given by the
        # input to the slicer. Fill in the relevant entries from the target.
        A_known = np.zeros((num_rows, target.shape[1]))
        if transpose:
            # Flip the domain and range indices in the transposed case.
            A_known[domain_inds] = target.toarray()[range_inds]
        else:
            A_known[range_inds] = target.toarray()[domain_inds]
        assert np.allclose(result.toarray(), A_known)
    else:  # AdArray.
        # The logic is the same as in the two above cases. Consider the val and jac
        # attributes of the AdArray separately.
        val_known = np.zeros(num_rows)
        if transpose:
            val_known[domain_inds] = target.val[range_inds]
        else:
            val_known[range_inds] = target.val[domain_inds]
        assert np.allclose(result.val, val_known)

        jac_known = np.zeros((num_rows, target.jac.shape[1]))
        if transpose:
            jac_known[domain_inds] = target.jac.toarray()[range_inds]
        else:
            jac_known[range_inds] = target.jac.toarray()[domain_inds]
        assert np.allclose(result.jac.toarray(), jac_known)


@pytest.mark.parametrize(
    "other_mode, target_mode, operator",
    [
        # The following tuples are in the form (A, B, operator), where A and B are the
        # types of the operands and operator is the operator to be applied. The cases
        # considered span all permissible combinations of operands and operators. Note
        # that the 'other' operator never is dense (numpy array), since delayed
        # evaluation is not supported in this case.
        ("sparse", "dense", "@"),  # Matrix-vector product
        (
            "ad",
            "dense",
            "*",
        ),  # Hadamar product between numpy array and AdArray, with Ad array as the left operand.
        ("sparse", "sparse", "@"),  # Matrix multiplication between sparse matrices.
        ("sparse", "ad", "@"),  # Matrix-AdArray product.
        (
            "sparse",
            "float",
            "@",
        ),  # Broadcasting of a float, followed by matrix-vector product.
        ("ad", "ad", "*"),  # Hadamar product between AdArrays.
        ("ad", "ad", "/"),  # Division between AdArrays.
        ("ad", "dense", "*"),  # Product between AdArray and numpy array.
        ("ad", "dense", "/"),  # Division between AdArray and numpy array.
        ("ad", "float", "*"),  # Broadcasting of a float, followed by Hadamar product.
        ("ad", "float", "/"),  # Broadcasting of a float, followed by division.
        ("float", "ad", "/"),  # Division between float and AdArray.
        ("float", "ad", "*"),  # Product between float and AdArray.
        ("float", "dense", "/"),  # Division between float and numpy array.
        ("float", "dense", "*"),  # Product between float and numpy array.
        (
            "float",
            "float",
            "*",
        ),  # Broadcasting of the target float, followed by scaling.
        (
            "float",
            "float",
            "/",
        ),  # Broadcasting of the target float, followed by scaling.
    ],
)
def test_matrix_slicer_delayed_evaluation(A, other_mode, target_mode, operator):
    """Test the delayed evaluation of the matrix slicer.

    When the matrix slicer is involved in a three-term operation on the form

        other_operand operator matrix_slicer @ target

    the matrix slicer will in some cases delay the evaluation of the operation, so that
    the expression is evaluated as

        other_operand operator (matrix_slicer @ target)

    This test checks that the delayed evaluation works as expected.

    Parameters:
        A: The matrix to be sliced.
        other_mode: The mode of the other operand. Can be 'dense', 'sparse', 'ad' or
            'float'.
        target_mode: The mode of the target matrix. Can be 'dense', 'sparse', 'ad' or
            'float'.
        operator: The operator to be applied.


    """
    other_operand = _get_arrayslicer_target(A, other_mode)
    slicer_target = _get_arrayslicer_target(A, target_mode)

    other_indices = np.array([0, 2])
    if other_mode == "dense" or other_mode == "ad":
        domain_indices = other_indices
    elif other_mode == "ad" and target_mode == "dense":
        domain_indices = other_indices
    else:
        domain_indices = np.array([0, 1, 2, 3])

    if other_mode == "float":
        other_operand_sliced = other_operand
    else:
        other_operand_sliced = other_operand[other_indices]

    slicer = matrix_operations.ArraySlicer(domain_indices=domain_indices)

    temp_result = eval(f"other_operand_sliced {operator} slicer")

    result = temp_result @ slicer_target

    if target_mode == "float":
        slicer_target = np.full(domain_indices.max() + 1, slicer_target)

    sliced_target = slicer_target[domain_indices]

    known_result = eval(f"other_operand_sliced {operator} sliced_target")
    if target_mode == "ad" or other_mode == "ad":
        assert np.allclose(result.val, known_result.val)
        assert np.allclose(result.jac.toarray(), known_result.jac.toarray())
    elif target_mode == "sparse":
        assert np.allclose(result.toarray(), known_result.toarray())
    else:
        assert np.allclose(result, known_result)


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


# ------------------ Test cs{r,s}_matrix_from_dense_blocks -----------------------


@pytest.mark.parametrize("block_size", [2, 3])
@pytest.mark.parametrize("format", ["csr", "csc"])
def test_dense_matrix_from_single_block(block_size, format):
    """Test the conversion of a dense block to a sparse matrix. Both for csr and csc
    format. A single block is used, hence the constructed and the original matrix
    should be the same.
    """
    arr = np.arange(block_size**2)

    known = arr.reshape((block_size, block_size))

    if format == "csc":
        # CSC will fill the matrix column-wise.
        known = known.T
        value = matrix_operations.csc_matrix_from_dense_blocks(arr, block_size, 1)
    else:
        value = matrix_operations.csr_matrix_from_dense_blocks(arr, block_size, 1)

    assert np.all(known == value.toarray())


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_dense_matrix_two_blocks(format):
    """Test the conversion of a dense block to a sparse matrix. Both for csr and csc
    format. Two blocks are provided. The matrix should be filled with the blocks on the
    block diagonal.
    """
    block_size = 2
    num_blocks = 2
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # The method will fill the diagonal blocks of the target matrix.
    known = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6], [0, 0, 7, 8]])

    if format == "csc":
        # CSC will fill the matrix column-wise.
        known = known.T
        value = matrix_operations.csc_matrix_from_dense_blocks(
            arr, block_size, num_blocks
        )
    else:
        value = matrix_operations.csr_matrix_from_dense_blocks(
            arr, block_size, num_blocks
        )

    assert np.all(known == value.toarray())


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_sparse_block_matrix_from_sparse(format):
    """Test the conversion of a sparse block matrix to a sparse matrix. Both for csr and
    csc format. The block matrix is constructed from two sparse blocks. The matrix
    should be filled with the blocks on the block diagonal.
    """
    # Make two blocks, one in csr and one in csc format.
    blocks = [
        sps.csr_matrix(np.array([[1, 2], [3, 4]])),
        sps.csc_matrix(np.array([[5, 6, 7], [8, 9, 10]])),
    ]
    # The method will fill the diagonal blocks of the target matrix.
    known = np.array(
        [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0], [0, 0, 5, 6, 7], [0, 0, 8, 9, 10]]
    )

    if format == "csc":
        # CSC will fill the matrix column-wise.
        value = matrix_operations.csc_matrix_from_sparse_blocks(blocks)
    else:
        value = matrix_operations.csr_matrix_from_sparse_blocks(blocks)

    assert np.all(known == value.toarray())


def test_diagonal_matrix_from_sparse_blocks():
    """Test the conversion of a sparse block matrix to a sparse matrix. The block matrix
    is constructed from two sparse blocks. The matrix should be filled with the blocks on
    the block diagonal.
    """
    # Make two blocks, one in csr and one in csc format.
    blocks = [
        sps.dia_matrix(np.array([[1, 0], [0, 2]])),
        sps.dia_matrix(
            np.array([[3, 0, 0], [0, 4, 0], [0, 0, 5]]),
        ),
    ]
    # The method will fill the diagonal blocks of the target matrix.
    known = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )

    value = matrix_operations.sparse_dia_from_sparse_blocks(blocks)

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
