"""
Tests for matrix operations for zeroing rows/columns, efficient slicing, stacking,
merging and construction from arrays.
"""

from typing import Any, Literal

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy import matrix_operations
from porepy.applications.md_grids.mdg_library import (
    cube_with_orthogonal_fractures,
    square_with_orthogonal_fractures,
)
from porepy.applications.test_utils.arrays import compare_matrices, compare_arrays

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
        # Extract selected columns. Both domain and range indices vary
        # non-monotonically.
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
        ("sparse", "dense", "@"),
        ("sparse", "sparse", "@"),
        ("sparse", "ad", "@"),
        ("sparse", "float", "@"),
    ],
)
def test_matrix_slicer_delayed_evaluation_sparse(
    A: sps.spmatrix,
    other_mode: Literal["sparse"],
    target_mode: Literal["dense", "sparse", "ad", "float"],
    operator: Literal["@"],
):
    """Test the application of the ArraySlicer to a sparse matrix, with delayed
    evaluation. This test is split from the other data types (see just below), since the
    sparse matrix can only be combined with the '@' operator (and the others should
    not), hence the parametrization is different.
    """
    # The actual test is left to a backend function, to avoid code duplication.
    _matrix_slicer_delayed_evaluation_backend(A, other_mode, target_mode, operator)


@pytest.mark.parametrize("other_mode", ["ad", "float"])
@pytest.mark.parametrize("target_mode", ["ad", "dense", "float"])
@pytest.mark.parametrize("operator", ["*", "/", "+", "-", "**"])
def test_matrix_slicer_delayed_evaluation_ad_dense_float(
    A: sps.spmatrix,
    other_mode: Literal["ad", "float"],
    target_mode: Literal["ad", "dense", "float"],
    operator: Literal["*", "/", "+", "-", "**"],
):
    """Test the application of the ArraySlicer to numpy and AdArrays, as well as scalars
    (floats), with delayed evaluation.

    Note that the parametrization of 'other_mode' does not include 'dense' (i.e. a numpy
    array), since this does not make sense in the context of the ArraySlicer. See the
    docstring of that class for more information.
    """
    # The actual test is left to a backend function, to avoid code duplication.
    _matrix_slicer_delayed_evaluation_backend(A, other_mode, target_mode, operator)


def _matrix_slicer_delayed_evaluation_backend(
    A: sps.spmatrix,
    other_mode: Literal["dense", "sparse", "ad", "float"],
    target_mode: Literal["dense", "sparse", "ad", "float"],
    operator: Literal["*", "/", "@", "+", "-", "**"],
):
    """Test the delayed evaluation of the matrix slicer.

    When the matrix slicer is involved in a three-term operation on the form

        (1) other_operand operator matrix_slicer @ target

    the matrix slicer delay the evaluation of the operation, so that the expression is
    evaluated as

        (2) other_operand operator (matrix_slicer @ target)

    This test checks that the delayed evaluation works as expected, that is, that the
    result of (1) is the same as the form (2), with explicitly enforced paratheses.

    Parameters:
        A: The matrix to be sliced.
        other_mode: The data type for the 'other' (leftmost) operand, as described
            above.
        target_mode: The data type for the operand to be sliced (the target of the
            slicing).
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

    # The ArraySlicer will leave a target with a reduced size, as specified by the
    # domain indices. To avoid a size mismatch in the evaluation of the delayed
    # expression, we need to slice the other operand to the same size as the domain
    # indices. The exception is if the other operand is a float, which cannot be sliced.
    if other_mode == "float":
        # Silence ruff errors here, we will use the variable in the below eval
        # statement.
        other_operand_sliced = other_operand  # noqa:F841
    else:
        other_operand_sliced = other_operand[other_indices]  # noqa:F841

    # Silence ruff error about the variable not being used.
    slicer = matrix_operations.ArraySlicer(domain_indices=domain_indices)  # noqa:F841

    # Evaluate the expression without explicit parentheses. This is the form (1)
    # described in the docstring. Under the hood, this forces python to interpret the
    # expression 'other_operand {operator} array_slicer' (ex: float + ArraySlicer).
    # This will invoke the respective __radd__, __rsub__, etc. methods of the
    # ArraySlicer, and the delayed evaluation will be triggered.
    result = eval(f"other_operand_sliced {operator} slicer @ slicer_target")

    # Next we construct a benchmark result by effectively imposing parentheses around
    # the slicing operation. This is the form (2) described in the docstring.
    if target_mode == "float":
        # If the target is a float, we expand it to a vector of the same size as the
        # domain indices. This mimics the behavior of the ArraySlicer, which is expected
        # to broadcast the float.
        slicer_target = np.full(domain_indices.max() + 1, slicer_target)

    # Mimic the effect of the array slicer to the target, by slicing it. This tests only
    # a subset of the functionality of the ArraySlicer, as it only considers the case
    # where the range is [0, 1, .., domain_indices.max()]. However, this is sufficient
    # to test the delayed evaluation; the full functionality of the ArraySlicer is
    # tested elsewhere.
    #
    # Silence ruff error about the variable not being used.
    sliced_target = slicer_target[domain_indices]  # noqa:F841

    # Combine the other operand with the sliced target, using the 'operator'.
    #
    # Implementation note: It is tempting to convert the slicer to a projection matrix,
    # and use that for constructing a known result. However, perhaps due to EK's lack of
    # imagination, this ran into various issues with sizes etc. The explicit slicing
    # above is equivalent, and as good a representation of what the slicer should do.
    known_result = eval(f"other_operand_sliced {operator} sliced_target")

    # Thanks to the delayed evaluation, the result of the two expressions should be the
    # same.
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
    is constructed from two sparse blocks. The matrix should be filled with the blocks
    on the block diagonal.
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


@pytest.fixture(
    params=[
        dict(
            A=sps.csr_matrix(
                np.array(
                    [
                        [1, 0, 2, 0, 0, 0],  # cell 0, var types 0,1,2
                        [0, 1, 0, 3, 0, 0],  # cell 1, var types 0,1,2
                        [1, 0, 4, 0, 0, 0],
                        [0, 1, 0, 5, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                    ]
                ),
                dtype=float,
            ),
            block_sizes=[2, 2, 1, 1],
            row_perm=[0, 2, 1, 3, 4, 5],
            col_perm=[0, 2, 1, 3, 4, 5],
        ),
        dict(
            A=sps.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 2, 0],
                        [0, 0, 0, 1, 0, 3],
                        [0, 0, 1, 0, 4, 0],
                        [0, 0, 0, 1, 0, 5],
                    ]
                ),
                dtype=float,
            ),
            block_sizes=[1, 1, 2, 2],
            row_perm=[0, 1, 2, 4, 3, 5],
            col_perm=[0, 1, 2, 4, 3, 5],
        ),
        dict(
            A=sps.csr_matrix(
                np.array(
                    [
                        [0, 0, 1, 1],
                        [0, 0, 0, 1],
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                    ]
                ),
                dtype=float,
            ),
            block_sizes=[2, 2],
            row_perm=[0, 1, 2, 3],
            col_perm=[2, 3, 0, 1],
        ),
    ]
)
def non_block_diag_matrix(request) -> dict[str, Any]:
    """Fixture to provide a non-diagonal block matrix for testing."""
    return request.param


def test_generate_permutation_to_block_diag_matrix(
    non_block_diag_matrix: dict[str, Any],
):
    """Test that generate_permutation_to_block_diag_matrix correctly identifies the
    blocks in the specified test cases, and that the permutation is represented in a
    format, which compatible with the intended use cases.

    """

    row_perm, col_perm, block_sizes = (
        matrix_operations.generate_permutation_to_block_diag_matrix(
            non_block_diag_matrix["A"]
        )
    )

    # Expect four blocks: two blocks of size 2 (for vars 0&1 on each cell),
    # and two singleton blocks (for var type 2 on each cell)
    assert list(block_sizes) == non_block_diag_matrix["block_sizes"]

    # The permutations  groups eqns/vars by cell:
    assert list(row_perm) == non_block_diag_matrix["row_perm"]
    assert list(col_perm) == non_block_diag_matrix["col_perm"]


def test_invert_permuted_block_diag_mat(non_block_diag_matrix: dict[str, Any]):
    """Test that invert_permuted_block_diag_mat correctly inverts a block-structured
    sparse matrix by permuting to block-diagonal, inverting each block, and
    undoing the permutations.
    """

    A = non_block_diag_matrix["A"]
    row_perm, col_perm, block_sizes = (
        matrix_operations.generate_permutation_to_block_diag_matrix(
            non_block_diag_matrix["A"]
        )
    )

    A_inv = matrix_operations.invert_permuted_block_diag_matrix(
        A, row_perm, col_perm, block_sizes
    )

    # Verify that A * A_inv is the identity matrix
    approx_identity = A.dot(A_inv).toarray()
    assert np.allclose(approx_identity, np.eye(A.shape[0]))


@pytest.mark.parametrize(
    "mdg",
    [
        square_with_orthogonal_fractures("cartesian", {"cell_size": 0.25}, [0, 1])[0],
        cube_with_orthogonal_fractures("cartesian", {"cell_size": 0.25}, [0, 1, 2])[0],
    ],
)
def test_invert_permuted_block_diag_mat_on_mdg(mdg: pp.MixedDimensionalGrid):
    """Construct a mixed-dimensional system, register variables and equations on
    subdomains and interfaces, then build and invert its Schur-complement secondary
    block via block-diagonal permutations."""

    # Instantiate an EquationSystem on the MD grid.
    equation_system = pp.ad.EquationSystem(mdg)

    # Create four “subdomain” variables (p1, p2, s1, s2), one DOF per cell.
    p1 = equation_system.create_variables(name="p1", subdomains=mdg.subdomains())
    p2 = equation_system.create_variables(name="p2", subdomains=mdg.subdomains())
    s1 = equation_system.create_variables(name="s1", subdomains=mdg.subdomains())
    s2 = equation_system.create_variables(name="s2", subdomains=mdg.subdomains())

    # Create three “interface” variables (pf, sf1, sf2), one DOF per interface cell.
    pf = equation_system.create_variables(name="pf", interfaces=mdg.interfaces())
    sf1 = equation_system.create_variables(name="sf1", interfaces=mdg.interfaces())
    sf2 = equation_system.create_variables(name="sf2", interfaces=mdg.interfaces())

    # Define equation‐to‐grid-entity mapping: one equation per cell.
    eq_per_gridEntity = {"cells": 1, "faces": 0, "nodes": 0}

    # On each subdomain cell, register 4 equations.
    Scalar = pp.ad.Scalar
    expr_p1 = p1 + s1 * Scalar(2.0) - Scalar(1.0)
    expr_p1.set_name("eq_p1")
    equation_system.set_equation(expr_p1, mdg.subdomains(), eq_per_gridEntity)

    expr_p2 = p2 * Scalar(1.0) + s2 * Scalar(2.0) - Scalar(2.0)
    expr_p2.set_name("eq_p2")
    equation_system.set_equation(expr_p2, mdg.subdomains(), eq_per_gridEntity)

    expr_s1 = p1 * Scalar(3.0) + s1 * Scalar(1.0) - Scalar(3.0)
    expr_s1.set_name("eq_s1")
    equation_system.set_equation(expr_s1, mdg.subdomains(), eq_per_gridEntity)

    expr_s2 = p2 * Scalar(3.0) + s2 * Scalar(1.0) - Scalar(4.0)
    expr_s2.set_name("eq_s2")
    equation_system.set_equation(expr_s2, mdg.subdomains(), eq_per_gridEntity)

    # On each interface, register 3 equations.
    eq_pf = pf ** Scalar(2.0) - Scalar(2.0)
    eq_pf.set_name("eq_p_f")
    equation_system.set_equation(eq_pf, mdg.interfaces(), eq_per_gridEntity)

    eq_sf1 = pf + sf2 + sf1 * Scalar(2.5) - Scalar(1.0)
    eq_sf1.set_name("eq_s_f_1")
    equation_system.set_equation(eq_sf1, mdg.interfaces(), eq_per_gridEntity)

    eq_sf2 = pf + sf2 * sf1 + sf1 - Scalar(10.0)
    eq_sf2.set_name("eq_s_f_2")
    equation_system.set_equation(eq_sf2, mdg.interfaces(), eq_per_gridEntity)

    # Define "secondary" list of equations & variables.
    secondaryEqList = ["eq_s_f_1", "eq_s_f_2"]
    secondaryVarList = ["sf1", "sf2"]

    # Extract the secondary block matrix.
    A_ss, _ = equation_system.assemble(
        equations=secondaryEqList,
        variables=secondaryVarList,
        state=np.zeros(equation_system.num_dofs()),
    )

    # Invert the non-diagonal block matrix A_ss.
    row_perm, col_perm, block_sizes = (
        matrix_operations.generate_permutation_to_block_diag_matrix(A_ss)
    )
    inv_A_ss = matrix_operations.invert_permuted_block_diag_matrix(
        A_ss, row_perm, col_perm, block_sizes
    )

    # Verify that A_ss * inv_A_ss is the identity matrix.
    approx_identity = A_ss.dot(inv_A_ss).toarray()
    assert np.allclose(approx_identity, np.eye(A_ss.shape[0]))


def test_diagonal_scaling_matrix():
    # Generate a matrix with a known row sum, check that the target function returns the
    # correct diagonal.
    A = np.array([[1, 2, 3], [0, -5, 6], [-7, 8, 0]])
    A_sum = np.array([6, 11, 15])
    values = 1 / A_sum

    D = pp.matrix_operations.diagonal_scaling_matrix(sps.csr_matrix(A))
    assert compare_arrays(values, D.diagonal())
