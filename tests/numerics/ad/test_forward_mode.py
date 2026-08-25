"""Collection of unit tests for the automatic differentiation forward mode. For the
class AdArray, tests are being conducted on the public attributes self.val, self.jac,
and self.copy. The tests also cover the initialization of AdArray (joint initiation of
multiple dependent variables) and the arithmetic operations implemented in AdArray,
e.g., add, sub, etc., which are also covered in other tests.

"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from porepy.applications.test_utils.arrays import compare_arrays, compare_matrices
from porepy.numerics.ad import functions as af
from porepy.numerics.ad.forward_mode import (
    AdArray,
    DiagonalAdArray,
    init_partial_ad_array,
    initialize_diagonal_ad_arrays,
)


def initAdArrays(variables: list[np.ndarray]) -> list[AdArray]:
    """Initialize a set of AdArrays, jointly dependent on each other.

    Test helper: creates one AdArray per entry in ``variables``, with the
    gradients taken with respect to all variables jointly (i.e. each returned
    AdArray has a unit derivative with respect to itself and a zero derivative
    with respect to the other variables).

    """
    num_values_per_variable = [v.size for v in variables]
    ad_arrays: list[AdArray] = []

    for i, val in enumerate(variables):
        # initiate zero jacobian
        n = num_values_per_variable[i]
        jac = [sps.csc_matrix((n, m)) for m in num_values_per_variable]
        # Set jacobian of variable i to I
        jac[i] = sps.diags(np.ones(num_values_per_variable[i])).tocsr()
        # initiate AdArray
        jac = sps.bmat([jac])
        ad_arrays.append(AdArray(val, jac))

    return ad_arrays


@pytest.fixture(params=[sps.csc_matrix, sps.csc_array])
def create_csc(request) -> type[sps.csc_matrix] | type[sps.csc_array]:
    return request.param


@pytest.fixture(params=[sps.csr_matrix, sps.csr_array])
def create_csr(request) -> type[sps.csr_matrix] | type[sps.csr_array]:
    return request.param


@pytest.fixture(params=[sps.diags, sps.diags_array])
def create_diags(request):
    return request.param


def test_quadratic_function():
    x, y = initAdArrays([np.array([1]), np.array([2])])
    z = 1 * x + 2 * y + 3 * x * y + 4 * x * x + 5 * y * y
    val = 35
    assert z.val == val and np.all(z.jac.toarray() == [15, 25])


def test_vector_quadratic():
    x, y = initAdArrays([np.array([1, 1]), np.array([2, 3])])
    z = 1 * x + 2 * y + 3 * x * y + 4 * x * x + 5 * y * y
    val = np.array([35, 65])
    J = np.array([[15, 0, 25, 0], [0, 18, 0, 35]])

    assert np.all(z.val == val) and np.sum(z.jac != J) == 0


def test_mapping_m_to_n(create_csc: type[sps.csc_matrix] | type[sps.csc_array]):
    x, y = initAdArrays([np.array([1, 1, 3]), np.array([2, 3])])
    A = create_csc(np.array([[1, 2, 1], [2, 3, 4]]))

    z = y * (A @ x)
    val = np.array([12, 51])
    J = np.array([[2, 4, 2, 6, 0], [6, 9, 12, 0, 17]])

    assert np.all(z.val == val) and np.sum(z.jac != J) == 0


def test_add_two_ad_variables_init():
    a, b = initAdArrays([np.array([1]), np.array([-10])])
    c = a + b
    assert c.val == -9 and np.all(c.jac.toarray() == [1, 1])
    assert a.val == 1 and np.all(a.jac.toarray() == [1, 0])
    assert b.val == -10 and np.all(b.jac.toarray() == [0, 1])


def test_sub_var_init_with_var_init():
    a, b = initAdArrays([np.array([3]), np.array([2])])
    c = b - a
    assert np.allclose(c.val, -1) and np.all(c.jac.toarray() == [-1, 1])
    assert a.val == 3 and np.all(a.jac.toarray() == [1, 0])
    assert b.val == 2 and np.all(b.jac.toarray() == [0, 1])


def test_mul_ad_var_init():
    a, b = initAdArrays([np.array([3]), np.array([2])])
    c = a * b
    assert a.val == 3 and np.all(a.jac.toarray() == [1, 0])
    assert b.val == 2 and np.all(b.jac.toarray() == [0, 1])
    assert c.val == 6 and np.all(c.jac.toarray() == [2, 3])


def test_mul_scal_ad_var_init():
    a, b = initAdArrays([np.array([3]), np.array([2])])
    d = 3.0
    c = d * a
    assert c.val == 9 and np.all(c.jac.toarray() == [3, 0])
    assert a.val == 3 and np.all(a.jac.toarray() == [1, 0])
    assert b.val == 2 and np.all(b.jac.toarray() == [0, 1])


def test_mul_sps_advar_init(create_csc: type[sps.csc_matrix] | type[sps.csc_array]):
    x = initAdArrays([np.array([1, 2, 3])])[0]
    A = create_csc(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    f = A @ x
    assert np.all(f.val == [14, 32, 50])
    assert np.all((f.jac == A).toarray())


def test_advar_init_diff_len(create_csc: type[sps.csc_matrix] | type[sps.csc_array]):
    a, b = initAdArrays([np.array([1, 2, 3]), np.array([1, 2])])
    A = create_csc(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    B = create_csc(np.array([[1, 2], [4, 5]]))

    f = A @ a
    g = B @ b
    zero_32 = create_csc((3, 2))
    zero_23 = create_csc((2, 3))

    jac_f = sps.hstack((A, zero_32))
    jac_g = sps.hstack((zero_23, B))
    assert np.all(f.val == [14, 32, 50])
    assert np.all((f.jac == jac_f).toarray())
    assert np.all(g.val == [5, 14])
    assert np.all((g.jac == jac_g).toarray())


def test_advar_init_cross_jacobi():
    x, y = initAdArrays([np.array([-1, 4]), np.array([1, 5])])

    z = x * y
    J = np.array([[1, 0, -1, 0], [0, 5, 0, 4]])
    assert np.all(z.val == [-1, 20])
    assert np.all((z.jac.toarray() == J))


def test_advar_mul_vec(create_diags):
    x = AdArray(np.array([1, 2, 3]), create_diags([3, 2, 1]))
    A = np.array([1, 3, 10])
    f = x * A
    sol = np.array([1, 6, 30])
    jac = np.diag([3, 6, 10])

    assert np.all(f.val == sol) and np.all(f.jac == jac)
    assert np.all(x.val == np.array([1, 2, 3])) and np.all(x.jac == np.diag([3, 2, 1]))


def test_advar_m_mul_vec_n(create_csc, create_diags):
    x = AdArray(np.array([1, 2, 3]), create_diags([3, 2, 1]))
    vec = np.array([1, 2])
    R = create_csc(np.array([[1, 0, 1], [0, 1, 0]]))
    y = R @ x
    z = y * vec
    Jy = np.array([[3, 0, 1], [0, 2, 0]])
    Jz = np.array([[1, 0, 3], [0, 4, 0]])
    assert np.all(y.val == [4, 2])
    assert np.sum(y.jac.toarray() - Jy) == 0
    assert np.all(z.val == [4, 4])
    assert np.sum(z.jac.toarray() - Jz) == 0


def test_mul_sps_advar(create_csc: type[sps.csc_matrix] | type[sps.csc_array]):
    J = create_csc(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
    x = AdArray(np.array([1, 2, 3]), J)
    A = create_csc(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    f = A @ x

    assert np.all(f.val == [14, 32, 50])
    assert np.all(f.jac == A @ J.toarray())


def test_mul_advar_vectors(create_csc: type[sps.csc_matrix] | type[sps.csc_array]):
    Ja = create_csc(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
    Jb = create_csc(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    a = AdArray(np.array([1, 2, 3]), Ja)
    b = AdArray(np.array([1, 1, 1]), Jb)
    A = create_csc(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    f = A @ a + b

    assert np.all(f.val == [15, 33, 51])
    assert compare_matrices(f.jac, A @ Ja + Jb)
    # Asterix stands for element-wise multiplication for sparrays and matrix
    # multiplication for spmatrix.
    if create_csc is sps.csc_matrix:
        assert compare_matrices(f.jac, A * Ja + Jb)
    elif create_csc is sps.csc_array:
        assert not compare_matrices(f.jac, A * Ja + Jb)
    else:
        raise ValueError(create_csc)
    assert compare_matrices(Ja, create_csc(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]])))
    assert compare_matrices(Jb, create_csc(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))


def test_copy_scalar(create_csr: type[sps.csr_matrix] | type[sps.csr_array]):
    a = AdArray(np.array([1]), create_csr([[0]]))
    b = a.copy()
    assert a.val == b.val
    assert a.jac == b.jac
    a.val = 2
    a.jac = 3
    assert b.val == 1
    assert b.jac == 0


def test_copy_vector(create_csr: type[sps.csr_matrix] | type[sps.csr_array]):
    a = AdArray(np.ones(3), create_csr(np.diag(np.ones((3)))))
    b = a.copy()
    assert np.allclose(a.val, b.val)
    assert np.allclose(a.jac.toarray(), b.jac.toarray())
    a.val[0] = 3
    a.jac[2] = 4
    assert np.allclose(b.val, np.ones(3))
    assert np.allclose(b.jac.toarray(), create_csr(np.diag(np.ones((3)))).toarray())


def test_exp_scalar_times_ad_var(create_csc, create_diags):
    val = np.array([1, 2, 3])
    J = create_diags(np.array([1, 1, 1]))
    a, _, _ = initAdArrays([val, val, val])
    c = 2.0
    b = af.exp(c * a)

    zero = create_csc((3, 3))
    jac = sps.hstack([c * create_diags(np.exp(c * val)) * J, zero, zero])
    jac_a = sps.hstack([J, zero, zero])
    assert np.allclose(b.val, np.exp(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.toarray() == jac_a.toarray())


@pytest.mark.parametrize(
    "index,index_c",
    [  # indices and their complement for tested array
        (np.array([1]), [0, 2, 3, 4, 5, 6, 7, 8, 9]),
        (slice(0, 10, 2), slice(1, 10, 2)),
        (np.array([0, 2, 4, 6, 8], dtype=int), np.array([1, 3, 5, 7, 9], dtype=int)),
    ],
)
def test_get_set_slice_ad_var(
    index, index_c, create_csr: type[sps.csr_matrix] | type[sps.csr_array]
):
    a = initAdArrays([np.arange(10)])[0]

    val = np.arange(10)
    jac = create_csr(np.eye(10))

    assert np.all(val == a.val)
    assert np.all(jac == a.jac.toarray())

    if isinstance(index, int):
        target_val = np.array([val[index]])
    else:
        target_val = val[index]
    target_jac = jac[index]

    # Testing slicing
    a_slice = a[index]

    # `initAdArrays` (the test helper above) builds its Jacobians with spmatrices,
    # not sparrays. Their slicing behavior is different: spmatrix does not flatten
    # the result, while sparray does. Some parts of the code may rely on this
    # assumption. This code will signalize if this assumption ever breaks.
    if isinstance(index, int):
        assert len(a_slice.jac.shape) == 2
        # Manually unraveling it for the sparrays to make the test consistent.
        target_jac = target_jac.reshape(1, 10)

    assert compare_arrays(a_slice.val, target_val)
    assert compare_matrices(a_slice.jac, target_jac)

    # testing setting values with slicing

    b = a[index] * 10.0
    assert compare_arrays(b.val, val[index] * 10.0)
    # Edge case, it is known that the test will fail, since sparrays flatten the slice.
    # This condition signalize if the behavior in `initAdArrays` is changed.
    if isinstance(jac, sps.csr_array) and isinstance(index, int):
        with pytest.raises(AssertionError):
            assert compare_matrices(b.jac, jac[index] * 10.0)
    else:
        assert compare_matrices(b.jac, jac[index] * 10.0)

    # setting an AD array should set val and jacobian row-wise
    a_copy = a.copy()
    a[index] = b
    assert compare_arrays(a[index].val, b.val)
    assert compare_matrices(a[index].jac, b.jac)
    # complement should not be affected
    assert compare_arrays(a[index_c].val, a_copy[index_c].val)
    assert compare_matrices(a[index_c].jac, a_copy[index_c].jac)

    # setting a numpy array should only modify the values of the ad array
    b = target_val * 10.0
    a = a_copy.copy()
    a[index] = b
    assert compare_arrays(a[index].val, b)
    assert compare_arrays(a[index_c].val, a_copy[index_c].val)
    assert compare_matrices(a.jac, a_copy.jac)


@pytest.mark.parametrize("N", [1, 3])
@pytest.mark.parametrize("logical_op", [">", ">=", "<", "<=", "==", "!="])
@pytest.mark.parametrize(
    "other",
    [
        1,
        np.ones(1),
        np.ones(2),
        np.ones(3),
        np.ones((2, 2)),
        AdArray(np.ones(1), sps.csr_matrix(np.eye(1))),
        AdArray(np.ones(2), sps.csr_matrix(np.eye(2))),
        AdArray(np.ones(3), sps.csr_matrix(np.eye(3))),
        AdArray(np.ones(1), sps.csr_array(np.eye(1))),
        AdArray(np.ones(2), sps.csr_array(np.eye(2))),
        AdArray(np.ones(3), sps.csr_array(np.eye(3))),
    ],
)
def test_logical_operation(
    N: int, logical_op: str, other: int | np.ndarray | AdArray, create_csr
):
    """Logical operations on Ad arrays are implemented such that they operate only on
    values, making them completely equivalent to what numpy does. This test is based
    onthat premise: Logical operations on AdArrays should yield results identical to
    operations on numpy arrays.

    Test that they work and that the result of the logical operation is doing the same
    as numpy for ``.val`` only.

    """

    val = np.arange(N)
    jac = create_csr(np.eye(N))
    # Ignore ad not being accessed, it is used in the exec statement.
    ad = AdArray(val, jac)  # noqa: F841

    global result_numpy, result_ad
    result_numpy = np.empty(N)
    result_ad = np.empty(N)

    # NOTE Numpy manages to compare arrays, if one of them has shape (1,) by treating
    # it as a scalar. All other cases should raise an error.
    try:
        # NOTE if the AD array is the right operand, the overload of numpy will be
        # invoked. Must use the .val member in this case
        if isinstance(other, AdArray):
            exec(f"global result_numpy; result_numpy = val {logical_op} other.val")
        else:
            exec(f"global result_numpy; result_numpy = val {logical_op} other")
    # If numpy failes to broadcast the shapes, so should the Ad Array.
    except ValueError as numpy_err:
        with pytest.raises(ValueError) as ad_error:
            exec(f"result_ad = ad {logical_op} other")

        # Comparison of exceptions by type and message content
        assert ad_error.type is type(numpy_err)
        assert str(ad_error.value) == str(numpy_err)
    # If numpy does not fail, the logical operation on AD should have same result,
    # dtype and shape
    else:
        exec(f"global result_ad; result_ad = ad {logical_op} other")

        assert result_ad.shape == result_numpy.shape
        assert result_ad.dtype == result_numpy.dtype
        assert np.all(result_ad == result_numpy)


@pytest.mark.parametrize("operator", ["+", "-", "*", "/", "**"])
def test_numpy_array_as_left_operand(operator: str, create_csr):
    """A numpy array to the left of an AdArray should give an AdArray.

    This tests the disabling of numpy's ufuncs for AdArray, by setting
    AdArray.__array_ufunc__ = None. By this trick, numpy broadcasts instead of deferring
    to __radd__ etc., and returns an object array holding one full AdArray -- values and
    Jacobian -- per element.

    """
    val = np.array([2.0, 3.0, 4.0])
    jac = create_csr(np.diag([5.0, 6.0, 7.0]))
    ad = AdArray(val, jac)
    b = np.array([3.0, 2.0, 5.0])

    res = eval(f"b {operator} ad")

    # d/dx of (b op x), differentiated by hand.
    derivative = {
        "+": np.ones_like(val),
        "-": -np.ones_like(val),
        "*": b,
        "/": -b / val**2,
        "**": b**val * np.log(b),
    }[operator]

    assert isinstance(res, AdArray)
    assert np.allclose(res.val, eval(f"b {operator} val"))
    assert np.allclose(res.jac.toarray(), np.diag(derivative) @ jac.toarray())
    # One Jacobian, not one per element.
    assert res.jac.nnz == jac.nnz


@pytest.mark.parametrize("logical_op", [">", ">=", "<", "<=", "==", "!="])
def test_numpy_array_as_left_operand_logical(logical_op: str, create_csr):
    """Test disabling of numpy's ufuncs for AdArray when used with logical operators.

    See test test_numpy_array_as_left_operand for details.
    """
    val = np.array([2.0, 3.0, 4.0])
    ad = AdArray(val, create_csr(np.eye(3)))
    b = np.array([3.0, 3.0, 1.0])

    res = eval(f"b {logical_op} ad")

    assert isinstance(res, np.ndarray) and res.dtype == np.bool_
    assert np.all(res == eval(f"b {logical_op} val"))


@pytest.mark.parametrize(
    "state, indices, expected_jac",
    [
        (
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1, 3]),
            np.diag([0.0, 1.0, 0.0, 1.0]),
        ),
        (np.array([1.0, 2.0, 3.0]), np.array([], dtype=int), np.zeros((3, 3))),
    ],
)
def test_init_partial_ad_array(state, indices, expected_jac):
    """The returned AdArray has a unit derivative at the given indices, and a zero
    derivative everywhere else."""
    var = init_partial_ad_array(state, indices)

    assert isinstance(var, AdArray)
    assert np.allclose(var.val, state)
    assert np.allclose(var.jac.toarray(), expected_jac)


def test_initialize_diagonal_ad_arrays_single_variable():
    """With a single variable, the returned array has a unit derivative of each
    entry with respect to itself, placed at the given global indices."""
    val = np.array([2.0, 3.0, 4.0])
    global_indices = [np.array([1, 3, 5])]
    num_derivatives = 6

    diag_vars = initialize_diagonal_ad_arrays([val], global_indices, num_derivatives)

    assert len(diag_vars) == 1
    var = diag_vars[0]
    assert isinstance(var, DiagonalAdArray)
    assert var.is_diagonal
    assert np.allclose(var.val, val)

    full_jac = var.to_full().jac.toarray()
    expected_jac = np.zeros((3, num_derivatives))
    expected_jac[np.arange(3), global_indices[0]] = 1.0
    assert np.allclose(full_jac, expected_jac)


def test_initialize_diagonal_ad_arrays_two_variables():
    """Each returned DiagonalAdArray depends only on itself (unit derivative), not on
    the other variable passed in the same call."""
    val_0 = np.array([1.0, 2.0])
    val_1 = np.array([3.0, 4.0])
    indices = [np.array([0, 1]), np.array([2, 3])]
    num_derivatives = 4

    diag_vars = initialize_diagonal_ad_arrays([val_0, val_1], indices, num_derivatives)

    assert len(diag_vars) == 2
    assert np.allclose(diag_vars[0].val, val_0)
    assert np.allclose(diag_vars[1].val, val_1)
    for var in diag_vars:
        assert isinstance(var, DiagonalAdArray)
        assert var.is_diagonal

    # Each array's raw (diagonal-representation) Jacobian has a unit derivative in
    # its own row, and zero in the row belonging to the other variable.
    assert np.allclose(diag_vars[0].jac, np.array([[1.0, 1.0], [0.0, 0.0]]))
    assert np.allclose(diag_vars[1].jac, np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_initialize_diagonal_ad_arrays_custom_derivatives():
    val = np.array([2.0, 4.0])
    indices = [np.array([0, 1])]
    num_derivatives = 2
    derivatives = [np.array([5.0, 6.0])]

    diag_vars = initialize_diagonal_ad_arrays(
        [val], indices, num_derivatives, derivatives=derivatives
    )

    full_jac = diag_vars[0].to_full().jac.toarray()
    expected_jac = np.diag([5.0, 6.0])
    assert np.allclose(full_jac, expected_jac)


@pytest.mark.parametrize(
    "vals, indices, num_derivatives",
    [
        ([np.array([1.0])], [np.array([0]), np.array([1])], 2),
        ([np.array([1.0, 2.0])], [np.array([0])], 2),
    ],
)
def test_initialize_diagonal_ad_arrays_mismatched(vals, indices, num_derivatives):
    with pytest.raises(ValueError):
        initialize_diagonal_ad_arrays(vals, indices, num_derivatives)


def test_diagonal_ad_array_coerces_int_val_and_jac_to_float():
    """DiagonalAdArray should enforce the same float-dtype invariant as AdArray."""
    val = np.array([1, 2, 3])
    jac = np.array([4, 5, 6])

    var = DiagonalAdArray(
        val,
        jac,
        row_indices=np.arange(3),
        col_indices=[np.arange(3)],
        num_derivatives=3,
    )

    assert var.val.dtype == float
    assert var.jac.dtype == float


def test_diagonal_ad_array_replace():
    """replace() should build a new DiagonalAdArray with new value/Jacobian data,
    while reusing (not copying) the original's structural indices."""
    row_indices = np.array([2, 5])
    col_indices = [np.array([2, 5])]
    var = DiagonalAdArray(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        row_indices=row_indices,
        col_indices=col_indices,
        num_derivatives=6,
    )

    new_val = np.array([10.0, 20.0])
    new_jac = np.array([30.0, 40.0])
    new_var = var.replace(new_val, new_jac)

    assert isinstance(new_var, DiagonalAdArray)
    assert np.allclose(new_var.val, new_val)
    assert np.allclose(new_var.jac, new_jac)
    assert new_var.row_indices is row_indices
    assert new_var.col_indices is col_indices
    assert new_var.num_derivatives == 6
