"""Test suite for operations with Ad_arrays.

NOTE: For combinations of numpy arrays and Ad_arrays, we do not test the reverse order
operations for addition, subtraction, multiplication, and division, since this will not
work properly with numpy arrays, see https://stackoverflow.com/a/58120561 for more
information. To circumwent this problem, parsing of Ad expressions ensures that numpy
arrays are always left added (and subtracted, multiplied) with Ad_arrays, but this
should be covered in tests to be written.

"""

import pytest
import scipy.sparse as sps
import numpy as np

from porepy.numerics.ad.forward_mode import Ad_array, initAdArrays
from porepy.numerics.ad import functions as af


def test_add():
    # Create an Ad_array, add it to the various Ad types (float, numpy arrays, scipy
    # matrices and other Ad_arrays). The tests verifies that the results are as
    # expected, or that an error is raised for operations that are not supported.

    val = np.array([1, 2, 3])
    jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    ad_arr = Ad_array(val, jac)

    # Test that a float can be added to an Ad_array
    ad_add_float = ad_arr + 1.0
    assert np.allclose(ad_add_float.val, np.array([2, 3, 4]))
    assert np.allclose(
        ad_add_float.jac.toarray(), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )
    # Also test reversed order
    float_add_ad = 1.0 + ad_arr
    assert np.allclose(float_add_ad.val, np.array([2, 3, 4]))
    assert np.allclose(
        float_add_ad.jac.toarray(), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )

    # Test that a 1d numpy array with the same size as the Ad_array can be added
    numpy_array = np.arange(ad_arr.val.size)
    ad_add_numpy = ad_arr + numpy_array
    assert np.allclose(ad_add_numpy.val, np.array([1, 3, 5]))
    assert np.allclose(
        ad_add_numpy.jac.toarray(), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )
    # See module-level comment regarding reverse mode.

    # Test that a 1d numpy array with a different size than the Ad_array raises an error
    other = np.arange(ad_arr.val.size + 1)
    with pytest.raises(ValueError):
        ad_arr + other

    # Test that a 2d numpy array raises an error
    other = np.arange(ad_arr.val.size).reshape((ad_arr.val.size, 1))
    with pytest.raises(ValueError):
        ad_arr + other

    # Test that a sparse matrix raises an error
    other = sps.csr_matrix(ad_arr.jac)
    with pytest.raises(ValueError):
        ad_arr + other
    # We should get an error also in reverse mode
    with pytest.raises(ValueError):
        other + ad_arr

    # Test that an Ad_array with the same size as the Ad_array can be added
    ad_add_ad = ad_arr + ad_arr
    assert np.allclose(ad_add_ad.val, np.array([2, 4, 6]))
    assert np.allclose(
        ad_add_ad.jac.toarray(), np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])
    )

    # Test that adding an Ad_array of different size than the Ad_array raises an error
    sz = ad_arr.val.size + 1
    other = Ad_array(np.arange(sz), sps.csr_matrix((np.zeros((sz, sz)))))
    with pytest.raises(ValueError):
        ad_arr + other

    # No need to test reversed order, since the other operand is an Ad_array


def test_subtract():
    # Create an Ad_array, subtract it from the various Ad types (float, numpy arrays,
    # scipy matrices and other Ad_arrays). The tests verifies that the results are as
    # expected, or that an error is raised for operations that are not supported.

    val = np.array([1, 2, 3])
    jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    ad_arr = Ad_array(val, jac)

    # Test that a float can be added to an Ad_array
    ad_subtract_float = ad_arr - 1.0
    assert np.allclose(ad_subtract_float.val, np.array([0, 1, 2]))
    assert np.allclose(
        ad_subtract_float.jac.toarray(), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )
    # Also test reversed order
    float_subtract_ad = 1.0 - ad_arr
    assert np.allclose(float_subtract_ad.val, np.array([0, -1, -2]))
    assert np.allclose(
        float_subtract_ad.jac.toarray(), -np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )

    # Test that a 1d numpy array with the same size as the Ad_array can be subtracted
    numpy_array = np.arange(ad_arr.val.size)
    ad_subtract_numpy = ad_arr - numpy_array
    assert np.allclose(ad_subtract_numpy.val, np.array([1, 1, 1]))
    assert np.allclose(
        ad_subtract_numpy.jac.toarray(), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )
    # See module-level comment regarding reverse mode.

    # Test that a 1d numpy array with a different size than the Ad_array raises an error
    other = np.arange(ad_arr.val.size + 1)
    with pytest.raises(ValueError):
        ad_arr - other

    # Test that a 2d numpy array raises an error
    other = np.arange(ad_arr.val.size).reshape((ad_arr.val.size, 1))
    with pytest.raises(ValueError):
        ad_arr - other

    # Test that a sparse matrix raises an error
    other = sps.csr_matrix(ad_arr.jac)
    with pytest.raises(ValueError):
        ad_arr - other
    # We should get an error also in reverse mode
    with pytest.raises(ValueError):
        other - ad_arr

    # Test that an Ad_array with the same size as the Ad_array can be subtracted
    ad_subtract_ad = ad_arr - ad_arr
    assert np.allclose(ad_subtract_ad.val, np.array([0, 0, 0]))
    assert np.allclose(ad_subtract_ad.jac.toarray(), np.zeros((3, 3)))

    # Test that subtracting an Ad_array from a different size than the Ad_array raises
    # an error
    sz = ad_arr.val.size + 1
    other = Ad_array(np.arange(sz), sps.csr_matrix((np.zeros((sz, sz)))))
    with pytest.raises(ValueError):
        ad_arr - other

    # No need to test reversed order, since the other operand is an Ad_array


def test_mul():
    # Create an Ad_array, multiply it from the various Ad types (float, numpy arrays,
    # scipy matrices and other Ad_arrays). The tests verifies that the results are as
    # expected, or that an error is raised for operations that are not supported.

    val = np.array([1, 2, 3])
    jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    ad_arr = Ad_array(val, jac)

    # Test that a float can be added to an Ad_array
    ad_mul_float = ad_arr * 2.0
    assert np.allclose(ad_mul_float.val, np.array([2, 4, 6]))
    assert np.allclose(
        ad_mul_float.jac.toarray(), 2 * np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )
    # Also test reversed order
    float_mul_ad = 2.0 * ad_arr
    assert np.allclose(float_mul_ad.val, np.array([2, 4, 6]))
    assert np.allclose(
        float_mul_ad.jac.toarray(), 2 * np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )

    # Test that a 1d numpy array with the same size as the Ad_array can be multiplied
    numpy_array = np.arange(ad_arr.val.size)
    ad_mul_numpy = ad_arr * numpy_array
    assert np.allclose(ad_mul_numpy.val, np.array([0, 2, 6]))
    assert np.allclose(
        ad_mul_numpy.jac.toarray(), np.array([[0, 0, 0], [4, 5, 6], [14, 16, 18]])
    )
    # See module-level comment regarding reverse mode.

    # Test that a 1d numpy array with a different size than the Ad_array raises an error
    other = np.arange(ad_arr.val.size + 1)
    with pytest.raises(ValueError):
        ad_arr * other

    # Test that a 2d numpy array raises an error
    other = np.arange(ad_arr.val.size).reshape((ad_arr.val.size, 1))
    with pytest.raises(ValueError):
        ad_arr * other

    # Test that a sparse matrix raises an error
    other = sps.csr_matrix(ad_arr.jac)
    with pytest.raises(ValueError):
        ad_arr * other
    # We should get an error also in reverse mode
    with pytest.raises(ValueError):
        other * ad_arr

    # Test that an Ad_array with the same size as the Ad_array can be multiplied
    ad_mul_ad = ad_arr * ad_arr
    # Hardcoded values of val * val
    assert np.allclose(ad_mul_ad.val, np.array([1, 4, 9]))

    # The derivative of x**2 = 2 * x * x'
    known_jac = 2 * np.vstack((val[0] * jac[0].A, val[1] * jac[1].A, val[2] * jac[2].A))
    assert np.allclose(ad_mul_ad.jac.toarray(), known_jac)

    # Test that multiplying an Ad_array with a different size than the Ad_array raises
    # an error
    sz = ad_arr.val.size + 1
    other = Ad_array(np.arange(sz), sps.csr_matrix((np.zeros((sz, sz)))))
    with pytest.raises(ValueError):
        ad_arr * other


def test_div():
    # Create an Ad_array, take its power with the various Ad types (float, numpy arrays,
    # scipy matrices and other Ad_arrays). The tests verifies that the results are as
    # expected, or that an error is raised for operations that are not supported.

    val = np.array([1, 2, 3])
    jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    ad_arr = Ad_array(val, jac)

    # Test that a float can be added to an Ad_array
    ad_div_float = ad_arr / 2.0
    # Hardcoded values for val**2.0
    assert np.allclose(ad_div_float.val, np.array([0.5, 1, 1.5]))
    # Hardcoded values for 2 * diag(val) @ jac
    assert np.allclose(ad_div_float.jac.toarray(), jac.A / 2.0)

    # Also test reversed order
    float_div_ad = 2.0 / ad_arr
    # Hardcoded values for 2.0**val
    assert np.allclose(float_div_ad.val, np.array([2, 2 / 2, 2 / 3]))
    # The derivative of 2.0 / val is -2.0 / val**2 * val'
    assert np.allclose(
        float_div_ad.jac.toarray(),
        np.vstack(
            (
                -2 / val[0] ** 2 * jac[0].A,
                -2 / val[1] ** 2 * jac[1].A,
                -2 / val[2] ** 2 * jac[2].A,
            )
        ),
    )

    # Test that an Ad_array can be divided by a 1d numpy array with the same size.
    numpy_array = np.arange(ad_arr.val.size) + 1
    ad_div_numpy = ad_arr / numpy_array
    # Hardcoded values of val**numpy_array
    assert np.allclose(ad_div_numpy.val, np.array([1, 1, 1]))

    # Hardcoded values of the derivative
    known_jac = np.vstack(
        (
            jac[0].A / numpy_array[0],
            jac[1].A / numpy_array[1],
            jac[2].A / numpy_array[2],
        )
    )

    assert np.allclose(ad_div_numpy.jac.toarray(), known_jac)

    # See module-level comment regarding reverse mode.

    # Test that a 1d numpy array with a different size than the Ad_array raises an error
    other = np.arange(ad_arr.val.size + 1)
    with pytest.raises(ValueError):
        ad_arr / other

    # Test that a 2d numpy array raises an error
    other = np.arange(ad_arr.val.size).reshape((ad_arr.val.size, 1))
    with pytest.raises(ValueError):
        ad_arr / other

    # Test that a sparse matrix raises an error
    other = sps.csr_matrix(ad_arr.jac)
    with pytest.raises(ValueError):
        ad_arr / other
    # We should get an error also in reverse mode
    with pytest.raises(ValueError):
        other / ad_arr

    # Test that an Ad_array can be raised to the power of an Ad_array with the same
    # size.
    ad_arr_2 = ad_arr + ad_arr**2.0
    # Make sure that the values of ad_arr_2 are correct before we start testing the
    # derivative.
    assert np.allclose(ad_arr_2.val, np.array([2, 6, 12]))
    ad_div_ad = ad_arr / ad_arr_2
    assert np.allclose(ad_div_ad.val, np.array([1 / 2, 2 / 6, 3 / 12]))
    # The derivative of arr / arr_2 is (arr_2 * arr' - arr * arr_2') / arr_2**2
    known_jac = np.vstack(
        (
            (ad_arr_2.val[0] * ad_arr.jac[0].A - ad_arr.val[0] * ad_arr_2.jac[0].A)
            / ad_arr_2.val[0] ** 2,
            (ad_arr_2.val[1] * ad_arr.jac[1].A - ad_arr.val[1] * ad_arr_2.jac[1].A)
            / ad_arr_2.val[1] ** 2,
            (ad_arr_2.val[2] * ad_arr.jac[2].A - ad_arr.val[2] * ad_arr_2.jac[2].A)
            / ad_arr_2.val[2] ** 2,
        )
    )
    assert np.allclose(ad_div_ad.jac.toarray(), known_jac)

    # Test that multiplying an Ad_array with a different size than the Ad_array raises
    # an error
    sz = ad_arr.val.size + 1
    other = Ad_array(np.arange(sz), sps.csr_matrix((np.zeros((sz, sz)))))
    with pytest.raises(ValueError):
        ad_arr / other

    # No need to test reversed order, since the other operand is an Ad_array


def test_pow():
    # Create an Ad_array, take its power with the various Ad types (float, numpy arrays,
    # scipy matrices and other Ad_arrays). The tests verifies that the results are as
    # expected, or that an error is raised for operations that are not supported.

    val = np.array([1, 2, 3])
    jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    ad_arr = Ad_array(val, jac)

    # Test that a float can be added to an Ad_array
    ad_pow_float = ad_arr**2.0
    # Hardcoded values for val**2.0
    assert np.allclose(ad_pow_float.val, np.array([1, 4, 9]))
    # Hardcoded values for 2 * diag(val) @ jac
    assert np.allclose(
        ad_pow_float.jac.toarray(),
        2 * np.vstack((val[0] * jac[0].A, val[1] * jac[1].A, val[2] * jac[2].A)),
    )
    # Also test reversed order
    float_pow_ad = 2.0**ad_arr
    # Hardcoded values for 2.0**val
    assert np.allclose(float_pow_ad.val, np.array([2, 4, 8]))
    # The derivative of 2.0**val is 2.0**val * log(2.0), then we need to multiply by the
    # Jacobian following the chain rule.
    assert np.allclose(
        float_pow_ad.jac.toarray(),
        np.vstack(
            (
                np.log(2.0) * (2 ** val[0]) * jac[0].A,
                np.log(2.0) * (2 ** val[1]) * jac[1].A,
                np.log(2.0) * (2 ** val[2]) * jac[2].A,
            )
        ),
    )

    # Test that a 1d numpy array with the same size as the Ad_array can be multiplied
    numpy_array = np.arange(ad_arr.val.size)
    ad_pow_numpy = ad_arr**numpy_array
    # Hardcoded values of val**numpy_array
    assert np.allclose(ad_pow_numpy.val, np.array([1, 2, 9]))

    # The derivative of x^numpy_array is numpy_array * x**(numpy_array - 1) * x'
    known_jac = np.vstack(
        (
            numpy_array[0] * (val[0] ** (numpy_array[0] - 1.0)) * jac[0].A,
            numpy_array[1] * (val[1] ** (numpy_array[1] - 1.0)) * jac[1].A,
            numpy_array[2] * (val[2] ** (numpy_array[2] - 1.0)) * jac[2].A,
        )
    )

    assert np.allclose(ad_pow_numpy.jac.toarray(), known_jac)
    # See module level docstring for a comment on the reverse order.

    # Test that a 1d numpy array with a different size than the Ad_array raises an error
    other = np.arange(ad_arr.val.size + 1)
    with pytest.raises(ValueError):
        ad_arr**other

    # Test that a 2d numpy array raises an error
    other = np.arange(ad_arr.val.size).reshape((ad_arr.val.size, 1))
    with pytest.raises(ValueError):
        ad_arr**other

    # Test that a sparse matrix raises an error
    other = sps.csr_matrix(ad_arr.jac)
    with pytest.raises(ValueError):
        ad_arr**other
    # We should get an error also in reverse mode
    with pytest.raises(ValueError):
        other**ad_arr

    # Test that an Ad_array can be raised to the power of an Ad_array with the same
    # size.
    # Make a new Ad_array with the same size as ad_arr.
    ad_arr_2 = ad_arr + ad_arr
    ad_pow_ad = ad_arr**ad_arr_2
    assert np.allclose(ad_pow_ad.val, np.array([1**1, 2**4, 3**6]))
    # The derivative of val**val is val**val * log(val), then we need to multiply by the
    # Jacobian following the chain rule.
    known_jac = np.vstack(
        (
            ad_arr_2.val[0] * ad_arr.val[0] ** (ad_arr_2.val[0] - 1.0) * ad_arr.jac[0].A
            + np.log(ad_arr.val[0])
            * (ad_arr.val[0] ** ad_arr_2.val[0])
            * ad_arr_2.jac[0].A,
            ad_arr_2.val[1] * ad_arr.val[1] ** (ad_arr_2.val[1] - 1.0) * ad_arr.jac[1].A
            + np.log(ad_arr.val[1])
            * (ad_arr.val[1] ** ad_arr_2.val[1])
            * ad_arr_2.jac[1].A,
            ad_arr_2.val[2] * ad_arr.val[2] ** (ad_arr_2.val[2] - 1.0) * ad_arr.jac[2].A
            + np.log(ad_arr.val[2])
            * (ad_arr.val[2] ** ad_arr_2.val[2])
            * ad_arr_2.jac[2].A,
        )
    )
    assert np.allclose(ad_pow_ad.jac.toarray(), known_jac)

    # Test that multiplying an Ad_array with a different size than the Ad_array raises
    # an error
    sz = ad_arr.val.size + 1
    other = Ad_array(np.arange(sz), sps.csr_matrix((np.zeros((sz, sz)))))
    with pytest.raises(ValueError):
        ad_arr**other

    # No need to test reversed order, since the other operand is an Ad_array


def test_matmul():
    # Create an Ad_array, use matmul (@) with the various Ad types (float, numpy arrays,
    # scipy matrices and other Ad_arrays). This should mostly raise errors, so verify
    # that, and also check that the correct numbers are returned when it works.

    val = np.array([1, 2, 3])
    jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    ad_arr = Ad_array(val, jac)

    # Right and left multiplication with a float should work. First test the right
    ad_matmul_float = ad_arr @ 2.0
    assert np.allclose(ad_matmul_float.val, np.array([2, 4, 6]))
    # The derivative of val * 2.0 is 2.0 * val'
    assert np.allclose(ad_matmul_float.jac.toarray(), 2.0 * jac.toarray())
    # Test the left multiplication
    ad_matmul_float = 2.0 @ ad_arr
    assert np.allclose(ad_matmul_float.val, np.array([2, 4, 6]))
    assert np.allclose(ad_matmul_float.jac.toarray(), 2.0 * jac.toarray())

    # Multiplication with a numpy array should raise an error
    other = np.arange(ad_arr.val.size)
    with pytest.raises(ValueError):
        ad_arr @ other
    # We should get an error also in reverse mode
    with pytest.raises(ValueError):
        other @ ad_arr

    # Left multiplication with a scipy matrix should work
    other = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    spmatrix_matmul_other = other @ ad_arr
    assert np.allclose(spmatrix_matmul_other.val, np.array([14, 32, 50]))
    # The derivative of other @ val is other @ val'
    known_jac = np.array([[30, 36, 42], [66, 81, 96], [102, 126, 150]])
    assert np.allclose(spmatrix_matmul_other.jac.toarray(), known_jac)

    # Right multiplication with a scipy matrix should raise an error
    with pytest.raises(ValueError):
        ad_arr @ other

    # Both left and right multiplication with an Ad_array should raise an error
    other = Ad_array(np.arange(ad_arr.val.size), sps.csr_matrix((np.zeros((3, 3)))))
    with pytest.raises(ValueError):
        ad_arr @ other
    # We should get an error also in reverse mode
    with pytest.raises(ValueError):
        other @ ad_arr


""" Below follows legacy (though updated) tests for the Ad_array class. These tests
    cover initiation of Ad_array (joint initiation of multiple dependent
variables). The test also partly cover the arithmetic operations implemented for
Ad_arrays, e.g., __add__, __sub__, etc., but these are also tested in different tests.
"""


def test_add_two_ad_variables_init():
    a, b = initAdArrays([np.array([1]), np.array([-10])])
    c = a + b
    assert c.val == -9 and np.all(c.jac.A == [1, 1])
    assert a.val == 1 and np.all(a.jac.A == [1, 0])
    assert b.val == -10 and np.all(b.jac.A == [0, 1])


def test_sub_var_init_with_var_init():
    a, b = initAdArrays([np.array([3]), np.array([2])])
    c = b - a
    assert np.allclose(c.val, -1) and np.all(c.jac.A == [-1, 1])
    assert a.val == 3 and np.all(a.jac.A == [1, 0])
    assert b.val == 2 and np.all(b.jac.A == [0, 1])


def test_mul_ad_var_init():
    a, b = initAdArrays([np.array([3]), np.array([2])])
    c = a * b
    assert a.val == 3 and np.all(a.jac.A == [1, 0])
    assert b.val == 2 and np.all(b.jac.A == [0, 1])
    assert c.val == 6 and np.all(c.jac.A == [2, 3])


def test_mul_scal_ad_var_init():
    a, b = initAdArrays([np.array([3]), np.array([2])])
    d = 3.0
    c = d * a
    assert c.val == 9 and np.all(c.jac.A == [3, 0])
    assert a.val == 3 and np.all(a.jac.A == [1, 0])
    assert b.val == 2 and np.all(b.jac.A == [0, 1])


def test_mul_sps_advar_init():
    x = initAdArrays([np.array([1, 2, 3])])[0]
    A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    f = A @ x
    assert np.all(f.val == [14, 32, 50])
    assert np.all((f.jac == A).A)


def test_advar_init_diff_len():
    a, b = initAdArrays([np.array([1, 2, 3]), np.array([1, 2])])
    A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    B = sps.csc_matrix(np.array([[1, 2], [4, 5]]))

    f = A @ a
    g = B @ b
    zero_32 = sps.csc_matrix((3, 2))
    zero_23 = sps.csc_matrix((2, 3))

    jac_f = sps.hstack((A, zero_32))
    jac_g = sps.hstack((zero_23, B))
    assert np.all(f.val == [14, 32, 50])
    assert np.all((f.jac == jac_f).A)
    assert np.all(g.val == [5, 14])
    assert np.all((g.jac == jac_g).A)


def test_advar_init_cross_jacobi():
    x, y = initAdArrays([np.array([-1, 4]), np.array([1, 5])])

    z = x * y
    J = np.array([[1, 0, -1, 0], [0, 5, 0, 4]])
    assert np.all(z.val == [-1, 20])
    assert np.all((z.jac == J).A)


def test_exp_scalar_times_ad_var():
    val = np.array([1, 2, 3])
    J = sps.diags(np.array([1, 1, 1]))
    a, _, _ = initAdArrays([val, val, val])
    c = 2.0
    b = af.exp(c * a)

    zero = sps.csc_matrix((3, 3))
    jac = sps.hstack([c * sps.diags(np.exp(c * val)) * J, zero, zero])
    jac_a = sps.hstack([J, zero, zero])
    assert np.allclose(b.val, np.exp(c * val)) and np.allclose(b.jac.A, jac.A)
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == jac_a.A)


def test_advar_mul_vec():
    x = Ad_array(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
    A = np.array([1, 3, 10])
    f = x * A
    sol = np.array([1, 6, 30])
    jac = np.diag([3, 6, 10])

    assert np.all(f.val == sol) and np.all(f.jac == jac)
    assert np.all(x.val == np.array([1, 2, 3])) and np.all(x.jac == np.diag([3, 2, 1]))


def test_advar_m_mul_vec_n():
    x = Ad_array(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
    vec = np.array([1, 2])
    R = sps.csc_matrix(np.array([[1, 0, 1], [0, 1, 0]]))
    y = R @ x
    z = y * vec
    Jy = np.array([[3, 0, 1], [0, 2, 0]])
    Jz = np.array([[1, 0, 3], [0, 4, 0]])
    assert np.all(y.val == [4, 2])
    assert np.sum(y.jac.A - Jy) == 0
    assert np.all(z.val == [4, 4])
    assert np.sum(z.jac.A - Jz) == 0


def test_mul_sps_advar():
    J = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
    x = Ad_array(np.array([1, 2, 3]), J)
    A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    f = A @ x

    assert np.all(f.val == [14, 32, 50])
    assert np.all(f.jac == A * J.A)


def test_mul_advar_vectors():
    Ja = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
    Jb = sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    a = Ad_array(np.array([1, 2, 3]), Ja)
    b = Ad_array(np.array([1, 1, 1]), Jb)
    A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    f = A @ a + b

    assert np.all(f.val == [15, 33, 51])
    assert np.sum(f.jac.A != A * Ja + Jb) == 0
    assert (
        np.sum(Ja != sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))) == 0
    )
    assert (
        np.sum(Jb != sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))) == 0
    )


def test_copy_scalar():
    a = Ad_array(np.array([1]), sps.csr_matrix([[0]]))
    b = a.copy()
    assert a.val == b.val
    assert a.jac == b.jac
    a.val = 2
    a.jac = 3
    assert b.val == 1
    assert b.jac == 0


def test_copy_vector():
    a = Ad_array(np.ones(3), sps.csr_matrix(np.diag(np.ones((3)))))
    b = a.copy()
    assert np.allclose(a.val, b.val)
    assert np.allclose(a.jac.A, b.jac.A)
    a.val[0] = 3
    a.jac[2] = 4
    assert np.allclose(b.val, np.ones(3))
    assert np.allclose(b.jac.A, sps.csr_matrix(np.diag(np.ones((3)))).A)
