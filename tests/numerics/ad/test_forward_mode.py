"""Collection of unit tests for the automatic differentiation forward mode. For the class
AdArray, tests are being conducted on the public attributes self.val, self.jac, and
self.copy. The tests also cover the initialization of AdArray (joint initiation of
multiple dependent variables) and the arithmetic operations implemented in AdArray, e.g.,
add, sub, etc., which are also covered in other tests.

"""
from __future__ import annotations

import pytest

import numpy as np
import scipy.sparse as sps

from porepy.numerics.ad import functions as af
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays


def test_quadratic_function():
    x, y = initAdArrays([np.array([1]), np.array([2])])
    z = 1 * x + 2 * y + 3 * x * y + 4 * x * x + 5 * y * y
    val = 35
    assert z.val == val and np.all(z.jac.A == [15, 25])


def test_vector_quadratic():
    x, y = initAdArrays([np.array([1, 1]), np.array([2, 3])])
    z = 1 * x + 2 * y + 3 * x * y + 4 * x * x + 5 * y * y
    val = np.array([35, 65])
    J = np.array([[15, 0, 25, 0], [0, 18, 0, 35]])

    assert np.all(z.val == val) and np.sum(z.jac != J) == 0


def test_mapping_m_to_n():
    x, y = initAdArrays([np.array([1, 1, 3]), np.array([2, 3])])
    A = sps.csc_matrix(np.array([[1, 2, 1], [2, 3, 4]]))

    z = y * (A @ x)
    val = np.array([12, 51])
    J = np.array([[2, 4, 2, 6, 0], [6, 9, 12, 0, 17]])

    assert np.all(z.val == val) and np.sum(z.jac != J) == 0


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


def test_advar_mul_vec():
    x = AdArray(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
    A = np.array([1, 3, 10])
    f = x * A
    sol = np.array([1, 6, 30])
    jac = np.diag([3, 6, 10])

    assert np.all(f.val == sol) and np.all(f.jac == jac)
    assert np.all(x.val == np.array([1, 2, 3])) and np.all(x.jac == np.diag([3, 2, 1]))


def test_advar_m_mul_vec_n():
    x = AdArray(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
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
    x = AdArray(np.array([1, 2, 3]), J)
    A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    f = A @ x

    assert np.all(f.val == [14, 32, 50])
    assert np.all(f.jac == A * J.A)


def test_mul_advar_vectors():
    Ja = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
    Jb = sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    a = AdArray(np.array([1, 2, 3]), Ja)
    b = AdArray(np.array([1, 1, 1]), Jb)
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
    a = AdArray(np.array([1]), sps.csr_matrix([[0]]))
    b = a.copy()
    assert a.val == b.val
    assert a.jac == b.jac
    a.val = 2
    a.jac = 3
    assert b.val == 1
    assert b.jac == 0


def test_copy_vector():
    a = AdArray(np.ones(3), sps.csr_matrix(np.diag(np.ones((3)))))
    b = a.copy()
    assert np.allclose(a.val, b.val)
    assert np.allclose(a.jac.A, b.jac.A)
    a.val[0] = 3
    a.jac[2] = 4
    assert np.allclose(b.val, np.ones(3))
    assert np.allclose(b.jac.A, sps.csr_matrix(np.diag(np.ones((3)))).A)


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


@pytest.mark.parametrize(
    'index,index_c', [  # indices and their complement for tested array
        (1, [0, 2, 3, 4, 5, 6, 7, 8, 9]),
        (slice(0, 10, 2), slice(1, 10, 2)),
        (np.array([0, 2, 4, 6, 8], dtype=int), np.array([1, 3, 5, 7, 9], dtype=int)),
    ]
)
def test_get_set_slice_ad_var(index, index_c):
    a = initAdArrays([np.arange(10)])[0]

    val = np.arange(10)
    jac = sps.csr_matrix(np.eye(10))

    assert np.all(val == a.val)
    assert np.all(jac == a.jac.toarray())

    if isinstance(index, int):
        target_val = np.array([val[index]])
    else:
        target_val = val[index]
    target_jac = jac[index].toarray()

    # Testing slicing
    a_slice = a[index]

    assert a_slice.val.shape == target_val.shape
    assert a_slice.jac.shape == target_jac.shape
    assert np.all(a_slice.val == target_val)
    assert np.all(a_slice.jac == target_jac)

    # testing setting values with slicing

    b = a[index] * 10.
    assert np.all(b.val == val[index] * 10.)
    assert np.all(b.jac.toarray() == jac[index] * 10.)

    # setting an AD array should set val and jacobian row-wise
    a_copy = a.copy()
    a[index] = b
    assert np.all(a[index].val == b.val)
    assert np.all(a[index].jac.A == b.jac.A)
    # complement should not be affected
    assert np.all(a[index_c].val == a_copy[index_c].val)
    assert np.all(a[index_c].jac.A == a_copy[index_c].jac.A)

    # setting a numpy array should only modify the values of the ad array
    b = target_val * 10.
    a = a_copy.copy()
    a[index] = b
    assert np.all(a[index].val == b)
    assert np.all(a[index_c].val == a_copy[index_c].val)
    assert np.all(a.jac.A == a_copy.jac.A)
