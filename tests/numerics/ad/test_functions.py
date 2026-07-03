"""Test collection of the functions being wrapped in a
:class:`~porepy.numerics.ad.operator_functions.Function`.

For each supported function, the value and jacobian are compared with a reference data.
For AD quantities, four cases are considered: scalar, variable, sparse Jacobian, and
scalar times a variable.

"""

import warnings

import numpy as np
import pytest
import scipy.sparse as sps

from porepy.numerics.ad import AdArray
from porepy.numerics.ad import functions as af

warnings.simplefilter("ignore", sps.SparseEfficiencyWarning)


def test_exp_scalar():
    a = AdArray(np.array([1]), sps.csr_matrix(np.array([0])))
    b = af.exp(a)
    assert b.val == np.exp(1) and b.jac == 0
    assert a.val == 1 and a.jac == 0


def test_exp_advar():
    a = AdArray(np.array([2]), sps.csr_matrix(np.array([3])))
    b = af.exp(a)
    assert b.val == np.exp(2) and b.jac == 3 * np.exp(2)
    assert a.val == 2 and a.jac == 3


def test_exp_vector():
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.exp(a)
    jac = np.dot(np.diag(np.exp(val)), J)

    assert np.all(b.val == np.exp(val)) and np.all(b.jac == jac)
    assert np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))


def test_exp_sparse_jac():
    val = np.array([1, 2, 3])
    J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
    a = AdArray(val, J)
    b = af.exp(a)
    jac = np.dot(np.diag(np.exp(val)), J.toarray())
    assert np.all(b.val == np.exp(val)) and np.all(b.jac == jac)


def test_exp_scalar_times_ad_var():
    val = np.array([1, 2, 3])
    J = sps.diags(np.array([1, 1, 1]))
    a = AdArray(val, J)
    c = 2.0
    b = af.exp(c * a)
    jac = c * sps.diags(np.exp(c * val)) * J

    assert np.allclose(b.val, np.exp(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.toarray() == J.toarray())


# Function: log
def test_log_scalar():
    a = AdArray(np.array([2]), sps.csr_matrix(np.array([0])))
    b = af.log(a)
    assert b.val == np.log(2) and b.jac == 0
    assert a.val == 2 and a.jac == 0


def test_log_advar():
    a = AdArray(np.array([2]), sps.csr_matrix(np.array([3])))
    b = af.log(a)
    assert b.val == np.log(2) and b.jac == 1 / 2 * 3
    assert a.val == 2 and a.jac == 3


def test_log_vector():
    val = np.array([1, 2, 3])
    J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
    a = AdArray(val, J)
    b = af.log(a)
    jac = sps.diags(1 / val) * J

    assert np.all(b.val == np.log(val)) and np.all(b.jac.toarray() == jac)


def test_log_sparse_jac():
    val = np.array([1, 2, 3])
    J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
    a = AdArray(val, J)
    b = af.log(a)
    jac = np.dot(np.diag(1 / val), J.toarray())
    assert np.all(b.val == np.log(val)) and np.all(b.jac == jac)


def test_log_scalar_times_ad_var():
    val = np.array([1, 2, 3])
    J = sps.diags(np.array([1, 1, 1]))
    a = AdArray(val, J)
    c = 2.0
    b = af.log(c * a)
    jac = sps.diags(1 / val) * J

    assert np.allclose(b.val, np.log(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.toarray() == J.toarray())


# Function: abs
def test_abs_no_advar():
    a = np.array([1, -10, 3, -np.pi])
    a_abs = af.abs(a)
    assert np.allclose(a_abs, [1, 10, 3, np.pi])
    assert np.allclose(a, [1, -10, 3, -np.pi])


def test_abs_advar():
    J = np.array([[1, -1, -np.pi, 3], [0, 0, 0, 0], [1, 2, -3.2, 4], [4, 2, 300000, 1]])
    a = AdArray(np.array([1, -10, 3, -np.pi]), sps.csc_matrix(J))
    a_abs = af.abs(a)
    J_abs = np.array(
        [[1, -1, -np.pi, 3], [0, 0, 0, 0], [1, 2, -3.2, 4], [-4, -2, -300000, -1]]
    )

    assert np.allclose(
        J,
        np.array(
            [
                [1, -1, -np.pi, 3],
                [0, 0, 0, 0],
                [1, 2, -3.2, 4],
                [4, 2, 300000, 1],
            ]
        ),
    )
    assert np.allclose(a_abs.val, [1, 10, 3, np.pi])
    assert np.allclose(a_abs.jac.toarray(), J_abs)


# Function: sin
def test_sin_scalar():
    a = AdArray(np.array([1]), sps.csr_matrix(np.array([0])))
    b = af.sin(a)
    assert b.val == np.sin(1) and b.jac == 0
    assert a.val == 1 and a.jac == 0


def test_sin_advar():
    a = AdArray(np.array([2]), sps.csr_matrix(np.array([3])))
    b = af.sin(a)
    assert b.val == np.sin(2) and b.jac == np.cos(2) * 3
    assert a.val == 2 and a.jac == 3


def test_sin_vector():
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.sin(a)
    jac = np.dot(np.diag(np.cos(val)), J)

    assert np.all(b.val == np.sin(val)) and np.all(b.jac == jac)
    assert np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))


def test_sin_sparse_jac():
    val = np.array([1, 2, 3])
    J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
    a = AdArray(val, J)
    b = af.sin(a)
    jac = np.dot(np.diag(np.cos(val)), J.toarray())
    assert np.all(b.val == np.sin(val)) and np.all(b.jac == jac)


def test_sin_scalar_times_ad_var():
    val = np.array([1, 2, 3])
    J = sps.diags(np.array([1, 1, 1]))
    a = AdArray(val, J)
    c = 2.0
    b = af.sin(c * a)
    jac = c * sps.diags(np.cos(c * val)) * J

    assert np.allclose(b.val, np.sin(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.toarray() == J.toarray())


# Function: cos
def test_cos_scalar():
    a = AdArray(np.array([1]), sps.csr_matrix(np.array([0])))
    b = af.cos(a)
    assert b.val == np.cos(1) and b.jac == 0
    assert a.val == 1 and a.jac == 0


def test_cos_advar():
    a = AdArray(np.array([2]), sps.csr_matrix(np.array([3])))
    b = af.cos(a)
    assert b.val == np.cos(2) and b.jac == -np.sin(2) * 3
    assert a.val == 2 and a.jac == 3


def test_cos_vector():
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.cos(a)
    jac = np.dot(-np.diag(np.sin(val)), J)

    assert np.all(b.val == np.cos(val)) and np.all(b.jac == jac)
    assert np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))


def test_cos_sparse_jac():
    val = np.array([1, 2, 3])
    J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
    a = AdArray(val, J)
    b = af.cos(a)
    jac = np.dot(-np.diag(np.sin(val)), J.toarray())
    assert np.all(b.val == np.cos(val)) and np.all(b.jac == jac)


def test_cos_scalar_times_ad_var():
    val = np.array([1, 2, 3])
    J = sps.diags(np.array([1, 1, 1]))
    a = AdArray(val, J)
    c = 2.0
    b = af.cos(c * a)
    jac = -c * sps.diags(np.sin(c * val)) * J

    assert np.allclose(b.val, np.cos(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.toarray() == J.toarray())


# Function: tan
def test_tan_scalar():
    a = AdArray(np.array([1]), sps.csr_matrix(np.array([0])))
    b = af.tan(a)
    assert b.val == np.tan(1) and b.jac == 0
    assert a.val == 1 and a.jac == 0


def test_tan_advar():
    a = AdArray(np.array([2]), sps.csr_matrix(np.array([3])))
    b = af.tan(a)
    assert b.val == np.tan(2) and b.jac == 1 / (np.cos(2) ** 2) * 3
    assert a.val == 2 and a.jac == 3


def test_tan_vector():
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.tan(a)
    jac = np.dot(np.diag((np.cos(val) ** 2) ** (-1)), J)

    assert np.all(b.val == np.tan(val)) and np.all(b.jac == jac)
    assert np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))


def test_tan_sparse_jac():
    val = np.array([1, 2, 3])
    J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
    a = AdArray(val, J)
    b = af.tan(a)
    jac = np.dot(np.diag((np.cos(val) ** 2) ** (-1)), J.toarray())
    assert np.all(b.val == np.tan(val)) and np.all(b.jac == jac)


def test_tan_scalar_times_ad_var():
    val = np.array([1, 2, 3])
    J = sps.diags(np.array([1, 1, 1]))
    a = AdArray(val, J)
    c = 2.0
    b = af.tan(c * a)
    jac = c * sps.diags((np.cos(c * val) ** 2) ** (-1)) * J

    assert np.allclose(b.val, np.tan(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.toarray() == J.toarray())


# Function: arcsin
def test_arcsin_scalar():
    a = AdArray(np.array([0.5]), sps.csr_matrix(np.array([0])))
    b = af.arcsin(a)
    assert b.val == np.arcsin(0.5) and b.jac == 0
    assert a.val == 0.5 and a.jac == 0


def test_arcsin_advar():
    a = AdArray(np.array([0.2]), sps.csr_matrix(np.array([0.3])))
    b = af.arcsin(a)
    assert b.val == np.arcsin(0.2) and b.jac == (1 - 0.2**2) ** (-0.5) * 0.3
    assert a.val == 0.2 and a.jac == 0.3


def test_arcsin_vector():
    val = np.array([0.1, 0.2, 0.3])
    J = np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.arcsin(a)
    jac = np.dot(np.diag((1 - val**2) ** (-0.5)), J)

    assert np.all(b.val == np.arcsin(val)) and np.all(b.jac == jac)
    assert np.all(J == np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))


def test_arcsin_sparse_jac():
    val = np.array([0.1, 0.2, 0.3])
    J = sps.csc_matrix(np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))
    a = AdArray(val, J)
    b = af.arcsin(a)
    jac = np.dot(np.diag((1 - val**2) ** (-0.5)), J.toarray())
    assert np.all(b.val == np.arcsin(val)) and np.all(b.jac == jac)


def test_arcsin_scalar_times_ad_var():
    val = np.array([0.1, 0.2, 0.3])
    J = sps.diags(np.array([0.1, 0.1, 0.1]))
    a = AdArray(val, J)
    c = 0.2
    b = af.arcsin(c * a)
    jac = sps.diags(c * (1 - (c * val) ** 2) ** (-0.5)) * J

    assert np.allclose(b.val, np.arcsin(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [0.1, 0.2, 0.3]) and np.all(a.jac.toarray() == J.toarray())


# Function: arccos
def test_arccos_scalar():
    a = AdArray(np.array([0.5]), sps.csr_matrix(np.array([0])))
    b = af.arccos(a)
    assert b.val == np.arccos(0.5) and b.jac == 0
    assert a.val == 0.5 and a.jac == 0


def test_arccos_advar():
    a = AdArray(np.array([0.2]), sps.csr_matrix(np.array([0.3])))
    b = af.arccos(a)
    assert b.val == np.arccos(0.2) and b.jac == -((1 - 0.2**2) ** (-0.5)) * 0.3
    assert a.val == 0.2 and a.jac == 0.3


def test_arccos_vector():
    val = np.array([0.1, 0.2, 0.3])
    J = np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.arccos(a)
    jac = np.dot(-np.diag((1 - val**2) ** (-0.5)), J)

    assert np.all(b.val == np.arccos(val)) and np.all(b.jac == jac)
    assert np.all(J == np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))


def test_arccos_sparse_jac():
    val = np.array([0.1, 0.2, 0.3])
    J = sps.csc_matrix(np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))
    a = AdArray(val, J)
    b = af.arccos(a)
    jac = np.dot(-np.diag((1 - val**2) ** (-0.5)), J.toarray())
    assert np.all(b.val == np.arccos(val)) and np.all(b.jac == jac)


def test_arccos_scalar_times_ad_var():
    val = np.array([0.1, 0.2, 0.3])
    J = sps.diags(np.array([0.1, 0.1, 0.1]))
    a = AdArray(val, J)
    c = 0.2
    b = af.arccos(c * a)
    jac = -sps.diags(c * (1 - (c * val) ** 2) ** (-0.5)) * J

    assert np.allclose(b.val, np.arccos(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [0.1, 0.2, 0.3]) and np.all(a.jac.toarray() == J.toarray())


# Function: arctan
def test_arctan_scalar():
    a = AdArray(np.array([0.5]), sps.csr_matrix(np.array([0])))
    b = af.arctan(a)
    assert b.val == np.arctan(0.5) and b.jac == 0
    assert a.val == 0.5 and a.jac == 0


def test_arctan_advar():
    a = AdArray(np.array([0.2]), sps.csr_matrix(np.array([0.3])))
    b = af.arctan(a)
    assert b.val == np.arctan(0.2) and b.jac == (1 + 0.2**2) ** (-1) * 0.3
    assert a.val == 0.2 and a.jac == 0.3


def test_arctan_vector():
    val = np.array([0.1, 0.2, 0.3])
    J = np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.arctan(a)
    jac = np.dot(np.diag((1 + val**2) ** (-1)), J)

    assert np.all(b.val == np.arctan(val)) and np.all(b.jac == jac)
    assert np.all(J == np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))


def test_arctan_sparse_jac():
    val = np.array([0.1, 0.2, 0.3])
    J = sps.csc_matrix(np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))
    a = AdArray(val, J)
    b = af.arctan(a)
    jac = np.dot(np.diag((1 + val**2) ** (-1)), J.toarray())
    assert np.all(b.val == np.arctan(val)) and np.all(b.jac == jac)


def test_arctan_scalar_times_ad_var():
    val = np.array([0.1, 0.2, 0.3])
    J = sps.diags(np.array([0.1, 0.1, 0.1]))
    a = AdArray(val, J)
    c = 0.2
    b = af.arctan(c * a)
    jac = sps.diags(c * (1 + (c * val) ** 2) ** (-1)) * J

    assert np.allclose(b.val, np.arctan(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [0.1, 0.2, 0.3]) and np.all(a.jac.toarray() == J.toarray())


# Function: sinh
def test_sinh_scalar():
    a = AdArray(np.array([1]), sps.csr_matrix(np.array([0])))
    b = af.sinh(a)
    assert b.val == np.sinh(1) and b.jac == 0
    assert a.val == 1 and a.jac == 0


def test_sinh_advar():
    a = AdArray(np.array([2]), sps.csr_matrix(np.array([3])))
    b = af.sinh(a)
    assert b.val == np.sinh(2) and b.jac == np.cosh(2) * 3
    assert a.val == 2 and a.jac == 3


def test_sinh_vector():
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.sinh(a)
    jac = np.dot(np.diag(np.cosh(val)), J)

    assert np.all(b.val == np.sinh(val)) and np.all(b.jac == jac)
    assert np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))


def test_sinh_sparse_jac():
    val = np.array([1, 2, 3])
    J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
    a = AdArray(val, J)
    b = af.sinh(a)
    jac = np.dot(np.diag(np.cosh(val)), J.toarray())
    assert np.all(b.val == np.sinh(val)) and np.all(b.jac == jac)


def test_sinh_scalar_times_ad_var():
    val = np.array([1, 2, 3])
    J = sps.diags(np.array([1, 1, 1]))
    a = AdArray(val, J)
    c = 2.0
    b = af.sinh(c * a)
    jac = c * sps.diags(np.cosh(c * val)) * J

    assert np.allclose(b.val, np.sinh(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.toarray() == J.toarray())


# Function: cosh
def test_cosh_scalar():
    a = AdArray(np.array([1]), sps.csr_matrix(np.array([0])))
    b = af.cosh(a)
    assert b.val == np.cosh(1) and b.jac == 0
    assert a.val == 1 and a.jac == 0


def test_cosh_advar():
    a = AdArray(np.array([2]), sps.csr_matrix(np.array([3])))
    b = af.cosh(a)
    assert b.val == np.cosh(2) and b.jac == np.sinh(2) * 3
    assert a.val == 2 and a.jac == 3


def test_cosh_vector():
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.cosh(a)
    jac = np.dot(np.diag(np.sinh(val)), J)

    assert np.all(b.val == np.cosh(val)) and np.all(b.jac == jac)
    assert np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))


def test_cosh_sparse_jac():
    val = np.array([1, 2, 3])
    J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
    a = AdArray(val, J)
    b = af.cosh(a)
    jac = np.dot(np.diag(np.sinh(val)), J.toarray())
    assert np.all(b.val == np.cosh(val)) and np.all(b.jac == jac)


def test_cosh_scalar_times_ad_var():
    val = np.array([1, 2, 3])
    J = sps.diags(np.array([1, 1, 1]))
    a = AdArray(val, J)
    c = 2.0
    b = af.cosh(c * a)
    jac = c * sps.diags(np.sinh(c * val)) * J

    assert np.allclose(b.val, np.cosh(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.toarray() == J.toarray())


# Function: tanh
def test_tanh_scalar():
    a = AdArray(np.array([1]), sps.csr_matrix(np.array([0])))
    b = af.tanh(a)
    assert b.val == np.tanh(1) and b.jac == 0
    assert a.val == 1 and a.jac == 0


def test_tanh_advar():
    a = AdArray(np.array([2]), sps.csr_matrix(np.array([3])))
    b = af.tanh(a)
    assert b.val == np.tanh(2) and b.jac == np.cosh(2) ** (-2) * 3
    assert a.val == 2 and a.jac == 3


def test_tanh_vector():
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.tanh(a)
    jac = np.dot(np.diag((np.cosh(val) ** 2) ** (-1)), J)

    assert np.all(b.val == np.tanh(val)) and np.all(b.jac == jac)
    assert np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))


def test_tanh_sparse_jac():
    val = np.array([1, 2, 3])
    J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
    a = AdArray(val, J)
    b = af.tanh(a)
    jac = np.dot(np.diag((np.cosh(val) ** 2) ** (-1)), J.toarray())
    assert np.all(b.val == np.tanh(val)) and np.all(b.jac == jac)


def test_tanh_scalar_times_ad_var():
    val = np.array([1, 2, 3])
    J = sps.diags(np.array([1, 1, 1]))
    a = AdArray(val, J)
    c = 2.0
    b = af.tanh(c * a)
    jac = c * sps.diags((np.cosh(c * val) ** 2) ** (-1)) * J

    assert np.allclose(b.val, np.tanh(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.toarray() == J.toarray())


# Function: arcsinh
def test_arcsinh_scalar():
    a = AdArray(np.array([0.5]), sps.csr_matrix(np.array([0])))
    b = af.arcsinh(a)
    assert np.isclose(b.val, np.arcsinh(0.5)) and b.jac == 0
    assert np.isclose(a.val, 0.5) and a.jac == 0


def test_arcsinh_advar():
    a = AdArray(np.array([0.2]), sps.csr_matrix(np.array([0.3])))
    b = af.arcsinh(a)
    assert np.isclose(b.val, np.arcsinh(0.2)) and np.isclose(
        b.jac.toarray(), (1 + 0.2**2) ** (-0.5) * 0.3
    )
    assert a.val == 0.2 and a.jac == 0.3


def test_arcsinh_vector():
    val = np.array([0.1, 0.2, 0.3])
    J = np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.arcsinh(a)
    jac = np.dot(np.diag((1 + val**2) ** (-0.5)), J)

    assert np.allclose(b.val, np.arcsinh(val)) and np.allclose(b.jac.toarray(), jac)
    assert np.all(J == np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))


def test_arcsinh_sparse_jac():
    val = np.array([0.1, 0.2, 0.3])
    J = sps.csc_matrix(np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))
    a = AdArray(val, J)
    b = af.arcsinh(a)
    jac = np.dot(np.diag((1 + val**2) ** (-0.5)), J.toarray())
    assert np.allclose(b.val, np.arcsinh(val)) and np.allclose(b.jac.toarray(), jac)


def test_arcsinh_scalar_times_ad_var():
    val = np.array([0.1, 0.2, 0.3])
    J = sps.diags(np.array([0.1, 0.1, 0.1]))
    a = AdArray(val, J)
    c = 0.2
    b = af.arcsinh(c * a)
    jac = sps.diags(c * (1 + (c * val) ** 2) ** (-0.5)) * J

    assert np.allclose(b.val, np.arcsinh(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [0.1, 0.2, 0.3]) and np.all(a.jac.toarray() == J.toarray())


# Function: arccosh
def test_arccosh_scalar():
    a = AdArray(np.array([2]), sps.csr_matrix(np.array([0])))
    b = af.arccosh(a)
    assert np.isclose(b.val, np.arccosh(2)) and b.jac == 0
    assert a.val == 2 and a.jac == 0


def test_arccosh_advar():
    a = AdArray(np.array([2]), sps.csr_matrix(np.array([3])))
    b = af.arccosh(a)
    assert (
        np.isclose(b.val, np.arccosh(2))
        and b.jac == (2 - 1) ** (-0.5) * (2 + 1) ** (-0.5) * 3
    )
    assert a.val == 2 and a.jac == 3


def test_arccosh_vector():
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.arccosh(a)
    jac = np.dot(np.diag((val - 1) ** (-0.5) * (val + 1) ** (-0.5)), J)

    assert np.allclose(b.val, np.arccosh(val)) and np.allclose(b.jac.toarray(), jac)
    assert np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))


def test_arccosh_sparse_jac():
    val = np.array([1, 2, 3])
    J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
    a = AdArray(val, J)
    b = af.arccosh(a)
    jac = np.dot(np.diag((val - 1) ** (-0.5) * (val + 1) ** (-0.5)), J.toarray())
    assert np.allclose(b.val, np.arccosh(val)) and np.allclose(b.jac.toarray(), jac)


def test_arccosh_scalar_times_ad_var():
    val = np.array([1, 2, 3])
    J = sps.diags(np.array([1, 1, 1]))
    a = AdArray(val, J)
    c = 2.0
    b = af.arccosh(c * a)
    jac = sps.diags(c * (c * val - 1) ** (-0.5) * (c * val + 1) ** (-0.5)) * J

    assert np.allclose(b.val, np.arccosh(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.toarray() == J.toarray())


# Function: arctanh
def test_arctanh_scalar():
    a = AdArray(np.array([0.5]), sps.csr_matrix(np.array([0])))
    b = af.arctanh(a)
    assert np.isclose(b.val, np.arctanh(0.5)) and b.jac == 0
    assert a.val == 0.5 and a.jac == 0


def test_arctanh_advar():
    a = AdArray(np.array([0.2]), sps.csr_matrix(np.array([0.3])))
    b = af.arctanh(a)
    assert np.isclose(b.val, np.arctanh(0.2)) and np.isclose(
        b.jac.toarray(), (1 - 0.2**2) ** (-1) * 0.3
    )
    assert a.val == 0.2 and a.jac == 0.3


def test_arctanh_vector():
    val = np.array([0.1, 0.2, 0.3])
    J = np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.arctanh(a)
    jac = np.dot(np.diag((1 - val**2) ** (-1)), J)

    assert np.allclose(b.val, np.arctanh(val)) and np.allclose(b.jac.toarray(), jac)
    assert np.all(J == np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))


def test_arctanh_sparse_jac():
    val = np.array([0.1, 0.2, 0.3])
    J = sps.csc_matrix(np.array([[0.3, 0.2, 0.1], [0.5, 0.6, 0.1], [0.2, 0.3, 0.5]]))
    a = AdArray(val, J)
    b = af.arctanh(a)
    jac = np.dot(np.diag((1 - val**2) ** (-1)), J.toarray())
    assert np.allclose(b.val, np.arctanh(val)) and np.allclose(b.jac.toarray(), jac)


def test_arctanh_scalar_times_ad_var():
    val = np.array([0.1, 0.2, 0.3])
    J = sps.diags(np.array([0.1, 0.1, 0.1]))
    a = AdArray(val, J)
    c = 0.2
    b = af.arctanh(c * a)
    jac = sps.diags(c * (1 - (c * val) ** 2) ** (-1)) * J

    assert np.allclose(b.val, np.arctanh(c * val)) and np.allclose(
        b.jac.toarray(), jac.toarray()
    )
    assert np.all(a.val == [0.1, 0.2, 0.3]) and np.all(a.jac.toarray() == J.toarray())


# Function: heaviside_smooth
def test_heaviside_smooth_scalar():
    a = AdArray(np.array([0.5]), sps.csr_matrix(np.array([0])))
    b = af.heaviside_smooth(a)
    val = 0.5 * (1 + 2 / np.pi * np.arctan(0.5 / 1e-3))
    assert b.val == val and b.jac == 0
    assert a.val == 0.5 and a.jac == 0


def test_heaviside_smooth_advar():
    a = AdArray(np.array([0.2]), sps.csr_matrix(np.array([0.3])))
    b = af.heaviside_smooth(a)
    val = 0.5 * (1 + (2 / np.pi) * np.arctan(0.2 / 1e-3))
    der = (1 / np.pi) * (1e-3 / (1e-3**2 + 0.2**2))
    assert np.isclose(b.val, val) and np.isclose(b.jac.toarray(), der * 0.3)
    assert a.val == 0.2 and a.jac == 0.3


def test_heaviside_smooth_vector():
    val = np.array([1, -2, 3])
    J = np.array([[3, -2, 1], [-5, 6, 1], [2, 3, -5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = af.heaviside_smooth(a)

    true_val = 0.5 * (1 + 2 * np.pi ** (-1) * np.arctan(val * 1e3))
    true_jac = np.dot(np.diag(np.pi ** (-1) * (1e-3 * (1e-3**2 + val**2) ** (-1))), J)

    assert np.allclose(b.val, true_val) and np.allclose(b.jac.toarray(), true_jac)
    assert np.all(J == np.array([[3, -2, 1], [-5, 6, 1], [2, 3, -5]]))


def test_heaviside_smooth_sparse_jac():
    val = np.array([1, -2, 3])
    J = sps.csc_matrix(np.array([[3, -2, 1], [-5, 6, 1], [2, 3, -5]]))
    a = AdArray(val, J)
    b = af.heaviside_smooth(a)

    true_val = 0.5 * (1 + 2 * np.pi ** (-1) * np.arctan(val * 1e3))
    true_jac = np.dot(
        np.diag(np.pi ** (-1) * (1e-3 * (1e-3**2 + val**2) ** (-1))), J.toarray()
    )

    assert np.allclose(b.val, true_val) and np.allclose(b.jac.toarray(), true_jac)


def test_heaviside_smooth_times_ad_var():
    val = np.array([1, -2, -3])
    J = sps.diags(np.array([0.1, 0.1, 0.1]))
    a = AdArray(val, J)
    c = 0.2
    b = af.heaviside_smooth(c * a)

    true_val = 0.5 * (1 + 2 * np.pi ** (-1) * np.arctan(c * val * 1e3))
    true_jac = (
        sps.diags(c * np.pi ** (-1) * (1e-3 * (1e-3**2 + (c * val) ** 2) ** (-1))) * J
    )

    assert np.allclose(b.val, true_val) and np.allclose(
        b.jac.toarray(), true_jac.toarray()
    )
    assert np.all(a.val == [1, -2, -3]) and np.all(a.jac.toarray() == J.toarray())


# Function: mask_by_threshold
@pytest.mark.parametrize(
    "char_var,var,tol,expected_val,expected_jac",
    [
        # Test case 1: scalar, no AdArray
        pytest.param(
            np.array([0.5, -0.1, 2.0]),
            10.0,
            0.0,
            np.array([10.0, 0.0, 10.0]),
            None,
            id="scalar_no_advar",
        ),
        # Test case 2: ndarray, no AdArray
        pytest.param(
            np.array([0.5, -0.1, 2.0]),
            np.array([10, 20, 30]),
            0.0,
            np.array([10, 0, 30]),
            None,
            id="array_no_advar",
        ),
        # Test case 3: NaN * 0 = 0
        pytest.param(
            np.array([0.5, -0.1, 2.0]),
            np.array([10.0, np.nan, 30.0]),
            0.0,
            np.array([10.0, 0.0, 30.0]),
            None,
            id="nan_times_zero",
        ),
        # Test case 4: AdArray with tolerance
        pytest.param(
            AdArray(np.array([0.05, 0.1, 1.0]), sps.csr_matrix((3, 3))),
            AdArray(np.array([10, 20, 30]), sps.csr_matrix(np.diag([1, 1, 1]))),
            0.08,
            np.array([0, 20, 30]),
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),
            id="adarray_with_tolerance",
        ),
        # Test case 5: all masked
        pytest.param(
            AdArray(np.array([0.1, 0.2, 0.3]), sps.csr_matrix((3, 3))),
            AdArray(np.array([10, 20, 30]), sps.csr_matrix(np.diag([1, 1, 1]))),
            0.5,
            np.array([0, 0, 0]),
            np.zeros((3, 3)),
            id="all_masked",
        ),
        # Test case 6: all kept
        pytest.param(
            AdArray(np.array([0.5, 1.0, 2.0]), sps.csr_matrix((3, 3))),
            AdArray(np.array([10, 20, 30]), sps.csr_matrix(np.diag([1, 1, 1]))),
            0.0,
            np.array([10, 20, 30]),
            np.diag([1, 1, 1]),
            id="all_kept",
        ),
    ],
)
def test_mask_by_threshold(char_var, var, tol, expected_val, expected_jac):
    """Parametrized test for mask_by_threshold covering multiple cases."""
    result = af.mask_by_threshold(tol, char_var, var)

    # Check values
    assert np.allclose(result.val if hasattr(result, "val") else result, expected_val)

    # Check Jacobian if expected
    if expected_jac is not None:
        assert hasattr(result, "jac"), "Expected AdArray with Jacobian"
        assert np.allclose(result.jac.toarray(), expected_jac)


# Function: clip


def test_clip_ndarray():
    # Values entirely within bounds, at min, at max, and outside both bounds.
    var = np.array([-2.0, 0.0, 1.5, 5.0])
    result = af.clip(var, 0.0, 3.0)
    assert np.allclose(result, np.array([0.0, 0.0, 1.5, 3.0]))


def test_clip_float():
    assert af.clip(2.0, 0.0, 3.0) == 2.0
    assert af.clip(-1.0, 0.0, 3.0) == 0.0
    assert af.clip(5.0, 0.0, 3.0) == 3.0


def test_clip_adarray_values():
    # Check that values are clipped correctly.
    val = np.array([-1.0, 1.0, 4.0])
    J = sps.eye(3, format="csr")
    a = AdArray(val, J)
    b = af.clip(a, 0.0, 3.0)
    assert np.allclose(b.val, np.array([0.0, 1.0, 3.0]))


def test_clip_adarray_jacobian_interior():
    # For values strictly inside [min_val, max_val], the Jacobian is preserved.
    val = np.array([1.0, 2.0])
    J = sps.csr_matrix(np.array([[3.0, 0.0], [0.0, 5.0]]))
    a = AdArray(val, J)
    b = af.clip(a, 0.0, 3.0)
    assert np.allclose(b.jac.toarray(), J.toarray())


def test_clip_adarray_jacobian_at_bounds():
    # For values exactly at min or max, the Jacobian is zeroed out.
    val = np.array([0.0, 3.0])
    J = sps.eye(2, format="csr")
    a = AdArray(val, J)
    b = af.clip(a, 0.0, 3.0)
    assert np.allclose(b.jac.toarray(), np.zeros((2, 2)))


def test_clip_adarray_jacobian_outside_bounds():
    # For values outside [min_val, max_val], the Jacobian is zeroed out.
    val = np.array([-1.0, 5.0])
    J = sps.eye(2, format="csr")
    a = AdArray(val, J)
    b = af.clip(a, 0.0, 3.0)
    assert np.allclose(b.jac.toarray(), np.zeros((2, 2)))


def test_clip_adarray_mixed():
    # Mix of clipped (below, above) and interior values.
    val = np.array([-1.0, 1.0, 2.0, 5.0])
    J = sps.csr_matrix(
        np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]], dtype=float)
    )
    a = AdArray(val, J)
    b = af.clip(a, 0.0, 3.0)

    expected_val = np.array([0.0, 1.0, 2.0, 3.0])
    # Interior rows (indices 1 and 2) keep their Jacobian; clipped rows (0, 3) are zero.
    expected_jac = np.array(
        [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 0]], dtype=float
    )
    assert np.allclose(b.val, expected_val)
    assert np.allclose(b.jac.toarray(), expected_jac)


def test_clip_adarray_dense_jacobian():
    # Verify correctness with a non-diagonal (dense) Jacobian.
    val = np.array([0.5, 2.5])
    J = sps.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    a = AdArray(val, J)
    b = af.clip(a, 0.0, 3.0)
    assert np.allclose(b.val, np.array([0.5, 2.5]))
    assert np.allclose(b.jac.toarray(), J.toarray())


def test_clip_does_not_mutate_input():
    # Ensure the original AdArray is unchanged after clipping.
    val = np.array([-1.0, 2.0, 5.0])
    J = sps.eye(3, format="csr")
    a = AdArray(val.copy(), J.copy())
    # Perform clipping, but ignore the result to check that 'a' is unchanged. The clip
    # affects the values -1 and 5.
    _ = af.clip(a, 0.0, 3.0)
    assert np.allclose(a.val, np.array([-1.0, 2.0, 5.0]))
    assert np.allclose(a.jac.toarray(), sps.eye(3).toarray())


# Function: mask_by_threshold
@pytest.mark.parametrize(
    "char_var,var,tol,expected_val,expected_jac",
    [
        # Test case 1: scalar, no AdArray
        pytest.param(
            np.array([0.5, -0.1, 2.0]),
            10.0,
            0.0,
            np.array([10.0, 0.0, 10.0]),
            None,
            id="scalar_no_advar",
        ),
        # Test case 2: ndarray, no AdArray
        pytest.param(
            np.array([0.5, -0.1, 2.0]),
            np.array([10, 20, 30]),
            0.0,
            np.array([10, 0, 30]),
            None,
            id="array_no_advar",
        ),
        # Test case 3: NaN * 0 = 0
        pytest.param(
            np.array([0.5, -0.1, 2.0]),
            np.array([10.0, np.nan, 30.0]),
            0.0,
            np.array([10.0, 0.0, 30.0]),
            None,
            id="nan_times_zero",
        ),
        # Test case 4: AdArray with tolerance
        pytest.param(
            AdArray(np.array([0.05, 0.1, 1.0]), sps.csr_matrix((3, 3))),
            AdArray(np.array([10, 20, 30]), sps.csr_matrix(np.diag([1, 1, 1]))),
            0.08,
            np.array([0, 20, 30]),
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),
            id="adarray_with_tolerance",
        ),
        # Test case 5: all masked
        pytest.param(
            AdArray(np.array([0.1, 0.2, 0.3]), sps.csr_matrix((3, 3))),
            AdArray(np.array([10, 20, 30]), sps.csr_matrix(np.diag([1, 1, 1]))),
            0.5,
            np.array([0, 0, 0]),
            np.zeros((3, 3)),
            id="all_masked",
        ),
        # Test case 6: all kept
        pytest.param(
            AdArray(np.array([0.5, 1.0, 2.0]), sps.csr_matrix((3, 3))),
            AdArray(np.array([10, 20, 30]), sps.csr_matrix(np.diag([1, 1, 1]))),
            0.0,
            np.array([10, 20, 30]),
            np.diag([1, 1, 1]),
            id="all_kept",
        ),
    ],
)
def test_mask_by_threshold(char_var, var, tol, expected_val, expected_jac):
    """Parametrized test for mask_by_threshold covering multiple cases."""
    result = af.mask_by_threshold(tol, char_var, var)

    # Check values
    assert np.allclose(result.val if hasattr(result, "val") else result, expected_val)

    # Check Jacobian if expected
    if expected_jac is not None:
        assert hasattr(result, "jac"), "Expected AdArray with Jacobian"
        assert np.allclose(result.jac.toarray(), expected_jac)
