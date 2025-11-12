"""Testing module for functionality regarding general solutions of cubic polynomials
dependent on coefficients and their derivatives."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from porepy.applications.test_utils.derivative_testing import (
    get_EOC_taylor,
    assert_order_at_least,
)

from porepy.compositional.peng_robinson.compressibility_factor import (
    A_CRIT,
    B_CRIT,
    Z_CRIT,
    c_from_AB,
)
from porepy.compositional.peng_robinson.cubic_polynomial import (
    calculate_root_derivatives,
    calculate_roots,
    d_one_root,
    d_three_roots,
    d_triple_root,
    d_two_roots,
    get_r1,
    get_root_case,
    one_root,
    three_roots,
    triple_root,
    two_roots,
)


def get_polynomial_residual(r: float | np.ndarray, c: np.ndarray) -> float | np.ndarray:
    """Computes the residual of a normalized polynomial.

    The polynomial is assumed to be of order ``c.size`` and if ``r`` is a root then it
    holds

    .. math::

        p(r) = r^n + c[0]r^{n-1} + \dots + c[n-1] r + c[n] = 0

    Parameters:
        r: Supposed roots of the polynomial
        c: Coefficients of the polynomial ending with the constant monomial coefficient.

    Returns:
        The absolute value of above polynomial expression.

    """
    n = c.size
    c_ = np.hstack([1, c])
    r_ = [r**i for i in range(n, -1, -1)]
    return np.abs(np.dot(c_, r_))


def _get_random_coeffs_for_two_root_case() -> np.ndarray:
    """Get random coefficients for the two root case which does not violate the
    constraint that the first reduced coefficient is not zero."""
    c2 = np.random.rand()
    c1 = np.random.rand()
    c0 = np.random.rand()

    while np.abs(get_r1(c2, c1)) < 0.1:
        c1 = np.random.rand()

    return np.array([c2, c1, c0])


@pytest.mark.parametrize(
    ["coefficients", "solution", "root_case"],
    [
        (np.array([-1, -1, -2]), np.array([2.0]), 1),
        (np.array([-3, -3, -1]), np.array([np.cbrt(4) + np.cbrt(2) + 1]), 1),
        (np.array([1, -2, -2]), np.array([np.sqrt(2), -np.sqrt(2), -1]), 3),
        (np.array([4, 2, -4]), np.array([-1 + np.sqrt(3), -1 - np.sqrt(3), -2]), 3),
        (  # (x-1)*(x-2)**2
            np.array([-5, 8, -4]),
            np.array([1.0, 2.0]),
            2,
        ),
        (  # (x-1)*(x+2)**2
            np.array([3, 0, -4]),
            np.array([1.0, -2.0]),
            2,
        ),
        (  # (x-sqrt(2))**2
            np.array([-3 * np.sqrt(2), 6, -2 * np.sqrt(2)]),
            np.array([np.sqrt(2)]),
            0,
        ),
        (  # x**3 + 1
            np.array([0.0, 0.0, 1.0]),
            np.array([-1]),
            1,
        ),
        (  # x**3 - 1
            np.array([0.0, 0.0, -1.0]),
            np.array([1]),
            1,
        ),
        (
            np.array([2.0, 4.0 / 3.0, 1.0]),
            np.array([-2 - np.cbrt(19)]) / 3.0,
            1,
        ),
        (  # Peng-Robinson EoS critical point
            np.array(
                [
                    B_CRIT - 1,
                    A_CRIT - 2.0 * B_CRIT - 3.0 * B_CRIT**2,
                    B_CRIT**3 + B_CRIT**2 - A_CRIT * B_CRIT,
                ]
            ),
            np.array([Z_CRIT]),
            0,
        ),
    ],
)
def test_known_root_case_calculations(
    coefficients: np.ndarray,
    solution: np.ndarray,
    root_case: int,
) -> None:
    """For given coefficients of the polynomial, tests if the root case is correctly
    deduced and then if the root is correctly calculated.

    The calculation is once done using the explicit function for the root case, and once
    using the general function. Both results should match the known solution.

    """

    # NOTE: Due to numerics, we must allow this tolerance. The current code does not
    # reach lower tolerances for all test cases.
    tol = 1e-14

    calculated_root_case = get_root_case(*coefficients, tol)

    # Test the calculated root case.
    assert calculated_root_case == root_case

    # Custom computations are supposed to be returned sorted as well (ascending).
    solution = np.sort(solution)

    vals: np.ndarray

    match root_case:
        case 0:
            assert solution.size == 1
            vals = triple_root(coefficients[0])
        case 1:
            assert solution.size == 1
            vals = one_root(*coefficients)
        case 2:
            assert solution.size == 2
            vals = two_roots(*coefficients)
        case 3:
            assert solution.size == 3
            vals = three_roots(*coefficients)
        case _:
            assert False, "Faulty test"

    # Test computed root.
    np.testing.assert_allclose(vals, solution, atol=tol, rtol=0.0)

    # Test that it is indeed a rood.
    residual = get_polynomial_residual(vals, coefficients)
    np.testing.assert_allclose(residual, 0.0, atol=tol, rtol=0.0)

    # Test that the call to the general function returns the same result.
    genvals = calculate_roots(*coefficients, tol)
    np.testing.assert_allclose(genvals, solution, atol=tol, rtol=0.0)


def test_triple_root_derivatives() -> None:
    """Specialized test for the triple root derivative, which is constant and simple."""

    h_values = np.logspace(-1, -10, 10)
    X0 = np.random.rand(3)

    r = triple_root(X0[0])
    dr = d_triple_root(X0[0])
    approximations = [(triple_root(X0[0] + h) - r) / h for h in h_values]
    np.testing.assert_allclose(np.array(approximations), -1 / 3, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(
        dr, np.array([-1 / 3, 0.0, 0.0]).reshape((1, 3)), atol=1e-14, rtol=0.0
    )


@pytest.mark.parametrize(
    ["func", "dfunc"],
    [
        (one_root, d_one_root),
        (two_roots, d_two_roots),
    ],
)
@pytest.mark.parametrize(
    "x0",
    [
        c_from_AB(A_CRIT, B_CRIT),
        c_from_AB(0.0, 0.0),
    ],
)
def test_single_double_root_derivatives_around_triple_point(
    func: Callable[..., np.ndarray], dfunc: Callable[..., np.ndarray], x0: np.ndarray
) -> None:
    """The computation of derivatives for single and double root functions loses 1
    order of convergence when using a known triple point as base point.

    This is tested for a random direction around the Peng-Robinson critical point.

    """
    d = np.random.rand(x0.size)
    orders = get_EOC_taylor(func, dfunc, x0, d, h=np.logspace(-1, -10, 10))
    assert_order_at_least(orders, 1.0, tol=1e-2)


@pytest.mark.parametrize(
    ["func", "dfunc", "x0"],
    [
        (one_root, d_one_root, c_from_AB(0.7, 0.05)),
        (one_root, d_one_root, c_from_AB(0.2, 0.05)),
        # The two root functions should be defined everywhere where the first reduced
        # coefficients is not zero, i.e. c_1 != c_2**2 / 3.
        (two_roots, d_two_roots, _get_random_coeffs_for_two_root_case()),
        (two_roots, d_two_roots, _get_random_coeffs_for_two_root_case()),
        (three_roots, d_three_roots, c_from_AB(0.2, 0.015)),
        (three_roots, d_three_roots, c_from_AB(0.0, 0.1)),
    ],
)
@pytest.mark.parametrize(
    "direction",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.ones(3),
    ],
)
def test_root_derivatives(
    func: Callable[..., np.ndarray],
    dfunc: Callable[..., np.ndarray],
    x0: np.ndarray,
    direction: np.ndarray,
) -> None:
    """Tests the computation of the root derivative functions by asserting the
    first-order Taylor expansion approximates the function at a perturbed point with
    order 2.

    This test is necessary because the analytical expressions for the derivatives are
    hard-coded.

    Note:
        The test is run at specific point, where we know the root case.
        This is because of the nature of the solution formulae, they might be
        ill-defined otherwise. Testing in truly arbitrary point is difficult,
        because the computations can fail due to an assertion, the approximation can
        loose an order, or succeed with order 2.

        The idea is to chose a point with an area around it where the root case is
        constant.

    """
    orders = get_EOC_taylor(func, dfunc, x0, direction, np.logspace(-2, -11, 10))
    assert_order_at_least(orders, 2.0, tol=5e-2)


def test_general_root_derivative() -> None:
    """Tests the generic derivative calculation for roots by using two random
    coefficient sets with fixed seeds, one where order 2 is reached and one where only
    order 1 is reached."""
    np.random.seed(42)

    x0 = np.random.rand(3)
    d = np.random.rand(3)
    d /= np.linalg.norm(d)
    h = np.logspace(-1, -10, 10)

    def func(*args: float) -> np.ndarray:
        return calculate_roots(*args, eps=1e-14)

    def dfunc(*args: float) -> np.ndarray:
        return calculate_root_derivatives(*args, eps=1e-14)

    orders = get_EOC_taylor(func, dfunc, x0, d, h)
    assert_order_at_least(orders, 2.0, tol=1e-2)

    np.random.seed(2)

    x0 = np.random.rand(3)
    d = np.random.rand(3)
    d /= np.linalg.norm(d)

    orders = get_EOC_taylor(func, dfunc, x0, d, h)
    assert_order_at_least(orders, 1.0, tol=1e-2)
