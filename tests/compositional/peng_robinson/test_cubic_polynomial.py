"""Testing module for functionality regarding general solutions of cubic polynomials
dependent on coefficients and their derivatives."""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pytest

# os.environ["NUMBA_DISABLE_JIT"] = "1"

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


def get_EOC_taylor(
    func: Callable[[tuple[float, ...]], np.ndarray],
    dfunc: Callable[[tuple[float, ...]], np.ndarray],
    x0: np.ndarray,
    d: np.ndarray,
    h: np.ndarray,
    verbose: bool = False,
    tol: float = 1e-14,
) -> np.ndarray:
    """Estimate the order of convergence (EOC) of the derivative
    computation of ``func`` at point ``X0`` along direction ``D`` using Taylor
    expansion.

    The EOC is estimated by computing the error between the exact function value and
    the first-order Taylor approximation for a sequence of step sizes.

    The function and its derivative function are assumed to take floats as arguments.
    The number of floats is indicated by the size of ``X0``. ``dfunc`` should return
    a 2D array with shape (m, n) where m is the number of function outputs and n is
    the number of input arguments (size of ``X0``).

    Parameters:
        func: Function for which the derivative is computed.
        dfunc: Function computing the derivative of `func`.
        x0: Point at which the derivative is computed.
        d: Direction along which the derivative is computed.
        h: Array of step sizes to use for the Taylor expansion.
        verbose: If True, prints detailed information about errors and estimated orders.
        tol: Tolerance below which errors are considered zero
            (i.e., exact approximation).

    Returns:
        Estimated EOC values for each consecutive pair of step sizes.

    """
    # Norming direction for sensible scaling.
    d = d / np.linalg.norm(d)

    # Testing derivative along a random direction.
    errors = []
    for h_ in h:
        approx = func(*x0) + h_ * (dfunc(*x0) @ d)
        exact = func(*(x0 + h_ * d))
        error = np.linalg.norm(exact - approx)
        # If errors are small, their ratios can falsely indicate order loss due to
        # floating point arithmetics.
        if error < tol:
            error = 0.0
        errors.append(error)

    errors = np.array(errors)

    h_ratios = h[1:] / h[:-1]

    # Safe division for error_ratios
    error_ratios = np.full_like(errors[1:], np.nan)
    mask = errors[:-1] > tol
    error_ratios[mask] = errors[1:][mask] / errors[:-1][mask]

    # Compute orders safely
    orders = np.full_like(error_ratios, np.inf)
    finite_mask = np.isfinite(error_ratios) & (error_ratios > tol)
    orders[finite_mask] = np.log(error_ratios[finite_mask]) / np.log(
        h_ratios[finite_mask]
    )

    if verbose:
        normalized_errors = errors / h**2
        print(
            f"{'h':>10} {'error':>12} {'error/h^2':>12} {'error_ratio':>12} "
            f"{'est_order':>10}"
        )
        print("-" * 60)
        for i in range(len(h)):
            if i == 0:
                print(
                    f"{h[i]:>10.2e} {errors[i]:>12.2e} {normalized_errors[i]:>12.2e} "
                    f"{'-':>12} {'-':>10}"
                )
            else:
                print(
                    f"{h[i]:>10.2e} {errors[i]:>12.2e} {normalized_errors[i]:>12.2e} "
                    f"{error_ratios[i - 1]:>12.2e} {orders[i - 1]:>10.2f}"
                )

    return orders


def assert_order_at_least(
    orders: np.ndarray, expected_order: float, tol: float = 0.1, err_msg: str = ""
) -> None:
    """Asserts that the average of the estimated orders are at least the expected order
    minus a tolerance.

    If orders are negative or nan, an error is raised.
    Order values of + infinity are treated as an exact approximation and treated as the
    expected order.

    """
    if np.any(orders < 0):
        raise ValueError("Estimated orders contain negative values.")
    if np.any(np.isnan(orders)):
        raise ValueError("Estimated orders contain NAN values")

    # If order all inf, we have an exact approximation.
    if not np.all(np.isinf(orders)):
        # Treat infinities as expected order
        orders[np.isinf(orders)] = expected_order
        order_avg = np.mean(orders)
        assert np.all(order_avg >= expected_order - tol), (
            f"Expected all orders to be at least {expected_order - tol}, "
            f"but got {order_avg}: {err_msg}"
        )


def get_polynomial_residual(r: float, c2: float, c1: float, c0: float) -> float:
    """Computes the residual of a normalized cubic polynomial.

    If ``r`` is indeed a root, than ``r**3 + c2*r**2 + c1*r + c0`` is zero.

    """
    return np.abs(r**3 + c2 * r**2 + c1 * r + c0)


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
        ((-1, -1, -2), np.array([2.0]), 1),
        ((-3, -3, -1), np.array([np.cbrt(4) + np.cbrt(2) + 1]), 1),
        ((1, -2, -2), np.array([np.sqrt(2), -np.sqrt(2), -1]), 3),
        ((4, 2, -4), np.array([-1 + np.sqrt(3), -1 - np.sqrt(3), -2]), 3),
        (  # (x-1)*(x-2)**2
            (-5, 8, -4),
            np.array(
                [
                    1.0,
                    2.0,
                ]
            ),
            2,
        ),
        (  # (x-1)*(x+2)**2
            (3, 0, -4),
            np.array(
                [
                    1.0,
                    -2.0,
                ]
            ),
            2,
        ),
        (  # (x-sqrt(2))**2
            (-3 * np.sqrt(2), 6, -2 * np.sqrt(2)),
            np.array([np.sqrt(2)]),
            0,
        ),
        (  # Peng-Robinson EoS critical point
            (
                B_CRIT - 1,
                A_CRIT - 2.0 * B_CRIT - 3.0 * B_CRIT**2,
                B_CRIT**3 + B_CRIT**2 - A_CRIT * B_CRIT,
            ),
            np.array([Z_CRIT]),
            0,
        ),
    ],
)
def test_known_root_case_calculations(
    coefficients: tuple[float, float, float],
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
    c2, c1, c0 = coefficients

    calculated_root_case = get_root_case(c2, c1, c0, tol)

    # Test the calculated root case.
    assert calculated_root_case == root_case

    # Custom computations are supposed to be returned sorted as well (ascending).
    solution = np.sort(solution)

    vals: np.ndarray

    match root_case:
        case 0:
            assert solution.size == 1
            vals = triple_root(c2)
        case 1:
            assert solution.size == 1
            vals = one_root(c2, c1, c0)
        case 2:
            assert solution.size == 2
            vals = two_roots(c2, c1, c0)
        case 3:
            assert solution.size == 3
            vals = three_roots(c2, c1, c0)
        case _:
            assert False, "Faulty test"

    # Test computed root.
    np.testing.assert_allclose(vals, solution, atol=tol, rtol=0.0)

    # Test that it is indeed a rood.
    # residual = vals**3 + c2 * vals**2 + c1 * vals + c0
    residual = get_polynomial_residual(vals, c2, c1, c0)
    np.testing.assert_allclose(residual, 0.0, atol=tol, rtol=0.0)

    # Test that the call to the general function returns the same result.
    genvals = calculate_roots(c2, c1, c0, tol)
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
