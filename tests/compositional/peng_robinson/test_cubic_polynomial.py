"""Testing module for functionality regarding general solutions of cubic polynomials
dependent on coefficients and their derivatives."""

from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np
import pytest

from porepy.applications.test_utils.derivative_testing import (
    assert_order_at_least,
    get_EOC_taylor,
)
from porepy.compositional.peng_robinson.compressibility_factor import (
    A_CRIT,
    B_CRIT,
    Z_CRIT,
    c_from_AB,
    critical_line,
    is_supercritical,
)
from porepy.compositional.peng_robinson.cubic_polynomial import (
    calculate_root_derivatives,
    calculate_roots,
    d_one_root,
    d_three_roots,
    d_triple_root,
    d_two_roots,
    get_root_case,
    one_root,
    r_from_c,
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
    c = np.random.rand(3)

    while np.abs(r_from_c(c)[0]) < 0.1:
        c[1] = np.random.rand()

    return c.astype(np.float64)


@pytest.mark.parametrize(
    ["A", "B", "is_sc", "rc"],
    [
        ## Tests around critical point, which is a triple root.
        (A_CRIT, B_CRIT, True, 0),
        # Supercritical points, must be cases with 1 real root, 2 complex-conjugated.
        # We operate in general with a float64-precision and due to floating point
        # arithmetics we can determin the root case down to this precision safely
        # without introducing errors.
        (A_CRIT, B_CRIT + 1e-14, True, 1),
        (A_CRIT + 1e-14, B_CRIT, True, 1),
        (A_CRIT - 1e-14, B_CRIT, True, 1),
        # First detection of 3-root region below critical point.
        (np.float64(0.4545650246634926), np.float64(0.07717075021475177), False, 3),
        (np.float64(0.4545650246634926), np.float64(0.07712476321475177), False, 3),
        # These points lead to numerically unstable discriminant checks due to floating
        # point cancelation. They can lead falsely into the 3 root case, where the
        # trigonometric formulae lead to wrong values and the polynomial residual is too
        # high.
        (np.float64(0.4572355289213818), np.float64(0.07781278127812781), True, 1),
        (np.float64(0.4572393501308955), np.float64(0.07779699945959057), True, 1),
        (np.float64(0.4572355289213818), np.float64(0.0777960606060606), False, 1),
        ## Tests around point 00.
        (np.float64(0.0), np.float64(0.0), True, 2),
        # Points too close where degeneracy is detected. With eps=1e-15 we cannot
        # get any closer when using float64
        (np.float64(1e-8), np.float64(0.0), False, 2),
        (np.float64(0.0), np.float64(1e-9), True, 2),
        (np.float64(1e-8), critical_line(np.float64(1e-8)), True, 2),
        # Further away the root cases should be correctly resolved.
        # More precise is not possible due to float64.
        (np.float64(1e-7), np.float64(0.0), False, 3),
        (np.float64(0.0), np.float64(1e-8), True, 3),
        (np.float64(1e-7), critical_line(np.float64(1e-7)), True, 1),
        # Numerically challenging points due to cancelation.
        (
            np.float64(9.858323132194671e-08),
            np.float64(1.677338672193383e-08),
            True,
            1,
        ),
        (
            np.float64(0.0007071067811865475),
            np.float64(0.0028768629157831494),
            True,
            3,
        ),
    ],
)
def test_floating_point_stability_for_degenerate_discriminant(
    A: float,
    B: float,
    is_sc: bool,
    rc: int,
) -> None:
    """Tests the stability of the root computations and root case computations in terms
    of floating point operations.

    The framework is build for float64, hence a precision of 1e-15 is what we consider
    zero. Differences above that should be detected by the framework.

    This test is performed around the critical point of the Peng-Robinson EoS,
    which is a known triple point with all root scenarios converging towards it, i.e.
    there are boarders between 1-2-3 distinct real root cases, which end there and the
    calculation of the discriminant is very sensitive.
    The 2-root case is never heat due to floating point arithmetics, but there the
    3-root case delivers a close enough approximation (yields 2 roots with difference
    on order of defined eps).

    It is also performed around the point AB=(0,0), which is a known 2-root case but
    here the floating point issues around a degenerate discriminant are more serious.
    We loose some precision because we fall logically into the trigonometric case
    computations for 3 roots (Trying to hit a point with limited number of floating
    points...)

    Important:
        If changes break this test, something is not alright. Don't go down this path.

    Parameters:
        A: Cohesion value.
        B: Covolume value.
        is_sc: Flag if the point is expected to be super-critical.
        rc: Expected root case.

    """
    eps = 1e-15
    c = c_from_AB(A, B)
    is_sc_comp = is_supercritical(A, B)
    rc_comp = get_root_case(c, eps)
    assert is_sc_comp == is_sc, "Wrong supercritical indication computed."
    assert rc_comp == rc, "Wrong root case computed"

    r = calculate_roots(c, eps)
    assert np.all(get_polynomial_residual(r, c) < eps), "Not actual root."


def test_numerical_precision():
    """Sweeps accross the interval [-1, 1] for all coefficients and checks that the
    computed roots satisfy the polynomial equation up to a specified precision."""
    eps = 1e-14
    N = 101
    c0 = np.linspace(-1, 1, N, endpoint=True)
    c1 = np.linspace(-1, 1, N, endpoint=True)
    c2 = np.linspace(-1, 1, N, endpoint=True)

    for c0_, c1_, c2_ in product(c0, c1, c2):
        c = np.array([c0_, c1_, c2_])
        r = calculate_roots(c, eps)
        assert np.all(get_polynomial_residual(r, c) < eps), f"Precision failure: {c}"


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
            c_from_AB(A_CRIT, B_CRIT),
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

    calculated_root_case = get_root_case(coefficients, tol)

    # Test the calculated root case.
    assert calculated_root_case == root_case

    # Custom computations are supposed to be returned sorted as well (ascending).
    solution = np.sort(solution)

    vals: np.ndarray

    match root_case:
        case 0:
            assert solution.size == 1
            vals = triple_root(coefficients.astype(np.float64))
        case 1:
            assert solution.size == 1
            vals = one_root(coefficients.astype(np.float64))
        case 2:
            assert solution.size == 2
            vals = two_roots(coefficients.astype(np.float64))
        case 3:
            assert solution.size == 3
            vals = three_roots(coefficients.astype(np.float64))
        case _:
            assert False, "Faulty test"

    # Test computed root.
    np.testing.assert_allclose(vals, solution, atol=tol, rtol=0.0)

    # Test that it is indeed a rood.
    residual = get_polynomial_residual(vals, coefficients)
    np.testing.assert_allclose(residual, 0.0, atol=tol, rtol=0.0)

    # Test that the call to the general function returns the same result.
    genvals = calculate_roots(coefficients, tol)
    np.testing.assert_allclose(genvals, solution, atol=tol, rtol=0.0)


def test_triple_root_derivatives() -> None:
    """Specialized test for the triple root derivative, which is constant and simple."""

    h_values = np.logspace(-1, -10, 10)
    X0 = np.random.rand(3)

    r = triple_root(X0)
    dr = d_triple_root(X0)
    approximations = [(triple_root(X0 + h) - r) / h for h in h_values]
    np.testing.assert_allclose(np.array(approximations), -1 / 3, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(
        dr, np.array([-1 / 3, 0.0, 0.0]).reshape((1, 3)), atol=1e-14, rtol=0.0
    )


@pytest.mark.parametrize(
    "d",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ],
)
@pytest.mark.parametrize(
    "x0",
    [
        c_from_AB(A_CRIT, B_CRIT),
        c_from_AB(0.0, 0.0),
    ],
)
@pytest.mark.parametrize(
    ["func", "dfunc"],
    [
        (one_root, d_one_root),
        (two_roots, d_two_roots),
    ],
)
def test_single_double_root_derivatives_around_triple_point(
    func: Callable[..., np.ndarray],
    dfunc: Callable[..., np.ndarray],
    x0: np.ndarray,
    d: np.ndarray,
) -> None:
    """The computation of derivatives for triple and double root functions loses 1
    order of convergence when because these points are usually limit cases where the
    formulas switch.

    """

    def _func(x):
        return func(np.array(x).astype(np.float64))

    def _dfunc(x):
        return dfunc(np.array(x).astype(np.float64))

    orders = get_EOC_taylor(_func, _dfunc, x0, d, h=np.logspace(-1, -10, 10))
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
    "d",
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
    d: np.ndarray,
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
    orders = get_EOC_taylor(func, dfunc, x0, d, np.logspace(-2, -11, 10))
    assert_order_at_least(orders, 2.0, tol=5e-2)


def test_general_root_derivative() -> None:
    """Tests the generic derivative calculation for roots by using two random
    coefficient sets with fixed seeds, one where order 2 is reached and one where only
    order 1 is reached."""
    np.random.seed(42)
    eps = 1e-14

    x0 = np.random.rand(3)
    d = np.random.rand(3)
    h = np.logspace(-1, -10, 10)

    def func(c: np.ndarray) -> np.ndarray:
        return calculate_roots(c, eps)

    def dfunc(c: np.ndarray) -> np.ndarray:
        return calculate_root_derivatives(c, eps)

    orders = get_EOC_taylor(func, dfunc, x0, d, h)
    assert_order_at_least(orders, 2.0, tol=1e-2)

    np.random.seed(2)

    x0 = np.random.rand(3)
    d = np.random.rand(3)

    orders = get_EOC_taylor(func, dfunc, x0, d, h)
    assert_order_at_least(orders, 1.0, tol=1e-2)
