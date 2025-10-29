"""Module for testing compressibility factor computation which is based on the
solution of real cubic polynomials."""

from __future__ import annotations


import numpy as np
import pytest

# import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"

from porepy.compositional.peng_robinson.compressibility_factor import (
    W_sub,
    get_compressibility_factor,
    _smooth_3root_region,
    _smooth_scl_transition,
    c_from_AB,
    critical_line,
    dc_from_AB,
    dW_sub,
    get_compressibility_factor_derivatives,
    is_extended_root,
    A_CRIT,
    B_CRIT,
    Z_CRIT,
)

from tests.compositional.peng_robinson.test_cubic_polynomial import (
    get_EOC_taylor,
    assert_order_at_least,
    get_polynomial_residual,
)


@pytest.fixture(scope="module")
def AB_refinement() -> int:
    """Refinement for AB space for testing."""
    return 100


@pytest.fixture(scope="module")
def A_range(AB_refinement) -> np.ndarray:
    """Range of tested cohesion values."""
    return np.linspace(1e-3, 1.0, AB_refinement, endpoint=True)


@pytest.fixture(scope="module")
def B_range(AB_refinement) -> np.ndarray:
    """Range of tested covolume values."""
    return np.linspace(1e-3, 0.206813 + 0.05, AB_refinement, endpoint=True)


@pytest.mark.parametrize("gaslike", [True, False])
def test_critical_point(gaslike: bool) -> None:
    """Tests the critical values of cohesion and covolume.

    They should lead to a triple root with the value of the critical compressibility
    factor.

    """
    tol = 1e-14
    s = 0.0

    Zval = get_compressibility_factor(A_CRIT, B_CRIT, gaslike, tol, s)
    c = c_from_AB(A_CRIT, B_CRIT)

    np.testing.assert_allclose(Zval, Z_CRIT, rtol=0.0, atol=tol)
    np.testing.assert_allclose(
        get_polynomial_residual(Z_CRIT, c), 0.0, rtol=0.0, atol=tol
    )


def test_root_computation_in_AB_space(A_range: np.ndarray, B_range: np.ndarray) -> None:
    """Tests root computation in the cohesion-covolume space and asserts that
    non-extended roots are actual roots."""

    tol = 1e-14

    Avec, Bvec = (v.flatten() for v in np.meshgrid(A_range, B_range))

    for A, B in zip(Avec, Bvec):
        err_msg = f" (A = {A}; B = {B}) "
        c = c_from_AB(A, B)
        Zg = get_compressibility_factor(A, B, True, tol, 0.0)
        Zl = get_compressibility_factor(A, B, False, tol, 0.0)

        assert Zl <= Zg, f"Liquid root must be smaller or equal gas root. {err_msg}"

        # If the gaslike root is not extended, it must be a real root
        if not is_extended_root(A, B, True, tol):
            assert get_polynomial_residual(Zg, c) < tol, (
                f"Real gas compressibility factor is not real root. {err_msg}"
            )
        # Analogous for liquidlike root.
        if not is_extended_root(A, B, False, tol):
            assert get_polynomial_residual(Zl, c) < tol, (
                f"Real liquid compressibility factor is not real root. {err_msg}"
            )


@pytest.mark.parametrize("gaslike", [True, False])
@pytest.mark.parametrize(
    "d",
    [  # Directions are choses such that they mimic the skewedness of the 3-root
        # region. And we rotate by 90 deg to cover roughly all directions.
        np.array([0.05, 0.01]),
        np.array([-0.05, -0.01]),
        np.array([-0.05, 0.01]),
        np.array([0.05, -0.01]),
    ],
)
@pytest.mark.parametrize(
    "x0",
    [
        # Sub-critical liquid area.
        np.array([0.5, 0.02]),
        # Sub-critical gas area.
        np.array([0.3, critical_line(0.3) - 0.001]),
        # 2-phase area.
        np.array([0.1, 0.01]),
        np.array([0.2, 0.02]),
        np.array([0.3, 0.04]),
        # Super-critical liquid area.
        np.array([0.7, 0.09]),
        np.array([0.9, B_CRIT]),
        # Super-critical gas area
        np.array([0.36, 0.065]),
    ],
)
def test_root_derivative_computation(
    gaslike: bool, d: np.ndarray, x0: np.ndarray
) -> None:
    """Tests the computation of root derivatives around specified points.

    Points are chosen such that they are in areas usually encountered in the
    computation.

    The expected order should be 2, but it can be smaller in areas where extended roots
    are smoothed.

    """
    tol = 1e-14

    def func(*x):
        return get_compressibility_factor(*x, gaslike, tol, 0.0)

    def dfunc(*x):
        return get_compressibility_factor_derivatives(*x, gaslike, tol, 0.0)

    orders = get_EOC_taylor(func, dfunc, x0, d, h=np.logspace(-1, -10, 10))
    # NOTE: There is a lot of trickery possible to make this test pass, but we try to
    # be fair. The changes in A,B of order 1e-3 are significant in the sense that it can
    # result in another root case region, hence we ignore the first 2 entries.
    # And in terms of tolerance, treating 1.995 as 2 is fair enough considering the
    # computations involved (considering also that the method uses the average order).
    assert_order_at_least(orders[2:], 2, tol=5e-3, err_msg=f"{x0}")


@pytest.mark.parametrize(
    "d",
    [  # Directions are choses such that they mimic the skewedness of the 3-root
        # region. And we rotate by 90 deg to cover roughly all directions.
        np.array([0.05, 0.01]),
        np.array([-0.05, -0.01]),
        np.array([-0.05, 0.01]),
        np.array([0.05, -0.01]),
    ],
)
@pytest.mark.parametrize(
    ["x0", "gaslike", "expected_order"],
    [
        # Sub-critical 3-root area where liquid is smoothed.
        (np.array([0.09, 0.007]), True, 2.0),
        (np.array([0.09, 0.007]), False, 1.0),
        (np.array([0.19, 0.0095]), True, 2.0),
        (np.array([0.19, 0.0095]), False, 0.98),
        (np.array([0.3, 0.045]), True, 2.0),
        (np.array([0.3, 0.045]), False, 1.0),
        # Sub-critical 3-root area where gas is smoothed.
        (np.array([0.264, 0.01]), True, 0.98),
        (np.array([0.264, 0.01]), False, 2),
        (np.array([0.344, 0.045]), True, 0.98),
        (np.array([0.344, 0.045]), False, 2),
        # Super-critical liquid area, gas root is smoothed.
        (np.array([0.7, 0.12]), True, 1.0),
        (np.array([0.7, 0.12]), False, 2.0),
        (np.array([0.5, 0.08]), True, 1.0),
        (np.array([0.5, 0.08]), False, 2.0),  # 13
        # Super-critical gas area, liquid root is smoothed.
        (np.array([0.5, 0.11]), True, 2),
        (np.array([0.5, 0.11]), False, 0.98),
        (np.array([0.4, B_CRIT]), True, 2),
        (np.array([0.4, B_CRIT]), False, 1.0),
    ],
)
def test_root_derivative_computation_smoothed(
    gaslike: bool, d: np.ndarray, x0: np.ndarray, expected_order: float | int
) -> None:
    """Analogous to the non-smooth test, but with different parametrization as one root
    liquid or gas, can be smoothed leading to a reduced order of the approximation."""
    tol = 1e-14

    # NOTE we also apply smoothing in the physical 2-phase region/3-root region
    def func(*x):
        return get_compressibility_factor(*x, gaslike, tol, 0.25)

    def dfunc(*x):
        return get_compressibility_factor_derivatives(*x, gaslike, tol, 0.25)

    rc = is_extended_root(*x0, gaslike, tol)

    orders = get_EOC_taylor(func, dfunc, x0, d, h=np.logspace(-1, -10, 10))
    assert_order_at_least(orders[2:], expected_order, tol=1e-2, err_msg=f"{x0}")


@pytest.mark.parametrize(
    "d", [np.array([1.0, 0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
)
def test_extended_root_derivative_function(d: np.ndarray) -> None:
    """Tests the derivative computation of the extended root. Taylorexpansion must
    converge with second order.

    This must hold for all points, hence tested at a random point with a quadratic
    proxy function for the compressibility factor.

    """

    def func(*args):
        Z = sum(a**2 for a in args)
        return W_sub(float(Z), float(args[-1]))

    def dfunc(*args):
        dz = np.array([2 * a for a in args]).astype(float)
        return dW_sub(dz)

    x0 = np.random.rand(2)
    orders = get_EOC_taylor(func, dfunc, x0, d, h=np.logspace(0, -10, 11))
    assert_order_at_least(orders, 2.0, tol=1e-3, err_msg=f"{x0}")


@pytest.mark.parametrize(
    "d", [np.array([1.0, 0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
)
def test_derivatives_of_polynom_coeffs_wrt_AB(d: np.ndarray) -> None:
    """Tests the computation of derivatives of the coefficients of the Peng-Robinson-EOS
    with respect to cohesion and covolume."""
    x0 = np.random.rand(2)
    orders = get_EOC_taylor(c_from_AB, dc_from_AB, x0, d, h=np.logspace(0, -10, 11))
    assert_order_at_least(orders, 2.0, tol=1e-3, err_msg=f"{x0}")


def test_supercritical_smoothing_function() -> None:
    """"""


def test_3root_smoothing_function() -> None:
    """"""


def test_limitcase_zero_cohesion(B_range: np.ndarray) -> None:
    """"""


def test_limitcase_zero_covolume(A_range: np.ndarray) -> None:
    """"""
