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
    COVOLUME_LIMIT,
)
from porepy.compositional.peng_robinson.cubic_polynomial import (
    get_root_case,
    calculate_roots,
)

from tests.compositional.peng_robinson.test_cubic_polynomial import (
    get_EOC_taylor,
    assert_order_at_least,
    get_polynomial_residual,
)


def _err_msg(A: float, B: float) -> str:
    return f"(A, B) = ({A}, {B})"


def assert_roots_correctly_sized(A: float, B: float, tol: float = 1e-14) -> None:
    """Asserts that it always holds ``B < Zl <= Zg``."""

    Zg = get_compressibility_factor(A, B, True, tol, 0.0)
    Zl = get_compressibility_factor(A, B, False, tol, 0.0)

    assert Zl <= Zg, f"Liquid root must be smaller or equal gas root. {_err_msg(A, B)}"
    assert B < Zl, (
        f"Liquid root must be greater than physical bound B. {_err_msg(A, B)}"
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
        err_msg = _err_msg(A, B)
        c = c_from_AB(A, B)

        assert_roots_correctly_sized(A, B, tol=tol)

        # If the gaslike root is not extended, it must be a real root
        if not is_extended_root(A, B, True, tol):
            assert (
                get_polynomial_residual(
                    get_compressibility_factor(A, B, True, tol, 0.0), c
                )
                < tol
            ), f"Real gas compressibility factor is not real root. {err_msg}"
        # Analogous for liquidlike root.
        if not is_extended_root(A, B, False, tol):
            assert (
                get_polynomial_residual(
                    get_compressibility_factor(A, B, False, tol, 0.0), c
                )
                < tol
            ), f"Real liquid compressibility factor is not real root. {err_msg}"


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
    assert_order_at_least(orders[2:], 2, tol=5e-3, err_msg=_err_msg(*x0))


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

    orders = get_EOC_taylor(func, dfunc, x0, d, h=np.logspace(-1, -10, 10))
    assert_order_at_least(orders[2:], expected_order, tol=1e-2, err_msg=_err_msg(*x0))


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
    assert_order_at_least(orders, 2.0, tol=1e-3, err_msg=_err_msg(*x0))


@pytest.mark.parametrize(
    "d", [np.array([1.0, 0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
)
def test_derivatives_of_polynom_coeffs_wrt_AB(d: np.ndarray) -> None:
    """Tests the computation of derivatives of the coefficients of the Peng-Robinson-EOS
    with respect to cohesion and covolume."""
    x0 = np.random.rand(2)
    orders = get_EOC_taylor(c_from_AB, dc_from_AB, x0, d, h=np.logspace(0, -10, 11))
    assert_order_at_least(orders, 2.0, tol=1e-3, err_msg=_err_msg(*x0))


def test_supercritical_smoothing_function() -> None:
    """"""


def test_3root_smoothing_function() -> None:
    """"""


@pytest.mark.parametrize(
    "d", [np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([1.0, -1.0])]
)
@pytest.mark.parametrize(
    ["gaslike", "expected_order"],
    [
        (True, 2.0),
        (False, 1.0),
    ],
)
def test_limitcase_zero_cohesion(
    d: np.ndarray, gaslike: bool, expected_order: float, B_range: np.ndarray
) -> None:
    """The case of zero cohesion is part of the nonphysical 3-root area where the
    smallest real root is smaller than the physical bound B.

    Test for proper extension and computation, as well as order of Taylor expansion.
    Liquid root is expteced to be extended and loosing order of convergence.

    """

    tol = 1e-14

    for B in B_range:
        x0 = np.array([0.0, B])
        err_msg = _err_msg(*x0)
        c = c_from_AB(*x0)
        rc = get_root_case(*c, tol)
        assert rc == 3, f"Expecting 3-root-case: {err_msg}"

        # Testing approximation
        def func(*x):
            return get_compressibility_factor(*x, gaslike, tol, 0.0)

        def dfunc(*x):
            return get_compressibility_factor_derivatives(*x, gaslike, tol, 0.0)

        assert_roots_correctly_sized(*x0, tol=tol)

        is_extended = is_extended_root(*x0, gaslike, tol)

        # Should be real root
        if gaslike:
            assert not is_extended, f"Expecting gas root to be real: {err_msg}"
            assert get_polynomial_residual(func(*x0), c) <= tol, (
                "Gas root not real root."
            )
        else:
            assert is_extended, f"Expecting liquid root to be extended: {err_msg}"

        orders = get_EOC_taylor(func, dfunc, x0, d, h=np.logspace(-1, -10, 10))
        assert_order_at_least(orders[2:], expected_order, tol=1e-2, err_msg=err_msg)


@pytest.mark.parametrize(
    "d", [np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([-1.0, 1.0])]
)
@pytest.mark.parametrize(
    ["gaslike", "expected_order"],
    [
        (True, 1.97),
        (False, 1.0),
    ],
)
def test_limitcase_zero_covolume(
    d: np.ndarray, gaslike: bool, expected_order: float, A_range: np.ndarray
) -> None:
    """Limit case with B = 0, where the smallest real root goes too zero.

    In the 3-root are, A in (0, 0.25), the smallest (liquid-like) root goes to zero.
    In the 1-root area, A > 0.25, the 1 real root, which is liquid, goes to zero.

    In both cases, the liquid like root needs to be extended.
    Currently it is approximated using lower bound on B which is positive but tiny,
    leading to an order loss.

    The roots goes to zero, but it is actually always bigger than B, until B=0, where
    they become equal.

    """

    tol = 1e-14
    # Special value: See test_limitcase_zero_covolume_liquid_saturated
    A_L = 0.25

    # Then we test the rest of the line.
    for A in A_range:
        x0 = np.array([A, 0.0])
        err_msg = _err_msg(*x0)
        c = c_from_AB(*x0)
        rc = get_root_case(*c, tol)
        if A < A_L:
            assert rc == 3, f"Expecting 3-root-case: {err_msg}"
        elif A > A_L:
            assert rc == 1, f"Expecting 1-root-case: {err_msg}"
        else:
            # Skip this case, see test_limitcase_zero_covolume_liquid_saturated
            continue

        # Testing approximation
        def func(*x):
            return get_compressibility_factor(*x, gaslike, tol, 0.0)

        def dfunc(*x):
            return get_compressibility_factor_derivatives(*x, gaslike, tol, 0.0)

        assert_roots_correctly_sized(*x0, tol=tol)

        is_extended = is_extended_root(*x0, gaslike, tol)

        if not gaslike:
            assert is_extended, f"Expecting liquid root to be extended: {err_msg}"

        if rc == 3:
            if gaslike:
                assert not is_extended, f"Expecting gas roots to be real: {err_msg}"
                assert get_polynomial_residual(func(*x0), c) <= tol, (
                    "Gas root not real root."
                )
            # Since we use lower bound instead of zero, the extended root should be
            # pretty close to zero.
            else:
                assert get_polynomial_residual(func(*x0), c) <= COVOLUME_LIMIT, (
                    "Extended liquid root too far away."
                )

        # Liquid-like root is extended but approximated with a close enough value.
        elif rc == 1:
            assert is_extended, f"Expecting root to be extended: {err_msg}"
            if not gaslike:
                assert COVOLUME_LIMIT == 1e-5, "COVOLUME_LIMIT expected to be 1e-5."
                # Because numerics sometimes does not care.
                assert get_polynomial_residual(func(*x0), c) <= 2e-5, (
                    "Liquid root not real root."
                )

        orders = get_EOC_taylor(func, dfunc, x0, d, h=np.logspace(-1, -10, 10))
        assert_order_at_least(
            orders,
            expected_order,
            tol=1e-1,
            err_msg=err_msg,
            # Liquid like root approximations are only asymptotic near liquid-saturated
            # line.
            asymptotic=None if gaslike else 5,
        )


@pytest.mark.parametrize(
    "d",
    [
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([-1.0, 1.0]),
        # Slope of liquid-saturated line.
        np.array([1.0, 2.0]),
    ],
)
@pytest.mark.parametrize(
    ["gaslike", "expected_order"],
    [
        # NOTE: Gas is about to disappear *and* we are in a limit case. This is likely
        # one of the points where things go haywire with cubic EoS.
        (True, 0.5),
        (False, 1.0),
    ],
)
def test_limitcase_zero_covolume_liquid_saturated(
    d: np.ndarray, gaslike: bool, expected_order: float
) -> None:
    """Test the special point (A, B) = (0.25, 0), which is the lower end of the liquid-
    saturated 2-phase regime, spanning from this  point to the critical point.

    We also know it's slope (2) there from Ben Gharbia 2021.

    """
    tol = 1e-14
    A_L = 0.25
    x0 = np.array([A_L, 0.0])
    err_msg = _err_msg(*x0)
    c = c_from_AB(*x0)
    rc = get_root_case(*c, tol)
    assert rc == 2, f"Expecting 2-root-case: {err_msg}"

    # The raw roots have special values.
    roots = calculate_roots(*c, eps=tol)
    np.testing.assert_allclose(roots, np.array([0.0, 0.5]), rtol=0.0, atol=tol)

    def func(*x):
        return get_compressibility_factor(*x, gaslike, tol, 0.0)

    def dfunc(*x):
        return get_compressibility_factor_derivatives(*x, gaslike, tol, 0.0)

    assert_roots_correctly_sized(*x0, tol=tol)
    is_extended = is_extended_root(*x0, gaslike, tol)
    if gaslike:
        assert not is_extended, (
            f"Expecting gas root to be real at liquid-saturated border: {err_msg}"
        )
        assert get_polynomial_residual(func(*x0), c) <= tol, (
            f"{'Gas' if gaslike else 'Liquid'} root not real root."
        )
    else:
        assert is_extended, f"Expecting liquid root to be bound: {err_msg}"
        assert COVOLUME_LIMIT == 1e-5, "COVOLUME_LIMIT value changed."
        assert get_polynomial_residual(func(*x0), c) <= 2e-5, (
            f"{'Gas' if gaslike else 'Liquid'} root not real root."
        )

    orders = get_EOC_taylor(func, dfunc, x0, d, h=np.logspace(-1, -10, 10))
    assert_order_at_least(
        orders,
        expected_order,
        tol=1e-2,
        err_msg=err_msg,
        asymptotic=None if gaslike else 5,
    )


@pytest.mark.parametrize("gaslike", [True, False])
@pytest.mark.parametrize(
    ["d", "expected_order_gas", "expected_order_liquid"],
    [
        (np.array([A_CRIT, critical_line(A_CRIT)]), 1.89, 1.0),
        (np.array([1.0, 1e3]), 1.89, 1.0),
        (np.array([1e3, 1.0]), 1.89, None),
        (np.array([1.0, 0.0]), 1.89, None),
        (np.array([0.0, 1.0]), 1.89, 1.0),
    ],
)
def test_limitcase_zero_cohesion_and_covolume(
    gaslike: bool,
    d: np.ndarray,
    expected_order_gas: float | None,
    expected_order_liquid: float | None,
) -> None:
    """Test evaluation and derivatives of the point (A, B) = (0, 0).

    This is a known 2-real-root case, so no extension procedure.
    It is also the point where the critical line and the gas-saturated 2-phase line
    start (both end at critical point).

    It also borders the non-physical 3-root area where the liquid-like root is capped
    to be around B and its derivatives are set to constant. Taylor approximation from
    that direction will fail.

    """
    tol = 1e-14
    x0 = np.zeros(2, dtype=float)
    err_msg = _err_msg(*x0)
    c = c_from_AB(*x0)
    rc = get_root_case(*c, tol)
    assert rc == 2, f"Expecting 2-root-case: {err_msg}"

    # The raw roots have special values.
    roots = calculate_roots(*c, eps=tol)
    np.testing.assert_allclose(roots, np.array([0.0, 1.0]), rtol=0.0, atol=tol)

    assert_roots_correctly_sized(0.0, 0.0, tol=tol)

    # Testing approximation
    def func(*x):
        return get_compressibility_factor(*x, gaslike, tol, 0.0)

    def dfunc(*x):
        return get_compressibility_factor_derivatives(*x, gaslike, tol, 0.0)

    is_extended = is_extended_root(*x0, gaslike, tol)

    if gaslike:
        assert not is_extended, f"Expecting gas root to be real: {err_msg}"
    else:
        assert is_extended, f"Expecting liqud root to be extended: {err_msg}"

    orders = get_EOC_taylor(func, dfunc, x0, d, h=np.logspace(-1, -10, 10))

    expected_order = expected_order_gas if gaslike else expected_order_liquid
    if isinstance(expected_order, (int, float)):
        assert_order_at_least(orders, expected_order, tol=1e-2, err_msg=err_msg)
    elif expected_order is None:
        assert np.any(orders < 0), (
            "Expecting negative orders where divergence indicated."
        )
