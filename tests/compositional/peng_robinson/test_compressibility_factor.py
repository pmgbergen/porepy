"""Module for testing compressibility factor computation which is based on the
solution of real cubic polynomials."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from porepy.applications.test_utils.derivative_testing import (
    assert_order_at_least,
    get_EOC_taylor,
)
from porepy.compositional.peng_robinson.compressibility_factor import (
    A_CRIT,
    B_CRIT,
    COVOLUME_LIMIT,
    CRITICAL_SLOPE,
    Z_CRIT,
    _smooth_3root_region,
    c_from_AB,
    dc_from_AB,
    extended_factor,
    extended_factor_derivatives,
    get_compressibility_factor,
    get_compressibility_factor_derivatives,
    is_extended_factor,
)
from porepy.compositional.peng_robinson.cubic_polynomial import (
    calculate_roots,
    get_root_case,
)
from tests.compositional.peng_robinson import calculate_expected_order
from tests.compositional.peng_robinson.test_cubic_polynomial import (
    get_polynomial_residual,
)


def _err_msg(A: float, B: float) -> str:
    return f"(A, B) = ({A}, {B})"


def assert_roots_correctly_sized(A: float, B: float, tol: float = 1e-14) -> None:
    """Asserts that it always holds ``B < Zl <= Zg``."""

    Zg = get_compressibility_factor(A, B, True, tol)
    Zl = get_compressibility_factor(A, B, False, tol)

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
    return np.linspace(0.0, 1.0, AB_refinement, endpoint=True)


@pytest.fixture(scope="module")
def B_range(AB_refinement) -> np.ndarray:
    """Range of tested covolume values."""
    return np.linspace(0.0, 0.23, AB_refinement, endpoint=True)


@pytest.mark.parametrize("gaslike", [True, False])
def test_critical_point(gaslike: bool) -> None:
    """Tests the critical values of cohesion and covolume.

    They should lead to a triple root with the value of the critical compressibility
    factor.

    """
    tol = 1e-14

    Zval = get_compressibility_factor(A_CRIT, B_CRIT, gaslike, tol)
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
        if not is_extended_factor(A, B, True, tol):
            assert (
                get_polynomial_residual(get_compressibility_factor(A, B, True, tol), c)
                < 2 * COVOLUME_LIMIT
                if B < COVOLUME_LIMIT
                else tol
            ), f"Real gas compressibility factor is not real root. {err_msg}"
        # Analogous for liquidlike root.
        if not is_extended_factor(A, B, False, tol):
            assert (
                get_polynomial_residual(get_compressibility_factor(A, B, False, tol), c)
                < 2 * COVOLUME_LIMIT
                if B < COVOLUME_LIMIT
                else tol
            ), f"Real liquid compressibility factor is not real root. {err_msg}"


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
        np.array([0.3, CRITICAL_SLOPE * 0.3 - 0.001]),
        # 2-phase area.
        np.array([0.1, 0.01]),
        np.array([0.2, 0.02]),
        np.array([0.3, 0.04]),
        # NOTE Expecting order loss in super-critical region
        # Super-critical liquid area.
        np.array([0.7, 0.09]),
        np.array([0.9, B_CRIT]),
        # Super-critical gas area
        np.array([0.36, 0.065]),
    ],
)
@pytest.mark.parametrize("gaslike", [True, False])
@pytest.mark.parametrize("sm", [0.0, 1e-4])
@pytest.mark.parametrize("sc_bw", [0.0, 1e-3])
def test_root_derivative_computation(
    sm: float,
    sc_bw: float,
    gaslike: bool,
    d: np.ndarray,
    x0: np.ndarray,
) -> None:
    """Tests the computation of root derivatives around specified points.

    Points are chosen such that they are in areas usually encountered in the
    computation.

    The expected order should be 2, but it can be smaller in areas where extended roots
    are smoothed.

    """
    tol = 1e-14

    def func(x):
        return get_compressibility_factor(*x, gaslike, tol)

    def dfunc(x):
        return get_compressibility_factor_derivatives(
            *x, gaslike, tol, sm, 1e-5, sc_bw, 1.0
        )

    orders = get_EOC_taylor(func, dfunc, x0, d, np.logspace(-1, -10, 10))
    # NOTE: There is a lot of trickery possible to make this test pass, but we try to
    # be fair. The changes in A,B of order 1e-3 are significant in the sense that it can
    # result in another root case region, hence we ignore the first 2 entries.
    # And in terms of tolerance, treating 1.995 as 2 is fair enough considering the
    # computations involved (considering also that the method uses the average order).
    expected_order = calculate_expected_order(gaslike, tol, sm=sm, sc_bw=sc_bw, AB=x0)
    # NOTE For the horizontal B_CRIT there occurs a jump when approximating from below
    # Total loss of order if not smoothed.
    if sc_bw == 0.0 and x0[1] == B_CRIT and d[1] < 0:
        expected_order = 0.0
    assert_order_at_least(
        orders, expected_order, tol=1e-2, err_msg=_err_msg(*x0), asymptotic=6
    )


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
    ["x0", "gaslike"],
    [
        # Sub-critical 3-root area where liquid is smoothed.
        (np.array([0.09, 0.007]), True),
        (np.array([0.09, 0.007]), False),
        (np.array([0.19, 0.0095]), True),
        (np.array([0.19, 0.0095]), False),
        (np.array([0.3, 0.045]), True),
        (np.array([0.3, 0.045]), False),
        # Sub-critical 3-root area where gas is smoothed.
        (np.array([0.264, 0.01]), True),
        (np.array([0.264, 0.01]), False),
        (np.array([0.344, 0.045]), True),
        (np.array([0.344, 0.045]), False),
        # Super-critical liquid area, gas root is smoothed.
        (np.array([0.7, 0.12]), True),
        (np.array([0.7, 0.12]), False),
        (np.array([0.5, 0.08]), True),
        (np.array([0.5, 0.08]), False),
        # Super-critical gas area, liquid root is smoothed.
        (np.array([0.5, 0.11]), True),
        (np.array([0.5, 0.11]), False),
        (np.array([0.4, B_CRIT]), True),
        (np.array([0.4, B_CRIT]), False),
    ],
)
@pytest.mark.parametrize("sm", [0.0, 1e-4, 0.25])
@pytest.mark.parametrize("sc_bw", [0.0, 1e-3])
def test_root_derivative_computation_smoothed(
    sm: float,
    sc_bw: float,
    gaslike: bool,
    d: np.ndarray,
    x0: np.ndarray,
) -> None:
    """Analogous to the non-smooth test, but with different parametrization as one root
    liquid or gas, can be smoothed leading to a reduced order of the approximation."""
    tol = 1e-14

    # NOTE we also apply smoothing in the physical 2-phase region/3-root region
    def func(x):
        return get_compressibility_factor(*x, gaslike, tol)

    def dfunc(x):
        return get_compressibility_factor_derivatives(
            *x, gaslike, tol, sm, 1e-4, sc_bw, 1.0
        )

    orders = get_EOC_taylor(func, dfunc, x0, d, np.logspace(-1, -10, 10))
    expected_order = calculate_expected_order(gaslike, tol, sm=sm, sc_bw=sc_bw, AB=x0)
    assert_order_at_least(
        orders, expected_order, tol=1e-2, err_msg=_err_msg(*x0), asymptotic=6
    )


@pytest.mark.parametrize(
    "d", [np.array([1.0, 0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
)
def test_extended_root_derivative_function(d: np.ndarray) -> None:
    """Tests the derivative computation of the extended root. Taylorexpansion must
    converge with second order.

    This must hold for all points, hence tested at a random point with a quadratic
    proxy function for the compressibility factor.

    """

    def func(x):
        Z = sum(a**2 for a in x)
        return extended_factor(float(Z), float(x[-1]))

    def dfunc(x):
        dz = np.array([2 * a for a in x]).astype(float)
        return extended_factor_derivatives(dz)

    x0 = np.random.rand(2)
    orders = get_EOC_taylor(func, dfunc, x0, d, np.logspace(0, -10, 11))
    assert_order_at_least(orders, 2.0, tol=1e-3, err_msg=_err_msg(*x0))


@pytest.mark.parametrize(
    "d", [np.array([1.0, 0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
)
def test_derivatives_of_polynom_coeffs_wrt_AB(d: np.ndarray) -> None:
    """Tests the computation of derivatives of the coefficients of the Peng-Robinson-EOS
    with respect to cohesion and covolume."""

    def func(x):
        return c_from_AB(*x)

    def dfunc(x):
        return dc_from_AB(*x)

    x0 = np.random.rand(2)
    orders = get_EOC_taylor(func, dfunc, x0, d, np.logspace(0, -10, 11))
    assert_order_at_least(orders, 2.0, tol=1e-3, err_msg=_err_msg(*x0))


@pytest.mark.parametrize("s", [1e-3, 0.1, 0.25])
@pytest.mark.parametrize("out_format", ["scalar", "array"])
@pytest.mark.parametrize("gaslike", [True, False])
def test_3root_smoothing_function(
    gaslike: bool, out_format: Literal["scalar", "array"], s: float
) -> None:
    """Tests the smoothing function in the physical three root area.

    The smallest (liquidlike) and biggest (gaslike) root are smoothed using a
    nonphysical intermediate root.

    """
    if out_format == "scalar":
        shape = (3,)
    elif out_format == "array":
        shape = (3, 4)
    else:
        assert False

    out = np.random.random(shape)
    out = np.abs(out)
    out_before = out.copy()

    z_b = 0.0
    z_t = 1.0

    # We test for some values in between.
    for z_m in np.linspace(0, 1, 100, endpoint=True):
        z = np.array([z_b, z_m, z_t])

        # Ratio where the intermediate route is.
        d = (z_m - z_b) / (z_t - z_b)

        _smooth_3root_region(z, s, gaslike, out)
        # middle route is in any case unchanged
        assert np.all(out[1] == out_before[1])

        check_is_between = False
        i = np.nan
        # In the gaslike case, the smoothing happens in the interval [1 - 2s, 1-s]
        # In [1-s, 1] the output value for the gas root should be an average of middle
        # root and gas root.
        # In [0, 1-2s] it should be unchanged.
        # In between it is a convex combination, i.e. with values in between.
        # NOTE we test the middle interval with less/greater equal because at case
        # borders it should be smooth as well. So we use if, and not elif as well.
        if gaslike:
            avg = (out_before[1] + out_before[2]) * 0.5
            # First assert liquid root is unchanged
            assert np.all(out[0] == out_before[0])
            if d <= 1 - 2 * s:
                # No change.
                assert np.all(out[2] == out_before[2])
            if 1 - 2 * s <= d <= 1 - s:
                check_is_between = True
                i = 2
            if 1 - s <= d:
                # Average with intermediate root.
                assert np.all(avg == out[2])
        # In the liquidlike case, the smoothing happens in the interval [s, 2s]
        # In [0, s] the output value for the liquid root should be an average of middle
        # root and liquid root.
        # In [2s, 1] it should be unchanged.
        # In between it is a convex combination, i.e. with values in between.
        else:
            avg = (out_before[1] + out_before[0]) * 0.5
            # First assert gas root is unchanged
            assert np.all(out[2] == out_before[2])
            if d <= s:
                # Average with intermediate root.
                assert np.all(avg == out[0])
            if s <= d <= 2 * s:
                check_is_between = True
                i = 0
            if 2 * s <= d:
                # No change.
                assert np.all(out[0] == out_before[0])

        if check_is_between:
            assert np.all(
                ((out_before[i] >= out[i]) & (out[i] >= avg))
                | ((out_before[i] <= out[i]) & (out[i] <= avg))
            )

        out = out_before.copy()


@pytest.mark.parametrize(
    "d", [np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([1.0, -1.0])]
)
@pytest.mark.parametrize("gaslike", [True, False])
@pytest.mark.parametrize("sm", [0.0, 1e-4])
@pytest.mark.parametrize("sc_bw", [0.0, 1e-3])
def test_limitcase_zero_cohesion(
    sm: float,
    sc_bw: float,
    gaslike: bool,
    d: np.ndarray,
    B_range: np.ndarray,
) -> None:
    """The case of zero cohesion is part of the nonphysical 3-root area where the
    smallest real root is smaller than the physical bound B.

    Test for proper extension and computation, as well as order of Taylor expansion.
    Liquid root is expteced to be extended and loosing order of convergence.

    """

    tol = 1e-14

    # See test for zero cohesion and covolume.
    B_range = B_range[B_range > 1e-7]

    for B in B_range:
        x0 = np.array([0.0, B])
        err_msg = _err_msg(*x0)
        c = c_from_AB(*x0)
        rc = get_root_case(c, tol)
        assert rc == 3, f"Expecting 3-root-case: {err_msg}"

        # Testing approximation
        def func(x):
            return get_compressibility_factor(*x, gaslike, tol)

        def dfunc(x):
            return get_compressibility_factor_derivatives(
                *x, gaslike, tol, sm, 1e-4, sc_bw, 1.0
            )

        assert_roots_correctly_sized(*x0, tol=tol)

        is_extended = is_extended_factor(*x0, gaslike, tol)

        # Should be real root
        if gaslike:
            assert not is_extended, f"Expecting gas root to be real: {err_msg}"
            assert get_polynomial_residual(func(x0), c) <= tol, (
                "Gas root not real root."
            )
        else:
            assert is_extended, f"Expecting liquid root to be extended: {err_msg}"

        orders = get_EOC_taylor(func, dfunc, x0, d, np.logspace(-1, -10, 10))
        expected_order = calculate_expected_order(
            gaslike, tol, sm=sm, sc_bw=sc_bw, AB=x0
        )
        # NOTE In any case, the liquid-like root is approximated when approaching
        # zero cohsion
        if not gaslike:
            expected_order = 1
        assert_order_at_least(
            orders, expected_order, tol=1e-2, err_msg=err_msg, asymptotic=5
        )


@pytest.mark.parametrize(
    "d", [np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([-1.0, 1.0])]
)
@pytest.mark.parametrize("gaslike", [True, False])
@pytest.mark.parametrize("sm", [0.0, 1e-4])
@pytest.mark.parametrize("sc_bw", [0.0, 1e-3])
def test_limitcase_zero_covolume(
    sm: float,
    sc_bw: float,
    gaslike: bool,
    d: np.ndarray,
    A_range: np.ndarray,
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

    # See test for zero cohesion and covolume.
    A_range = A_range[A_range > 1e-7]

    # Then we test the rest of the line.
    for A in A_range:
        x0 = np.array([A, 0.0])
        err_msg = _err_msg(*x0)
        c = c_from_AB(*x0)
        rc = get_root_case(c, tol)
        if A < A_L:
            assert rc == 3, f"Expecting 3-root-case: {err_msg}"
        elif A > A_L:
            assert rc == 1, f"Expecting 1-root-case: {err_msg}"
        else:
            # Skip this case, see test_limitcase_zero_covolume_liquid_saturated
            continue

        # Testing approximation
        def func(x):
            return get_compressibility_factor(*x, gaslike, tol)

        def dfunc(x):
            return get_compressibility_factor_derivatives(
                *x, gaslike, tol, sm, 1e-4, sc_bw, 1.0
            )

        assert_roots_correctly_sized(*x0, tol=tol)

        is_extended = is_extended_factor(*x0, gaslike, tol)

        # Since B is shifted to COVOLUME limit, the error should be linear in distance
        # from B = 0.
        if not (gaslike and A >= 0.25):  # Skip area where gas is extended.
            assert get_polynomial_residual(func(x0), c) <= 2 * COVOLUME_LIMIT, (
                "Root too far off."
            )

        if rc == 1 and gaslike:
            assert is_extended, f"Expecting gas root to be extended: {err_msg}"
        else:
            assert not is_extended, f"Expecting root to be real: {err_msg}"

        orders = get_EOC_taylor(func, dfunc, x0, d, np.logspace(-1, -10, 10))

        expected_order = calculate_expected_order(
            gaslike, tol, sm=sm, sc_bw=sc_bw, AB=x0
        )

        # Order reduction where gas is extended.
        if gaslike and A >= 0.25:
            expected_order = 1
        # Order reduction for liquid is always expected.
        if not gaslike:
            expected_order = 1

        assert_order_at_least(
            orders,
            expected_order,
            tol=2e-1,
            err_msg=err_msg,
            # Root approximations are only asymptotic near liquid-saturated
            # line.
            asymptotic=3,
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
@pytest.mark.parametrize("gaslike", [True, False])
@pytest.mark.parametrize("sm", [0.0, 1e-4])
@pytest.mark.parametrize("sc_bw", [0.0, 1e-3])
def test_limitcase_zero_covolume_liquid_saturated(
    sm: float,
    sc_bw: float,
    gaslike: bool,
    d: np.ndarray,
) -> None:
    """Test the special point (A, B) = (0.25, 0), which is the lower end of the liquid-
    saturated 2-phase regime, spanning from this  point to the critical point.

    We also know it's slope (2) there from Ben Gharbia 2021.

    This, with the points 0,0 and the critical point, are the most challenging points
    as the approximations traverse multiple phase regions and extension regimes.
    Approximation is at best linear very close to the point.

    """
    tol = 1e-14
    A_L = 0.25
    x0 = np.array([A_L, 0.0])
    err_msg = _err_msg(*x0)
    c = c_from_AB(*x0)
    rc = get_root_case(c, tol)
    assert rc == 2, f"Expecting 2-root-case: {err_msg}"

    # The raw roots have special values.
    roots = calculate_roots(c, tol)
    np.testing.assert_allclose(roots, np.array([0.0, 0.5]), rtol=0.0, atol=tol)
    # Gas root should not be modified, since real, but liquid-root should be bound
    # by limit value for covolume.
    Zg = get_compressibility_factor(*x0, True, tol)
    assert Zg == 0.5, "Unexpected value for gas root."
    Zl = get_compressibility_factor(*x0, False, tol)
    assert Zl >= COVOLUME_LIMIT, "Unexpected value for liquid root."

    def func(x):
        return get_compressibility_factor(*x, gaslike, tol)

    def dfunc(x):
        return get_compressibility_factor_derivatives(
            *x, gaslike, tol, sm, 1e-4, sc_bw, 1.0
        )

    assert_roots_correctly_sized(*x0, tol=tol)
    is_extended = is_extended_factor(*x0, gaslike, tol)
    assert (
        get_polynomial_residual(func(x0), c) <= tol if gaslike else 2 * COVOLUME_LIMIT
    ), "Root too far off."
    assert not is_extended, (
        f"Expecting root to be real at liquid-saturated border: {err_msg}"
    )

    orders = get_EOC_taylor(func, dfunc, x0, d, np.logspace(-2, -11, 10), tol=1e-3)
    assert_order_at_least(
        orders,
        1,
        tol=1e-2,
        err_msg=err_msg,
        asymptotic=5,
    )


@pytest.mark.parametrize(
    ["d", "expected_liquid_order_loss"],
    [
        (np.array([A_CRIT, CRITICAL_SLOPE * A_CRIT]), False),
        (np.array([1.0, 1e3]), False),
        (np.array([1e3, 1.0]), True),
        (np.array([1.0, 0.0]), True),
        (np.array([0.0, 1.0]), False),
    ],
)
@pytest.mark.parametrize("gaslike", [True, False])
@pytest.mark.parametrize("sm", [0.0, 1e-4])
@pytest.mark.parametrize("sc_bw", [0.0, 1e-3])
def test_limitcase_zero_cohesion_and_covolume(
    sm: float,
    sc_bw: float,
    gaslike: bool,
    d: np.ndarray,
    expected_liquid_order_loss: bool,
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
    rc = get_root_case(c, tol)
    assert rc == 2, f"Expecting 2-root-case: {err_msg}"

    # The raw roots have special values.
    roots = calculate_roots(c, tol)
    np.testing.assert_allclose(roots, np.array([0.0, 1.0]), rtol=0.0, atol=tol)

    assert_roots_correctly_sized(0.0, 0.0, tol=tol)

    # Testing approximation
    def func(x):
        return get_compressibility_factor(*x, gaslike, tol)

    def dfunc(x):
        return get_compressibility_factor_derivatives(
            *x, gaslike, tol, sm, 1e-4, sc_bw, 1.0
        )

    is_extended = is_extended_factor(*x0, gaslike, tol)

    if gaslike:
        assert not is_extended, f"Expecting gas root to be real: {err_msg}"
    else:
        assert is_extended, f"Expecting liquid root to be extended: {err_msg}"

    orders = get_EOC_taylor(func, dfunc, x0, d, np.logspace(-3, -12, 10))

    # Loss of order expected.
    # Gas: Approximation traverses multiple root areas and formulas leading to
    # semi-smooth approximations
    # Liquid is in any case extended, and if approimated from the wrong angles, total
    # loss of order.
    expected_order = 1
    if not gaslike and expected_liquid_order_loss:
        expected_order = 0

    assert_order_at_least(
        orders, expected_order, tol=1e-2, err_msg=err_msg, asymptotic=3
    )
