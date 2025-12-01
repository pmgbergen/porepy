"""Testing the assembly of the Peng-Robinson EOS, and its computations."""

from __future__ import annotations

import numpy as np
import pytest

import porepy as pp
import porepy.compositional.ideal as pid
import porepy.compositional.peng_robinson as pr
from porepy.applications.test_utils.derivative_testing import (
    assert_order_at_least,
    get_EOC_taylor,
)
from porepy.compositional.peng_robinson.eos import (
    a_dl,
    a_VdW,
    ac_component,
    alpha,
    b_dl,
    bc_component,
    compact_dense_symmat,
    covolume_dep,
    dalpha_dT,
    ddalpha_dTT,
    grad_a_dl,
    grad_a_VdW,
    grad_b_dl,
    grad_covolume_dep,
    grad_u_dep,
    hess_a_dl,
    hess_a_VdW,
    lnphis,
    lnphis_jac,
    u_dep,
)
from tests.compositional.peng_robinson import (
    calculate_expected_order,
    components,
    comps_and_phases,
    pr_eos,
)


def test_critical_values():
    """Test that critical cohesion and covolume are as expected near the critical
    point."""
    np.testing.assert_allclose(
        ac_component(1.0, 1.0),
        pr.A_CRIT * pp.compositional.THD_REF.R_U**2,
        atol=1e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        bc_component(1.0, 1.0),
        pr.B_CRIT * pp.compositional.THD_REF.R_U,
        atol=1e-14,
        rtol=0.0,
    )


def test_compact_dense_symmat() -> None:
    """Tests the algorithm for compacting and expanding a dense symmetric matrix."""
    N = 3

    A = np.random.random((N, N)) * 100
    A = (A + A.T) / 2.0

    A_arr = compact_dense_symmat(A.copy())
    assert A_arr.ndim == 1, "Expecting 1D array"
    assert A_arr.size == int(N * (N + 1) / 2), "Unexpected size of compacted matrix."

    A2 = compact_dense_symmat(A_arr)
    assert A2.shape == A.shape, "Expanded matrix of unexpected shape"
    (
        np.testing.assert_allclose(A2, A, atol=1e-14, rtol=0.0),
        "Matrix not equal after expansion.",
    )

    # When using a non-symmetric matrix, the algorithm should only use the upper
    # triangle.
    A = np.random.random((N, N)) * 100
    A2 = compact_dense_symmat(compact_dense_symmat(A.copy()))
    idx = np.triu_indices(N)
    np.testing.assert_allclose(A[idx], A2[idx], atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(A[idx], A2.T[idx], atol=1e-14, rtol=0.0)

    # Test that there is no unintended referencing of array elements.
    A = np.random.random((N, N)) * 100
    A_arr = compact_dense_symmat(A)
    A[0] = 200.0
    assert np.all(A[0] != A_arr[:N])


@pytest.mark.parametrize(
    ["d", "h"], [(d, h) for d, h in zip(np.eye(2), 2 * [np.logspace(0, -9, 10)])]
)
def test_repulsive_departure(d: np.ndarray, h: np.ndarray) -> None:
    """Tests the logarithmic term shared by all departure functions."""
    np.random.seed(2)
    B = np.random.rand() + pr.COVOLUME_LIMIT
    Z = 2 * np.random.rand() + B

    def func(x):
        return covolume_dep(*x)

    def dfunc(x):
        return grad_covolume_dep(*x)

    x0 = np.array((Z, B))

    orders = get_EOC_taylor(func, dfunc, x0, d, h)
    assert_order_at_least(orders, 2, tol=1e-2, err_msg=f"d = {d}", asymptotic=7)


@pytest.mark.parametrize("omega", [0.2, 0.8])
def test_alpha(omega: float):
    """Tests the implementation of the alpha term in the cohesion, it's first and
    second derivative."""

    Tc = 600.0
    omega = 0.7
    h = np.logspace(1, -8, 10)
    x0 = np.array([400.0])
    d = np.ones(1)

    # First test that first derivative approximates function properly.

    def func(x: np.ndarray) -> np.ndarray:
        T = x[0]
        return alpha(T, Tc, omega)

    def dfunc(x: np.ndarray) -> np.ndarray:
        T = x[0]
        return np.array([dalpha_dT(T, Tc, omega)])

    def ddfunc(x: np.ndarray) -> np.ndarray:
        T = x[0]
        return np.array([[ddalpha_dTT(T, Tc, omega)]])

    orders = get_EOC_taylor(func, dfunc, x0, d, h)
    assert_order_at_least(orders, 2, tol=1e-2)

    # Test that the second derivative approximates the first derivative.

    orders = get_EOC_taylor(dfunc, ddfunc, x0, d, h)
    assert_order_at_least(orders, 2, tol=1e-2)

    # Finally, test higher order approximation.

    orders = get_EOC_taylor(func, dfunc, x0, d, h, ddfunc=ddfunc)
    assert_order_at_least(orders, 3, tol=1e-2)


_dh_per_n = lambda n, pt: [
    (d, h)
    for d, h in (
        zip(np.eye(pt + n), (n + pt) * [np.logspace(1, -5, 7)])
        if n > 1
        else zip(np.eye(pt + n)[:pt], pt * [np.logspace(1, -5, 7)])
    )
]


@pytest.mark.parametrize(
    ["nc", "d", "h"], [(n, d, h) for n in [1, 2, 5] for d, h in _dh_per_n(n, 1)]
)
def test_cohesion_VdW(nc: int, d: np.ndarray, h: np.ndarray) -> None:
    """Tests the implementation of the cohesion, its gradient and Hessian."""

    np.random.seed(42)

    T = 300.0 + np.random.rand() * 100
    Tcs = 400 + np.random.rand(nc) * 10
    omegas = np.random.rand(nc) + 1e-5
    acs = 100.0 + np.random.rand(nc) * 10
    bips = np.random.random((nc, nc)) + 1e-5
    bips = (bips + bips.T) / 2.0
    np.fill_diagonal(bips, 0.0)

    x0 = np.array([T] + [1 / nc] * nc)

    def func(x: np.ndarray) -> float:
        T = x[0]
        x = x[1:]
        return a_VdW(T, x, Tcs, omegas, acs, bips)

    def dfunc(x: np.ndarray) -> float:
        T = x[0]
        x = x[1:]
        grad_a = grad_a_VdW(T, x, Tcs, omegas, acs, bips)
        if nc > 1:
            return grad_a
        else:
            return np.hstack((grad_a, 0.0))

    def ddfunc(x: np.ndarray) -> float:
        T = x[0]
        x = x[1:]
        hess_a = compact_dense_symmat(hess_a_VdW(T, x, Tcs, omegas, acs, bips))
        if nc > 1:
            return hess_a
        else:
            j = np.eye(2)
            j[:-1, :-1] = hess_a
            return j

    # NOTE: Due to floating point arithmetics, we loose digits and must consider
    # approximation errors below this value as zero.
    err_tol = 1e-12

    # Test grad approximates function.
    orders = get_EOC_taylor(func, dfunc, x0, d, h, tol=err_tol)
    assert_order_at_least(orders, 2, tol=1e-2)

    # Test Hessian approximates grad.
    orders = get_EOC_taylor(dfunc, ddfunc, x0, d, h, tol=err_tol)
    assert_order_at_least(orders, 2, tol=1e-2)

    # Test higher order approximation.
    orders = get_EOC_taylor(func, dfunc, x0, d, h, tol=err_tol, ddfunc=ddfunc)
    assert_order_at_least(orders, 3, tol=1e-2)


@pytest.mark.parametrize(
    ["nc", "d", "h"], [(n, d, h) for n in [1, 2, 5] for d, h in _dh_per_n(n, 2)]
)
def test_cohesion_VdW_dl(nc: int, d: np.ndarray, h: np.ndarray) -> None:
    """Tests the implementation of the dimensionless cohesion, its gradient and Hessian.

    The VdW cohesion is used as its base.

    """

    np.random.seed(42)

    p = 1e7 + np.random.rand() * 1e6
    T = 300.0 + np.random.rand() * 100
    Tcs = 400 + np.random.rand(nc) * 10
    omegas = np.random.rand(nc) + 1e-5
    acs = 100.0 + np.random.rand(nc) * 10
    bips = np.random.random((nc, nc)) + 1e-5
    bips = (bips + bips.T) / 2.0
    np.fill_diagonal(bips, 0.0)

    x0 = np.array([p, T] + [1 / nc] * nc)

    def func(x: np.ndarray) -> float:
        p = x[0]
        T = x[1]
        xn = x[2:]
        a = a_VdW(T, xn, Tcs, omegas, acs, bips)
        return a_dl(a, p, T)

    def dfunc(x: np.ndarray) -> float:
        p = x[0]
        T = x[1]
        xn = x[2:]
        a = a_VdW(T, xn, Tcs, omegas, acs, bips)
        grad_a = grad_a_VdW(T, xn, Tcs, omegas, acs, bips)
        grad_A = grad_a_dl(grad_a, a, p, T)
        if nc > 1:
            return grad_A
        else:
            return np.hstack((grad_A, 0.0))

    def ddfunc(x: np.ndarray) -> float:
        p = x[0]
        T = x[1]
        xn = x[2:]
        a = a_VdW(T, xn, Tcs, omegas, acs, bips)
        grad_a = grad_a_VdW(T, xn, Tcs, omegas, acs, bips)
        hess_a = hess_a_VdW(T, xn, Tcs, omegas, acs, bips)
        hess_A = compact_dense_symmat(hess_a_dl(hess_a, grad_a, a, p, T))
        if nc > 1:
            return hess_A
        else:
            j = np.eye(3)
            j[:-1, :-1] = hess_A
            return j

    # See note in cohesion_VdW test.
    err_tol = 1e-12

    # Test grad approximates function.
    orders = get_EOC_taylor(func, dfunc, x0, d, h, tol=err_tol)
    assert_order_at_least(orders, 2, tol=1e-2)

    # Test Hessian approximates grad.
    orders = get_EOC_taylor(dfunc, ddfunc, x0, d, h, tol=err_tol)
    assert_order_at_least(orders, 2, tol=1e-2)

    # Test higher order approximation.
    orders = get_EOC_taylor(func, dfunc, x0, d, h, tol=err_tol, ddfunc=ddfunc)
    assert_order_at_least(orders, 3, tol=1e-2)


@pytest.mark.parametrize(
    ["nc", "d", "h"], [(n, d, h) for n in [1, 2, 5] for d, h in _dh_per_n(n, 2)]
)
def test_covolume_dl(nc: int, d: np.ndarray, h: np.ndarray) -> None:
    """Tests the implementation of the dimensionless covolume and its gradient.

    Uses VdW covolume as its base
    """

    np.random.seed(42)

    p = 1e7 + np.random.rand() * 1e6
    T = 300.0 + np.random.rand() * 100
    bcs = 100.0 + np.random.rand(nc) * 10
    x0 = np.array([p, T] + [1 / nc] * nc)

    def func(x: np.ndarray) -> float:
        p = x[0]
        T = x[1]
        x = x[2:]
        b = np.dot(x, bcs)
        return b_dl(b, p, T)

    def dfunc(x: np.ndarray) -> float:
        p = x[0]
        T = x[1]
        x = x[2:]
        b = np.dot(x, bcs)
        grad_b = bcs.copy()
        grad_B = grad_b_dl(grad_b, b, p, T)
        if nc > 1:
            return grad_B
        else:
            return np.hstack((grad_B, 0.0))

    # Test grad approximates function.
    orders = get_EOC_taylor(func, dfunc, x0, d, h, tol=1e-9)
    assert_order_at_least(orders, 2, tol=1e-2)


@pytest.mark.parametrize(
    ["d", "h"],
    [
        (d, h)
        for d, h in zip(
            np.eye(5),
            [
                np.logspace(2, -7, 10),
            ]
            * 5,
        )
    ],
)
def test_u_dep(d: np.ndarray, h: np.ndarray):
    """Tests the correct implementation of the derivative of the departure internal
    energy."""

    np.random.seed(42)

    A = np.random.rand() * 100 + 10
    B = np.random.rand() * 100 + 10
    # Only restriction that Z must be greater than B.
    Z = np.random.rand() * 100 + B + 300
    T = 400.0
    dAdT = np.random.rand() * 100 + 10

    x0 = np.array((A, B, Z, T, dAdT))

    def func(x: np.ndarray) -> float:
        return u_dep(*x)

    def dfunc(x: np.ndarray) -> np.ndarray:
        return grad_u_dep(*x)

    # NOTE The computations suffers from loss of precision due to logarithms of small
    # numbers and its derivative.
    orders = get_EOC_taylor(func, dfunc, x0, d, h, tol=1e-10)
    assert_order_at_least(orders, 2, tol=1e-2, asymptotic=8)


@pytest.mark.parametrize(
    ["nc", "d", "h"],
    [
        (n, d, h)
        for n in [1, 2, 5]
        for d, h in zip(
            np.eye(5),
            [
                np.logspace(2, -7, 10),
            ]
            * 5,
        )
    ],
)
def test_lnphis(nc: int, d: np.ndarray, h: np.ndarray) -> None:
    """Tests that the derivative computation for the logarithm of fugacities is
    correctly implemented."""

    np.random.seed(42)

    A = np.random.rand() * 100 + 100
    B = np.random.rand() * 100 + 100
    # Only restriction that Z must be greater than B.
    Z = np.random.rand() * 100 + B + 300
    dadx = np.random.rand() * 100 + 100
    bi = np.random.rand() * 100 + 100

    x0 = np.array((A, B, Z, dadx, bi))

    def func(x: np.ndarray) -> np.ndarray:
        return lnphis(*x[:-2], np.ones(nc) * x[-2], np.ones(nc) * x[-1])

    def dfunc(x: np.ndarray) -> np.ndarray:
        return lnphis_jac(*x[:-2], np.ones(nc) * x[-2], np.ones(nc) * x[-1])

    # NOTE The computations suffers from loss of precision due to logarithms of small
    # numbers and its derivative.
    orders = get_EOC_taylor(func, dfunc, x0, d, h, tol=1e-12)
    assert_order_at_least(orders, 2, tol=1e-2, asymptotic=7)


_dh_per_cp_id = lambda cp: [
    (d, h)
    for d, h in zip(
        np.eye(1 + cp[0]) if cp[0] > 1 else np.eye(1),
        [np.logspace(2, -4, 7)]
        + (cp[0] * [np.logspace(0, -6, 7)] if cp[0] > 0 else []),
    )
]


@pytest.mark.parametrize(
    ["d", "h"], [(d, h) for d, h in zip(np.eye(2), 2 * [np.logspace(2, -7, 10)])]
)
@pytest.mark.parametrize(
    ["func", "dfunc"],
    [(pid.ideal_rho, pid.grad_ideal_rho), (pid.ideal_v, pid.grad_ideal_v)],
)
def test_ideal_density_and_volume(func, dfunc, d: np.ndarray, h: np.ndarray) -> None:
    """Tests the implementation of ideal density and volume and their derivative
    implementation."""

    np.random.seed(28)

    p = 1e7 + np.random.rand() * 1e6
    T = 300.0 + np.random.rand() * 1e2

    x0 = np.array((p, T))

    def func_(x):
        return func(x[0], x[1])

    def dfunc_(x):
        return dfunc(x[0], x[1])

    orders = get_EOC_taylor(func_, dfunc_, x0, d, h)
    assert_order_at_least(orders, 2, tol=1e-2)


@pytest.mark.skipped(reason="slow due to compilation")
@pytest.mark.parametrize(
    ["comps_and_phases", "d", "h"],
    [(cp, d, h) for cp in [(1, "V"), (2, "V"), (3, "V")] for d, h in _dh_per_cp_id(cp)],
    indirect=["comps_and_phases"],
)
@pytest.mark.parametrize("property_name", ["h", "u"])
def test_ideal_mixture_energies(
    comps_and_phases: tuple[int, str],
    d: np.ndarray,
    h: np.ndarray,
    property_name: str,
    pr_eos: pr.CompiledPengRobinson,
) -> None:
    """Test correctness of the Peng-Robinson EoS derivatives, i.e. that the Taylor
    approximation is of second order.

    Ideal energies do not depend on pressure, contrary to other or real properties.

    """

    ncomp = comps_and_phases[0]

    assert pr_eos.nc == ncomp, "Failure in test setup."
    assert pr_eos.is_compiled, "EoS not compiled."

    def func(x):
        T = x[0]
        xn = np.array(x[1:]) if ncomp > 1 else np.ones(1)
        assert xn.size == ncomp, "Invalid number of components."
        propfunc = pr_eos._ideal_funcs[property_name]
        return propfunc(T, xn)

    def dfunc(x):
        T = x[0]
        xn = np.array(x[1:]) if ncomp > 1 else np.ones(1)
        assert xn.size == ncomp, "Invalid number of components."
        dpropfunc = pr_eos._ideal_funcs[f"d{property_name}"]
        return dpropfunc(T, xn)

    x0 = np.array([400.0] + ([1.0 / ncomp] * ncomp if ncomp > 1 else []))

    # NOTE: Precision loss likely due to powers of temperature and division by
    # temperatures in interpolation.
    orders = get_EOC_taylor(func, dfunc, x0, d, h, tol=1e-9)
    assert_order_at_least(
        orders,
        2,
        tol=1e-2,
        asymptotic=5,
        err_msg=f"prop = {property_name}, x0 = {x0}, d = {d}",
    )


_dh_per_cp = lambda cp: [
    (d, h)
    for d, h in zip(
        np.eye(2 + cp[0]) if cp[0] > 1 else np.eye(2),
        [np.logspace(3, -3, 7), np.logspace(1, -5, 7)]
        + (cp[0] * [np.logspace(0, -6, 7)] if cp[0] > 0 else []),
    )
]


@pytest.mark.skipped(reason="slow due to compilation")
@pytest.mark.parametrize(
    ["comps_and_phases", "d", "h"],
    [
        (cp, d, h)
        for cp in [(1, "V"), (1, "L"), (2, "V"), (2, "L"), (3, "V"), (3, "L")]
        for d, h in _dh_per_cp(cp)
    ],
    indirect=["comps_and_phases"],
)
@pytest.mark.parametrize("property_name", ["h", "u", "rho", "v", "phis"])
@pytest.mark.parametrize("smooth3", [0.0, 1e-4])
@pytest.mark.parametrize("smooth_sc", [0.0, 1e-3])
@pytest.mark.parametrize(
    "x0_pT",
    [
        # Cover low and high pressure and temperature.
        np.array((1e6, 300.0)),
        np.array((1e6, 500.0)),
        np.array((20e6, 300.0)),
        np.array((20e6, 500.0)),
    ],
)
def test_property_derivatives(
    x0_pT: np.ndarray,
    smooth3: float,
    smooth_sc: float,
    comps_and_phases: tuple[int, str],
    d: np.ndarray,
    h: np.ndarray,
    property_name: str,
    pr_eos: pr.CompiledPengRobinson,
) -> None:
    """Test correctness of the Peng-Robinson EoS derivatives, i.e. that the Taylor
    approximation is of second order.

    The order can deteriorate if we are in the super-critical area where the
    derivatives of the compressibility factor are approximated when smoothing.
    The expected order is calculated accordingly.

    """

    tol = 1e-14
    # smooth3 = 0.0
    # smooth_sc = 1e-3
    params = np.array((smooth3, smooth_sc, tol))

    ncomp = comps_and_phases[0]

    if comps_and_phases[1] == "L":
        state = pp.compositional.PhysicalState.liquid
    elif comps_and_phases[1] == "V":
        state = pp.compositional.PhysicalState.gas
    else:
        assert False, "Invalid phase specification."

    assert pr_eos.nc == ncomp, "Failure in test setup."
    assert pr_eos.is_compiled, "EoS not compiled."

    def func(x):
        p = x[0]
        T = x[1]
        xn = np.array(x[2:]) if ncomp > 1 else np.ones(1)
        assert xn.size == ncomp, "Invalid number of components."
        preargfunc = pr_eos.funcs["prearg_val"]
        propfunc = pr_eos.funcs[property_name]
        return propfunc(preargfunc(state, p, T, xn, params), p, T, xn)

    def dfunc(x):
        p = x[0]
        T = x[1]
        xn = np.array(x[2:]) if ncomp > 1 else np.ones(1)
        assert xn.size == ncomp, "Invalid number of components."
        preargfunc = pr_eos.funcs["prearg_val"]
        preargdifffunc = pr_eos.funcs["prearg_jac"]
        dpropfunc = pr_eos.funcs[f"d{property_name}"]
        prearg_val = preargfunc(state, p, T, xn, params)
        return dpropfunc(
            preargfunc(state, p, T, xn, params),
            preargdifffunc(prearg_val, p, T, xn, params),
            p,
            T,
            xn,
        )

    if ncomp > 1:
        x0 = np.hstack((x0_pT, np.ones(ncomp) / ncomp))
    else:
        x0 = x0_pT

    # NOTE: Precision loss from ideal part is propagated to the real energies.
    orders = get_EOC_taylor(func, dfunc, x0, d, h, tol=1e-9)
    expected_order = calculate_expected_order(
        True if state == pp.compositional.PhysicalState.gas else False,
        tol,
        smooth_sc=smooth_sc,
        smooth3=smooth3,
        pTx=(x0[0], x0[1], x0[2:]) if ncomp > 1 else (x0[0], x0[1]),
        eos=pr_eos,
    )
    assert_order_at_least(
        orders,
        expected_order,
        tol=5e-2,
        err_msg=f"{property_name}; x0 = {x0[:2]}; d = {d}; state={state}",
    )
