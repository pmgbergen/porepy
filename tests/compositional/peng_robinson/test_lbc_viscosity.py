"""Testing the LBC viscosity implementation."""

from __future__ import annotations

import numpy as np
import pytest

import porepy as pp
import porepy.compositional.peng_robinson as pr
from porepy.applications.test_utils.derivative_testing import (
    assert_order_at_least,
    get_EOC_taylor,
)
from tests.compositional.peng_robinson import (
    calculate_expected_order,
    components,
    comps_and_phases,
    pr_eos,
)

_dh_per_cp = lambda cp: [
    (d, h)
    for d, h in zip(
        np.eye(2 + cp[0]) if cp[0] > 1 else np.eye(2),
        [np.logspace(3, -3, 7), np.logspace(2, -4, 7)]
        + (cp[0] * [np.logspace(0, -6, 7)] if cp[0] > 0 else []),
    )
]


# @pytest.mark.skipped(reason="slow due to compilation.")
@pytest.mark.parametrize(
    ["comps_and_phases", "d", "h"],
    [
        (cp, d, h)
        for cp in [(1, "V"), (1, "L"), (2, "V"), (2, "L"), (3, "V"), (3, "L")]
        for d, h in _dh_per_cp(cp)
    ],
    indirect=["comps_and_phases"],
)
@pytest.mark.parametrize("smooth3", [0.0, 1e-4, 1e-1])
@pytest.mark.parametrize("smooth_sc", [0.0, 1e-3])
@pytest.mark.parametrize(
    "x0_pT",
    [
        np.array((1e6, 300.0)),
        np.array((1e6, 500.0)),
        np.array((20e6, 300.0)),
        np.array((20e6, 500.0)),
    ],
)
def test_lbc_derivatives(
    x0_pT: np.ndarray,
    smooth3: float,
    smooth_sc: float,
    comps_and_phases: tuple[int, str],
    d: np.ndarray,
    h: np.ndarray,
    pr_eos: pr.CompiledPengRobinson,
) -> None:
    """Tests that the derivatives of the LBC viscosity model are correctly implemented,
    by assuring that the Taylor approximation is of expected order.

    The order can deteriorate if we are in the super-critical area where the
    derivatives of the compressibility factor are approximated when smoothing.
    The expected order is calculated accordingly.

    """
    tol = 1e-14

    params = np.array((smooth3, smooth_sc, tol))

    ncomp = comps_and_phases[0]
    assert pr_eos.nc == ncomp, "Failure in test setup."
    assert pr_eos.is_compiled, "EoS not compiled."

    if comps_and_phases[1] == "L":
        state = pp.compositional.PhysicalState.liquid
    elif comps_and_phases[1] == "V":
        state = pp.compositional.PhysicalState.gas
    else:
        assert False, "Invalid phase specification."

    def func(x):
        p = x[0]
        T = x[1]
        xn = np.array(x[2:]) if ncomp > 1 else np.ones(1)
        assert xn.size == ncomp, "Invalid number of components."
        preargfunc = pr_eos.funcs["prearg_val"]
        propfunc = pr_eos.funcs["mu"]
        return propfunc(preargfunc(state, p, T, xn, params), p, T, xn)

    def dfunc(x):
        p = x[0]
        T = x[1]
        xn = np.array(x[2:]) if ncomp > 1 else np.ones(1)
        assert xn.size == ncomp, "Invalid number of components."
        preargfunc = pr_eos.funcs["prearg_val"]
        preargdifffunc = pr_eos.funcs["prearg_jac"]
        dpropfunc = pr_eos.funcs["dmu"]
        prearg_val = preargfunc(state, p, T, xn, params)
        return dpropfunc(
            prearg_val,
            preargdifffunc(prearg_val, p, T, xn, params),
            p,
            T,
            xn,
        )

    if ncomp > 1:
        x0 = np.hstack((x0_pT, np.ones(ncomp) / ncomp))
    else:
        x0 = x0_pT

    orders = get_EOC_taylor(func, dfunc, x0, d, h, tol=1e-10)
    expected_order = calculate_expected_order(
        True if state == pp.compositional.PhysicalState.gas else False,
        tol,
        smooth_sc=smooth_sc,
        smooth3=smooth3,
        pTx=(x0[0], x0[1], x0[2:]) if ncomp > 1 else (x0[0], x0[1]),
        eos=pr_eos,
    )
    assert_order_at_least(orders, expected_order, tol=2e-2, asymptotic=3)
