"""Testing the assembly of the Peng-Robinson EOS, and its computations."""

from __future__ import annotations

import numpy as np
import pytest

import porepy as pp
import porepy.compositional.peng_robinson as pr
from porepy.applications.test_utils.derivative_testing import (
    assert_order_at_least,
    get_EOC_taylor,
)
from tests.compositional.peng_robinson import components, comps_and_phases, pr_eos


@pytest.mark.skipped(reason="slow due to compilation.")
@pytest.mark.parametrize("params", [np.zeros(0), np.ones(1) * 0.2])
@pytest.mark.parametrize("prop", ["h", "v", "rho", "phis"])
@pytest.mark.parametrize(
    "comps_and_phases",
    [(1, "V"), (1, "L"), (2, "V"), (2, "L"), (3, "V"), (3, "L")],
    indirect=True,
)
def test_property_derivatives(
    comps_and_phases: tuple[int, str],
    prop: str,
    params: np.ndarray,
    pr_eos: pr.CompiledPengRobinson,
) -> None:
    """Test correctness of the Peng-Robinson EoS derivatives, i.e. that the Taylor
    approximation is of second order."""

    dprop = f"d{prop}"
    ncomp = comps_and_phases[0]

    if comps_and_phases[1] == "L":
        state = pp.compositional.PhysicalState.liquid
    elif comps_and_phases[1] == "V":
        state = pp.compositional.PhysicalState.gas
    else:
        assert False, "Invalid phase specification."

    assert pr_eos._nc == ncomp, "Failure in test setup."
    assert pr_eos.is_compiled, "EoS not compiled."

    def func(*x):
        p = x[0]
        T = x[1]
        xn = np.array(x[2:])
        assert xn.size == ncomp, "Invalid number of components."
        preargfunc = pr_eos.funcs["prearg_val"]
        propfunc = pr_eos.funcs[prop]
        return propfunc(preargfunc(state, p, T, xn, params), p, T, xn)

    def dfunc(*x):
        p = x[0]
        T = x[1]
        xn = np.array(x[2:])
        xn = xn / xn.sum()
        assert xn.size == ncomp, "Invalid number of components."
        preargfunc = pr_eos.funcs["prearg_val"]
        preargdifffunc = pr_eos.funcs["prearg_jac"]
        dpropfunc = pr_eos.funcs[dprop]
        return dpropfunc(
            preargfunc(state, p, T, xn, params),
            preargdifffunc(state, p, T, xn, params),
            p,
            T,
            xn,
        )

    x0 = np.zeros(2 + ncomp)
    x0[0] = 1e7  # Pressure in Pa
    x0[1] = 400.0  # Temperature in K
    x0[2:] = 1.0 / ncomp  # Partial fractions
    directions = np.eye(2 + ncomp)
    h_p = np.logspace(3, -3, 7)
    h_T = np.logspace(2, -4, 7)
    h_x = np.logspace(0, -6, 7)

    # TODO: Reduce tolerance once sympy dependency is removed.
    # There is some suspected loss of precision due to floating point arithmetic.
    for d, h in zip(directions, [h_p, h_T] + [h_x] * ncomp):
        orders = get_EOC_taylor(func, dfunc, x0, d, h, tol=1e-10)
        assert_order_at_least(
            orders,
            2.0,
            tol=1.5e-1,
            err_msg=f"{prop} ({comps_and_phases}) {d}",
            asymptotic=5,
        )
