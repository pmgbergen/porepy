"""Tests the computation of ideal fluid properties."""

from __future__ import annotations

import numpy as np
import pytest

import porepy.compositional as pc
from porepy.applications.test_utils.derivative_testing import (
    assert_order_at_least,
    get_EOC_taylor,
)
from porepy.compositional import ideal
from porepy.compositional.ideal.collection import (
    H_FORMATION_H2O_G_deNevers,
    H_FORMATION_H2O_L_deNevers,
)


@pytest.fixture(scope="module")
def ideal_fluids(request) -> list[ideal.IdealFluid]:
    """Fixture list of compiled ideal fluids present in the subpackage."""
    out = [
        ideal.IdealH2O,
        ideal.IdealCO2,
        ideal.IdealH2S,
        ideal.IdealN2,
    ]
    for i in out:
        i.compile()

    return out


def test_ideal_water_obeys_reference_state(
    ideal_fluids: list[ideal.IdealFluid],
) -> None:
    """Tests that the implementation of the ideal water obeys the reference state
    assumptions.

    This tests the implementation of the ideal enthalpy from the book by de Nevers, and
    the correctness of the basis shift (since de Nevers uses a different reference state
    than porepy, which uses IAPWS).

    This test is critical for the exactness of our computations in terms of physical
    numbers.

    """
    h2o = ideal_fluids[0]
    # Assert that the ideal gas water enthalpy at the defined reference temperature is
    # equal to the difference in the given formation enthalpies.
    h = h2o.funcs["h"]
    latent_heat = H_FORMATION_H2O_G_deNevers - H_FORMATION_H2O_L_deNevers
    assert latent_heat > 0, "Expecting latent heat to be positive."
    np.testing.assert_allclose(
        h(pc.THD_REF.T) - pc.THD_REF.H, latent_heat, atol=1e-15, rtol=0.0
    )

    # The ideal gas internal energy should obey ideal gas law, where the difference
    # in volume is equal to RT, and we use the definition of h = u + RT
    # This holds only if the reference state is correctly used.
    u = h2o.funcs["u"]
    delta_u = latent_heat - pc.THD_REF.R_U * pc.THD_REF.T
    assert delta_u > 0, (
        "Expecting change in internal energy upon evaporation to be positive."
    )
    np.testing.assert_allclose(u(pc.THD_REF.T) - pc.THD_REF.U, delta_u)


@pytest.mark.parametrize("name", ["u", "h"])
@pytest.mark.parametrize("i", [i for i in range(4)])
def test_T_dependent_properties(
    i: int, name: str, ideal_fluids: list[ideal.IdealFluid]
) -> None:
    """Tests the correct implementation of a ideal property dependent on temperature
    and its derivative using Taylor expansion."""

    fluid = ideal_fluids[i]

    d = np.ones(1)
    h = np.logspace(2, -8, 10)

    T0 = np.ones(1) * 400.0

    def func(x: np.ndarray) -> float:
        return fluid.funcs[name](x[0])

    def dfunc(x: np.ndarray) -> np.ndarray:
        return np.array([fluid.funcs[f"d{name}"](x[0])])

    orders = get_EOC_taylor(func, dfunc, T0, d, h, tol=1e-9)
    assert_order_at_least(orders, 2.0, tol=1e-2)


@pytest.mark.parametrize(["d", "h"], [(d, np.logspace(2, -8, 10)) for d in np.eye(2)])
@pytest.mark.parametrize("name", ["rho", "v"])
@pytest.mark.parametrize("i", [i for i in range(4)])
def test_pT_dependent_properties(
    i: int,
    name: str,
    d: np.ndarray,
    h: np.ndarray,
    ideal_fluids: list[ideal.IdealFluid],
) -> None:
    """Tests the correct implementation of a ideal property dependent on both pressure
    and temperature, and its derivative using Taylor expansion."""

    fluid = ideal_fluids[i]
    X0 = np.array((1e6, 400.0))

    def func(x: np.ndarray) -> float:
        return fluid.funcs[name](*x)

    def dfunc(x: np.ndarray) -> np.ndarray:
        return fluid.funcs[f"d{name}"](*x)

    orders = get_EOC_taylor(func, dfunc, X0, d, h, tol=1e-9)
    assert_order_at_least(orders, 2.0, tol=1e-2)
