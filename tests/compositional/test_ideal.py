"""Tests the computation of ideal fluid properties."""

from __future__ import annotations

import numpy as np
import pytest

import porepy.compositional as pc
from porepy.compositional import ideal
from porepy.compositional.ideal.collection import (
    H_FORMATION_H2O_G_deNevers,
    H_FORMATION_H2O_L_deNevers,
)


def test_ideal_water_obeys_reference_state() -> None:
    """Tests that the implementation of the ideal water obeys the reference state
    assumptions.

    This tests the implementation of the ideal enthalpy from the book by de Nevers, and
    the correctness of the basis shift (since de Nevers uses a different reference state
    than porepy, which uses IAPWS).

    This test is critical for the exactness of our computations in terms of physical
    numbers.

    """
    h2o = ideal.IdealH2O
    h2o.compile()
    # Assert that the ideal gas water enthalpy at the defined reference temperature is
    # equal to the difference in the given formation enthalpies.
    h = h2o.funcs["h"]
    latent_heat = pc.H_REF + np.abs(
        H_FORMATION_H2O_G_deNevers - H_FORMATION_H2O_L_deNevers
    )
    np.testing.assert_allclose(h(pc.T_REF), latent_heat, atol=1e-15, rtol=0.0)

    # The ideal gas internal energy should obey ideal gas law, where the difference
    # in volume is equal to RT, and we use the definition of h = u + RT
    # This holds only if the reference state is correctly used.
    u = h2o.funcs["u"]
    delta_u = pc.U_REF + np.abs(latent_heat - pc.R_U_MOL * pc.T_REF)
    np.testing.assert_allclose(u(pc.T_REF), delta_u)
