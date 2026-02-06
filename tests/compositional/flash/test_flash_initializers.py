"""Testing methods in the flash initialization module."""

from __future__ import annotations

import pytest

from itertools import product

import numpy as np
import porepy as pp

import porepy.compositional.flash as pf
from porepy.compositional._global_thermodynamic_reference_state import T as T_REF

from tests.compositional.peng_robinson import components, comps_and_phases


@pytest.mark.parametrize("comps_and_phases", [(2, "V"), (3, "V")], indirect=True)
def test_dew_and_bubble_point_T_computation(
    comps_and_phases: tuple[int, str],
    components: list[pp.compositional.FluidComponent],
) -> None:
    """Tests the computation of dew and bubble point for a mixture of water and CO2."""

    p_cs = np.array([c.critical_pressure for c in components])
    T_cs = np.array([c.critical_temperature for c in components])
    omegas = np.array([c.acentric_factor for c in components])
    n_C1m = comps_and_phases[0] - 1

    N = 100
    z0 = np.linspace(1e-3, 1.0, N, endpoint=False)
    p = np.linspace(2e5, 50e6, N, endpoint=True)

    for z_, p_ in product(z0, p):
        z_r = (1.0 - z_) / n_C1m
        zs = np.array([z_] + [z_r] * n_C1m)
        T0 = (T_REF + np.sum(zs * T_cs)) * 0.5
        T_dew = pf.get_dew_point_T(T0, p_, zs, p_cs, T_cs, omegas)
        assert T_dew not in (np.nan, np.inf, -np.inf)
        T_bub = pf.get_bubble_point_T(T_dew, p_, zs, p_cs, T_cs, omegas)
        assert T_bub not in (np.nan, np.inf, -np.inf)

        # Assert solution found.
        K_dew = pf.K_Wilson(p_, T_dew, p_cs, T_cs, omegas)
        res_dew = np.sum(zs / K_dew) - 1.0
        assert np.abs(res_dew) < 1e-7

        K_bub = pf.K_Wilson(p_, T_bub, p_cs, T_cs, omegas)
        res_bub = np.sum(zs * K_bub) - 1.0
        assert np.abs(res_bub) < 1e-7
