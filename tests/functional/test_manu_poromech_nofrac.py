"""
This module contains light-weighted functional tests for a manufactured solution of
the poromechanical equations in an unfractured domain.

The set of equations is non-linear, with the non-linearity entering through the
dependency of the fluid density with the pressure as given in [1, 2]:

                    rho(p) = rho_0 * exp(c_f * (p - p_0)),

where rho_0 and p_0 are the fluid density and pressure at reference states, and c_f
is the (constant) fluid compressibility. For this setup, we employ rho_0=1 [kg * m^-3],
c_f=0.02 [Pa^-1], and p_0=0 [Pa].

The rest of the physical parameters are given unitary values, except, the reference
porosity phi_ref=0.1 [-] and the Biot coefficient alpha=0.5 [-]. For the exact pressure
and displacement solutions, we use the ones employed in [3].

Tests should pass as long as the `desired_error` matches the `actual_error`,
up to absolute (1e-5) and relative (1e-3) tolerances. The values of such tolerances
aim at keeping the test meaningful while minimizing the chances of failure due to
floating-point arithmetic close to machine precision.

For this functional test, we are comparing errors for the pressure, Darcy fluxes,
displacements, and poroelastic forces. The errors are measured in a discrete relative
L2-error norm (such as the ones defined in [3]). The desired errors were obtained by
running the :class:zManuPoromechanics2d` simulation setup with default parameters. We
test the errors for three different times, namely: 0.2 [s], 0.6 [s], and 1.0 [s].

References:

    - [1] Coussy, O. (2004). Poromechanics. John Wiley & Sons. ISO 690

    - [2] Garipov, T. T., & Hui, M. H. (2019). Discrete fracture modeling approach for
    simulating coupled thermo-hydro-mechanical effects in fractured reservoirs.
    International Journal of Rock Mechanics and Mining Sciences, 122, 104075.

    - [3] Nordbotten, J. M. (2016). Stable cell-centered finite volume discretization
    for Biot equations. SIAM Journal on Numerical Analysis, 54(2), 942-968.

"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest

import porepy as pp
from tests.functional.setups.manu_poromech_nofrac import (
    ManuPoroMechSaveData,
    ManuPoroMechSetup,
)


@pytest.fixture
def results() -> list[ManuPoroMechSaveData]:
    # Run verification setup and retrieve results for three different times
    fluid = pp.FluidConstants({"compressibility": 0.02})
    solid = pp.SolidConstants({"biot_coefficient": 0.5})
    material_constants = {"fluid": fluid, "solid": solid}
    params = {
        "mesh_arguments": {"mesh_size_frac": 0.1, "mesh_size_bound": 0.1},
        "manufactured_solution": "nordbotten_2016",
        "material_constants": material_constants,
        "time_manager": pp.TimeManager([0, 0.2, 0.6, 1], 0.2, True),
    }
    setup = ManuPoroMechSetup(params)
    pp.run_time_dependent_model(setup, params)
    return setup.results


# Desired errors
DesiredError = namedtuple(
    "DesiredError",
    "error_pressure, " "error_flux, " "error_displacement, " "error_force ",
)
desired_errors: list[DesiredError] = [
    # t = 0.2 [s]
    DesiredError(
        error_pressure=0.02726034893861471,
        error_flux=0.024928149988902644,
        error_displacement=0.037379176860992194,
        error_force=0.03376791679431919,
    ),
    # t = 0.6 [s]
    DesiredError(
        error_pressure=0.02132896228706064,
        error_flux=0.015744995552258594,
        error_displacement=0.037343144687286826,
        error_force=0.03376572087148504,
    ),
    # t = 1.0 [s]
    DesiredError(
        error_pressure=0.020586932670543838,
        error_flux=0.014707480468287515,
        error_displacement=0.03733585022249853,
        error_force=0.03376572087148504,
    ),
]


# Now, we write the actual test
@pytest.mark.parametrize("time_index", [0, 1, 2])
def test_manufactured_poromechanics_unfractured_2d(time_index: int, results):
    """Check errors for pressure, flux, displacement, and force.

    Nest test functions to avoid computing the solution multiple times.

    """
    actual = results[time_index]
    desired = desired_errors[time_index]

    np.testing.assert_allclose(
        actual.error_pressure, desired.error_pressure, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual.error_flux, desired.error_flux, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual.error_displacement, desired.error_displacement, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual.error_force, desired.error_force, atol=1e-5, rtol=1e-3
    )
