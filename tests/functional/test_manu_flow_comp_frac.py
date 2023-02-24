"""
This module contains light-weighted functional tests for a manufactured solution of
the compressible flow with a single, fully embedded vertical fracture.

The manufactured solution for the compressible flow verification is obtained as a
natural extension of the incompressible case, see [1]. The non-linearity is included
via the dependency of the fluid density with the fluid pressure as given in [3]:

.. math:

    \rho(p) = \rho_0 \exp{(c_f (p - p_0))},

where, for our setup, :math:`\rho_0 = 1` [kg * m^-3], :math:`c_f = 0.2` [Pa^-1],
and :math:`p_0 = 0` [Pa]. The rest of the physical parameter are given unitary
values, except for the reference porosity :math:`\phi_0 = 0.1` and normal
permeability :math:`\kappa = 0.5`.

Tests should pass as long as the `desired_error` matches the `actual_error`,
up to absolute (1e-5) and relative (1e-3) tolerances. The values of such tolerances
aim at keeping the test meaningful while minimizing the chances of failure due to
floating-point arithmetic close to machine precision.

For this functional test, we are comparing errors for the pressure (for the matrix and
the fracture) and fluxes (for the matrix, the fracture, and on the interface). The
errors are measured in a discrete relative L2-error norm (such as the ones defined
in [2]). The desired errors were obtained by running the
:class:`ManuCompFlowSetup` simulation setup with the parameters given in
the `Examples` section of the class docstring. We test the errors for three different
times, namely: 0.2 [s], 0.6 [s], and 1.0 [s].

References:

    - [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu,
    F. A. (2022). A posteriori error estimates for hierarchical mixed-dimensional
    elliptic equations. Journal of Numerical Mathematics.

    - [2] Nordbotten, J. M. (2016). Stable cell-centered finite volume discretization
    for Biot equations. SIAM Journal on Numerical Analysis, 54(2), 942-968.

    - [3] Garipov, T. T., & Hui, M. H. (2019). Discrete fracture modeling approach for
    simulating coupled thermo-hydro-mechanical effects in fractured reservoirs.
    International Journal of Rock Mechanics and Mining Sciences, 122, 104075.

"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest

import porepy as pp
from tests.functional.setups.manu_flow_comp_frac import (
    ManuCompFlowSetup,
    ManuCompSaveData,
    manu_comp_fluid,
    manu_comp_solid,
)


@pytest.fixture(scope="module")
def results() -> list[ManuCompSaveData]:
    # Run verification setup and retrieve results for the scheduled times
    material_constants = {
        "solid": pp.SolidConstants(manu_comp_solid),
        "fluid": pp.FluidConstants(manu_comp_fluid),
    }
    mesh_arguments = {"mesh_size_frac": 0.05, "mesh_size_bound": 0.05}
    time_manager = pp.TimeManager([0, 0.2, 0.8, 1.0], 0.2, True)
    params = {
        "material_constants": material_constants,
        "mesh_arguments": mesh_arguments,
        "time_manager": time_manager,
    }
    setup = ManuCompFlowSetup(params)
    pp.run_time_dependent_model(setup, params)
    return setup.results


DesiredError = namedtuple(
    "DesiredError",
    "error_matrix_pressure, "
    "error_matrix_flux, "
    "error_frac_pressure, "
    "error_frac_flux, "
    "error_intf_flux",
)


desired_errors: list[DesiredError] = [
    # t = 0.2 [s]
    DesiredError(
        error_matrix_pressure=0.0036348942968626903,
        error_matrix_flux=0.004269623457013436,
        error_frac_pressure=0.6583594878262145,
        error_frac_flux=0.0004147458073966614,
        error_intf_flux=0.4694294365244132,
    ),
    # t = 0.6 [s]
    DesiredError(
        error_matrix_pressure=0.003028196249337853,
        error_matrix_flux=0.0029348589540767584,
        error_frac_pressure=0.6596212819386692,
        error_frac_flux=0.0005294349183034246,
        error_intf_flux=0.4943796145199786,
    ),
    # t = 1.0 [s]
    DesiredError(
        error_matrix_pressure=0.0028115632547158387,
        error_matrix_flux=0.002959106297324448,
        error_frac_pressure=0.651546529657672,
        error_frac_flux=0.0005572271352671979,
        error_intf_flux=0.49619460256620995,
    ),
]


# Now, we write the actual test
@pytest.mark.parametrize("time_index", [0, 1, 2])
def test_manufactured_flow_compressible_fractured_2d(time_index, results):
    """Check errors for pressure and flux solutions.

    Nest test functions to avoid computing the solution multiple times.

    """
    actual = results[time_index]
    desired = desired_errors[time_index]

    np.testing.assert_allclose(
        actual.error_matrix_pressure,
        desired.error_matrix_pressure,
        atol=1e-5,
        rtol=1e-3,
    )

    np.testing.assert_allclose(
        actual.error_matrix_flux, desired.error_matrix_flux, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual.error_frac_pressure,
        desired.error_frac_pressure,
        atol=1e-5,
        rtol=1e-3,
    )

    np.testing.assert_allclose(
        actual.error_frac_flux, desired.error_frac_flux, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual.error_intf_flux, desired.error_intf_flux, atol=1e-5, rtol=1e-3
    )
