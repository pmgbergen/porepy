"""
This module contains light-weighted functional tests for a manufactured solution of
the compressible flow with a single, fully embedded vertical fracture.

The manufactured solution for the compressible flow verification is obtained as a
natural extension of the incompressible case, see [1]. The non-linearity is included
via the dependency of the fluid density with the fluid pressure as given in [3]:

                    rho(p) = rho_0 * exp(c_f * (p - p_0)),

where, for our setup, rho_0=1 [kg * m^-3], c_f=0.2 [Pa^-1], and p_0=0 [Pa]. The rest
of the physical parameter are given unitary values, except the porosity phi=0.1.

Tests should pass as long as the `desired_error` matches the `actual_error`,
up to absolute (1e-5) and relative (1e-3) tolerances. The values of such tolerances
aim at keeping the test meaningful while minimizing the chances of failure due to
floating-point arithmetic close to machine precision.

For this functional test, we are comparing errors for the pressure (for the rock and
the fracture) and fluxes (for the rock, the fracture, and on the interface). The
errors are measured in a discrete relative L2-error norm (such as the ones defined
in [2]). The desired errors were obtained by running the
::class::ManufacturedCompFlow2d simulation setup with default parameters. We test the
errors for three different times, namely: 0.2 [s], 0.6 [s], and 1.0 [s].

References:

    [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu,
    F. A. (2022). A posteriori error estimates for hierarchical mixed-dimensional
    elliptic equations. Journal of Numerical Mathematics.

    [2] Nordbotten, J. M. (2016). Stable cell-centered finite volume discretization
    for Biot equations. SIAM Journal on Numerical Analysis, 54(2), 942-968.

    [3] Garipov, T. T., & Hui, M. H. (2019). Discrete fracture modeling approach for
    simulating coupled thermo-hydro-mechanical effects in fractured reservoirs.
    International Journal of Rock Mechanics and Mining Sciences, 122, 104075.

"""

from collections import namedtuple

import numpy as np
import pytest

import porepy as pp
from porepy.models.verification_setups.manu_flow_comp_frac import ManufacturedCompFlow2d

# Run verification setup and retrieve results for three different times
stored_times = [0.2, 0.6, 1.0]
params = {"stored_times": stored_times}
setup = ManufacturedCompFlow2d(params)
pp.run_time_dependent_model(setup, params)

# Desired errors
DesiredError = namedtuple(
    "DesiredError",
    "error_rock_pressure, "
    "error_rock_flux, "
    "error_frac_pressure, "
    "error_frac_flux, "
    "error_intf_flux",
)

desired_errors: list[DesiredError, DesiredError, DesiredError] = [
    # t = 0.2 [s]
    DesiredError(
        error_rock_pressure=0.0036348942968626903,
        error_rock_flux=0.004269623457013436,
        error_frac_pressure=0.6583594878262145,
        error_frac_flux=0.0004147458073966614,
        error_intf_flux=0.4694294365244132,
    ),
    # t = 0.6 [s]
    DesiredError(
        error_rock_pressure=0.004065781090409986,
        error_rock_flux=0.0031985302018835153,
        error_frac_pressure=0.6435320945794909,
        error_frac_flux=0.00012455930072995976,
        error_intf_flux=0.4424159881895866,
    ),
    # t = 1.0 [s]
    DesiredError(
        error_rock_pressure=0.0037766760157519246,
        error_rock_flux=0.003168806344488447,
        error_frac_pressure=0.62511414569198,
        error_frac_flux=0.00011512160294788515,
        error_intf_flux=0.43780241971270234,
    ),
]


# Now, we write the actual test
@pytest.mark.parametrize(
    "desired, actual",
    [(desired, actual) for (desired, actual) in zip(desired_errors, setup.results)],
)
def test_manufactured_flow_compressible_fractured_2d(desired, actual):
    """Check errors for pressure and flux solutions."""

    np.testing.assert_allclose(
        actual.error_rock_pressure, desired.error_rock_pressure, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual.error_rock_flux, desired.error_rock_flux, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual.error_frac_pressure, desired.error_frac_pressure, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual.error_frac_flux, desired.error_frac_flux, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual.error_intf_flux, desired.error_intf_flux, atol=1e-5, rtol=1e-3
    )
