"""
This module contains light-weighted functional tests for a manufactured flow solution
with a single vertical fracture.
"""

import numpy as np

import porepy as pp
from porepy.models.verification_setups.manu_flow_incomp_frac import ManufacturedFlow2d


def test_pressure_and_fluxes():
    """Check errors for pressure and flux solutions.

    The manufactured solution assumes unitary physical parameters. For the details about
    the solution setup, see Section 7.1 from [1].

    Tests should pass as long as the `desired_error` matches the `actual_error`, up to
    absolute (1e-5) and relative (1e-3) tolerances. The values of such tolerances aim at
    keeping the test meaningful while minimizing the chances of failure due to
    floating-point arithmetic close to machine precision.

    In this test, we are comparing errors for the pressure (for the rock and the
    fracture) and fluxes (for the rock, the fracture, and on the interface). The errors
    are measured in a discrete relative L2-error norm (such as the ones defined in [2]).
    The desired errors were obtained by running the ::class::ManufacturedFlow2D
    simulation setup with default parameters.

    References:

        [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu, F. A.
            (2022). A posteriori error estimates for hierarchical mixed-dimensional
             elliptic equations. Journal of Numerical Mathematics.

        [2] Nordbotten, J. M. (2016). Stable cell-centered finite volume
            discretization for Biot equations. SIAM Journal on Numerical Analysis,
            54(2), 942-968.

    """
    setup = ManufacturedFlow2d({})
    pp.run_stationary_model(setup, {})

    desired_rock_pressure_error = 0.003967565610557834
    desired_rock_flux_error = 0.003137353186856967
    desired_frac_pressure_error = 0.7106357756119986
    desired_frac_flux_error = 0.00041600808643628363
    desired_intf_flux_error = 0.5023434810025456

    actual_rock_pressure_error = setup.sol.error_rock_pressure
    actual_rock_flux_error = setup.sol.error_rock_flux
    actual_frac_pressure_error = setup.sol.error_frac_pressure
    actual_frac_flux_error = setup.sol.error_frac_flux
    actual_intf_flux_error = setup.sol.error_intf_flux

    np.testing.assert_allclose(
        actual_rock_pressure_error, desired_rock_pressure_error, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual_rock_flux_error, desired_rock_flux_error, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual_frac_pressure_error, desired_frac_pressure_error, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual_frac_flux_error, desired_frac_flux_error, atol=1e-5, rtol=1e-3
    )

    np.testing.assert_allclose(
        actual_intf_flux_error, desired_intf_flux_error, atol=1e-5, rtol=1e-3
    )
