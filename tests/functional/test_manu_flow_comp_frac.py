"""
This module contains light-weighted functional tests for a manufactured solution of
the compressible flow with a single, fully embedded vertical fracture.

The manufactured solution for the compressible flow verification is obtained as a
natural extension of the incompressible case, see [1]. The non-linearity is included
via the dependency of the fluid density with the fluid pressure as given in [3]:

.. math:

    \rho(p) = \rho_0 \\exp{(c_f (p - p_0))},

where, for our setup, :math:`\\rho_0 = 1` [kg * m^-3], :math:`c_f = 0.2` [Pa^-1],
and :math:`p_0 = 0` [Pa]. The rest of the physical parameter are given unitary
values, except for the reference porosity :math:`\\phi_0 = 0.1` and normal
permeability :math:`\\kappa = 0.5`.

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

import numpy as np
import pytest

import porepy as pp
from tests.functional.setups.functional_utils import DesiredValuesFlow
from tests.functional.setups.manu_flow_comp_frac import (
    ManuCompFlowSetup,
    ManuCompSaveData,
    manu_comp_fluid,
    manu_comp_solid,
)


@pytest.fixture(scope="module")
def results() -> list[ManuCompSaveData]:
    """Run verification setup and retrieve results for the scheduled times."""
    material_constants = {
        "solid": pp.SolidConstants(manu_comp_solid),
        "fluid": pp.FluidConstants(manu_comp_fluid),
    }
    mesh_arguments = {"mesh_size_frac": 0.125}
    time_manager = pp.TimeManager([0, 0.2, 0.8, 1.0], 0.2, True)
    params = {
        "material_constants": material_constants,
        "mesh_arguments": mesh_arguments,
        "time_manager": time_manager,
    }
    setup = ManuCompFlowSetup(params)
    pp.run_time_dependent_model(setup, params)
    return setup.results


@pytest.fixture(scope="module")
def desired() -> list[DesiredValuesFlow]:
    """Set desired errors."""
    desired_error_0 = DesiredValuesFlow(  # t = 0.2 [s]
        error_matrix_pressure=0.0036348942968626903,
        error_matrix_flux=0.004269623457013436,
        error_frac_pressure=0.6583594878262145,
        error_frac_flux=0.0004147458073966614,
        error_intf_flux=0.4694294365244132,
    )
    desired_error_1 = DesiredValuesFlow(  # t = 0.8 [s]
        error_matrix_pressure=0.003028196249337853,
        error_matrix_flux=0.0029348589540767584,
        error_frac_pressure=0.6596212819386692,
        error_frac_flux=0.0005294349183034246,
        error_intf_flux=0.4943796145199786,
    )
    desired_error_2 = DesiredValuesFlow(  # t = 1.0 [s]
        error_matrix_pressure=0.0028115632547158387,
        error_matrix_flux=0.002959106297324448,
        error_frac_pressure=0.651546529657672,
        error_frac_flux=0.0005572271352671979,
        error_intf_flux=0.49619460256620995,
    )
    return [desired_error_0, desired_error_1, desired_error_2]


@pytest.mark.parametrize("tst_idx", [0, 1, 2])
def test_manufactured_flow_compressible_fractured_2d(tst_idx, results, desired):
    """Check L2-relative errors for primary and secondary variables.

    Nest test functions to avoid computing the solution multiple times.

    """
    var_to_test = [
        "error_matrix_pressure",
        "error_matrix_flux",
        "error_frac_pressure",
        "error_frac_pressure",
        "error_intf_flux",
    ]

    for attr in var_to_test:
        np.testing.assert_allclose(
            getattr(results[tst_idx], attr),
            getattr(desired[tst_idx], attr),
            atol=1e-5,
            rtol=1e-3,
        )
