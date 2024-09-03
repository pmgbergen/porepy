"""
Module containing functional tests for Mandel's problem.

We consider two functional tests:

    - The first test compares desired and actual relative errors for the pressure,
      displacement, flux, poroelastic force, and degrees of consolidation, for two
      different times, namely 25 and 50 seconds.

    - The second test checks that the same errors for pressure and displacement (up to
      5 decimals) are obtained for scaled and unscaled problems. Scaling is
      performed for length and mass.

"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest

import porepy as pp

from porepy.examples.mandel_biot import (
    MandelSaveData,
    MandelSetup,
    mandel_fluid_constants,
    mandel_solid_constants,
)


@pytest.fixture(scope="module")
def results() -> list[MandelSaveData]:
    # Run verification setup and retrieve results for three different times
    material_constants = {
        "fluid": pp.FluidConstants(mandel_fluid_constants),
        "solid": pp.SolidConstants(mandel_solid_constants),
    }
    time_manager = pp.TimeManager([0, 25, 50], 25, True)
    model_params = {
        "material_constants": material_constants,
        "time_manager": time_manager,
    }
    setup = MandelSetup(model_params)
    pp.run_time_dependent_model(setup)
    return setup.results


# Desired errors
DesiredError = namedtuple(
    "DesiredError",
    "error_pressure, "
    "error_flux, "
    "error_displacement, "
    "error_force, "
    "error_consolidation_degree",
)

desired_errors: list[DesiredError] = [
    # t = 25 [s]
    DesiredError(
        error_pressure=0.02468899282842615,
        error_flux=0.4277241964548815,
        error_displacement=0.0007426275934204356,
        error_force=0.007199561782640474,
        error_consolidation_degree=(0.005794353839056594, 2.983724378680108e-16),
    ),
    # t = 50 [s]
    DesiredError(
        error_pressure=0.01604388082216944,
        error_flux=0.17474770500364722,
        error_displacement=0.000710048736369786,
        error_force=0.0047015988044755526,
        error_consolidation_degree=(0.004196023265526011, 2.983724378680108e-16),
    ),
]


@pytest.mark.parametrize("time_index", [0, 1])
def test_error_primary_and_secondary_variables(time_index: int, results):
    """Checks error for pressure, displacement, flux, force, and consolidation degree.

    Physical parameters used in this test have been adapted from [1].

    Tests should pass as long as a `desired_error` matches the `actual_error` up to
    absolute and relative tolerances. In this particular case, we are comparing pressure
    errors, flux errors, displacement errors, poroelastic force errors (measured
    using the L2-discrete relative error given in [2]) and degrees of consolidation
    errors in the horizontal and vertical directions (measured in absolute terms). The
    `desired_error` was obtained by copying the simulation results at the time this
    test was written.

    For this test, we require an absolute tolerance of 1e-5 and a relative tolerance of
    1e-3 between the `desired_error` and the `actual_error`. The choice of such
    tolerances aim at keeping the test meaningful while minimizing the chances of
    failure due to floating-point arithmetic close to machine precision.

    References:

        - [1] Mikelić, A., Wang, B., & Wheeler, M. F. (2014). Numerical convergence
          study of iterative coupling for coupled flow and geomechanics.
          Computational Geosciences, 18(3), 325-341.

        - [2] Nordbotten, J. M. (2016). Stable cell-centered finite volume
          discretization for Biot equations. SIAM Journal on Numerical Analysis,
          54(2), 942-968.

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

    np.testing.assert_allclose(
        actual.error_consolidation_degree[0],
        desired.error_consolidation_degree[0],
        atol=1e-5,
        rtol=1e-3,
    )

    np.testing.assert_allclose(
        actual.error_consolidation_degree[1],
        desired.error_consolidation_degree[1],
        atol=1e-5,
        rtol=1e-3,
    )


def test_scaled_vs_unscaled_systems():
    """Checks that the same results are obtained for scaled and unscaled systems."""

    # The unscaled problem
    material_constants_unscaled = {
        "fluid": pp.FluidConstants(mandel_fluid_constants),
        "solid": pp.SolidConstants(mandel_solid_constants),
    }
    time_manager_unscaled = pp.TimeManager([0, 10], 10, True)
    model_params_unscaled = {
        "material_constants": material_constants_unscaled,
        "time_manager": time_manager_unscaled,
    }
    model_unscaled = MandelSetup(params=model_params_unscaled)
    pp.run_time_dependent_model(model_unscaled)

    # The scaled problem
    material_constants_scaled = {
        "fluid": pp.FluidConstants(mandel_fluid_constants),
        "solid": pp.SolidConstants(mandel_solid_constants),
    }
    time_manager_scaled = pp.TimeManager([0, 10], 10, True)
    scaling = {"m": 1e-3, "kg": 1e-3}  # length in millimeters and mass in grams
    units = pp.Units(**scaling)
    model_params_scaled = {
        "material_constants": material_constants_scaled,
        "time_manager": time_manager_scaled,
        "units": units,
    }
    scaled_model = MandelSetup(params=model_params_scaled)
    pp.run_time_dependent_model(model=scaled_model)

    # Compare results
    np.testing.assert_almost_equal(
        model_unscaled.results[-1].error_pressure,
        scaled_model.results[-1].error_pressure,
        decimal=5,
    )
    np.testing.assert_almost_equal(
        model_unscaled.results[-1].error_displacement,
        scaled_model.results[-1].error_displacement,
        decimal=5,
    )
