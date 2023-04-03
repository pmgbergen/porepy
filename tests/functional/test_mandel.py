"""
Module containing functional tests for Mandel's problem.

We consider two functional tests:

    - The first test compares desired and actual relative errors for the pressure,
      displacement, flux, poroelastic force, and degrees of consolidation, for three
      different times, namely 10, 30, and 50 seconds.

    - The second test checks that the same errors for pressure and displacement (up to
      5 decimals) are obtained for scaled and unscaled problems. Scaling is
      performed for length and mass.

"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest

import porepy as pp
from porepy.applications.complete_setups.mandel_biot import (
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
    time_manager = pp.TimeManager([0, 10, 30, 50], 10, True)
    params = {
        "material_constants": material_constants,
        "time_manager": time_manager,
    }
    setup = MandelSetup(params)
    pp.run_time_dependent_model(setup, params)
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
    # t = 10 [s]
    DesiredError(
        error_pressure=0.020915626239924504,
        error_flux=0.4673715742577367,
        error_displacement=0.00036006562251599753,
        error_force=0.006174477658138146,
        error_consolidation_degree=(0.004112823366753098, 2.983724378680108e-16),
    ),
    # t = 30 [s]
    DesiredError(
        error_pressure=0.01002508912855953,
        error_flux=0.11759895811865957,
        error_displacement=0.0003363008805572937,
        error_force=0.003024676802218547,
        error_consolidation_degree=(0.002324983805104465, 2.983724378680108e-16),
    ),
    # t = 50 [s]
    DesiredError(
        error_pressure=0.007059420526689479,
        error_flux=0.06788475278093752,
        error_displacement=0.0003094287000315224,
        error_force=0.002150211938046511,
        error_consolidation_degree=(0.0018043500295539042, 2.983724378680108e-16),
    ),
]


@pytest.mark.parametrize("time_index", [0, 1, 2])
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
    params_unscaled = {
        "material_constants": material_constants_unscaled,
        "time_manager": time_manager_unscaled,
    }
    unscaled = MandelSetup(params=params_unscaled)
    pp.run_time_dependent_model(model=unscaled, params=params_unscaled)

    # The scaled problem
    material_constants_scaled = {
        "fluid": pp.FluidConstants(mandel_fluid_constants),
        "solid": pp.SolidConstants(mandel_solid_constants),
    }
    time_manager_scaled = pp.TimeManager([0, 10], 10, True)
    scaling = {"m": 1e-3, "kg": 1e-3}  # length in millimeters and mass in grams
    units = pp.Units(**scaling)
    params_scaled = {
        "material_constants": material_constants_scaled,
        "time_manager": time_manager_scaled,
        "units": units,
    }
    scaled = MandelSetup(params=params_scaled)
    pp.run_time_dependent_model(model=scaled, params=params_scaled)

    # Compare results
    np.testing.assert_almost_equal(
        unscaled.results[-1].error_pressure,
        scaled.results[-1].error_pressure,
        decimal=5,
    )
    np.testing.assert_almost_equal(
        unscaled.results[-1].error_displacement,
        scaled.results[-1].error_displacement,
        decimal=5,
    )
