"""
Module containing functional tests for Terzaghi's consolidation problem.

We consider two functional test:

    - The first test guarantees that the results obtained with the Biot class
      `cls:porepy.applications.classic_models.Biot` matches `exactly` the results
      obtained with the full poromechanical model `cls:porepy:models.poromechanics`
      when biot_coefficient = 1 and the fluid compressibility = 0.

    - The second test compares desired and actual relative errors for the pressure and
      degree of consolidation obtained from the MPFA/MPSA-FV solutions and the
      exact Terzaghi's solution.

"""

import numpy as np

import porepy as pp

from porepy.models.poromechanics import Poromechanics
from porepy.models.verification_setups.terzaghi import (
    TerzaghiSetup,
    ModifiedGeometry,
    SetupUtilities,
    ModifiedPoromechanicsBoundaryConditions,
    ModifiedSolutionStrategy,
)


class TerzaghiSetupPoromechanics(
    ModifiedPoromechanicsBoundaryConditions,
    ModifiedSolutionStrategy,
    ModifiedGeometry,
    SetupUtilities,
    Poromechanics,
):
    ...


def test_biot_equal_to_incompressible_poromechanics():
    """Checks if the same results are obtained for Biot and incompressible
    poromechanics"""

    # Run Terzaghi setup with full poromechanics model
    params_poromech = {"time_manager": pp.TimeManager([0, 1], 1, True), "num_cells": 10}
    setup_poromech = TerzaghiSetupPoromechanics(params_poromech)
    pp.run_time_dependent_model(setup_poromech, params_poromech)
    p_poromechanics = setup_poromech.results[0].numerical_pressure
    U_poromechanics = setup_poromech.results[0].numerical_consolidation_degree

    # Run Terzaghi setup with Biot model
    params_biot = {"time_manager": pp.TimeManager([0, 1], 1, True), "num_cells": 10}
    setup_biot = TerzaghiSetup(params_biot)
    pp.run_time_dependent_model(setup_biot, params_biot)
    p_biot = setup_biot.results[0].numerical_pressure
    U_biot = setup_biot.results[0].numerical_consolidation_degree

    np.testing.assert_almost_equal(p_poromechanics, p_biot)
    np.testing.assert_almost_equal(U_poromechanics, U_biot)


def test_pressure_and_consolidation_degree_errors():
    """Checks error for pressure solution and degree of consolidation.

    Physical parameters used in this test have been adapted from [1].

    Tests should pass as long as a `desired_error` matches the `actual_error` up to
    absolute and relative tolerances. In this particular case, we are comparing pressure
    errors (measured using the L2-discrete relative error given in [2]) and degree of
    consolidation errors (measured in absolute terms). The `desired_error` was obtained
    by copying the simulation results at the time this test was written.

    For this test, we require an absolute tolerance of 1e-5 and a relative tolerance of
    1e-3 between the `desired_error` and the `actual_error`. The choice of such tolerances
    aim at keeping the test meaningful while minimizing the chances of failure due to
    floating-point arithmetic close to machine precision.

    References:

    [1] Mikelić, A., Wang, B., & Wheeler, M. F. (2014). Numerical convergence study of
        iterative coupling for coupled flow and geomechanics. Computational Geosciences,
        18(3), 325-341.

    [2] Nordbotten, J. M. (2016). Stable cell-centered finite volume discretization for
        Biot equations. SIAM Journal on Numerical Analysis, 54(2), 942-968.

    """

    params = {
        "time_manager": pp.TimeManager([0, 0.05, 0.1, 0.3], 0.05, constant_dt=True),
        "stored_times": [0.05, 0.1, 0.3],
        "num_cells": 10,
    }
    setup = TerzaghiSetup(params)
    pp.run_time_dependent_model(setup, params)

    # Check pressure error
    desired_error_p = [
        0.06884857957987067,
        0.0455347877406611,
        0.022391576732172767
    ]
    actual_error_p = [result.pressure_error for result in setup.results]
    np.testing.assert_allclose(actual_error_p, desired_error_p, rtol=1e-3, atol=1e-5)

    # Check consolidation degree error
    desired_error_consol = [
        0.0270053052913328,
        0.018839104164792175,
        0.010927390550216687
    ]
    actual_error_consol = [
        result.consolidation_degree_error for result in setup.results
    ]
    np.testing.assert_allclose(
        actual_error_consol, desired_error_consol, rtol=1e-3, atol=1e-5
    )
