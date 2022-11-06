"""
This module contains light-weighted functional tests for Terzaghi's consolidation problem.
"""

import porepy as pp
import numpy as np

from porepy.models.applications.terzaghi_model import Terzaghi


def test_pressure_and_consolidation_degree():
    """Checks error for pressure solution and degree of consolidation.

    Model parameters have been taken from [1].

    [1] Mikelić, A., Wang, B., & Wheeler, M. F. (2014). Numerical convergence study of
        iterative coupling for coupled flow and geomechanics. Computational Geosciences,
        18(3), 325-341.
    """

    model_params = {
        "alpha_biot": 1.0,  # [-]
        "height": 1.0,  # [m]
        "lambda_lame": 1.65e9,  # [Pa]
        "mu_lame": 1.475e9,  # [Pa]
        "num_cells": 10,
        "permeability": 9.86e-14,  # [m^2]
        "perturbation_factor": 1e-6,
        "plot_results": False,
        "specific_weight": 9.943e3,  # [Pa * m^-1]
        "time_manager": pp.TimeManager([0, 0.05, 0.1, 0.3], 0.05, constant_dt=True),
        "upper_limit_summation": 1000,
        "use_ad": True,
        "vertical_load": 6e8,  # [N * m^-1]
        "viscosity": 1e-3,  # [Pa * s]
    }
    model = Terzaghi(params=model_params)
    pp.run_time_dependent_model(model, model.params)

    # Check pressure error
    desired_error_p = [0.0, 0.06884914805129305, 0.04553528965962772, 0.022392040856504078]
    actual_error_p = [sol.pressure_error for sol in model.solutions]
    np.testing.assert_allclose(actual_error_p, desired_error_p, rtol=1e-3, atol=1e-5)

    # Check consolidation degree
    desired_error_consol = [
        0.0,
        0.02700528627244403,
        0.01883909437589315,
        0.010927389539718946,
    ]
    actual_error_consol = [sol.consolidation_degree_error for sol in model.solutions]
    np.testing.assert_allclose(actual_error_consol, desired_error_consol, rtol=1e-3, atol=1e-5)
