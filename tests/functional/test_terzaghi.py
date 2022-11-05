"""
This module contains light-weighted functional tests for Terzaghi's consolidation problem.
"""

import porepy as pp
import numpy as np
import pytest

from porepy.models.applications.terzaghi_model import Terzaghi


# -----> Test pressure error
def test_pressure_error():
    """Checks L2-discrete relative error for the pressure solution"""
    model_params = {
        'alpha_biot': 1.0,  # [-]
        'height': 1.0,  # [m]
        'lambda_lame': 1.65E9,  # [Pa]
        'mu_lame': 1.475E9,  # [Pa]
        'num_cells': 20,
        'permeability': 9.86E-14,  # [m^2]
        'perturbation_factor': 1E-6,
        'plot_results': False,
        'specific_weight': 9.943E3,  # [Pa * m^-1]
        'time_manager': pp.TimeManager([0, 0.05, 0.1, 0.3], 0.05, constant_dt=True),
        'upper_limit_summation': 1000,
        'use_ad': True,
        'vertical_load': 6E8,  # [N * m^-1]
        'viscosity': 1E-3,  # [Pa * s]
    }
    model = Terzaghi(params=model_params)
    pp.run_time_dependent_model(model, model.params)
    desired = [0, 0.059455475183486366, 0.04111629146957893, 0.020505285438958992]
    actual = [model.sol[key]["p_error"] for key in model.sol.keys()]
    np.testing.assert_array_almost_equal(actual, desired, 1E-6)


