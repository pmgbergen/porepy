"""mD Model of cold CO2 injection with randomized seed."""

from __future__ import annotations

import logging
import os
import time
import warnings

from typing import Any

os.environ["LINE_PROFILE"] = "1"
from line_profiler import profile

# os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
import porepy as pp
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.applications.material_values.solid_values import basalt
from porepy.examples.cold_co2_injection.model import (
    ColdInjectionModelFF,
    BuoyancyModel,
    ColdInjectionModel,
    NoFluxRediscretization,
)
from porepy.examples.cold_co2_injection.solver import NewtonArmijoAndersonSolver


warnings.filterwarnings("ignore", category=RuntimeWarning)
BUOYANCY_ON = False
VERBOSE = True

# max_iterations = 40 if BUOYANCY_ON else 30
max_iterations = 30
iter_range = (21, 28) if BUOYANCY_ON else (15, 25)
newton_tol = 1e-3
newton_tol_increment = 1e-3
T_end_months = 100

time_schedule = [i * pp.DAY for i in range(121)]
assert T_end_months > 5
time_schedule += [i * 30 * pp.DAY for i in range(5, T_end_months + 1)]

time_schedule = time_schedule[:4]
dt_init = 20 * pp.MINUTE
dt_min = 10 * pp.MINUTE
dt_max = 3 * pp.HOUR

time_manager = pp.TimeManager(
    schedule=time_schedule,
    dt_init=dt_init,
    dt_min_max=(dt_min, dt_max),
    iter_max=max_iterations,
    iter_optimal_range=iter_range,
    iter_relax_factors=(0.75, 1.5),
    recomp_factor=0.5,
    recomp_max=10,
    print_info=VERBOSE,
    rtol=0.0,
)

phase_property_params = {
    "phase_property_params": [0.0],
}

basalt_ = basalt.copy()
basalt_["permeability"] = 1e-14
well_surrounding_permeability = 1e-14
material_params = {"solid": pp.SolidConstants(**basalt_)}

flash_params: dict[Any, Any] = {
    "mode": "parallel",
    "solver": "npipm",
    "solver_params": {
        "tolerance": 1e-3,
        "max_iterations": 80,  # 150
        "armijo_rho": 0.99,
        "armijo_kappa": 0.4,
        "armijo_max_iterations": 30,
        "npipm_u1": 10,
        "npipm_u2": 10,
        "npipm_eta": 0.5,
    },
    "global_iteration_stride": 3,
    "fallback_to_iterate": True,
}
flash_params.update(phase_property_params)

solver_params = {
    "max_iterations": max_iterations,
    "nl_convergence_tol": newton_tol_increment,
    "nl_convergence_tol_res": newton_tol,
    "apply_schur_complement_reduction": True,
    # "linear_solver": "scipy_sparse",
    "linear_solver": "pypardiso",
    "nonlinear_solver": NewtonArmijoAndersonSolver,
    "armijo_line_search": True,
    "armijo_line_search_weight": 0.9,
    "armijo_line_search_incline": 0.2,
    "armijo_line_search_max_iterations": 10,
    "armijo_start_after_residual_reaches": np.inf,
    "armijo_stop_after_residual_reaches": 1e-5,
    "appplyard_chop": 0.2,
    "anderson_acceleration": False,
    "anderson_acceleration_depth": 3,
    "anderson_acceleration_constrained": True,
    "anderson_acceleration_regularization_parameter": 1e-3,
    "anderson_acceleration_relaxation_parameter": 0.,
    "anderson_start_after_residual_reaches": np.inf,
    "anderson_stop_after_residual_reaches": 1e1,
    "solver_statistics_file_name": "solver_statistics.json",
    "flag_failure_as_diverged": True,
}

meshing_params = {
    "grid_type": "simplex",
    "meshing_arguments": {
        "cell_size": 1.,
        "cell_size_fracture": 1.,
    },
}

model_params: dict[str, Any] = {
    "equilibrium_condition": "unified-p-h",
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "flash_params": flash_params,
    "fractional_flow": BUOYANCY_ON,
    "material_constants": material_params,
    "time_manager": time_manager,
    "prepare_simulation": False,
    "enable_buoyancy_effects": BUOYANCY_ON,
    "compile": True,
    "flash_compiler_args": ("p-T", "p-h"),
    "_lbc_viscosity": False,
    "fracture_permeability": 1e-11,
    "impermeable_fracture_permeability": 1e-11,
    "_num_fractures": 50,
    "_well_surrounding_permeability": well_surrounding_permeability,
    "folder_name": f"visualization/md_case/",
    "progressbars": not VERBOSE,
    "times_to_export": time_schedule,
}

model_params.update(phase_property_params)
model_params.update(solver_params)
model_params.update(meshing_params)

class ModelSetup(
    BuoyancyModel if BUOYANCY_ON else NoFluxRediscretization,
    ColdInjectionModelFF if BUOYANCY_ON else ColdInjectionModel,
):
    """Model setup for cold CO2 injection."""
    _domain_x_length = 50
    _domain_y_length = 15
    _INJECTION_POINTS = [np.array([5., 3])]
    _PRODUCTION_POINTS= [np.array([45.0, 10])]
    _T_HEATED = 630.0
    _p_INIT = 20e6
    _T_INIT = 400.
    _p_OUT = 18e6
    _TOTAL_INJECTED_MASS: float = 10 * 27430.998956110157 / (60 * 60)

    # _z_INIT: dict[str, float] = {"H2O": 0.995, "CO2": 0.005}
    _z_INIT: dict[str, float] = {"H2O": 0.9, "H2S": 0.1}

    # _z_IN: dict[str, float] = {"H2O": 0.9, "CO2": 0.1}
    _z_IN: dict[str, float] = {"H2O": 0.995, "H2S": 0.005}


if __name__ == "__main__":

    model = ModelSetup(model_params)

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy").setLevel(logging.DEBUG)
    t_0 = time.time()
    model.prepare_simulation()
    prep_sim_time = time.time() - t_0
    if VERBOSE:
        logging.getLogger("porepy").setLevel(logging.DEBUG)
    else:
        logging.getLogger("porepy").setLevel(logging.WARNING)

    # Defining sub system for Schur complement reduction.
    primary_equations = cfle.cf.get_primary_equations_cf(model)
    primary_equations += [
        eq for eq in model.equation_system.equations.keys() if "flux" in eq
    ]
    primary_equations += [
        "production_pressure_constraint",
        "injection_temperature_constraint",
    ]
    primary_variables = cfle.cf.get_primary_variables_cf(model)
    primary_variables += list(
        set([v.name for v in model.equation_system.variables if "flux" in v.name])
    )

    model.schur_complement_primary_equations = primary_equations
    model.schur_complement_primary_variables = primary_variables

    t_0 = time.time()
    SIMULATION_SUCCESS: bool = True
    pp.run_time_dependent_model(model, model_params)
    sim_time = time.time() - t_0
