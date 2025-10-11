"""mD Model of cold CO2 injection with randomized seed."""

from __future__ import annotations

import logging
import os
import time
import warnings

from typing import Any

# os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
import porepy as pp
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.applications.material_values.solid_values import basalt
from porepy.examples.cold_co2_injection.model import (
    ColdCO2InjectionModelFF,
    BuoyancyModel,
    ColdCO2InjectionModel,
    NoFluxRediscretization,
)
from porepy.examples.cold_co2_injection.solver import NewtonArmijoAndersonSolver
from porepy.applications.test_utils.models import add_mixin


warnings.filterwarnings("ignore", category=RuntimeWarning)
BUOYANCY_ON = False
VERBOSE = True

max_iterations = 40 if BUOYANCY_ON else 30
iter_range = (21, 35) if BUOYANCY_ON else (15, 25)
newton_tol = 1e-5
newton_tol_increment = 1e-5
T_end_months = 100

time_schedule = [i * 30 * pp.DAY for i in range(T_end_months + 1)]
dt_init = pp.HOUR
dt_min = pp.MINUTE
dt_max = 30 * pp.DAY

time_manager = pp.TimeManager(
    schedule=time_schedule,
    dt_init=dt_init,
    dt_min_max=(dt_min, dt_max),
    iter_max=max_iterations,
    iter_optimal_range=iter_range,
    iter_relax_factors=(0.75, 2),
    recomp_factor=0.6,
    recomp_max=10,
    print_info=VERBOSE,
    rtol=0.0,
)

phase_property_params = {
    "phase_property_params": [0.0],
}

basalt_ = basalt.copy()
basalt_["permeability"] = 1e-14
well_surrounding_permeability = 1e-13
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
    "armijo_line_search_weight": 0.95,
    "armijo_line_search_incline": 0.2,
    "armijo_line_search_max_iterations": 10,
    "armijo_start_after_residual_reaches": np.inf,
    "armijo_stop_after_residual_reaches": 1e-3,
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
        "cell_size": 2.,
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
    "fracture_permeability": 1e-12,
    "impermeable_fracture_permeability": 1e-12,
    "_num_fractures": 5,
    "_well_surrounding_permeability": well_surrounding_permeability,
    "folder_name": f"visualization/md_case/",
    "progressbars": not VERBOSE,
}

model_params.update(phase_property_params)
model_params.update(solver_params)
model_params.update(meshing_params)


if __name__ == "__main__":
    if BUOYANCY_ON:
        model_class = add_mixin(BuoyancyModel, ColdCO2InjectionModelFF)
        # model_class = add_mixin(pp.constitutive_laws.DarcysLawAd, model_class)
    else:
        model_class = add_mixin(NoFluxRediscretization, ColdCO2InjectionModel)

    model = model_class(model_params)

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy").setLevel(logging.DEBUG)
    t_0 = time.time()
    model.prepare_simulation()
    prep_sim_time = time.time() - t_0
    if VERBOSE:
        logging.getLogger("porepy").setLevel(logging.INFO)
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
