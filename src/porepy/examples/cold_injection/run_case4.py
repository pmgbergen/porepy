"""Cold CO2 injection in 3D fractured setting."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

os.environ["NUMBA_DISABLE_JIT"] = "1"

import porepy as pp
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.examples.cold_injection.config import (
    get_default_convergence_criteria,
    get_default_params,
    set_schur_complement,
)
from porepy.examples.cold_injection.geometry import GeometryBenchmark3d_case4
from porepy.examples.cold_injection.model import (
    BuoyancyModel,
    ColdInjectionMixins,
    NoFluxRediscretization,
)

BUOYANCY_ON = False

max_iterations = 40 if BUOYANCY_ON else 30
iter_range = (21, 35) if BUOYANCY_ON else (15, 25)
newton_tol_res = 1e-5
newton_tol_inc = 1e-3
newton_tol_res_isofug = 1e-2
T_end_months = 30

time_schedule = [i * 30 * pp.DAY for i in range(T_end_months + 1)]
dt_init = 3 * pp.HOUR
dt_min = pp.HOUR

time_manager = pp.TimeManager(
    schedule=time_schedule,
    dt_init=dt_init,
    dt_min_max=(dt_min, 30 * pp.DAY),
    iter_max=max_iterations,
    iter_optimal_range=iter_range,
    iter_relax_factors=(0.75, 2),
    recomp_factor=0.5,
    recomp_max=10,
    print_info=True,
    rtol=0.0,
)

model_params, solver_params = get_default_params()
model_params["max_iterations"] = max_iterations
model_params["nl_convergence_tol"] = newton_tol_res
model_params["nl_convergence_tol_increment"] = newton_tol_inc
model_params["time_manager"] = time_manager
model_params["times_to_export"] = time_schedule

model_params["fractional_flow"] = BUOYANCY_ON
model_params["enable_buoyancy_effects"] = BUOYANCY_ON


if BUOYANCY_ON:
    # 'Yippee Ki Yay, motherfucker' - John McClane, Die Hard.
    class ModelClass(  # type:ignore[misc]
        BuoyancyModel,
        GeometryBenchmark3d_case4,
        ColdInjectionMixins,
        cfle.EnthalpyBasedCFFLETemplate,
    ):
        pass

else:

    class ModelClass(  # type:ignore
        NoFluxRediscretization,
        GeometryBenchmark3d_case4,
        ColdInjectionMixins,
        cfle.EnthalpyBasedCFLETemplate,
    ):
        pass


ModelClass._PERM_AROUND_WELLS = 1e-13
ModelClass._HEATED_BOUNDARY_ON = True
ModelClass._COMPONENT_NAMES = ["H2O", "CO2"]
ModelClass._IDEAL_COMPONENTS = [
    pp.compositional.ideal.IdealH2O,
    pp.compositional.ideal.IdealCO2,
]
# NOTE density of injected mixture under T_IN p_OUT conditions
ModelClass._TOTAL_INJECTED_MASS = 10 * 27430.998956110157 / (60 * 60)
ModelClass._p_INIT = 20e6
ModelClass._p_OUT = 19e6
ModelClass._p_BC = 19e6
ModelClass._T_INIT = 450.0
ModelClass._T_IN = 300.0
ModelClass._T_BC = 640.0
ModelClass._z_INIT = {"H2O": 0.995, "CO2": 0.005}
ModelClass._z_IN = {"H2O": 0.9, "CO2": 0.1}


if __name__ == "__main__":
    timestamp = datetime.today().strftime("%d%B%Y_%H-%M-%S")
    sub_folder = f"CI_CASE4/{timestamp}_BUOY_{BUOYANCY_ON}"
    model_params["folder_name"] = f"visualization/{sub_folder}"

    model = ModelClass(model_params)  # type:ignore[abstract]

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy").setLevel(logging.DEBUG)
    t_0 = time.time()
    model.prepare_simulation()
    prep_sim_time = time.time() - t_0
    logging.getLogger("porepy").setLevel(logging.INFO)

    # Defining sub system for Schur complement reduction.
    set_schur_complement(model)  # type:ignore[arg-type]
    solver_params.update(
        get_default_convergence_criteria(
            model, max_iterations, newton_tol_res, newton_tol_inc, newton_tol_res_isofug
        )
    )

    t_0 = time.time()
    pp.run_time_dependent_model(model, solver_params)
    sim_time = time.time() - t_0

    print(f"Simulation prepared after {prep_sim_time:.2f} (s).")
    print(f"Simulation finished after {sim_time / 60.0:.2f} (m).")
