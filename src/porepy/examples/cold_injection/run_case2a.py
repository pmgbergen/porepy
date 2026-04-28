"""2D, 2-phase water flow through horizontal fracture domain with temporal aperture
jump.

Isothermal model with nonlinear preconditioning using the vT flash.

"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta

# os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np

import porepy as pp
from porepy.examples.cold_injection.config import (
    get_default_convergence_criteria,
    get_default_params,
    get_rpc,
    set_schur_complement,
)
from porepy.examples.cold_injection.geometry import HorizontalFractureAndPointWells2D
from porepy.examples.cold_injection.model import (
    ColdInjectionMixins,
    FluidPoreInteraction,
    IsothermalModelTemplate,
    NoFluxRediscretization,
)

V_PRIMARY = True
ISOCHORIC_NPC = True

APERTURE_JUMP_SCHEDULE: list[tuple[float, pp.number]] = [
    (25 * pp.DAY, 100),
]

newton_tol_res = 1e-7
newton_tol_res_isofug = 1e-2
newton_tol_inc = 1.0
max_iterations = 25
iter_range = (15, max_iterations)

T_END_DAYS = 50

time_schedule = [i * pp.DAY for i in range(T_END_DAYS)]
if APERTURE_JUMP_SCHEDULE:
    t_jump = APERTURE_JUMP_SCHEDULE[0][0]
    t = np.array(time_schedule)
    t_before: list[float] = t[t < t_jump].tolist()
    t_after: list[float] = t[t > t_jump].tolist()
    if t_before[-1] < t_jump - pp.HOUR:
        t_before += [t_jump - pp.HOUR]
    time_schedule = t_before + np.arange(t_jump, t_after[0], pp.HOUR).tolist() + t_after

dt_init = pp.DAY * 0.5
dt_min = pp.SECOND
dt_max = np.max(np.diff(np.array(time_schedule)))

time_manager = pp.TimeManager(
    schedule=time_schedule,
    dt_init=dt_init,
    dt_min_max=(dt_min, dt_max),
    iter_max=max_iterations,
    iter_optimal_range=iter_range,
    iter_relax_factors=(0.75, 1.5),
    recomp_factor=0.5,
    recomp_max=10,
    print_info=True,
    atol=5e-15,
)

model_params, solver_params = get_default_params(
    base_permeability=1e-14,
)

# model_params["linear_solver"] = "pypardiso"  # scipy_sparse default
model_params["time_manager"] = time_manager
model_params["times_to_export"] = time_schedule
model_params["meshing_arguments"] = {
    "cell_size": 5.0,
    "cell_size_boundary": 5.0,
    "cell_size_fracture": 1.0,
    "refinement_proximity_multiplier": 1.0,
    "refinement_size_multiplier": 1.0,
    "background_transition_multiplier": 15,
}
# model_params["grid_type"] = "cartesian"
# model_params["meshing_arguments"] = {
#     "cell_size": 10.0,
#     "cell_size_fracture": 10.0,
# }


eos_params = [1e-4, 1e-2, 1e-3, 10.0]
model_params["flash_params"]["gen_arg_params"] = eos_params
model_params["flash_params"]["phase_property_params"] = eos_params
model_params["phase_property_params"] = eos_params
model_params["flash_params"]["global_iteration_stride"] = None
model_params["flash_params"]["solver_params"]["atol_res"] = 1e-5
model_params["flash_params"]["solver_params"]["max_iterations"] = 25

model_params["equilibrium_specification"] = (
    pp.compositional.FlashSpec.vT,
    "persistent-variables",
)
model_params["flash_params"]["compile_args"] = (
    pp.compositional.FlashSpec.pT,
    pp.compositional.FlashSpec.vT,
)

model_params["use_logp_nonlinear_rpc"] = False

solver_params["atol_objective"] = newton_tol_res
solver_params["newton_chop"] = None
solver_params["appleyard_chop"] = 0.3
solver_params["pressure_clip"] = (0.9, 1.1)  # (0.8, 1.2)
solver_params["volume_clip"] = (0.9, 1.1)  # (0.8, 1.2)
if (
    model_params["use_logp_nonlinear_rpc"]
    and solver_params["pressure_clip"] is not None
):
    solver_params["pressure_clip"] = tuple(
        [np.log(c) for c in solver_params["pressure_clip"]]
    )

solver_params["do_armijo_line_search"] = False
solver_params["armijo_line_search_weight"] = 0.9
solver_params["armijo_line_search_incline"] = 1e-4
solver_params["armijo_line_search_max_iterations"] = 20
solver_params["armijo_stop_after_residual_reaches"] = 1e-5

solver_params["do_ntrdc"] = True
solver_params["ntrdc_scale_with_inf"] = True
solver_params["ntrdc_return_nan"] = False
solver_params["ntrdc_eta_3"] = 0.5
solver_params["ntrdc_eta_2"] = 0.1
solver_params["ntrdc_delta_tol"] = 1e-7

solver_params["in_physical_space"] = True


class ModelClass(  # type:ignore
    FluidPoreInteraction,
    NoFluxRediscretization,
    HorizontalFractureAndPointWells2D,
    ColdInjectionMixins,
    IsothermalModelTemplate,
):
    pass


# ModelClass._PRESSURE_BOUNDARY_ON = False
ModelClass._COMPONENT_NAMES = ["H2O"]
ModelClass._IDEAL_COMPONENTS = [pp.compositional.ideal.IdealH2O]
# NOTE water density in mol / m^3 at 15 MPa and 300 K using Peng-Robinson.
ModelClass._TOTAL_INJECTED_MASS = 10 * 47134.59273520758 / (60 * 60)
ModelClass._p_INIT = 10e6
ModelClass._p_OUT = 10e6
ModelClass._p_BC = 10e6
ModelClass._T_INIT = 450.0
ModelClass._T_IN = 450.0
ModelClass._T_BC = 450.0
ModelClass._z_INIT = {"H2O": 1.0}
ModelClass._z_IN = {"H2O": 1.0}
ModelClass._APERTURE_FACTOR_AFTER_TIME = APERTURE_JUMP_SCHEDULE

if ISOCHORIC_NPC:
    ModelClass._ISOCHORIC_NPC_SPEC = pp.compositional.FlashSpec.vT


if __name__ == "__main__":
    timestamp = datetime.today().strftime("%d%B%Y_%H-%M-%S")
    _ajump = False if len(APERTURE_JUMP_SCHEDULE) == 0 else APERTURE_JUMP_SCHEDULE[0][1]
    _stride = model_params["flash_params"]["global_iteration_stride"]
    sub_folder = (
        "CI_CASE2A/"
        f"{timestamp}"
        f"_AJUMP_{_ajump}"
        f"_ICHOR_{bool(ISOCHORIC_NPC)}"
        f"_VPRIM_{bool(V_PRIMARY)}"
        f"_STRIDE_{_stride}"
    )
    model_params["folder_name"] = f"visualization/{sub_folder}"

    print(f"\nStarting simulation : {sub_folder}\n")
    print(
        f"Solver parameters:\n"
        + pp.compositional.safe_sum([f"{k}: {v}\n" for k, v in solver_params.items()])
    )

    model = ModelClass(model_params)  # type:ignore[abstract]

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy").setLevel(logging.DEBUG)
    t_0 = time.time()
    model.prepare_simulation()
    prep_sim_time = time.time() - t_0
    logging.getLogger("porepy").setLevel(logging.INFO)

    model.params["linear_right_preconditioner"] = get_rpc(model)  # type:ignore

    # Defining sub system for Schur complement reduction.
    set_schur_complement(model)  # type:ignore[arg-type]
    if V_PRIMARY:
        model.schur_complement_primary_variables.remove("pressure")
        model.schur_complement_primary_variables.append("fluid_specific_volume")
    solver_params.update(
        get_default_convergence_criteria(
            model, max_iterations, newton_tol_res, newton_tol_inc, newton_tol_res_isofug
        )
    )

    t_0 = time.time()
    pp.run_time_dependent_model(model, solver_params)
    sim_time = time.time() - t_0

    print(f"Simulation prepared after {str(timedelta(seconds=prep_sim_time))}")
    print(f"Simulation finished after {str(timedelta(seconds=sim_time))}")
