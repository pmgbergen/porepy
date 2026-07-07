"""2D, 2-phase water flow through horizontal fracture domain with temporal aperture
jump.

Thermal model with nonlinear preconditioning using the uv flash.
Temperature is initially constant, during injection and on the boundary.
Temperature drop is expected when fracture opens.

Full uv-based model with volume and internal energy as primary variables.

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
    NoFluxRediscretization,
)
from porepy.examples.cold_injection.run_case2a import (
    Case2DataCollection,
    dt_init,
    dt_min,
    eos_params,
    get_case2_argparser,
    modify_schedule,
    resolve_args,
    time_schedule,
)
from porepy.models.compositional_flow_with_equilibrium import CFLEModelTemplate

max_iterations = 30
iter_range = (15, 25)
newton_tol_res = 1e-5
newton_tol_res_isofug = 1e-2
newton_tol_inc = 1.0

model_params, solver_params = get_default_params()

# model_params["linear_solver"] = "pypardiso"  # scipy_sparse default
# model_params["times_to_export"] = time_schedule

model_params["flash_params"]["gen_arg_params"] = eos_params
model_params["flash_params"]["phase_property_params"] = eos_params
model_params["phase_property_params"] = eos_params
model_params["flash_params"]["global_iteration_stride"] = None
model_params["flash_params"]["solver_params"]["atol_res"] = 1e-5
model_params["flash_params"]["solver_params"]["max_iterations"] = 80
model_params["flash_params"]["solver_params"]["rpc_p"] = -1

model_params["equilibrium_specification"] = (
    pp.compositional.FlashSpec.vu,
    "persistent-variables",
)
model_params["flash_params"]["compile_args"] = (
    pp.compositional.FlashSpec.pT,
    pp.compositional.FlashSpec.vu,
)

solver_params["atol_objective"] = newton_tol_res
solver_params["newton_chop"] = None
solver_params["appleyard_chop"] = 0.3
solver_params["pressure_clip"] = (0.9, 1.1)  # (0.8, 1.2)
solver_params["volume_clip"] = (0.9, 1.1, 2e-5)  # (0.8, 1.2)
solver_params["energy_clip"] = (0.9, 1.1)  # (0.8, 1.2)
model_params["use_logp_nonlinear_rpc"] = False

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
    Case2DataCollection,
    FluidPoreInteraction,
    NoFluxRediscretization,
    HorizontalFractureAndPointWells2D,
    ColdInjectionMixins,
    CFLEModelTemplate,
):
    pass


model_params["create_fluid_volume_variable"] = True
model_params["create_fluid_internal_energy_variable"] = True
model_params["create_fluid_enthalpy_variable"] = False


ModelClass._HEATED_BOUNDARY_ON = False
ModelClass._COMPONENT_NAMES = ["H2O"]
ModelClass._IDEAL_COMPONENTS = [pp.compositional.ideal.IdealH2O]
# NOTE water density in mol / m^3 at 15 MPa and 300 K using Peng-Robinson.
ModelClass._TOTAL_INJECTED_MASS = 10 * 47134.59273520758 / (60 * 60)
ModelClass._p_INIT = 10e6
ModelClass._p_OUT = 10e6
ModelClass._p_BC = 10e6
ModelClass._T_INIT = 450.0
ModelClass._T_IN = 450.0
ModelClass._T_BC = 450.0  # 640.
ModelClass._z_INIT = {"H2O": 1.0}
ModelClass._z_IN = {"H2O": 1.0}


if __name__ == "__main__":
    parser = get_case2_argparser("CI Case 2e.")
    APERTURE_JUMP_SCHEDULE, E_PRIMARY, ISOCHORIC_NPC = resolve_args(parser.parse_args())

    # NOTE for debugging
    from porepy.examples.cold_injection.run_case2a import JUMP_TIME

    APERTURE_JUMP_SCHEDULE = [(JUMP_TIME, 3.0)]

    ajump: float | None
    if APERTURE_JUMP_SCHEDULE:
        ajump = APERTURE_JUMP_SCHEDULE[0][1]
        time_schedule = modify_schedule(time_schedule)
        time_schedule[25] = JUMP_TIME - 5.0
    else:
        ajump = None

    ModelClass._APERTURE_FACTOR_AFTER_TIME = APERTURE_JUMP_SCHEDULE
    model_params["time_manager"] = pp.TimeManager(
        schedule=time_schedule,
        dt_init=dt_init,
        dt_min_max=(dt_min, np.max(np.diff(np.array(time_schedule)))),
        iter_max=max_iterations,
        iter_optimal_range=iter_range,
        iter_relax_factors=(0.75, 1.5),
        recomp_factor=0.5,
        recomp_max=10,
        print_info=True,
        atol=5e-15,
    )

    if ISOCHORIC_NPC:
        ModelClass._ISOCHORIC_NPC_SPEC = pp.compositional.FlashSpec.vu

    timestamp = datetime.today().strftime("%d%B%Y_%H-%M-%S")
    sub_folder = (
        f"CI_CASE2E/"
        f"{timestamp}"
        f"_AJUMP_{ajump}"
        f"_ICHOR_{bool(ISOCHORIC_NPC)}"
        # f"_STRIDE_{model_params["flash_params"]["global_iteration_stride"]}"
        f"_EPRIM_{bool(E_PRIMARY)}"
    )
    model_params["folder_name"] = f"visualization/{sub_folder}"

    print(
        f"\nStarting simulation : {sub_folder}\n"
        f"Aperture jump: {ajump}\n"
        f"Extensives primary: {E_PRIMARY}\n"
        f"Do isochoric NPC: {ISOCHORIC_NPC}\n"
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
    set_schur_complement(model, use_extensives=E_PRIMARY)  # type:ignore[arg-type]
    solver_params.update(
        get_default_convergence_criteria(
            model,
            max_iterations,
            newton_tol_res,
            newton_tol_inc,
            newton_tol_res_isofug,
            atol_div=1e12,
        )
    )

    t_0 = time.time()
    pp.run_time_dependent_model(model, solver_params)
    sim_time = time.time() - t_0

    print(f"Simulation prepared after {str(timedelta(seconds=prep_sim_time))}")
    print(f"Simulation finished after {str(timedelta(seconds=sim_time))}")
