"""2-phase water flow through single fracture domain with temporal aperture jump.

Isothermal model with nonlinear preconditioning using the vT flash.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

# import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np

import porepy as pp

from porepy.applications.test_utils.models import add_mixin
from porepy.examples.cold_injection.config import (
    get_default_convergence_criteria,
    get_default_params,
)
from porepy.examples.cold_injection.geometry import HorizontalFractureAndPointWells2D
from porepy.examples.cold_injection.model import (
    ColdInjectionMixins,
    DataCollectionMixin,
    FluidPoreInteraction,
    NoFluxRediscretization,
    set_schur_complement,
    IsothermalModelTemplate,
)


ISOCHORIC_NPC = False
BUOYANCY_ON = False
COLLECT_DATA = True

assert not BUOYANCY_ON, "Fractional flow not supported for case 2a."

APERTURE_JUMP_SCHEDULE: list[tuple[float, float]] = [
    # (25 * pp.DAY, 5.0),
]

max_iterations = 40 if BUOYANCY_ON else 30
iter_range = (21, 28) if BUOYANCY_ON else (15, 25)
newton_tol_res = 1e-5
newton_tol_res_isofug = 1e-2
newton_tol_inc = 1e-5

T_END_DAYS = 50

time_schedule = [i * pp.DAY for i in range(T_END_DAYS)]


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
    rtol=0.0,
)

model_params, solver_params = get_default_params(
    base_permeability=1e-14,
)

model_params["time_manager"] = time_manager
# model_params["times_to_export"] = time_schedule
model_params["meshing_arguments"]["cell_size"] = 2.0
model_params["meshing_arguments"]["cell_size_fracture"] = 1.0

model_params["_well_surrounding_permeability"] = 1e-12
model_params["_fracture_permeability"] = 1e-10

model_params["fractional_flow"] = BUOYANCY_ON
model_params["enable_buoyancy_effects"] = BUOYANCY_ON
model_params["_heated_boundary_on"] = False

model_params["flash_params"]["gen_arg_params"] = [1e-4, 1e-2, 1e-3, 10.0]
model_params["flash_params"]["phase_property_params"] = [1e-4, 1e-2, 1e-3, 10.0]
model_params["flash_params"]["global_iteration_stride"] = max_iterations
model_params["phase_property_params"] = [1e-4, 1e-2, 1e-3, 10.0]

model_params["equilibrium_specification"] = (
    pp.compositional.FlashSpec.vT,
    "persistent-variables",
)
if ISOCHORIC_NPC:
    model_params["flash_compiler_args"] = (
        pp.compositional.FlashSpec.pT,
        pp.compositional.FlashSpec.vT,
    )
    model_params["_do_isochoric_npc"] = pp.compositional.FlashSpec.vT
else:
    if model_params["flash_params"]["global_iteration_stride"] < max_iterations:
        model_params["flash_compiler_args"] = (
            pp.compositional.FlashSpec.pT,
            pp.compositional.FlashSpec.vT,
        )
    else:
        model_params["flash_compiler_args"] = (pp.compositional.FlashSpec.pT,)
    model_params["_do_isochoric_npc"] = pp.compositional.FlashSpec.none

model_params["variable_scaling_linear_rpc"] = {
    "pressure": 22064000.0,
    "temperature": 647.096,
    "enthalpy": 524641.0735546586,
}

solver_params["armijo_line_search_weight"] = 0.9
solver_params["armijo_line_search_incline"] = 1e-4
solver_params["armijo_line_search_max_iterations"] = 20
solver_params["armijo_stop_after_residual_reaches"] = 1e-5
solver_params["armijo_least_squares_form"] = False
solver_params["newton_chop"] = 0.4


class Case2aMixin:
    """Model configuration for case 1e: 2-phase water flow."""

    _COMPONENT_NAMES: list[str] = ["H2O"]

    _IDEAL_COMPONENTS: list[pp.compositional.ideal.IdealFluid] = [
        pp.compositional.ideal.IdealH2O,
    ]

    # NOTE water density in mol / m^3 at 15 MPa and 300 K using Peng-Robinson.
    _TOTAL_INJECTED_MASS: float = 10 * 47134.59273520758 / (60 * 60)

    _p_INIT: float = 10e6
    _T_INIT: float = 300.0

    _p_OUT: float = 10e6  # roughly hydrostatic pressure of water at depth of 1 km.
    _T_IN: float = 300.0

    _T_BC: float = 300.0

    _APERTURE_FACTOR_AFTER_TIME = APERTURE_JUMP_SCHEDULE


class ModelClass(  # type:ignore
    Case2aMixin,
    FluidPoreInteraction,
    NoFluxRediscretization,
    HorizontalFractureAndPointWells2D,
    ColdInjectionMixins,
    IsothermalModelTemplate,
):
    pass


model_class = ModelClass

if COLLECT_DATA:
    model_class = add_mixin(DataCollectionMixin, model_class)  # type:ignore


if __name__ == "__main__":
    timestamp = datetime.today().strftime("%d%B%Y_%H-%M-%S")
    sub_folder = (
        "m2d_case2a"
        f"_{timestamp}"
        f"_BUOY_{BUOYANCY_ON}"
        f"_AJUMP_{bool(model_class._APERTURE_FACTOR_AFTER_TIME)}"
        f"_ICHOR_{bool(ISOCHORIC_NPC)}"
    )
    model_params["folder_name"] = f"visualization/{sub_folder}"

    model = model_class(model_params)  # type:ignore[abstract]

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy").setLevel(logging.DEBUG)
    t_0 = time.time()
    model.prepare_simulation()
    prep_sim_time = time.time() - t_0
    logging.getLogger("porepy").setLevel(logging.INFO)

    # Defining sub system for Schur complement reduction.
    set_schur_complement(model)  # type:ignore[arg-type]
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

    print(f"Simulation prepared after {prep_sim_time:.2f} (s).")
    print(f"Simulation finished after {sim_time / 60.0:.2f} (m).")
