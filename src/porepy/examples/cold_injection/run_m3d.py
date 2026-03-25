"""Cold CO2 injection in 3D fractured setting."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np

import porepy as pp
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.applications.test_utils.models import add_mixin
from porepy.examples.cold_injection.config import (
    get_default_convergence_criteria,
    get_default_params,
)
from porepy.examples.cold_injection.model import (
    BuoyancyModel,
    ColdInjectionMixins,
    DataCollectionMixin,
    ModelConfig,
    NoFluxRediscretization,
    set_schur_complement,
)
from porepy.examples.flow_benchmark_3d_case_4 import Geometry as Geometry3D

BUOYANCY_ON = False
COLLECT_DATA = False

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

model_params["_well_surrounding_permeability"] = 1e-13
model_params["_impermeable_fracture_permeability"] = 1e-16
model_params["_fracture_permeability"] = 1e-10

model_params["fractional_flow"] = BUOYANCY_ON
model_params["enable_buoyancy_effects"] = BUOYANCY_ON
model_params["_heated_boundary_on"] = False


class Mixin3d(ModelConfig):
    """Modifies the setup to be compatible with 3D."""

    def _heated_boundary_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define heated boundary with D-type conditions for conductive flux."""
        sides = self.domain_boundary_sides(sd)

        heated = np.zeros(sd.num_faces, dtype=bool)
        heated[sides.bottom] = True

        x = sd.face_centers[0]
        y = sd.face_centers[0]

        L = self.domain.bounding_box["xmax"] - self.domain.bounding_box["xmin"]
        W = self.domain.bounding_box["ymax"] - self.domain.bounding_box["ymin"]
        x0 = self.domain.bounding_box["xmin"] + 0.5 * L
        y0 = self.domain.bounding_box["ymin"] + 0.5 * W
        r = 0.5 * min(W, L)
        circle = (x - x0) ** 2 + (y - y0) ** 2 <= r**2

        heated &= circle

        return heated


if BUOYANCY_ON:
    # 'Yippee Ki Yay, motherfucker' - John McClane, Die Hard.
    class ModelClass(  # type:ignore[misc]
        Mixin3d,
        BuoyancyModel,
        Geometry3D,
        ColdInjectionMixins,
        cfle.EnthalpyBasedCFFLETemplate,
    ):
        pass

else:

    class ModelClass(  # type:ignore
        Mixin3d,
        NoFluxRediscretization,
        Geometry3D,
        ColdInjectionMixins,
        cfle.EnthalpyBasedCFLETemplate,
    ):
        pass


model_class = ModelClass

if COLLECT_DATA:
    model_class = add_mixin(DataCollectionMixin, model_class)  # type:ignore


if __name__ == "__main__":
    timestamp = datetime.today().strftime("%d%B%Y_%I-%M-%S")
    sub_folder = f"m3d_{timestamp}_BUOY_{BUOYANCY_ON}"
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
