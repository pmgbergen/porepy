"""Run script for the cold CO2 injection model.

Script is executable from command line. For a list of possible flags and
model configurations run

Windows:

> python.exe ./run.py --help

Linux:

> python run.py --help


Note:
    The code within the docker image from Zenodo should be static and unchanging for
    all time.
    There is also a repository on Github containing a snapshot of the code, which is
    perhaps easier accessible. But the continuous availability of that repo is not
    guaranteed.

Note:
    For some reason, the flag ``-p`` must be given a value, otherwise the parsing won't
    work when executing from bash script. Any integer must be given, value is
    meaningless. Just a work-around.

"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

import porepy as pp
import porepy.models.compositional_flow_with_equilibrium as cfle

from porepy.examples.cold_injection.config import MODEL_PARAMS
from porepy.examples.cold_injection.geometry import PointWells
from porepy.examples.cold_injection.model import (
    BuoyancyModel,
    ColdInjectionMixins,
    NoFluxRediscretization,
    set_schur_complement,
)


BUOYANCY_ON = False

max_iterations = 40 if BUOYANCY_ON else 30
iter_range = (21, 28) if BUOYANCY_ON else (15, 25)
newton_tol = 1e-7
newton_tol_increment = 5e-6

time_schedule = [i * 30 * pp.DAY for i in range(30 + 1)]
time_schedule = [0.0, pp.DAY]
dt_init = 3 * pp.HOUR
dt_max = 30 * pp.DAY

time_manager = pp.TimeManager(
    schedule=time_schedule,
    dt_init=dt_init,
    dt_min_max=(pp.HOUR, dt_max),
    iter_max=max_iterations,
    iter_optimal_range=iter_range,
    iter_relax_factors=(0.75, 2),
    recomp_factor=0.5,
    recomp_max=10,
    print_info=True,
    rtol=0.0,
)

model_params = MODEL_PARAMS.copy()
model_params["max_iterations"] = max_iterations
model_params["nl_convergence_tol"] = newton_tol
model_params["nl_convergence_tol_increment"] = newton_tol_increment
model_params["time_manager"] = time_manager
model_params["times_to_export"] = time_schedule

model_params["fractional_flow"] = BUOYANCY_ON
model_params["enable_buoyancy_effects"] = BUOYANCY_ON


if BUOYANCY_ON:

    class ModelClass(  # type:ignore[misc]
        BuoyancyModel,
        PointWells,
        ColdInjectionMixins,
        cfle.EnthalpyBasedCFFLETemplate,
    ):
        pass
else:

    class ModelClass(  # type:ignore
        NoFluxRediscretization,
        PointWells,
        ColdInjectionMixins,
        cfle.EnthalpyBasedCFLETemplate,
    ):
        pass


if __name__ == "__main__":
    timestamp = datetime.today().strftime("%d%B%Y_%I-%M-%S")
    model_params["folder_name"] = f"visualization/{timestamp}"

    model = ModelClass(model_params)  # type:ignore[abstract]

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy").setLevel(logging.DEBUG)
    t_0 = time.time()
    model.prepare_simulation()
    prep_sim_time = time.time() - t_0
    logging.getLogger("porepy").setLevel(logging.INFO)

    # Defining sub system for Schur complement reduction.
    set_schur_complement(model)  # type:ignore[arg-type]

    t_0 = time.time()
    SIMULATION_SUCCESS: bool = True
    pp.run_time_dependent_model(model, model_params)
    try:
        pp.run_time_dependent_model(model, model_params)
    except Exception as err:
        SIMULATION_SUCCESS = False
        print(f"Simulation failed:\n{str(err)}")
    sim_time = time.time() - t_0

    print(f"Simulation prepared after {prep_sim_time:.2f} (s).")
    print(f"Simulation finished after {sim_time / 60.0:.2f} (m).")
