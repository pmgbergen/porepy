from __future__ import annotations

import os
import pickle

import time
from typing import cast

import numpy as np

import porepy as pp

# geometry description horizontal case
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    SimpleGeometryHorizontal as ModelGeometryH,
)
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    SimpleGeometryVertical as ModelGeometryV,
)

# Figure 4 single with low pressure (lP) condition
# Figure 4 single with moderate pressure (mP) condition
# Figure 4 single with high pressure (hP) condition
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_single_phase_high_pressure as BC_hP,
)
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_single_phase_low_pressure as BC_lP,
)
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_single_phase_moderate_pressure as BC_mP,
)
from porepy.examples.geothermal_flow.model_configuration.DriesnerModelConfiguration import (  # noqa: E501
    DriesnerBrineFractionalFlowModel as FlowModel,   # fractional_flow=True pairs with the FF template
)
from porepy.examples.geothermal_flow.model_configuration.flow_model_base import (  # noqa: E501
    geothermal_nonlinear_solver,
)

from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_single_phase_high_pressure as IC_hP,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_single_phase_low_pressure as IC_lP,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_single_phase_moderate_pressure as IC_mP,
)
from porepy.examples.geothermal_flow.obl_sampler import NSplineSampler, VTKSampler

# --------------------------------------------------------------------------------------- #
#  Weis et al. (2014) Fig. 4: six single-phase heating fronts -- {hP, mP, lP} x
#  {horizontal, vertical} on the 2 km domain, each run to the paper's snapshot instant.
#  Profiles are pickled to _cache/ (resumable); VTUs go to per-case subfolders of
#  single_phase_visualization/ (the porepy_1d_solver outputs are never touched).
# --------------------------------------------------------------------------------------- #
DAY = 86400.0
TO_MEGA = 1.0e-6
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_cache")

# Constitutive approach shared by every subsection_4_2 solver: Driesner opensowat OBL
# tables sampled with the C2 tensor-spline backend (consistent value/Jacobian).
TABLE_LEVEL = 3                           # opensowat .vtr level (0..4 available)
USE_SPLINE = True                         # True -> NSplineSampler; False -> VTKSampler probe
_TABLE_DIR = os.path.join(
    HERE, os.pardir, os.pardir, "model_configuration", "constitutive_description",
    "driesner_vtk_files")

FINAL_TIME_DAYS = {                       # paper snapshot instants (years x 365 d)
    "horizontal": {"case_hP": 91250.0, "case_mP": 43800.0, "case_lP": 547500.0},
    "vertical": {"case_hP": 273750.0, "case_mP": 127750.0, "case_lP": 547500.0},
}
CASES = {"case_hP": (BC_hP, IC_hP), "case_mP": (BC_mP, IC_mP), "case_lP": (BC_lP, IC_lP)}
GEOMETRIES = {"horizontal": (ModelGeometryH, 0), "vertical": (ModelGeometryV, 1)}


def _attach_samplers(model) -> None:
    """Attach the level-``TABLE_LEVEL`` Driesner OBL samplers (phz + ptz), exactly as
    porepy_1d_solver / porepy_3d_solver do."""
    Sampler = NSplineSampler if USE_SPLINE else VTKSampler
    phz = Sampler(os.path.join(_TABLE_DIR, f"opensowat_xph_l_{TABLE_LEVEL}.vtr"))
    phz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, h, p)
    model.obl_sampler = phz
    ptz = Sampler(os.path.join(_TABLE_DIR, f"opensowat_xpt_l_{TABLE_LEVEL}.vtr"))
    ptz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, t, p)
    ptz.translation_factors = (0.0, -273.15, 0.0)            # T in degC -> K in the sampler
    model.obl_sampler_ptz = ptz


def _cache_path(case_name, geometry_case):
    return os.path.join(CACHE_DIR,
                        f"single_phase_{case_name}_{geometry_case}_l{TABLE_LEVEL}.pkl")


def run_case(case_name: str, geometry_case: str, cache: bool = True) -> dict:
    """Run one Fig.-4 case to its snapshot instant and pickle the final (x, T, p) profile.

    Resumable: with ``cache`` the existing pickle is returned without re-running."""
    path = _cache_path(case_name, geometry_case)
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    tf = FINAL_TIME_DAYS[geometry_case][case_name] * DAY
    BC, IC = CASES[case_name]
    ModelGeometry, axis = GEOMETRIES[geometry_case]
    time_manager = pp.TimeManager(schedule=[0.0, tf], dt_init=365.0 * DAY,
                                  constant_dt=True, iter_max=50, print_info=True)
    solid = pp.SolidConstants(permeability=1e-15, porosity=0.1,
                              thermal_conductivity=2.0 * TO_MEGA, density=2700.0,
                              specific_heat_capacity=880.0 * TO_MEGA)
    params = {
        "folder_name": os.path.join(HERE, "single_phase_visualization",
                                    f"{case_name}_{geometry_case}"),
        "fractional_flow": True,
        "enable_buoyancy_effects": True,
        "material_constants": {"solid": solid},
        "time_manager": time_manager,
        "prepare_simulation": False,
        "apply_schur_complement_reduction": False,
        "use_petsc": True,
        "petsc_preconditioner": "lu",
    }

    class GeothermalWaterFlowModel(ModelGeometry, BC, IC, FlowModel):
        def after_nonlinear_convergence(self) -> None:
            super().after_nonlinear_convergence()  # type:ignore[safe-super]
            print("Number of iterations: ",
                  self.nonlinear_solver_statistics.num_iterations)
            print("Time value (year): ", self.time_manager.time / (365.0 * DAY))
            print("Time index: ", self.time_manager.time_index)
            print("")

    model = GeothermalWaterFlowModel(params)
    _attach_samplers(model)
    solver_params = {
        "nl_convergence_criteria": {
            "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                tol=1.0e-4, metric=pp.EquationBasedLebesgueMetric(model)),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=100),
        },
    }
    runner = pp.ModelRunner(model, solver_params,
                            nonlinear_solver=geothermal_nonlinear_solver(solver_params))
    print(f"=== {case_name} / {geometry_case}: tf = {tf / (365.0 * DAY):.0f} yr, "
          f"DoF = {model.equation_system.num_dofs()} ===", flush=True)
    model.schur_complement_primary_equations = (
        pp.compositional_flow.get_primary_equations_cf(model))
    model.schur_complement_primary_variables = (
        pp.compositional_flow.get_primary_variables_cf(model))
    model.exporter.write_vtu()                               # t = 0 snapshot
    tb = time.time()
    runner.run()
    print(f"  wall: {(time.time() - tb) / 60.0:.1f} min", flush=True)

    sd = model.mdg.subdomains()[0]
    ev = model.equation_system.evaluate
    x = np.asarray(sd.cell_centers[axis])                    # distance [m], 0..2000
    p = np.asarray(ev(model.pressure([sd])))                 # [MPa] (model-native)
    T = np.asarray(ev(model.temperature([sd])))              # [K]
    o = np.argsort(x)
    keep = {"case": case_name, "geometry": geometry_case,
            "t_years": tf / (365.0 * DAY), "x": x[o], "T": T[o], "p": p[o],
            "level": TABLE_LEVEL}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(keep, f)
    return keep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Weis et al. (2014) Fig. 4 single-phase heating fronts: "
                    "{hP, mP, lP} x {horizontal, vertical} to the paper's snapshot "
                    "instants; final profiles pickled to _cache/.")
    ap.add_argument("--case", default="all", choices=[*CASES, "all"])
    ap.add_argument("--geometry", default="all", choices=[*GEOMETRIES, "all"])
    ap.add_argument("--no-cache", action="store_true",
                    help="re-run even when a cached profile exists")
    a = ap.parse_args()
    for c in (list(CASES) if a.case == "all" else [a.case]):
        for g in (list(GEOMETRIES) if a.geometry == "all" else [a.geometry]):
            run_case(c, g, cache=not a.no_cache)
