"""PorePy 2D column (Weis 2014, Figure 5) for the subsection 4.1 overlay.

Runs the four cases needed by the figure overlays -- {horizontal, vertical} x {HU, HU-mw} -- at the
geometry's native N=800 and fixed dt = 0.125 yr, level-5 Driesner tables, and writes each converged
1D profile (distance, T, p, s_liq) extracted from the live model to

    subsection_4_1/_cache/porepy_{case}_{scheme}_N800_l5.pkl

with keys y[m], T[K], p[Pa], s_liq -- exactly what plot_style.to_plot_units consumes. PorePy still
writes its usual VTU/PVD output alongside (periodic snapshots).

The two scheme knobs:
  HU    -> buoyancy_upwinding="hybrid", mass_mobility_weighted_permeability=False
  HU-mw -> buoyancy_upwinding="hybrid", mass_mobility_weighted_permeability=True

Run: ``python porepy_1d_solver.py`` (heavy -- vertical is 1000 yr / 8000 steps). Requires the PorePy
environment.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

import porepy as pp

from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    SimpleGeometryHorizontal as ModelGeometryH,
)
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    SimpleGeometryVertical as ModelGeometryV,
)
from porepy.examples.geothermal_flow.model_configuration.DriesnerModelConfiguration import (  # noqa: E501
    DriesnerBrineFlowModel as FlowModel,
)
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_two_phase_moderate_pressure as BC,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_two_phase_moderate_pressure as IC,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

# --------------------------------------------------------------------------------------------- #
#  Fixed benchmark parameters (shared by all four cases)
# --------------------------------------------------------------------------------------------- #
DAY = 86400.0
TO_MEGA = 1.0e-6
DT = 0.125 * 365.0 * DAY                  # fixed time step: 0.125 yr (matches the 1D solver DT0)
TABLE_LEVEL = 5                           # Driesner opensowat .vtr level
EXPORT_EVERY = 8                          # VTU snapshot cadence (in time steps)

FINAL_TIME_DAYS = {"horizontal": 73000.0, "vertical": 365000.0}   # 200 yr / 1000 yr
GEOMETRY = {"horizontal": ModelGeometryH, "vertical": ModelGeometryV}
DIST_AXIS = {"horizontal": 0, "vertical": 1}   # distance = cell_centers x (horiz) / y (vert)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_cache")
_TABLE_DIR = os.path.join(
    HERE, os.pardir, "model_configuration", "constitutive_description", "driesner_vtk_files")


def _attach_samplers(model) -> None:
    """Attach the level-``TABLE_LEVEL`` Driesner VTK samplers (phz + ptz) to ``model``."""
    phz = VTKSampler(os.path.join(_TABLE_DIR, f"opensowat_xph_l_{TABLE_LEVEL}_grads.vtr"))
    phz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, h, p)
    model.vtk_sampler = phz
    ptz = VTKSampler(os.path.join(_TABLE_DIR, f"opensowat_xpt_l_{TABLE_LEVEL}_grads.vtr"))
    ptz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, t, p)
    ptz.translation_factors = (0.0, -273.15, 0.0)            # T in degC -> K in the sampler
    model.vtk_sampler_ptz = ptz


def run_case(geometry_case: str, weighted_perm: bool) -> dict:
    """Run one (orientation, scheme) case and pickle its converged 1D profile to ``_cache/``.

    ``weighted_perm=False`` -> HU (upwinded total mobility); ``True`` -> HU-mw (mobility-weighted).
    Both use ``buoyancy_upwinding='hybrid'``. Returns the saved dict.
    """
    scheme = "hu_mw" if weighted_perm else "hu"
    tf = FINAL_TIME_DAYS[geometry_case] * DAY

    time_manager = pp.TimeManager(schedule=[0.0, tf], dt_init=DT, constant_dt=True,
                                  iter_max=50, print_info=True)
    solid = pp.SolidConstants(permeability=1e-15, porosity=0.1,
                              thermal_conductivity=2.0 * TO_MEGA, density=2700.0,
                              specific_heat_capacity=880.0 * TO_MEGA)
    times_to_export = list(np.arange(0.0, tf, DT * EXPORT_EVERY)) + [tf]
    params = {
        "ad_backend": "native",
        "fractional_flow": False,
        "mass_mobility_weighted_permeability": weighted_perm,
        "enable_buoyancy_effects": True,
        "buoyancy_upwinding": "hybrid",
        "material_constants": {"solid": solid},
        "time_manager": time_manager,
        "times_to_export": times_to_export,
        "solver_statistics_file_name": f"solver_statistics_{geometry_case}_{scheme}",
        "use_petsc": False,
        "step_control_method": "None",
    }

    ModelGeometry = GEOMETRY[geometry_case]

    class GeothermalWaterFlowModel(ModelGeometry, BC, IC, FlowModel):
        pass

    model = GeothermalWaterFlowModel(params)
    _attach_samplers(model)

    solver_params = {
        "nl_convergence_criteria": {
            "res_abs": pp.ResidualBasedAbsoluteCriterion(
                tol=1.0e-4, metric=pp.EquationBasedLebesgueMetric(model)),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.MaxIterationsCriterion(max_iterations=30),
        },
    }

    print(f"\n=== PorePy {geometry_case} / {scheme}  "
          f"(tf={tf / (365.0 * DAY):.0f} yr, dt=0.125 yr, level {TABLE_LEVEL}) ===", flush=True)
    runner = pp.ModelRunner(model, solver_params)
    print("  DoF:", model.equation_system.num_dofs())
    model.schur_complement_primary_equations = (
        pp.compositional_flow.get_primary_equations_cf(model))
    model.schur_complement_primary_variables = (
        pp.compositional_flow.get_primary_variables_cf(model))
    model.exporter.write_vtu()                              # t=0 snapshot
    t0 = time.time()
    runner.run()
    print(f"  run wall: {(time.time() - t0) / 60.0:.1f} min", flush=True)

    # --- converged 1D profile, extracted from the live model (no VTU round-trip) ---
    sd = model.mdg.subdomains()[0]
    ev = model.equation_system.evaluate
    y = np.asarray(sd.cell_centers[DIST_AXIS[geometry_case]])       # distance [m], 0..2000
    p = np.asarray(ev(model.pressure([sd])))                        # [Pa]
    T = np.asarray(ev(model.temperature([sd])))                     # [K]
    gas = next(ph for ph in model.fluid.phases if ph.name == "gas")
    s_liq = 1.0 - np.asarray(ev(gas.saturation([sd])))             # [-]
    o = np.argsort(y)

    stats = getattr(model, "nonlinear_solver_statistics", None)
    total_it = int(sum(getattr(stats, "num_iterations_history", []) or [])) if stats else -1

    keep = {"y": y[o], "T": T[o], "p": p[o], "s_liq": s_liq[o],
            "case": geometry_case, "scheme": scheme, "n_cells": int(sd.num_cells),
            "total_it": total_it, "level": TABLE_LEVEL}
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(
        CACHE_DIR, f"porepy_{geometry_case}_{scheme}_N{sd.num_cells}_l{TABLE_LEVEL}.pkl")
    with open(path, "wb") as f:
        pickle.dump(keep, f)
    print(f"  wrote {os.path.relpath(path, HERE)}  (N={sd.num_cells}, total_it={total_it})",
          flush=True)
    return keep


def main() -> None:
    for geometry_case in ("horizontal", "vertical"):
        for weighted_perm in (False, True):
            run_case(geometry_case, weighted_perm)


if __name__ == "__main__":
    main()
