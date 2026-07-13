"""PorePy 2D column (Weis 2014, Figure 5) for the subsection 4.1 overlay.

Runs the four cases needed by the figure overlays -- {horizontal, vertical} x {HU, HU-mw} -- at the
geometry's native N=800 and nominal dt = 0.25 yr, level-3 Driesner tables (matching weis_1d_solver),
and writes each converged 1D profile (distance, T, p, s_liq) extracted from the live model to

    subsection_4_1/_cache/porepy_{case}_{scheme}_N800_l3[_spline].pkl

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
    DriesnerBrineFlowModel,               # HU  (standard primary equations)
    DriesnerBrineFractionalFlowModel,     # HU-mw (fractional-flow primary equations)
)
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_two_phase_moderate_pressure as BC,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_two_phase_moderate_pressure as IC,
)
from porepy.examples.geothermal_flow.obl_sampler import NSplineSampler, VTKSampler

# --------------------------------------------------------------------------------------------- #
#  Fixed benchmark parameters (shared by all four cases)
# --------------------------------------------------------------------------------------------- #
DAY = 86400.0
TO_MEGA = 1.0e-6
DT = 0.25 * 365.0 * DAY                  # nominal time step: 0.25 yr (matches the 1D solver DT0)
TABLE_LEVEL = 3                           # Driesner opensowat .vtr level (0..4 available; 3 matches weis_1d_solver)
EXPORT_EVERY = 4                          # VTU snapshot cadence (in time steps)
USE_SPLINE = True                         # OBL sampler backend: True -> NSplineSampler (value and
#                                           gradient from one C2 tensor spline; consistent Jacobian);
#                                           False -> VTKSampler (probe of the stored value/grad_ fields).
_SAMPLER_SUFFIX = "_spline" if USE_SPLINE else ""   # keep spline vs VTK output caches distinct

FINAL_TIME_DAYS = {"horizontal": 73000.0, "vertical": 365000.0}   # 200 yr / 1000 yr
GEOMETRY = {"horizontal": ModelGeometryH, "vertical": ModelGeometryV}
DIST_AXIS = {"horizontal": 0, "vertical": 1}   # distance = cell_centers x (horiz) / y (vert)
N_CELLS = 800                                  # geometry's native cell count (ref_level 0.25)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_cache")
_TABLE_DIR = os.path.join(
    HERE, os.pardir, "model_configuration", "constitutive_description", "driesner_vtk_files")


def _pickle_path(geometry_case: str, scheme: str) -> str:
    """Per-case output pickle path in _cache/ (keyed by orientation, scheme, N, table level)."""
    return os.path.join(
        CACHE_DIR, f"porepy_{geometry_case}_{scheme}_N{N_CELLS}_l{TABLE_LEVEL}{_SAMPLER_SUFFIX}.pkl")


def _stats_path(geometry_case: str, scheme: str) -> str:
    """Companion human-readable solver-statistics text file next to the pickle."""
    return os.path.join(
        CACHE_DIR, f"porepy_{geometry_case}_{scheme}_N{N_CELLS}_l{TABLE_LEVEL}{_SAMPLER_SUFFIX}_stats.txt")


def _stats_pkl_path(geometry_case: str, scheme: str) -> str:
    """Companion pickle holding the model's :class:`NonlinearRunStats` dataclass."""
    return os.path.join(
        CACHE_DIR, f"porepy_{geometry_case}_{scheme}_N{N_CELLS}_l{TABLE_LEVEL}{_SAMPLER_SUFFIX}_stats.pkl")


def _save_stats(geometry_case: str, scheme: str, stats, tf: float) -> tuple[int, int]:
    """Persist ``stats`` (a :class:`NonlinearRunStats` from ``model.collect_run_stats()``): pickle
    the dataclass to ``_stats.pkl`` and write its ``as_text()`` -- prefixed with the run context
    (case, N, level, nominal-step count) -- to ``_stats.txt``. Returns ``(nominal, n_extra)``, the
    fixed-dt step count and how many extra accepted sub-steps the dt-cuts produced."""
    nominal = int(round(tf / DT))                        # steps if dt had stayed at the 0.25 yr cap
    n_extra = max(0, stats.n_accepted_steps - nominal)   # extra accepted steps from dt-cuts
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_stats_pkl_path(geometry_case, scheme), "wb") as f:
        pickle.dump(stats, f)
    header = (
        f"# PorePy solver statistics -- {geometry_case} / {scheme}  (N={N_CELLS}, level {TABLE_LEVEL})\n"
        f"# final_time_yr {tf / (365.0 * DAY):.0f}  nominal_dt_yr {DT / (365.0 * DAY):.4f}  "
        f"nominal_steps {nominal}  extra_substeps {n_extra}\n")
    with open(_stats_path(geometry_case, scheme), "w") as f:
        f.write(header)
        f.write(stats.as_text())
    return nominal, n_extra


def _attach_samplers(model) -> None:
    """Attach the level-``TABLE_LEVEL`` Driesner OBL samplers (phz + ptz) to ``model``. Backend is
    ``NSplineSampler`` (value and gradient from one C2 tensor spline -> consistent Jacobian) when
    ``USE_SPLINE`` else ``VTKSampler`` (pyvista probe of the stored value/``grad_`` fields)."""
    Sampler = NSplineSampler if USE_SPLINE else VTKSampler
    phz = Sampler(os.path.join(_TABLE_DIR, f"opensowat_xph_l_{TABLE_LEVEL}_grads.vtr"))
    phz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, h, p)
    model.obl_sampler = phz
    ptz = Sampler(os.path.join(_TABLE_DIR, f"opensowat_xpt_l_{TABLE_LEVEL}_grads.vtr"))
    ptz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, t, p)
    ptz.translation_factors = (0.0, -273.15, 0.0)            # T in degC -> K in the sampler
    model.obl_sampler_ptz = ptz


def run_case(geometry_case: str, weighted_perm: bool, cache: bool = True) -> dict:
    """Run one (orientation, scheme) case and pickle its converged 1D profile to ``_cache/``.

    ``weighted_perm=False`` -> HU (upwinded total mobility); ``True`` -> HU-mw (mobility-weighted).
    Both use ``buoyancy_upwinding='hybrid'``. Resumable: if ``cache`` and the target pickle already
    exists it is loaded and the (heavy) run is skipped; delete the pickle (or pass ``cache=False``)
    to recompute. Returns the saved dict.
    """
    scheme = "hu_mw" if weighted_perm else "hu"
    path = _pickle_path(geometry_case, scheme)
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            keep = pickle.load(f)
        print(f"\n=== PorePy {geometry_case} / {scheme}: cached "
              f"({os.path.relpath(path, HERE)}) -- skipping run ===", flush=True)
        return keep
    tf = FINAL_TIME_DAYS[geometry_case] * DAY

    # Adaptive dt CAPPED at DT (0.25 yr): it stays at DT on easy steps (dt_min_max max = DT means
    # it never grows above the cap) and only cuts -- by recomp_factor=0.5, down to DT/64, up to
    # recomp_max retries -- through hard steps such as the ~169 yr transition. The schedule end tf
    # is always hit exactly, so the extracted final-time profile is unaffected.
    time_manager = pp.TimeManager(
        schedule=[0.0, tf], dt_init=DT, constant_dt=False,
        dt_min_max=(DT / 64.0, DT), iter_max=20, iter_optimal_range=(3, 10),
        recomp_factor=0.5, recomp_max=10, print_info=True)
    solid = pp.SolidConstants(permeability=1e-15, porosity=0.1,
                              thermal_conductivity=2.0 * TO_MEGA, density=2700.0,
                              specific_heat_capacity=880.0 * TO_MEGA)
    times_to_export = list(np.arange(0.0, tf, DT * EXPORT_EVERY)) + [tf]
    params = {
        "ad_backend": "sparsa",
        "fractional_flow": weighted_perm,
        "enable_buoyancy_effects": True,
        "buoyancy_upwinding": "hybrid",
        "material_constants": {"solid": solid},
        "time_manager": time_manager,
        "times_to_export": times_to_export,
        "use_petsc": False,
        "step_control_method": "None",
    }

    ModelGeometry = GEOMETRY[geometry_case]
    # HU-mw uses the fractional-flow template, HU the standard one -- the discretisation distinction
    # is the base template, not a runtime parameter.
    FlowModel = DriesnerBrineFractionalFlowModel if weighted_perm else DriesnerBrineFlowModel

    class GeothermalWaterFlowModel(ModelGeometry, BC, IC, FlowModel):

        def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

        def fourier_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.fourier_keyword, list(subdomains))

        pass

    model = GeothermalWaterFlowModel(params)
    _attach_samplers(model)

    solver_params = {
        "nl_convergence_criteria": {
            "res_abs": pp.ResidualBasedAbsoluteCriterion(
                tol=1.0e-5, metric=pp.EquationBasedLebesgueMetric(model)),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.MaxIterationsCriterion(max_iterations=20),
        },
    }

    print(f"\n=== PorePy {geometry_case} / {scheme}  "
          f"(tf={tf / (365.0 * DAY):.0f} yr, dt=0.25 yr, level {TABLE_LEVEL}) ===", flush=True)
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

    # Solver statistics -> the base model's picklable NonlinearRunStats dataclass (collect_run_stats,
    # available to every model deriving from FlowModelBase). Persist it as a .pkl + human .txt.
    stats = model.collect_run_stats()
    nominal, n_extra = _save_stats(geometry_case, scheme, stats, tf)
    n_steps, total_it, n_cuts = (
        stats.n_accepted_steps, stats.total_newton_iterations, stats.n_time_step_cuts)
    print(f"  accepted steps: {n_steps} (nominal {nominal}"
          + (f", +{n_extra} sub-steps" if n_extra else "")
          + f"; {n_cuts} dt-cut{'' if n_cuts == 1 else 's'})"
          + f"   total Newton iterations (accepted): {total_it}"
          + f"   -> {os.path.relpath(_stats_path(geometry_case, scheme), HERE)}", flush=True)

    keep = {"y": y[o], "T": T[o], "p": p[o], "s_liq": s_liq[o],
            "case": geometry_case, "scheme": scheme, "n_cells": int(sd.num_cells),
            "n_steps": n_steps, "total_it": total_it, "n_time_step_cuts": n_cuts,
            "level": TABLE_LEVEL}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:                        # path fixed at the top (N_CELLS, level)
        pickle.dump(keep, f)
    print(f"  wrote {os.path.relpath(path, HERE)}  (N={sd.num_cells}, total_it={total_it})",
          flush=True)
    return keep


def main() -> None:
    for geometry_case in ["horizontal","vertical"]:
        for weighted_perm in [False]:
            run_case(geometry_case, weighted_perm)


if __name__ == "__main__":
    main()
