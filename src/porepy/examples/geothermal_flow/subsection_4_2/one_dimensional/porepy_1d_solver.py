"""PorePy 2D column (Weis 2014, Figure 5) for the subsection 4.1 overlay.

Runs the four cases needed by the figure overlays -- {horizontal, vertical} x {HU, HU-mwp} -- at the
geometry's native N=800 and nominal dt = 0.25 yr, level-3 Driesner tables (matching weis_1d_solver),
and writes each converged 1D profile (distance, T, p, s_liq) extracted from the live model to

    subsection_4_1/_cache/porepy_{case}_{scheme}_N800_l3.pkl

with keys y[m], T[K], p[Pa], s_liq -- exactly what plot_style.to_plot_units consumes. PorePy still
writes its usual VTU/PVD output alongside (periodic snapshots).

The two scheme knobs:
  HU    -> buoyancy_upwinding="hybrid", mass_mobility_weighted_permeability=False
  HU-mwp -> buoyancy_upwinding="hybrid", mass_mobility_weighted_permeability=True

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
    DriesnerBrineFractionalFlowModel,     # HU-mwp (fractional-flow primary equations)
)
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_two_phase_moderate_pressure as BC,
    BC_H2O_NaCl_Figure_6 as BC_fig6,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_two_phase_moderate_pressure as IC,
    IC_H2O_NaCl_Figure_6_pure_water as IC_fig6_pw,
    IC_H2O_NaCl_Figure_6_salt as IC_fig6_salt,
)
from porepy.examples.geothermal_flow.model_configuration.geothermal_export import (  # noqa: E501
    DriesnerPhaseExport,
)
from porepy.examples.geothermal_flow.model_configuration.flow_model_base import (  # noqa: E501
    geothermal_nonlinear_solver,  # NewtonSolver that dispatches to model.solve_linear_system
)
from porepy.examples.geothermal_flow.obl_sampler import VTKSampler

# --------------------------------------------------------------------------------------------- #
#  Fixed benchmark parameters (shared by all four cases)
# --------------------------------------------------------------------------------------------- #
DAY = 86400.0
TO_MEGA = 1.0e-6
DT = 0.25 * 365.0 * DAY                  # nominal time step: 0.25 yr (matches the 1D solver DT0)
TABLE_LEVEL = 3                           # Driesner opensowat .vtr level (0..4 available; 3 matches weis_1d_solver)
EXPORT_EVERY = 4                          # VTU snapshot cadence (in time steps)
# OBL sampling: the unified VTKSampler tensor backend -- multilinear value + analytic gradient from
# the SAME interpolant (the sampled gradient is the derivative of the sampled value). This is the
# identical construction weis_1d_solver uses, so the two solvers are directly comparable.

FINAL_TIME_DAYS = {"horizontal": 73000.0, "vertical": 365000.0}   # 200 yr / 1000 yr
GEOMETRY = {"horizontal": ModelGeometryH, "vertical": ModelGeometryV}
DIST_AXIS = {"horizontal": 0, "vertical": 1}   # distance = cell_centers x (horiz) / y (vert)
N_CELLS = 800                                  # geometry's native cell count (ref_level 0.25)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_cache")
_TABLE_DIR = os.path.join(
    HERE, os.pardir, os.pardir, "model_configuration", "constitutive_description",
    "driesner_vtk_files")


def _pickle_path(geometry_case: str, scheme: str) -> str:
    """Per-case output pickle path in _cache/ (keyed by orientation, scheme, N, table level)."""
    return os.path.join(
        CACHE_DIR, f"porepy_{geometry_case}_{scheme}_N{N_CELLS}_l{TABLE_LEVEL}.pkl")


def _stats_path(geometry_case: str, scheme: str) -> str:
    """Companion human-readable solver-statistics text file next to the pickle."""
    return os.path.join(
        CACHE_DIR, f"porepy_{geometry_case}_{scheme}_N{N_CELLS}_l{TABLE_LEVEL}_stats.txt")


def _stats_pkl_path(geometry_case: str, scheme: str) -> str:
    """Companion pickle holding the model's :class:`NonlinearRunStats` dataclass."""
    return os.path.join(
        CACHE_DIR, f"porepy_{geometry_case}_{scheme}_N{N_CELLS}_l{TABLE_LEVEL}_stats.pkl")


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


def _attach_samplers(model, xph_name: str = None, xpt_name: str = None) -> None:
    """Attach the Driesner OBL samplers (phz + ptz) to ``model``. The ``VTKSampler`` tensor backend
    gives a multilinear value and the analytic gradient of that same interpolant (no stored ``grad_``
    fields) -- the construction weis_1d_solver also uses. Defaults to the level-``TABLE_LEVEL``
    opensowat tables; pass ``xph_name``/``xpt_name`` to sample a different .vtr (Fig 6 pure-water
    column uses the fine purewater tables)."""
    xph_name = xph_name or f"opensowat_xph_l_{TABLE_LEVEL}.vtr"
    xpt_name = xpt_name or f"opensowat_xpt_l_{TABLE_LEVEL}.vtr"
    Sampler = VTKSampler
    phz = Sampler(os.path.join(_TABLE_DIR, xph_name))
    phz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, h, p)
    model.obl_sampler = phz
    ptz = Sampler(os.path.join(_TABLE_DIR, xpt_name))
    ptz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, t, p)
    ptz.translation_factors = (0.0, -273.15, 0.0)            # T in degC -> K in the sampler
    model.obl_sampler_ptz = ptz


def run_case(geometry_case: str, weighted_perm: bool, cache: bool = True) -> dict:
    """Run one (orientation, scheme) case and pickle its converged 1D profile to ``_cache/``.

    ``weighted_perm=False`` -> HU (upwinded total mobility); ``True`` -> HU-mwp (mobility-weighted).
    Both use ``buoyancy_upwinding='hybrid'``. Resumable: if ``cache`` and the target pickle already
    exists it is loaded and the (heavy) run is skipped; delete the pickle (or pass ``cache=False``)
    to recompute. Returns the saved dict.
    """
    scheme = "hu_mwp" if weighted_perm else "hu"
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
        "ad_backend": "native",
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
    # HU-mwp uses the fractional-flow template, HU the standard one -- the discretisation distinction
    # is the base template, not a runtime parameter.
    FlowModel = DriesnerBrineFractionalFlowModel if weighted_perm else DriesnerBrineFlowModel

    class GeothermalWaterFlowModel(DriesnerPhaseExport, ModelGeometry, BC, IC, FlowModel):

        def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

        def fourier_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.fourier_keyword, list(subdomains))

        pass

    model = GeothermalWaterFlowModel(params)
    _attach_samplers(model)

    solver_params = {
        "nl_convergence_criteria": {
            "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                tol=1.0e-5, metric=pp.EquationBasedLebesgueMetric(model)),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=20),
        },
    }

    print(f"\n=== PorePy {geometry_case} / {scheme}  "
          f"(tf={tf / (365.0 * DAY):.0f} yr, dt=0.25 yr, level {TABLE_LEVEL}) ===", flush=True)
    runner = pp.ModelRunner(model, solver_params,
                            nonlinear_solver=geothermal_nonlinear_solver(solver_params))
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


# --------------------------------------------------------------------------------------------- #
#  Weis (2014) Figure 6 -- H2O-NaCl, horizontal, 2000 yr. Two columns: 'pw' pure water (z=0, sampled
#  from the fine purewater_x*.vtr) and 'salt' (z_init=0.42, immobile halite, opensowat tables).
# --------------------------------------------------------------------------------------------- #
FIG6_TF_YEARS = 2000.0
FIG6_TABLES = {                                   # column -> (xph .vtr, xpt .vtr) sampled by porepy
    "pw":   ("purewater_xph.vtr", "purewater_xpt.vtr"),                        # Fig-6 left (fine z=0)
    "salt": (f"opensowat_xph_l_{TABLE_LEVEL}.vtr", f"opensowat_xpt_l_{TABLE_LEVEL}.vtr"),
}


def _fig6_pickle_path(column: str) -> str:
    return os.path.join(CACHE_DIR, f"porepy_fig6_{column}_hu_N{N_CELLS}_l{TABLE_LEVEL}.pkl")


def run_fig6_case(column: str, cache: bool = True, tf_years: float = FIG6_TF_YEARS,
                  write_pickle: bool = True) -> dict:
    """Run the Weis (2014) Fig-6 HU case for ``column`` ('pw' pure water z=0, or 'salt' z_init=0.42
    with immobile halite), horizontal, tf=``tf_years`` (2000 yr), and pickle the converged 1D profile
    (y, T, p, s_liq, s_halite) to _cache/porepy_fig6_{column}_hu_N800_l3.pkl. The pure-water column
    samples the fine purewater_x*.vtr tables (Fig-6 left); the salt column samples opensowat. Only the
    final VTU snapshot is exported."""
    if column not in FIG6_TABLES:
        raise ValueError(f"column must be one of {list(FIG6_TABLES)}")
    path = _fig6_pickle_path(column)
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            keep = pickle.load(f)
        print(f"\n=== PorePy fig6 {column}: cached ({os.path.relpath(path, HERE)}) -- skipping ===",
              flush=True)
        return keep
    tf = tf_years * 365.0 * DAY
    xph_name, xpt_name = FIG6_TABLES[column]
    ICcls = IC_fig6_salt if column == "salt" else IC_fig6_pw

    time_manager = pp.TimeManager(
        schedule=[0.0, tf], dt_init=DT, constant_dt=False,
        dt_min_max=(DT / 64.0, DT), iter_max=20, iter_optimal_range=(3, 10),
        recomp_factor=0.5, recomp_max=10, print_info=True)
    solid = pp.SolidConstants(permeability=1e-15, porosity=0.1,
                              thermal_conductivity=2.0 * TO_MEGA, density=2700.0,
                              specific_heat_capacity=880.0 * TO_MEGA)
    params = {
        "ad_backend": "native", "fractional_flow": False, "enable_buoyancy_effects": True,
        "buoyancy_upwinding": "hybrid", "material_constants": {"solid": solid},
        "time_manager": time_manager, "times_to_export": [tf],       # final snapshot only (save disk)
        "use_petsc": False, "step_control_method": "None",
        # thermal-overshoot postprocessing destabilises the strongly halite-forming salt column;
        # disable it (the physical-bound clip stays on). No effect where s_h = 0 (pw column).
        "enable_thermal_overshoot_postprocessing": False,
        # bound the per-iteration gas-saturation step to damp the vapor phase-appearance oscillation
        # at the inlet (s_gas flip-flopping 0.2<->1.0) that otherwise stalls the salt column.
        "max_gas_saturation_step": 0.2,
    }

    class GeothermalWaterFlowModel(DriesnerPhaseExport, ModelGeometryH, BC_fig6, ICcls,
                                   DriesnerBrineFlowModel):
        def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

        def fourier_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.fourier_keyword, list(subdomains))

    model = GeothermalWaterFlowModel(params)
    _attach_samplers(model, xph_name=xph_name, xpt_name=xpt_name)
    solver_params = {
        "nl_convergence_criteria": {"res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
            tol=1.0e-5, metric=pp.EquationBasedLebesgueMetric(model))},
        "nl_divergence_criteria": {"max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=20)},
    }
    print(f"\n=== PorePy fig6 {column} (tf={tf_years:.0f} yr, tables={xph_name}) ===", flush=True)
    runner = pp.ModelRunner(model, solver_params,
                            nonlinear_solver=geothermal_nonlinear_solver(solver_params))
    print("  DoF:", model.equation_system.num_dofs())
    model.schur_complement_primary_equations = (
        pp.compositional_flow.get_primary_equations_cf(model))
    model.schur_complement_primary_variables = (
        pp.compositional_flow.get_primary_variables_cf(model))
    t0 = time.time()
    runner.run()
    print(f"  run wall: {(time.time() - t0) / 60.0:.1f} min", flush=True)

    sd = model.mdg.subdomains()[0]
    ev = model.equation_system.evaluate
    y = np.asarray(sd.cell_centers[0])                              # horizontal -> distance = x [m]
    p = np.asarray(ev(model.pressure([sd])))                        # model-native [MPa] (loader ->Pa)
    T = np.asarray(ev(model.temperature([sd])))                     # [K]
    liq = next(ph for ph in model.fluid.phases if ph.name == "liq")
    s_liq = np.asarray(ev(liq.saturation([sd])))                    # true liquid saturation (3-phase)
    hal = [ph for ph in model.fluid.phases if ph.name == "halite"]
    s_halite = np.asarray(ev(hal[0].saturation([sd]))) if hal else np.zeros_like(T)
    o = np.argsort(y)
    stats = model.collect_run_stats()
    keep = {"y": y[o], "T": T[o], "p": p[o], "s_liq": s_liq[o], "s_halite": s_halite[o],
            "column": column, "scheme": "hu", "n_cells": int(sd.num_cells),
            "n_steps": stats.n_accepted_steps, "total_it": stats.total_newton_iterations,
            "n_time_step_cuts": stats.n_time_step_cuts, "level": TABLE_LEVEL}
    if write_pickle:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(keep, f)
        print(f"  wrote {os.path.relpath(path, HERE)}  (total_it={keep['total_it']})", flush=True)
    return keep


def main() -> None:
    for geometry_case in ["horizontal","vertical"]:
        for weighted_perm in [False]:
            run_case(geometry_case, weighted_perm)


if __name__ == "__main__":
    main()
