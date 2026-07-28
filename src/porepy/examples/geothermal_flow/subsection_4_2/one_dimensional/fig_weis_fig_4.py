"""Weis (2014) Figure 4 -- single-phase heating fronts. 3x2 panels: rows {high, moderate, low}
pressure (hP/mP/lP), columns {horizontal, vertical}, each showing temperature (red, left axis) and
pressure (blue, right axis) vs distance. The single weis brine engine at z=0 for PPU / HU / HU-mwp,
overlaid on the digitized paper reference. Each panel has its own BC preset and final time; the heavy
N=800 runs are cached in ``_cache/`` (some are 1500 yr).

    python fig_weis_fig_4.py            # compute (or load cache) + render figures/fig_weis_fig_4
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_weis_common as C  # noqa: E402  (pins BLAS threads on import)
import weis_1d_solver as m   # noqa: E402
import plot_style as ps      # noqa: E402

N = 800


def _K(celsius):
    return celsius + 273.15


# Fig-4 single-phase BC presets in SI (p [Pa], T [K], z=0); linear initial pressure, T_init = T_out.
# Extracted from model_configuration/bc_description/bc_market.py (BC_single_phase_{high,moderate,low}).
FIG4_BC = {
    "hp": dict(p_left=50e6, T_left=_K(350), z_left=0.0, p_right=25e6, T_right=_K(150), T_init=_K(150), z_init=0.0),
    "mp": dict(p_left=40e6, T_left=_K(450), z_left=0.0, p_right=20e6, T_right=_K(300), T_init=_K(300), z_init=0.0),
    "lp": dict(p_left=15e6, T_left=_K(500), z_left=0.0, p_right=1e6,  T_right=_K(350), T_init=_K(350), z_init=0.0),
}
# Final time [yr] per (pressure level, orientation) -- paper Fig-4 snapshots (single_phase solver).
FIG4_TF = {
    ("hp", "horizontal"): 250,  ("hp", "vertical"): 750,
    ("mp", "horizontal"): 120,  ("mp", "vertical"): 350,
    ("lp", "horizontal"): 1500, ("lp", "vertical"): 1500,
}
LEVELS = ("hp", "mp", "lp")
LEVEL_LABEL = {"hp": "high", "mp": "moderate", "lp": "low"}
ORIENTS = ("horizontal", "vertical")

# PorePy single-phase overlay (mirrors fig_weis_fig_5): the converged profile from
# single_phase_porepy_1d_solver, cached as _cache/single_phase_case_{hP,mP,lP}_{orient}_l3.pkl -- the
# SAME opensowat level-3 OBL as Fig 5. Drawn as black x markers over the weis-HU T/p curves.
POREPY_C = "black"
POREPY_LABEL = r"HU-PorePy"
AUTORUN_POREPY = True                          # generate a missing overlay pickle by running PorePy
_PP_CASE = {"hp": "case_hP", "mp": "case_mP", "lp": "case_lP"}   # Fig-4 level -> single-phase case name


def _load_porepy(lvl, orient, level=None):
    """single_phase_porepy_1d_solver pickle (x[m], T[K], p[MPa]) for Fig-4 panel (lvl, orient),
    normalised to the SI plot convention (y[m], p -> Pa) that ``ps.to_plot_units`` consumes. If the
    pickle is missing and ``AUTORUN_POREPY``, run the PorePy single-phase solver to make it (lazy
    import, so a warm-cache re-plot never imports porepy). Returns the dict, or None if unavailable."""
    import pickle
    level = m.TABLE_LEVEL if level is None else level
    case_name = _PP_CASE[lvl]
    path = os.path.join(C.CACHE_DIR, f"single_phase_{case_name}_{orient}_l{level}.pkl")
    if not os.path.exists(path) and AUTORUN_POREPY:
        try:
            import single_phase_porepy_1d_solver as sp1d    # lazy: imports porepy only on a cold cache
            if level == sp1d.TABLE_LEVEL:
                print(f"[fig4] porepy overlay cache missing for {case_name}/{orient} -- running "
                      f"single_phase_porepy_1d_solver.run_case (heavy: PorePy solve) ...", flush=True)
                sp1d.run_case(case_name, orient)            # writes the same pickle path
            else:
                print(f"[fig4] porepy overlay skipped for {case_name}/{orient}: fig level {level} "
                      f"!= single_phase solver level {sp1d.TABLE_LEVEL}", flush=True)
        except Exception as exc:                            # never let an overlay break the figure
            print(f"[fig4] porepy overlay auto-run failed for {case_name}/{orient}: {exc}", flush=True)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        d = dict(pickle.load(f))
    print(f"[fig4-porepy] ({case_name!r}, {orient!r})   cached", flush=True)
    return {"y": d["x"], "T": d["T"], "p": d["p"] * 1.0e6}  # x->y[m], p MPa->Pa (SI, ps.to_plot_units)


def compute(N=N, level=None, parallel=True, skip=frozenset()):
    level = m.TABLE_LEVEL if level is None else level
    m.prebuild_table_caches(level)
    tasks = []
    for lvl in LEVELS:
        for orient in ORIENTS:
            bc = {**FIG4_BC[lvl], "tf_yr": FIG4_TF[(lvl, orient)]}
            for sk in C.active_schemes(skip):               # drop --skip-solver weis schemes
                tasks.append(((sk, lvl, orient), f"fig4_{lvl}", sk, orient, bc, N, level, False, False))
    return C.run_tasks("fig4", tasks, parallel=parallel)


def plot(out, stem="fig_weis_fig_4", skip=frozenset()):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ps.apply_style()
    fig, axes = plt.subplots(3, 2, figsize=(ps.TEXTWIDTH_IN, 7.4), sharex=True)
    letters = (("(a)", "(b)"), ("(c)", "(d)"), ("(e)", "(f)"))
    panels = []
    for i, lvl in enumerate(LEVELS):
        for j, orient in enumerate(ORIENTS):
            ax_tp = axes[i, j]; ax_p = ax_tp.twinx(); ax_p.grid(False)
            res = {sk: out[(sk, lvl, orient)] for sk in ps.SCHEMES if (sk, lvl, orient) in out}
            panels.append(res)
            C.draw_tp(ax_tp, ax_p, res,
                      ref_T=C.ref_csv(f"fig_4_{lvl}_{orient}_temperature_raw.csv"),
                      ref_p=C.ref_csv(f"fig_4_{lvl}_{orient}_pressure_raw.csv"))
            pp_res = None if C.is_skipped("hu-porepy", skip) else _load_porepy(lvl, orient)  # PorePy overlay
            if pp_res is not None:
                step = max(1, len(pp_res["y"]) // 24)       # ~24 markers across the 2 km column
                mk = dict(color=POREPY_C, marker="x", ms=4.2, ls="none", mew=0.9, zorder=6)
                for ax, fld in ((ax_tp, "T"), (ax_p, "p")):
                    xx, yy = ps.to_plot_units(pp_res, fld)
                    ax.plot(xx[::step], yy[::step], **mk)
            ax_tp.set_xlim(0.0, 2.0)
            ps.panel_tag(ax_tp, letters[i][j], loc=(0.04, 0.09), va="bottom")
            if i == 0:
                ax_tp.set_title(orient)
            # per-panel iteration counts, captioned with this panel's pressure level + time
            C.iteration_legend(ax_tp, res, loc="upper right", fontsize=6.0,
                               title=fr"{LEVEL_LABEL[lvl]} $p$, ${FIG4_TF[(lvl, orient)]}$ yr")
            if j == 0:
                ax_tp.set_ylabel(ps.FIELD_LABEL["T"], color=C.WEIS_T)
                ax_tp.tick_params(axis="y", colors=C.WEIS_T)
                ax_p.tick_params(axis="y", labelright=False)
            else:
                ax_p.set_ylabel(ps.FIELD_LABEL["p"], color=C.WEIS_P)
                ax_p.tick_params(axis="y", colors=C.WEIS_P)
                ax_tp.tick_params(axis="y", labelleft=False)
    for ax in axes[-1, :]:
        ax.set_xlabel(ps.DIST_LABEL)

    handles = C.scheme_handles(only=C.active_schemes(skip))
    if not C.is_skipped("hu-porepy", skip):
        handles.append(Line2D([0], [0], color=POREPY_C, marker="x", ms=5, mew=1.2, ls="none", label=POREPY_LABEL))
    handles += [Line2D([0], [0], color="black", ls="-", label=r"$T$ (left)"),
                Line2D([0], [0], color="black", ls=C.P_LS, label=r"$p$ (right)"),
                C.ref_legend_handle()]
    fig.tight_layout()
    ps.bottom_legend(fig, handles, [h.get_label() for h in handles], ncol=4)
    ps.savefig(fig, stem, C.OUT_DIR)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Weis (2014) Fig 4 (single-phase heating).")
    ap.add_argument("--skip-solver", default="", dest="skip",
                    help="comma-separated solvers to skip: ppu, hu, hu-mwp, hu-porepy")
    args = ap.parse_args(argv)
    skip = C.parse_skip(args.skip)
    plot(compute(skip=skip), skip=skip)


if __name__ == "__main__":
    main()
