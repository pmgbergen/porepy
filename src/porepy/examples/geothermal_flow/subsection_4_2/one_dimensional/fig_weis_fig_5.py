"""Weis (2014) Figure 5 -- two-phase, pure water. 2x2 panels: columns {horizontal 200 yr,
vertical 1000 yr}, rows {temperature+pressure, liquid saturation}. The single weis brine engine at
z=0 (``weis_1d_solver.run_brine(**FIG5)``) for PPU / HU / HU-mwp, overlaid on the digitized paper
reference. The heavy N=800 runs are cached in ``_cache/`` (delete to recompute).

    python fig_weis_fig_5.py            # compute (or load cache) + render figures/fig_weis_fig_5
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_weis_common as C  # noqa: E402  (pins BLAS threads on import)
import weis_1d_solver as m   # noqa: E402
import plot_style as ps      # noqa: E402

N = 800
TAG = "fig5"
CASES = ("horizontal", "vertical")
YEARS = {"horizontal": "200", "vertical": "1000"}

# PPU-Weis (Fig 5 ONLY): PPU with the Weis (2014) discretization -- properties AND upwind directions
# lagged once per step (lag_upwind), and the interface gravity densities upwinded fully-upstream
# (Eq.25, grav_upstream) rather than the consistent face average. Purple, so it stands out. It runs on
# the Weis (2014) grid of N=200 cells, independent of the N used for the other schemes.
PPU_WEIS_C = "#984EA3"
PPU_WEIS_N = 200
PPU_WEIS_LABEL = rf"PPU-Weis ($N={PPU_WEIS_N}$)"

# PorePy approximation overlay: the converged HU profile from porepy_1d_solver, cached as
# _cache/porepy_{case}_hu_N800_l3.pkl. Drawn as black x markers over the weis-HU reference so the
# agreement (or drift) reads directly.
POREPY_C = "black"
POREPY_LABEL = r"HU-PorePy"
# If a porepy overlay pickle is missing, run porepy_1d_solver to generate it (so one `python
# fig_weis_fig_5.py` produces the porepy data AND the figure). HEAVY -- the PorePy solve is minutes
# per case (vertical is 1000 yr). Set False to only plot porepy caches that already exist.
AUTORUN_POREPY = True


def _load_porepy(case, scheme="hu", N=N, level=None):
    """porepy_1d_solver profile pickle (y[m], T[K], p[MPa], s_liq) for ``case``/``scheme``, normalised
    to the SI plot convention (p -> Pa) that ``ps.to_plot_units`` consumes. If the pickle is missing
    and ``AUTORUN_POREPY``, run ``porepy_1d_solver.run_case`` to make it (lazy import, so a warm-cache
    re-plot never imports porepy). Returns the dict, or None if unavailable."""
    import pickle
    level = m.TABLE_LEVEL if level is None else level
    path = os.path.join(C.CACHE_DIR, f"porepy_{case}_{scheme}_N{N}_l{level}.pkl")
    if not os.path.exists(path) and AUTORUN_POREPY:
        try:
            import porepy_1d_solver as pp1d                 # lazy: only imports porepy on a cold cache
            if (N, level) == (pp1d.N_CELLS, pp1d.TABLE_LEVEL):
                print(f"[fig5] porepy overlay cache missing for {case}/{scheme} -- running "
                      f"porepy_1d_solver.run_case (heavy: PorePy solve) ...", flush=True)
                pp1d.run_case(case, weighted_perm=(scheme == "hu_mwp"))   # writes the same pickle path
            else:
                print(f"[fig5] porepy overlay skipped for {case}/{scheme}: fig (N={N}, l{level}) "
                      f"!= porepy_1d_solver (N={pp1d.N_CELLS}, l{pp1d.TABLE_LEVEL})", flush=True)
        except Exception as exc:                            # never let an overlay break the figure
            print(f"[fig5] porepy overlay auto-run failed for {case}/{scheme}: {exc}", flush=True)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        d = dict(pickle.load(f))
    d["p"] = d["p"] * 1.0e6                                # MPa (PorePy native) -> Pa (SI, as weis)
    print(f"[fig5-porepy] ({scheme!r}, {case!r})   cached total_it={int(d['total_it'])}", flush=True)
    return d


def compute(N=N, level=None, parallel=True, skip=frozenset()):
    level = m.TABLE_LEVEL if level is None else level
    out = C.sweep(TAG, list(CASES), m.FIG5, N, level, parallel=parallel,     # PPU / HU / HU-mwp (minus skipped)
                  schemes=C.active_schemes(skip))
    if not C.is_skipped("ppu-weis", skip):
        weis_N = min(PPU_WEIS_N, N)                      # exactly 200 for the figure; scaled in --quick
        weis_tasks = [(("ppu_weis", case), "fig5weis", "ppu", case, m.FIG5, weis_N, level, True, True)
                      for case in CASES]                                      # grav_upstream, lag_upwind
        out.update(C.run_tasks("fig5-weis", weis_tasks, parallel=parallel))
    return out


def plot(out, stem="fig_weis_fig_5", skip=frozenset()):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ps.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(ps.TEXTWIDTH_IN, 5.6), sharex="col")
    tags = (("(a)", "(b)"), ("(c)", "(d)"))
    panels = []
    for j, case in enumerate(CASES):
        ax_tp, ax_s = axes[0, j], axes[1, j]
        ax_p = ax_tp.twinx(); ax_p.grid(False)
        res = {sk: out[(sk, case)] for sk in ps.SCHEMES if (sk, case) in out}
        panels.append(res)
        C.draw_tp(ax_tp, ax_p, res,
                  ref_T=C.ref_csv(f"fig_5_{case}_temperature_raw.csv"),
                  ref_p=C.ref_csv(f"fig_5_{case}_pressured_raw.csv"))
        C.draw_s(ax_s, res, ref_s=C.ref_csv(f"fig_5_{case}_saturation_liq_raw.csv"))
        extra_it = []
        if not C.is_skipped("ppu-weis", skip) and ("ppu_weis", case) in out:
            w = out[("ppu_weis", case)]                       # PPU-Weis 4th curve (Fig 5 only)
            ax_tp.plot(*ps.to_plot_units(w, "T"), color=PPU_WEIS_C, ls="-", lw=1.3, zorder=3)
            ax_p.plot(*ps.to_plot_units(w, "p"), color=PPU_WEIS_C, ls=C.P_LS, lw=1.1, zorder=3)
            ax_s.plot(*ps.to_plot_units(w, "s_liq"), color=PPU_WEIS_C, lw=1.3, zorder=3)
            extra_it.append((PPU_WEIS_C, w["total_it"]))
        pp_res = None if C.is_skipped("hu-porepy", skip) else _load_porepy(case)  # PorePy HU overlay
        if pp_res is not None:
            step = max(1, len(pp_res["y"]) // 24)             # ~24 markers across the 2 km column
            mk = dict(color=POREPY_C, marker="x", ms=4.2, ls="none", mew=0.9, zorder=6)
            for ax, fld in ((ax_tp, "T"), (ax_p, "p"), (ax_s, "s_liq")):
                x, y = ps.to_plot_units(pp_res, fld)
                ax.plot(x[::step], y[::step], **mk)
            extra_it.append((POREPY_C, pp_res["total_it"]))
        C.iteration_legend(ax_s, res, loc="lower right", extra=extra_it)
        ax_tp.set_title(fr"{case} orientation, ${YEARS[case]}$ years")
        ax_tp.set_xlim(0.0, 2.0)
        ps.panel_tag(ax_tp, tags[0][j], loc=(0.04, 0.09), va="bottom")   # T+p high at top-left -> tag low
        ps.panel_tag(ax_s, tags[1][j])                                    # s_liq low at top-left -> tag high
        # left column carries the T / s_liq axes; right column carries the pressure axis
        if j == 0:
            ax_tp.set_ylabel(ps.FIELD_LABEL["T"], color=C.WEIS_T)
            ax_tp.tick_params(axis="y", colors=C.WEIS_T)
            ax_s.set_ylabel(ps.FIELD_LABEL["s_liq"])
            ax_p.tick_params(axis="y", labelright=False)
        else:
            ax_p.set_ylabel(ps.FIELD_LABEL["p"], color=C.WEIS_P)
            ax_p.tick_params(axis="y", colors=C.WEIS_P)
            ax_tp.tick_params(axis="y", labelleft=False)
            ax_s.tick_params(axis="y", labelleft=False)
        ax_s.set_xlabel(ps.DIST_LABEL)

    # single bottom legend: scheme colours (names only) + the T/p and reference key. Per-case
    # iteration counts live in each panel's small legend (they differ by orientation).
    handles = C.scheme_handles(only=C.active_schemes(skip))
    if not C.is_skipped("ppu-weis", skip):
        handles.append(Line2D([0], [0], color=PPU_WEIS_C, lw=1.8, label=PPU_WEIS_LABEL))
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
    ap = argparse.ArgumentParser(description="Weis (2014) Fig 5 (two-phase pure water).")
    ap.add_argument("--skip-solver", default="", dest="skip",
                    help="comma-separated solvers to skip: ppu, hu, hu-mwp, hu-porepy, ppu-weis")
    args = ap.parse_args(argv)
    skip = C.parse_skip(args.skip)
    plot(compute(skip=skip), skip=skip)


if __name__ == "__main__":
    main()
