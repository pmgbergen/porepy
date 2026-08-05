"""Weis (2014) Figure 6 -- H2O-NaCl, horizontal column, 2000 yr. 2x2 panels: columns {pure water
z=0, salt + immobile halite z>0}, rows {temperature+pressure, liquid(+halite) saturation}. The
single weis brine engine for PPU / HU / HU-mwp, both columns sampling the graded OBL (z=0 slice for
pure water, z=salt_z for salt). The digitized Weis (2014) Fig-6 reference is overlaid from
benchmark_figures_data/fig_6_{pw,salt}_*.csv. If the salt column fails to converge it is drawn as a
placeholder, so the figure always renders.

    python reproduce_weis_fig_6.py                         # pure water + salt (z_init=0.42 -> S_h~0.1)
    python reproduce_weis_fig_6.py --salt-z-init 0.3 --N 200
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_weis_common as C  # noqa: E402
import weis_1d_solver as m   # noqa: E402
import plot_style as ps      # noqa: E402

N = 800                      # Fig 6 grid (dx = 10 m, paper); lighter than the N=800 Fig 4/5
SALT_Z = 0.42                # z_init giving S_h ~ 0.1 at the IC (from the z_init sweep)
TF = 2000.0
COLS = (("pw", "Water"), ("salt", "Water + salt"))

# PorePy overlay (mirrors reproduce_weis_fig_5): the converged HU profile from porepy_1d_solver.run_fig6_case,
# cached as _cache/fig6_porepy_{pw,salt}_hu_N800_lgraded.pkl. Both columns sample the same graded OBL
# (z=0 slice for pw, z=salt_z for salt). Black x markers.
POREPY_C = "#0b2e6f"          # deep navy overlay, drawn as a dash-dot line (clearer than the old x)
POREPY_LS = (0, (7, 3, 1.5, 3))   # dash-dot-dash pattern for the PorePy curve
POREPY_LABEL = r"HU-PorePy"
AUTORUN_POREPY = True         # generate a missing overlay pickle by running PorePy (heavy: 2000 yr)


def _load_porepy(column):
    """porepy_1d_solver Fig-6 pickle (y[m], T[K], p[MPa], s_liq, s_halite) for ``column`` ('pw'|'salt'),
    normalised to the SI plot convention (p -> Pa) that ``ps.to_plot_units`` consumes. If the pickle is
    missing and ``AUTORUN_POREPY``, run ``porepy_1d_solver.run_fig6_case`` to make it (lazy import, so a
    warm-cache re-plot never imports porepy). Returns the dict, or None if unavailable."""
    import pickle
    path = os.path.join(C.CACHE_DIR, f"fig6_porepy_{column}_hu_N800_l{m.TABLE_LEVEL}.pkl")
    if not os.path.exists(path) and AUTORUN_POREPY:
        try:
            import porepy_1d_solver as pp1d                 # lazy: imports porepy only on a cold cache
            print(f"[fig6] porepy overlay cache missing for {column} -- running "
                  f"porepy_1d_solver.run_fig6_case (heavy: 2000 yr PorePy solve) ...", flush=True)
            pp1d.run_fig6_case(column)                      # writes the same pickle path
        except Exception as exc:                            # never let an overlay break the figure
            print(f"[fig6] porepy overlay auto-run failed for {column}: {exc}", flush=True)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        d = dict(pickle.load(f))
    d["p"] = d["p"] * 1.0e6                                # MPa (PorePy native) -> Pa (SI, as weis)
    print(f"[fig6-porepy] ({column!r})   cached total_it={int(d.get('total_it', 0))}", flush=True)
    return d


def compute(N=N, salt_z=SALT_Z, parallel=True, skip=frozenset()):
    """PPU/HU/HU-mwp for the pure-water (z=0) and salt (z>0) columns at the Fig-6 BCs, both sampling the
    single graded OBL (z=0 slice for pw, z=salt_z for salt). The salt column is resilient: on divergence
    it is returned as ``None`` and drawn as a placeholder."""
    schemes = C.active_schemes(skip)                            # drop --skip-solver weis schemes
    pw = C.sweep("fig6_pw", ["horizontal"], {**m.FIG6, "z_init": 0.0, "tf_yr": TF}, N,
                 parallel=parallel, schemes=schemes)
    try:
        salt = C.sweep("fig6_salt", ["horizontal"], {**m.FIG6, "z_init": salt_z, "tf_yr": TF},
                       N, parallel=parallel, schemes=schemes)
    except Exception as exc:
        print(f"[fig6] salt column failed ({type(exc).__name__}: {exc}) -> placeholder", flush=True)
        salt = None

    def _byscheme(d):
        return None if d is None else {sk: d[(sk, "horizontal")] for sk in ps.SCHEMES
                                       if (sk, "horizontal") in d}
    return {"pw": _byscheme(pw), "salt": _byscheme(salt)}


def plot(out, stem="reproduce_weis_fig_6", skip=frozenset()):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ps.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(ps.TEXTWIDTH_IN, 5.4), sharex="col")
    tags = (("(a)", "(b)"), ("(c)", "(d)"))
    for j, (col, title) in enumerate(COLS):
        ax_tp, ax_s = axes[0, j], axes[1, j]
        ax_p = ax_tp.twinx(); ax_p.grid(False)
        res = out[col]
        ax_tp.set_title(title)
        ax_tp.set_xlim(0.0, 2.0)
        ps.panel_tag(ax_tp, tags[0][j], loc=(0.04, 0.09), va="bottom")
        ps.panel_tag(ax_s, tags[1][j])
        ax_h = None
        # Always draw the digitized Weis (2014) Fig-6 reference (benchmark_figures_data/fig_6_{col}_*.csv)
        # so the target is visible even when the solver is pending; the solver curves, iteration counts
        # and PorePy overlay draw only on a converged result.
        C.draw_tp(ax_tp, ax_p, res or {},
                  ref_T=C.ref_csv(f"fig_6_{col}_temperature_raw.csv"),
                  ref_p=C.ref_csv(f"fig_6_{col}_pressure_raw.csv"))
        ax_h = C.draw_s(ax_s, res or {}, ref_s=C.ref_csv(f"fig_6_{col}_saturation_liq_raw.csv"),
                        halite=(col == "salt"))
        if col == "salt":                                    # halite-saturation reference, twin axis
            ref_sh = C.ref_csv("fig_6_salt_saturation_halite_raw.csv")
            if ref_sh is not None:
                if ax_h is None:
                    ax_h = ax_s.twinx(); ax_h.grid(False); ax_h.set_ylim(-0.03, 1.03)
                ax_h.plot(ref_sh[0], ref_sh[1], color=C.WEIS_S, ls=(0, (1, 1)), lw=C.REF_LW,
                          marker="D", mfc="none", mec=C.WEIS_S, ms=C.REF_MS, mew=0.8, zorder=2,
                          markevery=max(1, len(ref_sh[0]) // C.REF_NMARK))
        if res is None:                                      # solver unavailable -> reference only
            for ax in (ax_tp, ax_s):
                ax.text(0.5, 0.93, "solver pending", transform=ax.transAxes, ha="center",
                        va="top", fontsize=9, color="0.55", style="italic")
        else:
            pp_res = None if C.is_skipped("hu-porepy", skip) else _load_porepy(col)  # PorePy HU overlay
            extra = [(POREPY_C, pp_res["total_it"])] if pp_res is not None else None
            C.iteration_legend(ax_s, res, loc="center left", extra=extra)  # empty vapor column, clear
            if pp_res is not None:
                mk = dict(color=POREPY_C, ls=POREPY_LS, lw=1.3, zorder=6)  # navy dash-dot line
                for ax, fld in ((ax_tp, "T"), (ax_p, "p"), (ax_s, "s_liq")):
                    xx, yy = ps.to_plot_units(pp_res, fld)
                    ax.plot(xx, yy, **mk)
                if ax_h is not None:                         # halite saturation on its own twin axis
                    ax_h.plot(pp_res["y"] / 1000.0, pp_res["s_halite"], **mk)
        # left column: T / s_liq axes; right column: p (+ halite) axes
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
            if ax_h is not None:
                ax_h.set_ylabel(r"Halite saturation $[-]$")
        ax_s.set_xlabel(ps.DIST_LABEL)

    handles = C.scheme_handles(only=C.active_schemes(skip))
    if not C.is_skipped("hu-porepy", skip):
        handles.append(Line2D([0], [0], color=POREPY_C, ls=POREPY_LS, lw=1.6, label=POREPY_LABEL))
    handles += [Line2D([0], [0], color="black", ls="-", label=r"$T$ (left)"),
                Line2D([0], [0], color="black", ls=C.P_LS, label=r"$p$ (right)"),
                Line2D([0], [0], color="black", ls=(0, (1, 1)), label=r"halite sat.")]
    fig.tight_layout()
    ps.bottom_legend(fig, handles, [h.get_label() for h in handles], ncol=4)
    ps.savefig(fig, stem, C.OUT_DIR)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weis (2014) Fig 6 (H2O-NaCl, PPU/HU/HU-mwp).")
    ap.add_argument("--salt-z-init", type=float, default=SALT_Z, dest="salt_z",
                    help="initial NaCl composition for the salt column (default 0.42 -> S_h~0.1)")
    ap.add_argument("--N", type=int, default=N, help=f"cells (default {N})")
    ap.add_argument("--no-porepy", action="store_true", dest="no_porepy",
                    help="skip the PorePy overlay (fast weis-vs-reference run, clean timing)")
    ap.add_argument("--halite-perm", choices=["A", "B"], default=m.HALITE_PERM_OPTION, dest="halite_perm",
                    help="halite->flow convention: A = p.349 rel-perm (k_rl+k_rv=1-S_h); "
                         "B = Eq 28 abs-perm (k=k0(1-S_h)^2). Weis Fig 6 used B (default %(default)s).")
    ap.add_argument("--skip-solver", default="", dest="skip",
                    help="comma-separated solvers to skip (avoid expensive sweeps): ppu, hu, hu-mwp, hu-porepy")
    args = ap.parse_args(argv)
    os.environ["WEIS_HALITE_PERM"] = args.halite_perm   # spawned parallel-sweep workers read this at import
    m.HALITE_PERM_OPTION = args.halite_perm             # this process + the non-parallel path
    if args.no_porepy:
        global AUTORUN_POREPY
        AUTORUN_POREPY = False        # no auto-run; with no graded overlay cache -> weis + reference only
    skip = C.parse_skip(args.skip)
    plot(compute(N=args.N, salt_z=args.salt_z, skip=skip), stem="reproduce_weis_fig_6", skip=skip)


if __name__ == "__main__":
    main()
