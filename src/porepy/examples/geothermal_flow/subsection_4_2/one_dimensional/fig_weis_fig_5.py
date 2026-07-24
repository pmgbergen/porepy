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


def compute(N=N, level=None, parallel=True):
    level = m.TABLE_LEVEL if level is None else level
    out = C.sweep(TAG, list(CASES), m.FIG5, N, level, parallel=parallel)     # PPU / HU / HU-mwp
    weis_N = min(PPU_WEIS_N, N)                          # exactly 200 for the figure; scaled in --quick
    weis_tasks = [(("ppu_weis", case), "fig5weis", "ppu", case, m.FIG5, weis_N, level, True, True)
                  for case in CASES]                                          # grav_upstream, lag_upwind
    out.update(C.run_tasks("fig5-weis", weis_tasks, parallel=parallel))
    return out


def plot(out, stem="fig_weis_fig_5"):
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
        w = out[("ppu_weis", case)]                           # PPU-Weis 4th curve (Fig 5 only)
        ax_tp.plot(*ps.to_plot_units(w, "T"), color=PPU_WEIS_C, ls="-", lw=1.3, zorder=3)
        ax_p.plot(*ps.to_plot_units(w, "p"), color=PPU_WEIS_C, ls=C.P_LS, lw=1.1, zorder=3)
        ax_s.plot(*ps.to_plot_units(w, "s_liq"), color=PPU_WEIS_C, lw=1.3, zorder=3)
        C.iteration_legend(ax_s, res, loc="lower right", extra=[(PPU_WEIS_C, w["total_it"])])
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
    handles = C.scheme_handles() + [
        Line2D([0], [0], color=PPU_WEIS_C, lw=1.8, label=PPU_WEIS_LABEL),
        Line2D([0], [0], color="black", ls="-", label=r"$T$ (left)"),
        Line2D([0], [0], color="black", ls=C.P_LS, label=r"$p$ (right)"),
        C.ref_legend_handle()]
    fig.tight_layout()
    ps.bottom_legend(fig, handles, [h.get_label() for h in handles], ncol=4)
    ps.savefig(fig, stem, C.OUT_DIR)


def main():
    plot(compute())


if __name__ == "__main__":
    main()
