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


def compute(N=N, level=None, parallel=True):
    level = m.TABLE_LEVEL if level is None else level
    return C.sweep(TAG, list(CASES), m.FIG5, N, level, parallel=parallel)


def plot(out, stem="fig_weis_fig_5"):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ps.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(ps.TEXTWIDTH_IN, 5.4), sharex="col")
    tags = (("(a)", "(b)"), ("(c)", "(d)"))
    for j, case in enumerate(CASES):
        ax_tp, ax_s = axes[0, j], axes[1, j]
        ax_p = ax_tp.twinx(); ax_p.grid(False)
        res = {sk: out[(sk, case)] for sk in ps.SCHEMES if (sk, case) in out}
        h, lab = C.draw_tp(ax_tp, ax_p, res,
                           ref_T=C.ref_csv(f"fig_5_{case}_temperature_raw.csv"),
                           ref_p=C.ref_csv(f"fig_5_{case}_pressured_raw.csv"))
        C.draw_s(ax_s, res, ref_s=C.ref_csv(f"fig_5_{case}_saturation_liq_raw.csv"))
        ax_tp.set_title(fr"{case} orientation, ${YEARS[case]}$ years")
        ax_tp.set_xlim(0.0, 2.0)
        ps.panel_tag(ax_tp, tags[0][j]); ps.panel_tag(ax_s, tags[1][j])
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
        # per-column scheme + iteration-count key (counts differ between orientations)
        key = ax_s.legend(h, lab, loc="center right", fontsize=7, frameon=True, fancybox=True,
                          framealpha=1.0, edgecolor="0.6", borderpad=0.4, handlelength=1.4)
        key.get_frame().set_boxstyle("round,pad=0.25,rounding_size=0.3")

    # global key: solid = T / dashed = p (both left/right axes), and the reference band
    style = [Line2D([0], [0], color="black", ls="-", label=r"$T$ (left)"),
             Line2D([0], [0], color="black", ls=C.P_LS, label=r"$p$ (right)"),
             Line2D([0], [0], color=C.WEIS_T_LIGHT, lw=C.REF_LW, label=r"Weis et al.\ (2014)")]
    fig.tight_layout()
    ps.bottom_legend(fig, style, [s.get_label() for s in style], ncol=3)
    ps.savefig(fig, stem, C.OUT_DIR)


def main():
    plot(compute())


if __name__ == "__main__":
    main()
