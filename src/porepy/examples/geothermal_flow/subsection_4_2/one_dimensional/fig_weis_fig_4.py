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


def compute(N=N, level=None, parallel=True):
    level = m.TABLE_LEVEL if level is None else level
    m.prebuild_table_caches(level)
    tasks = []
    for lvl in LEVELS:
        for orient in ORIENTS:
            bc = {**FIG4_BC[lvl], "tf_yr": FIG4_TF[(lvl, orient)]}
            for sk in ps.SCHEMES:
                tasks.append(((sk, lvl, orient), f"fig4_{lvl}", sk, orient, bc, N, level, False, False))
    return C.run_tasks("fig4", tasks, parallel=parallel)


def plot(out, stem="fig_weis_fig_4"):
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

    handles = C.scheme_handles() + [
        Line2D([0], [0], color="black", ls="-", label=r"$T$ (left)"),
        Line2D([0], [0], color="black", ls=C.P_LS, label=r"$p$ (right)"),
        C.ref_legend_handle()]
    fig.tight_layout()
    ps.bottom_legend(fig, handles, [h.get_label() for h in handles], ncol=3)
    ps.savefig(fig, stem, C.OUT_DIR)


def main():
    plot(compute())


if __name__ == "__main__":
    main()
