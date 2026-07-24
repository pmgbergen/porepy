"""Weis (2014) Figure 6 -- H2O-NaCl, horizontal column, 2000 yr. 2x2 panels: columns {pure water
z=0, salt + immobile halite z>0}, rows {temperature+pressure, liquid(+halite) saturation}. The
single weis brine engine for PPU / HU / HU-mwp. The digitized Fig-6 reference is not in the repo yet,
so the reference curve is a labelled placeholder. If the salt column fails to converge it is drawn as
a placeholder too, so the figure always renders.

    python fig_weis_fig_6.py                         # pure water + salt (z_init=0.42 -> S_h~0.1)
    python fig_weis_fig_6.py --salt-z-init 0.3 --N 200
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_weis_common as C  # noqa: E402
import weis_1d_solver as m   # noqa: E402
import plot_style as ps      # noqa: E402

N = 200                      # Fig 6 grid (dx = 10 m, paper); lighter than the N=800 Fig 4/5
SALT_Z = 0.42                # z_init giving S_h ~ 0.1 at the IC (from the z_init sweep)
TF = 2000.0
COLS = (("pw", "pure water"), ("salt", "salt + halite"))


def compute(N=N, level=None, salt_z=SALT_Z, parallel=True):
    """PPU/HU/HU-mwp for the pure-water (z=0) and salt (z>0) columns at the Fig-6 BCs. The salt column
    is resilient: on divergence it is returned as ``None`` and drawn as a placeholder."""
    level = m.TABLE_LEVEL if level is None else level
    pw = C.sweep("fig6_pw", ["horizontal"], {**m.FIG6, "z_init": 0.0, "tf_yr": TF}, N, level,
                 parallel=parallel)
    try:
        salt = C.sweep("fig6_salt", ["horizontal"], {**m.FIG6, "z_init": salt_z, "tf_yr": TF},
                       N, level, parallel=parallel)
    except Exception as exc:
        print(f"[fig6] salt column failed ({type(exc).__name__}: {exc}) -> placeholder", flush=True)
        salt = None

    def _byscheme(d):
        return None if d is None else {sk: d[(sk, "horizontal")] for sk in ps.SCHEMES
                                       if (sk, "horizontal") in d}
    return {"pw": _byscheme(pw), "salt": _byscheme(salt)}


def plot(out, stem="fig_weis_fig_6"):
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
        ps.panel_tag(ax_tp, tags[0][j]); ps.panel_tag(ax_s, tags[1][j])
        ax_h = None
        if res is None:                                  # salt solve not available -> placeholder
            for ax in (ax_tp, ax_s):
                ax.text(0.5, 0.5, "salt case\n(pending)", transform=ax.transAxes, ha="center",
                        va="center", fontsize=10, color="0.55", style="italic")
        else:
            # Fig-6 reference CSVs are not digitized yet -> ref_csv returns None -> no band drawn
            h, lab = C.draw_tp(ax_tp, ax_p, res,
                               ref_T=C.ref_csv(f"fig_6_{col}_temperature_raw.csv"),
                               ref_p=C.ref_csv(f"fig_6_{col}_pressure_raw.csv"))
            ax_h = C.draw_s(ax_s, res, ref_s=C.ref_csv(f"fig_6_{col}_saturation_liq_raw.csv"),
                            halite=(col == "salt"))
            ax_tp.text(0.5, 0.06, "reference: to add", transform=ax_tp.transAxes, ha="center",
                       fontsize=6.5, color="0.55", style="italic")
            key = ax_s.legend(h, lab, loc="center right", fontsize=7, frameon=True, fancybox=True,
                              framealpha=1.0, edgecolor="0.6", borderpad=0.4, handlelength=1.4)
            key.get_frame().set_boxstyle("round,pad=0.25,rounding_size=0.3")
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

    style = [Line2D([0], [0], color="black", ls="-", label=r"$T$ (left)"),
             Line2D([0], [0], color="black", ls=C.P_LS, label=r"$p$ (right)"),
             Line2D([0], [0], color="black", ls=(0, (1, 1)), label=r"halite sat.")]
    fig.tight_layout()
    ps.bottom_legend(fig, style, [s.get_label() for s in style], ncol=3)
    ps.savefig(fig, stem, C.OUT_DIR)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weis (2014) Fig 6 (H2O-NaCl, PPU/HU/HU-mwp).")
    ap.add_argument("--salt-z-init", type=float, default=SALT_Z, dest="salt_z",
                    help="initial NaCl composition for the salt column (default 0.42 -> S_h~0.1)")
    ap.add_argument("--N", type=int, default=N, help="cells (default 200)")
    args = ap.parse_args(argv)
    plot(compute(N=args.N, salt_z=args.salt_z))


if __name__ == "__main__":
    main()
