#!/usr/bin/env python
"""Weis (2014) Fig 6: horizontal 1-D H2O-NaCl column with an immobile solid-halite phase.

Full 2x2 layout: LEFT column = pure water (panels A: T+p, B: liquid saturation); RIGHT column =
salt-saturated + halite (panels C, D, with the halite curve). The engine is
``weis_1d_solver.run_brine`` (mass + salt + energy, horizontal so no buoyancy).

The right (salt) column is drawn only when ``--salt-z-init`` is supplied; otherwise it is a labelled
placeholder -- so the figure is resilient while the full-range brine table is still being built.

The P/T boundary conditions and the initial temperature are exposed as CLI flags (temperatures in
degC, pressures in MPa); defaults are the Fig-6 values, so a bare call reproduces the paper setup.

Usage:
    python fig_weis_fig6.py                       # A,B populated; C,D placeholder
    python fig_weis_fig6.py --tf 2000             # ... integrated to 2000 yr
    python fig_weis_fig6.py --salt-z-init 0.3     # also fills C,D (once the salt table is ready)
    python fig_weis_fig6.py --T-left 400 --p-left 20 --tf 200   # e.g. the Fig-5 hot-steam inlet
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import weis_1d_solver as W                                # noqa: E402
import plot_style as ps                                   # noqa: E402

OUT_DIR = os.path.join(HERE, "figures")
_RED, _BLUE, _GREEN, _BLACK = "#D55E00", "#0072B2", "#009E73", "0.15"
_DASH = (0, (4, 2))
_S_LIM = (-0.03, 1.03)                                   # saturation is always [0, 1]


def _auto_lims(bc_deg, results):
    """T and p axis limits derived from the boundary data (degC / MPa), widened to cover the actual
    profiles so nothing clips, with a small margin. Replaces the old hard-coded Fig-6 ranges, so any
    --T-left/--p-left (e.g. the 400 degC / 20 MPa Fig-5 inlet) is framed correctly."""
    Ts = [bc_deg["T_left"], bc_deg["T_right"], bc_deg["T_init"]]
    Ps = [bc_deg["p_left"], bc_deg["p_right"]]
    for res in results:
        if res is not None:
            Ts += [float((res["T"] - 273.15).min()), float((res["T"] - 273.15).max())]
            Ps += [float((res["p"] / 1.0e6).min()), float((res["p"] / 1.0e6).max())]

    def pad(lo, hi, frac, lo_floor=None):
        m = frac * max(hi - lo, 1e-9)
        lo2 = lo - m if lo_floor is None else max(lo_floor, lo - m)
        return (lo2, hi + m)

    return pad(min(Ts), max(Ts), 0.06), pad(min(Ps), max(Ps), 0.05, lo_floor=0.0)


def run_case(z_init, N, level, tf, bc=None, verbose=True):
    """Run one horizontal brine column; return the result dict, or None on failure (resilient).
    ``bc`` overrides the FIG6 boundary/initial conditions in SI (p_left/T_left/p_right/T_right/T_init).
    ``verbose`` streams run_brine's per-50-step progress (t / dt / Newton iters)."""
    try:
        return W.run_brine(N=N, level=level, z_init=z_init, tf_yr=tf, verbose=verbose, **(bc or {}))
    except Exception as exc:                              # table not ready / divergence -> placeholder
        print(f"  [skip] salt case z_init={z_init}: {type(exc).__name__}: {exc}", flush=True)
        return None


def _draw_column(ax_tp, ax_p, ax_s, res, tags, t_lim, p_lim):
    """Draw one Fig-6 column (T+p over saturations) into pre-made axes; placeholder if res is None.
    ``t_lim``/``p_lim`` are the shared, boundary-data-driven axis ranges from ``_auto_lims``."""
    # letters set low so they clear the near-inlet flat top curves (T=300, s_liq=1) at large t
    ps.panel_tag(ax_tp, tags[0], loc=(0.04, 0.82)); ps.panel_tag(ax_s, tags[1], loc=(0.04, 0.82))
    ax_tp.set_ylim(*t_lim); ax_p.set_ylim(*p_lim); ax_s.set_ylim(*_S_LIM)
    if res is None:
        for ax in (ax_tp, ax_s):
            ax.text(0.5, 0.5, "salt case\n(awaiting table)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="0.55", style="italic")
        return None
    x = res["y"] / 1.0e3
    ax_tp.plot(x, res["T"] - 273.15, color=_RED, lw=1.8)
    ax_p.plot(x, res["p"] / 1.0e6, color=_BLUE, lw=1.8, ls=_DASH)
    ax_s.plot(x, res["s_liq"], color=_GREEN, lw=1.8)
    if np.max(np.abs(res["s_halite"])) > 1.0e-6:         # halite twin only when present
        ax_h = ax_s.twinx(); ax_h.grid(False)
        ax_h.plot(x, res["s_halite"], color=_BLACK, lw=1.6, ls=_DASH)
        ax_h.set_ylim(*_S_LIM); ax_h.set_ylabel(r"Halite saturation $[-]$", color=_BLACK)
        ax_h.tick_params(axis="y", colors=_BLACK)
        return ax_h
    return None


def plot_full(res_pw, res_salt, stem, bc_deg):
    ps.apply_style()
    t_lim, p_lim = _auto_lims(bc_deg, [res_pw, res_salt])   # ranges from the boundary data + profiles
    fig, ((ax_tpL, ax_tpR), (ax_sL, ax_sR)) = plt.subplots(2, 2, figsize=(7.9, 6.3), sharex=True)
    ax_pL = ax_tpL.twinx(); ax_pR = ax_tpR.twinx()
    for a in (ax_pL, ax_pR):
        a.grid(False)
    _draw_column(ax_tpL, ax_pL, ax_sL, res_pw, ("(A)", "(B)"), t_lim, p_lim)
    ax_hR = _draw_column(ax_tpR, ax_pR, ax_sR, res_salt, ("(C)", "(D)"), t_lim, p_lim)

    # T and liquid saturation labelled on the far left; pressure on the data side (shared BC scales,
    # so the inner twin ticks are hidden). Keeps the figure readable whether or not C,D are filled.
    ax_tpL.set_ylabel(ps.FIELD_LABEL["T"], color=_RED); ax_tpL.tick_params(axis="y", colors=_RED)
    ax_sL.set_ylabel(ps.FIELD_LABEL["s_liq"], color=_GREEN); ax_sL.tick_params(axis="y", colors=_GREEN)
    ax_tpR.tick_params(axis="y", labelleft=False)
    ax_sR.tick_params(axis="y", labelleft=False)
    if res_salt is not None:
        ax_pR.set_ylabel(ps.FIELD_LABEL["p"], color=_BLUE); ax_pR.tick_params(axis="y", colors=_BLUE)
        ax_pL.tick_params(axis="y", labelright=False)
    else:
        ax_pL.set_ylabel(ps.FIELD_LABEL["p"], color=_BLUE); ax_pL.tick_params(axis="y", colors=_BLUE)
        ax_pR.set_yticks([])

    for a in (ax_sL, ax_sR):
        a.set_xlabel(ps.DIST_LABEL)
    ax_tpL.set_xlim(0.0, W.L_COLUMN / 1.0e3)
    handles = [plt.Line2D([], [], color=_RED, lw=1.8, label="temperature"),
               plt.Line2D([], [], color=_BLUE, lw=1.8, ls=_DASH, label="pressure"),
               plt.Line2D([], [], color=_GREEN, lw=1.8, label="liquid sat."),
               plt.Line2D([], [], color=_BLACK, lw=1.6, ls=_DASH, label="halite sat.")]
    ps.bottom_legend(fig, handles, [h.get_label() for h in handles], ncol=4)
    fig.tight_layout()
    ps.savefig(fig, stem, OUT_DIR)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weis (2014) Fig 6 (H2O-NaCl horizontal column, 2x2).")
    ap.add_argument("--salt-z-init", type=float, default=None, dest="salt_z_init",
                    help="initial NaCl composition for the RIGHT (salt) panels C,D; omit -> placeholder")
    ap.add_argument("--N", type=int, default=200, help="cells (default 200)")
    ap.add_argument("--level", type=int, default=W.TABLE_LEVEL, help="Driesner table level")
    ap.add_argument("--tf", type=float, default=W.FIG6["tf_yr"], help="final time [yr]")
    # Boundary / initial conditions in paper units (temperatures in degC, pressures in MPa); defaults
    # are the Fig-6 values from W.FIG6, so omitting all of them reproduces the paper setup.
    ap.add_argument("--T-left", type=float, default=W.FIG6["T_left"] - 273.15, dest="T_left",
                    help="left (hot vapor inlet) boundary temperature [degC] (default 300)")
    ap.add_argument("--T-right", type=float, default=W.FIG6["T_right"] - 273.15, dest="T_right",
                    help="right (cool outlet) boundary temperature [degC] (default 150)")
    ap.add_argument("--T-init", type=float, default=W.FIG6["T_init"] - 273.15, dest="T_init",
                    help="initial domain temperature [degC] (default 150)")
    ap.add_argument("--p-left", type=float, default=W.FIG6["p_left"] / 1.0e6, dest="p_left",
                    help="left boundary pressure [MPa] (default 4.0)")
    ap.add_argument("--p-right", type=float, default=W.FIG6["p_right"] / 1.0e6, dest="p_right",
                    help="right boundary pressure [MPa] (default 1.0)")
    args = ap.parse_args(argv)

    bc = dict(T_left=args.T_left + 273.15, T_right=args.T_right + 273.15, T_init=args.T_init + 273.15,
              p_left=args.p_left * 1.0e6, p_right=args.p_right * 1.0e6)
    bc_deg = dict(T_left=args.T_left, T_right=args.T_right, T_init=args.T_init,
                  p_left=args.p_left, p_right=args.p_right)      # display units for the axis ranges
    print(f"  Fig 6 conditions: left {args.T_left:.0f}degC/{args.p_left:g}MPa  ->  "
          f"right {args.T_right:.0f}degC/{args.p_right:g}MPa;  IC {args.T_init:.0f}degC", flush=True)

    print(f"  Fig 6: pure-water column (A,B), tf={args.tf:.0f} yr ...", flush=True)
    res_pw = run_case(0.0, args.N, args.level, args.tf, bc)
    res_salt = None
    if args.salt_z_init is not None:
        print(f"  Fig 6: salt column (C,D), z_init={args.salt_z_init}, tf={args.tf:.0f} yr ...",
              flush=True)
        res_salt = run_case(args.salt_z_init, args.N, args.level, args.tf, bc)

    plot_full(res_pw, res_salt, "fig_weis_fig6", bc_deg)
    tag = "A,B + C,D" if res_salt is not None else "A,B (C,D placeholder)"
    print(f"  Fig 6 done: {tag}", flush=True)


if __name__ == "__main__":
    main()
