#!/usr/bin/env python
"""Weis et al. (2014, Geofluids 14:347-371) Fig. 4 reproduction.

Six single-phase heating fronts -- rows {hP, mP, lP}, columns {Horizontal, Vertical} --
with temperature (red, left axis) and fluid pressure (blue, right axis) vs distance.
PorePy profiles (solid) come from the ``single_phase_porepy_1d_solver.py`` cache
(run it first: ``python single_phase_porepy_1d_solver.py``); the digitized reference
curves (dashed dark gray) from ``benchmark_figures_data/fig_4_*_raw.csv``.

Usage: python fig_weis_single_phase.py   ->  figures/fig_4_single_phase.{png,pdf}
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, os.pardir, os.pardir, "benchmark_figures_data")
CACHE_DIR = os.path.join(HERE, "_cache")
FIG_DIR = os.path.join(HERE, "figures")

sys.path.insert(0, HERE)
import plot_style as ps                              # noqa: E402  (SAVE_PDF toggle for run_workflow)

ROWS = ("case_hP", "case_mP", "case_lP")
COLS = ("horizontal", "vertical")
T_LIM = {"case_hP": (150, 350), "case_mP": (290, 450), "case_lP": (290, 500)}
P_LIM = {"case_hP": (25, 50), "case_mP": (20, 40), "case_lP": (0, 15)}
YEARS = {("case_hP", "horizontal"): 250, ("case_hP", "vertical"): 750,
         ("case_mP", "horizontal"): 120, ("case_mP", "vertical"): 350,
         ("case_lP", "horizontal"): 1500, ("case_lP", "vertical"): 1500}
_ABC = "ABCDEF"

C_T, C_P, C_REF = "tab:red", "tab:blue", "0.25"


def _ref(case, geom, field):
    """Digitized reference curve (distance [km], value) for one panel/field."""
    tag = case.replace("case_", "").lower()                      # hp / mp / lp
    path = os.path.join(REF_DIR, f"fig_4_{tag}_{geom}_{field}_raw.csv")
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    o = np.argsort(d[:, 0])
    return d[o, 0], d[o, 1]


def _porepy(case, geom):
    """Cached PorePy profile for one panel (any table level)."""
    hits = sorted(glob.glob(os.path.join(
        CACHE_DIR, f"single_phase_{case}_{geom}_l*.pkl")))
    if not hits:
        return None                                  # not computed yet -> placeholder panel
    with open(hits[-1], "rb") as f:
        return pickle.load(f)


def main():
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 10.8), sharex=True)
    for i, case in enumerate(ROWS):
        for j, geom in enumerate(COLS):
            ax = axes[i][j]
            axp = ax.twinx()
            xr, Tr = _ref(case, geom, "temperature")
            ax.plot(xr, Tr, ls="--", color=C_REF, lw=2.0, zorder=2)
            xrp, Pr = _ref(case, geom, "pressure")
            axp.plot(xrp, Pr, ls="--", color=C_REF, lw=2.0, zorder=2)
            d = _porepy(case, geom)
            if d is not None:
                ax.plot(d["x"] / 1.0e3, d["T"] - 273.15, color=C_T, lw=1.6, zorder=3)
                axp.plot(d["x"] / 1.0e3, d["p"], color=C_P, lw=1.6, zorder=3)
            else:
                ax.text(0.5, 0.5, "PorePy solution\npending", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="0.45",
                        style="italic")
            ax.set_xlim(0.0, 2.0)
            ax.set_ylim(*T_LIM[case])
            axp.set_ylim(*P_LIM[case])
            ax.text(0.03, 0.86, f"({_ABC[i * 2 + j]})", transform=ax.transAxes,
                    fontsize=12, va="top", fontweight="bold")
            ax.text(0.97, 0.94, f"{YEARS[(case, geom)]} years", transform=ax.transAxes,
                    fontsize=11, va="top", ha="right")
            if i == 0:
                ax.set_title(geom.capitalize(), fontsize=13)
            if j == 0:
                ax.set_ylabel(r"Temperature ($^\circ$C)", color=C_T)
                axp.set_yticklabels([])
            else:
                axp.set_ylabel("Pressure (MPa)", color=C_P)
                ax.set_yticklabels([])
            ax.tick_params(axis="y", colors=C_T)
            axp.tick_params(axis="y", colors=C_P)
            if i == 2:
                ax.set_xlabel("Distance (km)")
    # figure-level line-style key: solid = PorePy, dashed = reference
    fig.legend(handles=[
        plt.Line2D([], [], color=C_T, lw=1.6, label="Temperature (PorePy)"),
        plt.Line2D([], [], color=C_P, lw=1.6, label="Pressure (PorePy)"),
        plt.Line2D([], [], color=C_REF, ls="--", lw=2.0, label="Weis et al. (2014)")],
        loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10,
        frameon=True, fancybox=True)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in (("png", "pdf") if ps.SAVE_PDF else ("png",)):
        fig.savefig(os.path.join(FIG_DIR, f"fig_4_single_phase.{ext}"),
                    dpi=300, bbox_inches="tight")
    print("wrote", os.path.join("figures", "fig_4_single_phase.png"),
          "(+ .pdf)" if ps.SAVE_PDF else "")


if __name__ == "__main__":
    main()
