#!/usr/bin/env python
"""Plot the outlet CO2 breakthrough curves from outlet_co2_breakthrough.csv.

Reads one CSV per run folder and overlays the runs. By default a single panel -- the outlet
CO2 fraction (the breakthrough curve); --all adds the CO2 outflow rate and cumulative CO2 out.

Run:  python plot_breakthrough.py                      # single panel, auto-find visualization_co2_*
      python plot_breakthrough.py --all                # all three panels
      python plot_breakthrough.py FOLDER [FOLDER ...]   # specific run folders
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = "outlet_co2_breakthrough.csv"

LABELS = {"visualization_co2_inject_hu_md":              "TPFA",
          "visualization_co2_inject_hu_md_mpfa":         "MPFA",
          "visualization_co2_inject_hu_md_mpfa_nograv":  "MPFA - no gravity"}

PANELS = [("outlet_z_co2_avg",          "Outlet CO2 fraction",  "CO2 fraction [-]"),
          ("co2_mass_outflow_rate_kg_s", "CO2 outflow rate",     "rate [kg/s]"),
          ("co2_cumulative_out_kg",      "Cumulative CO2 out",   "mass [kg]")]


def find_runs(args):
    folders = args if args else sorted(glob.glob(os.path.join(HERE, "visualization_co2_*")))
    runs = []
    for f in folders:
        path = os.path.join(f, CSV)
        if os.path.exists(path):
            base = os.path.basename(f.rstrip("/"))
            label = LABELS.get(base, base.replace("visualization_co2_", ""))
            runs.append((label, path))
    return runs


def main():
    ap = argparse.ArgumentParser(description="Outlet CO2 breakthrough curves.")
    ap.add_argument("--all", action="store_true",
                    help="show all 3 panels (fraction, rate, cumulative); default is fraction only")
    ap.add_argument("folders", nargs="*", help="run folders (default: auto-find visualization_co2_*)")
    a = ap.parse_args()

    runs = find_runs(a.folders)
    if not runs:
        print(f"no {CSV} found (run the solver first, or pass run folders as arguments)")
        return

    panels = PANELS if a.all else PANELS[:1]       # default: outlet CO2 fraction only
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.4), squeeze=False)
    axes = axes[0]
    for label, path in runs:
        d = np.genfromtxt(path, delimiter=",", names=True)
        t = np.atleast_1d(d["time_days"])
        for ax, (col, _, _) in zip(axes, panels):
            ax.plot(t, np.atleast_1d(d[col]), marker=".", ms=4, label=label)
    for ax, (_, title, ylab) in zip(axes, panels):
        ax.set_title(title)
        ax.set_xlabel("time [days]")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    if len(runs) > 1:
        axes[0].legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"co2_breakthrough.{ext}"), dpi=150, bbox_inches="tight")
    print(f"wrote co2_breakthrough.{{png,pdf}} from {len(runs)} run(s): "
          + ", ".join(lbl for lbl, _ in runs))


if __name__ == "__main__":
    main()
