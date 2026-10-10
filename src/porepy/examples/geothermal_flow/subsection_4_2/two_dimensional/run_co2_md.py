#!/usr/bin/env python
"""Run the H2O-CO2 injection cases on the mixed-dimensional (--md) grid, then make the figures.

Solver runs (sequential; each writes its own visualization_co2_* folder):
  1) TPFA            (default)
  2) MPFA            (--consistent)
  3) MPFA no-gravity (--consistent --no-gravity)
Post-processing figures (png + pdf):
  plot_breakthrough.py      outlet CO2 breakthrough curves
  plot_rho_h_contours.py    rho / enthalpy contours, no-gravity vs gravity

Run (from the porepy env):
  python run_co2_md.py                 # run the 3 cases, then plot
  python run_co2_md.py --plots-only    # skip the solver, just (re)make the figures
  python run_co2_md.py --no-plots      # run the 3 cases only
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable                                  # whatever interpreter runs this script
SOLVER = os.path.join(HERE, "porepy_2d_solver_co2.py")
BASE = ["--case", "inject", "--cell-size", "0.1", "--tf", "50.0",
        "--dt-init", "1.0", "--n-snap", "100", "--md"]

RUNS = [("TPFA", BASE),
        ("MPFA", BASE + ["--consistent"]),
        ("MPFA no-gravity", BASE + ["--consistent", "--no-gravity"])]

PLOTS = ["plot_breakthrough.py", "plot_rho_h_contours.py"]


def main():
    ap = argparse.ArgumentParser(description="Run the CO2 --md cases and make the figures.")
    ap.add_argument("--plots-only", action="store_true", help="skip the solver; only (re)make figures")
    ap.add_argument("--no-plots", action="store_true", help="run the solver only; skip the figures")
    a = ap.parse_args()

    if not a.plots_only:
        for i, (label, extra) in enumerate(RUNS, 1):
            print(f"=== [{i}/{len(RUNS)}] {label} : {' '.join(extra)} ===", flush=True)
            subprocess.run([PY, SOLVER] + extra, cwd=HERE, check=True)

    if not a.no_plots:
        for p in PLOTS:
            print(f"=== plotting: {p} ===", flush=True)
            subprocess.run([PY, os.path.join(HERE, p)], cwd=HERE, check=True)

    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
