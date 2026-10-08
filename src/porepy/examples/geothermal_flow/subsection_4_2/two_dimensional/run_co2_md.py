#!/usr/bin/env python
"""Run the H2O-CO2 injection case on the mixed-dimensional (--md) grid, twice:
  1) TPFA   (default)
  2) MPFA   (--consistent)
Same geometry/schedule; outputs land in the two case-tagged visualization folders.

Run (from the porepy env):  python run_co2_md.py
"""
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


def main():
    for i, (label, extra) in enumerate(RUNS, 1):
        cmd = [PY, SOLVER] + extra
        print(f"=== [{i}/{len(RUNS)}] {label} : {' '.join(extra)} ===", flush=True)
        subprocess.run(cmd, cwd=HERE, check=True)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
