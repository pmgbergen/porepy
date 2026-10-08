#!/usr/bin/env python
"""Workflow: regenerate the H2O-CO2 compositional OBL data AND the figures.

Data (two VTR tables + offset sidecar):
  h2o_co2_xpt.vtr     p-T-z table
  h2o_co2_xph.vtr     p-h-z table
  h2o_co2_offset.txt  enthalpy offset [J/kg]

Figures (images + error plots):
  co2_phase_diagram.{png,pdf}         p-T / p-h phase diagram + densities
  co2_ph_slices.{png,pdf}             compositional p-h slices z = 0 / 0.1 / 0.25 / 0.5
  co2_phz_diagrams.{png,pdf}          phase regions + OBL saturation error vs the true flash (z=0.1/0.3)
  co2_err_{saturation,density,enthalpy,temperature}.{png,pdf}   grouped L2 errors

Case window: p in [4, 10] MPa, h in [0.075, 0.22] MJ/kg (pt T-axis [1, 60] C). The enthalpy offset is
the canonical min(H)=0 shift over the FULL physical range. Table resolution is 1.5x the previous
per-axis node counts.

Run (from inside this folder):
  python generate_obl_data.py                 # tables + figures
  python generate_obl_data.py --no-figures    # tables only
  python generate_obl_data.py --figures-only  # figures from the existing tables
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import pyvista as pv

import build_co2_compositional_table as B

HERE = os.path.dirname(os.path.abspath(__file__))

# case axes -- 1.5x the previous per-axis resolution (was z=43, T=161, p=141, h=161)
P_RANGE, NP = (4.0, 10.0), 212          # MPa      (141 -> 212)
T_RANGE, NT = (1.0, 60.0), 242          # degC     (161 -> 242)
H_RANGE, NH = (0.075, 0.22), 242        # MJ/kg    (161 -> 242)
Z_AX = np.unique(np.concatenate([np.linspace(0.0, 1.0, 62), [0.001, 0.999]]))   # ~64 nodes (43 -> ~64)


def build_tables() -> None:
    print("[tables 1/3] canonical enthalpy offset (full physical range) ...")
    off = B.canonical_offset()
    print(f"             offset = {off / 1e3:.3f} kJ/kg")

    print("[tables 2/3] building the two case tables (1.5x resolution) ...")
    T_AX = np.linspace(T_RANGE[0], T_RANGE[1], NT)
    P_AX = np.linspace(P_RANGE[0], P_RANGE[1], NP) * 1e6
    B.build_and_write(Z_AX, T_AX, P_AX, tag="", off=off,
                      h_min=H_RANGE[0], h_max=H_RANGE[1], nh=NH)

    print("[tables 3/3] sanity check ...")
    for name in ("h2o_co2_xpt.vtr", "h2o_co2_xph.vtr"):
        m = pv.read(os.path.join(HERE, name))
        bad = [k for k in m.point_data.keys() if not np.isfinite(m.point_data[k]).all()]
        assert not bad, f"{name}: non-finite fields {bad}"
        print(f"             {name}  dims {m.dimensions}  fields {len(m.point_data.keys())}  OK")


def make_figures() -> None:
    # imported here (after the tables exist) so the table-reading figures see the fresh data
    import co2_phase_diagram
    import co2_ph_slices
    import co2_phz_diagrams
    import co2_table_errors
    for label, mod in (("phase diagram", co2_phase_diagram),
                       ("p-h slices", co2_ph_slices),
                       ("phz diagrams", co2_phz_diagrams),
                       ("grouped L2 errors", co2_table_errors)):
        print(f"[figures] {label} ...")
        mod.main()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the H2O-CO2 OBL tables and figures.")
    ap.add_argument("--no-figures", action="store_true", help="build the tables only")
    ap.add_argument("--figures-only", action="store_true", help="regenerate figures from existing tables")
    a = ap.parse_args()

    if not a.figures_only:
        build_tables()
    if not a.no_figures:
        make_figures()
    print("done.")


if __name__ == "__main__":
    main()
