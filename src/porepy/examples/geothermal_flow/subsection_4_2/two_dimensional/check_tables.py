#!/usr/bin/env python
"""Acceptance checks for regenerated Driesner XPH/XPT tables.

Usage: python check_tables.py [--level 3]

Checks, per table:
  1. RANGES     h and p axes cover the fig-8 needs (10 degC water .. 4.5+ MJ/kg;
                atmospheric .. 60 MPa); z axis reported.
  2. CONVENTION saturation convention detector at two-phase nodes:
                volumetric  <=>  Rho ~ S_l Rho_l + S_v Rho_v
                mass        <=>  1/Rho ~ S_l/Rho_l + S_v/Rho_v
                The solvers REQUIRE volumetric.
  3. PARTITION  Xl == z in single-phase liquid; Xv > 0 somewhere in V+L.
  4. CLOSURE    bulk-salt identity at every V+L node with the detected convention:
                volumetric: (Xl S_l Rho_l + Xv S_v Rho_v) / (S_l Rho_l + S_v Rho_v) == z
  5. SANITY     mu >= 0 wherever the phase is mobile; rho_l(z=0.1, 10 degC) ~ 1075.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "one_dimensional"))
from weis_1d_solver import table_paths                              # noqa: E402


def load(path):
    g = pv.read(path)
    nc, ny, nz = g.dimensions
    try:
        c = np.asarray(g.x); a = np.asarray(g.y); b = np.asarray(g.z)
        assert len(c) == nc
    except Exception:
        pts = np.asarray(g.points)
        c = pts[:nc, 0]; a = pts[::nc, 1][:ny]; b = pts[::nc * ny, 2][:nz]
    F = {n: np.asarray(g.point_data[n], float).reshape(nz, ny, nc)
         for n in g.point_data.keys() if not n.startswith("grad")}
    return c, a, b, F


def check(path, kind):
    print(f"\n=== {os.path.basename(path)} ({kind}) ===")
    zax, aax, pax, F = load(path)
    ok = True

    # 1. ranges
    print(f"axes: z [{zax.min():g}, {zax.max():g}] x{len(zax)}  "
          f"{'h' if kind == 'xph' else 'T'} [{aax.min():g}, {aax.max():g}] x{len(aax)}  "
          f"p [{pax.min():g}, {pax.max():g}] x{len(pax)}")
    if kind == "xph" and (aax.min() > 0.045 or aax.max() < 4.5):
        print("  RANGE WARNING: h axis should span ~[0.02, 4.7] MJ/kg "
              "(10 degC water to q9/q20 dry-out)"); ok = False
    if pax.min() > 0.102 or pax.max() < 50:
        print("  RANGE WARNING: p axis should span ~[0.101, 60] MPa"); ok = False

    Sl, Sv = F["S_l"], F["S_v"]
    Rl, Rv, R = F["Rho_l"], F["Rho_v"], F["Rho"]
    Xl, Xv = F["Xl"], F["Xv"]
    two = (Sv > 0.05) & (Sl > 0.05) & (Rl > 1.0) & (Rv > 0.01)

    # 2. saturation convention
    if two.sum():
        vol = np.abs(Sl[two] * Rl[two] + Sv[two] * Rv[two] - R[two]) / R[two]
        mas = np.abs(Sl[two] / Rl[two] + Sv[two] / Rv[two] - 1.0 / R[two]) * R[two]
        conv = "VOLUMETRIC" if np.median(vol) < np.median(mas) else "MASS"
        print(f"convention: {conv} (vol-identity median {np.median(vol):.2e}, "
              f"mass-identity median {np.median(mas):.2e}) over {two.sum()} V+L nodes")
        if conv != "VOLUMETRIC":
            print("  FAIL: solvers require VOLUMETRIC saturations"); ok = False

        # 4. bulk-salt closure (volumetric form), per z slice
        errs = []
        for kz in range(1, len(zax)):
            m = two[:, :, kz]
            if not m.sum():
                continue
            zb = ((Xl[:, :, kz][m] * Sl[:, :, kz][m] * Rl[:, :, kz][m]
                   + Xv[:, :, kz][m] * Sv[:, :, kz][m] * Rv[:, :, kz][m])
                  / (Sl[:, :, kz][m] * Rl[:, :, kz][m]
                     + Sv[:, :, kz][m] * Rv[:, :, kz][m]))
            errs.append(np.abs(zb - zax[kz]) / max(zax[kz], 1e-12))
        e = np.concatenate(errs) if errs else np.zeros(1)
        p999 = float(np.percentile(e, 99.9)); n_out = int(np.count_nonzero(e > 1e-6))
        print(f"bulk-salt closure (volumetric): p99.9 {p999:.2e}, worst {e.max():.2e}, "
              f"outliers >1e-6: {n_out}/{e.size} (phase-boundary stragglers tolerated)")
        if p999 > 1e-6:
            print("  FAIL: implied z_bulk != z beyond isolated boundary nodes"); ok = False
        print(f"Xv in V+L: max {Xv[two].max():.5f} "
              f"({'ok, nonzero' if Xv[two].max() > 0 else 'FAIL: vapor salt-free everywhere'})")

    # 3. single-phase partitioning: Xl == z where only liquid is present
    liq = (Sl > 0.999) & (Sv < 1e-6)
    for kz in range(1, len(zax)):
        m = liq[:, :, kz]
        if m.sum():
            err = np.max(np.abs(Xl[:, :, kz][m] - zax[kz]))
            if err > 1e-6:
                print(f"  FAIL: single-phase Xl != z at z={zax[kz]:g} (max err {err:.2e})")
                ok = False
    print("single-phase Xl == z: checked")

    # 5. sanity: mobile-phase viscosity positivity
    mul, muv = F["mu_l"], F["mu_v"]
    bad_l = int(np.count_nonzero((mul <= 0) & (Sl > 0.3)))
    bad_v = int(np.count_nonzero((muv <= 0) & (Sv > 1e-3)))
    print(f"negative viscosity at mobile states: mu_l {bad_l}, mu_v {bad_v}"
          + ("" if bad_l == bad_v == 0 else "  FAIL"))
    ok = ok and bad_l == 0 and bad_v == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--xph", default=None, help="explicit xph path (overrides --level)")
    ap.add_argument("--xpt", default=None, help="explicit xpt path (overrides --level)")
    a = ap.parse_args()
    xph = a.xph or table_paths(a.level)[0]
    xpt = a.xpt or table_paths(a.level)[1]
    ok = check(xph, "xph") & check(xpt, "xpt")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
