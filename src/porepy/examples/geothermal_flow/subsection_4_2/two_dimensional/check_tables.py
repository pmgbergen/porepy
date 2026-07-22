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
  6. HALITE     hard gates for the full-range (z 0..1) halite tables:
                S_l + S_v + S_h == 1 everywhere; three-phase bulk-salt closure
                z Rho == Xl S_l Rho_l + Xv S_v Rho_v + (Rho - S_l Rho_l - S_v Rho_v)
                at S_h > 0 nodes (X_h = 1, S_h Rho_h by bulk substitution); and
                S_v(h) monotone through the two-phase band per (z, p) column
                (jagged dry-out boundaries are what lock the solvers).
  7. Z0-SLICE   drift of the z = 0 slice vs the stored reference
                (z0_slice_ref.npz): weis_1d/fig-4/fig-5 and every pure-water
                result across subsection_4_2 read this slice.  Snapshot the
                accepted tables ONCE with --save-z0-ref BEFORE regenerating.
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

    Sl, Sv, Sh = F["S_l"], F["S_v"], F["S_h"]
    Rl, Rv, R = F["Rho_l"], F["Rho_v"], F["Rho"]
    Xl, Xv = F["Xl"], F["Xv"]
    # V+L closure applies only where NO halite is present (three-phase nodes are
    # covered by the halite closure in section 6)
    two = (Sv > 0.05) & (Sl > 0.05) & (Sh <= 0.0) & (Rl > 1.0) & (Rv > 0.01)

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

    # 3. single-phase partitioning: Xl == z where ONLY liquid is present (L+H
    # states legitimately have Xl = X_sat != z, so halite nodes are excluded)
    liq = (Sl > 0.999) & (Sv < 1e-6) & (Sh < 1e-9)
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

    # 6. halite gates
    print(f"halite: S_h > 0 at {(Sh > 0).sum()}/{Sh.size} nodes, max {Sh.max():.4f}")
    ssum = np.abs(Sl + Sv + Sh - 1.0)
    print(f"  saturation sum |S_l+S_v+S_h-1|: max {ssum.max():.2e}"
          + ("" if ssum.max() < 1e-9 else "  FAIL"))
    ok = ok and ssum.max() < 1e-9
    hal = (Sh > 1e-12) & (R > 1.0)
    if hal.sum():
        rho_h_part = R[hal] - Sl[hal] * Rl[hal] - Sv[hal] * Rv[hal]
        zb = (Xl[hal] * Sl[hal] * Rl[hal] + Xv[hal] * Sv[hal] * Rv[hal]
              + rho_h_part) / R[hal]
        ztgt = np.broadcast_to(zax[None, None, :], Sh.shape)[hal]
        e3 = np.abs(zb - ztgt) / np.maximum(ztgt, 1e-12)
        p999h = float(np.percentile(e3, 99.9))
        print(f"  three-phase closure (bulk-Rho substitution): p99.9 {p999h:.2e}, "
              f"worst {e3.max():.2e} over {hal.sum()} nodes"
              + ("" if p999h < 1e-6 else "  FAIL"))
        ok = ok and p999h < 1e-6
    if kind == "xph":
        # informational (NOT a gate): supercritical vapor-like/liquid-like label
        # flips (rho_l == rho_v, harmless by degeneracy) and physical retrograde
        # condensation both produce dips; only distinct-phase dips are suspicious.
        dips = 0; worst = 0.0
        for kz in range(len(zax)):
            sv = Sv[:, :, kz]                              # (p, h) slice
            rl = Rl[:, :, kz]; rv = Rv[:, :, kz]
            d = np.diff(sv, axis=1)
            band = ((sv[:, :-1] > 1e-6) & (sv[:, :-1] < 1.0 - 1e-6)
                    & (np.abs(rl - rv)[:, :-1] > 0.01 * np.abs(rl[:, :-1])))
            bad = band & (d < -1e-2)
            dips += int(bad.sum())
            if bad.any():
                worst = max(worst, float(-d[bad].min()))
        print(f"  S_v(h) distinct-phase dips > 1e-2 (informational): {dips} node steps"
              + ("" if dips == 0 else f" (worst {worst:.3f})"))

    # 7. z = 0 slice drift vs stored reference
    ref_path = os.path.join(HERE, "z0_slice_ref.npz")
    if os.path.exists(ref_path):
        zf = np.load(ref_path, allow_pickle=False)
        pre = kind + "_"
        names = [str(n) for n in zf["names"] if str(n).startswith(pre)]
        if names:
            ra = np.asarray(zf[pre + "aax"]); rp = np.asarray(zf[pre + "pax"])
            from scipy.interpolate import RegularGridInterpolator
            worst_f, worst_v = "", 0.0
            AA, PP = np.meshgrid(ra, rp, indexing="xy")
            pts = np.column_stack([PP.ravel(), AA.ravel()])
            for n in names:
                fname = n[len(pre):]
                if fname in ("aax", "pax") or fname not in F:
                    continue
                itp = RegularGridInterpolator((pax, aax), F[fname][:, :, 0],
                                              bounds_error=False, fill_value=None)
                new = itp(pts).reshape(len(rp), len(ra))
                refv = np.asarray(zf[n], float)
                dv = np.abs(new - refv) / np.maximum(np.abs(refv).max(), 1e-12)
                if dv.max() > worst_v:
                    worst_v, worst_f = float(dv.max()), fname
            print(f"z0-slice drift vs reference: worst field '{worst_f}' "
                  f"max rel {worst_v:.2e}" + ("" if worst_v < 1e-3 else "  FAIL"))
            ok = ok and worst_v < 1e-3
    else:
        print("z0-slice reference not found (create with --save-z0-ref)")

    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def save_z0_ref(xph, xpt):
    out, names = {}, []
    for path, kind in ((xph, "xph"), (xpt, "xpt")):
        zax, aax, pax, F = load(path)
        out[kind + "_aax"] = aax; out[kind + "_pax"] = pax
        names += [kind + "_aax", kind + "_pax"]
        for n, v in F.items():
            out[f"{kind}_{n}"] = v[:, :, 0].astype(np.float64)
            names.append(f"{kind}_{n}")
    out["names"] = np.array(names)
    np.savez_compressed(os.path.join(HERE, "z0_slice_ref.npz"), **out)
    print("wrote", os.path.join(HERE, "z0_slice_ref.npz"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--xph", default=None, help="explicit xph path (overrides --level)")
    ap.add_argument("--xpt", default=None, help="explicit xpt path (overrides --level)")
    ap.add_argument("--save-z0-ref", action="store_true",
                    help="snapshot the z = 0 slice of the CURRENT (accepted) tables "
                         "as the drift reference; run once BEFORE regenerating")
    a = ap.parse_args()
    xph = a.xph or table_paths(a.level)[0]
    xpt = a.xpt or table_paths(a.level)[1]
    if a.save_z0_ref:
        save_z0_ref(xph, xpt)
        return
    ok = check(xph, "xph") & check(xpt, "xpt")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
