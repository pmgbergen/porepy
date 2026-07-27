"""Fig 6 RIGHT (salt, z > 0): a solver-free consistency test of the phz table against its ptz sibling.

Analogue of ``fig6_pw_ph_consistency.py`` for the salt column. Where solid halite is present the state
is over-determined, so (following the s_h != 0 idea) we bootstrap the composition and enthalpy from the
ptz (T-mode) table and then use them to CHECK the phz (h-mode) table:

  at each x with s_h > 0 (p = p_ref, T = T_ref fixed):
    1) COMPOSITION  z(x): invert the ptz table  S_h(z, T_ref, p_ref) = s_halite_ref  for z.
    2) ENTHALPY     h(x): read it straight from the ptz table,  h = H(z, T_ref, p_ref)   (T pins h in
                          single-liquid+halite; no phz inversion, so the phz table stays independent).
    3) CHECK phz: sample the phz table at (z, h, p_ref) and compare its Temperature / S_l / S_h back to
                  the reference. Small residuals => the two tables are mutually consistent AND the phz
                  table is resolution-adequate here; growing residuals (or wiggly z/h) => too coarse.

Tables are passed as the phz (xph) name; the matching ptz (xpt) sibling (xph -> xpt) must exist.

    python fig6_salt_ph_consistency.py                          # brine_graded vs opensowat_l3
    python fig6_salt_ph_consistency.py --tables my_xph.vtr

NOTE: z is only pinned by s_h in the (cold) halite region; the rest of the column is under-determined
and is skipped -- this is the well-conditioned INVERSE, a necessary consistency condition.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pyvista as pv
from porepy.examples.geothermal_flow.obl_sampler import VTKSampler

HERE = os.path.dirname(os.path.abspath(__file__))
VTK_DIR = os.path.join(HERE, os.pardir, os.pardir, "model_configuration",
                       "constitutive_description", "driesner_vtk_files")
REF_DIR = os.path.join(HERE, os.pardir, os.pardir, "benchmark_figures_data")

DEFAULT_TABLES = ["brine_graded_xph.vtr", "opensowat_xph_l_3.vtr"]   # purewater has no z-axis -> excluded


def _load_ref(name):
    a = np.loadtxt(os.path.join(REF_DIR, name), delimiter=",", skiprows=1)
    o = np.argsort(a[:, 0])
    return a[o, 0], a[o, 1]


def _invert(x, y, target):
    """First x where the (monotone-ish) curve y(x) crosses ``target``; np.nan if never."""
    k = np.where(np.diff(np.sign(y - target)) != 0)[0]
    if len(k) == 0:
        return np.nan
    j = k[0]
    return x[j] + (x[j + 1] - x[j]) * (target - y[j]) / (y[j + 1] - y[j] + 1e-300)


def _largest_run(mask):
    """Boolean mask keeping only the largest contiguous True run (drops isolated digitization spikes)."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.zeros_like(mask)
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    out = np.zeros_like(mask); out[max(runs, key=len)] = True
    return out


def reconstruct(xph_path, xpt_path, xg, P, Tc, SL, SH, n_grid=2000):
    """Return z, h [MJ/kg], and the phz residuals (dT[C], dSL, dSH) on the contiguous halite region."""
    Vph = VTKSampler(xph_path); Vpt = VTKSampler(xpt_path)
    zmax = float(np.asarray(pv.read(xpt_path).x)[-1])
    zgrid = np.linspace(0.0, min(zmax, 0.9), n_grid)
    n = len(xg)
    z = np.full(n, np.nan); h = np.full(n, np.nan)
    dT = np.full(n, np.nan); dSL = np.full(n, np.nan); dSH = np.full(n, np.nan)
    halite = _largest_run(SH > 0.02)                         # contiguous halite region only
    for i in np.where(halite)[0]:
        # (1) composition z from s_halite, ptz table
        Vpt.sample_at(np.column_stack([zgrid, np.full(n_grid, Tc[i]), np.full(n_grid, P[i])]))
        z[i] = _invert(zgrid, np.asarray(Vpt.sampled_could.point_data["S_h"]), SH[i])
        if not np.isfinite(z[i]):
            continue
        # (2) enthalpy straight from the ptz table: h = H(z, T, p)  (kJ/kg field -> MJ/kg)
        Vpt.sample_at(np.array([[z[i], Tc[i], P[i]]]))
        h[i] = float(np.asarray(Vpt.sampled_could.point_data["H"])[0]) * 1e-3
        # (3) CHECK the phz table at (z, h, p): does it reproduce the reference T / s_l / s_h?
        Vph.sample_at(np.array([[z[i], h[i], P[i]]]))
        pd = Vph.sampled_could.point_data
        dT[i] = (float(np.asarray(pd["Temperature"])[0]) - 273.15) - Tc[i]
        dSL[i] = float(np.asarray(pd["S_l"])[0]) - SL[i]
        dSH[i] = float(np.asarray(pd["S_h"])[0]) - SH[i]
    return z, h, dT, dSL, dSH, halite


def wiggle(xg, v, mask):
    """(sign_changes, tv_ratio) on the masked finite part -- 0 and ~1 for a smooth monotone curve."""
    m = mask & np.isfinite(v); vv = v[m]
    if vv.size < 3:
        return np.nan, np.nan
    dv = np.diff(vv); span = max(vv.max() - vv.min(), 1e-12)
    return int(np.sum(np.diff(np.sign(dv)) != 0)), float(np.sum(np.abs(dv)) / span)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fig 6 right (salt): phz-vs-ptz table consistency test.")
    ap.add_argument("--tables", nargs="+", default=DEFAULT_TABLES,
                    help="phz (xph) .vtr names in driesner_vtk_files/ (their xpt siblings must exist)")
    ap.add_argument("--npts", type=int, default=160)
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "fig6_salt_ph_consistency.png"))
    args = ap.parse_args(argv)

    xp, P = _load_ref("fig_6_salt_pressure_raw.csv")
    xt, Tc = _load_ref("fig_6_salt_temperature_raw.csv")
    xl, SL = _load_ref("fig_6_salt_saturation_liq_raw.csv")
    xh, SH = _load_ref("fig_6_salt_saturation_halite_raw.csv")
    lo = max(xp.min(), xt.min(), xl.min(), xh.min()); hi = min(xp.max(), xt.max(), xl.max(), xh.max())
    xg = np.linspace(lo, hi, args.npts)
    Pg = np.interp(xg, xp, P); Tg = np.interp(xg, xt, Tc)
    Sg = np.interp(xg, xl, SL); Hg = np.interp(xg, xh, SH)
    print(f"Fig-6-right reference: {args.npts} pts on x=[{lo:.3f},{hi:.3f}] km  s_hal[{Hg.min():.3f},{Hg.max():.3f}]\n"
          f"  method: z from s_h (ptz) -> h = ptz.H(z,T,p) -> CHECK phz at (z,h,p) vs reference T/s_l/s_h\n")

    res = {}; znodes = {}; hnodes = {}
    for t in args.tables:
        xph = t if os.path.isabs(t) else os.path.join(VTK_DIR, t)
        xpt = xph.replace("xph", "xpt")
        if not os.path.exists(xpt):
            print(f"  !! skip {os.path.basename(t)}: xpt sibling missing ({os.path.basename(xpt)})"); continue
        res[os.path.basename(t)] = reconstruct(xph, xpt, xg, Pg, Tg, Sg, Hg)
        znodes[os.path.basename(t)] = int(np.asarray(pv.read(xpt).x).size)
        hnodes[os.path.basename(t)] = int(np.asarray(pv.read(xph).y).size)
    if not res:
        print("no usable tables"); return

    hdr = (f"{'table':<24} {'z-nod':>5} {'h-nod':>5} | {'z:sgn':>5} {'h:sgn':>5} | "
           f"{'phz dT[C]':>9} {'phz dS_l':>9} {'phz dS_h':>9}   (max over halite region)")
    print(hdr); print("-" * len(hdr))
    for name, (z, h, dT, dSL, dSH, hal) in res.items():
        zsc, _ = wiggle(xg, z, hal); hsc, _ = wiggle(xg, h, hal)
        mx = lambda a: float(np.nanmax(np.abs(a[hal]))) if np.isfinite(a[hal]).any() else np.nan
        print(f"{name:<24} {znodes[name]:>5} {hnodes[name]:>5} | {zsc:>5d} {hsc:>5d} | "
              f"{mx(dT):>9.3f} {mx(dSL):>9.4f} {mx(dSH):>9.4f}")
    print("\n  phz dT/dS_l/dS_h = max| phz(z,h,p) - reference |. Small => brine_graded_xph.vtr is consistent"
          "\n  with its xpt sibling and resolution-adequate in the halite region; large/wiggly => too coarse.")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    for name, (z, h, dT, dSL, dSH, hal) in res.items():
        ax[0].plot(xg[hal], z[hal], "o-", ms=3, label=name)
        ax[1].plot(xg[hal], h[hal], "o-", ms=3, label=name)
        ax[2].plot(xg[hal], dT[hal], "-", label=f"{name} dT")
    ax[0].set(title="z(x) from s_halite (ptz)", xlabel="km", ylabel="z (NaCl overall)")
    ax[1].set(title="h(x) = ptz.H(z,T,p)  [MJ/kg]", xlabel="km", ylabel="h [MJ/kg]")
    ax[2].axhline(0, color="0.7", lw=0.8)
    ax[2].set(title="phz check: T(z,h,p) - T_ref  [degC]", xlabel="km", ylabel="dT [degC]")
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\n[fig] {args.out}")


if __name__ == "__main__":
    main()
