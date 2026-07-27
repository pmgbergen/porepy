"""Fig 6 LEFT (pure water, z = 0): a solver-free p-h table-resolution test.

Idea: reconstruct the enthalpy profile h(x) along the Fig-6-left column directly from the digitized
reference (p, T, s_liq), by INVERTING a candidate OBL phz table -- no PDE solve. For z = 0 the pressure
is taken from the reference curve, so p = p_ref(x) is fixed and we sweep x to recover h:

  * two-phase   (0 < s_liq < 1): T is pinned to the boiling curve T_sat(p) and cannot identify h, so
                                 invert  S_l(z=0, h, p_ref) = s_liq_ref  for h.
  * single-phase (s_liq ~ 0/1):  s_liq is flat (0 or 1) and cannot identify h, so invert
                                 Temperature(z=0, h, p_ref) = T_ref  for h.

A table that is well resolved in the p-h plane yields a smooth, monotone h(x). A table that is too
coarse makes the reconstructed h(x) WIGGLE (its piecewise interpolant kinks / overshoots leak into the
inverse). Run several tables and read off the coarsest one that stays smooth.

    python fig6_pw_ph_consistency.py                         # purewater vs graded vs opensowat_l3
    python fig6_pw_ph_consistency.py --tables my_coarse.vtr  # any candidate .vtr (phz: z, h[MJ/kg], p[MPa])
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

DEFAULT_TABLES = ["purewater_xph.vtr", "brine_graded_xph.vtr", "opensowat_xph_l_3.vtr"]


def _load_ref(name):
    a = np.loadtxt(os.path.join(REF_DIR, name), delimiter=",", skiprows=1)
    o = np.argsort(a[:, 0])
    return a[o, 0], a[o, 1]


def _invert(h, y, target):
    """First h where the (monotone-ish) curve y(h) crosses ``target``; np.nan if never. Works for
    increasing or decreasing y (uses the first sign change of y - target)."""
    s = np.sign(y - target)
    k = np.where(np.diff(s) != 0)[0]
    if len(k) == 0:
        return np.nan
    j = k[0]
    y0, y1 = y[j], y[j + 1]
    return h[j] + (h[j + 1] - h[j]) * (target - y0) / (y1 - y0 + 1e-300)


def reconstruct_h(table_path, xg, P, Tc, SL, n_h=2000):
    """Reconstruct h(x) [MJ/kg] by inverting the phz table at z = 0 against the reference.
    Returns (h, mode) with mode in {'s_l', 'T'} per point."""
    V = VTKSampler(table_path)
    hax = np.asarray(pv.read(table_path).y)                 # coord2 = enthalpy axis [MJ/kg]
    hgrid = np.linspace(hax[0], hax[-1], n_h)               # sample the interpolant finely (expose nodes)
    zc = np.zeros_like(hgrid)
    h = np.full(len(xg), np.nan); mode = np.empty(len(xg), dtype=object)
    for i in range(len(xg)):
        V.sample_at(np.column_stack([zc, hgrid, np.full_like(hgrid, P[i])]))
        if 0.02 < SL[i] < 0.98:                             # two-phase -> invert S_l(h) = s_liq
            Sl = np.asarray(V.sampled_could.point_data["S_l"])
            h[i] = _invert(hgrid, Sl, SL[i]); mode[i] = "s_l"
        else:                                               # single-phase -> invert T(h) = T_ref
            T = np.asarray(V.sampled_could.point_data["Temperature"]) - 273.15   # K -> degC
            h[i] = _invert(hgrid, T, Tc[i]); mode[i] = "T"
    return h, mode


def wiggle_metrics(xg, h):
    """(sign_changes, rms_curvature, tv_ratio) -- a smooth monotone h(x) gives 0, ~0, ~1."""
    m = np.isfinite(h); hh = h[m]
    if hh.size < 3:
        return np.nan, np.nan, np.nan
    dh = np.diff(hh)
    sign_changes = int(np.sum(np.diff(np.sign(dh)) != 0))   # 0 if monotone
    rms_curv = float(np.sqrt(np.mean(np.diff(hh, 2) ** 2)))
    span = max(hh.max() - hh.min(), 1e-12)
    tv_ratio = float(np.sum(np.abs(dh)) / span)             # 1 if monotone, >1 the more it back-tracks
    return sign_changes, rms_curv, tv_ratio


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fig 6 left (pure water) p-h table resolution test.")
    ap.add_argument("--tables", nargs="+", default=DEFAULT_TABLES,
                    help="phz .vtr tables (name in driesner_vtk_files/, or an absolute path)")
    ap.add_argument("--npts", type=int, default=140, help="reference resample points along the column")
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "fig6_pw_ph_consistency.png"))
    args = ap.parse_args(argv)

    xp, P = _load_ref("fig_6_pw_pressure_raw.csv")          # km, MPa
    xt, Tc = _load_ref("fig_6_pw_temperature_raw.csv")      # km, degC
    xs, SL = _load_ref("fig_6_pw_saturation_liq_raw.csv")   # km, -
    lo = max(xp.min(), xt.min(), xs.min()); hi = min(xp.max(), xt.max(), xs.max())
    xg = np.linspace(lo, hi, args.npts)
    Pg = np.interp(xg, xp, P); Tg = np.interp(xg, xt, Tc); Sg = np.interp(xg, xs, SL)
    print(f"Fig-6-left reference: {args.npts} pts on x=[{lo:.3f},{hi:.3f}] km  "
          f"p[{Pg.min():.2f},{Pg.max():.2f}]MPa  T[{Tg.min():.1f},{Tg.max():.1f}]C  s_liq[{Sg.min():.2f},{Sg.max():.2f}]")
    n_two = int(np.sum((Sg > 0.02) & (Sg < 0.98)))
    print(f"  phase split: {n_two} two-phase pts (invert S_l), {args.npts - n_two} single-phase (invert T)\n")

    results = {}; nodes = {}
    for t in args.tables:
        path = t if os.path.isabs(t) else os.path.join(VTK_DIR, t)
        h, mode = reconstruct_h(path, xg, Pg, Tg, Sg)
        results[os.path.basename(t)] = (h, mode)
        nodes[os.path.basename(t)] = int(np.asarray(pv.read(path).y).size)
    # deviation from the FINEST table (most h-nodes) = "truth"; isolates table wiggle from physics
    truth = max(nodes, key=nodes.get); h_truth = results[truth][0]
    print(f"reference ('truth') = finest table: {truth} ({nodes[truth]} h-nodes)\n")
    print(f"{'table':<26} {'h-nodes':>8} {'sign-chg':>9} {'tv/span':>8} {'maxdev':>9} {'rmsdev':>9}  (dev = |h - truth|, MJ/kg)")
    for name, (h, _) in results.items():
        sc, rc, tv = wiggle_metrics(xg, h)
        m = np.isfinite(h) & np.isfinite(h_truth)
        maxdev = float(np.max(np.abs(h[m] - h_truth[m]))); rmsdev = float(np.sqrt(np.mean((h[m] - h_truth[m])**2)))
        print(f"{name:<26} {nodes[name]:>8} {sc:>9d} {tv:>8.3f} {maxdev:>9.4f} {rmsdev:>9.4f}")

    # --- figure: reconstructed h(x) overlay + the reference context ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    two = (Sg > 0.02) & (Sg < 0.98)
    for name, (h, mode) in results.items():
        ax[0].plot(xg, h, "-", lw=1.4, label=name)
    ax[0].axvspan(xg[two].min() if two.any() else 0, xg[two].max() if two.any() else 0,
                  color="0.9", zorder=0, label="two-phase band")
    ax[0].set(title="reconstructed enthalpy h(x)  [MJ/kg]", xlabel="distance [km]", ylabel="h [MJ/kg]")
    ax[0].legend(fontsize=7)
    ax[1].plot(xg, Tg, "r-", label="T [degC]"); ax[1].plot(xg, Pg * 60, "b--", label="p x60 [MPa]")
    ax[1].plot(xg, Sg * 300, "g:", label="s_liq x300"); ax[1].set(title="Fig-6-left reference",
              xlabel="distance [km]"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\n[fig] {args.out}\n  -> smooth/monotone h(x) (sign-chg ~ 0, tv/span ~ 1) = table fine enough;"
          " wiggles = too coarse in p-h.")


if __name__ == "__main__":
    main()
