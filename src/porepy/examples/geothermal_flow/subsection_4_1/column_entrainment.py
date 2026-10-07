#!/usr/bin/env python
"""Intermediate-phase shear-entrainment column test (subsection 4.1).

A 1-D vertical column, density-inverted: heavy alpha on top, light gamma on the bottom,
a passive intermediate phase beta in a middle band. Alpha sinks and gamma rises in a
counter-current; the question is which stream entrains beta. Three runs sweep the
intermediate density across the alpha-gamma pair:

    near-heavy   rho_b = 3/4 rho_a + 1/4 rho_g = 1250   -> m_b^{ag} = 1    (joins heavy)
    exact        rho_b = 1/2 rho_a + 1/2 rho_g = 1000   -> m_b^{ag} = 1/2  (on the separator)
    near-light   rho_b = 1/4 rho_a + 3/4 rho_g =  750   -> m_b^{ag} = 0    (joins light)

m_b^{ag} is the midpoint background weight of Definition 3.3 (hamon_2d_solver._side_masks):
the arithmetic mean of the pair densities is the separator, so beta's mobility is grouped
with the heavy or the light stream depending on its side of it. This test shows beta is
sheared DOWN / stays / is entrained UP accordingly -- the operator transfers drag to the
passive phase rather than leaving it a non-interacting spectator.

Reuses the real solver (same flux operator, mobilities, Newton) from hamon_2d_solver; only
the grid (barrier-free column), the IC (alpha/beta/gamma bands) and the densities differ.

Run:  python column_entrainment.py            # writes figures/column_entrainment.{png,pdf}
      python column_entrainment.py --scheme hu-mp --ny 100 --t-end-days 30
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hamon_2d_solver as H  # noqa: E402  (solver primitives + the HU-BM operator under test)

RHO_A, RHO_G = 1500.0, 500.0                       # heavy (alpha) / light (gamma) [kg/m^3]
CASES = {                                          # label -> (rho_beta, expected m_beta)
    "near_heavy": 0.75 * RHO_A + 0.25 * RHO_G,     # 1250
    "exact":      0.50 * RHO_A + 0.50 * RHO_G,     # 1000
    "near_light": 0.25 * RHO_A + 0.75 * RHO_G,     # 750
}
CASE_TITLE = {
    "near_heavy": r"near-heavy  $\rho_\beta=1250$",
    "exact":      r"exact mean  $\rho_\beta=1000$",
    "near_light": r"near-light  $\rho_\beta=750$",
}
CASE_COLOR = {"near_heavy": "#b2182b", "exact": "#4d4d4d", "near_light": "#2166ac"}
SCHEMES = ("hu-mp", "ppu")                         # HU (= hu-mp) vs PPU
DISPLAY = {"hu-mp": "HU", "ppu": "PPU"}            # figure labels (paper terminology)


# --------------------------------------------------------------------------- phase system
def set_densities(rho_beta: float) -> float:
    """Configure the module for 3 phases with densities [1500, rho_beta, 500] and return the
    midpoint background weight m_beta of the (alpha, gamma) pair (Definition 3.3)."""
    H.set_phase_system(3)                          # N=3 scaffolding (pairs, perm, bands)
    H.RHO = np.array([RHO_A, float(rho_beta), RHO_G])
    H.RHO_BANDS = H.RHO.copy()
    H._MASKS = H._side_masks()                     # rebuild the midpoint masks for the new RHO
    H._THETA = H._pinned_theta()
    return float(H._MASKS[(0, 2)][1])              # m_beta for pair (alpha=0, gamma=2)


# --------------------------------------------------------------------------- column grid / IC
def column_grid(ny: int) -> H.Grid:
    """Barrier-free, uniform-permeability 1-D column (nx=1): only vertical faces, with gravity."""
    nx, dx, dy = 1, H.LX, H.LY / ny
    ncell = ny
    yc = (np.arange(ny) + 0.5) * dy
    Tf = np.full(ny - 1, (dx / dy) * H.K_ROCK)     # harmonic(K, K) = K on a uniform column
    return H.Grid(
        nx=nx, ny=ny, dx=dx, dy=dy, ncell=ncell,
        xc=np.full(ncell, 0.5 * dx), yc=yc,
        Kcell=np.full(ncell, H.K_ROCK), barrier=np.zeros(ncell, bool),
        fL=np.arange(ny - 1), fR=np.arange(1, ny),
        Tf=Tf, GC=Tf * H.G * dy, Vcell=dx * dy,
        frac=np.zeros(ncell, bool), Vc=np.full(ncell, dx * dy),
        xe=np.array([0.0, dx]), ye=np.arange(ny + 1) * dy)


def column_ic(grid: H.Grid, mid_frac: float) -> np.ndarray:
    """(3, ncell) IC: alpha fills the top, gamma the bottom, beta a central band of height
    ``mid_frac * LY`` -- the density-inverted column that segregates."""
    y = grid.yc
    lo, hi = 0.5 * H.LY - 0.5 * mid_frac * H.LY, 0.5 * H.LY + 0.5 * mid_frac * H.LY
    s = np.zeros((3, grid.ncell))
    s[0][y >= hi] = 1.0                            # alpha (heavy) on top
    s[2][y <= lo] = 1.0                            # gamma (light) on the bottom
    s[1][(y > lo) & (y < hi)] = 1.0                # beta in the middle band
    return s


def hydrostatic_p(grid: H.Grid, s: np.ndarray) -> np.ndarray:
    """Hydrostatic pressure of the initial density column (so grad p = rho g within each band
    and only the interface density jumps drive flow at t=0). Datum: zero mean (Lagrange gauge)."""
    rho = (s * H.RHO[:, None]).sum(axis=0)         # per-cell mixture density
    above = np.cumsum(rho[::-1])[::-1] - rho       # sum of rho in the cells strictly above
    p = H.G * grid.dy * (above + 0.5 * rho)
    return p - p.mean()


def com_beta(grid: H.Grid, s_beta: np.ndarray) -> float:
    """Saturation-weighted centre-of-mass height of phase beta [m]."""
    w = np.clip(s_beta, 0.0, None)
    return float((w * grid.yc).sum() / max(w.sum(), 1e-30))


# --------------------------------------------------------------------------- one case
def run_case(rho_beta, scheme, ny=100, mid_frac=1.0 / 3.0, t_end_days=30.0,
             dt_days=0.25, dt_init_days=0.05, fixed_dt_days=None, verbose=True):
    """Advance one column to ``t_end_days`` with the real solver. Returns
    (m_beta, times[days], com[m], profiles s_beta[(ntime, ny)], y[m])."""
    m_beta = set_densities(rho_beta)
    grid = column_grid(ny)
    pattern = H.sparsity_pattern(grid)
    linsolve = H.make_linear_solver("scipy")       # 300-DoF system: direct solve is instant
    s0 = column_ic(grid, mid_frac)
    nc = grid.ncell
    x = np.concatenate([hydrostatic_p(grid, s0), s0[0], s0[1]])   # [p, s_alpha, s_beta]

    def _reorder(s_col):                            # cell j=0 is the bottom -> flip to top-down row
        return s_col[::-1]

    times = [0.0]
    com = [com_beta(grid, s0[1])]
    prof = [_reorder(s0[1])]
    t, t_end = 0.0, t_end_days * H.DAY
    dt0, dt_init = dt_days * H.DAY, dt_init_days * H.DAY
    dt_floor = min(dt0 / 64.0, dt_init)
    dt = min(max(dt_init, dt_floor), dt0)
    maxit = H.NEWTON_MAXIT
    if fixed_dt_days is not None:                        # fixed step: no growth, no cutting
        dt = dt0 = dt_floor = fixed_dt_days * H.DAY
        maxit = 60                                       # allow more Newton iters for the large step
    n_steps = total_it = max_it = n_cuts = it_wasted = 0
    while t < t_end - 1e-9:
        step_dt = min(dt, t_end - t)
        s_old = x[nc:].reshape(2, nc).copy()
        dirs = H.frozen_directions(x, grid, scheme)
        r = H.make_residual(grid, step_dt, s_old, dirs)

        def relag(xx, _d=dirs):
            nd = H.frozen_directions(xx, grid, scheme)
            _d.upT, _d.up_phase = nd.upT, nd.up_phase

        xn, its, m, ok = H.newton(r, x, pattern, grid, step_dt, maxit=maxit,
                                  linsolve=linsolve, relag=relag)
        if not ok and step_dt > dt_floor + 1e-30:
            n_cuts += 1
            it_wasted += its
            dt = max(step_dt * 0.5, dt_floor)
            continue
        x, t = xn, t + step_dt
        n_steps += 1
        total_it += its
        max_it = max(max_it, its)
        s = H._saturations(x, nc)
        times.append(t / H.DAY)
        com.append(com_beta(grid, s[1]))
        prof.append(_reorder(s[1]))
        if ok and its < 5 and dt < dt0:
            dt = min(dt * 2.0, dt0)
    s_fin = H._saturations(x, nc)                   # final full saturations (3, ncell)
    rho_fin = (s_fin * H.RHO[:, None]).sum(axis=0)  # mixture density rho = sum_k s_k rho_k (bottom-up)
    stats = {"n_steps": n_steps, "total_it": total_it, "max_it": max_it,
             "avg_it": total_it / max(n_steps, 1), "n_cuts": n_cuts, "it_wasted": it_wasted,
             "rho_final": rho_fin, "t_final_days": t / H.DAY}
    if verbose:
        print(f"  {scheme:6s} rho_b={rho_beta:6.0f}  m_beta={m_beta:.2f}  "
              f"com: {com[0]:.2f} -> {com[-1]:.2f} m  (d={com[-1]-com[0]:+.2f} m)  "
              f"steps={n_steps}  total_it={total_it}  avg={stats['avg_it']:.2f}  "
              f"max={max_it}  cuts={n_cuts}", flush=True)
    return m_beta, np.array(times), np.array(com), np.array(prof), grid.yc[::-1], stats


# --------------------------------------------------------------------------- figure
def make_figure(results, out_dir, suffix="", perm_factor=1.0, dt_note="", ny=None):
    """results[case][scheme] = (m_beta, t, com, prof, y, stats). Top row: beta saturation space-time
    map (HU) per case with the HU (solid) and PPU (dashed) centre-of-mass tracks overlaid. Bottom
    row: the HU-vs-PPU beta-saturation difference |s_beta^HU - s_beta^PPU| per case (both schemes
    interpolated onto a common time grid). Barriers colormap (vlag)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import plot_reference as PR            # same colormap the barriers figures use
        cmap = PR._cmap("vlag")
    except Exception:                           # pragma: no cover
        import seaborn as sns
        cmap = sns.color_palette("vlag", as_cmap=True)

    labels = ["near_heavy", "exact", "near_light"]
    t_end = max(results[l]["hu-mp"][1][-1] for l in labels)
    tg = np.linspace(0.0, t_end, 240)           # common time grid for the difference

    def on_grid(times, prof):                   # (ntime, ny) -> (len(tg), ny), interp per height
        return np.column_stack([np.interp(tg, times, prof[:, j]) for j in range(prof.shape[1])])

    # beta-saturation difference per case (HU vs PPU on the shared grid)
    diffs = {}
    for lab in labels:
        prof_h, prof_p = results[lab]["hu-mp"][3], results[lab]["ppu"][3]
        t_h, t_p = results[lab]["hu-mp"][1], results[lab]["ppu"][1]
        diffs[lab] = np.abs(on_grid(t_h, prof_h) - on_grid(t_p, prof_p))
    dmax = max(float(d.max()) for d in diffs.values()) or 1.0

    fig = plt.figure(figsize=(13.0, 8.6), constrained_layout=True)
    axes = fig.subplots(2, 3)

    s_im = d_im = None
    for k, lab in enumerate(labels):
        m_beta, t, com, prof, yc, _ = results[lab]["hu-mp"]      # top map = HU field
        _, t_p, com_p, _, _, _ = results[lab]["ppu"]
        ax = axes[0][k]
        s_im = ax.pcolormesh(t, yc, prof.T, cmap=cmap, vmin=0.0, vmax=1.0,
                             shading="gouraud", rasterized=True)
        ax.plot(t, com, color="k", lw=1.6, label="HU")                 # beta CoM, HU
        ax.plot(t_p, com_p, color="#15b01a", lw=1.5, ls="--", label="PPU")
        ax.set_title(f"{CASE_TITLE[lab]}\n$m_\\beta^{{\\alpha\\gamma}}={m_beta:.1f}$", fontsize=12)
        ax.set_xlim(0, t_end)
        ax.set_ylim(0, H.LY)
        ax.tick_params(labelbottom=False)
        if k == 0:
            ax.set_ylabel("height $y$ [m]")
            ax.legend(loc="lower left", fontsize=8, framealpha=0.85,
                      title=r"$\bar{y}_\beta=\dfrac{\int_\Omega s_\beta\,y\,\mathrm{d}V}"
                            r"{\int_\Omega s_\beta\,\mathrm{d}V}$")

        axd = axes[1][k]                                              # bottom = |s_beta^HU - s_beta^PPU|
        d_im = axd.pcolormesh(tg, yc, diffs[lab].T, cmap=cmap, vmin=0.0, vmax=dmax,
                              shading="gouraud", rasterized=True)
        axd.set_xlim(0, t_end)
        axd.set_ylim(0, H.LY)
        axd.set_xlabel("time [days]")
        if k == 0:
            axd.set_ylabel("height $y$ [m]")

    cb = fig.colorbar(s_im, ax=list(axes[0]), location="right", fraction=0.04, pad=0.02, shrink=0.9)
    cb.set_label(r"$\beta$ saturation $s_\beta$ [-]  (HU)")
    cd = fig.colorbar(d_im, ax=list(axes[1]), location="right", fraction=0.04, pad=0.02, shrink=0.9)
    cd.set_label(r"$|s_\beta^{\mathrm{HU}}-s_\beta^{\mathrm{PPU}}|$ [-]")

    K_SI = 1000.0 * H.MILLI_DARCY * perm_factor                      # rock permeability [m^2]
    dy = (H.LY / ny) if ny else None
    htag = f",  $\\Delta y={dy:g}$ m" if dy else ""
    fig.suptitle(
        f"HU vs PPU:  $\\rho_\\alpha={RHO_A:.0f}$ and $\\rho_\\gamma={RHO_G:.0f}$ kg m$^{{-3}}$     "
        f"($K={K_SI:.3e}$ m$^2$,  $t_f={t_end:.0f}$ d{dt_note}{htag})", fontsize=13)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"column_entrainment{suffix}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("wrote", os.path.relpath(path, HERE))
    plt.close(fig)


# --------------------------------------------------------------------------- density 3-panel
def make_density_comparison(results, case, out_dir, suffix="", perm_factor=1.0, dt_note=""):
    """Three panels for one column (final time): mixture density rho = sum_k s_k rho_k for PPU
    (left) and HU (middle) -- same barriers colormap (vlag reversed, heavy=blue), with the cell
    mesh as a wireframe -- and their relative difference |rho_PPU - rho_HU|/||rho_PPU|| (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import plot_reference as PR            # same colormap helper the barriers figures use
        cm = PR._cmap("vlag")
    except Exception:                           # pragma: no cover
        import seaborn as sns
        cm = sns.color_palette("vlag", as_cmap=True)
    cm_rho = cm.reversed()                       # heavy (high rho) -> blue, light -> red

    rho_ppu = results[case]["ppu"][5]["rho_final"]      # bottom-up (ny,)
    rho_hu = results[case]["hu-mp"][5]["rho_final"]
    ny = rho_ppu.size
    ye = np.linspace(0.0, H.LY, ny + 1)                 # cell edges in height [m]
    W = 14.0                                            # display width of the column strip
    xe = np.array([0.0, W])
    diff = np.abs(rho_ppu - rho_hu) / np.sqrt(np.mean(rho_ppu ** 2))   # barriers-style relative
    dmax = max(float(diff.max()), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 6.4), constrained_layout=True)

    def strip(ax, vals, cmap, vmin, vmax):
        pcm = ax.pcolormesh(xe, ye, vals.reshape(ny, 1), cmap=cmap, vmin=vmin, vmax=vmax,
                            shading="flat")
        ax.set_xlim(0, W)
        ax.set_ylim(0, H.LY)
        ax.set_xticks([])
        return pcm

    im = strip(axes[0], rho_ppu, cm_rho, 500.0, 1500.0)
    strip(axes[1], rho_hu, cm_rho, 500.0, 1500.0)
    dim = strip(axes[2], diff, cm, 0.0, dmax)

    it_p = results[case]["ppu"][5]["total_it"]
    it_h = results[case]["hu-mp"][5]["total_it"]
    axes[0].set_title(f"PPU\n({it_p} it)", fontsize=12)
    axes[1].set_title(f"HU\n({it_h} it)", fontsize=12)
    axes[2].set_title("L2 difference", fontsize=12)
    axes[0].set_ylabel("height $y$ [m]")
    for ax in (axes[1], axes[2]):
        ax.set_yticklabels([])

    cb = fig.colorbar(im, ax=[axes[0], axes[1]], location="bottom", fraction=0.05,
                      pad=0.03, aspect=32, ticks=[500, 1000, 1500])
    cb.set_label(r"mixture density $\rho=\sum_k s_k\,\rho_k$  [kg m$^{-3}$]", fontsize=10)
    dcb = fig.colorbar(dim, ax=axes[2], location="bottom", fraction=0.05, pad=0.03, aspect=12)
    dcb.set_label(r"$|\rho_{\mathrm{PPU}}-\rho_{\mathrm{HU}}|\,/\,\|\rho_{\mathrm{PPU}}\|$",
                  fontsize=9)

    t_end = results[case]["hu-mp"][5]["t_final_days"]
    rho_beta = CASES[case]
    k_md = perm_factor * 1000.0                           # rock permeability [mD] (base 1000 mD)
    fig.suptitle(
        f"$\\rho_\\alpha={RHO_A:.0f}$,  $\\rho_\\beta={rho_beta:.0f}$,  "
        f"$\\rho_\\gamma={RHO_G:.0f}$ kg m$^{{-3}}$     "
        f"($K={k_md:g}$ mD,  $t={t_end:.0f}$ d{dt_note},  max diff ${dmax:.1e}$)", fontsize=12)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"column_density_{case}{suffix}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("wrote", os.path.relpath(path, HERE))
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ny", type=int, default=200,
                    help="column cells in the vertical (default 200 = 2x the base refinement)")
    ap.add_argument("--mid-frac", type=float, default=1.0 / 3.0,
                    help="height fraction of the central beta band (default 1/3)")
    ap.add_argument("--t-end-days", type=float, default=60.0, help="run horizon [days] (default 60)")
    ap.add_argument("--perm-factor", type=float, default=1.0,
                    help="scale the rock permeability by this factor (default 1.0 = 1000 mD); "
                         "higher K speeds up the segregation")
    ap.add_argument("--dt-fixed-days", type=float, default=None,
                    help="use a FIXED time step of this many days (disables the adaptive dt and "
                         "step-cutting); default None = adaptive, capped at 0.25 d")
    ap.add_argument("--out", default=os.path.join(HERE, "passive_phase_example"),
                    help="output folder (default: passive_phase_example/)")
    args = ap.parse_args()

    H.K_ROCK = 1000.0 * H.MILLI_DARCY * args.perm_factor      # scaled rock permeability
    suffix = "" if args.perm_factor == 1.0 else f"_kx{args.perm_factor:g}"
    dt_note = ""
    if args.dt_fixed_days is not None:
        suffix += f"_dt{args.dt_fixed_days:g}"
        dt_note = f",  $\\Delta t={args.dt_fixed_days:g}$ d"

    print(f"=== intermediate-phase shear entrainment  (HU = hu-mp  vs  PPU, "
          f"K x{args.perm_factor:g}, dt="
          f"{('fixed ' + str(args.dt_fixed_days) + ' d') if args.dt_fixed_days else 'adaptive'}) ===")
    results = {}
    for lab, rho_beta in CASES.items():
        results[lab] = {}
        for scheme in SCHEMES:
            results[lab][scheme] = run_case(rho_beta, scheme, ny=args.ny, mid_frac=args.mid_frac,
                                            t_end_days=args.t_end_days,
                                            fixed_dt_days=args.dt_fixed_days)
    make_figure(results, args.out, suffix=suffix, perm_factor=args.perm_factor, dt_note=dt_note,
                ny=args.ny)
    for lab in CASES:                                   # PPU | HU | L2-difference mixture density
        make_density_comparison(results, lab, args.out, suffix=suffix,
                                perm_factor=args.perm_factor, dt_note=dt_note)


if __name__ == "__main__":
    main()
