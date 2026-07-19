#!/usr/bin/env python
"""PorePy figures for subsection 4.2, from the exported VTU snapshots (no simulations).

Per N in --nphase (default 3 4) and per case (fixed-dim, --md):
  saturation_grid_pp_hu[_md]  -- per-phase saturation grid (rows = days, cols = phases),
                                 layout mirroring plot_reference.plot_grid
  conservation_pp_hu[_md]     -- RELATIVE losses of the volume-averaged totals vs time:
                                     mass   |<rho>(t) - <rho>(0)| / <rho>(0),
                                     energy |<E>(t) - <E>(t_1)| / |<E>(t_1)|,
                                     rho_mix = sum_i s_i rho_i,
                                     E = phi (rho h - p) + (1-phi) rho_s c_ps T,
                                 with the actual specific volumes A (2D), len*a (1D), a^2 (0D).
                                 Energy is referenced to the FIRST post-step snapshot t_1: the
                                 first Newton step re-gauges the elliptic mixture h (constant
                                 ~1e3 offset vs the t=0 export), after which E is conserved.
  saturation_conservation_pp_hu[_md] -- per-phase RELATIVE pore-volume losses
                                 |V_i(t) - V_i(0)| / V_pore, V_i = phi int s_i dV,
                                 V_pore = phi V_tot (uniform phi cancels); these decompose the
                                 mass drift: M(t) - M(0) = sum_i rho_i (V_i(t) - V_i(0)).

Reads visualization_barriers[_frac]_hu_N<n>/ ; writes into figures[_n4]/.
Usage: python plot_porepy.py [--nphase 3 4] [--days 0 78 571]
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np

import hamon_2d_solver as H                       # barrier_mask + scheme conventions (no porepy)

HERE = os.path.dirname(os.path.abspath(__file__))
DAY = 86400.0
LX = LY = 100.0
PHI = 0.3                                          # = porepy_2d_solver porosity
APERTURE = 1.0e-1                                  # = porepy_2d_solver FRACTURE_APERTURE
H_MIN, H_MAX = 1.0, 3.0                            # = porepy_2d_solver enthalpy bounds
RHO_S = 2500.0                                     # = solid_constants density
CP_S = 0.0035                                      # = solver C_P (caloric T = h/C_P AND rock c_p)

_PHASE_FIELDS = {3: ("s_water", "s_oil", "s_gas"),
                 4: ("s_water", "s_oil1", "s_oil2", "s_gas")}


def _fractures_ref():
    """The (x0, y0, x1, y1) fracture list, parsed from porepy_2d_solver.py's source."""
    src = open(os.path.join(HERE, "porepy_2d_solver.py")).read()
    m = re.search(r"_FRACTURES_REF = (\[.*?\n\])", src, re.S)
    return ast.literal_eval(re.sub(r"#.*", "", m.group(1)))


def _case_dir(n, md):
    return os.path.join(HERE, f"visualization_barriers{'_frac' if md else ''}_hu_N{n}")


def read_pvd(case_dir):
    """Master .pvd -> sorted list of (t_days, {dim: vtu_path}); mortar files skipped."""
    master = [f for f in os.listdir(case_dir)
              if f.endswith(".pvd") and not re.search(r"_\d{6}\.pvd$", f)][0]
    by_time: dict[float, dict[int, str]] = {}
    for ds in ET.parse(os.path.join(case_dir, master)).getroot().iter("DataSet"):
        f = ds.attrib["file"]
        if "mortar" in f:
            continue
        dim = int(re.search(r"_(\d)_\d{6}\.vtu$", f).group(1))
        by_time.setdefault(float(ds.attrib["timestep"]) / DAY, {})[dim] = os.path.join(case_dir, f)
    return sorted(by_time.items())


def _image(arr1d, nx, ny):
    """Cell array (c = j*nx + i, j = 0 bottom) -> image with row 0 = top (as plot_reference)."""
    return np.flipud(np.asarray(arr1d, float).reshape(ny, nx))


def _cell_measures(mesh, dim):
    """Per-cell SPECIFIC volume: area (2D quads), length*a (1D lines), a^2 (0D vertices)."""
    pts = mesh.points
    if dim == 2:
        quads = mesh.cells_dict["quad"]
        x, y = pts[:, 0][quads], pts[:, 1][quads]
        # shoelace: abs of the SIGNED sum (per-edge abs would weight area by elevation)
        return 0.5 * np.abs((x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1))
    if dim == 1:
        lines = mesh.cells_dict["line"]
        return np.linalg.norm(pts[lines[:, 1]] - pts[lines[:, 0]], axis=1) * APERTURE
    return np.full(sum(len(b) for b in mesh.cells), APERTURE ** 2)


def _field(mesh, name):
    return np.concatenate([np.asarray(b, float) for b in mesh.cell_data[name]])


def totals(files, n):
    """Volume-AVERAGED mixture mass and energy for one snapshot, as in
    tests/functional/setups/buoyancy_flow_model.py:

        mass   = sum_grids int rho_mix dV / V_tot,        rho_mix = sum_i s_i rho_i,
        energy = sum_grids int [ phi (rho_mix h - p) + (1 - phi) rho_s c_ps T ] dV / V_tot,

    with the ACTUAL specific volumes of every dimension (area, len*a, a^2).  The energy is the
    TOTAL internal energy: fluid phi(rho h - p) (h, p = exported mixture fields) + rock
    (1-phi) rho_s c_ps T, with the solver's solid constants."""
    rho = np.linspace(1500.0, 500.0, n)
    m_int = e_int = v_tot = 0.0
    for dim, path in files.items():
        mesh = meshio.read(path)
        v = _cell_measures(mesh, dim)
        s = [_field(mesh, key) for key in _PHASE_FIELDS[n]]
        rho_mix = sum(rho[i] * s[i] for i in range(n))
        e_fluid = PHI * (rho_mix * _field(mesh, "enthalpy") - _field(mesh, "pressure"))
        e_rock = (1.0 - PHI) * RHO_S * CP_S * _field(mesh, "temperature")
        m_int += float(np.sum(rho_mix * v))
        e_int += float(np.sum((e_fluid + e_rock) * v))
        v_tot += float(v.sum())
    return m_int / v_tot, e_int / v_tot


def phase_volumes(files, n):
    """Phase volumes V_i = int s_i dV for one snapshot (s_i is already intensive; its
    volume integral IS the conserved measure -- no rho, phi, or V_tot factors)."""
    vol = np.zeros(n)
    for dim, path in files.items():
        mesh = meshio.read(path)
        v = _cell_measures(mesh, dim)
        for i, key in enumerate(_PHASE_FIELDS[n]):
            vol[i] += float(np.sum(_field(mesh, key) * v))
    return vol


def saturation_conservation(n, md, out_dir):
    """Per-phase PORE-volume losses |V_i(t) - V_i(0)| / V_pore vs time, with
    V_i = phi int s_i dV and V_pore = phi V_tot (phi is uniform, so it cancels and the
    ratio is the relative loss |int s_i dV| / V_tot).  rho_i is constant, so these
    decompose the mass drift exactly: M(t) - M(0) = sum_i rho_i (V_i(t) - V_i(0))."""
    series = read_pvd(_case_dir(n, md))
    t = np.array([s[0] for s in series])
    S = np.array([phase_volumes(s[1], n) for s in series])      # (nt, n) bulk int s_i dV
    v_tot = float(S[0].sum())                                   # sum_i s_i = 1 -> V_tot at t=0
    loss, t = np.abs(S[1:] - S[0]) / v_tot, t[1:]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, key in enumerate(_PHASE_FIELDS[n]):
        ax.semilogy(t, loss[:, i], "o-", ms=3, label=key.replace("s_", "$s_{") + "}$")
    for level in (1.0e-3, 1.0e-4):
        ax.axhline(level, color="0.6", lw=1.0, ls="--", zorder=0)
        ax.text(t[-1], level, f" {level:.0e}", color="0.45", fontsize=8, va="bottom")
    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"pore-volume loss  $|V_i(t)-V_i(0)| / V_{pore}$,"
                  "\n"
                  r"$V_i = \phi\int s_i\,dV$,  $V_{pore} = \phi V_{tot}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)
    tag = "mixed-dimensional" if md else "fixed-dimensional"
    ax.set_title(f"PorePy HU-BM(mp), N={n}, {tag}: saturation conservation")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"saturation_conservation_pp_hu{'_md' if md else ''}.png"))


def _save(fig, png):
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(png[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(png, HERE)} (+ .pdf)")


def _overlay(ax, md):
    B = _image(H.barrier_mask(100, 100), 100, 100).astype(bool)   # mask is FLAT (c = j*nx+i)
    rgba = np.zeros(B.shape + (4,))
    rgba[..., 3] = np.where(B, 0.95, 0.0)
    ax.imshow(rgba, extent=[0, LX, LY, 0], aspect="equal", interpolation="nearest")
    if md:
        for x0, y0, x1, y1 in _fractures_ref():
            ax.plot([x0, x1], [LY - y0, LY - y1], color="k", lw=1.0, alpha=0.8)


def saturation_grid(n, md, days, out_dir):
    snaps = dict(read_pvd(_case_dir(n, md)))
    keys = _PHASE_FIELDS[n]
    fig, axes = plt.subplots(len(days), n, figsize=(3.1 * n, 3.3 * len(days)), squeeze=False)
    im = None
    for i, day in enumerate(days):
        t = min(snaps, key=lambda s: abs(s - day))
        mesh = meshio.read(snaps[t][2])
        for j, key in enumerate(keys):
            ax = axes[i][j]
            im = ax.imshow(_image(_field(mesh, key), 100, 100), extent=[0, LX, LY, 0],
                           aspect="equal", cmap="coolwarm", vmin=0.0, vmax=1.0,
                           interpolation="nearest")
            _overlay(ax, md)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(key.replace("s_", "$s_{") + "}$", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{int(round(t))} days", fontsize=11)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, ticks=[0, 0.5, 1])
    cbar.set_label("phase saturation $s_k$")
    tag = "mixed-dimensional" if md else "fixed-dimensional"
    fig.suptitle(f"PorePy HU-BM(mp), N={n}, {tag}: per-phase saturations", y=1.00)
    _save(fig, os.path.join(out_dir, f"saturation_grid_pp_hu{'_md' if md else ''}.png"))


def conservation(n, md, out_dir):
    """Mass/energy losses |q(t) - q(0)| of the volume-averaged quantities (test-style,
    absolute, NOT relative): mass on the left axis, energy on the right, gray order
    lines at 1e-3 and 1e-4 marking the expected mass-preservation decades."""
    series = read_pvd(_case_dir(n, md))
    t = np.array([s[0] for s in series])
    q = np.array([totals(s[1], n) for s in series])             # (nt, 2): mass, energy
    # drop the zero-by-definition reference points: mass vs t=0 (exact IC); energy vs the FIRST
    # post-step snapshot t_1 (the first Newton step re-gauges the elliptic mixture h -- a
    # constant ~1e3 offset vs the t=0 export that would bury the actual drift)
    # RELATIVE losses (dimensionless): each drift normalized by its reference value
    loss_m, t_m = np.abs(q[1:, 0] - q[0, 0]) / abs(q[0, 0]), t[1:]
    loss_e, t_e = np.abs(q[2:, 1] - q[1, 1]) / abs(q[1, 1]), t[2:]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()
    l1, = ax1.semilogy(t_m, loss_m, "o-", color="C0", ms=3, label="mass")
    l2, = ax2.semilogy(t_e, loss_e, "s-", color="C3", ms=3, label="energy")
    # order line at 1e-3 on EACH axis, colored like its curve (left/mass blue, right/energy red)
    ax1.axhline(1.0e-3, color="C0", lw=1.0, ls="--", alpha=0.6, zorder=0)
    ax1.text(t_m[-1], 1.0e-3, " 1e-03", color="C0", fontsize=8, va="bottom")
    ax2.axhline(1.0e-3, color="C3", lw=1.0, ls="--", alpha=0.6, zorder=0)
    ax2.text(t_m[0], 1.0e-3, " 1e-03", color="C3", fontsize=8, va="bottom", ha="left")
    ax1.set_xlabel("time [days]")
    ax1.set_ylabel(r"relative mass loss  "
                   r"$|\langle\rho\rangle(t)-\langle\rho\rangle(0)| / \langle\rho\rangle(0)$",
                   color="C0")
    ax2.set_ylabel(r"relative total-energy loss  "
                   r"$|\langle E \rangle(t) - \langle E \rangle(t_1)| / |\langle E \rangle(t_1)|$,"
                   "\n"
                   r"$E = \phi(\rho h - p) + (1{-}\phi)\rho_s c_{ps} T$", color="C3")
    ax1.tick_params(axis="y", colors="C0"); ax2.tick_params(axis="y", colors="C3")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(handles=[l1, l2], loc="lower right", fontsize=9)
    tag = "mixed-dimensional" if md else "fixed-dimensional"
    ax1.set_title(f"PorePy HU-BM(mp), N={n}, {tag}: conservation vs time")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"conservation_pp_hu{'_md' if md else ''}.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nphase", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--days", type=float, nargs="+", default=[0.0, 78.0, 571.0])
    args = ap.parse_args(argv)
    for n in args.nphase:
        out_dir = os.path.join(HERE, "figures" if n == 3 else f"figures_n{n}")
        os.makedirs(out_dir, exist_ok=True)
        for md in (False, True):
            if not os.path.isdir(_case_dir(n, md)):
                print(f"  (skip N={n} md={md}: no output dir)")
                continue
            saturation_grid(n, md, args.days, out_dir)
            conservation(n, md, out_dir)
            saturation_conservation(n, md, out_dir)


if __name__ == "__main__":
    main()
