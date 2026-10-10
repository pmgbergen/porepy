#!/usr/bin/env python
"""Contours of mixture density (rho) and enthalpy for the MPFA no-gravity and gravity runs.

2x2 layout: rows = case (no gravity, gravity), columns = quantity (rho, enthalpy).
The 2-D matrix field is drawn as filled contours; the 1-D fractures (grid dim == 1) are overlaid
as thick lines (lw = 10) colored by the SAME field and the SAME hardcoded color range as the
matrix. One colorbar per column (shared across the two rows).

Reads the latest snapshot of each run (read-only). Writes co2_rho_h_contours.{png,pdf} here.

Run:  python plot_rho_h_contours.py
      python plot_rho_h_contours.py --index N
"""
import argparse
import glob
import os
import re

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

plt.rcParams.update({"font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16,
                     "xtick.labelsize": 14, "ytick.labelsize": 14})

HERE = os.path.dirname(os.path.abspath(__file__))

# rows = case (no gravity first, gravity second)
CASES = [("no gravity", "visualization_co2_inject_hu_md_mpfa_nograv"),
         ("gravity",    "visualization_co2_inject_hu_md_mpfa")]

# subsection_4_1 colormap: seaborn vlag (fallback coolwarm), matching plot_reference._cmap
def _cmap(name="vlag"):
    try:
        import seaborn as sns
        return sns.color_palette(name, as_cmap=True)
    except Exception:
        try:
            return plt.get_cmap(name)
        except ValueError:
            return plt.get_cmap("coolwarm" if name == "vlag" else "viridis")


CMAP = _cmap("vlag")

# columns = quantity: (vtu field, title, colorbar label, hardcoded range, colormap)
RHO_RANGE = (800.0, 1000.0)      # kg/m3   -- hardcoded
H_RANGE = (0.10, 0.14)           # MJ/kg   -- hardcoded
COLS = [("rho",      "mixture density",  "$\\rho$ [kg/m$^3$]", RHO_RANGE, CMAP),
        ("enthalpy", "mixture enthalpy", "$h$ [MJ/kg]",        H_RANGE,   CMAP)]

L_DOMAIN = 10.0                  # m
FRAC_LW = 1                      # fracture line width (was 10)
NLEV = 16                        # contour levels


def _latest_file(folder, dim, index=None):
    best, bi = None, -1
    for f in glob.glob(os.path.join(folder, f"*_{dim}_*.vtu")):
        m = re.search(rf"_{dim}_0*(\d+)\.vtu$", os.path.basename(f))
        if not m:
            continue
        i = int(m.group(1))
        if index is not None:
            if i == index:
                return f, i
        elif i > bi:
            bi, best = i, f
    return (best, bi) if index is None else (None, index)


def _time_days(folder, index):
    """Simulation time [days] of a snapshot, read from the pvd (timestep is in seconds)."""
    for pvd in (os.path.join(folder, f"data_{index:06d}.pvd"), os.path.join(folder, "data.pvd")):
        if os.path.exists(pvd):
            with open(pvd) as fh:
                ts = [float(m) for m in re.findall(r'timestep="([0-9.eE+-]+)"', fh.read())]
            if ts:
                return max(ts) / 86400.0
    return None


def _panel(ax, folder, field, vmin, vmax, cmap, index):
    f2, idx = _latest_file(folder, 2, index)
    m2 = pv.read(f2)
    cc = m2.cell_centers().points
    v2 = np.asarray(m2.cell_data[field], float)
    levels = np.linspace(vmin, vmax, NLEV)
    ax.tricontourf(cc[:, 0], cc[:, 1], v2, levels=levels, cmap=cmap,
                   vmin=vmin, vmax=vmax, extend="both")

    # gray wireframe of the matrix mesh
    e = m2.extract_all_edges()
    conn = e.lines.reshape(-1, 3)[:, 1:]
    ax.add_collection(LineCollection(e.points[conn][:, :, :2], colors="0.35",
                                     linewidths=0.3, zorder=2))

    f1, _ = _latest_file(folder, 1, index)
    if f1 is not None:
        m1 = pv.read(f1)
        conn = m1.cells_dict.get(3)      # VTK_LINE -> fracture (dim == 1) cells
        if conn is not None and field in m1.cell_data:
            segs = m1.points[conn][:, :, :2]
            ax.add_collection(LineCollection(segs, colors="0.15", linewidths=FRAC_LW + 1.2,
                                             capstyle="projecting", zorder=4))  # dark outline
            lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(vmin, vmax),
                                linewidths=FRAC_LW, capstyle="projecting", zorder=5)
            lc.set_array(np.asarray(m1.cell_data[field], float))
            ax.add_collection(lc)
    ax.set_xlim(0, L_DOMAIN)
    ax.set_ylim(0, L_DOMAIN)
    ax.set_aspect("equal")
    return idx


def main():
    ap = argparse.ArgumentParser(description="rho / enthalpy contours, no-gravity vs gravity.")
    ap.add_argument("--index", type=int, default=None, help="snapshot index (default: latest)")
    a = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 10.5), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.20, hspace=0.05))
    used = {}
    for r, (case_label, folder) in enumerate(CASES):
        for c, (field, title, _, (vmin, vmax), cmap) in enumerate(COLS):
            ax = axes[r, c]
            idx = _panel(ax, os.path.join(HERE, folder), field, vmin, vmax, cmap, a.index)
            used[case_label] = idx
            if r == 0:
                ax.set_title(title)
            if r == 1:
                ax.set_xlabel("x [m]")
            if c == 0:
                ax.set_ylabel(f"{case_label}\ny [m]")

    for c, (field, title, clabel, (vmin, vmax), cmap) in enumerate(COLS):
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap)
        cb = fig.colorbar(sm, ax=axes[:, c], location="bottom", shrink=0.9, pad=0.08)
        cb.set_label(clabel)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"co2_rho_h_contours.{ext}"), dpi=150, bbox_inches="tight")
    print("wrote co2_rho_h_contours.{png,pdf}  snapshot index: "
          + ", ".join(f"{k}={v}" for k, v in used.items()))


if __name__ == "__main__":
    main()
