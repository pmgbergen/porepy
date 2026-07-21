#!/usr/bin/env python
"""Weis et al. (2014) Fig. 8 (A-C) reproduction, panels placed HORIZONTALLY.

Three snapshots (5 / 15 / 50 kyr) of the heat-flux plume, rendered with pyvista from the
``visualization_<tag>/`` VTUs: the enthalpy field by default or vapor saturation with
``--field s_v`` (vlag colormap, the subsection_4_1 convention), rendered as the actual
piecewise-constant FV solution (no cell-to-point smoothing), with the paper's temperature
isotherms (100..400 degC, red) and fluid-pressure contours (5 / 15 / 25 MPa, blue), on
the central 4 km of the 9 km x 3 km domain (depth 0 at the top).

Usage: python fig_weis_2d_plume.py [--scheme hu] [--consistent] [--grid-type ...]
       [--cell-size M] [--q-anomaly W/M2] [--z-anomaly Z] [--times 5000 15000 50000]
       ->  figures/fig_8_plume_<tag>.png   (tag = case_naming.case_tag of the run flags)
"""
from __future__ import annotations

import argparse
import os
import re
import xml.etree.ElementTree as ET

import sys

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case_naming import case_tag                     # noqa: E402

try:
    import seaborn as sns

    def _cmap(name="vlag"):
        return sns.color_palette(name, as_cmap=True)
except ImportError:                                  # seaborn optional (= plot_reference)
    import matplotlib.pyplot as plt

    def _cmap(name="vlag"):
        try:
            return plt.get_cmap(name)
        except ValueError:
            return plt.get_cmap("coolwarm" if name == "vlag" else "viridis")

HERE = os.path.dirname(os.path.abspath(__file__))
DAY = 86400.0
YEAR = 365.0 * DAY
X_CENTER, X_HALF = 4500.0, 2000.0                    # central 4 km excerpt [m]
DEPTH = 3000.0
T_ISO = [100.0, 200.0, 300.0, 400.0]      # degC -- the paper's annotated isotherms
P_ISO = [5.0, 15.0, 25.0]                                    # MPa  (paper's blue contours)
_ABC = "ABC"


def _snapshots(folder):
    """Master .pvd -> sorted [(t_years, vtu_path)] for the 2D subdomain files."""
    master = [f for f in os.listdir(folder)
              if f.endswith(".pvd") and not re.search(r"_\d+\.pvd$", f)][0]
    out = []
    for ds in ET.parse(os.path.join(folder, master)).getroot().iter("DataSet"):
        f = ds.attrib["file"]
        if "mortar" in f or not re.search(r"_2_\d+\.vtu$", f):
            continue
        out.append((float(ds.attrib["timestep"]) / YEAR, os.path.join(folder, f)))
    return sorted(out)


def _panel_mesh(path):
    """Load one snapshot, clip to the central 4 km, center x about the source.
    CELL data is kept as-is so the render shows the piecewise-constant FV solution."""
    grid = pv.read(path)
    grid = grid.clip_box([X_CENTER - X_HALF, X_CENTER + X_HALF, 0.0, DEPTH,
                          -1.0, 1.0], invert=False)
    return grid.translate((-X_CENTER, 0.0, 0.0))


def _render_panel(mesh, field, cm, clim):
    """pyvista render of one panel (cell field -> flat per-cell shading), returned as
    an image array that exactly frames the 4 km x 3 km excerpt (parallel projection)."""
    pl = pv.Plotter(off_screen=True, window_size=(1200, 900))     # 4:3 = domain aspect
    pl.add_mesh(mesh, scalars=field, preference="cell", cmap=cm, clim=clim,
                show_scalar_bar=False)
    pl.view_xy()
    pl.enable_parallel_projection()
    pl.camera.focal_point = (0.0, DEPTH / 2.0, 0.0)
    pl.camera.position = (0.0, DEPTH / 2.0, 1.0e4)
    pl.camera.parallel_scale = DEPTH / 2.0            # half-height -> exact vertical frame
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors, cm as mcm

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="enthalpy", choices=["enthalpy", "s_v"],
                    help="colormap field: enthalpy (default) or vapor saturation")
    ap.add_argument("--scheme", default="hu")
    ap.add_argument("--consistent", action="store_true",
                    help="figure for the MPFA (consistent) run")
    ap.add_argument("--grid-type", default=None, choices=["cartesian", "simplex"])
    ap.add_argument("--cell-size", type=float, default=None, metavar="M")
    ap.add_argument("--q-anomaly", type=float, default=None, metavar="W/M2",
                    help="figure for a non-default anomaly heat flux run")
    ap.add_argument("--z-init", type=float, default=None, metavar="Z",
                    help="figure for a non-default initial NaCl composition run")
    ap.add_argument("--flux-anomaly", type=float, default=None, metavar="KG/S",
                    help="figure for a fluid-injection (condition 3) run")
    ap.add_argument("--z-anomaly", type=float, default=None, metavar="Z",
                    help="figure for a non-default injected-salinity run")
    ap.add_argument("--dt-nominal", type=float, default=None, metavar="YR",
                    help="figure for a non-default nominal-dt run")
    ap.add_argument("--dt-min", type=float, default=None, metavar="YR")
    ap.add_argument("--dt-max", type=float, default=None, metavar="YR")
    ap.add_argument("--tf", type=float, default=None, metavar="YR",
                    help="figure for a run with a non-default final time [years]")
    ap.add_argument("--times", type=float, nargs="+", default=[5000.0, 15000.0, 50000.0],
                    metavar="YEARS", help="snapshot instants [years] (default 5/15/50 kyr)")
    args = ap.parse_args()

    tag = case_tag(args.scheme, args.consistent, args.grid_type, args.cell_size,
                   args.q_anomaly, args.z_init, args.dt_nominal, args.dt_min,
                   args.dt_max, args.tf, flux_anomaly=args.flux_anomaly,
                   z_anomaly=args.z_anomaly)
    folder = os.path.join(HERE, f"visualization_{tag}")
    if not os.path.isdir(folder):
        have = sorted(d for d in os.listdir(HERE) if d.startswith("visualization_"))
        raise SystemExit(
            f"no data for this parametrization ({os.path.basename(folder)}); run "
            "`python porepy_2d_solver.py` with the same flags first.\n"
            f"available: {', '.join(have) or 'none'}")
    snaps = _snapshots(folder)
    cmv = _cmap("vlag")

    picked = [min(snaps, key=lambda s: abs(s[0] - tt)) for tt in args.times]
    meshes = [_panel_mesh(path) for _, path in picked]
    f_all = np.concatenate([np.asarray(m.cell_data[args.field], float) for m in meshes])
    clim = (float(f_all.min()), float(f_all.max()))
    if clim[0] == clim[1]:                            # constant field (e.g. s_v = 0)
        clim = (clim[0], clim[0] + 1.0)

    fig, axes = plt.subplots(1, len(args.times), figsize=(4.6 * len(args.times), 4.0),
                             sharey=True)
    axes = [axes] if len(args.times) == 1 else list(axes)
    for k, (ax, (t, path), mesh) in enumerate(zip(axes, picked, meshes)):
        img = _render_panel(mesh, args.field, cmv, clim)
        ax.imshow(img, extent=[-X_HALF / 1e3, X_HALF / 1e3, DEPTH / 1e3, 0.0],
                  aspect="auto", interpolation="nearest")
        # paper-style annotated contours (drawn in matplotlib so clabel can tag them);
        # contours need point values -> convert a copy, the field render stays cellwise
        pmesh = mesh.cell_data_to_point_data()
        dist = pmesh.points[:, 0] / 1.0e3
        depth = (DEPTH - pmesh.points[:, 1]) / 1.0e3
        ct = ax.tricontour(dist, depth, np.asarray(pmesh.point_data["T_C"], dtype=float),
                           levels=T_ISO, colors="firebrick", linewidths=1.4)
        ax.clabel(ct, fmt=lambda v: f"{v:.0f}°C", fontsize=8, inline=True,
                  colors="black")
        cp = ax.tricontour(dist, depth, np.asarray(pmesh.point_data["pressure"],
                                                   dtype=float),
                           levels=P_ISO, colors="royalblue", linewidths=1.4)
        ax.clabel(cp, fmt=lambda v: f"{v:.0f} MPa", fontsize=8, inline=True,
                  colors="black")
        ax.set_xlim(-X_HALF / 1e3, X_HALF / 1e3)
        ax.set_ylim(DEPTH / 1e3, 0.0)                 # depth increases downward
        ax.set_xlabel("Distance (km)")
        if k == 0:
            ax.set_ylabel("Depth (km)")
        ax.text(0.97, 0.05, f"{t / 1000.0:.0f} kyrs", transform=ax.transAxes,
                fontsize=12, va="bottom", ha="right", zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2))
    sm = mcm.ScalarMappable(norm=colors.Normalize(*clim), cmap=cmv)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label(r"Enthalpy (MJ kg$^{-1}$)" if args.field == "enthalpy"
                   else r"Vapor saturation $s_v$ (-)")

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    suffix = "" if args.field == "enthalpy" else f"_{args.field}"
    out = os.path.join(HERE, "figures", f"fig_8_plume_{tag}{suffix}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")
    print("wrote", os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
