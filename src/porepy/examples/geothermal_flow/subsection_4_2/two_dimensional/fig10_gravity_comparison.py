#!/usr/bin/env python
"""Gravity-treatment comparison for the Q=9 W/m^2 two-phase plume (Weis et al. 2014, Fig. 10A).

A 2x2 grid of color maps (columns: TPFA / MPFA; rows: vapor saturation / fluid enthalpy) with the
Fig. 10A contour comparison spanning the full width beneath it, over the truncated simplex domain at
the final (~25 kyr) quasi-steady snapshot:

  row 1. vapor saturation s_v   -- TPFA | MPFA (central dashed axis, mesh wireframe)
  row 2. fluid enthalpy h        -- TPFA | MPFA (shared scale across both cases)
  row 3. Fig. 10A reproduction: fluid pressure (blue), temperature (red) and liquid-saturation (green)
         contours, overlaying BOTH cases -- MPFA solid, TPFA dashed (as the paper does CSMP++ solid /
         HYDROTHERM dashed).

Reads the two run folders directly (no re-simulation):
  visualization_hu_simplex_trunc_q9        -> TPFA  (--grid-type simplex --truncated-domain --q-anomaly 9)
  visualization_hu_mpfa_simplex_trunc_q9   -> MPFA  ( ... --consistent )

Run:  python fig10_gravity_comparison.py   [--out NAME]   [--step N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 12,
})

TPFA_DIR = "visualization_hu_simplex_trunc_q9"       # inconsistent gravity (TPFA)
MPFA_DIR = "visualization_hu_mpfa_simplex_trunc_q9"  # consistent gravity (MPFA, --consistent)

# Absolute-coordinate reference (see porepy_2d_solver.py: coords stay absolute under --truncated-domain)
X_PLUME = 4500.0     # plume / heat-anomaly axis [m]  -> distance = (x - X_PLUME) / 1000
Y_SURFACE = 3000.0   # top (surface) elevation [m]    -> depth = (Y_SURFACE - y) / 1000

# Fig. 10 contour levels (kept identical to the paper; the truncated domain simply omits the
# deepest ones, e.g. 25 MPa / 600 C, which lie below the removed bottom 1 km).
P_LEVELS = [5.0, 15.0, 25.0]          # fluid pressure [MPa]
T_LEVELS = [100.0, 300.0, 600.0]      # temperature [degC]
SL_LEVELS = [0.9]                     # liquid saturation contour (just inside the two-phase field)
H_LEVELS = [1.0]                      # fluid enthalpy contour [MJ/kg]
P_COLOR, T_COLOR, S_COLOR, H_COLOR = "#1f4e9c", "#c0392b", "#2ca25f", "#8e44ad"

# Vapor-saturation colormap: seaborn "vlag" -- exactly what subsection_4_1's maps use
# (plot_reference._cmap("vlag") = sns.color_palette("vlag", as_cmap=True)).
try:
    import seaborn as _sns  # noqa: E402
    SAT_CMAP = _sns.color_palette("vlag", as_cmap=True)
except Exception:                                    # pragma: no cover
    SAT_CMAP = plt.get_cmap("vlag")

X_LIM = (-2.0, 2.0)          # horizontal window [km] (the paper's Fig. 10 crop)


def _matrix_vtu(folder: str, step: int | None) -> str:
    """Path to the 2D-matrix VTU: the requested step, or the last one."""
    files = glob.glob(os.path.join(HERE, folder, "data_2_*.vtu"))
    if not files:
        raise FileNotFoundError(f"no matrix VTU (data_2_*.vtu) in {folder}")
    files.sort(key=lambda f: int(re.search(r"_(\d+)\.vtu$", f).group(1)))
    return files[-1] if step is None else next(
        f for f in files if int(re.search(r"_(\d+)\.vtu$", f).group(1)) == step)


def snapshot_time_label(folder: str, step: int | None) -> str:
    """Simulated time of the plotted snapshot, formatted as in the paper (e.g. '25 kyrs')."""
    try:
        times = json.load(open(os.path.join(HERE, folder, "times.json")))["time"]
        idx = int(re.search(r"_(\d+)\.vtu$", _matrix_vtu(folder, step)).group(1))
        yrs = (times[idx] if idx < len(times) else times[-1]) / (365.0 * 86400.0)
    except Exception:                                    # pragma: no cover
        return ""
    return f"{yrs / 1000.0:g} kyrs" if yrs >= 1000.0 else f"{yrs:g} yrs"


def load(folder: str, step: int | None):
    """Return (Triangulation in distance/depth [km], cell-data dict) for the matrix snapshot."""
    m = meshio.read(_matrix_vtu(folder, step))
    x = (m.points[:, 0] - X_PLUME) / 1000.0            # distance [km], 0 = plume axis
    depth = (Y_SURFACE - m.points[:, 1]) / 1000.0      # depth [km], 0 = surface
    tris = m.cells_dict["triangle"]
    tri = mtri.Triangulation(x, depth, triangles=tris)
    cd = {k: np.concatenate(m.cell_data[k]).astype(float) for k in m.cell_data}
    return tri, cd


def cell_to_point(tri: mtri.Triangulation, cellvals: np.ndarray) -> np.ndarray:
    """Area-agnostic cell -> point average (adjacent-cell mean), for tricontour."""
    npts = len(tri.x)
    acc = np.zeros(npts)
    cnt = np.zeros(npts)
    np.add.at(acc, tri.triangles.ravel(), np.repeat(cellvals, 3))
    np.add.at(cnt, tri.triangles.ravel(), 1.0)
    return acc / np.maximum(cnt, 1.0)


def _style_axes(ax, title=None, ylabel=True, xlabel=True):
    if ylabel:
        ax.set_ylabel("depth [km]")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel("distance from plume axis [km]")
    ax.set_aspect("equal")
    ax.set_xlim(*X_LIM)        # horizontal crop to the paper's window
    ax.set_ylim(2.0, 0.0)      # depth increases downward, surface at top
    if not ylabel:
        ax.tick_params(labelleft=False)
    if not xlabel:
        ax.tick_params(labelbottom=False)


def _map_panel(ax, tri, vals, *, cmap, vmin, vmax, title=None, ylabel=True, xlabel=True):
    """One cell-data color map with the mesh wireframe and central bisecting axis."""
    tpc = ax.tripcolor(tri, facecolors=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="flat", rasterized=True)
    ax.triplot(tri, color="0.2", lw=0.3, alpha=0.55, zorder=3)   # mesh wireframe overlay
    ax.axvline(0.0, color="0.4", ls="--", lw=1.3, zorder=5)      # central bisecting axis
    _style_axes(ax, title, ylabel=ylabel, xlabel=xlabel)
    return tpc


def _clabel(ax, cs, fmt):
    """Inline contour labels on a white background box, so the (other-case, dashed) curves that
    cross a label do not obscure the number."""
    for tx in ax.clabel(cs, fmt=fmt, fontsize=9, inline=True, inline_spacing=5):
        tx.set_bbox(dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def _fig10_contours(ax, tri, cd, ls, label_prefix):
    """Overlay the Fig. 10A field contours (P blue, T red, s_l green) for one case."""
    p = cell_to_point(tri, cd["pressure"])
    t = cell_to_point(tri, cd["T_C"])
    sl = cell_to_point(tri, cd["s_l"])
    h = cell_to_point(tri, cd["enthalpy"])
    kw = dict(linestyles=ls, linewidths=1.3)
    cp = ax.tricontour(tri, p, levels=P_LEVELS, colors=P_COLOR, **kw)
    ct = ax.tricontour(tri, t, levels=T_LEVELS, colors=T_COLOR, **kw)
    cs = ax.tricontour(tri, sl, levels=SL_LEVELS, colors=S_COLOR, **kw)
    ch = ax.tricontour(tri, h, levels=H_LEVELS, colors=H_COLOR, **kw)
    if ls == "solid":                                # label only once (on the MPFA pass)
        _clabel(ax, cp, "%g MPa")
        _clabel(ax, ct, "%g°C")
        _clabel(ax, cs, "%.2f")                      # liquid saturation value (0.90)
        _clabel(ax, ch, "%.1f MJ/kg")                # fluid enthalpy value (1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="fig10_gravity_comparison",
                    help="output basename (written as .png and .pdf)")
    ap.add_argument("--step", type=int, default=None,
                    help="snapshot index (default: last = quasi-steady ~25 kyr)")
    args = ap.parse_args()

    tri_tpfa, cd_tpfa = load(TPFA_DIR, args.step)
    tri_mpfa, cd_mpfa = load(MPFA_DIR, args.step)

    # Shared enthalpy scale across both cases so the two columns are directly comparable.
    h_all = np.concatenate([cd_tpfa["enthalpy"], cd_mpfa["enthalpy"]])
    h_vmin, h_vmax = float(h_all.min()), float(h_all.max())

    # 2x2 color-map grid (cols: TPFA / MPFA, rows: s_v / enthalpy) + full-width contour panel below.
    fig = plt.figure(figsize=(9.0, 11.2), constrained_layout=True)
    axd = fig.subplot_mosaic(
        [["sv_t", "sv_m"],
         ["h_t",  "h_m"],
         ["cont", "cont"]],
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.9]},
    )

    # row 1: vapor saturation (column headers here; no x-labels -- the enthalpy row carries them)
    tpc_s = _map_panel(axd["sv_t"], tri_tpfa, cd_tpfa["s_v"], cmap=SAT_CMAP, vmin=0.0, vmax=1.0,
                       title="TPFA (inconsistent gravity)", ylabel=True, xlabel=False)
    _map_panel(axd["sv_m"], tri_mpfa, cd_mpfa["s_v"], cmap=SAT_CMAP, vmin=0.0, vmax=1.0,
               title="MPFA (consistent gravity)", ylabel=False, xlabel=False)

    # row 2: fluid enthalpy
    tpc_h = _map_panel(axd["h_t"], tri_tpfa, cd_tpfa["enthalpy"], cmap=SAT_CMAP,
                       vmin=h_vmin, vmax=h_vmax, ylabel=True, xlabel=True)
    _map_panel(axd["h_m"], tri_mpfa, cd_mpfa["enthalpy"], cmap=SAT_CMAP,
               vmin=h_vmin, vmax=h_vmax, ylabel=False, xlabel=True)

    cbar_s = fig.colorbar(tpc_s, ax=[axd["sv_t"], axd["sv_m"]], location="bottom",
                          fraction=0.09, pad=0.04, aspect=45, shrink=0.6)
    cbar_s.set_label("vapor saturation $s_v$ [-]")
    cbar_h = fig.colorbar(tpc_h, ax=[axd["h_t"], axd["h_m"]], location="bottom",
                          fraction=0.09, pad=0.04, aspect=45, shrink=0.6)
    cbar_h.set_label("fluid enthalpy $h$ [MJ/kg]")

    # bottom panel: Fig. 10A style, both cases overlaid (MPFA solid, TPFA dashed)
    axc = axd["cont"]
    _fig10_contours(axc, tri_mpfa, cd_mpfa, "solid", "MPFA")
    _fig10_contours(axc, tri_tpfa, cd_tpfa, "dashed", "TPFA")
    _style_axes(axc, "MPFA vs TPFA", ylabel=True, xlabel=True)

    # simulated-time stamp, as in the paper's Fig. 10A ("25 kyrs", top-right corner)
    tlabel = snapshot_time_label(MPFA_DIR, args.step)
    if tlabel:
        axc.text(0.97, 0.95, tlabel, transform=axc.transAxes,
                 ha="right", va="top", fontsize=12)

    # Column-major fill with ncol=4 -> row 1: the four fields; row 2: MPFA / TPFA (line styles),
    # padded with two blank entries so the style row sits under the first two fields.
    _blank = Line2D([], [], color="none", label="")
    handles = [
        Line2D([], [], color=P_COLOR, lw=1.6, label="Pressure"),
        Line2D([], [], color="0.2", lw=1.6, ls="solid", label="MPFA"),
        Line2D([], [], color=T_COLOR, lw=1.6, label="Temperature"),
        Line2D([], [], color="0.2", lw=1.6, ls="dashed", label="TPFA"),
        Line2D([], [], color=S_COLOR, lw=1.6, label="Liquid saturation"),
        _blank,
        Line2D([], [], color=H_COLOR, lw=1.6, label="Enthalpy"),
        _blank,
    ]
    axc.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
               ncol=4, fontsize=12, frameon=True, columnspacing=1.4,
               handlelength=1.6, handletextpad=0.5)

    for ext in ("png", "pdf"):
        path = os.path.join(HERE, f"{args.out}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print("wrote", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
