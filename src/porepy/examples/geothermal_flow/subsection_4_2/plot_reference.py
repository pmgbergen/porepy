#!/usr/bin/env python
"""Figures for subsection 4.2 (three-phase gravity segregation through barriers, Bosma 2022
Ex. 6.3) from the ``run_reference.py`` output.

Two figures (both saved to ``figures/``):

  --maps  (per method, from step-1 output ``vtr/``): a 3-panel saturation map at 0 / 78 / 571
          days in the style of the paper's Fig. 5 -- a diverging map of ``s_gas - s_water``
          (BLUE = heavy/water, RED = light/gas, WHITE = intermediate/oil), barriers overlaid in
          dark grey, depth (0 at top) on the y-axis. One figure per scheme.

  --gas   (from step-2 output ``output_ref_<scheme>/``): a 3-panel comparison of the GAS
          saturation at 78 days -- ppu (left), hu (middle), hu-mw (right) -- each panel titled
          with that scheme's total Newton iterations and number of time-step cuts (read from
          ``stats_<scheme>.txt``).

Usage:
    python plot_reference.py               # both figures (needs run_reference.py output)
    python plot_reference.py --maps        # only the per-scheme saturation maps
    python plot_reference.py --gas         # only the gas-saturation comparison
    python plot_reference.py --maps --days 0 78   # override the snapshot days (for testing)
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")                       # file output only, no display
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import pyvista as pv                        # noqa: E402

try:
    import seaborn as sns                    # noqa: E402

    def _cmap(name="vlag"):
        return sns.color_palette(name, as_cmap=True)
except ImportError:                          # seaborn optional
    def _cmap(name="vlag"):
        try:
            return plt.get_cmap(name)        # matplotlib built-in
        except ValueError:                   # seaborn-only name (e.g. vlag) -> nearest built-in
            return plt.get_cmap("coolwarm")

HERE = os.path.dirname(os.path.abspath(__file__))
LX = LY = 100.0                              # domain [m]
SCHEMES = ("hu", "ppu", "hu-mw")
DEFAULT_DAYS = (0, 78, 571)
_ABC = "abcdefghijklmnop"


# --------------------------------------------------------------------------------------- #
#  I/O helpers
# --------------------------------------------------------------------------------------- #
def _vtr_path(out_dir, scheme, day):
    return os.path.join(out_dir, f"hamon_{scheme.replace('-', '_')}_{int(round(day))}d.vtr")


def load_vtr(path):
    """Return ``(nx, ny, fields)`` from a RectilinearGrid ``.vtr`` (cell-centred fields)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path}\n  (run `python run_reference.py` first to produce the output).")
    m = pv.read(path)
    nx = m.dimensions[0] - 1
    ny = m.dimensions[1] - 1
    fields = {k: np.asarray(m.cell_data[k]) for k in m.cell_data.keys()}
    return nx, ny, fields


def _image(arr1d, nx, ny):
    """Cell array (c = j*nx + i, j=0 bottom) -> 2-D image with row 0 = domain TOP (depth 0)."""
    return np.flipud(np.asarray(arr1d, float).reshape(ny, nx))


_STAT_FIELDS = {"n_accepted_steps": int, "total_newton_iters": int,
                "avg_iters_per_step": float, "max_iters_per_step": int,
                "n_time_step_cuts": int, "wasted_iters_on_cuts": int}


def parse_stats(path):
    """Return a dict of the (deterministic) solver stats from a ``stats_<scheme>.txt``.

    Keys are those of :data:`_STAT_FIELDS` (steps, total/avg/max Newton iterations, dt-cuts,
    wasted iterations). Missing / unparsable fields are simply absent from the dict. Wall time
    is intentionally NOT returned -- it is machine-dependent, not a property of the scheme.
    """
    out = {}
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) >= 2 and p[0] in _STAT_FIELDS:
                try:
                    out[p[0]] = _STAT_FIELDS[p[0]](p[1])
                except ValueError:
                    pass
    return out


# --------------------------------------------------------------------------------------- #
#  Drawing primitives
# --------------------------------------------------------------------------------------- #
def _overlay_barriers(ax, barrier1d, nx, ny):
    B = _image(barrier1d, nx, ny) > 0.5
    rgba = np.zeros(B.shape + (4,))
    rgba[..., :3] = 0.20                     # dark grey
    rgba[..., 3] = np.where(B, 0.95, 0.0)    # opaque on barriers, transparent elsewhere
    ax.imshow(rgba, extent=[0, LX, LY, 0], aspect="equal", interpolation="nearest")


def _style_axes(ax, caption):
    ax.set_xlim(0, LX)
    ax.set_ylim(LY, 0)                       # depth: 0 at top, 100 at bottom
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))
    ax.set_aspect("equal")
    ax.text(0.5, -0.16, caption, transform=ax.transAxes, ha="center", va="top", fontsize=10)


def _save(fig, png_path):
    """Save the figure as both a PNG (150 dpi) and a vector PDF; return the PNG path."""
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.splitext(png_path)[0] + ".pdf", bbox_inches="tight")
    return png_path


# --------------------------------------------------------------------------------------- #
#  Figure 1: per-scheme 3-panel saturation maps (paper Fig. 5 style)
# --------------------------------------------------------------------------------------- #
def plot_maps(scheme, vtr_dir, days, out_dir, cmap="vlag"):
    cm = _cmap(cmap)
    fig, axes = plt.subplots(1, len(days), figsize=(4.4 * len(days), 4.6))
    if len(days) == 1:
        axes = [axes]
    im = None
    for k, (ax, day) in enumerate(zip(axes, days)):
        nx, ny, f = load_vtr(_vtr_path(vtr_dir, scheme, day))
        composite = _image(f["s_g"] - f["s_w"], nx, ny)     # +1 gas(red) .. -1 water(blue)
        im = ax.imshow(composite, extent=[0, LX, LY, 0], aspect="equal",
                       cmap=cm, vmin=-1.0, vmax=1.0, interpolation="nearest")
        _overlay_barriers(ax, f["barrier"], nx, ny)
        _style_axes(ax, f"({_ABC[k]}) Saturation map at {int(round(day))} days")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                        ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["water", "oil", "gas"])
    st = parse_stats(os.path.join(vtr_dir, f"stats_{scheme.replace('-', '_')}.txt"))
    stat_line = "     ".join(s for s in (
        None if "total_newton_iters" not in st else f"total iterations: {st['total_newton_iters']}",
        None if "avg_iters_per_step" not in st else f"avg iterations/step: {st['avg_iters_per_step']:.2f}",
        None if "n_time_step_cuts" not in st else f"dt-cuts: {st['n_time_step_cuts']}") if s)
    fig.suptitle(f"Three-phase segregation through barriers -- scheme: {scheme}", y=1.00)
    if stat_line:
        fig.text(0.5, 0.935, stat_line, ha="center", va="top", fontsize=10, color="0.35")
    path = _save(fig, os.path.join(out_dir, f"saturation_maps_{scheme.replace('-', '_')}.png"))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------- #
#  Figure 2: gas-saturation comparison at 78 days (ppu | hu | hu-mw)
# --------------------------------------------------------------------------------------- #
def plot_gas_comparison(base, schemes, day, out_dir, cmap="vlag"):
    cm = _cmap(cmap)
    order = ["ppu", "hu", "hu-mw"]                          # left -> right, as requested
    schemes = [s for s in order if s in schemes] or order
    fig, axes = plt.subplots(1, len(schemes), figsize=(4.4 * len(schemes), 4.8))
    if len(schemes) == 1:
        axes = [axes]
    im = None
    for ax, scheme in zip(axes, schemes):
        tag = scheme.replace("-", "_")
        out = os.path.join(base, f"output_ref_{tag}")
        nx, ny, f = load_vtr(_vtr_path(out, scheme, day))
        sg = _image(f["s_g"], nx, ny)
        im = ax.imshow(sg, extent=[0, LX, LY, 0], aspect="equal",
                       cmap=cm, vmin=0.0, vmax=1.0, interpolation="nearest")
        _overlay_barriers(ax, f["barrier"], nx, ny)
        st = parse_stats(os.path.join(out, f"stats_{tag}.txt"))
        it_s = str(st.get("total_newton_iters", "n/a"))
        cut_s = str(st.get("n_time_step_cuts", "n/a"))
        _style_axes(ax, "")
        ax.set_title(f"{scheme.upper()}\niterations: {it_s}   dt-cuts: {cut_s}", fontsize=11)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=[0, 0.5, 1])
    cbar.set_label("gas saturation $s_g$")
    fig.suptitle(f"Gas saturation at {int(round(day))} days", y=0.99)
    path = _save(fig, os.path.join(out_dir, f"gas_comparison_{int(round(day))}d.png"))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot the subsection 4.2 reference results.")
    ap.add_argument("--maps", action="store_true", help="per-scheme 3-panel saturation maps")
    ap.add_argument("--gas", action="store_true", help="gas-saturation comparison at 78 days")
    ap.add_argument("--vtr-dir", default=os.path.join(HERE, "vtr"),
                    help="step-1 output dir for --maps (default: ./vtr)")
    ap.add_argument("--base", default=HERE,
                    help="dir containing output_ref_<scheme>/ for --gas (default: here)")
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    ap.add_argument("--days", type=int, nargs="+", default=list(DEFAULT_DAYS),
                    help="snapshot days for --maps (default: 0 78 571)")
    ap.add_argument("--gas-day", type=int, default=78, help="day for --gas (default: 78)")
    ap.add_argument("--cmap", default="vlag",
                    help="matplotlib/seaborn colormap name (default: vlag)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "figures"))
    args = ap.parse_args(argv)
    if not (args.maps or args.gas):
        args.maps = args.gas = True                        # default: both

    if args.maps:
        for scheme in args.schemes:
            print("wrote", os.path.relpath(
                plot_maps(scheme, args.vtr_dir, args.days, args.out_dir, cmap=args.cmap), HERE))
    if args.gas:
        print("wrote", os.path.relpath(
            plot_gas_comparison(args.base, args.schemes, args.gas_day, args.out_dir,
                                cmap=args.cmap), HERE))


if __name__ == "__main__":
    main()
