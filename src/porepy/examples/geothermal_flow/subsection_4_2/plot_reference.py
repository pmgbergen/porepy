#!/usr/bin/env python
"""Figures for subsection 4.2 (N-phase gravity segregation through barriers, Bosma 2022
Ex. 6.3 at --nphase 3) from the ``run_reference.py`` output.

The number of phases N is auto-detected from each ``.vtr`` (the ``s_0 .. s_{N-1}`` fields), so the
same script draws the 3-phase Bosma reference and the 4-phase (oil -> mid-heavy + mid-light)
variant. Pass ``--nphase 4`` to read the suffixed ``vtr_n4/`` / ``output_ref_<scheme>_n4/`` dirs
that ``run_reference.py --nphase 4`` writes.

Three figures (all saved to ``figures/``):

  --maps  (per scheme, from step-1 output ``vtr/``): a 3-panel DIVERGING saturation map at
          0 / 78 / 571 days in the style of the paper's Fig. 5. The scalar is the
          density-ranked composite ``sum_k c_k s_k`` with ``c = linspace(-1, +1, N)``
          (BLUE = heaviest phase, RED = lightest, WHITE = mid). At N=3 this is exactly
          ``s_gas - s_water``; at N=4 the two interior phases tint the hue toward their density.

  --grid  (per scheme): a per-phase saturation grid -- rows = snapshot days, columns = phases
          ``s_0 .. s_{N-1}`` (heaviest -> lightest, labelled by density), each a 0..1 ``vlag``
          map with barriers overlaid. This resolves the interior phases that the diverging
          composite collapses (essential for N >= 4).

  --gas   (from step-2 output ``output_ref_<scheme>/``): a comparison of the LIGHTEST phase's
          saturation ``s_{N-1}`` at 78 days -- PPU, HU-BM(ff/mw/mp) -- each panel titled with
          that scheme's total Newton iterations and number of time-step cuts (read from
          ``stats_<scheme>.txt``). At N=3 the lightest phase is the gas.

Usage:
    python plot_reference.py               # all three figures (needs run_reference.py output)
    python plot_reference.py --maps        # only the per-scheme diverging saturation maps
    python plot_reference.py --grid        # only the per-scheme per-phase saturation grids
    python plot_reference.py --gas         # only the lightest-phase comparison
    python plot_reference.py --nphase 4    # read the 4-phase run (vtr_n4/, output_ref_*_n4/)
    python plot_reference.py --maps --days 0 78   # override the snapshot days (for testing)
"""
from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")                       # file output only, no display
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import pyvista as pv                         # noqa: E402

try:
    import seaborn as sns                    # noqa: E402

    def _cmap(name="vlag"):
        return sns.color_palette(name, as_cmap=True)
except ImportError:                          # seaborn optional
    def _cmap(name="vlag"):
        try:
            return plt.get_cmap(name)        # matplotlib built-in
        except ValueError:                   # seaborn-only name (e.g. vlag) -> nearest built-in
            return plt.get_cmap("coolwarm" if name == "vlag" else "viridis")

HERE = os.path.dirname(os.path.abspath(__file__))
LX = LY = 100.0                              # domain [m]
SCHEMES = ("hu", "ppu", "hu-mw", "hu-mp")
# Display names for the HU-BM (Hybrid Upwinding with Background Mobility) family. Keep in sync with
# hamon_2d_solver.SCHEME_LABELS (duplicated here so plotting stays a standalone post-processor with
# no solver/scipy import). Tokens stay the filename keys; HU-BM(...) labels appear in figure text.
SCHEME_LABELS = {
    "hu": "HU-BM(ff)", "hu-mp": "HU-BM(mp)", "hu-mw": "HU-BM(mw)", "ppu": "PPU",
}


def scheme_label(scheme):
    """Human-facing display name for a scheme token (falls back to the token itself)."""
    return SCHEME_LABELS.get(scheme, scheme)


DEFAULT_DAYS = (0, 78, 571)
RHO_HEAVY, RHO_LIGHT = 1500.0, 500.0         # solver's linspace(1500, 500, N) density ladder
_ABC = "abcdefghijklmnop"


# --------------------------------------------------------------------------------------- #
#  Phase bookkeeping (N auto-detected from the VTR fields)
# --------------------------------------------------------------------------------------- #
def _phase_fields(fields):
    """Sorted phase-saturation field names ``s_0 .. s_{N-1}`` present in a VTR (heavy->light)."""
    ks = [k for k in fields if re.fullmatch(r"s_\d+", k)]
    return sorted(ks, key=lambda k: int(k.split("_")[1]))


def _nphase(fields):
    return len(_phase_fields(fields))


def _densities(n):
    """Phase densities matching the solver: linspace(1500, 500, N), heaviest first."""
    return np.linspace(RHO_HEAVY, RHO_LIGHT, n)


def _suffix(nphase):
    """Input-dir suffix matching run_reference.py (empty at N=3, ``_n4`` at N=4, ...)."""
    return "" if int(nphase) == 3 else f"_n{int(nphase)}"


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


def _composite(fields, nx, ny):
    """Diverging density-ranked composite ``sum_k c_k s_k`` with ``c = linspace(-1, +1, N)``.

    Heaviest phase ``s_0`` -> -1 (blue), lightest ``s_{N-1}`` -> +1 (red), interior phases at
    intermediate ranks. Since ``sum_k s_k = 1``, the result stays in ``[-1, +1]``. At N=3,
    ``c = [-1, 0, +1]`` gives exactly ``s_2 - s_0 = s_gas - s_water`` (the original Fig. 5 map).
    The accumulation stays in the fields' native dtype (float32 as stored in the VTR) so the N=3
    result is bit-for-bit identical to the legacy ``f["s_g"] - f["s_w"]`` computation.
    """
    pf = _phase_fields(fields)
    arrs = [np.asarray(fields[k]) for k in pf]
    c = np.linspace(-1.0, 1.0, len(pf)).astype(arrs[0].dtype)   # match dtype -> no float promotion
    acc = c[0] * arrs[0]
    for ck, a in zip(c[1:], arrs[1:]):
        acc = acc + ck * a
    return _image(acc, nx, ny)


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
    if caption:
        ax.text(0.5, -0.16, caption, transform=ax.transAxes, ha="center", va="top", fontsize=10)


def _save(fig, png_path):
    """Save the figure as both a PNG (150 dpi) and a vector PDF; return the PNG path."""
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.splitext(png_path)[0] + ".pdf", bbox_inches="tight")
    return png_path


def _diverging_tick_labels(n):
    """Colorbar labels for the [-1, 0, +1] diverging composite ticks, given N phases."""
    if n == 3:
        return ["water", "oil", "gas"]                       # Bosma names (N=3 unchanged)
    rho = _densities(n)
    return [f"heavy\n$s_0$\n{rho[0]:.0f}", "mid", f"light\n$s_{{{n - 1}}}$\n{rho[-1]:.0f}"]


def _phase_header(n):
    """Column header per phase for the grid figure: name (N=3) or index + density (general)."""
    rho = _densities(n)
    if n == 3:
        names = ["water", "oil", "gas"]
        return [f"{names[k]}\n$s_{{{k}}}$   $\\rho$={rho[k]:.0f}" for k in range(n)]
    return [f"$s_{{{k}}}$   $\\rho$={rho[k]:.0f}" for k in range(n)]


# --------------------------------------------------------------------------------------- #
#  Figure 1: per-scheme 3-panel DIVERGING saturation maps (paper Fig. 5 style)
# --------------------------------------------------------------------------------------- #
def plot_maps(scheme, vtr_dir, days, out_dir, cmap="vlag"):
    cm = _cmap(cmap)
    fig, axes = plt.subplots(1, len(days), figsize=(4.4 * len(days), 4.6))
    if len(days) == 1:
        axes = [axes]
    im, n = None, 3
    for k, (ax, day) in enumerate(zip(axes, days)):
        nx, ny, f = load_vtr(_vtr_path(vtr_dir, scheme, day))
        n = _nphase(f)
        composite = _composite(f, nx, ny)                   # +1 lightest(red) .. -1 heaviest(blue)
        im = ax.imshow(composite, extent=[0, LX, LY, 0], aspect="equal",
                       cmap=cm, vmin=-1.0, vmax=1.0, interpolation="nearest")
        _overlay_barriers(ax, f["barrier"], nx, ny)
        _style_axes(ax, f"({_ABC[k]}) Saturation map at {int(round(day))} days")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(_diverging_tick_labels(n))
    st = parse_stats(os.path.join(vtr_dir, f"stats_{scheme.replace('-', '_')}.txt"))
    stat_line = "     ".join(s for s in (
        None if "total_newton_iters" not in st else f"total iterations: {st['total_newton_iters']}",
        None if "avg_iters_per_step" not in st else f"avg iterations/step: {st['avg_iters_per_step']:.2f}",
        None if "n_time_step_cuts" not in st else f"dt-cuts: {st['n_time_step_cuts']}") if s)
    title_n = "Three-phase" if n == 3 else f"{n}-phase"
    fig.suptitle(f"{title_n} segregation through barriers -- scheme: {scheme_label(scheme)}", y=1.00)
    if stat_line:
        fig.text(0.5, 0.935, stat_line, ha="center", va="top", fontsize=10, color="0.35")
    path = _save(fig, os.path.join(out_dir, f"saturation_maps_{scheme.replace('-', '_')}.png"))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------- #
#  Figure 2: per-scheme per-phase saturation GRID (rows = days, cols = phases s_0..s_{N-1})
# --------------------------------------------------------------------------------------- #
def plot_grid(scheme, vtr_dir, days, out_dir, cmap="vlag"):
    cm = _cmap(cmap)
    # Load every (day) snapshot once; N is taken from the first.
    snaps = [load_vtr(_vtr_path(vtr_dir, scheme, day)) for day in days]
    n = _nphase(snaps[0][2])
    pf = _phase_fields(snaps[0][2])
    headers = _phase_header(n)
    fig, axes = plt.subplots(len(days), n, figsize=(3.1 * n, 3.3 * len(days)), squeeze=False)
    im = None
    for i, (day, (nx, ny, f)) in enumerate(zip(days, snaps)):
        for j, key in enumerate(pf):
            ax = axes[i][j]
            im = ax.imshow(_image(f[key], nx, ny), extent=[0, LX, LY, 0], aspect="equal",
                           cmap=cm, vmin=0.0, vmax=1.0, interpolation="nearest")
            _overlay_barriers(ax, f["barrier"], nx, ny)
            _style_axes(ax, "")
            if i == 0:
                ax.set_title(headers[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{int(round(day))} days", fontsize=11)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, ticks=[0, 0.5, 1])
    cbar.set_label("phase saturation $s_k$")
    title_n = "Three-phase" if n == 3 else f"{n}-phase"
    fig.suptitle(f"{title_n} per-phase saturations (heavy -> light) -- scheme: {scheme_label(scheme)}", y=1.00)
    path = _save(fig, os.path.join(out_dir, f"saturation_grid_{scheme.replace('-', '_')}.png"))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------- #
#  Figure 3: lightest-phase saturation comparison at 78 days (PPU | HU-BM ff | mw | mp)
# --------------------------------------------------------------------------------------- #
def plot_gas_comparison(base, schemes, day, out_dir, cmap="vlag", suffix=""):
    cm = _cmap(cmap)
    order = ["ppu", "hu", "hu-mw", "hu-mp"]                 # left -> right, as requested
    schemes = [s for s in order if s in schemes] or order
    fig, axes = plt.subplots(1, len(schemes), figsize=(4.4 * len(schemes), 4.8))
    if len(schemes) == 1:
        axes = [axes]
    im, n = None, 3
    for ax, scheme in zip(axes, schemes):
        tag = scheme.replace("-", "_")
        out = os.path.join(base, f"output_ref_{tag}{suffix}")
        nx, ny, f = load_vtr(_vtr_path(out, scheme, day))
        n = _nphase(f)
        light = _phase_fields(f)[-1]                        # lightest phase s_{N-1} (= s_g at N=3)
        sg = _image(f[light], nx, ny)
        im = ax.imshow(sg, extent=[0, LX, LY, 0], aspect="equal",
                       cmap=cm, vmin=0.0, vmax=1.0, interpolation="nearest")
        _overlay_barriers(ax, f["barrier"], nx, ny)
        st = parse_stats(os.path.join(out, f"stats_{tag}.txt"))
        it_s = str(st.get("total_newton_iters", "n/a"))
        cut_s = str(st.get("n_time_step_cuts", "n/a"))
        _style_axes(ax, "")
        ax.set_title(f"{scheme_label(scheme)}\niterations: {it_s}   dt-cuts: {cut_s}", fontsize=11)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=[0, 0.5, 1])
    phase_name = "gas" if n == 3 else f"lightest phase $s_{{{n - 1}}}$"
    cbar.set_label(f"{'gas' if n == 3 else 'lightest'} saturation "
                   f"$s_{{{'g' if n == 3 else n - 1}}}$")
    fig.suptitle(f"{phase_name.capitalize()} saturation at {int(round(day))} days", y=0.99)
    path = _save(fig, os.path.join(out_dir, f"gas_comparison_{int(round(day))}d.png"))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot the subsection 4.2 reference results.")
    ap.add_argument("--maps", action="store_true", help="per-scheme diverging saturation maps")
    ap.add_argument("--grid", action="store_true", help="per-scheme per-phase saturation grids")
    ap.add_argument("--gas", action="store_true", help="lightest-phase comparison at 78 days")
    ap.add_argument("--nphase", type=int, default=3,
                    help="number of phases to read (default 3). Selects the input dirs written by "
                         "run_reference.py: N=3 -> vtr/ & output_ref_<scheme>/, N=4 -> vtr_n4/ & "
                         "output_ref_<scheme>_n4/. The actual N is still auto-detected per VTR.")
    ap.add_argument("--vtr-dir", default=None,
                    help="step-1 output dir for --maps/--grid (default: ./vtr[_nN])")
    ap.add_argument("--base", default=HERE,
                    help="dir containing output_ref_<scheme>[_nN]/ for --gas (default: here)")
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    ap.add_argument("--days", type=int, nargs="+", default=list(DEFAULT_DAYS),
                    help="snapshot days for --maps/--grid (default: 0 78 571)")
    ap.add_argument("--gas-day", type=int, default=78, help="day for --gas (default: 78)")
    ap.add_argument("--cmap", default="vlag",
                    help="colormap for all figures --maps/--grid/--gas (default: vlag)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir for the figures (default: ./figures[_nN], so N=4 output "
                         "never clobbers the N=3 figures)")
    args = ap.parse_args(argv)
    if not (args.maps or args.grid or args.gas):
        args.maps = args.grid = args.gas = True            # default: all three

    sfx = _suffix(args.nphase)
    vtr_dir = args.vtr_dir if args.vtr_dir is not None else os.path.join(HERE, f"vtr{sfx}")
    out_dir = args.out_dir if args.out_dir is not None else os.path.join(HERE, f"figures{sfx}")

    if args.maps:
        for scheme in args.schemes:
            print("wrote", os.path.relpath(
                plot_maps(scheme, vtr_dir, args.days, out_dir, cmap=args.cmap), HERE))
    if args.grid:
        for scheme in args.schemes:
            print("wrote", os.path.relpath(
                plot_grid(scheme, vtr_dir, args.days, out_dir, cmap=args.cmap), HERE))
    if args.gas:
        print("wrote", os.path.relpath(
            plot_gas_comparison(args.base, args.schemes, args.gas_day, out_dir,
                                cmap=args.cmap, suffix=sfx), HERE))


if __name__ == "__main__":
    main()
