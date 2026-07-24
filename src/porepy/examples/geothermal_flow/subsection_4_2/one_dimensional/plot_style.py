"""Shared publication aesthetics for the subsection 4.1 (Weis 1D) figures.

LaTeX (usetex) + serif to match the paper, with a graceful mathtext fallback when no LaTeX
build is on PATH. Provides the colour/marker registry for the three schemes and the two
gravity-density treatments, unit conversions, and a PDF(+PNG) save helper.
"""
from __future__ import annotations

import os
import shutil

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Text width [in] of the paper; a full-width figure spans it, two panels sharing it side by side.
TEXTWIDTH_IN = 6.5

# Okabe-Ito colour-blind-safe triad for the three schemes.
_BLUE, _VERMILLION, _GREEN = "#0072B2", "#D55E00", "#009E73"

# The three schemes carried through the §4.1 comparison, mapped to weis_1d_solver.run kwargs.
#   key -> run kwargs (scheme, weighted_perm) + display label / colour / marker
SCHEMES = {
    "ppu":   dict(scheme="ppu", weighted_perm=False, label="PPU",
                  color=_BLUE, marker="o"),
    "hu":    dict(scheme="hu",  weighted_perm=False, label="HU",
                  color=_VERMILLION, marker="s"),
    "hu_mwp": dict(scheme="hu",  weighted_perm=True,  label=r"HU-$\mathrm{mwp}$",
                  color=_GREEN, marker="^"),
}

# Gravity-term density treatment (fig weis_verification) -> run kwarg + line style.
DENSITY = {
    "averaged": dict(grav_upstream=False, label="averaged", ls="-"),
    "upwinded": dict(grav_upstream=True,  label="upwinded", ls=(0, (4, 2))),
}

# Digitized-reference marker style.
REF_KW = dict(marker="s", ls="none", ms=3.2, mfc="none", mec="0.2", mew=0.6,
              label=r"Weis et al.\ (2014)", zorder=5)

FIELD_LABEL = {
    "T": r"Temperature $[^{\circ}\mathrm{C}]$",
    "p": r"Pressure $[\mathrm{MPa}]$",
    "s_liq": r"Liquid saturation $[-]$",
}
DIST_LABEL = r"Distance $[\mathrm{km}]$"


def apply_style(usetex=True):
    """Apply the publication rcParams. Uses LaTeX if ``usetex`` and a ``latex`` binary is on
    PATH; otherwise falls back to matplotlib's Computer-Modern mathtext."""
    use = bool(usetex) and shutil.which("latex") is not None
    if usetex and not use:
        print("[plot_style] no 'latex' on PATH -> mathtext (Computer Modern) fallback")
    mpl.rcParams.update({
        "text.usetex": use,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # larger + bold, so the labels stay legible after the paper scales the figure down
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
        "mathtext.default": "bf",                      # bold math in the mathtext fallback
        "axes.linewidth": 0.9, "lines.linewidth": 1.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "legend.frameon": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    if use:
        # usetex ignores the weight rcParams -> bold via the preamble: bold text series + bold math.
        mpl.rcParams["text.latex.preamble"] = (
            r"\usepackage{amsmath}\renewcommand{\familydefault}{\bfdefault}\boldmath")


def to_plot_units(res, field):
    """weis_1d_solver.run result -> (distance_km, value in plotted units)."""
    x_km = res["y"] / 1000.0
    val = {"T": res["T"] - 273.15, "p": res["p"] / 1e6, "s_liq": res["s_liq"]}[field]
    return x_km, val


def bottom_legend(fig, handles, labels, ncol, y=-0.02, fontsize=9):
    """A rounded-box legend centred just below the figure (call after ``fig.tight_layout()``; the
    tight save bbox includes it). ``loc='upper center'`` anchors its top so it sits clear beneath
    the axis label."""
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                     columnspacing=1.2, handlelength=1.6, fontsize=fontsize, borderpad=0.5,
                     frameon=True, fancybox=True, framealpha=1.0, edgecolor="0.6")
    leg.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.4")
    return leg


def panel_tag(ax, text, loc=(0.04, 0.93), va="top", ha="left"):
    """Place a bold panel tag, e.g. ``(a)``, in axis coordinates, on a subtle white backing box so
    it stays legible wherever it lands."""
    ax.text(loc[0], loc[1], text, transform=ax.transAxes, fontweight="bold", va=va, ha=ha,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2))


SAVE_PDF = True     # also write a vector PDF next to each PNG (run_workflow toggles this via --pdf)


def savefig(fig, stem, out_dir):
    """Save ``fig`` as a PNG (preview) plus, when ``SAVE_PDF``, a vector PDF (for \\includegraphics)
    under ``out_dir``.

    If a LaTeX render error occurs (usetex on but a package/glyph missing), retry once with
    usetex disabled so a figure is still produced."""
    os.makedirs(out_dir, exist_ok=True)
    exts = ("pdf", "png") if SAVE_PDF else ("png",)
    paths = [os.path.join(out_dir, f"{stem}.{ext}") for ext in exts]
    try:
        for p in paths:
            fig.savefig(p)
    except Exception as exc:  # LaTeX rendering failure -> fall back and retry
        print(f"[plot_style] savefig failed under usetex ({exc}); retrying with mathtext")
        mpl.rcParams["text.usetex"] = False
        for p in paths:
            fig.savefig(p)
    for p in paths:
        print(f"wrote {p}")
