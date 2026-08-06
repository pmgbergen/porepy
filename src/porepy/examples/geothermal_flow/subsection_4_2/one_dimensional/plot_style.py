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

# Curve encoding (all three figures): the WARM palette is the LEFT-axis quantity (temperature,
# liquid saturation); the COOL palette is the RIGHT-axis quantity (pressure, halite saturation).
# Scheme identity is the DASH pattern -- distinct periods, so where curves coincide the dashes
# interleave instead of hiding each other; the per-scheme shade is a redundant second cue. Validated
# (OKLab dE): warm<->cool family gap ~29 (holds under deuteranopia), within-family adjacent ~10-13
# (the dash carries scheme identity, so that is below the colour-only floor by design).
SCHEMES = {
    "ppu":    dict(scheme="ppu", weighted_perm=False, label="PPU",
                   warm="#F5A623", cool="#5FB0E0", dash=(0, (1, 1.4))),
    "hu":     dict(scheme="hu",  weighted_perm=False, label="HU",
                   warm="#EE6C2C", cool="#3E8FCA", dash=(0, (5.5, 2.2))),
    "hu_mwp": dict(scheme="hu",  weighted_perm=True,  label=r"HU-$\mathrm{mwp}$",
                   warm="#D6352A", cool="#2666AE", dash=(0, (6.5, 1.6, 1.2, 1.6))),
}
# Overlay curves (not weis schemes): same warm/cool idiom, own distinct dash. PorePy takes the
# darkest shade (the reference dark red/blue sits just beyond it); the fig-5 PPU-Weis curve reuses
# PPU's shade -- it IS PPU under Weis's discretisation -- and is told apart by its dash alone.
POREPY = dict(label=r"HU-PorePy", warm="#A81D2E", cool="#143C86", dash=(0, (3.0, 2.6)))
PPU_WEIS = dict(warm=SCHEMES["ppu"]["warm"], cool=SCHEMES["ppu"]["cool"], dash=(0, (7, 1.5, 1.5, 1.5)))
CURVE_LW = 2.0        # scheme / overlay line width (the reference band stays thin, ~1.0)

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
DIST_LABEL = r"\textbf{Distance} $[\mathrm{km}]$"


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
    """weis_1d_solver.run result -> (distance_km, value in plotted units). Only the requested field is
    evaluated, so a result lacking the others still works (e.g. a single-phase porepy overlay with no
    ``s_liq``)."""
    x_km = res["y"] / 1000.0
    if field == "T":
        val = res["T"] - 273.15
    elif field == "p":
        val = res["p"] / 1e6
    else:
        val = res["s_liq"]
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
