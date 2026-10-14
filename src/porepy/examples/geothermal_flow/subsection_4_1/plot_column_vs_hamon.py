#!/usr/bin/env python
"""Two-row column-segregation figure: PorePy vs the hamon reference.

Row 1: PorePy ``hu`` beta-saturation space-time map per density case, with the beta
centre-of-mass track for ``hu`` (solid) and ``hu-mw`` (dashed) overlaid.
Row 2: the cross-code difference ``|s_beta^{PorePy hu} - s_beta^{hamon hu-mp}|`` per
case, both interpolated onto a common time grid (ny=200, kx10, 60 d).

PorePy profiles are read from the .npz that porepy_column_segregation.run_case writes;
the hamon reference (hu-mp) is run in-process (fast numpy). Writes the two-row
deliverable passive_phase_example/porepy_column_segregation_kx10.{png,pdf}.

Run:  python plot_column_vs_hamon.py   # after the .npz profiles have been written
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hamon_2d_solver as H  # noqa: E402  (reference solver)
import column_segregation as CS  # noqa: E402  (hamon column run_case + CASES)

VIS = os.path.join(HERE, "visualization_column")
OUT = os.path.join(HERE, "passive_phase_example")
NY, PERM, TEND, LY = 200, 10.0, 60.0, 100.0
CASES = [
    ("near_heavy", 1250, r"near-heavy  $\rho_\beta=1250$", 1.0),
    ("exact", 1000, r"exact mean  $\rho_\beta=1000$", 0.5),
    ("near_light", 750, r"near-light  $\rho_\beta=750$", 0.0),
]


def load_porepy(tag: str) -> dict:
    d = np.load(os.path.join(VIS, f"{tag}.npz"))
    return dict(
        t=d["t"], y=d["y"], prof=d["prof"], com=d["com"], m_beta=float(d["m_beta"])
    )


def hamon_hu_mp(rho: float) -> dict:
    """hamon hu-mp reference, re-oriented to ascending y (to match the PorePy npz)."""
    H.K_ROCK = 1000.0 * H.MILLI_DARCY * PERM
    _, times, com, prof, y_desc, _ = CS.run_case(
        rho, "hu-mp", ny=NY, mid_frac=1.0 / 3.0, t_end_days=TEND, verbose=False
    )
    return dict(
        t=np.asarray(times),
        y=np.asarray(y_desc)[::-1],
        prof=np.asarray(prof)[:, ::-1],
        com=np.asarray(com),
    )


def main() -> None:
    try:
        import seaborn as sns

        cmap = sns.color_palette("vlag", as_cmap=True)
    except Exception:  # pragma: no cover
        cmap = plt.get_cmap("coolwarm")

    tg = np.linspace(0.0, TEND, 240)  # common time grid for the difference

    def on_grid(t, prof):  # (ntime, ny) -> (len(tg), ny)
        return np.column_stack(
            [np.interp(tg, t, prof[:, j]) for j in range(prof.shape[1])]
        )

    por_hu, por_ff, ham, diffs = {}, {}, {}, {}
    for lab, rho, _, _ in CASES:
        por_hu[lab] = load_porepy(f"hu_{rho}")
        por_ff[lab] = load_porepy(f"hu-mw_{rho}")
        ham[lab] = hamon_hu_mp(rho)
        diffs[lab] = np.abs(
            on_grid(por_hu[lab]["t"], por_hu[lab]["prof"])
            - on_grid(ham[lab]["t"], ham[lab]["prof"])
        )
    dmax = max(float(d.max()) for d in diffs.values()) or 1.0

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.6), constrained_layout=True)
    s_im = d_im = None
    for k, (lab, rho, title, mb) in enumerate(CASES):
        P, F = por_hu[lab], por_ff[lab]
        ax = axes[0][k]
        s_im = ax.pcolormesh(
            P["t"],
            P["y"],
            P["prof"].T,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            shading="gouraud",
            rasterized=True,
        )
        ax.plot(P["t"], P["com"], color="k", lw=1.6, label="HU (mp)")
        ax.plot(F["t"], F["com"], color="#15b01a", lw=1.5, ls="--", label="HU (ff)")
        ax.set_title(f"{title}\n$m_\\beta^{{\\alpha\\gamma}}={mb:.1f}$", fontsize=12)
        ax.set_xlim(0, TEND)
        ax.set_ylim(0, LY)
        ax.tick_params(labelbottom=False)
        if k == 0:
            ax.set_ylabel("height $y$ [m]")
            ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

        axd = axes[1][k]
        d_im = axd.pcolormesh(
            tg,
            ham[lab]["y"],
            diffs[lab].T,
            cmap=cmap,
            vmin=0.0,
            vmax=dmax,
            shading="gouraud",
            rasterized=True,
        )
        axd.set_xlim(0, TEND)
        axd.set_ylim(0, LY)
        axd.set_xlabel("time [days]")
        if k == 0:
            axd.set_ylabel("height $y$ [m]")
        dmx = float(diffs[lab].max())
        axd.text(
            0.97,
            0.04,
            f"max $|\\Delta|$={dmx:.1e}",
            transform=axd.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.75, lw=0),
        )

    cb = fig.colorbar(
        s_im, ax=list(axes[0]), location="right", fraction=0.04, pad=0.02, shrink=0.9
    )
    cb.set_label(r"$\beta$ saturation $s_\beta$ [-]  (PorePy HU)")
    cd = fig.colorbar(
        d_im, ax=list(axes[1]), location="right", fraction=0.04, pad=0.02, shrink=0.9
    )
    cd.set_label(r"$|s_\beta^{\mathrm{PorePy\,HU}}-s_\beta^{\mathrm{hamon\,HU}}|$ [-]")

    fig.suptitle(
        f"PorePy vs hamon column segregation  "
        f"($K$={PERM * 1000:g} mD, $t_f$={TEND:.0f} d, ny={NY}):  "
        f"top = PorePy $s_\\beta$ + CoM,  bottom = cross-code difference",
        fontsize=13,
    )
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"porepy_column_segregation_kx10.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("wrote", os.path.relpath(p, HERE))
    plt.close(fig)


if __name__ == "__main__":
    main()
