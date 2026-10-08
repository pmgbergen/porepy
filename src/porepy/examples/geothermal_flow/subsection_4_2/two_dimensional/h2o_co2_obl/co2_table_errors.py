#!/usr/bin/env python
"""L2 error of the compositional H2O-CO2 p-h table vs the direct ("true") flash, by phase group.

Four figures, each 4 panels z in {0.0, 0.1, 0.25, 0.5}, error map over (h,p) with the L2 norm
(RMS over the reachable domain) and max annotated:
  - saturations   ||S_tab - S_exact||_2            (absolute, 3 phases)
  - densities      sqrt(sum_k S_k (dRho_k/Rho_k)^2)  (saturation-weighted relative)
  - enthalpies     sqrt(sum_k mf_k dH_k^2)  [kJ/kg]  (mass-fraction weighted)
  - temperature   |T_tab - T_exact|  [K]
"truth" = the same SPE2003 + Garcia + heat-of-solution model evaluated directly at fine T-resolution,
so this is the OBL interpolation error. Unreachable (z,h) corners are masked.

Run:  python co2_table_errors.py
"""
from __future__ import annotations
import os
import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator as RGI
from CoolProp.CoolProp import PropsSI

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import build_co2_compositional_table as B
from co2_phz_diagrams import precompute_col, load, exact_fields  # exact_fields: reference flash
from co2_plot_style import paper_cmap

HERE = os.path.dirname(os.path.abspath(__file__))
TK = 273.15
Z_SLICES = (0.0, 0.1, 0.25, 0.5)
NH, NP = 200, 160
SLOTS = ["S_l", "S_h", "S_v", "Rho_l", "Rho_h", "Rho_v", "H_l", "H_h", "H_v", "Temperature"]


def group_error(tab, ex):
    """Per-node error for each phase group from table/exact field dicts."""
    dSl, dSh, dSv = tab["S_l"] - ex["S_l"], tab["S_h"] - ex["S_h"], tab["S_v"] - ex["S_v"]
    e_sat = np.sqrt(dSl**2 + dSh**2 + dSv**2)
    # saturation-weighted relative density error
    e_rho = np.zeros_like(e_sat)
    for s, r in (("S_l", "Rho_l"), ("S_h", "Rho_h"), ("S_v", "Rho_v")):
        w = np.clip(ex[s], 0, 1)
        e_rho = e_rho + w * ((tab[r] - ex[r]) / np.maximum(np.abs(ex[r]), 1e-9))**2
    e_rho = np.sqrt(e_rho)
    # mass-fraction weighted enthalpy error [kJ/kg]
    rho_mix = ex["S_l"] * ex["Rho_l"] + ex["S_h"] * ex["Rho_h"] + ex["S_v"] * ex["Rho_v"]
    e_h = np.zeros_like(e_sat)
    for s, r, h in (("S_l", "Rho_l", "H_l"), ("S_h", "Rho_h", "H_h"), ("S_v", "Rho_v", "H_v")):
        mf = ex[s] * ex[r] / np.maximum(rho_mix, 1e-9)
        e_h = e_h + mf * (tab[h] - ex[h])**2
    e_h = np.sqrt(e_h)
    e_T = np.abs(tab["Temperature"] - ex["Temperature"])
    return dict(saturation=e_sat, density=e_rho, enthalpy=e_h, temperature=e_T)


_UNITS = dict(saturation="[-]", density="(rel.)", enthalpy="[kJ/kg]", temperature="[K]")
_TITLES = dict(saturation="Saturation", density="Density (sat-weighted rel.)",
               enthalpy="Enthalpy (mass-weighted)", temperature="Temperature")


def main():
    (xz, yh, zp), d = load("h2o_co2_xph.vtr")
    off = float(open(os.path.join(HERE, "h2o_co2_offset.txt")).read())
    itp = {k: RGI((xz, yh, zp), d[k], bounds_error=False, fill_value=None) for k in SLOTS}

    h_ax = np.linspace(yh.min(), yh.max(), NH)
    p_ax = np.linspace(zp.min(), zp.max(), NP)
    Tfine = np.linspace(1.0, 60.0, 360)
    HH, PP = np.meshgrid(h_ax, p_ax, indexing="ij")
    cols = [precompute_col(P * 1e6, Tfine) for P in p_ax]

    errs = {g: [] for g in ("saturation", "density", "enthalpy", "temperature")}
    l2 = {g: {} for g in errs}
    for z in Z_SLICES:
        ex = {k: np.full(HH.shape, np.nan) for k in SLOTS}
        reach = np.zeros(HH.shape, bool)
        for j, P in enumerate(p_ax):
            colex, rc = exact_fields(z, h_ax * 1e6, cols[j], off, Tfine)
            for k in SLOTS:
                ex[k][:, j] = colex[k]
            reach[:, j] = rc
        q = np.column_stack([np.full(HH.size, z), HH.ravel(), PP.ravel()])
        tab = {k: itp[k](q).reshape(HH.shape) for k in SLOTS}
        ge = group_error(tab, ex)
        for g in errs:
            e = np.where(reach, ge[g], np.nan)
            errs[g].append(e)
            l2[g][z] = np.sqrt(np.nanmean(e**2))

    for g in errs:
        vmax = max(np.nanpercentile(e, 99.5) for e in errs[g])
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), sharex=True, sharey=True)
        for ax, z, e in zip(axes.ravel(), Z_SLICES, errs[g]):
            im = ax.pcolormesh(HH, PP, e, cmap=paper_cmap(), shading="nearest", vmin=0, vmax=vmax)
            ax.axhline(B.PC / 1e6, color="cyan", ls=":", lw=1)
            ax.set_title(f"$z_{{\\mathrm{{CO_2}}}} = {z:g}$", fontsize=11)
            ax.text(0.03, 0.96, f"L2 = {l2[g][z]:.3g}\nmax = {np.nanmax(e):.3g}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(fc="white", alpha=0.75, ec="none"))
        for ax in axes[1, :]:
            ax.set_xlabel("mixture enthalpy $h$ [MJ/kg]")
        for ax in axes[:, 0]:
            ax.set_ylabel("pressure $p$ [MPa]")
        cb = fig.colorbar(im, ax=axes, location="right", shrink=0.85)
        cb.set_label(f"{_TITLES[g]} error {_UNITS[g]}")
        fig.suptitle(f"{_TITLES[g]} error: compositional p-h table vs true flash "
                     f"(p in [{zp.min():.0f},{zp.max():.0f}] MPa)", fontsize=12, y=0.98)
        out = os.path.join(HERE, f"co2_err_{g}")
        for ext in ("png", "pdf"):
            fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("wrote co2_err_{saturation,density,enthalpy,temperature}.{png,pdf}")
    print(f"{'group':12s} " + "  ".join(f"z={z:<5g}" for z in Z_SLICES) + "   (L2 over reachable domain)")
    for g in ("saturation", "density", "enthalpy", "temperature"):
        print(f"{g:12s} " + "  ".join(f"{l2[g][z]:.3e}" for z in Z_SLICES) + f"   {_UNITS[g]}")


if __name__ == "__main__":
    main()
