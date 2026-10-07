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
from co2_table_phase_error import precompute_col, load
from co2_plot_style import paper_cmap

HERE = os.path.dirname(os.path.abspath(__file__))
TK = 273.15
Z_SLICES = (0.0, 0.1, 0.25, 0.5)
NH, NP = 200, 160
SLOTS = ["S_l", "S_h", "S_v", "Rho_l", "Rho_h", "Rho_v", "H_l", "H_h", "H_v", "Temperature"]


def exact_fields(z, h_post, c, off, Tfine):
    """Direct fine-resolution slot fields at (z, h_post[J/kg], P). Units match the table
    (S [-], Rho [kg/m3], H [kJ/kg post-offset], Temperature [K]). Returns (dict, reach)."""
    satb = {k: np.full(Tfine.size, c["sat"][k][0]) for k in c["sat"]}
    ff = B.flash_pt(np.full((1, Tfine.size), z), c["Pv"][None, :], Tfine[None, :], c["rw"][None],
                    c["hw"][None] - B.HW_REF, c["hc"][None] - B.HC_REF, c["rc"][None], c["mw"][None],
                    c["mc"][None], c["liq"][None], c["ca"][None], c["cc"][None], satb["rl"][None],
                    satb["rv"][None], satb["hl"][None], satb["hv"][None], satb["ml"][None], satb["mv"][None])
    Hf = ff["H"][0] + off
    reach = (h_post >= Hf.min()) & (h_post <= Hf.max())
    out = {k: np.full(h_post.shape, np.nan) for k in SLOTS}

    hlo = hhi = None
    if c["band_ok"] and (c["caL"] < z < c["ccL"]) and (c["caV"] < z < c["ccV"]):
        q = np.linspace(0, 1, 64)
        caq = c["caL"] * (1 - q) + c["caV"] * q
        ccq = c["ccL"] * (1 - q) + c["ccV"] * q
        hco2 = c["hl_s"] * (1 - q) + c["hv_s"] * q
        dhsq = c["dhsL"] * (1 - q) + c["dhsV"] * q
        Fq = np.clip((z - caq) / np.maximum(ccq - caq, 1e-12), 0, 1)
        hmix_q = (1 - Fq) * ((1 - caq) * c["hw_s"] + caq * (hco2 + dhsq / B.M_CO2)) \
            + Fq * (ccq * hco2 + (1 - ccq) * c["hw_s"]) + off
        order = np.argsort(hmix_q)
        hlo, hhi = float(hmix_q[order][0]), float(hmix_q[order][-1])

    def gather(mask, branch):
        idx = np.where(branch)[0]
        if idx.size == 0 or not np.any(mask):
            return
        o = np.argsort(Hf[idx])
        hq = h_post[mask]
        Tq = np.interp(hq, Hf[idx][o], (Tfine[idx])[o])
        tk = (Tfine[idx])[o] + TK
        pos = np.clip(np.searchsorted(tk, Tq + TK), 1, tk.size - 1)
        left = (Tq + TK - tk[pos - 1]) <= (tk[pos] - (Tq + TK))
        i0 = np.where(left, idx[o][pos - 1], idx[o][pos])
        for k in ("S_l", "S_h", "S_v", "Rho_l", "Rho_h", "Rho_v"):
            out[k][mask] = ff[k][0, i0]
        for k in ("H_l", "H_h", "H_v"):
            out[k][mask] = (ff[k][0, i0] + off) / 1e3
        out["Temperature"][mask] = Tq + TK

    if hlo is not None:
        Ts = c["Ts"]
        rw_s = PropsSI("D", "P", c["Pv"][0], "T", Ts, "Water")
        rl_s, rv_s = c["sat"]["rl"][0], c["sat"]["rv"][0]
        inb = reach & (h_post >= hlo) & (h_post <= hhi)
        if np.any(inb):
            qgrid = np.linspace(0, 1, 64)
            caq = c["caL"] * (1 - qgrid) + c["caV"] * qgrid
            ccq = c["ccL"] * (1 - qgrid) + c["ccV"] * qgrid
            hco2 = c["hl_s"] * (1 - qgrid) + c["hv_s"] * qgrid
            dhsq = c["dhsL"] * (1 - qgrid) + c["dhsV"] * qgrid
            Fg = np.clip((z - caq) / np.maximum(ccq - caq, 1e-12), 0, 1)
            hmix_q = (1 - Fg) * ((1 - caq) * c["hw_s"] + caq * (hco2 + dhsq / B.M_CO2)) \
                + Fg * (ccq * hco2 + (1 - ccq) * c["hw_s"]) + off
            o = np.argsort(hmix_q)
            qh = np.interp(h_post[inb], hmix_q[o], qgrid[o])
            cai = c["caL"] * (1 - qh) + c["caV"] * qh
            cci = c["ccL"] * (1 - qh) + c["ccV"] * qh
            hco2i = c["hl_s"] * (1 - qh) + c["hv_s"] * qh
            dhsi = c["dhsL"] * (1 - qh) + c["dhsV"] * qh
            Fi = np.clip((z - cai) / np.maximum(cci - cai, 1e-12), 0, 1)
            rho_lq = B.rho_aqueous(c["Pv"][0], Ts - TK, rw_s, cai)
            Va = (1 - Fi) / rho_lq
            Vh = Fi * (1 - qh) / rl_s
            Vg = Fi * qh / rv_s
            Vt = Va + Vh + Vg
            out["S_l"][inb] = Va / Vt
            out["S_h"][inb] = Vh / Vt
            out["S_v"][inb] = Vg / Vt
            out["Rho_l"][inb] = rho_lq
            out["Rho_h"][inb] = rl_s
            out["Rho_v"][inb] = rv_s
            out["H_l"][inb] = ((1 - cai) * c["hw_s"] + cai * (hco2i + dhsi / B.M_CO2) + off) / 1e3
            out["H_h"][inb] = (c["hl_s"] + off) / 1e3
            out["H_v"][inb] = (c["hv_s"] + off) / 1e3
            out["Temperature"][inb] = Ts
        gather(reach & (~inb) & (h_post < hlo), c["liq"])
        gather(reach & (~inb) & (h_post > hhi), ~c["liq"])
    else:
        gather(reach, np.ones(Tfine.size, bool))
    return out, reach


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
