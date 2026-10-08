#!/usr/bin/env python
"""Compositional H2O-CO2 p-h diagrams: phase regions (top) and saturation error (bottom).

Two slices z_CO2 in {0.1, 0.3}:
  (top)    phase regions (analytic fill: aqueous / aq+CO2-liquid / aq+CO2-gas / three-phase band)
  (bottom) ||S_table - S_exact||_2 over (h,p), the OBL saturation interpolation error, where S_exact
           is the direct fine-resolution flash (SPE2003 + Garcia + heat of solution).
Physically-unreachable (z,h) corners are masked (white).

Run:  python co2_phz_diagrams.py
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
from matplotlib.patches import Rectangle  # noqa: E402

import build_co2_compositional_table as B
import co2_ph_slices as S                                   # analytic divider + band + region fills
from co2_plot_style import paper_cmap

HERE = os.path.dirname(os.path.abspath(__file__))
TK = 273.15
Z_SLICES = (0.1, 0.3)
NH, NP = 200, 160                                      # plot grid (finer than the table)
SLOTS = ["S_l", "S_h", "S_v", "Rho_l", "Rho_h", "Rho_v", "H_l", "H_h", "H_v", "Temperature"]


def load(name):
    m = pv.read(os.path.join(HERE, name))
    nx, ny, nz = m.dimensions
    return (np.asarray(m.x), np.asarray(m.y), np.asarray(m.z)), \
           {k: np.asarray(m.point_data[k]).reshape((nx, ny, nz), order="F") for k in m.point_data}


def precompute_col(P, Tfine):
    """z-independent per-pressure data: fine CoolProp arrays, SPE compositions, band endpoints."""
    Tk = Tfine + TK
    Pv = np.full(Tfine.size, P)
    c = dict(Pv=Pv, Tk=Tk,
             rw=PropsSI("D", "P", Pv, "T", np.maximum(Tk, B.TW_MIN), "Water"),
             hw=PropsSI("H", "P", Pv, "T", np.maximum(Tk, B.TW_MIN), "Water"),
             mw=PropsSI("V", "P", Pv, "T", np.maximum(Tk, B.TW_MIN), "Water"),
             rc=PropsSI("D", "P", Pv, "T", Tk, "CO2"),
             hc=PropsSI("H", "P", Pv, "T", Tk, "CO2"),
             mc=PropsSI("V", "P", Pv, "T", Tk, "CO2"))
    c["liq"] = B._co2_liquid(Tk, Pv, c["rc"])
    ca, cc = B.comp_grid(Pv[None, :], Tk[None, :])
    c["ca"], c["cc"] = ca[0], cc[0]
    sat = B._co2_sat(np.array([P]))
    c["sat"] = sat
    Ts = sat["Ts"][0]
    c["Ts"] = Ts
    c["band_ok"] = np.isfinite(Ts) and (Ts >= B.TW_MIN)
    if c["band_ok"]:
        c["caL"], c["ccL"] = B.comp_at(P + B.DP, Ts)
        c["caV"], c["ccV"] = B.comp_at(P - B.DP, Ts)
        c["hw_s"] = PropsSI("H", "P", P, "T", Ts, "Water") - B.HW_REF
        c["hl_s"] = sat["hl"][0] - B.HC_REF
        c["hv_s"] = sat["hv"][0] - B.HC_REF
        c["dhsL"] = float(B.dHsol(np.array(Ts - TK), np.array(True)))
        c["dhsV"] = float(B.dHsol(np.array(Ts - TK), np.array(False)))
    return c


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


# region code -> colour (1 aqueous, 3 a+l, 5 a+g, 7 a+l+g)
_COL = {1: "#cfe3f7", 2: "#2a7f3f", 3: "#8fcf8f", 4: "#d4a017", 5: "#f3b65a", 6: "#cccccc", 7: "#9a9a9a"}
_ROI_H = (0.1, 0.2)                                    # dashed ROI box: h [MJ/kg]
_ROI_P = (4.0, 8.0)                                    #                 p [MPa]


def _roi_box(ax):
    """Dashed gray ROI rectangle p in [4,8] MPa, h in [0.1,0.2] MJ/kg."""
    ax.add_patch(Rectangle((_ROI_H[0], _ROI_P[0]), _ROI_H[1] - _ROI_H[0], _ROI_P[1] - _ROI_P[0],
                           fill=False, edgecolor="0.25", ls="--", lw=1.3, zorder=6))


def _slice_fill(ax, z, off, h_ax, p_ax):
    """Top-row analytic phase-region fill (same construction as co2_ph_slices)."""
    pc_mpa = S.PC / 1e6
    ax.set_facecolor(_COL[1])
    pcv, hcv = S.liqgas_curve(z, off)                       # smooth rho_CO2 = rho_c divider, p >= p_c
    pb, hlo, hhi = S.band_curves(z)
    hlo = (hlo + off) / 1e6 if hlo.size else hlo
    hhi = (hhi + off) / 1e6 if hhi.size else hhi
    if pb.size:                                             # two-phase present
        m = pb < pc_mpa
        gp = np.concatenate([pb[m], pcv])
        g_right = np.concatenate([hlo[m], hcv])
        o_left = np.concatenate([hhi[m], hcv])
        o = np.argsort(gp)
        gp, g_right, o_left = gp[o], g_right[o], o_left[o]
        ax.fill_betweenx(gp, h_ax.min(), g_right, color=_COL[3], zorder=1)
        ax.fill_betweenx(gp, o_left, h_ax.max(), color=_COL[5], zorder=1)
        ax.fill_betweenx(pb, hlo, hhi, color=_COL[7], zorder=3)
        ax.plot(hlo, pb, color="0.3", lw=0.8, zorder=4)
        ax.plot(hhi, pb, color="0.3", lw=0.8, zorder=4)
    if pcv.size:
        ax.plot(hcv, pcv, color="0.12", lw=1.4, zorder=5)
    ax.set_xlim(h_ax.min(), h_ax.max())
    ax.set_ylim(p_ax.min(), p_ax.max())
    ax.axhline(B.PC / 1e6, color="firebrick", ls=":", lw=1)


def main():
    (xz, yh, zp), d = load("h2o_co2_xph.vtr")
    off = float(open(os.path.join(HERE, "h2o_co2_offset.txt")).read())
    Sitp = {k: RGI((xz, yh, zp), d[k], bounds_error=False, fill_value=None)
            for k in ("S_l", "S_h", "S_v")}

    h_ax = np.linspace(yh.min(), yh.max(), NH)                 # MJ/kg -- table range
    p_ax = np.linspace(zp.min(), zp.max(), NP)                 # MPa  -- table range
    Tfine = np.linspace(1.0, 60.0, 360)
    HH, PP = np.meshgrid(h_ax, p_ax, indexing="ij")            # (NH,NP)
    cols = [precompute_col(P * 1e6, Tfine) for P in p_ax]      # p_ax is MPa; flash wants Pa

    panels = []
    emax = 0.0
    for z in Z_SLICES:
        ex = {k: np.full(HH.shape, np.nan) for k in ("S_l", "S_h", "S_v")}
        reach = np.zeros(HH.shape, bool)
        for j, P in enumerate(p_ax):
            colex, rc = exact_fields(z, h_ax * 1e6, cols[j], off, Tfine)   # h in J/kg
            for k in ("S_l", "S_h", "S_v"):
                ex[k][:, j] = colex[k]
            reach[:, j] = rc
        q = np.column_stack([np.full(HH.size, z), HH.ravel(), PP.ravel()])
        tab = {k: Sitp[k](q).reshape(HH.shape) for k in ("S_l", "S_h", "S_v")}
        e_sat = np.sqrt((tab["S_l"] - ex["S_l"])**2 + (tab["S_h"] - ex["S_h"])**2
                        + (tab["S_v"] - ex["S_v"])**2)
        e_sat = np.where(reach, e_sat, np.nan)
        panels.append((z, e_sat))
        emax = max(emax, np.nanpercentile(e_sat, 99.5))

    fig, axes = plt.subplots(2, len(Z_SLICES), figsize=(4.6 * len(Z_SLICES), 8.4),
                             sharex=True, sharey=True)
    for col, (z, e_sat) in enumerate(panels):
        ax0, ax1 = axes[0, col], axes[1, col]
        _slice_fill(ax0, z, off, h_ax, p_ax)
        _roi_box(ax0)
        ax0.set_title(f"$z_{{\\mathrm{{CO_2}}}} = {z:g}$")
        im = ax1.pcolormesh(HH, PP, e_sat, cmap=paper_cmap(), shading="nearest", vmin=0, vmax=emax)
        ax1.axhline(B.PC / 1e6, color="cyan", ls=":", lw=1)
        _roi_box(ax1)
        ax1.set_xlabel("mixture enthalpy $h$ [MJ/kg]")
        l2 = np.sqrt(np.nanmean(e_sat**2))
        ax1.text(0.03, 0.95, f"L2 = {l2:.3g}\nmax = {np.nanmax(e_sat):.3g}",
                 transform=ax1.transAxes, va="top", fontsize=8,
                 bbox=dict(fc="white", alpha=0.7, ec="none"))
    axes[0, 0].set_ylabel("pressure $p$ [MPa]\n(phase regions)")
    axes[1, 0].set_ylabel("pressure $p$ [MPa]\n(saturation error)")

    from matplotlib.patches import Patch
    handles = [Patch(color=_COL[1], label="aqueous"), Patch(color=_COL[3], label="aq + CO$_2$-liq"),
               Patch(color=_COL[5], label="aq + CO$_2$-gas"), Patch(color=_COL[7], label="a+l+g")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.975))
    # horizontal colorbar at the bottom, stealing uniformly from all axes so the 2x2 stays aligned
    fig.colorbar(im, ax=axes, location="bottom", shrink=0.5, aspect=40, pad=0.08,
                 label="saturation error  [-]")
    fig.suptitle("Compositional $p$-$h$ H$_2$O-CO$_2$ diagrams", y=1.02, fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"co2_phz_diagrams.{ext}"), dpi=150, bbox_inches="tight")
    print("wrote co2_phz_diagrams.{png,pdf}")
    for z, e_sat in panels:
        print(f"  z={z:<5g} sat-error  L2={np.sqrt(np.nanmean(e_sat**2)):.3e}  max={np.nanmax(e_sat):.3e}")


if __name__ == "__main__":
    main()
