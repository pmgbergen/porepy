#!/usr/bin/env python
"""Phase diagram + temperature-error map of the compositional H2O-CO2 p-h table.

For each slice z in {0.0, 0.1, 0.25, 0.5}:
  (top)    phase regions sampled from h2o_co2_xph.vtr  (1 aqueous, 3 a+l, 5 a+g, 7 a+l+g)
  (bottom) |T_table - T_exact|  where T_exact is a direct fine-resolution inversion of the SAME
           flash model (SPE2003 + Garcia + heat of solution) -- i.e. the OBL interpolation error.
Physically-unreachable (z,h) corners are masked (white).

Run:  python co2_table_phase_error.py
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
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402

import build_co2_compositional_table as B
from co2_plot_style import paper_cmap

HERE = os.path.dirname(os.path.abspath(__file__))
TK = 273.15
Z_SLICES = (0.0, 0.1, 0.25, 0.5)
NH, NP = 200, 160                                      # plot grid (offset from the 161x141 table)


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


def exact_from_col(z, h_post, c, off, Tfine):
    """Direct T_exact + region for query h-array at fixed (z,P), using a precomputed column c."""
    satb = {k: np.full(Tfine.size, c["sat"][k][0]) for k in c["sat"]}
    ff = B.flash_pt(np.full((1, Tfine.size), z), c["Pv"][None, :], Tfine[None, :], c["rw"][None],
                    c["hw"][None] - B.HW_REF, c["hc"][None] - B.HC_REF, c["rc"][None], c["mw"][None],
                    c["mc"][None], c["liq"][None], c["ca"][None], c["cc"][None], satb["rl"][None],
                    satb["rv"][None], satb["hl"][None], satb["hv"][None], satb["ml"][None], satb["mv"][None])
    Hf = ff["H"][0] + off
    TC = np.full(h_post.shape, np.nan)
    reg = np.full(h_post.shape, np.nan)
    reach = (h_post >= Hf.min()) & (h_post <= Hf.max())

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
        hlo, hhi = float(min(hmix_q[0], hmix_q[-1])), float(max(hmix_q[0], hmix_q[-1]))

    def invert(mask, branch):
        idx = np.where(branch)[0]
        if idx.size == 0:
            return
        order = np.argsort(Hf[idx])
        Tq = np.interp(h_post[mask], Hf[idx][order], (Tfine[idx])[order])
        caq2 = np.interp(Tq, Tfine, c["ca"])
        ccq2 = np.interp(Tq, Tfine, c["cc"])
        liqq = np.interp(Tq, Tfine, c["liq"].astype(float)) > 0.5
        TC[mask] = Tq
        reg[mask] = np.where(z <= caq2, 1.0, np.where(z >= ccq2, np.where(liqq, 2.0, 4.0),
                             np.where(liqq, 3.0, 5.0)))

    if hlo is not None:
        inb = reach & (h_post >= hlo) & (h_post <= hhi)
        TC[inb] = c["Ts"] - TK
        reg[inb] = 7.0
        invert(reach & (~inb) & (h_post < hlo), c["liq"])
        invert(reach & (~inb) & (h_post > hhi), ~c["liq"])
    else:
        invert(reach, np.ones(Tfine.size, bool))
    return TC, reg, reach


_COL = {1: "#cfe3f7", 2: "#2a7f3f", 3: "#8fcf8f", 4: "#d4a017", 5: "#f3b65a", 6: "#cccccc", 7: "#9a9a9a"}
_CODES = [1, 2, 3, 4, 5, 6, 7]
_CMAP = ListedColormap([_COL[c] for c in _CODES])
_NORM = BoundaryNorm([c - 0.5 for c in _CODES] + [7.5], _CMAP.N)


def main():
    (xz, yh, zp), d = load("h2o_co2_xph.vtr")
    off = float(open(os.path.join(HERE, "h2o_co2_offset.txt")).read())
    Ttab = RGI((xz, yh, zp), d["Temperature"], bounds_error=False, fill_value=None)
    Rtab = RGI((xz, yh, zp), d["phase_region"], bounds_error=False, fill_value=None)

    h_ax = np.linspace(yh.min(), yh.max(), NH)                 # MJ/kg (post-offset coordinate)
    p_ax = np.linspace(max(zp.min(), 0.5), min(zp.max(), 12.0), NP)
    Tfine = np.linspace(1.0, 60.0, 360)
    HH, PP = np.meshgrid(h_ax, p_ax, indexing="ij")            # (NH,NP)

    fig, axes = plt.subplots(2, len(Z_SLICES), figsize=(4.0 * len(Z_SLICES), 8.2),
                             sharex=True, sharey=True)
    cols = [precompute_col(P * 1e6, Tfine) for P in p_ax]   # p_ax is MPa; flash wants Pa
    emax = 0.0
    panels = []
    for z in Z_SLICES:
        Texact = np.full(HH.shape, np.nan)
        reach = np.zeros(HH.shape, bool)
        for j, P in enumerate(p_ax):
            Tc, _reg, rc = exact_from_col(z, h_ax * 1e6, cols[j], off, Tfine)   # h in J/kg
            Texact[:, j] = Tc
            reach[:, j] = rc
        q = np.column_stack([np.full(HH.size, z), HH.ravel(), PP.ravel()])
        Ttab_g = Ttab(q).reshape(HH.shape) - TK                 # degC
        Rtab_g = np.round(Rtab(q)).reshape(HH.shape)
        err = np.where(reach, np.abs(Ttab_g - Texact), np.nan)
        Rtab_g = np.where(reach, Rtab_g, np.nan)
        panels.append((z, HH, PP, Rtab_g, err))
        emax = max(emax, np.nanpercentile(err, 99.5))

    for col, (z, HH, PP, R, err) in enumerate(panels):
        ax0, ax1 = axes[0, col], axes[1, col]
        ax0.pcolormesh(HH, PP, R, cmap=_CMAP, norm=_NORM, shading="nearest")
        ax0.axhline(B.PC / 1e6, color="firebrick", ls=":", lw=1)
        ax0.set_title(f"$z_{{\\mathrm{{CO_2}}}} = {z:g}$")
        im = ax1.pcolormesh(HH, PP, err, cmap=paper_cmap(), shading="nearest", vmin=0, vmax=emax)
        ax1.axhline(B.PC / 1e6, color="cyan", ls=":", lw=1)
        ax1.set_xlabel("mixture enthalpy $h$ [MJ/kg]")
        mx = np.nanmax(err)
        ax1.text(0.03, 0.95, f"max {mx:.2f} K\nmean {np.nanmean(err):.3f} K",
                 transform=ax1.transAxes, va="top", fontsize=8,
                 bbox=dict(fc="white", alpha=0.7, ec="none"))
    axes[0, 0].set_ylabel("pressure $p$ [MPa]\n(phase regions)")
    axes[1, 0].set_ylabel("pressure $p$ [MPa]\n(|$T_{tab}-T_{exact}$|)")

    from matplotlib.patches import Patch
    handles = [Patch(color=_COL[1], label="aqueous"), Patch(color=_COL[3], label="aq + CO$_2$-liq"),
               Patch(color=_COL[5], label="aq + CO$_2$-gas"), Patch(color=_COL[7], label="a+l+g"),
               Patch(color=_COL[2], label="CO$_2$-liq"), Patch(color=_COL[4], label="CO$_2$-gas")]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.colorbar(im, ax=axes[1, :], location="right", shrink=0.9,
                 label="temperature error [K]")
    fig.suptitle("Compositional H$_2$O-CO$_2$ $p$-$h$ table: phase diagram (top) and "
                 "OBL temperature error vs direct flash (bottom)", y=0.965, fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"co2_table_phase_error.{ext}"), dpi=150, bbox_inches="tight")
    print("wrote co2_table_phase_error.{png,pdf}")
    for z, _, _, _, err in panels:
        print(f"  z={z:<5g} T-error  max={np.nanmax(err):.3f} K  mean={np.nanmean(err):.4f} K")


if __name__ == "__main__":
    main()
