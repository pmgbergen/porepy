#!/usr/bin/env python
"""Preview the COMPOSITIONAL (mutual-solubility) p-h phase diagram at fixed overall z_CO2.

Three slides, z_CO2 in {0.0, 0.1, 0.25}, built the same way the phz table will be: for each
(p, T) the SPE2003 flash gives the equilibrium phase compositions, a lever rule on the overall
CO2 mass fraction z gives the phase split, and the mixture specific enthalpy (water + CO2 +
heat of solution of the dissolved CO2) is the x-axis. Regions: 1 aqueous only, 3 aqueous+CO2-liquid,
5 aqueous+CO2-gas, 7 three-phase a+l+g (the T=Tsat band). Enthalpy reference matches
build_co2_obl_table.py so the h-axis is comparable to h2o_co2_xph.vtr.

Run:  python co2_ph_slices.py
"""
from __future__ import annotations

import os
import numpy as np
from CoolProp.CoolProp import PropsSI

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402

from co2_mutual_solubility import mutual_solubility  # SPE2003 flash

HERE = os.path.dirname(os.path.abspath(__file__))
TK = 273.15
TC = PropsSI("Tcrit", "CO2")          # 304.13 K
PC = PropsSI("pcrit", "CO2")          # 7.377 MPa
RHOC = PropsSI("rhocrit", "CO2")      # 467.6 kg/m3
TW_MIN = 274.15                       # water above freezing
M_H2O, M_CO2 = 18.015e-3, 44.01e-3    # kg/mol
DHSOL = -19.4e3                       # heat of solution of CO2 in water [J/mol] (exothermic, ~25 C)

HW_REF = PropsSI("H", "P", 1.0e5, "T", TW_MIN, "Water")
HC_REF = PropsSI("H", "P", 15.0e6, "T", TW_MIN, "CO2")

Z_SLICES = (0.0, 0.1, 0.25)
TD = np.linspace(1.0, 60.0, 170)      # T [degC]
PM = np.linspace(0.5, 12.0, 150)      # p [MPa]
DP = 50.0                             # Pa offset to pick the liquid/gas branch on the sat line

# domain pressure window (CO2-fault case): ~1 MPa top, ~10.8 MPa fault base
P_TOP, P_BASE = 1.0, 10.8


def w_aq(x):
    """CO2 mass fraction in the aqueous phase from its CO2 mole fraction x."""
    return x * M_CO2 / (x * M_CO2 + (1.0 - x) * M_H2O)


def w_cr(y):
    """CO2 mass fraction in the CO2-rich phase from its H2O mole fraction y."""
    return (1.0 - y) * M_CO2 / ((1.0 - y) * M_CO2 + y * M_H2O)


def _psat(Td):
    p = np.full_like(Td, np.inf)
    m = Td + TK < TC
    p[m] = PropsSI("P", "T", Td[m] + TK, "Q", 0, "CO2")
    return p


def grid_fields():
    """(T,p) grid: pure-component h/rho (vectorized) + equilibrium compositions (looped, z-free)."""
    T2, P2 = np.meshgrid(TD, PM, indexing="ij")          # [degC], [MPa]
    Tk, Pa = T2 + TK, P2 * 1e6
    hw = PropsSI("H", "P", Pa.ravel(), "T", np.maximum(Tk, TW_MIN).ravel(), "Water").reshape(T2.shape)
    hc = PropsSI("H", "P", Pa.ravel(), "T", Tk.ravel(), "CO2").reshape(T2.shape)
    rc = PropsSI("D", "P", Pa.ravel(), "T", Tk.ravel(), "CO2").reshape(T2.shape)
    psat = np.tile(_psat(TD)[:, None], (1, PM.size))
    co2_liq = ((Tk < TC) & (Pa >= psat)) | ((Tk >= TC) & (Pa >= PC) & (rc >= RHOC))

    ca = np.empty(T2.shape)                              # aqueous CO2 mass frac (saturated)
    cc = np.empty(T2.shape)                              # CO2-rich CO2 mass frac (saturated)
    for i in range(TD.size):
        for j in range(PM.size):
            x, y = mutual_solubility(Pa[i, j], Tk[i, j])
            ca[i, j], cc[i, j] = w_aq(np.clip(x, 0, 0.3)), w_cr(np.clip(y, 0, 0.1))
    return T2, P2, (hw - HW_REF), (hc - HC_REF), co2_liq, ca, cc


def mix_ph(z, hw, hc, co2_liq, ca, cc):
    """Mixture enthalpy [J/kg] and phase-region code on the (T,p) grid for overall mass frac z."""
    h_diss = hc + DHSOL / M_CO2                          # dissolved-CO2 partial enthalpy
    single_aq = z <= ca
    Fcr = np.where(cc > ca, (z - ca) / np.maximum(cc - ca, 1e-9), 0.0)
    Fcr = np.clip(Fcr, 0.0, 1.0)
    h_aq = (1 - ca) * hw + ca * h_diss
    h_cr = cc * hc + (1 - cc) * hw
    h = np.where(single_aq, (1 - z) * hw + z * h_diss, (1 - Fcr) * h_aq + Fcr * h_cr)
    reg = np.where(single_aq, 1, np.where(co2_liq, 3, 5)).astype(float)
    return h, reg


def band_curves(z):
    """Three-phase band endpoints h_lo(p), h_hi(p) [J/kg] along p where a+l+g coexist."""
    plo, phi, hlo, hhi = [], [], [], []
    for p in PM:
        Pa = p * 1e6
        if not (Pa < PC - 1.0):
            continue
        Ts = PropsSI("T", "P", Pa, "Q", 0, "CO2")
        if Ts < TW_MIN:                                  # water would freeze -> no band
            continue
        hw_s = PropsSI("H", "P", Pa, "T", Ts, "Water") - HW_REF
        hcL = PropsSI("H", "P", Pa, "Q", 0, "CO2") - HC_REF
        hcV = PropsSI("H", "P", Pa, "Q", 1, "CO2") - HC_REF
        xL, yL = mutual_solubility(Pa + DP, Ts)          # CO2-rich liquid side
        xV, yV = mutual_solubility(Pa - DP, Ts)          # CO2-rich gas side
        caL, ccL = w_aq(np.clip(xL, 0, 0.3)), w_cr(np.clip(yL, 0, 0.1))
        caV, ccV = w_aq(np.clip(xV, 0, 0.3)), w_cr(np.clip(yV, 0, 0.1))
        if not (caL < z < ccL and caV < z < ccV):        # need aqueous + CO2-rich both present
            continue
        FL = (z - caL) / (ccL - caL)
        FV = (z - caV) / (ccV - caV)
        h_d = DHSOL / M_CO2
        hL = (1 - FL) * ((1 - caL) * hw_s + caL * (hcL + h_d)) + FL * (ccL * hcL + (1 - ccL) * hw_s)
        hV = (1 - FV) * ((1 - caV) * hw_s + caV * (hcV + h_d)) + FV * (ccV * hcV + (1 - ccV) * hw_s)
        plo.append(p); phi.append(p); hlo.append(hL); hhi.append(hV)
    return np.array(plo), np.array(hlo), np.array(hhi)


# region code -> colour (1 aqueous, 3 a+l, 5 a+g, 7 a+l+g); others unused
_CODES = [1, 2, 3, 4, 5, 6, 7]
_COL = {1: "#cfe3f7", 2: "#2a7f3f", 3: "#8fcf8f", 4: "#d4a017", 5: "#f3b65a", 6: "#cccccc",
        7: "#9a9a9a"}
_CMAP = ListedColormap([_COL[c] for c in _CODES])
_NORM = BoundaryNorm([c - 0.5 for c in _CODES] + [7.5], _CMAP.N)


def main():
    T2, P2, hw, hc, co2_liq, ca, cc = grid_fields()

    # common enthalpy offset so min h = 0 (Driesner convention; z=0 cold water sets it)
    h0, _ = mix_ph(0.0, hw, hc, co2_liq, ca, cc)
    off = -float(np.min(h0))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    hmax = 0.0
    panels = []
    for z in Z_SLICES:
        h, reg = mix_ph(z, hw, hc, co2_liq, ca, cc)
        pb, hlo, hhi = band_curves(z)
        H = (h + off) / 1e6
        hlo = (hlo + off) / 1e6 if hlo.size else hlo
        hhi = (hhi + off) / 1e6 if hhi.size else hhi
        panels.append((z, H, reg, pb, hlo, hhi))
        hmax = max(hmax, float(H.max()))

    for ax, (z, H, reg, pb, hlo, hhi) in zip(axes, panels):
        ax.pcolormesh(H, P2, reg, cmap=_CMAP, norm=_NORM, shading="gouraud")
        if pb.size:
            ax.fill_betweenx(pb, hlo, hhi, color=_COL[7], zorder=3,
                             label="a+l+g (three-phase)")
            ax.plot(hlo, pb, color="0.3", lw=0.8, zorder=4)
            ax.plot(hhi, pb, color="0.3", lw=0.8, zorder=4)
        cs = ax.contour(H, P2, T2, levels=[5, 10, 20, 30, 40, 50], colors="0.35",
                        linewidths=0.6, linestyles="--")
        ax.clabel(cs, fmt=lambda v: f"{v:g}$^\\circ$C", fontsize=7, inline=True)
        ax.axhline(PC / 1e6, color="firebrick", lw=1.0, ls=":")
        ax.text(0.02 * hmax, PC / 1e6 + 0.1, "$p_c$ (CO$_2$)", color="firebrick", fontsize=8)
        ax.axhspan(P_TOP, P_BASE, color="0.5", alpha=0.06, zorder=0)
        ax.set_xlim(0, hmax)
        ax.set_ylim(PM.min(), PM.max())
        ax.set_xlabel("mixture enthalpy  $h$  [MJ/kg]")
        ax.set_title(f"$z_{{\\mathrm{{CO_2}}}} = {z:g}$")
    axes[0].set_ylabel("pressure  $p$  [MPa]")

    # legend of region colours
    from matplotlib.patches import Patch
    handles = [Patch(color=_COL[1], label="aqueous (1 phase)"),
               Patch(color=_COL[3], label="aqueous + CO$_2$-liquid"),
               Patch(color=_COL[5], label="aqueous + CO$_2$-gas"),
               Patch(color=_COL[7], label="a+l+g (three-phase)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("H$_2$O-CO$_2$ compositional $p$-$h$ phase diagram (Spycher-Pruess 2003 flash, "
                 "mixture $h$ incl. heat of solution)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"co2_ph_slices.{ext}"), dpi=160, bbox_inches="tight")
    print("wrote co2_ph_slices.{png,pdf}")
    print(f"enthalpy offset = {off/1e3:.1f} kJ/kg, h-axis [MJ/kg] in [0, {hmax:.3f}]")
    for z, H, reg, pb, hlo, hhi in panels:
        codes, cnt = np.unique(reg.astype(int), return_counts=True)
        frac = ", ".join(f"{c}:{100*n/reg.size:.0f}%" for c, n in zip(codes, cnt))
        band = f"band p in [{pb.min():.1f},{pb.max():.1f}] MPa" if pb.size else "no band"
        print(f"  z={z:<5g} regions({frac});  {band}")


if __name__ == "__main__":
    main()
