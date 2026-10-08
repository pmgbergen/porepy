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

Z_SLICES = (0.0, 0.1, 0.25, 0.5)
TD = np.linspace(1.0, 60.0, 280)      # T [degC] (fine, to resolve the critical region smoothly)
PM = np.linspace(4.0, 10.0, 260)      # p [MPa] -- OBL table range
H_WINDOW = (0.075, 0.22)              # MJ/kg -- OBL table enthalpy range
DP = 50.0                             # Pa offset to pick the liquid/gas branch on the sat line

# OBL enthalpy offset (canonical, from the table sidecar) so h aligns with h2o_co2_xph.vtr
_OFF_FILE = os.path.join(HERE, "h2o_co2_offset.txt")
OBL_OFFSET = float(open(_OFF_FILE).read()) if os.path.exists(_OFF_FILE) else None


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


def liqgas_curve(z, off):
    """Supercritical CO2-liquid|CO2-gas divider h*(p) [MJ/kg], p >= p_c, as the analytic
    rho_CO2 = rho_c locus. Smooth 1-D curve -- replaces contouring the discrete region label
    (a step field, which can only staircase between grid nodes). Below p_c the green/orange
    regions are separated by the three-phase band, not by a single line."""
    pp, hh = [], []
    h_d = DHSOL / M_CO2
    for p in np.linspace(PC / 1e6, PM.max(), 200):
        Pa = p * 1e6
        try:
            Tk = PropsSI("T", "P", Pa, "D", RHOC, "CO2")     # T where rho_CO2 = rho_c (unique p>=p_c)
        except ValueError:
            continue
        if not (TD.min() + TK <= Tk <= TD.max() + TK):
            continue
        hw = PropsSI("H", "P", Pa, "T", max(Tk, TW_MIN), "Water") - HW_REF
        hc = PropsSI("H", "P", Pa, "T", Tk, "CO2") - HC_REF
        x, y = mutual_solubility(Pa, Tk)
        ca, cc = w_aq(np.clip(x, 0, 0.3)), w_cr(np.clip(y, 0, 0.1))
        if z <= ca:                                          # no CO2-rich phase -> no divider
            continue
        Fcr = np.clip((z - ca) / max(cc - ca, 1e-9), 0.0, 1.0)
        h = (1 - Fcr) * ((1 - ca) * hw + ca * (hc + h_d)) + Fcr * (cc * hc + (1 - cc) * hw)
        pp.append(p); hh.append((h + off) / 1e6)
    return np.array(pp), np.array(hh)


# region code -> colour (1 aqueous, 3 a+l, 5 a+g, 7 a+l+g); others unused
_CODES = [1, 2, 3, 4, 5, 6, 7]
_COL = {1: "#cfe3f7", 2: "#2a7f3f", 3: "#8fcf8f", 4: "#d4a017", 5: "#f3b65a", 6: "#cccccc",
        7: "#9a9a9a"}
_CMAP = ListedColormap([_COL[c] for c in _CODES])
_NORM = BoundaryNorm([c - 0.5 for c in _CODES] + [7.5], _CMAP.N)


def main():
    T2, P2, hw, hc, co2_liq, ca, cc = grid_fields()

    # enthalpy offset: the OBL table's canonical value (so h matches h2o_co2_xph.vtr); fall back
    # to the self-consistent min-h=0 shift only if the sidecar is absent.
    if OBL_OFFSET is not None:
        off = OBL_OFFSET
    else:
        h0, _ = mix_ph(0.0, hw, hc, co2_liq, ca, cc)
        off = -float(np.min(h0))

    fig, axes = plt.subplots(1, len(Z_SLICES), figsize=(5.0 * len(Z_SLICES), 5.2), sharey=True)
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

    pc_mpa = PC / 1e6
    for ax, (z, H, reg, pb, hlo, hhi) in zip(axes, panels):
        # analytic region fill -- NOT contourf of the discrete 1/3/5 label (that staircases at the
        # divider). aqueous base; green = aq+CO2-liquid, left of the band (below p_c) / left of the
        # rho=rho_c divider (above p_c); orange = aq+CO2-gas, to the right; grey = three-phase band.
        ax.set_facecolor(_COL[1])
        pcv, hcv = liqgas_curve(z, off)                      # smooth rho_CO2 = rho_c divider, p >= p_c
        if np.any(reg > 2) and pb.size:                      # two-phase present (skip pure-water z=0)
            m = pb < pc_mpa
            gp = np.concatenate([pb[m], pcv])                # pressures (increasing after sort)
            g_right = np.concatenate([hlo[m], hcv])          # green | band/divider
            o_left = np.concatenate([hhi[m], hcv])           # band/divider | orange
            o = np.argsort(gp)
            gp, g_right, o_left = gp[o], g_right[o], o_left[o]
            ax.fill_betweenx(gp, H_WINDOW[0], g_right, color=_COL[3], zorder=1)
            ax.fill_betweenx(gp, o_left, H_WINDOW[1], color=_COL[5], zorder=1)
        if pb.size:
            ax.fill_betweenx(pb, hlo, hhi, color=_COL[7], zorder=3,
                             label="a+l+g (three-phase)")
            ax.plot(hlo, pb, color="0.3", lw=0.8, zorder=4)
            ax.plot(hhi, pb, color="0.3", lw=0.8, zorder=4)
        if pcv.size:
            ax.plot(hcv, pcv, color="0.12", lw=1.4, zorder=5)
        cs = ax.contour(H, P2, T2, levels=[5, 10, 20, 30, 40, 50], colors="0.35",
                        linewidths=0.6, linestyles="--")
        ax.clabel(cs, fmt=lambda v: f"{v:g}$^\\circ$C", fontsize=7, inline=True)
        ax.axhline(PC / 1e6, color="firebrick", lw=1.0, ls=":")
        ax.text(H_WINDOW[0] + 0.004, PC / 1e6 + 0.1, "$p_c$ (CO$_2$)", color="firebrick", fontsize=8)
        ax.set_xlim(*H_WINDOW)
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
