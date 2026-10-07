#!/usr/bin/env python
"""Phase-diagram evaluation for the proposed CO2-leakage-along-a-fault case (subsection 4.2).

Immiscible two-component H2O-CO2 model, phases a (aqueous liquid = pure water), l (CO2-rich
liquid), g (CO2-rich gas). Water stays aqueous liquid over the whole range; the CO2 component
is liquid / gas / supercritical per its own equation of state. Properties from CoolProp
(Span-Wagner CO2, IAPWS-95 water).

Shows the behavior in the two OBL table spaces -- p-T-z and p-h-z -- over the fault's initial
p-T path (hydrostatic from 1 MPa at 100 m depth, geothermal 30 C/km from 10 C), and the three
phase densities, flagging: CO2 boiling depth, the liquid-CO2 density crossing the aqueous-gas
midpoint (the density-weight switch of the midpoint background rule), and critical coalescence.

Run:  python co2_phase_diagram.py
"""
from __future__ import annotations

import os
import numpy as np
from CoolProp.CoolProp import PropsSI

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from co2_plot_style import paper_cmap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.80665
TK = 273.15

TC_CO2 = PropsSI("Tcrit", "CO2")          # 304.13 K
PC_CO2 = PropsSI("pcrit", "CO2")          # 7.377 MPa
TTRIP = PropsSI("Ttriple", "CO2")         # 216.59 K (-56.6 C)

# ---- domain / initial-state definition --------------------------------------------------
D_TOP, D_BOT = 100.0, 1100.0              # depth range [m]
P_TOP = 1.0e6                             # top pressure [Pa]
T_TOP_C = 10.0                            # top temperature [C]
GEOTHERM = 30.0 / 1000.0                  # geothermal gradient [C/m]


def fault_path(n=401):
    """Initial (depth, p, T) along the vertical fault: hydrostatic water column + geothermal T."""
    d = np.linspace(D_TOP, D_BOT, n)
    T = T_TOP_C + GEOTHERM * (d - D_TOP) + TK          # [K]
    p = np.empty(n)
    p[0] = P_TOP
    for i in range(1, n):
        rho_w = PropsSI("D", "P", p[i - 1], "T", 0.5 * (T[i - 1] + T[i]), "Water")
        p[i] = p[i - 1] + rho_w * G * (d[i] - d[i - 1])
    return d, p, T


def co2_phase(p, T):
    """'liquid' / 'gas' / 'supercritical' for pure CO2 at (p,T)."""
    if T >= TC_CO2 and p >= PC_CO2:
        return "supercritical"
    if T >= TC_CO2:
        return "gas"
    return "liquid" if p >= PropsSI("P", "T", T, "Q", 0, "CO2") else "gas"


def main():
    d, p, T = fault_path()
    Tc = T - TK

    # CO2 saturation curve (l-g) from 1 C (water stays unfrozen) up to critical
    Ts = np.linspace(1.0 + TK, TC_CO2 - 1e-4, 300)
    psat = np.array([PropsSI("P", "T", t, "Q", 0, "CO2") for t in Ts])

    # boiling crossing: where the fault path meets the CO2 saturation line (T < Tc)
    below = T < TC_CO2
    psat_path = np.full_like(p, np.nan)
    psat_path[below] = [PropsSI("P", "T", t, "Q", 0, "CO2") for t in T[below]]
    diff = p - psat_path
    kx = np.where(np.diff(np.sign(diff[below])) != 0)[0]
    d_boil = p_boil = T_boil = None
    if len(kx):
        idx = np.where(below)[0][kx[0]]
        d_boil, p_boil, T_boil = d[idx], p[idx], T[idx]

    # three saturated-phase densities along the CO2 saturation line (vs T) + aqueous-gas midpoint
    rho_a = np.array([PropsSI("D", "P", pp, "T", tt, "Water") for pp, tt in zip(psat, Ts)])
    rho_l = np.array([PropsSI("D", "T", tt, "Q", 0, "CO2") for tt in Ts])   # sat liquid CO2
    rho_g = np.array([PropsSI("D", "T", tt, "Q", 1, "CO2") for tt in Ts])   # sat vapor  CO2
    mid = 0.5 * (rho_a + rho_g)
    cx = np.where(np.diff(np.sign(rho_l - mid)) != 0)[0]
    T_cross = Ts[cx[0]] if len(cx) else None

    # common enthalpy reference (1 MPa, 10 C) so water and CO2 enthalpies combine meaningfully
    H_W_REF = PropsSI("H", "P", 1.0e6, "T", 283.15, "Water")
    H_C_REF = PropsSI("H", "P", 1.0e6, "T", 283.15, "CO2")

    # injection state at the fault base
    p_inj, T_inj = p[-1], T[-1]
    rho_inj = PropsSI("D", "P", p_inj, "T", T_inj, "CO2")
    h_inj = PropsSI("H", "P", p_inj, "T", T_inj, "CO2") - H_C_REF

    # ---------------------------------------------------------------- figure
    fig, (axpt, axph, axr) = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # (A) p-T-z phase diagram
    axpt.plot(Ts - TK, psat / 1e6, "k-", lw=2, label="CO$_2$ saturation (l--g)")
    axpt.plot(TC_CO2 - TK, PC_CO2 / 1e6, "ko", ms=7)
    axpt.annotate("critical\n%.1f$\\,^\\circ$C, %.2f MPa" % (TC_CO2 - TK, PC_CO2 / 1e6),
                  (TC_CO2 - TK, PC_CO2 / 1e6), textcoords="offset points", xytext=(-6, 8),
                  fontsize=8, ha="right")
    axpt.axhline(PC_CO2 / 1e6, color="0.7", ls=":", lw=0.8)
    axpt.axvline(TC_CO2 - TK, color="0.7", ls=":", lw=0.8)
    sc = axpt.scatter(Tc, p / 1e6, c=d, cmap=paper_cmap(), s=12, zorder=5)
    cb = fig.colorbar(sc, ax=axpt, pad=0.02)
    cb.set_label("depth [m]")
    axpt.plot(T_inj - TK, p_inj / 1e6, "r*", ms=15, zorder=6)
    axpt.annotate("injection (CO$_2$)\nsupercritical", (T_inj - TK, p_inj / 1e6),
                  textcoords="offset points", xytext=(-8, -4), fontsize=8, ha="right", color="r")
    if d_boil is not None:
        axpt.plot(T_boil - TK, p_boil / 1e6, "o", mfc="none", mec="darkorange", mew=2, ms=11, zorder=6)
        axpt.annotate("boiling\n%.0f m" % d_boil, (T_boil - TK, p_boil / 1e6),
                      textcoords="offset points", xytext=(8, 2), fontsize=8, color="darkorange")
    axpt.text(15, 10.5, "liquid CO$_2$", fontsize=9, color="#1f4e9c")
    axpt.text(5, 2.5, "gas CO$_2$", fontsize=9, color="#b8860b")
    axpt.text(33.0, 8.1, "supercritical", fontsize=9, color="#6a3d9a", rotation=12)
    axpt.set(xlabel="temperature [$^\\circ$C]", ylabel="pressure [MPa]",
             title="(A)  p--T--z  (phase type is z-independent)")
    axpt.set_xlim(0, 45)
    axpt.set_ylim(0, 12)
    axpt.legend(fontsize=8, loc="lower right")

    # (B) p-h-z MIXTURE phase diagram: the a+l+g three-phase region is an AREA in enthalpy.
    #     Mixture enthalpy h = (1-z) h_H2O + z h_CO2, both referenced to 1 MPa, 10 C.
    #     At p < Pc the CO2 boils at T=Tsat(p); over that T the CO2 enthalpy runs from its
    #     saturated-liquid to saturated-vapor value, so the MIXTURE enthalpy spans a band:
    #       h in [(1-z)h_w + z h_l^CO2 ,  (1-z)h_w + z h_g^CO2]  -> three phases a+l+g.
    # three-phase band lives only where the aqueous phase is liquid: CO2 must boil at T>=domain
    # minimum (10 C), i.e. p >= Psat_CO2(10 C) ~ 4.5 MPa, up to the CO2 critical pressure.
    p_lo3 = PropsSI("P", "T", T_TOP_C + TK, "Q", 0, "CO2")
    pb = np.linspace(p_lo3, PC_CO2 - 2e3, 220)

    def band(z):
        hmn, hmx = np.empty_like(pb), np.empty_like(pb)
        for i, q in enumerate(pb):
            tsat = PropsSI("T", "P", q, "Q", 0, "CO2")
            hw = PropsSI("H", "P", q, "T", tsat, "Water") - H_W_REF
            hmn[i] = (1 - z) * hw + z * (PropsSI("H", "P", q, "Q", 0, "CO2") - H_C_REF)
            hmx[i] = (1 - z) * hw + z * (PropsSI("H", "P", q, "Q", 1, "CO2") - H_C_REF)
        return hmn / 1e3, hmx / 1e3

    z_ref = 0.5
    h0, h1 = band(z_ref)
    axph.fill_betweenx(pb / 1e6, h0, h1, color="#c994c7", alpha=0.75,
                       label=r"$a+l+g$ (three-phase), $z=%.2f$" % z_ref)
    axph.plot(h0, pb / 1e6, "k-", lw=1.5)
    axph.plot(h1, pb / 1e6, "k-", lw=1.5)
    for zz, ls in ((0.25, ":"), (0.75, ":")):            # band widens with z
        a0, a1 = band(zz)
        axph.plot(a0, pb / 1e6, color="0.45", lw=0.8, ls=ls)
        axph.plot(a1, pb / 1e6, color="0.45", lw=0.8, ls=ls)
    hwc = PropsSI("H", "P", PC_CO2, "T", TC_CO2, "Water") - H_W_REF   # critical: band closes
    hcc = PropsSI("H", "P", PC_CO2, "T", TC_CO2, "CO2") - H_C_REF
    axph.plot(((1 - z_ref) * hwc + z_ref * hcc) / 1e3, PC_CO2 / 1e6, "ko", ms=7)
    axph.annotate("critical\n(band closes)", (((1 - z_ref) * hwc + z_ref * hcc) / 1e3, PC_CO2 / 1e6),
                  textcoords="offset points", xytext=(6, -2), fontsize=8)
    if p_boil is not None:
        axph.axhline(p_boil / 1e6, color="darkorange", ls=":", lw=1.0)
        axph.annotate("boiling p", (axph.get_xlim()[0], p_boil / 1e6), fontsize=8,
                      color="darkorange", va="bottom")
    axph.text(h0.min() - 25, 3.0, "$a+l$\n(2-phase)", fontsize=9, ha="right", color="#1f4e9c")
    axph.text(h1.max() + 10, 3.0, "$a+g$\n(2-phase)", fontsize=9, ha="left", color="#b8860b")
    axph.set(xlabel="mixture enthalpy $h-h_{\\mathrm{ref}}$ [kJ/kg]", ylabel="pressure [MPa]",
             title="(B)  p--h--z  (a+l+g three-phase region is an AREA)")
    axph.set_ylim(0, 12)
    axph.legend(fontsize=8, loc="upper right")

    # (C) phase densities along the CO2 saturation line + aqueous-gas midpoint crossover
    axr.plot(Ts - TK, rho_a, color="#1f4e9c", lw=2, label="aqueous (H$_2$O)")
    axr.plot(Ts - TK, rho_l, color="#2a7f3f", lw=2, label="liquid CO$_2$")
    axr.plot(Ts - TK, rho_g, color="#b8860b", lw=2, label="gas CO$_2$")
    axr.plot(Ts - TK, mid, color="0.4", lw=1.5, ls="--", label=r"$\frac{1}{2}(\rho_a+\rho_g)$")
    if T_cross is not None:
        axr.axvline(T_cross - TK, color="red", ls=":", lw=1.2)
        axr.annotate("liquid CO$_2$ crosses\nthe a--g midpoint\n(%.1f$^\\circ$C)" % (T_cross - TK),
                     (T_cross - TK, 0.5 * (rho_a[cx[0]] + rho_g[cx[0]])),
                     textcoords="offset points", xytext=(-90, 10), fontsize=8, color="red")
    if T_boil is not None:
        axr.axvline(T_boil - TK, color="darkorange", ls=":", lw=1.2)
    axr.set(xlabel="temperature [$^\\circ$C]  (along CO$_2$ saturation line)",
            ylabel="density [kg/m$^3$]",
            title="(C)  three-phase densities + midpoint switch")
    axr.set_xlim(Ts[0] - TK, TC_CO2 - TK + 0.5)
    axr.legend(fontsize=8, loc="center left")

    fig.suptitle("CO$_2$ leakage along a fault — phase-diagram evaluation (H$_2$O/CO$_2$ immiscible; "
                 "CoolProp)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        path = os.path.join(HERE, f"co2_phase_diagram.{ext}")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print("wrote", os.path.relpath(path, HERE))

    # ---------------------------------------------------------------- key numbers
    print("\n=== key states ===")
    print(f"top     : {D_TOP:.0f} m   p={P_TOP/1e6:.2f} MPa  T={T_TOP_C:.0f} C")
    print(f"base    : {D_BOT:.0f} m   p={p[-1]/1e6:.2f} MPa  T={T[-1]-TK:.0f} C   "
          f"CO2={co2_phase(p[-1],T[-1])}  rho={rho_inj:.0f} kg/m3  h={h_inj/1e3:.1f} kJ/kg")
    if d_boil is not None:
        print(f"boiling : {d_boil:.0f} m   p={p_boil/1e6:.2f} MPa  T={T_boil-TK:.1f} C")
    if T_cross is not None:
        d_cross = (T_cross - TK - T_TOP_C) / GEOTHERM + D_TOP
        print(f"rho mid : T={T_cross-TK:.1f} C  (~{d_cross:.0f} m)   "
              f"rho_l={rho_l[cx[0]]:.0f}  mid={mid[cx[0]]:.0f} kg/m3")
    print(f"critical: {TC_CO2-TK:.2f} C / {PC_CO2/1e6:.3f} MPa  "
          f"(~{(TC_CO2-TK-T_TOP_C)/GEOTHERM+D_TOP:.0f} m along the path)")


if __name__ == "__main__":
    main()
