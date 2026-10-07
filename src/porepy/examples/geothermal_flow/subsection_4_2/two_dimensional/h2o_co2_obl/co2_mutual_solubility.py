#!/usr/bin/env python
"""True H2O-CO2 mutual solubility (Spycher, Pruess & Ennis-King 2003) for the CO2-fault case.

Replaces the immiscible assumption: CO2 dissolves in the aqueous phase and H2O in the CO2-rich
phase, so the partial compositions x_CO2(aqueous) and y_H2O(CO2-rich) VARY with p-T. This script
evaluates the mutual solubilities and the genuine three-phase (aqueous + CO2-liquid + CO2-gas)
locus -- the step before rebuilding the OBL table with real compositions.

SPE 2003 (Geochim. Cosmochim. Acta 67:3015): a modified Redlich-Kwong EOS for the CO2-rich phase
+ equilibrium constants + Poynting corrections; valid ~12-100 C, up to ~600 bar.

Run:  python co2_mutual_solubility.py
"""
from __future__ import annotations

import os
import numpy as np
from CoolProp.CoolProp import PropsSI

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TK = 273.15
R = 83.14472                      # cm^3 bar / (mol K)
TC = PropsSI("Tcrit", "CO2")
PC = PropsSI("pcrit", "CO2")
M_H2O, M_CO2 = 18.015e-3, 44.01e-3   # kg/mol

# SPE 2003 parameters
B_CO2, B_H2O = 27.80, 18.18       # cm^3/mol
A_CO2H2O = 7.89e7                 # bar cm^6 K^0.5 / mol^2
VBAR_H2O, VBAR_CO2 = 18.1, 32.6   # cm^3/mol (avg partial molar volumes)
P0 = 1.0                          # reference pressure [bar]


def _a_co2(T):
    return 7.54e7 - 4.13e4 * T     # bar cm^6 K^0.5 / mol^2


def _rk_volume(P, T, yco2, liquid):
    """Molar volume [cm^3/mol] of the CO2-rich phase from the RK EOS (liquid or vapor root)."""
    yh2o = 1.0 - yco2
    a = yco2**2 * _a_co2(T) + 2 * yco2 * yh2o * A_CO2H2O             # a_H2O = 0
    b = yco2 * B_CO2 + yh2o * B_H2O
    A = a * P / (R**2 * T**2.5)
    B = b * P / (R * T)
    roots = np.roots([1.0, -1.0, A - B - B**2, -A * B])
    Z = roots[np.isreal(roots)].real
    Z = Z[Z > B]
    if Z.size == 0:
        Z = np.array([max(roots.real)])
    z = Z.min() if liquid else Z.max()
    return z * R * T / P, a, b


def _ln_phi(P, T, V, yco2, a, b):
    """ln fugacity coefficients (CO2, H2O) in the CO2-rich phase (SPE 2003 eq. 7)."""
    yh2o = 1.0 - yco2
    out = {}
    for k, bk, aik in (("co2", B_CO2, yco2 * _a_co2(T) + yh2o * A_CO2H2O),
                       ("h2o", B_H2O, yco2 * A_CO2H2O + yh2o * 0.0)):
        ln = (np.log(V / (V - b)) + bk / (V - b)
              - 2.0 * aik / (R * T**1.5 * b) * np.log((V + b) / V)
              + a * bk / (R * T**1.5 * b**2) * (np.log((V + b) / V) - b / (V + b))
              - np.log(P * V / (R * T)))
        out[k] = ln
    return out["co2"], out["h2o"]


def _K0(Tc_deg, liquid):
    logKH2O = -2.209 + 3.097e-2 * Tc_deg - 1.098e-4 * Tc_deg**2 + 2.048e-7 * Tc_deg**3
    if liquid:
        logKCO2 = 1.169 + 1.368e-2 * Tc_deg - 5.380e-5 * Tc_deg**2
    else:
        logKCO2 = 1.189 + 1.304e-2 * Tc_deg - 5.446e-5 * Tc_deg**2
    return 10.0**logKH2O, 10.0**logKCO2


def mutual_solubility(P_pa, T_k, n_iter=25):
    """Return (x_CO2 in aqueous [mole frac], y_H2O in CO2-rich [mole frac]) at (P[Pa], T[K])."""
    P = P_pa / 1e5                                   # bar
    Tdeg = T_k - TK
    # CO2-rich phase is liquid if below its vapour pressure curve (T<Tc and P>Psat)
    liquid = (T_k < TC) and (P_pa >= PropsSI("P", "T", T_k, "Q", 0, "CO2"))
    KH2O, KCO2 = _K0(Tdeg, liquid)
    yco2 = 1.0
    xco2 = yh2o = 0.0
    for _ in range(n_iter):
        V, a, b = _rk_volume(P, T_k, yco2, liquid)
        lnphi_co2, lnphi_h2o = _ln_phi(P, T_k, V, yco2, a, b)
        phi_co2, phi_h2o = np.exp(lnphi_co2), np.exp(lnphi_h2o)
        Ap = KH2O / (phi_h2o * P) * np.exp((P - P0) * VBAR_H2O / (R * T_k))
        Bp = phi_co2 * P / (55.508 * KCO2) * np.exp(-(P - P0) * VBAR_CO2 / (R * T_k))
        yh2o_new = (1.0 - Bp) / (1.0 / Ap - Bp)
        xco2_new = Bp * (1.0 - yh2o_new)
        if abs(yh2o_new - yh2o) < 1e-10 and abs(xco2_new - xco2) < 1e-12:
            yh2o, xco2 = yh2o_new, xco2_new
            break
        yh2o, xco2 = yh2o_new, xco2_new
        yco2 = 1.0 - yh2o
    return max(xco2, 0.0), max(yh2o, 0.0)


def molality(xco2):
    return xco2 * 55.508 / max(1.0 - xco2, 1e-12)      # mol CO2 / kg H2O


def main():
    print("=== validation: CO2 solubility in pure water (molality, mol/kg) ===")
    print("   T[C]  p[bar] | SPE2003 |  literature")
    targets = [(25, 60, 1.15), (25, 100, 1.35), (50, 100, 1.08), (40, 200, 1.45), (75, 150, 0.95)]
    for Tc, Pbar, lit in targets:
        x, y = mutual_solubility(Pbar * 1e5, Tc + TK)
        print(f"   {Tc:4.0f}  {Pbar:5.0f} | {molality(x):6.2f}  |  ~{lit:.2f}")

    # ---- mutual solubilities along the genuine three-phase locus (~CO2 saturation line) -------
    Td = np.linspace(2.0, TC - TK - 0.1, 160)
    psat = np.array([PropsSI("P", "T", t + TK, "Q", 0, "CO2") for t in Td])    # Pa
    xco2 = np.empty_like(Td)
    yh2o = np.empty_like(Td)
    for i, (t, p) in enumerate(zip(Td, psat)):
        xco2[i], yh2o[i] = mutual_solubility(p + 50.0, t + TK)     # just inside the liquid side

    fig, (axs, axc) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axs.plot(Td, 100 * xco2, color="#2a7f3f", lw=2, label="CO$_2$ in aqueous  ($x_{\\mathrm{CO_2}}$)")
    axs.plot(Td, 100 * yh2o, color="#1f4e9c", lw=2, label="H$_2$O in CO$_2$-rich ($y_{\\mathrm{H_2O}}$)")
    axs.set(xlabel="T [$^\\circ$C] (three-phase locus $\\approx$ CO$_2$ saturation)",
            ylabel="mole fraction [%]", title="(A)  mutual solubilities (vary with T!)")
    axs.legend(fontsize=9)
    axs.grid(alpha=0.3)
    ax2 = axs.twinx()
    ax2.plot(Td, [molality(x) for x in xco2], color="#2a7f3f", lw=0.8, ls=":")
    ax2.set_ylabel("CO$_2$ molality [mol/kg] (dotted)", color="#2a7f3f", fontsize=9)

    # phase compositions vs the immiscible model (0/1 constant)
    axc.plot(Td, 100 * (1 - xco2), color="#1f4e9c", lw=2, label="aqueous: H$_2$O")
    axc.plot(Td, 100 * xco2, color="#1f4e9c", lw=2, ls="--", label="aqueous: CO$_2$")
    axc.plot(Td, 100 * (1 - yh2o), color="#b8860b", lw=2, label="CO$_2$-rich: CO$_2$")
    axc.plot(Td, 100 * yh2o, color="#b8860b", lw=2, ls="--", label="CO$_2$-rich: H$_2$O")
    axc.axhline(100, color="0.6", ls=":", lw=0.8)
    axc.axhline(0, color="0.6", ls=":", lw=0.8)
    axc.set(xlabel="T [$^\\circ$C]", ylabel="phase composition [mol %]",
            title="(B)  phase compositions  (immiscible = flat 100/0)")
    axc.legend(fontsize=8, ncol=2)
    axc.set_ylim(-3, 103)

    fig.suptitle("H$_2$O-CO$_2$ TRUE mutual solubility (Spycher-Pruess 2003)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"co2_mutual_solubility.{ext}"), dpi=160, bbox_inches="tight")
    print("\nwrote co2_mutual_solubility.{png,pdf}")
    print("\nAt the fault boiling point (~28 C, ~6.9 MPa):")
    xb, yb = mutual_solubility(6.9e6, 28 + TK)
    print(f"   x_CO2(aqueous) = {100*xb:.2f} mol%  ({molality(xb):.2f} mol/kg);  "
          f"y_H2O(CO2-rich) = {100*yb:.2f} mol%")


if __name__ == "__main__":
    main()
