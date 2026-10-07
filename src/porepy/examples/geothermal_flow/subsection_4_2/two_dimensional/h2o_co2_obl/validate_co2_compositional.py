#!/usr/bin/env python
"""Validate the compositional H2O-CO2 OBL tables.

(1) reduction-to-immiscible: build_co2_compositional_table --immiscible must reproduce the backed-up
    immiscible tables (h2o_co2_immisc_x{pt,ph}.vtr) field by field.
(2) spot-checks: Xl, Rho_l, H_l, Temperature at sample (z,T,p) vs the direct SPE/Garcia/heat-of-solution
    recipe and CoolProp.
(3) per-node consistency: S sum = 1; mixture Rho = 1/sum(S_k/Rho_k); mixture H = sum(phase-mass * H_k).
(4) a p-h phase map read straight from the table (compare with co2_ph_slices.py).
"""
from __future__ import annotations
import os
import numpy as np
import pyvista as pv
from CoolProp.CoolProp import PropsSI

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import build_co2_compositional_table as B
import co2_mutual_solubility as spe

HERE = os.path.dirname(os.path.abspath(__file__))
TK = 273.15


def load(name):
    m = pv.read(os.path.join(HERE, name))
    nx, ny, nz = m.dimensions
    ax = (np.asarray(m.x), np.asarray(m.y), np.asarray(m.z))
    d = {k: np.asarray(m.point_data[k]).reshape((nx, ny, nz), order="F") for k in m.point_data.keys()}
    return ax, d


def reduction():
    print("=" * 78, "\n(1) REDUCTION TO IMMISCIBLE  (compositional --immiscible  vs  backup immiscible)")
    for tag in ("xpt", "xph"):
        _, a = load(f"h2o_co2_immisc_check_check_{tag}.vtr")
        _, b = load(f"h2o_co2_immisc_{tag}.vtr")
        print(f"  --- {tag} ---")
        for k in sorted(set(a) & set(b)):
            da = np.abs(a[k] - b[k])
            sc = np.maximum(np.abs(b[k]).max(), 1e-30)
            print(f"    {k:13s} max|Δ|={da.max():.3e}  rel={da.max()/sc:.2e}")
        extra = set(a) - set(b)
        if extra:
            print(f"    (new field not in immiscible: {sorted(extra)})")


def spot():
    print("=" * 78, "\n(2) COMPOSITIONAL SPOT-CHECKS  (table vs direct recipe)")
    (xz, yT, zp), d = load("h2o_co2_xpt.vtr")
    from scipy.interpolate import RegularGridInterpolator as RGI
    itp = {k: RGI((xz, yT, zp), d[k], bounds_error=False, fill_value=None) for k in d}
    print("   z    T[C] p[MPa] | Xl tab/dir | Rho_l tab/dir | H_l tab/dir[kJ/kg] | T tab/dir")
    for z, Tc, pM in [(0.1, 28, 6.9), (0.25, 40, 10.0), (0.05, 20, 5.0), (0.5, 50, 12.0)]:
        q = np.array([[z, Tc, pM]])
        P, Tk = pM * 1e6, Tc + TK
        x, _ = spe.mutual_solubility(P, Tk)
        ca = B.w_aq(min(max(x, 0), 0.3))
        w = min(z, ca)                                   # dissolved CO2 mass frac in aqueous
        rw = PropsSI("D", "P", P, "T", max(Tk, B.TW_MIN), "Water")
        rho_l_dir = B.rho_aqueous(P, Tc, rw, w)
        liq = (Tk < B.TC) and (P >= PropsSI("P", "T", Tk, "Q", 0, "CO2")) if Tk < B.TC else (P >= B.PC)
        dhs = float(B.dHsol(np.array(float(Tc)), np.array(bool(liq))))
        hw = PropsSI("H", "P", P, "T", max(Tk, B.TW_MIN), "Water") - B.HW_REF
        hc = PropsSI("H", "P", P, "T", Tk, "CO2") - B.HC_REF
        h_l_dir = ((1 - w) * hw + w * (hc + dhs / B.M_CO2))      # J/kg, pre-offset (table has +offset)
        print(f"  {z:.2f} {Tc:4.0f} {pM:5.1f} | {float(itp['Xl'](q)):.4f}/{w:.4f} | "
              f"{float(itp['Rho_l'](q)):6.1f}/{rho_l_dir:6.1f} | "
              f"{float(itp['H_l'](q)):7.1f}/({h_l_dir/1e3:7.1f}+off) | "
              f"{float(itp['Temperature'](q)):.1f}/{Tk:.1f}")


def consistency():
    print("=" * 78, "\n(3) PER-NODE CONSISTENCY")
    for tag in ("xpt", "xph"):
        _, d = load(f"h2o_co2_{tag}.vtr")
        S = d["S_l"] + d["S_h"] + d["S_v"]
        # mixture density from saturations and slot densities
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = d["S_l"] / d["Rho_l"] + d["S_h"] / d["Rho_h"] + d["S_v"] / d["Rho_v"]
            rho_recon = 1.0 / inv
        drho = np.abs(rho_recon - d["Rho"]) / np.maximum(d["Rho"], 1e-9)
        # mixture enthalpy = sum phase-mass-frac * H_k ; phase mass frac = S_k Rho_k / Rho
        mf_l = d["S_l"] * d["Rho_l"] / d["Rho"]
        mf_h = d["S_h"] * d["Rho_h"] / d["Rho"]
        mf_v = d["S_v"] * d["Rho_v"] / d["Rho"]
        H_recon = mf_l * d["H_l"] + mf_h * d["H_h"] + mf_v * d["H_v"]
        dH = np.abs(H_recon - d["H"])
        print(f"  {tag}: S-sum max|1-Σ|={np.abs(1-S).max():.2e}  "
              f"Rho recon rel={np.nanmax(drho):.2e}  H recon max|Δ|={np.nanmax(dH):.3e} kJ/kg "
              f"(massfrac Σ max|1-Σ|={np.abs(1-(mf_l+mf_h+mf_v)).max():.2e})")


def ph_map():
    print("=" * 78, "\n(4) p-h PHASE MAP FROM THE TABLE  -> co2_table_ph_map.png")
    (xz, yh, zp), d = load("h2o_co2_xph.vtr")
    from scipy.interpolate import RegularGridInterpolator as RGI
    reg = RGI((xz, yh, zp), d["phase_region"], bounds_error=False, fill_value=None)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    H, P = np.meshgrid(yh, zp, indexing="ij")
    for ax, z in zip(axes, (0.0, 0.1, 0.25)):
        R = reg(np.column_stack([np.full(H.size, z), H.ravel(), P.ravel()])).reshape(H.shape)
        im = ax.pcolormesh(H, P, np.round(R), cmap="Set2", shading="auto", vmin=1, vmax=7)
        ax.axhline(B.PC / 1e6, color="firebrick", ls=":", lw=1)
        ax.set(xlabel="h [MJ/kg]", title=f"$z_{{CO_2}}={z:g}$")
    axes[0].set_ylabel("p [MPa]")
    fig.colorbar(im, ax=axes, ticks=[1, 2, 3, 4, 5, 6, 7], label="phase_region (a=1,l=2,g=4; 7=a+l+g)")
    fig.suptitle("Compositional H2O-CO2 table: phase regions read from h2o_co2_xph.vtr")
    fig.savefig(os.path.join(HERE, "co2_table_ph_map.png"), dpi=150, bbox_inches="tight")
    print("   wrote co2_table_ph_map.png")


if __name__ == "__main__":
    reduction()
    spot()
    consistency()
    ph_map()
