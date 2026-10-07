#!/usr/bin/env python
"""Build the H2O-CO2 OBL tables (ptz + phz) in the Driesner .vtr format, using CoolProp.

Immiscible two-component H2O-CO2; phases a (aqueous = pure water), l (CO2-rich liquid),
g (CO2-rich gas). Mapped to the Driesner three-phase slots so the VTKSampler reads them
unchanged:   l-slot = aqueous (water),   v-slot = CO2 gas,   h-slot = CO2 liquid.
An extra mu_h field carries the (mobile) CO2-liquid viscosity.

Axes match the brine tables:  x = z_CO2 [-],  y = T [degC] (ptz) or h [MJ/kg] (phz),
z = p [MPa].  Fields [units]: H,H_h,H_l,H_v [kJ/kg]; Rho,Rho_h,Rho_l,Rho_v [kg/m3];
S_h,S_l,S_v [-]; Temperature [K]; Xl,Xv [CO2 mass frac in l-/v-slot]; mu_l,mu_v,mu_h [Pa s];
phase_region (bitmask 1..7: a=1, l=2, g=4).

Run:  python build_co2_obl_table.py        # writes h2o_co2_x{pt,ph}.vtr + a validation figure
"""
from __future__ import annotations

import os
import numpy as np
import pyvista as pv
from CoolProp.CoolProp import PropsSI

HERE = os.path.dirname(os.path.abspath(__file__))
TK = 273.15
TC = PropsSI("Tcrit", "CO2")          # 304.13 K
PC = PropsSI("pcrit", "CO2")          # 7.377 MPa
RHOC = PropsSI("rhocrit", "CO2")      # 467.6 kg/m3
TW_MIN = 274.15                       # keep water above freezing

# common enthalpy reference: water at (0.1 MPa, 1 C), CO2 at (15 MPa, 1 C, dense liquid)
HW_REF = PropsSI("H", "P", 1.0e5, "T", TW_MIN, "Water")
HC_REF = PropsSI("H", "P", 15.0e6, "T", TW_MIN, "CO2")

# table axes (sized for the CO2-fault case; refine later if needed)
Z_AX = np.unique(np.concatenate([np.linspace(0, 1, 41), [0.001, 0.999]]))     # z_CO2
P_AX = np.linspace(0.1, 15.0, 141) * 1e6                                       # p [Pa]
T_AX = np.linspace(1.0, 60.0, 161)                                            # T [degC] (ptz y-axis)


def _props(fluid, P, T):
    """Vectorized single-phase (D, H, V) of a pure fluid at (P[Pa], T[K]); water T-clipped."""
    Tc = np.maximum(T, TW_MIN) if fluid == "Water" else T
    D = PropsSI("D", "P", P, "T", Tc, fluid)
    H = PropsSI("H", "P", P, "T", Tc, fluid)
    V = PropsSI("V", "P", P, "T", Tc, fluid)
    return D, H, V


def _co2_sat(P):
    """Saturated CO2 (Tsat, rho_l, rho_v, h_l, h_v, mu_l, mu_v) at each P<PC (nan where P>=PC)."""
    out = {k: np.full_like(P, np.nan, dtype=float) for k in
           ("Ts", "rl", "rv", "hl", "hv", "ml", "mv")}
    m = P < PC - 1.0
    if m.any():
        q = P[m]
        out["Ts"][m] = PropsSI("T", "P", q, "Q", 0, "CO2")
        for key, what, Q in (("rl", "D", 0), ("rv", "D", 1), ("hl", "H", 0),
                             ("hv", "H", 1), ("ml", "V", 0), ("mv", "V", 1)):
            out[key][m] = PropsSI(what, "P", q, "Q", Q, "CO2")
    return out


def flash_pt_grid():
    """All fields on the (z, T, p) grid (single-phase CO2 -> l or g; no 3-phase in p-T)."""
    Z, Tc, P = np.meshgrid(Z_AX, T_AX, P_AX, indexing="ij")
    T = Tc + TK
    Zf, Tf, Pf = Z.ravel(), T.ravel(), P.ravel()

    rw, hw, mw = _props("Water", Pf, Tf)
    rc, hc, mc = _props("CO2", Pf, Tf)
    sat = _co2_sat(Pf)
    psat = np.where(Tf < TC, _safe_psat(Tf), np.inf)       # CO2 saturation pressure at T

    co2_liquid = ((Tf < TC) & (Pf >= psat)) | ((Tf >= TC) & (Pf >= PC) & (rc >= RHOC))
    # everything not liquid and not supercritical-labelled-liquid is gas
    hw_r, hc_r = hw - HW_REF, hc - HC_REF

    # per-phase (slot) properties, filling the absent CO2 phase with its saturated value
    rho_l = rw                                              # aqueous (l-slot)
    h_l, mu_l = hw_r, mw
    rho_h = np.where(co2_liquid, rc, np.where(np.isfinite(sat["rl"]), sat["rl"], rc))   # CO2 liq
    h_h = np.where(co2_liquid, hc_r, np.where(np.isfinite(sat["hl"]), sat["hl"] - HC_REF, hc_r))
    mu_h = np.where(co2_liquid, mc, np.where(np.isfinite(sat["ml"]), sat["ml"], mc))
    rho_v = np.where(~co2_liquid, rc, np.where(np.isfinite(sat["rv"]), sat["rv"], rc))  # CO2 gas
    h_v = np.where(~co2_liquid, hc_r, np.where(np.isfinite(sat["hv"]), sat["hv"] - HC_REF, hc_r))
    mu_v = np.where(~co2_liquid, mc, np.where(np.isfinite(sat["mv"]), sat["mv"], mc))

    # saturations (per unit total mass): aqueous + one CO2 phase
    Va = (1 - Zf) / rw
    Vl_co2 = np.where(co2_liquid, Zf / rho_h, 0.0)         # CO2 liquid volume (h-slot)
    Vg_co2 = np.where(~co2_liquid, Zf / rho_v, 0.0)        # CO2 gas volume (v-slot)
    Vtot = Va + Vl_co2 + Vg_co2
    S_l, S_h, S_v = Va / Vtot, Vl_co2 / Vtot, Vg_co2 / Vtot
    rho_mix = 1.0 / Vtot
    h_mix = (1 - Zf) * hw_r + Zf * np.where(co2_liquid, hc_r, hc_r)   # single CO2 phase enthalpy

    return _assemble(Z.shape, Zf, S_l, S_h, S_v, rho_mix, rho_l, rho_h, rho_v,
                     h_mix, h_l, h_h, h_v, mu_l, mu_h, mu_v, Tf)


def flash_ph_grid(h_ax):
    """All fields on the (z, h, p) grid, including the three-phase a+l+g band."""
    nz, nh, npz = Z_AX.size, h_ax.size, P_AX.size
    # per-(z,p): invert h->T in the single-phase branches (from a fine T curve), 3-phase in the band
    Tfine = np.linspace(1.0, 60.0, 400) + TK
    out = {k: np.empty((nz, nh, npz)) for k in _FIELDS}
    for ip, P in enumerate(P_AX):
        Pv = np.full_like(Tfine, P)
        rw_f, hw_f, mw_f = _props("Water", Pv, Tfine)
        rc_f, hc_f, mc_f = _props("CO2", Pv, Tfine)
        psat = _safe_psat(Tfine)
        liq_f = ((Tfine < TC) & (P >= psat)) | ((Tfine >= TC) & (P >= PC) & (rc_f >= RHOC))
        sat = _co2_sat(np.array([P]))
        Ts = sat["Ts"][0]
        band_ok = np.isfinite(Ts) and (Ts >= TW_MIN)       # water must stay liquid
        if band_ok:
            rw_s, hw_s = _props("Water", np.array([P]), np.array([Ts]))[0][0], \
                _props("Water", np.array([P]), np.array([Ts]))[1][0]
            rl_s, rv_s = sat["rl"][0], sat["rv"][0]
            hl_s, hv_s = sat["hl"][0] - HC_REF, sat["hv"][0] - HC_REF
            ml_s, mv_s = sat["ml"][0], sat["mv"][0]
        for iz, Z in enumerate(Z_AX):
            hmix_f = (1 - Z) * (hw_f - HW_REF) + Z * (hc_f - HC_REF)
            for ih, h in enumerate(h_ax):
                if band_ok:
                    hmin = (1 - Z) * (hw_s - HW_REF) + Z * hl_s
                    hmax = (1 - Z) * (hw_s - HW_REF) + Z * hv_s
                else:
                    hmin = hmax = np.nan
                if band_ok and hmin <= h <= hmax:          # three-phase a+l+g
                    x = 0.0 if hmax == hmin else (h - hmin) / (hmax - hmin)
                    Va = (1 - Z) / rw_s
                    Vl = Z * (1 - x) / rl_s
                    Vg = Z * x / rv_s
                    Vt = Va + Vl + Vg
                    _store(out, iz, ih, ip, Z, Va / Vt, Vl / Vt, Vg / Vt, 1.0 / Vt,
                           rw_s, rl_s, rv_s, h, hw_s - HW_REF, hl_s, hv_s, mw_f_of(rw_s, P, Ts),
                           ml_s, mv_s, Ts)
                else:                                      # single CO2 phase: invert h->T
                    branch = liq_f if (not band_ok or h < (hmin if band_ok else np.inf)) else ~liq_f
                    if band_ok and h > hmax:
                        branch = ~liq_f
                    elif band_ok and h < hmin:
                        branch = liq_f
                    T, i0 = _invert(hmix_f, h, branch, Tfine)
                    co2liq = liq_f[i0]
                    rw0, hw0, mw0 = rw_f[i0], hw_f[i0] - HW_REF, mw_f[i0]
                    rc0, hc0, mc0 = rc_f[i0], hc_f[i0] - HC_REF, mc_f[i0]
                    rho_h = rc0 if co2liq else (rl_s if band_ok else rc0)
                    rho_v = rc0 if not co2liq else (rv_s if band_ok else rc0)
                    hh = hc0 if co2liq else (hl_s if band_ok else hc0)
                    hv = hc0 if not co2liq else (hv_s if band_ok else hc0)
                    muh = mc0 if co2liq else (ml_s if band_ok else mc0)
                    muv = mc0 if not co2liq else (mv_s if band_ok else mc0)
                    Va = (1 - Z) / rw0
                    Vl = Z / rho_h if co2liq else 0.0
                    Vg = Z / rho_v if not co2liq else 0.0
                    Vt = Va + Vl + Vg
                    _store(out, iz, ih, ip, Z, Va / Vt, Vl / Vt, Vg / Vt, 1.0 / Vt,
                           rw0, rho_h, rho_v, h, hw0, hh, hv, mw0, muh, muv, T)
    return out


# -- small helpers ----------------------------------------------------------------------
_FIELDS = ("phase_region", "H", "H_h", "H_l", "H_v", "Rho", "Rho_h", "Rho_l", "Rho_v",
           "S_h", "S_l", "S_v", "Temperature", "Xl", "Xv", "mu_l", "mu_v", "mu_h")


def _safe_psat(T):
    p = np.full_like(T, np.inf)
    m = T < TC
    if np.ndim(T) == 0:
        return PropsSI("P", "T", float(T), "Q", 0, "CO2") if T < TC else np.inf
    p[m] = PropsSI("P", "T", T[m], "Q", 0, "CO2")
    return p


def mw_f_of(rho, P, T):
    return PropsSI("V", "P", float(P), "T", float(T), "Water")


def _invert(hcurve, h, branch, Tfine):
    """T with hcurve(T)=h along the monotone branch (index into Tfine too)."""
    idx = np.where(branch)[0]
    if idx.size == 0:
        idx = np.arange(Tfine.size)
    hc, tc = hcurve[idx], Tfine[idx]
    order = np.argsort(hc)
    T = float(np.interp(h, hc[order], tc[order]))
    i0 = idx[int(np.argmin(np.abs(tc - T)))]
    return T, i0


def _region(S_l, S_h, S_v):
    tol = 1e-6
    return (1 * (S_l > tol) + 2 * (S_h > tol) + 4 * (S_v > tol)).astype(float)


def _assemble(shape, Z, S_l, S_h, S_v, rho, rho_l, rho_h, rho_v, H, H_l, H_h, H_v,
              mu_l, mu_h, mu_v, T):
    d = dict(phase_region=_region(S_l, S_h, S_v),
             H=H, H_h=H_h, H_l=H_l, H_v=H_v, Rho=rho, Rho_h=rho_h, Rho_l=rho_l, Rho_v=rho_v,
             S_h=S_h, S_l=S_l, S_v=S_v, Temperature=T,
             Xl=np.zeros_like(Z), Xv=np.ones_like(Z), mu_l=mu_l, mu_v=mu_v, mu_h=mu_h)
    return {k: v.reshape(shape) for k, v in d.items()}


def _store(out, iz, ih, ip, Z, S_l, S_h, S_v, rho, rho_l, rho_h, rho_v, H, H_l, H_h, H_v,
           mu_l, mu_h, mu_v, T):
    vals = dict(phase_region=float(1 * (S_l > 1e-6) + 2 * (S_h > 1e-6) + 4 * (S_v > 1e-6)),
                H=H, H_h=H_h, H_l=H_l, H_v=H_v, Rho=rho, Rho_h=rho_h, Rho_l=rho_l, Rho_v=rho_v,
                S_h=S_h, S_l=S_l, S_v=S_v, Temperature=T, Xl=0.0, Xv=1.0,
                mu_l=mu_l, mu_v=mu_v, mu_h=mu_h)
    for k, v in vals.items():
        out[k][iz, ih, ip] = v


def _write_vtr(path, x, y, z, fields):
    grid = pv.RectilinearGrid()
    grid.x, grid.y, grid.z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    for k, A in fields.items():
        grid.point_data[k] = A.ravel(order="F").astype(np.float32)
    grid.save(path)
    print("wrote", os.path.relpath(path, HERE), grid.dimensions)


def main():
    print("building p-T-z table ...")
    ptf = flash_pt_grid()
    # shift enthalpies so min H = 0 (Driesner convention; same offset for both tables)
    off = -float(np.nanmin(ptf["H"]))
    for k in ("H", "H_h", "H_l", "H_v"):
        ptf[k] = ptf[k] + off
    H_kJ = ptf.copy()
    for k in ("H", "H_h", "H_l", "H_v"):
        H_kJ[k] = ptf[k] / 1e3                             # J/kg -> kJ/kg
    _write_vtr(os.path.join(HERE, "h2o_co2_xpt.vtr"), Z_AX, T_AX, P_AX / 1e6, H_kJ)

    print("building p-h-z table ...")
    h_lo, h_hi = 0.0, float(np.nanmax(ptf["H"]))           # J/kg range from the p-T table
    h_ax = np.linspace(h_lo, h_hi, 161)                    # mixture enthalpy [J/kg]
    phf = flash_ph_grid(h_ax)
    for k in ("H", "H_h", "H_l", "H_v"):                   # offset already baked into h_ax via ptf
        phf[k] = phf[k] + off
    for k in ("H", "H_h", "H_l", "H_v"):
        phf[k] = phf[k] / 1e3
    _write_vtr(os.path.join(HERE, "h2o_co2_xph.vtr"), Z_AX, h_ax / 1e6, P_AX / 1e6, phf)
    print("enthalpy offset [kJ/kg] = %.1f,  h-axis [MJ/kg] in [%.3f, %.3f]"
          % (off / 1e3, h_lo / 1e6, h_hi / 1e6))


if __name__ == "__main__":
    main()
