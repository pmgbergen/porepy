#!/usr/bin/env python
"""Build the COMPOSITIONAL H2O-CO2 OBL tables (ptz + phz) with true mutual solubility.

Upgrade of build_co2_obl_table.py (immiscible): CO2 dissolves in the aqueous phase and H2O in the
CO2-rich phase, so the partial compositions Xl (CO2 mass frac in aqueous), Xv/Xh (CO2 mass frac in the
CO2 gas/liquid slots) now VARY with (p,T). Recipe (cross-checked, SPE2003 flash):
  - equilibrium compositions: Spycher-Pruess-Ennis-King 2003 (co2_mutual_solubility.mutual_solubility),
    mole->mass via w_aq/w_cr; lever rule on the overall CO2 mass fraction z.
  - aqueous enthalpy: heat of solution DHsol(T) from van't Hoff on the SPE2003 K_CO2 (liquid branch).
  - aqueous density: Garcia (2001) apparent molar volume of dissolved CO2 + CoolProp pure water.
  - CO2-rich density/enthalpy: pure CoolProp CO2 (dissolved water <0.17% effect); store Xv,Xh anyway.
  - three-phase a+l+g band at T=Tsat_CO2(p) for p in [~3.58, 7.38] MPa (water-liquid .. CO2 critical).
Slot map / fields / axes / enthalpy reference identical to build_co2_obl_table.py, plus the new Xh.

Run:  python build_co2_compositional_table.py            # full grid -> h2o_co2_x{pt,ph}.vtr
      python build_co2_compositional_table.py --coarse   # fast dry run (small axes, _coarse names)
      python build_co2_compositional_table.py --immiscible   # reduction check: must match immiscible
"""
from __future__ import annotations

import argparse
import os
import numpy as np
from CoolProp.CoolProp import PropsSI

import build_co2_obl_table as imm
from build_co2_obl_table import (TK, TC, PC, RHOC, TW_MIN, HW_REF, HC_REF,
                                 _safe_psat, _co2_sat, _invert, _write_vtr)
import co2_mutual_solubility as spe

HERE = os.path.dirname(os.path.abspath(__file__))
R = 8.314462618
LN10 = np.log(10.0)
M_H2O, M_CO2 = 18.015e-3, 44.01e-3
DP = 50.0                                 # Pa offset to pick the L/V branch on the sat line

# Garcia (2001) apparent molar volume of dissolved CO2 [cm3/mol], T in degC
_G = (37.51, -9.585e-2, 8.740e-4, -5.044e-7)

# toggled on for the reduction-to-immiscible check
IMMISCIBLE = False


def dHsol(Tc, liquid):
    """Heat of solution of CO2 [J/mol] via van't Hoff on SPE2003 K_CO2 (exothermic)."""
    if IMMISCIBLE:
        return np.zeros_like(np.asarray(Tc, float))
    b, c = np.where(liquid, 1.368e-2, 1.304e-2), np.where(liquid, -5.380e-5, -5.446e-5)
    T = Tc + TK
    return -R * LN10 * T**2 * (b + 2.0 * c * Tc)


def w_aq(x):
    return x * M_CO2 / (x * M_CO2 + (1.0 - x) * M_H2O)


def w_cr(y):
    return (1.0 - y) * M_CO2 / ((1.0 - y) * M_CO2 + y * M_H2O)


def rho_aqueous(P, Tc, rw, w):
    """Aqueous density [kg/m3]: Garcia (2001) apparent molar volume + CoolProp pure water rw."""
    Vphi = (_G[0] + _G[1] * Tc + _G[2] * Tc**2 + _G[3] * Tc**3) * 1e-6     # m3/mol
    m = np.where(w < 1.0, w / (M_CO2 * np.maximum(1.0 - w, 1e-12)), 0.0)   # mol CO2 / kg H2O
    return (1.0 + m * M_CO2) / (1.0 / rw + m * Vphi)


def comp_at(P, Tk):
    """SPE2003 saturated CO2 mass fractions (c_a aqueous, c_c CO2-rich) at scalar (P[Pa],T[K])."""
    if IMMISCIBLE:
        return 0.0, 1.0
    x, y = spe.mutual_solubility(P, Tk)
    return w_aq(min(max(x, 0.0), 0.3)), w_cr(min(max(y, 0.0), 0.1))


def comp_grid(P2d, Tk2d):
    """c_a(p,T), c_c(p,T) over a 2D (T,p) grid (scalar SPE flash loop)."""
    ca = np.empty(P2d.shape)
    cc = np.empty(P2d.shape)
    it = np.nditer(P2d, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        ca[i], cc[i] = comp_at(float(P2d[i]), float(Tk2d[i]))
    return ca, cc


def _co2_liquid(Tk, P, rc):
    psat = np.where(Tk < TC, _safe_psat(Tk), np.inf)
    return ((Tk < TC) & (P >= psat)) | ((Tk >= TC) & (P >= PC) & (rc >= RHOC))


def flash_pt(Z, P, Tc, rw, hw_r, hc_r, rc, mw, mc, co2_liq, ca, cc,
             sat_rl, sat_rv, sat_hl, sat_hv, sat_ml, sat_mv):
    """All slot fields on an array grid (Z plus all (T,p) inputs broadcastable to Z.shape).

    Returns a dict of arrays shaped like Z. Enthalpies are J/kg (pre global offset)."""
    dhs = dHsol(Tc, co2_liq)
    h_diss = hc_r + dhs / M_CO2
    single_aq = Z <= ca
    single_cr = Z >= cc
    two = (~single_aq) & (~single_cr)
    Fcr = np.where(two, (Z - ca) / np.maximum(cc - ca, 1e-12), np.where(single_cr, 1.0, 0.0))
    Fcr = np.clip(Fcr, 0.0, 1.0)
    w_aqc = np.where(single_aq, Z, ca)                     # CO2 mass frac in aqueous (= Xl)
    w_crc = np.where(single_cr, Z, cc)                     # CO2 mass frac in CO2-rich (= Xv, Xh)
    aq_mass = np.where(single_cr, 0.0, np.where(single_aq, 1.0, 1.0 - Fcr))
    cr_mass = 1.0 - aq_mass

    rho_l = rho_aqueous(P, Tc, rw, w_aqc)
    h_l = (1.0 - w_aqc) * hw_r + w_aqc * h_diss
    h_cr = w_crc * hc_r + (1.0 - w_crc) * hw_r

    Va = aq_mass / rho_l
    Vcr = np.where(cr_mass > 0.0, cr_mass / rc, 0.0)
    Vt = Va + Vcr
    rho = 1.0 / Vt
    S_l = Va / Vt
    S_h = np.where(co2_liq, Vcr / Vt, 0.0)
    S_v = np.where(~co2_liq, Vcr / Vt, 0.0)
    h_mix = aq_mass * h_l + cr_mass * h_cr

    # CO2 slot fills (z-independent): present slot = pure CO2, absent slot = saturated value
    def fill(liqval, satval):
        return np.where(co2_liq, liqval, np.where(np.isfinite(satval), satval, liqval))
    rho_h = fill(rc, sat_rl)
    rho_v = np.where(~co2_liq, rc, np.where(np.isfinite(sat_rv), sat_rv, rc))
    h_h = fill(hc_r, sat_hl - HC_REF)
    h_v = np.where(~co2_liq, hc_r, np.where(np.isfinite(sat_hv), sat_hv - HC_REF, hc_r))
    mu_h = fill(mc, sat_ml)
    mu_v = np.where(~co2_liq, mc, np.where(np.isfinite(sat_mv), sat_mv, mc))

    region = 1.0 * (S_l > 1e-6) + 2.0 * (S_h > 1e-6) + 4.0 * (S_v > 1e-6)
    ones = np.ones_like(Z)
    return dict(phase_region=region, H=h_mix, H_h=h_h * ones, H_l=h_l, H_v=h_v * ones,
                Rho=rho, Rho_h=rho_h * ones, Rho_l=rho_l, Rho_v=rho_v * ones,
                S_h=S_h, S_l=S_l, S_v=S_v, Temperature=(Tc + TK) * ones,
                Xl=w_aqc, Xv=w_crc, Xh=w_crc, mu_l=mw * ones, mu_v=mu_v * ones, mu_h=mu_h * ones)


_FIELDS = ("phase_region", "H", "H_h", "H_l", "H_v", "Rho", "Rho_h", "Rho_l", "Rho_v",
           "S_h", "S_l", "S_v", "Temperature", "Xl", "Xv", "Xh", "mu_l", "mu_v", "mu_h")


def build_pt(Z_AX, T_AX, P_AX):
    """p-T-z table (single CO2 phase in p-T; three-phase only in p-h)."""
    Tc2, P2 = np.meshgrid(T_AX, P_AX, indexing="ij")                      # (nT,nP) degC, Pa
    Tk2 = Tc2 + TK
    rw = PropsSI("D", "P", P2.ravel(), "T", np.maximum(Tk2, TW_MIN).ravel(), "Water").reshape(P2.shape)
    hw = PropsSI("H", "P", P2.ravel(), "T", np.maximum(Tk2, TW_MIN).ravel(), "Water").reshape(P2.shape)
    mw = PropsSI("V", "P", P2.ravel(), "T", np.maximum(Tk2, TW_MIN).ravel(), "Water").reshape(P2.shape)
    rc = PropsSI("D", "P", P2.ravel(), "T", Tk2.ravel(), "CO2").reshape(P2.shape)
    hc = PropsSI("H", "P", P2.ravel(), "T", Tk2.ravel(), "CO2").reshape(P2.shape)
    mc = PropsSI("V", "P", P2.ravel(), "T", Tk2.ravel(), "CO2").reshape(P2.shape)
    co2_liq = _co2_liquid(Tk2, P2, rc)
    ca, cc = comp_grid(P2, Tk2)
    sat = _co2_sat(P_AX)                                                   # per-p (nP,)
    sat_b = {k: np.broadcast_to(v, P2.shape) for k, v in sat.items()}

    Z = Z_AX[:, None, None]
    def b(a):
        return np.broadcast_to(a, (Z_AX.size,) + P2.shape)
    out = flash_pt(Z, b(P2), b(Tc2), b(rw), b(hw) - HW_REF, b(hc) - HC_REF, b(rc), b(mw), b(mc),
                   b(co2_liq), b(ca), b(cc), b(sat_b["rl"]), b(sat_b["rv"]), b(sat_b["hl"]),
                   b(sat_b["hv"]), b(sat_b["ml"]), b(sat_b["mv"]))
    return out


def build_ph(Z_AX, h_ax, P_AX, nfine=320):
    """p-h-z table: per p build the fine-T compositional curve, invert h->T, insert the 3-phase band."""
    nz, nh, npz = Z_AX.size, h_ax.size, P_AX.size
    out = {k: np.empty((nz, nh, npz)) for k in _FIELDS}
    Tc_f = np.linspace(1.0, 60.0, nfine)
    Tk_f = Tc_f + TK
    for ip, P in enumerate(P_AX):
        Pv = np.full(nfine, P)
        rw_f = PropsSI("D", "P", Pv, "T", np.maximum(Tk_f, TW_MIN), "Water")
        hw_f = PropsSI("H", "P", Pv, "T", np.maximum(Tk_f, TW_MIN), "Water")
        mw_f = PropsSI("V", "P", Pv, "T", np.maximum(Tk_f, TW_MIN), "Water")
        rc_f = PropsSI("D", "P", Pv, "T", Tk_f, "CO2")
        hc_f = PropsSI("H", "P", Pv, "T", Tk_f, "CO2")
        mc_f = PropsSI("V", "P", Pv, "T", Tk_f, "CO2")
        liq_f = _co2_liquid(Tk_f, Pv, rc_f)
        ca_f, cc_f = comp_grid(Pv[None, :], Tk_f[None, :])
        ca_f, cc_f = ca_f[0], cc_f[0]
        sat = _co2_sat(np.array([P]))
        satb = {k: np.full(nfine, sat[k][0]) for k in sat}

        # fine-T fields over (z, Tfine)
        Zc = Z_AX[:, None]
        def bb(a):
            return np.broadcast_to(a, (nz, nfine))
        ff = flash_pt(Zc, bb(Pv), bb(Tc_f), bb(rw_f), bb(hw_f) - HW_REF, bb(hc_f) - HC_REF, bb(rc_f),
                      bb(mw_f), bb(mc_f), bb(liq_f), bb(ca_f), bb(cc_f), bb(satb["rl"]), bb(satb["rv"]),
                      bb(satb["hl"]), bb(satb["hv"]), bb(satb["ml"]), bb(satb["mv"]))
        Hf = ff["H"]                                                       # (nz, nfine)

        # three-phase band at Ts (water must stay liquid)
        Ts = sat["Ts"][0]
        band_ok = np.isfinite(Ts) and (Ts >= TW_MIN)
        if band_ok:
            caL, ccL = comp_at(P + DP, Ts)
            caV, ccV = comp_at(P - DP, Ts)
            hw_s = PropsSI("H", "P", P, "T", Ts, "Water") - HW_REF
            mw_s = PropsSI("V", "P", P, "T", Ts, "Water")
            rw_s = PropsSI("D", "P", P, "T", Ts, "Water")
            rl_s, rv_s = sat["rl"][0], sat["rv"][0]
            hl_s, hv_s = sat["hl"][0] - HC_REF, sat["hv"][0] - HC_REF
            ml_s, mv_s = sat["ml"][0], sat["mv"][0]
            dhsL = float(dHsol(np.array(Ts - TK), np.array(True)))
            dhsV = float(dHsol(np.array(Ts - TK), np.array(False)))

        for iz, Z in enumerate(Z_AX):
            h = h_ax
            done = np.zeros(nh, bool)
            if band_ok and (caL < Z < ccL) and (caV < Z < ccV):
                q = np.linspace(0.0, 1.0, 64)
                ca_q = caL * (1 - q) + caV * q
                cc_q = ccL * (1 - q) + ccV * q
                hcrL, hcrV = hl_s, hv_s
                hco2_q = hcrL * (1 - q) + hcrV * q
                dhs_q = dhsL * (1 - q) + dhsV * q
                Fcr_q = np.clip((Z - ca_q) / np.maximum(cc_q - ca_q, 1e-12), 0.0, 1.0)
                aqm = 1.0 - Fcr_q
                h_aq_q = (1 - ca_q) * hw_s + ca_q * (hco2_q + dhs_q / M_CO2)
                h_cr_q = cc_q * hco2_q + (1 - cc_q) * hw_s
                hmix_q = aqm * h_aq_q + Fcr_q * h_cr_q
                hlo, hhi = hmix_q[0], hmix_q[-1]
                inb = (h >= min(hlo, hhi)) & (h <= max(hlo, hhi))
                if inb.any():
                    order = np.argsort(hmix_q)
                    qh = np.interp(h[inb], hmix_q[order], q[order])
                    caq = caL * (1 - qh) + caV * qh
                    ccq = ccL * (1 - qh) + ccV * qh
                    hco2q = hl_s * (1 - qh) + hv_s * qh          # CO2 ref enthalpy through the band
                    dhsq = dhsL * (1 - qh) + dhsV * qh
                    Fq = np.clip((Z - caq) / np.maximum(ccq - caq, 1e-12), 0.0, 1.0)
                    rho_lq = rho_aqueous(P, Ts - TK, rw_s, caq)
                    h_l_q = (1 - caq) * hw_s + caq * (hco2q + dhsq / M_CO2)
                    Va = (1 - Fq) / rho_lq
                    Vh = Fq * (1 - qh) / rl_s
                    Vg = Fq * qh / rv_s
                    Vt = Va + Vh + Vg
                    _put(out, iz, inb, ip, dict(
                        phase_region=7.0, H=h[inb], H_h=hl_s, H_l=h_l_q, H_v=hv_s,
                        Rho=1.0 / Vt, Rho_h=rl_s, Rho_l=rho_lq, Rho_v=rv_s,
                        S_h=Vh / Vt, S_l=Va / Vt, S_v=Vg / Vt, Temperature=Ts,
                        Xl=caq, Xv=ccq, Xh=ccq, mu_l=mw_s, mu_v=mv_s, mu_h=ml_s))
                    done |= inb
                # single-phase branches outside the band
                lo_mask = (~done) & (h < min(hlo, hhi))
                hi_mask = (~done) & (h > max(hlo, hhi))
                _invert_fill(out, iz, lo_mask, ip, h, Hf, liq_f, ff, Tk_f, branch=liq_f)
                _invert_fill(out, iz, hi_mask, ip, h, Hf, ~liq_f, ff, Tk_f, branch=~liq_f)
            else:
                full = np.ones(nfine, bool)
                _invert_fill(out, iz, ~done, ip, h, Hf, full, ff, Tk_f, branch=full)
    return out


def _invert_fill(out, iz, hmask, ip, h_ax, Hf, _unused, ff, Tk_f, branch):
    """Invert h->T along the monotone branch and gather every field at the nearest fine index."""
    if not np.any(hmask):
        return
    idx = np.where(branch)[0]
    if idx.size == 0:
        idx = np.arange(Tk_f.size)
    hc = Hf[iz, idx]
    tk = Tk_f[idx]
    order = np.argsort(hc)
    hc_s, tk_s, idx_s = hc[order], tk[order], idx[order]
    hq = h_ax[hmask]
    T = np.interp(hq, hc_s, tk_s)
    # nearest fine index on the branch for each query
    pos = np.searchsorted(tk_s, T)
    pos = np.clip(pos, 1, tk_s.size - 1)
    left = (T - tk_s[pos - 1]) <= (tk_s[pos] - T)
    i0 = np.where(left, idx_s[pos - 1], idx_s[pos])
    rec = {k: (ff[k][iz, i0] if ff[k].ndim == 2 else ff[k]) for k in _FIELDS}
    rec["Temperature"] = T
    rec["H"] = hq
    _put(out, iz, hmask, ip, rec)


def _put(out, iz, hmask, ip, rec):
    sel = np.where(hmask)[0] if hmask.dtype == bool else hmask
    for k in _FIELDS:
        v = rec[k]
        out[k][iz, sel, ip] = v


def _axes(coarse):
    if coarse:
        Z = np.unique(np.concatenate([np.linspace(0, 1, 11), [0.001, 0.999]]))
        return Z, np.linspace(1.0, 60.0, 41), np.linspace(0.1, 15.0, 41) * 1e6, 80
    return imm.Z_AX, imm.T_AX, imm.P_AX, 320


# full physical range whose min(H)=0 fixes the canonical Driesner enthalpy offset
REF_P = (0.1, 15.0, 141)
REF_T = (1.0, 60.0, 161)


def canonical_offset():
    """Enthalpy offset [J/kg] so min(H)=0 over the FULL physical range (p 0.1-15 MPa, T 1-60 C).
    Keeps the h-frame identical for any narrowed sub-window built with it."""
    Pf = np.linspace(REF_P[0], REF_P[1], REF_P[2]) * 1e6
    Tf = np.linspace(REF_T[0], REF_T[1], REF_T[2])
    return -float(np.nanmin(build_pt(imm.Z_AX, Tf, Pf)["H"]))


def build_and_write(Z_AX, T_AX, P_AX, nfine=320, tag="", off=None, h_min=None, h_max=None, nh=161):
    """Build and write the ptz + phz VTR tables (+ offset sidecar). Returns the offset used."""
    print("building p-T-z table (compositional) ...", "IMMISCIBLE" if IMMISCIBLE else "")
    ptf = build_pt(Z_AX, T_AX, P_AX)
    raw_min = float(np.nanmin(ptf["H"]))                 # pre-offset enthalpy range
    raw_max = float(np.nanmax(ptf["H"]))
    if off is None:
        off = -raw_min
    for k in ("H", "H_h", "H_l", "H_v"):
        ptf[k] = ptf[k] + off
    H_kJ = {k: (ptf[k] / 1e3 if k in ("H", "H_h", "H_l", "H_v") else ptf[k]) for k in _FIELDS}
    _write_vtr(os.path.join(HERE, f"h2o_co2{tag}_xpt.vtr"), Z_AX, T_AX, P_AX / 1e6, H_kJ)

    print("building p-h-z table (compositional) ...")
    if h_min is not None and h_max is not None:
        h_ax = np.linspace(h_min * 1e6 - off, h_max * 1e6 - off, nh)    # raw; coord = h_ax+off
    else:
        h_ax = np.linspace(raw_min, raw_max, nh)
    phf = build_ph(Z_AX, h_ax, P_AX, nfine=nfine)
    for k in ("H", "H_h", "H_l", "H_v"):
        phf[k] = phf[k] + off
    H_kJ = {k: (phf[k] / 1e3 if k in ("H", "H_h", "H_l", "H_v") else phf[k]) for k in _FIELDS}
    _write_vtr(os.path.join(HERE, f"h2o_co2{tag}_xph.vtr"), Z_AX, (h_ax + off) / 1e6, P_AX / 1e6, H_kJ)
    with open(os.path.join(HERE, f"h2o_co2{tag}_offset.txt"), "w") as fh:
        fh.write(repr(off))                              # J/kg; error/validation scripts read this
    print("axes: z[%d] p[%.1f,%.1f]MPa x%d, T[%.0f,%.0f]C x%d, h[%.3f,%.3f]MJ/kg x%d  (off %.1f kJ/kg)"
          % (Z_AX.size, P_AX[0] / 1e6, P_AX[-1] / 1e6, P_AX.size, T_AX[0], T_AX[-1], T_AX.size,
             (h_ax[0] + off) / 1e6, (h_ax[-1] + off) / 1e6, h_ax.size, off / 1e3))
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--immiscible", action="store_true", help="reduction check (no solubility)")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--p-min", type=float, default=0.1, help="pressure min [MPa]")
    ap.add_argument("--p-max", type=float, default=15.0, help="pressure max [MPa]")
    ap.add_argument("--t-min", type=float, default=1.0, help="temperature min [degC] (ptz axis)")
    ap.add_argument("--t-max", type=float, default=60.0, help="temperature max [degC]")
    ap.add_argument("--h-min", type=float, default=None, help="phz enthalpy min [MJ/kg] (coordinate)")
    ap.add_argument("--h-max", type=float, default=None, help="phz enthalpy max [MJ/kg]")
    ap.add_argument("--np", type=int, default=141, dest="np_", help="# pressure nodes")
    ap.add_argument("--nt", type=int, default=161, help="# temperature nodes")
    ap.add_argument("--nh", type=int, default=161, help="# enthalpy nodes")
    ap.add_argument("--off", type=float, default=None, help="fixed enthalpy offset [J/kg]")
    a = ap.parse_args()
    global IMMISCIBLE
    IMMISCIBLE = a.immiscible

    if a.coarse:
        Z_AX, T_AX, P_AX, nfine = _axes(True)
        a.nh = 161
    else:
        Z_AX = imm.Z_AX
        T_AX = np.linspace(a.t_min, a.t_max, a.nt)
        P_AX = np.linspace(a.p_min, a.p_max, a.np_) * 1e6
        nfine = 320
    tag = ("_coarse" if a.coarse else "") + ("_immisc_check" if a.immiscible else "") + a.suffix
    build_and_write(Z_AX, T_AX, P_AX, nfine=nfine, tag=tag, off=a.off,
                    h_min=a.h_min, h_max=a.h_max, nh=a.nh)


if __name__ == "__main__":
    main()
