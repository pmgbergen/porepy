#!/usr/bin/env python
"""Two-cell thermal-compositional numerical-flux maps (PPU vs HU), TPFA, over OVERALL COMPOSITION.

Complements the monotonicity plots: those sweep the overall composition z_CH4 on the two-cell
column; this shows the numerical FLUX itself as a function of the two cells' overall compositions,
so the PPU kink along each phase-potential-reversal locus (the source of flip-flopping) and its
absence under HU are visible -- the thermal-compositional analog of Hamon, Mallison & Tchelepi
(2016, CMAME 311) Fig. 1.

Compositional idea: the independent transport variable is the overall composition z (mass
fractions); saturations come from the IMMISCIBLE flash s_l(z) = (z_l/rho_l) / sum_k (z_k/rho_k)
(= buoyancy_flow_model.py's s(z)). Components and densities are the monotonicity model's: H2O
(1000), C5H12 (700), CH4 (200) -- used as ratios; viscosities equal (the real 100:10:1 contrast
blows Delta p up at fixed total velocity and swamps the comparison). Quadratic k_r=s^2, h_l=c_p T.
Fluxes evaluated at a FIXED total velocity (Delta p solved per point) so the phase-potential signs
vary with z. Per-component mass flux rho_l u_l and advected energy sum_l (rho_l h_l) u_l. Schemes:
PPU (phase-potential upwinding) and HU (total-velocity viscous split + mobility-product buoyancy;
hamon_2d_solver 'hu-mp').

  N=2 (H2O/CH4):        F surface over (z_CH4^L, z_CH4^R) in [0,1]^2   -> Fig. 1 analog
  N=3 (H2O/C5H12/CH4):  F map over the left-cell composition triangle (z_R fixed)

Run:  python thermal_flux_surfaces.py           # writes figures/thermal_flux_surfaces_{2,3}phase.*
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# fixed interface parameters (nondimensional; Fig. 1 uses T=1, g_l = rho_l, u_T fixed)
TF = 1.0            # transmissibility
GC = 1.0            # gravity coefficient T_f * g * dz  (so the buoyancy weight of phase l is GC*rho_l)
CP = 1.0            # phase specific heat (h_l = CP * T)
EPS = 1.0e-30


# Components of the monotonicity model (tests/functional/setups/buoyancy_flow_model.py),
# nondimensionalized by the gas values (rho/200, mu/1e-5) -- ratios are what the flux and the
# immiscible flash s(z) depend on: H2O 1000/1e-3, C5H12 700/1e-4, CH4 200/1e-5.
_RHO_REAL = {"H2O": 1000.0, "C5H12": 700.0, "CH4": 200.0}
_MU_REAL = {"H2O": 1.0e-3, "C5H12": 1.0e-4, "CH4": 1.0e-5}
_COMPS = {2: ["H2O", "CH4"], 3: ["H2O", "C5H12", "CH4"]}      # heavy (index 0) -> light
_TEX = {"H2O": r"H_2O", "C5H12": r"C_5H_{12}", "CH4": r"CH_4"}


def _phase_system(n):
    """(RHO, MU, names) for the n monotonicity components, heavy->light, nondimensionalized.

    Densities keep the real component ratios (they drive the immiscible flash s(z) and buoyancy).
    Viscosities are set EQUAL: the real 100:10:1 contrast only rescales total mobility, but at a
    fixed total velocity that forces Delta p to blow up in the low-mobility (water-rich) corner and
    swamps the PPU-vs-HU kink comparison; equal mu keeps lambda_T O(1) and the surfaces readable."""
    names = _COMPS[n]
    rho = np.array([_RHO_REAL[c] / _RHO_REAL["CH4"] for c in names])   # -> [5, 3.5, 1]
    mu = np.ones(len(names))
    return rho, mu, names


def s_of_z(z, rho):
    """Immiscible flash: saturations (phase volume fractions) from the overall mass composition z,
    s_l = (z_l/rho_l) / sum_k (z_k/rho_k)  (= buoyancy_flow_model's s(z), density-ratio invariant)."""
    v = np.clip(z, 0.0, 1.0) / rho
    tot = v.sum()
    return v / tot if tot > 0 else np.full_like(v, 1.0 / len(v))


def _zlabel(name, side=""):
    sup = f"^{side}" if side else ""
    return r"$z%s_{\mathrm{%s}}$" % (sup, _TEX.get(name, name))


def _side_masks(rho, pairs):
    """Midpoint background weight m_l in {0,1/2,1} per pair (hamon_2d_solver._side_masks)."""
    m = {}
    for a, b in pairs:
        m[(a, b)] = 0.5 * (1.0 + np.sign((rho[a] - rho[b]) * (2.0 * rho - rho[a] - rho[b])))
    return m


def _lam(s, mu):
    sc = np.clip(s, 0.0, 1.0)
    return sc * sc / mu


# --------------------------------------------------------------------------- flux kernels
def ppu_phase_fluxes(sL, sR, dp, rho, mu):
    """PPU phase volumetric fluxes q_l = lam_l[up] * Phi_l, Phi_l = T_f dp - GC rho_l."""
    lamL, lamR = _lam(sL, mu), _lam(sR, mu)
    phi = TF * dp - GC * rho
    lam_up = np.where(phi >= 0.0, lamL, lamR)
    return lam_up * phi


def hu_phase_fluxes(sL, sR, dp, rho, mu, masks, pairs):
    """HU (hu-mp) phase volumetric fluxes: total-velocity viscous split + mobility-product buoyancy."""
    lamL, lamR = _lam(sL, mu), _lam(sR, mu)
    lamTL, lamTR = lamL.sum(), lamR.sum()
    rffL = (lamL * rho).sum() / lamTL if lamTL > 0 else rho.mean()
    rffR = (lamR * rho).sum() / lamTR if lamTR > 0 else rho.mean()
    V_T = TF * dp - GC * 0.5 * (rffL + rffR)
    if V_T >= 0.0:                                       # upwind total mobility by sign(V_T)
        qT = lamTL * V_T
        f_up = lamL / lamTL if lamTL > 0 else np.zeros_like(lamL)
    else:
        qT = lamTR * V_T
        f_up = lamR / lamTR if lamTR > 0 else np.zeros_like(lamR)
    q = f_up * qT                                        # viscous fractional-flow split
    for a, b in pairs:
        if rho[a] == rho[b]:
            continue
        w = -GC * (rho[a] - rho[b])                      # inter-phase gravity direction
        th = masks[(a, b)]
        lam_with = np.where(w >= 0.0, (th * lamL).sum(), (th * lamR).sum())
        lam_against = np.where(-w >= 0.0, ((1 - th) * lamL).sum(), ((1 - th) * lamR).sum())
        lamT_pair = lam_with + lam_against
        la = lamL[a] if w >= 0.0 else lamR[a]
        lb = lamL[b] if -w >= 0.0 else lamR[b]
        b_ab = (la * lb / (lamT_pair + EPS)) * w
        q[a] += b_ab
        q[b] -= b_ab
    return q


def _solve_dp(qT_of_dp, u_T):
    """Scalar monotone root: dp such that qT(dp) = u_T (expanding bracket + bisection)."""
    lo, hi = -1.0, 1.0
    for _ in range(60):
        if qT_of_dp(lo) <= u_T <= qT_of_dp(hi):
            break
        lo *= 2.0
        hi *= 2.0
    for _ in range(80):                                  # bisection
        mid = 0.5 * (lo + hi)
        if qT_of_dp(mid) < u_T:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def evaluate(scheme, sL, sR, u_T, rho, mu, masks, pairs, T_L, T_R):
    """Return (phase_mass_fluxes[N], energy_flux, dp) for one cell pair at fixed total velocity."""
    if scheme == "ppu":
        dp = _solve_dp(lambda d: ppu_phase_fluxes(sL, sR, d, rho, mu).sum(), u_T)
        q = ppu_phase_fluxes(sL, sR, dp, rho, mu)
    else:
        dp = _solve_dp(lambda d: hu_phase_fluxes(sL, sR, d, rho, mu, masks, pairs).sum(), u_T)
        q = hu_phase_fluxes(sL, sR, dp, rho, mu, masks, pairs)
    rh_L, rh_R = rho * CP * T_L, rho * CP * T_R          # (rho h)_l, upwind by phase-flux sign
    rh_up = np.where(q >= 0.0, rh_L, rh_R)
    energy = float((rh_up * q).sum())
    return rho * q, energy, dp                           # component mass flux = rho_l * q_l


# --------------------------------------------------------------------------- grids
def grid_2phase(u_T, rho, mu, masks, pairs, T_L, T_R, ng=81):
    """Flux fields over (z_CH4^R, z_CH4^L) in [0,1]^2 (saturations via s(z)) for each scheme."""
    zc = np.linspace(0.0, 1.0, ng)                       # overall CH4 composition (light, last comp)
    ZL, ZR = np.meshgrid(zc, zc, indexing="ij")          # ZL varies along axis 0
    out = {sc: {"mass": np.zeros((2, ng, ng)), "energy": np.zeros((ng, ng)),
                "dp": np.zeros((ng, ng))} for sc in ("ppu", "hu")}
    for i in range(ng):
        for j in range(ng):
            sl = s_of_z(np.array([1.0 - zc[i], zc[i]]), rho)
            sr = s_of_z(np.array([1.0 - zc[j], zc[j]]), rho)
            for sc in ("ppu", "hu"):
                m, e, dp = evaluate(sc, sl, sr, u_T, rho, mu, masks, pairs, T_L, T_R)
                out[sc]["mass"][:, i, j] = m
                out[sc]["energy"][i, j] = e
                out[sc]["dp"][i, j] = dp
    return zc, ZL, ZR, out


def grid_3phase(u_T, rho, mu, masks, pairs, T_L, T_R, zR, ng=121):
    """Flux fields over the left-cell COMPOSITION triangle (z_H2O^L, z_C5H12^L), z_CH4^L = 1-z0-z1,
    right-cell composition zR fixed. Saturations via the immiscible flash s(z)."""
    z = np.linspace(0.0, 1.0, ng)
    Z0, Z1 = np.meshgrid(z, z, indexing="ij")            # z_H2O, z_C5H12
    inside = (Z0 + Z1) <= 1.0 + 1e-9
    sR = s_of_z(zR, rho)
    out = {sc: {"mass": np.full((3, ng, ng), np.nan), "energy": np.full((ng, ng), np.nan),
                "dp": np.full((ng, ng), np.nan)} for sc in ("ppu", "hu")}
    for i in range(ng):
        for j in range(ng):
            if not inside[i, j]:
                continue
            sl = s_of_z(np.array([z[i], z[j], 1.0 - z[i] - z[j]]), rho)
            for sc in ("ppu", "hu"):
                m, e, dp = evaluate(sc, sl, sR, u_T, rho, mu, masks, pairs, T_L, T_R)
                out[sc]["mass"][:, i, j] = m
                out[sc]["energy"][i, j] = e
                out[sc]["dp"][i, j] = dp
    return z, Z0, Z1, inside, out


# --------------------------------------------------------------------------- figures
def _cmap():
    try:
        import plot_reference as PR
        return PR._cmap("vlag")
    except Exception:
        try:
            import seaborn as sns
            return sns.color_palette("vlag", as_cmap=True)
        except Exception:
            import matplotlib.pyplot as plt
            return plt.get_cmap("coolwarm")


def _figure_path(rho, mu, masks, pairs, path, axis_base, comp_rows, title, stem,
                 u_T, T_L, T_R, out_dir, z_slice=0.3, ng=81, nz=500):
    """Generic N=2-style figure over a 1-D COMPOSITION PATH z=path(t), t in [0,1]: per flux a PPU
    map, an HU map (over (t_R,t_L), isolines + red PPU creases) and a 1-D slice (PPU solid vs HU
    dashed) at t_R=z_slice. ``comp_rows`` = [(component index, row title), ...] (energy added)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = _cmap()
    n = len(rho)

    t = np.linspace(0.0, 1.0, ng)
    TL, TR = np.meshgrid(t, t, indexing="ij")
    mass = {sc: np.zeros((n, ng, ng)) for sc in ("ppu", "hu")}
    energy = {sc: np.zeros((ng, ng)) for sc in ("ppu", "hu")}
    dpf = {sc: np.zeros((ng, ng)) for sc in ("ppu", "hu")}
    for i in range(ng):
        for j in range(ng):
            sL, sR = s_of_z(path(t[i]), rho), s_of_z(path(t[j]), rho)
            for sc in ("ppu", "hu"):
                m, e, d = evaluate(sc, sL, sR, u_T, rho, mu, masks, pairs, T_L, T_R)
                mass[sc][:, i, j], energy[sc][i, j], dpf[sc][i, j] = m, e, d

    tl = np.linspace(0.0, 1.0, nz)                       # slice at t_R = z_slice
    sR_s = s_of_z(path(z_slice), rho)
    sl = {sc: {"mass": np.zeros((n, nz)), "energy": np.zeros(nz)} for sc in ("ppu", "hu")}
    dp_ppu = np.zeros(nz)
    for i, tt in enumerate(tl):
        sLv = s_of_z(path(tt), rho)
        for sc in ("ppu", "hu"):
            m, e, d = evaluate(sc, sLv, sR_s, u_T, rho, mu, masks, pairs, T_L, T_R)
            sl[sc]["mass"][:, i], sl[sc]["energy"][i] = m, e
            if sc == "ppu":
                dp_ppu[i] = d
    levels = sorted(set(np.round(GC * rho, 6)))          # merged densities -> one crease level
    creases = [tl[k] for lvl in levels
               for k in np.where(np.diff(np.sign(dp_ppu - lvl)) != 0)[0]]

    rows = [("mass", idx, rt) for idx, rt in comp_rows] + [("energy", None, "energy flux")]
    xlab, ylab = f"${axis_base}^{{R}}$", f"${axis_base}^{{L}}$"
    fig, axes = plt.subplots(len(rows), 3, figsize=(12.8, 3.5 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]}, squeeze=False)
    for r, (fld, k, title_r) in enumerate(rows):
        fields = [mass[sc][k] if fld == "mass" else energy[sc] for sc in ("ppu", "hu")]
        vmax = max(float(np.max(np.abs(f))) for f in fields) or 1.0
        for c, sc in enumerate(("ppu", "hu")):
            ax = axes[r][c]
            pc = ax.pcolormesh(TR, TL, fields[c], cmap=cmap, vmin=-vmax, vmax=vmax,
                               shading="auto", rasterized=True)
            ax.contour(TR, TL, fields[c], levels=12, colors="0.35", linewidths=0.5)
            if sc == "ppu":
                ax.contour(TR, TL, dpf["ppu"], levels=levels, colors="#d62728", linewidths=1.4)
            ax.axvline(z_slice, color="0.15", ls=":", lw=1.0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel("%s\n(%s)" % (ylab, title_r))
            if r == 0:
                ax.set_title("PPU" if sc == "ppu" else "HU", fontsize=12)
            fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
        axs = axes[r][2]
        yp = sl["ppu"]["mass"][k] if fld == "mass" else sl["ppu"]["energy"]
        yh = sl["hu"]["mass"][k] if fld == "mass" else sl["hu"]["energy"]
        axs.plot(tl, yp, color="#1f4e9c", lw=2.2, label="PPU")
        axs.plot(tl, yh, color="#c0392b", lw=2.2, ls="--", label="HU")
        for zcr in creases:
            axs.axvline(zcr, color="#d62728", ls=":", lw=1.0)
        axs.set_xlim(0, 1)
        axs.set_xlabel(ylab)
        axs.set_ylabel(title_r)
        axs.grid(alpha=0.25)
        if r == 0:
            axs.set_title(f"slice at ${axis_base}^{{R}} = {z_slice:.2f}$", fontsize=11)
            axs.legend(fontsize=9, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0 - 0.3 / len(rows)])
    _save(fig, out_dir, stem)


def figure_2phase(u_T, rho, mu, masks, pairs, T_L, T_R, names, out_dir, z_slice=0.3, tag=""):
    rho_s = "[" + ", ".join(f"{x:g}" for x in rho) + "]"
    _figure_path(
        rho, mu, masks, pairs, lambda t: np.array([1.0 - t, t]),
        r"z_{\mathrm{CH_4}}", [(0, r"mass flux $\mathrm{H_2O}$"), (1, r"mass flux $\mathrm{CH_4}$")],
        f"Two-cell thermal-compositional flux  —  H$_2$O / CH$_4$   ($u_T={u_T:g}$,  "
        f"$\\rho={rho_s}$;  maps over $z$ via $s(z)$,  grey: isolines,  red: PPU "
        f"$\\Delta\\Phi_\\ell=0$;  right: PPU kinks, HU smooth)",
        "thermal_flux_surfaces_2phase" + tag, u_T, T_L, T_R, out_dir, z_slice)


def figure_reduce_demo(u_T, T_L, T_R, out_dir, z_slice=0.3, tag=""):
    """N=3 with TWO densities exactly equal (H2O 5, C5H12 1, CH4 1), shown in the N=2 style along
    the heavy-vs-merged-light composition path z_C5H12 = z_CH4 = z_light/2. Demonstrates the exact
    equal-density aggregation: the C5H12 and CH4 flux rows are identical and the two creases coincide,
    so N=3 collapses to a two-phase (H2O vs merged light) structure."""
    rho = np.array([5.0, 1.0, 1.0])
    mu = np.ones(3)
    pairs = tuple(itertools.combinations(range(3), 2))
    masks = _side_masks(rho, pairs)
    _figure_path(
        rho, mu, masks, pairs, lambda t: np.array([1.0 - t, 0.5 * t, 0.5 * t]),
        r"z_{\mathrm{C_5H_{12}}+\mathrm{CH_4}}",
        [(0, r"mass flux $\mathrm{H_2O}$"), (1, r"mass flux $\mathrm{C_5H_{12}}$"),
         (2, r"mass flux $\mathrm{CH_4}$")],
        f"N=3 with two EQUAL densities ($\\rho=[5, 1, 1]$) in the N=2 style   "
        f"($u_T={u_T:g}$;  path $z_{{\\mathrm{{C_5H_{{12}}}}}}=z_{{\\mathrm{{CH_4}}}}$;  "
        f"C$_5$H$_{{12}}$ & CH$_4$ fluxes coincide -> N=3 reduces to 2-phase)",
        "thermal_flux_surfaces_3phase_reduced" + tag, u_T, T_L, T_R, out_dir, z_slice)


def figure_3phase(u_T, rho, mu, masks, pairs, T_L, T_R, zR, names, out_dir, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, Z0, Z1, inside, out = grid_3phase(u_T, rho, mu, masks, pairs, T_L, T_R, zR)
    cmap = _cmap()
    rows = [("mass", k, r"mass flux $\mathrm{%s}$" % _TEX[names[k]]) for k in range(3)]
    rows += [("energy", None, "energy flux")]
    fig, axes = plt.subplots(len(rows), 2, figsize=(8.2, 3.3 * len(rows)))
    for r, (fld, k, title) in enumerate(rows):
        fields = [out[sc]["mass"][k] if fld == "mass" else out[sc]["energy"] for sc in ("ppu", "hu")]
        vmax = np.nanmax([np.nanmax(np.abs(f)) for f in fields])
        for c, sc in enumerate(("ppu", "hu")):
            ax = axes[r][c]
            Z = np.where(inside, fields[c], np.nan)
            pc = ax.pcolormesh(Z1, Z0, Z, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto",
                               rasterized=True)
            with np.errstate(invalid="ignore"):          # flux isolines: kink at PPU creases, smooth for HU
                ax.contour(Z1, Z0, Z, levels=9, colors="0.35", linewidths=0.5)
            if sc == "ppu":                               # 3 potential-reversal creases
                ax.contour(Z1, Z0, np.where(inside, out["ppu"]["dp"], np.nan),
                           levels=sorted(GC * rho), colors="#d62728", linewidths=1.4)
            ax.plot([0, 1, 0, 0], [1, 0, 0, 1], color="0.3", lw=1.0)   # triangle edges
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xlabel(_zlabel(names[1], "L"))
            if c == 0:
                ax.set_ylabel("%s   (%s)" % (_zlabel(names[0], "L"), title))
            if r == 0:
                ax.set_title("PPU" if sc == "ppu" else "HU", fontsize=12)
            fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    rho_s = "[" + ", ".join(f"{x:g}" for x in rho) + "]"
    zR_s = "[" + ", ".join(f"{x:.2f}" for x in zR) + "]"
    fig.suptitle(f"Two-cell thermal-compositional flux maps over the left-cell composition simplex "
                 f"(H$_2$O / C$_5$H$_{{12}}$ / CH$_4$)\n"
                 f"$z_R={zR_s}$,  $u_T={u_T:g}$,  $\\rho={rho_s}$   "
                 f"(axes: overall composition $z$ via $s(z)$;  grey: flux isolines;  "
                 f"red: PPU $\\Delta\\Phi_\\ell=0$ creases)", fontsize=10)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    _save(fig, out_dir, "thermal_flux_surfaces_3phase" + tag)


# --------------------------------------------------------------------------- helpers
def _save(fig, out_dir, stem):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print("wrote", os.path.relpath(p, HERE))
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    import sys
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nphase", type=int, default=0, choices=[0, 2, 3],
                    help="0 = both N=2 and N=3 (default); or just one")
    ap.add_argument("--u-t", type=float, default=1.0, help="fixed total velocity (default 1.0)")
    ap.add_argument("--dT", type=float, default=0.0,
                    help="temperature drop T_L-T_R across the face (default 0 = uniform T; "
                         "energy flux then inherits the scheme smoothness)")
    ap.add_argument("--rho", default=None,
                    help="override phase densities, comma-separated heavy->light (e.g. 5,1.1,1 to "
                         "push two of them close); default = the monotonicity component ratios")
    ap.add_argument("--tag", default="", help="suffix appended to the output filename")
    ap.add_argument("--reduce-demo", action="store_true",
                    help="also make the N=3-with-two-EQUAL-densities figure in the N=2 style "
                         "(equal-density aggregation / reduction to 2-phase)")
    ap.add_argument("--out", default=os.path.join(HERE, "figures"), help="output dir")
    args = ap.parse_args()
    T_L, T_R = 1.0 + 0.5 * args.dT, 1.0 - 0.5 * args.dT
    rho_override = np.array([float(x) for x in args.rho.split(",")]) if args.rho else None

    if args.reduce_demo:
        figure_reduce_demo(args.u_t, T_L, T_R, args.out, tag=args.tag)
        return

    for n in ([2, 3] if args.nphase == 0 else [args.nphase]):
        rho, mu, names = _phase_system(n)
        if rho_override is not None:
            if len(rho_override) != n:
                continue
            rho = rho_override
        pairs = tuple(itertools.combinations(range(n), 2))
        masks = _side_masks(rho, pairs)
        if n == 2:
            figure_2phase(args.u_t, rho, mu, masks, pairs, T_L, T_R, names, args.out, tag=args.tag)
        else:
            zR = np.full(3, 1.0 / 3.0)                    # right-cell overall composition
            figure_3phase(args.u_t, rho, mu, masks, pairs, T_L, T_R, zR, names, args.out, tag=args.tag)


if __name__ == "__main__":
    main()
