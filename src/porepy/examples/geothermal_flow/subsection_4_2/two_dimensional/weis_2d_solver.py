"""Independent 2-D finite-volume solver for the Weis (2014) fig-8 heat-flux plume.

Fractional-flow form of the weighted fluid formulation with THREE conservation laws
per cell -- total mass, NaCl component, energy -- and primaries (p [Pa], h [J/kg],
z [-]).  The component equation mirrors the fully discrete system of the theory:

    component flux    F_z = upw(c_z) * F_total - B_z,
    component buoyancy B_z = sum_pairs upw(x_za) upw(pair mobility) G[rho_b - rho_a],

assembled with EXACTLY the same upwind machinery as the energy equation: the same
pair directions (ug, ud), the same total-flux direction (ut), the same
mobility-product pair operator, with the advected weight (h_l - h_v) replaced by the
salt mass fractions (X_l - X_v).  Properties are sampled TRILINEARLY from the
Driesner opensowat tables over (z, h, p), full salinity range 0..1 with HALITE:
S_h is a third (immobile) saturation from the flash -- storage-only per Weis 2014
Eqs 2-4 (bulk rho/h/z carry it automatically with (p, h, z) primaries), with pore
blocking through the relperms  R_l = 0.3 (1 - S_h),  kr_l + kr_v = 1 - S_h.
z = 0 reproduces the previous two-equation solver on the z = 0 slice.

Reference-mimicking switches (unchanged):
    --grav-upstream  gravity-term densities from the (lagged) upstream node (Eq. 25);
    --lag-upwind     upwind directions frozen at the previous time level (sec. 2.7);
    --lag-props      ALL nonlinear flux coefficients at the previous time level.

Output: ``data_2_*.vtu`` + ``data.pvd`` in ``visualization_<tag>/`` with porepy-named
cell fields (now including z_NaCl, x_NaCl_liq, x_NaCl_gas), rendered unchanged by
``fig_weis_2d_plume.py --scheme <token>``; --z-init joins the tag when non-default.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "one_dimensional"))
sys.path.insert(0, HERE)
from case_naming import case_tag                                    # noqa: E402
from weis_1d_solver import (                                        # noqa: E402
    G, K_PERM, PHI, K_E, RHO_S, C_S, S_R_LIQ, table_paths,
)

# --------------------------------------------------------------------------------------- #
#  fig-8 condition-2 data (SI)
# --------------------------------------------------------------------------------------- #
LX, LY = 9000.0, 3000.0       # domain [m]
P_TOP = 0.101325e6            # surface pressure [Pa]
T_TOP = 283.15                # surface temperature [K]
Q_BACKGROUND = 0.05           # background bottom heat flux [W/m^2]
Q_ANOMALY = 5.0               # anomaly heat flux [W/m^2] over |x-4500| <= 500 m
Z_INIT = 0.0                  # initial/recharge NaCl overall composition (--z-init)
X_SRC, HALF_SRC = 4500.0, 500.0
YEAR = 365.0 * 86400.0

DT_NOMINAL, DT_MIN, DT_MAX = 0.5, 0.001, 25.0        # [yr]
_DEFAULT_SNAP_YEARS = tuple(float(y) for y in range(0, 50001, 2500))

RHO_REF, T_REF, Z_REF = 800.0, 500.0, 0.1            # residual row scales
T_SCALE = 5.0 * YEAR      # FIXED residual time scale -- deliberately NOT DT_NOMINAL:
                          # coupling the row scales to a tunable dt silently rescales the
                          # convergence test (0.5 yr loosened it 10x and let a spurious
                          # frozen state pass; the 5.4 kW input hid at ~2 W/cell under tol)
TABLE_LEVEL = 3
NV = 3                    # unknowns per cell: p, h, z


# --------------------------------------------------------------------------------------- #
#  Trilinear table over (z_NaCl, h|T, p) -- the 3-D extension of weis_1d_solver.Table
# --------------------------------------------------------------------------------------- #
class Table3:
    """O(1) vectorised TRILINEAR sampler of a Driesner VTK table over its full
    (z_NaCl, second-axis, p) box.  Solver inputs are SI; ``a_in``/``b_in`` convert the
    second axis (h [J/kg] or T [degC]) and the pressure to table units; the z axis is
    dimensionless.  Field values return in SI via per-field scales."""

    def __init__(self, file_name, fields, a_in=1.0, b_in=1.0):
        key = ("|".join(f"{k}:{v}" for k, v in sorted(fields.items()))
               + f"|{a_in}|{b_in}|3d|{os.path.getmtime(file_name):.0f}")
        cache = file_name + ".sicache3.npz"
        if not (os.path.exists(cache) and self._load(cache, key)):
            self._build(file_name, fields, a_in, b_in, key, cache)
        self.names = list(self.V.keys())
        self.V_stack = np.ascontiguousarray(np.stack([self.V[n] for n in self.names]))

    def _load(self, cache, key):
        try:
            zf = np.load(cache, allow_pickle=False)
            if str(zf["key"]) != key:
                return False
            for s in ("nc", "ny", "nz"):
                setattr(self, s, int(zf[s]))
            for s in ("c0", "dc", "a0", "da", "b0", "db", "a_in", "b_in",
                      "a_min", "a_max", "b_min", "b_max", "c_min", "c_max"):
                setattr(self, s, float(zf[s]))
            self.cax = np.asarray(zf["cax"], float)
            self.c_uniform = bool(zf["c_uniform"])
            self.V = {str(n): zf["V_" + str(n)] for n in zf["names"]}
            return True
        except Exception:
            return False

    def _build(self, file_name, fields, a_in, b_in, key, cache):
        import pyvista as pv
        g = pv.read(file_name)
        nc, ny, nz = g.dimensions                          # (z_comp, second, p)
        if hasattr(g, "x") and np.asarray(g.x).ndim == 1 and len(np.asarray(g.x)) == nc:
            c = np.asarray(g.x); a = np.asarray(g.y); b = np.asarray(g.z)
        else:
            # legacy StructuredGrid: extract the 1-D axes from the point coordinates
            # (VTK ordering: x fastest, then y, then z)
            pts = np.asarray(g.points)
            c = pts[:nc, 0]
            a = pts[::nc, 1][:ny]
            b = pts[::nc * ny, 2][:nz]
        self.nc, self.ny, self.nz = int(nc), int(ny), int(nz)
        self.c0, self.dc = float(c[0]), float(c[1] - c[0])
        self.a0, self.da = float(a[0]), float(a[1] - a[0])
        self.b0, self.db = float(b[0]), float(b[1] - b[0])
        self.cax = np.asarray(c, float)               # full axes: the z axis of the
        self.c_uniform = bool(np.allclose(np.diff(c), self.dc))   # regenerated tables
        #                                               is quasi-log, NOT uniform
        self.a_in, self.b_in = float(a_in), float(b_in)
        self.a_min, self.a_max = float(a[0] / a_in), float(a[-1] / a_in)
        self.b_min, self.b_max = float(b[0] / b_in), float(b[-1] / b_in)
        self.c_min, self.c_max = float(c[0]), float(c[-1])
        self.V = {name: np.asarray(g.point_data[name]).reshape(nz, ny, nc) * scale
                  for name, scale in fields.items()}
        out = {"key": np.array(key), "nc": self.nc, "ny": self.ny, "nz": self.nz,
               "c0": self.c0, "dc": self.dc, "a0": self.a0, "da": self.da,
               "b0": self.b0, "db": self.db, "a_in": self.a_in, "b_in": self.b_in,
               "a_min": self.a_min, "a_max": self.a_max, "b_min": self.b_min,
               "b_max": self.b_max, "c_min": self.c_min, "c_max": self.c_max,
               "cax": self.cax, "c_uniform": self.c_uniform,
               "names": np.array(list(self.V.keys()))}
        out.update({"V_" + n: A for n, A in self.V.items()})
        try:
            np.savez(cache, **out)
        except Exception:
            pass

    def sample_many(self, zc, a, b):
        """Trilinear interpolation of ALL stored fields at (z_comp, a, b)."""
        zc = np.atleast_1d(np.asarray(zc, float))
        a = np.atleast_1d(np.asarray(a, float)) * self.a_in
        b = np.atleast_1d(np.asarray(b, float)) * self.b_in
        if self.c_uniform:
            fc = ((zc - self.c0) / self.dc).clip(0.0, self.nc - 1 - 1e-9)
            jc = fc.astype(np.intp); tc = fc - jc
        else:                                          # non-uniform z axis: bracket by
            jc = (np.searchsorted(self.cax, zc, side="right") - 1)   # binary search
            jc = np.clip(jc, 0, self.nc - 2)
            tc = ((zc - self.cax[jc]) / (self.cax[jc + 1] - self.cax[jc])).clip(0.0, 1.0)
        fa = ((a - self.a0) / self.da).clip(0.0, self.ny - 1 - 1e-9)
        fb = ((b - self.b0) / self.db).clip(0.0, self.nz - 1 - 1e-9)
        ja = fa.astype(np.intp); jb = fb.astype(np.intp)
        ta = fa - ja; tb = fb - jb
        jc1 = jc + 1; ja1 = ja + 1; jb1 = jb + 1
        Vs = self.V_stack
        vals = ((1 - tc) * ((1 - ta) * (1 - tb) * Vs[:, jb, ja, jc]
                            + ta * (1 - tb) * Vs[:, jb, ja1, jc]
                            + (1 - ta) * tb * Vs[:, jb1, ja, jc]
                            + ta * tb * Vs[:, jb1, ja1, jc])
                + tc * ((1 - ta) * (1 - tb) * Vs[:, jb, ja, jc1]
                        + ta * (1 - tb) * Vs[:, jb, ja1, jc1]
                        + (1 - ta) * tb * Vs[:, jb1, ja, jc1]
                        + ta * tb * Vs[:, jb1, ja1, jc1]))
        return {n: vals[i] for i, n in enumerate(self.names)}

    def __call__(self, name, zc, a, b):
        return self.sample_many(zc, a, b)[name]


# Driesner opensowat vtr tables (via weis_1d_solver.table_paths), the accepted set:
# volumetric saturations, real Xl/Xv partitioning, h 0.0001..4.7 MJ/kg, p down to
# atmospheric.  Axis units are mapped by a_in/b_in; field scales bring values to SI.
_XPH_FIELDS = {"Rho_l": 1.0, "Rho_v": 1.0, "H_l": 1e3, "H_v": 1e3,
               "mu_l": 1.0, "mu_v": 1.0, "S_v": 1.0, "S_h": 1.0, "Rho": 1.0,
               "Temperature": 1.0, "Xl": 1.0, "Xv": 1.0}
_XPH_A_IN, _XPH_B_IN = 1e-6, 1e-6
_XPT_FIELDS = {"H": 1e3}
_XPT_A_IN, _XPT_B_IN = 1.0, 1e-6


# --------------------------------------------------------------------------------------- #
#  Constitutive closure (mirrors weis_1d_solver.eval_props + salt fractions)
# --------------------------------------------------------------------------------------- #
@dataclass
class Props3:
    rho_l: np.ndarray; rho_v: np.ndarray
    s_v: np.ndarray; s_l: np.ndarray
    h_l: np.ndarray; h_v: np.ndarray
    T: np.ndarray
    rho_mix: np.ndarray
    lam_T: np.ndarray
    f_l: np.ndarray; f_v: np.ndarray
    rho_ff: np.ndarray
    mm_l: np.ndarray; mm_v: np.ndarray
    adv_h: np.ndarray
    x_l: np.ndarray; x_v: np.ndarray       # NaCl mass fraction in liquid / vapor
    adv_z: np.ndarray                       # x_l mm_l + x_v mm_v (component advection)
    s_h: np.ndarray                         # halite saturation (immobile, X_h = 1)


def eval_props(table, z, p, h):
    s = table.sample_many(z, h, p)
    rho_l = s["Rho_l"]; rho_v = s["Rho_v"]
    s_v = np.clip(s["S_v"], 0.0, 1.0)
    s_h = np.clip(s["S_h"], 0.0, 1.0 - s_v)
    s_l = np.clip(1.0 - s_v - s_h, 0.0, 1.0)
    h_l = s["H_l"]; h_v = s["H_v"]
    mu_l = s["mu_l"]; mu_v = s["mu_v"]
    T = s["Temperature"]
    x_l = np.clip(s["Xl"], 0.0, 1.0)
    x_v = np.clip(s["Xv"], 0.0, 1.0)

    # Weis 2014 relperms: linear, R_l = S_R_LIQ (1 - S_h), R_v = 0, kr_l + kr_v =
    # 1 - S_h (halite blocks pore space); S_h = 0 reduces to the previous model
    pore = np.maximum(1.0 - s_h, 1.0e-12)
    kr_l = pore * np.clip((s_l / pore - S_R_LIQ) / (1.0 - S_R_LIQ), 0.0, 1.0)
    kr_v = pore - kr_l
    mm_l = rho_l * kr_l / mu_l
    mm_v = rho_v * kr_v / mu_v
    lam_T = mm_l + mm_v
    inv = 1.0 / np.where(lam_T > 0.0, lam_T, 1.0)
    f_l = mm_l * inv
    f_v = mm_v * inv
    rho_ff = f_l * rho_l + f_v * rho_v
    rho_mix = s["Rho"]          # bulk density incl. the halite contribution S_h rho_h
    adv_h = h_l * mm_l + h_v * mm_v
    adv_z = x_l * mm_l + x_v * mm_v
    return Props3(rho_l, rho_v, s_v, s_l, h_l, h_v, T, rho_mix, lam_T,
                  f_l, f_v, rho_ff, mm_l, mm_v, adv_h, x_l, x_v, adv_z, s_h=s_h)


# --------------------------------------------------------------------------------------- #
#  Grid: uniform cartesian, unified face list (x-faces have GA=0, y-faces carry gravity)
# --------------------------------------------------------------------------------------- #
@dataclass
class Grid2D:
    nx: int; ny: int; d: float; ncell: int
    fL: np.ndarray; fR: np.ndarray          # face cell pairs; flux positive L->R (right/up)
    Tf: np.ndarray; GA: np.ndarray; TFf: np.ndarray
    top: np.ndarray                          # cell index of each top-row cell
    Tb: float; GAb: float; TFb: float        # top half-face coefficients
    bot: np.ndarray; q_bot: np.ndarray       # bottom cells and their INTEGRATED influx [W]
    Vcell: float; ms: float; es: float; zs: float
    xc: np.ndarray; yc: np.ndarray


def make_grid(cell: float, q_anomaly: float) -> Grid2D:
    nx, ny = round(LX / cell), round(LY / cell)
    d = LX / nx
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    cid = (ix + nx * iy).ravel()
    xc = (ix.ravel() + 0.5) * d; yc = (iy.ravel() + 0.5) * d
    mask_x = (ix < nx - 1)
    fLx = cid[mask_x.ravel()]; fRx = fLx + 1
    mask_y = (iy < ny - 1)
    fLy = cid[mask_y.ravel()]; fRy = fLy + nx
    fL = np.concatenate([fLx, fLy]); fR = np.concatenate([fRx, fRy])
    A = d * 1.0                                             # unit thickness
    Tf = np.full(fL.size, K_PERM * A / d)
    GA = np.concatenate([np.zeros(fLx.size), np.full(fLy.size, K_PERM * A * G)])
    TFf = np.full(fL.size, K_E * A / d)
    top = cid[(iy == ny - 1).ravel()]
    bot = cid[(iy == 0).ravel()]
    q = np.where(np.abs(xc[bot] - X_SRC) <= HALF_SRC, q_anomaly, Q_BACKGROUND) * A
    Vcell = A * d
    ms = Vcell * PHI * RHO_REF / T_SCALE
    es = Vcell * (1 - PHI) * RHO_S * C_S * T_REF / T_SCALE
    zs = Vcell * PHI * RHO_REF * Z_REF / T_SCALE
    return Grid2D(nx=nx, ny=ny, d=d, ncell=nx * ny, fL=fL, fR=fR, Tf=Tf, GA=GA,
                  TFf=TFf, top=top, Tb=2.0 * K_PERM * A / d, GAb=K_PERM * A * G,
                  TFb=2.0 * K_E * A / d, bot=bot, q_bot=q, Vcell=Vcell, ms=ms, es=es,
                  zs=zs, xc=xc, yc=yc)


def _upwind(direction, fL, fR):
    return np.where(direction >= 0.0, fL, fR)


def _harmonic_face(cell_field, fL, fR):
    """Harmonic face average 2 aL aR / (aL + aR), zero where the sum vanishes."""
    aL = cell_field[fL]; aR = cell_field[fR]
    s = aL + aR
    return np.where(s > 0.0, 2.0 * aL * aR / np.where(s > 0.0, s, 1.0), 0.0)


# --------------------------------------------------------------------------------------- #
#  Lagged (old-state) face directions -- identical to the two-equation solver
# --------------------------------------------------------------------------------------- #
def frozen_directions(grid, p, pr, scheme):
    fL, fR = grid.fL, grid.fR
    rho_l_f = 0.5 * (pr.rho_l[fL] + pr.rho_l[fR])
    rho_v_f = 0.5 * (pr.rho_v[fL] + pr.rho_v[fR])
    rho_ff_f = 0.5 * (pr.rho_ff[fL] + pr.rho_ff[fR])
    dp = grid.Tf * (p[fL] - p[fR])
    i_tot = _upwind(dp - grid.GA * rho_ff_f, fL, fR)
    if scheme in ("hu", "hu-mw"):
        ddf = -grid.GA * (rho_l_f - rho_v_f)
        return _upwind(ddf, fL, fR), _upwind(-ddf, fL, fR), i_tot
    if scheme == "ppu":
        return (_upwind(dp - grid.GA * rho_l_f, fL, fR),
                _upwind(dp - grid.GA * rho_v_f, fL, fR), i_tot)
    raise ValueError(f"unknown scheme {scheme!r}")


def cfl_dt(grid, table, x, cfl):
    """Weis Eq. 26 advective CFL + Eq. 27 mass-based criterion from the OLD state.
    The salt rides the phase fluxes and adds no independent wave family, so the
    phase-based limits remain the governing ones."""
    p = x[0::NV]; h = x[1::NV]; z = x[2::NV]
    pr = eval_props(table, z, p, h)
    fL, fR = grid.fL, grid.fR
    dp = grid.Tf * (p[fL] - p[fR])
    rho_l_f = 0.5 * (pr.rho_l[fL] + pr.rho_l[fR])
    rho_v_f = 0.5 * (pr.rho_v[fL] + pr.rho_v[fR])
    Psi_l = dp - grid.GA * rho_l_f
    Psi_v = dp - grid.GA * rho_v_f
    iu_l = _upwind(Psi_l, fL, fR); iu_v = _upwind(Psi_v, fL, fR)
    Q_l = np.abs(Psi_l) * pr.mm_l[iu_l] / np.maximum(pr.rho_l[iu_l], 1.0)
    Q_v = np.abs(Psi_v) * pr.mm_v[iu_v] / np.maximum(pr.rho_v[iu_v], 1.0)
    lim26 = PHI * grid.d * grid.d / np.maximum(np.maximum(Q_l, Q_v), 1.0e-30)

    F_l = Psi_l * pr.mm_l[iu_l]                        # phase MASS fluxes [kg/s]
    F_v = Psi_v * pr.mm_v[iu_v]
    out_l = np.zeros(grid.ncell); out_v = np.zeros(grid.ncell)
    np.add.at(out_l, iu_l, np.abs(F_l))
    np.add.at(out_v, iu_v, np.abs(F_v))
    tc = grid.top
    V_t = grid.Tb * (p[tc] - P_TOP) - grid.GAb * 1000.0
    outflow = V_t > 0.0
    np.add.at(out_l, tc[outflow], (V_t * pr.mm_l[tc])[outflow])
    np.add.at(out_v, tc[outflow], (V_t * pr.mm_v[tc])[outflow])
    store_l = PHI * pr.s_l * pr.rho_l * grid.Vcell
    store_v = PHI * pr.s_v * pr.rho_v * grid.Vcell
    lim_l = np.where(out_l > 1.0e-12, np.maximum(store_l, 0.0) / np.maximum(out_l, 1e-30),
                     np.inf)
    lim_v = np.where(out_v > 1.0e-12, np.maximum(store_v, 0.0) / np.maximum(out_v, 1e-30),
                     np.inf)
    lim27 = np.minimum(lim_l, lim_v)
    return cfl * float(min(lim26.min(), lim27.min()))


# --------------------------------------------------------------------------------------- #
#  Residual (3N, interleaved [mass_0, energy_0, comp_0, mass_1, ...])
# --------------------------------------------------------------------------------------- #
@dataclass
class Opts:
    scheme: str
    grav_upstream: bool
    lag_upwind: bool
    lag_props: bool


def residual(x, acc_m_o, acc_e_o, acc_z_o, dt, grid, table, btop, opts, ug, ud, ut,
             pr_old, pr=None):
    fL, fR = grid.fL, grid.fR
    p = x[0::NV]; h = x[1::NV]; z = x[2::NV]
    if pr is None:
        pr = eval_props(table, z, p, h)
    prf = pr_old if opts.lag_props else pr            # coefficients in the FLUXES

    acc_m = grid.Vcell * PHI * pr.rho_mix
    acc_e = grid.Vcell * (PHI * (pr.rho_mix * h - p) + (1 - PHI) * RHO_S * C_S * pr.T)
    acc_z = grid.Vcell * PHI * pr.rho_mix * z

    dp = grid.Tf * (p[fL] - p[fR])
    rho_l_f = 0.5 * (prf.rho_l[fL] + prf.rho_l[fR])
    rho_v_f = 0.5 * (prf.rho_v[fL] + prf.rho_v[fR])
    F_four = grid.TFf * (pr.T[fL] - pr.T[fR])         # conduction stays implicit

    if opts.scheme == "ppu":
        if opts.lag_upwind or opts.lag_props:
            iu_l, iu_v = ug, ud
        else:
            iu_l = _upwind(dp - grid.GA * rho_l_f, fL, fR)
            iu_v = _upwind(dp - grid.GA * rho_v_f, fL, fR)
        rho_l_g = prf.rho_l[iu_l] if opts.grav_upstream else rho_l_f
        rho_v_g = prf.rho_v[iu_v] if opts.grav_upstream else rho_v_f
        Psi_l = dp - grid.GA * rho_l_g
        Psi_v = dp - grid.GA * rho_v_g
        F_mass = Psi_l * prf.mm_l[iu_l] + Psi_v * prf.mm_v[iu_v]
        F_en = (F_four + Psi_l * prf.h_l[iu_l] * prf.mm_l[iu_l]
                + Psi_v * prf.h_v[iu_v] * prf.mm_v[iu_v])
        F_z = (Psi_l * prf.x_l[iu_l] * prf.mm_l[iu_l]
               + Psi_v * prf.x_v[iu_v] * prf.mm_v[iu_v])
    else:                                             # hu / hu-mw
        rho_ff_f = 0.5 * (prf.rho_ff[fL] + prf.rho_ff[fR])
        rho_ff_g = prf.rho_ff[ut] if opts.grav_upstream else rho_ff_f
        V_T = dp - grid.GA * rho_ff_g
        up = ut if (opts.lag_upwind or opts.lag_props) else _upwind(V_T, fL, fR)
        if opts.scheme == "hu-mw":
            # viscous term only: harmonic face average of the total mobility; the
            # advected weights per unit mass (<hbar>, <zbar>) stay upwinded
            lam_face = _harmonic_face(prf.lam_T, fL, fR)
            F_mass = V_T * lam_face
            denom = np.where(prf.lam_T[up] > 0.0, prf.lam_T[up], 1.0)
            F_en_adv = (prf.adv_h[up] / denom) * F_mass
            F_z_adv = (prf.adv_z[up] / denom) * F_mass
        else:
            F_mass = V_T * prf.lam_T[up]
            F_en_adv = V_T * prf.adv_h[up]
            F_z_adv = V_T * prf.adv_z[up]
        # pairwise buoyancy: identical operator for energy and component, only the
        # advected weight differs ((h_l - h_v) vs (X_l - X_v))
        w_flux = -grid.GA * (rho_l_f - rho_v_f)       # face-centered driving force
        lam_pair = prf.mm_l[ug] + prf.mm_v[ud]
        common = prf.mm_l[ug] * prf.mm_v[ud] / (lam_pair + 1.0e-30)
        F_en = F_four + F_en_adv + common * w_flux * (prf.h_l[ug] - prf.h_v[ud])
        F_z = F_z_adv + common * w_flux * (prf.x_l[ug] - prf.x_v[ud])

    div_m = np.zeros(grid.ncell); div_e = np.zeros(grid.ncell)
    div_z = np.zeros(grid.ncell)
    np.add.at(div_m, fL, F_mass); np.add.at(div_m, fR, -F_mass)
    np.add.at(div_e, fL, F_en);   np.add.at(div_e, fR, -F_en)
    np.add.at(div_z, fL, F_z);    np.add.at(div_z, fR, -F_z)

    # ---- top boundary (Dirichlet p, T, z), directional advection; V out-positive ------
    tc = grid.top
    prb = btop.pr
    V_t = grid.Tb * (p[tc] - btop.p) - grid.GAb * prb.rho_ff[0]
    out = V_t >= 0.0
    lam_t = np.where(out, prf.lam_T[tc], prb.lam_T[0])
    adv_t = np.where(out, prf.adv_h[tc], prb.adv_h[0])
    advz_t = np.where(out, prf.adv_z[tc], prb.adv_z[0])
    div_m[tc] += V_t * lam_t
    div_e[tc] += grid.TFb * (pr.T[tc] - btop.T) + V_t * adv_t
    div_z[tc] += V_t * advz_t

    # ---- bottom: no mass/salt flux; prescribed heat INFLUX (integrated per cell) ------
    div_e[grid.bot] -= grid.q_bot

    r = np.empty(NV * grid.ncell)
    r[0::NV] = ((acc_m - acc_m_o) / dt + div_m) / grid.ms
    r[1::NV] = ((acc_e - acc_e_o) / dt + div_e) / grid.es
    r[2::NV] = ((acc_z - acc_z_o) / dt + div_z) / grid.zs
    return r


@dataclass
class BoundaryState:
    p: float; h: float; z: float; pr: object; T: float


# --------------------------------------------------------------------------------------- #
#  Coloured FD Jacobian (5-point stencil, 3 vars -> 27 colours), SuperLU solve
# --------------------------------------------------------------------------------------- #
def build_jac_plan(grid):
    nc, nx = grid.ncell, grid.nx
    nbr = [[c] for c in range(nc)]
    for L, R in zip(grid.fL, grid.fR):
        nbr[L].append(R); nbr[R].append(L)
    ix = np.arange(nc) % nx; iy = np.arange(nc) // nx
    cell_color = (ix % 3) + 3 * (iy % 3)
    col_perturb, gat_rows, gat_owner = [], [], []
    for cc in range(9):
        for v in range(NV):
            cols, rows, owner = [], [], []
            for c in np.where(cell_color == cc)[0]:
                j = NV * c + v
                cols.append(j)
                rr = [NV * k + e for k in nbr[c] for e in range(NV)]
                rows.extend(rr); owner.extend([j] * len(rr))
            col_perturb.append(np.array(cols, dtype=np.intp))
            gat_rows.append(np.array(rows, dtype=np.intp))
            gat_owner.append(np.array(owner, dtype=np.intp))
    all_rows = np.concatenate(gat_rows); all_cols = np.concatenate(gat_owner)
    vscale = np.array([1.0e6, 1.0e5, 1.0e-2])            # p ~ MPa, h ~ 1e5, z ~ 0.01
    scale = np.tile(vscale, nc)
    return dict(n=NV * nc, ncolor=9 * NV, col_perturb=col_perturb, gat_rows=gat_rows,
                gat_owner=gat_owner, all_rows=all_rows, all_cols=all_cols, scale=scale)


def jacobian_fd(x, r0, args, plan, eps_rel=1e-7):
    """Coloured FD Jacobian, bit-identical to the naive sweep but sampling the table
    ONLY at each colour's perturbed cells (batched into one trilinear call).
    Patch-consistent step: when a state sits within eps BELOW a table node, a forward
    difference samples the neighbouring trilinear patch and its slope kink poisons the
    Newton direction (observed as a dt-independent |r| plateau with a collapsing line
    search when a quasi-steady boiling front parks a cell enthalpy on an h-node); the
    perturbation flips sign there so the quotient stays in the patch of the base state."""
    table = args[5]
    eps = eps_rel * np.maximum(np.abs(x), plan["scale"])
    p0 = x[0::NV].copy(); h0 = x[1::NV].copy(); z0 = x[2::NV].copy()

    ep = eps[0::NV]; eh = eps[1::NV]; ez = eps[2::NV]
    fb = (p0 * table.b_in - table.b0) / table.db
    dup = (np.ceil(fb) - fb) * table.db / table.b_in
    eps[0::NV] = np.where((dup > 0.0) & (dup < ep) & (p0 - ep >= table.b_min), -ep, ep)
    fa = (h0 * table.a_in - table.a0) / table.da
    dup = (np.ceil(fa) - fa) * table.da / table.a_in
    eps[1::NV] = np.where((dup > 0.0) & (dup < eh) & (h0 - eh >= table.a_min), -eh, eh)
    jc = np.clip(np.searchsorted(table.cax, z0, side="right") - 1, 0, table.nc - 2)
    dup = table.cax[jc + 1] - z0
    eps[2::NV] = np.where((dup > 0.0) & (dup < ez) & (z0 - ez >= table.c_min), -ez, ez)
    pr0 = eval_props(table, z0, p0, h0)
    fields = [f.name for f in dataclasses.fields(pr0)]

    ncolor = plan["ncolor"]
    cells_c, pp, hh, zz = [], [], [], []
    for c in range(ncolor):
        cols = plan["col_perturb"][c]
        cells = cols // NV
        pc = p0[cells].copy(); hc = h0[cells].copy(); zc = z0[cells].copy()
        v = c % NV
        if v == 0:
            pc = pc + eps[cols]
        elif v == 1:
            hc = hc + eps[cols]
        else:
            zc = zc + eps[cols]
        cells_c.append(cells); pp.append(pc); hh.append(hc); zz.append(zc)
    sizes = np.cumsum([0] + [len(c) for c in cells_c])
    pr_b = eval_props(table, np.concatenate(zz), np.concatenate(pp), np.concatenate(hh))

    parts = []
    for c in range(ncolor):
        cols = plan["col_perturb"][c]
        cells = cells_c[c]
        sl = slice(sizes[c], sizes[c + 1])
        dx = np.zeros(plan["n"]); dx[cols] = eps[cols]
        saved = []
        for name in fields:
            arr = getattr(pr0, name)
            saved.append(arr[cells].copy())
            arr[cells] = getattr(pr_b, name)[sl]
        dr = residual(x + dx, *args, pr=pr0) - r0
        for name, old_vals in zip(fields, saved):
            getattr(pr0, name)[cells] = old_vals
        parts.append(dr[plan["gat_rows"][c]] / eps[plan["gat_owner"][c]])
    A = sp.coo_matrix((np.concatenate(parts), (plan["all_rows"], plan["all_cols"])),
                      shape=(plan["n"], plan["n"])).tocsc()
    return A


# --------------------------------------------------------------------------------------- #
#  Newton + adaptive dt with exact schedule landing
# --------------------------------------------------------------------------------------- #
def newton_step(x0, x_old, dt, grid, table, btop, opts, plan, atol=1.0e-5, maxit=17):
    p_old = x_old[0::NV]; h_old = x_old[1::NV]; z_old = x_old[2::NV]
    pr_old = eval_props(table, z_old, p_old, h_old)
    ug, ud, ut = frozen_directions(grid, p_old, pr_old, opts.scheme)
    acc_m_o = grid.Vcell * PHI * pr_old.rho_mix
    acc_e_o = grid.Vcell * (PHI * (pr_old.rho_mix * h_old - p_old)
                            + (1 - PHI) * RHO_S * C_S * pr_old.T)
    acc_z_o = grid.Vcell * PHI * pr_old.rho_mix * z_old
    args = (acc_m_o, acc_e_o, acc_z_o, dt, grid, table, btop, opts, ug, ud, ut, pr_old)
    pclip = (table.b_min * (1 + 1e-9), table.b_max * (1 - 1e-9))
    hclip = (table.a_min * (1 + 1e-9), table.a_max * (1 - 1e-9))
    zclip = (table.c_min, table.c_max * (1 - 1e-9))
    sqrtN = np.sqrt(grid.ncell)
    _metric = lambda rr: max(np.linalg.norm(rr[0::NV]), np.linalg.norm(rr[1::NV]),
                             np.linalg.norm(rr[2::NV])) / sqrtN

    x = x0.copy()
    r = residual(x, *args)
    nrm = np.linalg.norm(r)
    for it in range(maxit):
        m = _metric(r)
        if m <= atol:
            return x, it, m, True
        A = jacobian_fd(x, r, args, plan)
        try:
            dx = spla.splu(A, permc_spec="MMD_AT_PLUS_A").solve(-r)
        except Exception:
            dx = np.zeros_like(r)
        step = 1.0
        for _ in range(15):
            xn = x + step * dx
            xn[0::NV] = np.clip(xn[0::NV], *pclip)
            xn[1::NV] = np.clip(xn[1::NV], *hclip)
            xn[2::NV] = np.clip(xn[2::NV], *zclip)
            r_new = residual(xn, *args); nrm_new = np.linalg.norm(r_new)
            if nrm_new < nrm:
                break
            step *= 0.5
        x = xn; r = r_new; nrm = nrm_new
    return x, maxit, _metric(r), False


# --------------------------------------------------------------------------------------- #
#  Hydrostatic IC + export
# --------------------------------------------------------------------------------------- #
def hydrostatic_column(table, xpt, ny, d, z_init):
    """Cell-center hydrostatic p at uniform 10 degC and composition z_init."""
    yc = (np.arange(ny) + 0.5) * d
    depth = LY - yc
    p = P_TOP + 1000.0 * G * depth
    zz = np.full(ny, z_init)
    for _ in range(10):
        h = xpt("H", zz, np.full(ny, T_TOP - 273.15), p)
        rho = eval_props(table, zz, p, h).rho_mix
        p_new = np.empty(ny)
        p_new[-1] = P_TOP + rho[-1] * G * (d / 2.0)          # top cell: half-cell column
        for j in range(ny - 2, -1, -1):
            p_new[j] = p_new[j + 1] + 0.5 * (rho[j] + rho[j + 1]) * G * d
        if np.max(np.abs(p_new - p)) < 1.0:
            p = p_new; break
        p = p_new
    return p


def export_vtu(folder, k, grid, table, x):
    import pyvista as pv
    p = x[0::NV]; h = x[1::NV]; z = x[2::NV]
    pr = eval_props(table, z, p, h)
    img = pv.ImageData(dimensions=(grid.nx + 1, grid.ny + 1, 1),
                       spacing=(grid.d, grid.d, 1.0), origin=(0.0, 0.0, 0.0))
    ug = img.cast_to_unstructured_grid()
    cd = ug.cell_data
    cd["pressure"] = p * 1e-6                     # MPa (porepy field names/units)
    cd["enthalpy"] = h * 1e-6                     # MJ/kg
    cd["temperature"] = pr.T
    cd["T_C"] = pr.T - 273.15
    cd["s_v"] = pr.s_v; cd["s_l"] = pr.s_l; cd["s_h"] = pr.s_h
    cd["rho"] = pr.rho_mix; cd["rho_l"] = pr.rho_l; cd["rho_v"] = pr.rho_v
    cd["h_l"] = pr.h_l * 1e-6; cd["h_v"] = pr.h_v * 1e-6
    cd["z_NaCl"] = z
    cd["x_NaCl_liq"] = pr.x_l; cd["x_NaCl_gas"] = pr.x_v
    ug.save(os.path.join(folder, f"data_2_{k:06d}.vtu"))


def write_pvd(folder, entries):
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1">', "<Collection>"]
    lines += [f'<DataSet group="" part="" timestep="{t:.6f}" file="{f}"/>'
              for t, f in entries]
    lines += ["</Collection>", "</VTKFile>"]
    with open(os.path.join(folder, "data.pvd"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------------------- #
def scheme_token(scheme, opts):
    tok = f"weis-{scheme}"
    tok += "-gu" if opts.grav_upstream else ""
    tok += "-ld" if opts.lag_upwind else ""
    tok += "-lp" if opts.lag_props else ""
    return tok


def run(scheme="hu", cell=100.0, q_anomaly=Q_ANOMALY, z_init=Z_INIT,
        snap_years=_DEFAULT_SNAP_YEARS, dt_nom=DT_NOMINAL, dt_min=DT_MIN,
        dt_max=DT_MAX, grav_upstream=False, lag_upwind=False, lag_props=False,
        cfl=None, level=TABLE_LEVEL, folder=None, verbose=True):
    opts = Opts(scheme=scheme, grav_upstream=grav_upstream,
                lag_upwind=lag_upwind, lag_props=lag_props)
    xph_path, xpt_path = table_paths(level)[:2]
    table = Table3(xph_path, _XPH_FIELDS, a_in=_XPH_A_IN, b_in=_XPH_B_IN)
    xpt = Table3(xpt_path, _XPT_FIELDS, a_in=_XPT_A_IN, b_in=_XPT_B_IN)
    if verbose:
        print(f"  tables: h [{table.a_min/1e6:g}, {table.a_max/1e6:g}] MJ/kg, "
              f"p [{table.b_min/1e6:g}, {table.b_max/1e6:g}] MPa, "
              f"z [{table.c_min:g}, {table.c_max:g}]")
    if P_TOP < table.b_min:
        print(f"  WARNING: P_TOP = {P_TOP/1e6:g} MPa is below the table pressure floor "
              f"{table.b_min/1e6:g} MPa -- near-surface states will clamp; consider "
              "raising P_TOP for this table set", flush=True)
    if z_init < table.c_min:
        z_init = table.c_min           # clamp below the table z floor (if any)
    if z_init > table.c_max:
        raise SystemExit(f"--z-init {z_init} above the table range [.., {table.c_max}]")
    grid = make_grid(cell, q_anomaly)
    if verbose:
        print(f"  weis_2d: {grid.nx}x{grid.ny} cells, scheme={scheme}, "
              f"gu={grav_upstream} ld={lag_upwind} lp={lag_props}, z_init={z_init:g}, "
              f"input {grid.q_bot.sum():.0f} W (grid-quantized inlet)")

    h_top = float(xpt("H", np.array([z_init]), np.array([T_TOP - 273.15]),
                      np.array([P_TOP]))[0])
    btop = BoundaryState(p=P_TOP, h=h_top, z=z_init,
                         pr=eval_props(table, np.array([z_init]), np.array([P_TOP]),
                                       np.array([h_top])),
                         T=T_TOP)
    p_col = hydrostatic_column(table, xpt, grid.ny, grid.d, z_init)
    p0 = p_col[np.arange(grid.ncell) // grid.nx]
    h0 = xpt("H", np.full(grid.ncell, z_init), np.full(grid.ncell, T_TOP - 273.15), p0)
    x = np.empty(NV * grid.ncell)
    x[0::NV] = p0; x[1::NV] = h0; x[2::NV] = z_init

    if folder is None:
        tag = case_tag(scheme_token(scheme, opts),
                       cell_size=None if cell == 100.0 else cell,
                       q_anomaly=q_anomaly, z_init=z_init)
        folder = os.path.join(HERE, "visualization_" + tag)
    os.makedirs(folder, exist_ok=True)

    plan = build_jac_plan(grid)
    schedule = sorted(set(float(s) for s in snap_years))
    tf = schedule[-1] * YEAR
    entries = []
    export_vtu(folder, 0, grid, table, x); entries.append((0.0, "data_2_000000.vtu"))
    write_pvd(folder, entries)
    next_i = 1

    cfl_val = cfl if cfl is not None else (
        0.9 if (opts.lag_props or opts.lag_upwind) else None)
    if verbose and cfl_val is not None:
        print(f"  advective CFL limiter ON (factor {cfl_val})")
    t = 0.0; dt = dt_nom * YEAR; step = 0; nits = 0; n_rej = 0; n_stuck = 0
    while t < tf - 1.0:
        t_target = schedule[next_i] * YEAR
        dt = min(dt, dt_max * YEAR, t_target - t)
        if cfl_val is not None:
            dt = max(min(dt, cfl_dt(grid, table, x, cfl_val)), dt_min * YEAR)
        x_old = x.copy()
        xn, nit, nrm, ok = newton_step(x, x_old, dt, grid, table, btop, opts, plan)
        if not ok and dt > dt_min * YEAR:
            n_rej += 1
            if verbose:
                print(f"    reject @ t={t / YEAR:.1f} yr: |r|={nrm:.2e} after {nit} its, "
                      f"dt {dt / YEAR:.3f} -> {max(0.5 * dt, dt_min * YEAR) / YEAR:.3f} yr",
                      flush=True)
            dt = max(0.5 * dt, dt_min * YEAR)
            continue
        if not ok:
            n_stuck += 1
            if verbose:
                print(f"    WARNING: accepting non-converged step at dt_min "
                      f"(t={t / YEAR:.1f} yr, |r|={nrm:.2e})", flush=True)
            if n_stuck >= 25:
                export_vtu(folder, 999, grid, table, x)
                entries.append((t, "data_2_000999.vtu"))
                write_pvd(folder, entries)
                raise RuntimeError(
                    f"solver locked at t={t / YEAR:.1f} yr: {n_stuck} consecutive "
                    f"non-converged dt_min steps (|r|={nrm:.2e}); state exported as "
                    "data_2_000999.vtu for diagnosis")
        else:
            n_stuck = 0
        x = xn; t += dt; step += 1; nits += nit
        if verbose and step % 200 == 0:
            print(f"    ... t={t / YEAR:8.1f} yr  step={step}  dt={dt / YEAR:.3f} yr  "
                  f"rejects={n_rej}", flush=True)
        if ok and nit < 5:
            dt = min(dt * 1.5, dt_max * YEAR)
        if abs(t - t_target) < 1.0:
            export_vtu(folder, next_i, grid, table, x)
            entries.append((t, f"data_2_{next_i:06d}.vtu"))
            write_pvd(folder, entries)
            if verbose:
                print(f"  t={t / YEAR:8.0f} yr  step={step}  <nit>={nits / max(step, 1):.2f}")
            next_i += 1
            if next_i >= len(schedule):
                break
    if verbose:
        print(f"  done: {step} steps, {nits} Newton its total -> {folder}")
    return folder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheme", default="hu", choices=["hu", "hu-mw", "ppu"],
                    help="hu = upwinded total mobility; hu-mw = harmonic total mobility\n"
                         "in the viscous term (buoyancy stays mobility-product); ppu")
    ap.add_argument("--cell-size", type=float, default=100.0, metavar="M")
    ap.add_argument("--q-anomaly", type=float, default=Q_ANOMALY, metavar="W/M2",
                    help="inlet heat flux over the central 1 km [W/m^2]; "
                         f"default {Q_ANOMALY} (fig 8), 9 = fig 10A")
    ap.add_argument("--z-init", type=float, default=Z_INIT, metavar="Z",
                    help="initial (uniform) and top-recharge NaCl overall composition; "
                         f"default {Z_INIT}")
    ap.add_argument("--grav-upstream", action="store_true",
                    help="Weis Eq. 25 upstream gravity densities")
    ap.add_argument("--lag-upwind", action="store_true",
                    help="upwind directions frozen at the old time level (Weis 2.7)")
    ap.add_argument("--lag-props", action="store_true",
                    help="flux coefficients at the old time level (semi-implicit, 2.6)")
    ap.add_argument("--cfl", type=float, default=None, metavar="C",
                    help="advective CFL dt limiter (Weis Eq. 26+27), available in ALL "
                         "modes; give a factor (e.g. 0.9) to enable. Default: off for "
                         "the fully-implicit modes, 0.9 for the lagged modes (which "
                         "are only conditionally stable and need it)")
    ap.add_argument("--snap-years", type=float, nargs="+",
                    default=list(_DEFAULT_SNAP_YEARS), metavar="YR")
    ap.add_argument("--dt-nominal", type=float, default=DT_NOMINAL, metavar="YR")
    ap.add_argument("--dt-min", type=float, default=DT_MIN, metavar="YR")
    ap.add_argument("--dt-max", type=float, default=DT_MAX, metavar="YR")
    a = ap.parse_args()
    run(scheme=a.scheme, cell=a.cell_size, q_anomaly=a.q_anomaly, z_init=a.z_init,
        snap_years=a.snap_years, dt_nom=a.dt_nominal, dt_min=a.dt_min,
        dt_max=a.dt_max, grav_upstream=a.grav_upstream, lag_upwind=a.lag_upwind,
        lag_props=a.lag_props, cfl=a.cfl)


if __name__ == "__main__":
    main()
