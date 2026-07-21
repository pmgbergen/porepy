"""Independent 2-D finite-volume solver for the Weis (2014) fig-8 heat-flux plume.

The 2-D port of ``../one_dimensional/weis_1d_solver.py`` (same (p, h) primaries, same
Driesner opensowat table closure, same HU/PPU flux forms, strict SI), on the 9 x 3 km
fig-8 domain with the condition-2 boundary conditions.  Its purpose is to make the
reference paper's discretization choices SWITCHABLE:

    --grav-upstream  gravity-term densities from the (lagged) upstream node
                     (Weis 2014 Eq. 25) instead of the arithmetic face average;
    --lag-upwind     upwind directions frozen at the previous time level
                     (Weis sec. 2.7, "the old velocity field defines the upwind nodes");
    --lag-props      ALL nonlinear flux coefficients (rho, mu, k_r, h_g, lambda) at the
                     previous time level -> fluxes linear in the current p (CSMP++'s
                     semi-implicit pressure equation, sec. 2.6-2.7).  Accumulation
                     stays implicit.

All three off = our sharp fully-implicit scheme (the porepy HU analog); all three on
(with --scheme ppu) = the closest monolithic analog of the CSMP++ discretization.

Output: ``data_2_*.vtu`` + ``data.pvd`` in ``visualization_<tag>/`` with porepy-named
cell fields, so ``fig_weis_2d_plume.py --scheme <token>`` renders the results
unchanged.  The scheme token encodes the switches: e.g. ``weis-hu``,
``weis-ppu-gu-ld-lp`` (gu = grav-upstream, ld = lag-upwind, lp = lag-props).
"""
from __future__ import annotations

import argparse
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
    G, K_PERM, PHI, K_E, RHO_S, C_S, Table, _XPH_FIELDS, eval_props, table_paths,
)

# --------------------------------------------------------------------------------------- #
#  fig-8 condition-2 data (SI)
# --------------------------------------------------------------------------------------- #
LX, LY = 9000.0, 3000.0       # domain [m]
P_TOP = 0.101325e6            # surface pressure [Pa] (atmospheric; table floor clips)
T_TOP = 283.15                # surface temperature [K]
Q_BACKGROUND = 0.05           # background bottom heat flux [W/m^2]
Q_ANOMALY = 5.0               # anomaly heat flux [W/m^2] over |x-4500| <= 500 m
X_SRC, HALF_SRC = 4500.0, 500.0
YEAR = 365.0 * 86400.0

DT_NOMINAL, DT_MIN, DT_MAX = 0.5, 0.001, 25.0        # [yr] (= porepy_2d_solver defaults)
_DEFAULT_SNAP_YEARS = tuple(float(y) for y in range(0, 50001, 2500))

RHO_REF, T_REF = 800.0, 500.0                        # residual row scales (as 1-D)
T_SCALE = 5.0 * YEAR      # FIXED residual time scale -- deliberately NOT DT_NOMINAL:
                          # coupling the row scales to a tunable dt silently rescales the
                          # convergence test (0.5 yr loosened it 10x and let a spurious
                          # frozen state pass; the 5.4 kW input hid at ~2 W/cell under tol)
TABLE_LEVEL = 3


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
    Vcell: float; ms: float; es: float
    xc: np.ndarray; yc: np.ndarray


def make_grid(cell: float, q_anomaly: float) -> Grid2D:
    nx, ny = round(LX / cell), round(LY / cell)
    d = LX / nx
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    cid = (ix + nx * iy).ravel()
    xc = (ix.ravel() + 0.5) * d; yc = (iy.ravel() + 0.5) * d
    # x-faces (L right-neighbor pairs), then y-faces (L upper-neighbor pairs)
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
    return Grid2D(nx=nx, ny=ny, d=d, ncell=nx * ny, fL=fL, fR=fR, Tf=Tf, GA=GA,
                  TFf=TFf, top=top, Tb=2.0 * K_PERM * A / d, GAb=K_PERM * A * G,
                  TFb=2.0 * K_E * A / d, bot=bot, q_bot=q, Vcell=Vcell, ms=ms, es=es,
                  xc=xc, yc=yc)


def _upwind(direction, fL, fR):
    return np.where(direction >= 0.0, fL, fR)


def _harmonic_face(cell_field, fL, fR):
    """Harmonic face average 2 aL aR / (aL + aR), zero where the sum vanishes."""
    aL = cell_field[fL]; aR = cell_field[fR]
    s = aL + aR
    return np.where(s > 0.0, 2.0 * aL * aR / np.where(s > 0.0, s, 1.0), 0.0)


# --------------------------------------------------------------------------------------- #
#  Lagged (old-state) face directions -- as weis_1d_solver.buoyancy_directions
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
    """Weis (2014) Eq. 26 advective CFL limit from the OLD state: dt <= cfl * min over
    faces/phases of phi * d * A / Q_g, with Q_g the volumetric phase flux across the
    face.  The lagged (semi-implicit) modes are only conditionally stable, exactly like
    CSMP++'s explicit advection -- running them beyond this dt oscillates even though
    the (nearly linear) Newton still converges."""
    p = x[0::2]; h = x[1::2]
    pr = eval_props(table, p, h)
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

    # Eq. 27 (mass-based criterion): the outflow of a phase during one step must not
    # exceed that phase's mass stored in the upstream cell -- "essential to avoid
    # oscillations in the highly-compressible two-phase region" (and empirically: the
    # h-floor drain-lock at the boiling-column top under --lag-props).
    F_l = Psi_l * pr.mm_l[iu_l]                        # phase MASS fluxes [kg/s]
    F_v = Psi_v * pr.mm_v[iu_v]
    out_l = np.zeros(grid.ncell); out_v = np.zeros(grid.ncell)
    np.add.at(out_l, iu_l, np.abs(F_l))                # tally on the upstream cell
    np.add.at(out_v, iu_v, np.abs(F_v))
    tc = grid.top                                      # top-boundary outflow counts too
    V_t = grid.Tb * (p[tc] - P_TOP) - grid.GAb * 1000.0
    outflow = V_t > 0.0
    np.add.at(out_l, tc[outflow], (V_t * pr.mm_l[tc])[outflow])
    np.add.at(out_v, tc[outflow], (V_t * pr.mm_v[tc])[outflow])
    store_l = PHI * pr.s_l * pr.rho_l * grid.Vcell
    store_v = PHI * pr.s_v * pr.rho_v * grid.Vcell
    # a phase with no outflow imposes NO constraint (0-storage/0-outflow cells --
    # e.g. the vapor phase in a cold liquid cell -- must not clamp dt to zero)
    lim_l = np.where(out_l > 1.0e-12, np.maximum(store_l, 0.0) / np.maximum(out_l, 1e-30),
                     np.inf)
    lim_v = np.where(out_v > 1.0e-12, np.maximum(store_v, 0.0) / np.maximum(out_v, 1e-30),
                     np.inf)
    lim27 = np.minimum(lim_l, lim_v)
    return cfl * float(min(lim26.min(), lim27.min()))


# --------------------------------------------------------------------------------------- #
#  Residual (2N, interleaved [mass_0, energy_0, ...])
# --------------------------------------------------------------------------------------- #
@dataclass
class Opts:
    scheme: str
    grav_upstream: bool
    lag_upwind: bool
    lag_props: bool


def residual(x, acc_m_o, acc_e_o, dt, grid, table, btop, opts, ug, ud, ut, pr_old,
             pr=None):
    fL, fR = grid.fL, grid.fR
    p = x[0::2]; h = x[1::2]
    if pr is None:
        pr = eval_props(table, p, h)
    prf = pr_old if opts.lag_props else pr            # coefficients in the FLUXES

    acc_m = grid.Vcell * PHI * pr.rho_mix
    acc_e = grid.Vcell * (PHI * (pr.rho_mix * h - p) + (1 - PHI) * RHO_S * C_S * pr.T)

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
    else:                                             # hu
        rho_ff_f = 0.5 * (prf.rho_ff[fL] + prf.rho_ff[fR])
        rho_ff_g = prf.rho_ff[ut] if opts.grav_upstream else rho_ff_f
        V_T = dp - grid.GA * rho_ff_g
        up = ut if (opts.lag_upwind or opts.lag_props) else _upwind(V_T, fL, fR)
        if opts.scheme == "hu-mw":
            # viscous term only: harmonic face average of the total mobility (the
            # lambda*K joint transmissibility, 1-D engine's weighted_perm / Remark 3.2);
            # smooth in the state -- no upwind switch in m_e. The advected enthalpy
            # per unit mass <hbar> stays upwinded.
            lam_face = _harmonic_face(prf.lam_T, fL, fR)
            F_mass = V_T * lam_face
            hbar_up = prf.adv_h[up] / np.where(prf.lam_T[up] > 0.0, prf.lam_T[up], 1.0)
            F_en_adv = hbar_up * F_mass
        else:
            F_mass = V_T * prf.lam_T[up]
            F_en_adv = V_T * prf.adv_h[up]
        w_flux = -grid.GA * (rho_l_f - rho_v_f)       # face-centered driving force
        lam_pair = prf.mm_l[ug] + prf.mm_v[ud]
        common = prf.mm_l[ug] * prf.mm_v[ud] / (lam_pair + 1.0e-30)
        F_en = F_four + F_en_adv + common * w_flux * (prf.h_l[ug] - prf.h_v[ud])

    div_m = np.zeros(grid.ncell); div_e = np.zeros(grid.ncell)
    np.add.at(div_m, fL, F_mass); np.add.at(div_m, fR, -F_mass)
    np.add.at(div_e, fL, F_en);   np.add.at(div_e, fR, -F_en)

    # ---- top boundary (Dirichlet p, T -> h_bc), directional advection; V out-positive --
    tc = grid.top
    prb = btop.pr
    V_t = grid.Tb * (p[tc] - btop.p) - grid.GAb * prb.rho_ff[0]
    out = V_t >= 0.0
    lam_t = np.where(out, prf.lam_T[tc], prb.lam_T[0])
    adv_t = np.where(out, prf.adv_h[tc], prb.adv_h[0])
    div_m[tc] += V_t * lam_t
    div_e[tc] += grid.TFb * (pr.T[tc] - btop.T) + V_t * adv_t

    # ---- bottom: no mass flux; prescribed heat INFLUX (integrated per cell) -----------
    div_e[grid.bot] -= grid.q_bot

    r = np.empty(2 * grid.ncell)
    r[0::2] = ((acc_m - acc_m_o) / dt + div_m) / grid.ms
    r[1::2] = ((acc_e - acc_e_o) / dt + div_e) / grid.es
    return r


@dataclass
class BoundaryState:
    p: float; h: float; pr: object; T: float


# --------------------------------------------------------------------------------------- #
#  Coloured FD Jacobian (5-point stencil, 2 vars -> 18 colours), SuperLU solve
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
        for v in range(2):
            cols, rows, owner = [], [], []
            for c in np.where(cell_color == cc)[0]:
                j = 2 * c + v
                cols.append(j)
                rr = [2 * k + e for k in nbr[c] for e in (0, 1)]
                rows.extend(rr); owner.extend([j] * len(rr))
            col_perturb.append(np.array(cols, dtype=np.intp))
            gat_rows.append(np.array(rows, dtype=np.intp))
            gat_owner.append(np.array(owner, dtype=np.intp))
    all_rows = np.concatenate(gat_rows); all_cols = np.concatenate(gat_owner)
    scale = np.where(np.arange(2 * nc) % 2 == 0, 1.0e6, 1.0e5)     # p ~ MPa, h ~ 1e5
    return dict(n=2 * nc, col_perturb=col_perturb, gat_rows=gat_rows,
                gat_owner=gat_owner, all_rows=all_rows, all_cols=all_cols, scale=scale)


def jacobian_fd(x, r0, args, plan, eps_rel=1e-7):
    """Coloured FD Jacobian, bit-identical to the naive 18-sweep version but sampling
    the property table ONLY at each colour's perturbed cells (~1/9 of the domain,
    batched into one table call); unperturbed cells reuse the base-state values,
    which a full re-evaluation would reproduce bitwise anyway."""
    import dataclasses
    table = args[4]
    eps = eps_rel * np.maximum(np.abs(x), plan["scale"])
    p0 = x[0::2].copy(); h0 = x[1::2].copy()
    pr0 = eval_props(table, p0, h0)
    fields = [f.name for f in dataclasses.fields(pr0)]

    # one batched table evaluation for ALL colours' perturbed cells
    cells_c, pp, hh = [], [], []
    for c in range(18):
        cols = plan["col_perturb"][c]
        cells = cols // 2
        pc = p0[cells].copy(); hc = h0[cells].copy()
        if c % 2 == 0:
            pc = pc + eps[cols]
        else:
            hc = hc + eps[cols]
        cells_c.append(cells); pp.append(pc); hh.append(hc)
    sizes = np.cumsum([0] + [len(c) for c in cells_c])
    pr_b = eval_props(table, np.concatenate(pp), np.concatenate(hh))

    parts = []
    for c in range(18):
        cols = plan["col_perturb"][c]
        cells = cells_c[c]
        sl = slice(sizes[c], sizes[c + 1])
        dx = np.zeros(plan["n"]); dx[cols] = eps[cols]
        saved = []
        for name in fields:                       # patch perturbed cells in place
            arr = getattr(pr0, name)
            saved.append(arr[cells].copy())
            arr[cells] = getattr(pr_b, name)[sl]
        dr = residual(x + dx, *args, pr=pr0) - r0
        for name, old_vals in zip(fields, saved):  # restore
            getattr(pr0, name)[cells] = old_vals
        parts.append(dr[plan["gat_rows"][c]] / eps[plan["gat_owner"][c]])
    A = sp.coo_matrix((np.concatenate(parts), (plan["all_rows"], plan["all_cols"])),
                      shape=(plan["n"], plan["n"])).tocsc()
    return A


# --------------------------------------------------------------------------------------- #
#  Newton + adaptive dt with exact schedule landing
# --------------------------------------------------------------------------------------- #
def newton_step(x0, x_old, dt, grid, table, btop, opts, plan, atol=1e-5, maxit=13):
    p_old = x_old[0::2]; h_old = x_old[1::2]
    pr_old = eval_props(table, p_old, h_old)
    ug, ud, ut = frozen_directions(grid, p_old, pr_old, opts.scheme)
    acc_m_o = grid.Vcell * PHI * pr_old.rho_mix
    acc_e_o = grid.Vcell * (PHI * (pr_old.rho_mix * h_old - p_old)
                            + (1 - PHI) * RHO_S * C_S * pr_old.T)
    args = (acc_m_o, acc_e_o, dt, grid, table, btop, opts, ug, ud, ut, pr_old)
    pclip = (table.b_min * (1 + 1e-9), table.b_max * (1 - 1e-9))
    hclip = (table.a_min * (1 + 1e-9), table.a_max * (1 - 1e-9))
    sqrtN = np.sqrt(grid.ncell)
    _metric = lambda rr: max(np.linalg.norm(rr[0::2]), np.linalg.norm(rr[1::2])) / sqrtN

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
            xn[0::2] = np.clip(xn[0::2], *pclip)
            xn[1::2] = np.clip(xn[1::2], *hclip)
            r_new = residual(xn, *args); nrm_new = np.linalg.norm(r_new)
            if nrm_new < nrm:
                break
            step *= 0.5
        x = xn; r = r_new; nrm = nrm_new
    return x, maxit, _metric(r), False


# --------------------------------------------------------------------------------------- #
#  Hydrostatic IC + export
# --------------------------------------------------------------------------------------- #
def hydrostatic_column(table, xpt, ny, d):
    """Cell-center hydrostatic p at uniform 10 degC, integrated with the table density."""
    yc = (np.arange(ny) + 0.5) * d
    depth = LY - yc
    p = P_TOP + 1000.0 * G * depth
    for _ in range(10):
        h = xpt("H", np.full(ny, T_TOP - 273.15), p)
        rho = eval_props(table, p, h).rho_mix
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
    p = x[0::2]; h = x[1::2]
    pr = eval_props(table, p, h)
    img = pv.ImageData(dimensions=(grid.nx + 1, grid.ny + 1, 1),
                       spacing=(grid.d, grid.d, 1.0), origin=(0.0, 0.0, 0.0))
    ug = img.cast_to_unstructured_grid()
    cd = ug.cell_data
    cd["pressure"] = p * 1e-6                     # MPa (porepy field names/units)
    cd["enthalpy"] = h * 1e-6                     # MJ/kg
    cd["temperature"] = pr.T
    cd["T_C"] = pr.T - 273.15
    cd["s_v"] = pr.s_v; cd["s_l"] = pr.s_l
    cd["rho"] = pr.rho_mix; cd["rho_l"] = pr.rho_l; cd["rho_v"] = pr.rho_v
    cd["h_l"] = pr.h_l * 1e-6; cd["h_v"] = pr.h_v * 1e-6
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


def run(scheme="hu", cell=100.0, q_anomaly=Q_ANOMALY, snap_years=_DEFAULT_SNAP_YEARS,
        dt_nom=DT_NOMINAL, dt_min=DT_MIN, dt_max=DT_MAX, grav_upstream=False,
        lag_upwind=False, lag_props=False, cfl=None, level=TABLE_LEVEL, folder=None,
        verbose=True):
    opts = Opts(scheme=scheme, grav_upstream=grav_upstream,
                lag_upwind=lag_upwind, lag_props=lag_props)
    xph_path, xpt_path = table_paths(level)
    table = Table(xph_path, _XPH_FIELDS, a_in=1e-6, b_in=1e-6)
    xpt = Table(xpt_path, {"H": 1e3}, a_in=1.0, b_in=1e-6)
    grid = make_grid(cell, q_anomaly)
    if verbose:
        print(f"  weis_2d: {grid.nx}x{grid.ny} cells, scheme={scheme}, "
              f"gu={grav_upstream} ld={lag_upwind} lp={lag_props}, "
              f"input {grid.q_bot.sum():.0f} W (grid-quantized inlet)")

    h_top = float(xpt("H", np.array([T_TOP - 273.15]), np.array([P_TOP]))[0])
    btop = BoundaryState(p=P_TOP, h=h_top,
                         pr=eval_props(table, np.array([P_TOP]), np.array([h_top])),
                         T=T_TOP)
    p_col = hydrostatic_column(table, xpt, grid.ny, grid.d)
    p0 = p_col[np.arange(grid.ncell) // grid.nx]
    h0 = xpt("H", np.full(grid.ncell, T_TOP - 273.15), p0)
    x = np.empty(2 * grid.ncell); x[0::2] = p0; x[1::2] = h0

    if folder is None:
        tag = case_tag(scheme_token(scheme, opts), cell_size=None if cell == 100.0 else cell,
                       q_anomaly=q_anomaly)
        folder = os.path.join(HERE, "visualization_" + tag)
    os.makedirs(folder, exist_ok=True)

    plan = build_jac_plan(grid)
    schedule = sorted(set(float(s) for s in snap_years))
    tf = schedule[-1] * YEAR
    entries = []
    export_vtu(folder, 0, grid, table, x); entries.append((0.0, "data_2_000000.vtu"))
    write_pvd(folder, entries)
    next_i = 1

    # The CFL limiter (Weis Eq. 26) is available in EVERY mode via --cfl; the lagged
    # (semi-implicit) modes are conditionally stable and get it by default (0.9).
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
                # Hard lock: dt_min steps that neither converge nor change the state.
                # Abort with a diagnostic snapshot instead of spinning forever.
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
    ap.add_argument("--grav-upstream", action="store_true",
                    help="Weis Eq. 25 upstream gravity densities")
    ap.add_argument("--lag-upwind", action="store_true",
                    help="upwind directions frozen at the old time level (Weis 2.7)")
    ap.add_argument("--lag-props", action="store_true",
                    help="flux coefficients at the old time level (semi-implicit, 2.6)")
    ap.add_argument("--cfl", type=float, default=None, metavar="C",
                    help="advective CFL dt limiter (Weis Eq. 26), available in ALL "
                         "modes; give a factor (e.g. 0.9) to enable. Default: off for "
                         "the fully-implicit modes, 0.9 for the lagged modes (which "
                         "are only conditionally stable and need it)")
    ap.add_argument("--snap-years", type=float, nargs="+",
                    default=list(_DEFAULT_SNAP_YEARS), metavar="YR")
    ap.add_argument("--dt-nominal", type=float, default=DT_NOMINAL, metavar="YR")
    ap.add_argument("--dt-min", type=float, default=DT_MIN, metavar="YR")
    ap.add_argument("--dt-max", type=float, default=DT_MAX, metavar="YR")
    a = ap.parse_args()
    run(scheme=a.scheme, cell=a.cell_size, q_anomaly=a.q_anomaly,
        snap_years=a.snap_years, dt_nom=a.dt_nominal, dt_min=a.dt_min,
        dt_max=a.dt_max, grav_upstream=a.grav_upstream, lag_upwind=a.lag_upwind,
        lag_props=a.lag_props, cfl=a.cfl)


if __name__ == "__main__":
    main()
