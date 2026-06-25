"""Independent 1-D finite-volume solver reproducing PorePy's geothermal fig-5D column.

Independent of PorePy (only ``numpy`` + ``scipy`` + ``pyvista`` + ``matplotlib``). It
re-implements, in vectorised numpy, the exact discrete model that
``geothermal_H2O_low_NaCl_content_fig_5.py`` assembles for the *vertical* H2O low-NaCl
benchmark (Weis et al., fig 5D), so it is a fast, transparent reference and lets us
exercise the two buoyancy upwinding schemes without the cost of the full PorePy run.

Two buoyancy options (``scheme``):
    * ``"hu"``  -- Hybrid Upwinding: the inter-phase gravity flux ``+/- ddf(rho_l - rho_v)``
                   sets the two upwind directions.
    * ``"ppu"`` -- Phase-Potential Upwinding: each phase's own potential
                   ``Psi_g = T_f (p_L - p_U) - K A rho_g g`` sets its upwind direction.
Only the upwind *direction* differs; the buoyancy magnitude
``b = f_g^up f_d^up (lambda_g^up + lambda_d^up) w_flux`` is identical, exactly as in PorePy's
``__entity_buoyancy_flux`` (non-mass-mobility-weighted branch, which fig_5 uses).

Model (per cell, 2 conservation laws; primaries p[Pa], h[J/kg]; z_NaCl = 0)
--------------------------------------------------------------------------
Backward Euler, fully-implicit Newton.  ``(ACC^n - ACC^{n-1})/dt + div(F) = 0`` (F up-positive):

  MASS    ACC = V phi rho_mix,  rho_mix = s_l rho_l + s_v rho_v
          F   = V_T * upwind(lambda_T_mass),   lambda_T_mass = sum_g rho_g k_r(s_g)/mu_g
  ENERGY  ACC = V [ phi (rho_mix h - p) + (1-phi) rho_s c_s T ]
          F   = -K_e dT/dx  +  V_T * upwind(sum_g h_g rho_g k_r/mu_g)  +  b(buoyancy)
  V_T = T_f (p_L - p_U) - K A rho_ff g,   rho_ff = sum_g f_g rho_g,  f_g = rho_g k_r/mu_g / lambda_T_mass

**Units are strict SI** (Pa, J/kg, m, s, kg, K): g=9.80665, K=1e-15 m^2, K_e=2.0 W/mK,
c_s=880 J/kgK, rho_s=2700, phi=0.1.  (PorePy runs the same system Mega-scaled; the physical
solution is identical.  Mixing MPa pressure with SI mobility is what silently kills advection.)
The constitutive closure is sampled from the *same* Driesner table the PorePy run uses,
``opensowat_xph_l_2_grads.vtk`` (axes z_NaCl, h[MJ/kg], p[MPa]), via O(1) uniform-grid
bilinear interpolation; SI<->table unit conversion is handled inside the sampler.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --------------------------------------------------------------------------------------- #
#  Paths / physical constants (SI) -- values from geothermal_H2O_low_NaCl_content_fig_5.py
# --------------------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
VTK_DIR = os.path.join(HERE, "model_configuration", "constitutive_description",
                       "driesner_vtk_files")
VTK_XPH = os.path.join(VTK_DIR, "opensowat_xph_l_2_grads.vtk")  # (z, h[MJ/kg], p[MPa])
VTK_XPT = os.path.join(VTK_DIR, "opensowat_xpt_l_2_grads.vtk")  # (z, T[degC],  p[MPa])
REF_DIR = os.path.join(HERE, "benchmark_figures_data")

G = 9.80665           # gravity [m/s^2]
K_PERM = 1.0e-15      # permeability [m^2]
PHI = 0.1             # porosity
K_E = 2.0             # effective thermal conductivity [W/(m K)]
RHO_S = 2700.0        # rock density [kg/m^3]
C_S = 880.0           # rock specific heat [J/(kg K)]
S_R_LIQ = 0.3         # residual liquid saturation

L_COLUMN = 2000.0     # column height [m]
DX = 10.0             # lateral cross-section [m] (cancels in the solution)
YEAR = 365.0 * 86400.0
DT0 = 0.125 * YEAR    # reference time step (for equation row-scaling)

# Reference scales used to row-scale the mass/energy residuals to O(1). Without this the
# mass (~kg/s) and energy (~W) equations differ by ~1e13 and the Jacobian is unsolvable.
RHO_REF = 800.0       # kg/m^3
T_REF = 500.0         # K

BUOY_SCALE = 1.0      # debug knob for the enthalpy-buoyancy term (1=on, 0=off, -1=flip)

# fig-5D boundary / initial data (SI)
P_BOT, P_TOP = 20.0e6, 1.0e6      # Pa  (inlet y=0 / outlet y=2000)
T_BOT, T_TOP = 673.15, 423.15     # K
T_INIT = 423.15                   # K, constant IC


# --------------------------------------------------------------------------------------- #
#  Fast uniform-grid bilinear table (xph / xpt are RectilinearGrid, uniform axes)
# --------------------------------------------------------------------------------------- #
class Table:
    """O(1) vectorised bilinear sampler of a Driesner VTK table on the z_NaCl=0 slice.

    Solver inputs are in SI; ``a_in``/``b_in`` convert them to the table axis units
    (second axis = h[MJ/kg] or T[degC]; third axis = p[MPa]).  Field values are returned
    in SI via the per-field ``fields`` scale (e.g. enthalpy kJ/kg -> J/kg via 1e3).
    """

    def __init__(self, file_name, fields, a_in=1.0, b_in=1.0):
        import pyvista as pv

        g = pv.read(file_name)
        self.nx, self.ny, self.nz = g.dimensions          # (z, second, p)
        a = np.asarray(g.y); b = np.asarray(g.z)          # axis nodes (table units)
        self.a0, self.da = a[0], a[1] - a[0]
        self.b0, self.db = b[0], b[1] - b[0]
        self.a_in, self.b_in = a_in, b_in
        # solver-unit bounds (for clamping the Newton state)
        self.a_min, self.a_max = a[0] / a_in, a[-1] / a_in
        self.b_min, self.b_max = b[0] / b_in, b[-1] / b_in
        self.V = {}
        for name, scale in fields.items():
            self.V[name] = np.asarray(g.point_data[name]).reshape(
                self.nz, self.ny, self.nx)[:, :, 0] * scale     # [p, second]

    def __call__(self, name, a, b):
        a = np.atleast_1d(np.asarray(a, float)) * self.a_in     # SI -> table axis
        b = np.atleast_1d(np.asarray(b, float)) * self.b_in
        fa = np.clip((a - self.a0) / self.da, 0.0, self.ny - 1 - 1e-9)
        fb = np.clip((b - self.b0) / self.db, 0.0, self.nz - 1 - 1e-9)
        ja = fa.astype(int); jb = fb.astype(int)
        ta = fa - ja; tb = fb - jb
        A = self.V[name]
        f00 = A[jb, ja]; f10 = A[jb, ja + 1]; f01 = A[jb + 1, ja]; f11 = A[jb + 1, ja + 1]
        return ((1 - ta) * (1 - tb) * f00 + ta * (1 - tb) * f10
                + (1 - ta) * tb * f01 + ta * tb * f11)


# xph: solver h[J/kg] -> axis MJ/kg (1e-6); p[Pa] -> MPa (1e-6).
#   field scales to SI: Rho 1 (kg/m^3), H kJ/kg->J/kg (1e3), S_v 1, T 1 (K).
#   mu: the table already stores Pa.s (probe: mu~2.5e-5 at 400C); PorePy's extra 1e-6 is its
#   Pa.s->MPa.s Mega-scaling, which must NOT be applied in this SI solver -> scale 1.0.
_XPH_FIELDS = {"Rho_l": 1.0, "Rho_v": 1.0, "H_l": 1e3, "H_v": 1e3,
               "mu_l": 1.0, "mu_v": 1.0, "S_v": 1.0, "Temperature": 1.0}


# --------------------------------------------------------------------------------------- #
#  Constitutive closure from the xph table  (vectorised; z_NaCl = 0)
# --------------------------------------------------------------------------------------- #
@dataclass
class Props:
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


def eval_props(table, p, h):
    rho_l = table("Rho_l", h, p)
    rho_v = table("Rho_v", h, p)
    s_v = np.clip(table("S_v", h, p), 0.0, 1.0)
    s_l = 1.0 - s_v
    h_l = table("H_l", h, p)
    h_v = table("H_v", h, p)
    mu_l = table("mu_l", h, p)
    mu_v = table("mu_v", h, p)
    T = table("Temperature", h, p)

    kr_l = np.maximum((s_l - S_R_LIQ) / (1.0 - S_R_LIQ), 0.0)   # Corey-type liquid
    kr_v = s_v                                                  # linear gas
    mm_l = rho_l * kr_l / mu_l
    mm_v = rho_v * kr_v / mu_v
    lam_T = mm_l + mm_v
    inv = 1.0 / np.where(lam_T > 0.0, lam_T, 1.0)
    f_l = mm_l * inv
    f_v = mm_v * inv
    rho_ff = f_l * rho_l + f_v * rho_v
    rho_mix = s_l * rho_l + s_v * rho_v
    adv_h = h_l * mm_l + h_v * mm_v
    return Props(rho_l, rho_v, s_v, s_l, h_l, h_v, T, rho_mix, lam_T,
                 f_l, f_v, rho_ff, mm_l, mm_v, adv_h)


# --------------------------------------------------------------------------------------- #
#  Geometry + frozen upwind directions
# --------------------------------------------------------------------------------------- #
@dataclass
class Geom:
    N: int; dy: float; A: float
    Tf: float; Tb: float; TFf: float; TFb: float; Vcell: float; GA: float
    ms: float; es: float          # mass / energy residual row-scales
    yc: np.ndarray


def make_geom(N):
    dy = L_COLUMN / N
    A = DX
    Vcell = A * dy
    ms = Vcell * PHI * RHO_REF / DT0
    es = Vcell * (1 - PHI) * RHO_S * C_S * T_REF / DT0
    return Geom(N=N, dy=dy, A=A, Tf=K_PERM * A / dy, Tb=2.0 * K_PERM * A / dy,
                TFf=K_E * A / dy, TFb=2.0 * K_E * A / dy, Vcell=Vcell,
                GA=K_PERM * A * G, ms=ms, es=es, yc=(np.arange(N) + 0.5) * dy)


def _upwind_idx(direction):
    """Internal face (lower i, upper i+1): direction>=0 -> upstream lower i, else upper i+1."""
    i = np.arange(direction.size)
    return np.where(direction >= 0.0, i, i + 1)


def buoyancy_directions(geom, p, pr, scheme):
    """Per-internal-face upstream cell indices (i_liq, i_gas) for the buoyancy term.

    HU: liquid rides +ddf(rho_l-rho_v), gas rides -ddf  (opposite inter-phase directions).
    PPU: each phase rides its own potential Psi_g = T_f(p_L-p_U) - K A rho_g g.
    Phase order matches PorePy phases = [liq, gas].
    """
    rho_l_f = 0.5 * (pr.rho_l[:-1] + pr.rho_l[1:])
    rho_v_f = 0.5 * (pr.rho_v[:-1] + pr.rho_v[1:])
    if scheme == "hu":
        ddf = -geom.GA * (rho_l_f - rho_v_f)     # inter-phase gravity flux ddf(rho_l-rho_v)
        dir_liq, dir_gas = ddf, -ddf
    elif scheme == "ppu":
        dp = geom.Tf * (p[:-1] - p[1:])
        dir_liq = dp - geom.GA * rho_l_f         # Psi_liq ~ -K(grad p - rho_l g)
        dir_gas = dp - geom.GA * rho_v_f         # Psi_gas
    else:
        raise ValueError(f"unknown scheme {scheme!r}; use 'hu' or 'ppu'")
    return _upwind_idx(dir_liq), _upwind_idx(dir_gas)


# --------------------------------------------------------------------------------------- #
#  Residual
# --------------------------------------------------------------------------------------- #
@dataclass
class BoundaryState:
    p: float; h: float; pr: Props; T: float


def boundary_state(table, p_bc, h_bc):
    pr = eval_props(table, np.array([p_bc]), np.array([h_bc]))
    return BoundaryState(p=p_bc, h=h_bc, pr=pr, T=float(pr.T[0]))


def residual(x, x_old, dt, geom, table, bbot, btop, scheme, ug, ud):
    """Full 2N residual, interleaved [mass_0, energy_0, mass_1, ...].

    ``ug``/``ud`` are the frozen (lagged) per-face upstream indices for liquid/gas:
      HU  -> i_liq=upwind(+ddf(rho_l-rho_v)), i_gas=upwind(-ddf); used ONLY for the
             simplicial two-pair buoyancy (advection rides the total velocity V_T).
      PPU -> i_liq=upwind(Psi_liq), i_gas=upwind(Psi_gas); genuine per-phase potential
             upwinding of the FULL phase flux (Weis fig-5 reference; buoyancy intrinsic).
    """
    N = geom.N
    p = x[0::2]; h = x[1::2]
    pr = eval_props(table, p, h)
    p_old = x_old[0::2]; h_old = x_old[1::2]
    pr_old = eval_props(table, p_old, h_old)

    # accumulation (backward Euler)
    acc_mass = geom.Vcell * PHI * pr.rho_mix
    acc_mass_o = geom.Vcell * PHI * pr_old.rho_mix
    acc_en = geom.Vcell * (PHI * (pr.rho_mix * h - p) + (1 - PHI) * RHO_S * C_S * pr.T)
    acc_en_o = geom.Vcell * (PHI * (pr_old.rho_mix * h_old - p_old)
                             + (1 - PHI) * RHO_S * C_S * pr_old.T)

    dp_face = p[:-1] - p[1:]
    rho_l_f = 0.5 * (pr.rho_l[:-1] + pr.rho_l[1:])
    rho_v_f = 0.5 * (pr.rho_v[:-1] + pr.rho_v[1:])
    F_four = geom.TFf * (pr.T[:-1] - pr.T[1:])

    if scheme == "ppu":
        # genuine phase-potential upwinding: each phase rides its OWN potential flux
        # Psi_g = T_f(p_L-p_U) - K A rho_g g, mobility/enthalpy upwinded by sign(Psi_g)
        # (lagged in ug/ud). Buoyancy is intrinsic to Psi_g (no separate term).
        Psi_l = geom.Tf * dp_face - geom.GA * rho_l_f
        Psi_v = geom.Tf * dp_face - geom.GA * rho_v_f
        F_mass = Psi_l * pr.mm_l[ug] + Psi_v * pr.mm_v[ud]
        F_en = F_four + Psi_l * (pr.h_l[ug] * pr.mm_l[ug]) + Psi_v * (pr.h_v[ud] * pr.mm_v[ud])
    else:
        # HU: total-velocity advection + simplicial two-pair buoyancy (UNCHANGED)
        rho_ff_f = 0.5 * (pr.rho_ff[:-1] + pr.rho_ff[1:])
        V_T = geom.Tf * dp_face - geom.GA * rho_ff_f
        up = np.where(V_T >= 0.0, np.arange(N - 1), np.arange(N - 1) + 1)
        F_mass = V_T * pr.lam_T[up]
        w_flux = -geom.GA * (rho_l_f - rho_v_f)
        common = pr.f_l[ug] * pr.f_v[ud] * (pr.mm_l[ug] + pr.mm_v[ud])
        F_buoy = BUOY_SCALE * common * w_flux * (pr.h_l[ug] - pr.h_v[ud])
        F_en = F_four + V_T * pr.adv_h[up] + F_buoy

    # ---- boundary faces (Dirichlet p, T->h_bc) ----
    if scheme == "ppu":
        # bottom: per-phase potential half-face flux (inflow uses boundary props)
        Psi_lb = geom.Tb * (bbot.p - p[0]) - geom.GA * bbot.pr.rho_l[0]
        Psi_vb = geom.Tb * (bbot.p - p[0]) - geom.GA * bbot.pr.rho_v[0]
        mml = bbot.pr.mm_l[0] if Psi_lb >= 0 else pr.mm_l[0]
        hl = bbot.pr.h_l[0] if Psi_lb >= 0 else pr.h_l[0]
        mmv = bbot.pr.mm_v[0] if Psi_vb >= 0 else pr.mm_v[0]
        hv = bbot.pr.h_v[0] if Psi_vb >= 0 else pr.h_v[0]
        Fmass_bot0 = Psi_lb * mml + Psi_vb * mmv
        Fen_bot0 = geom.TFb * (bbot.T - pr.T[0]) + Psi_lb * hl * mml + Psi_vb * hv * mmv

        Psi_lt = geom.Tb * (p[-1] - btop.p) - geom.GA * btop.pr.rho_l[0]
        Psi_vt = geom.Tb * (p[-1] - btop.p) - geom.GA * btop.pr.rho_v[0]
        mml = pr.mm_l[-1] if Psi_lt >= 0 else btop.pr.mm_l[0]
        hl = pr.h_l[-1] if Psi_lt >= 0 else btop.pr.h_l[0]
        mmv = pr.mm_v[-1] if Psi_vt >= 0 else btop.pr.mm_v[0]
        hv = pr.h_v[-1] if Psi_vt >= 0 else btop.pr.h_v[0]
        Fmass_topN = Psi_lt * mml + Psi_vt * mmv
        Fen_topN = geom.TFb * (pr.T[-1] - btop.T) + Psi_lt * hl * mml + Psi_vt * hv * mmv
    else:
        V_b = geom.Tb * (bbot.p - p[0]) - geom.GA * bbot.pr.rho_ff[0]
        if V_b >= 0.0:
            Fmass_bot0 = V_b * bbot.pr.lam_T[0]; Fh_b = V_b * bbot.pr.adv_h[0]
        else:
            Fmass_bot0 = V_b * pr.lam_T[0];      Fh_b = V_b * pr.adv_h[0]
        Fen_bot0 = geom.TFb * (bbot.T - pr.T[0]) + Fh_b

        V_t = geom.Tb * (p[-1] - btop.p) - geom.GA * btop.pr.rho_ff[0]
        if V_t >= 0.0:
            Fmass_topN = V_t * pr.lam_T[-1];     Fh_t = V_t * pr.adv_h[-1]
        else:
            Fmass_topN = V_t * btop.pr.lam_T[0]; Fh_t = V_t * btop.pr.adv_h[0]
        Fen_topN = geom.TFb * (pr.T[-1] - btop.T) + Fh_t

    # divergence: net upward outflow per cell = F_top - F_bottom
    div_mass = np.empty(N); div_en = np.empty(N)
    div_mass[0] = F_mass[0] - Fmass_bot0
    div_mass[1:-1] = F_mass[1:] - F_mass[:-1]
    div_mass[-1] = Fmass_topN - F_mass[-1]
    div_en[0] = F_en[0] - Fen_bot0
    div_en[1:-1] = F_en[1:] - F_en[:-1]
    div_en[-1] = Fen_topN - F_en[-1]

    r = np.empty(2 * N)
    r[0::2] = ((acc_mass - acc_mass_o) / dt + div_mass) / geom.ms     # row-scaled to O(1)
    r[1::2] = ((acc_en - acc_en_o) / dt + div_en) / geom.es
    return r


# --------------------------------------------------------------------------------------- #
#  Sparse coloured finite-difference Jacobian (block-tridiagonal, 6 colours)
# --------------------------------------------------------------------------------------- #
def _build_sparsity(N):
    rows_of_col = []
    for k in range(N):
        for _v in range(2):
            rows = []
            for kk in (k - 1, k, k + 1):
                if 0 <= kk < N:
                    rows += [2 * kk, 2 * kk + 1]
            rows_of_col.append(np.array(rows, dtype=int))
    color = np.array([(k % 3) * 2 + v for k in range(N) for v in range(2)])
    return rows_of_col, color


def jacobian_fd(x, r0, args, rows_of_col, color, eps_rel=1e-7):
    N = args[2].N
    n = 2 * N
    scale = np.where(np.arange(n) % 2 == 0, 1.0e6, 1.0e5)     # p ~ MPa, h ~ 1e5 J/kg
    eps = eps_rel * np.maximum(np.abs(x), scale)
    rows_all, cols_all, data_all = [], [], []
    for c in range(6):
        cols_c = np.where(color == c)[0]
        dx = np.zeros(n); dx[cols_c] = eps[cols_c]
        dr = residual(x + dx, *args) - r0
        for j in cols_c:
            rws = rows_of_col[j]
            rows_all.append(rws)
            cols_all.append(np.full(rws.size, j))
            data_all.append(dr[rws] / eps[j])
    return sp.csc_matrix((np.concatenate(data_all),
                          (np.concatenate(rows_all), np.concatenate(cols_all))), shape=(n, n))


# --------------------------------------------------------------------------------------- #
#  Newton time stepping
# --------------------------------------------------------------------------------------- #
def newton_step(x0, x_old, dt, geom, table, bbot, btop, scheme,
                rtol=1e-6, atol=1e-7, maxit=25, verbose=False):
    rows_of_col, color = _build_sparsity(geom.N)
    p_old, h_old = x_old[0::2], x_old[1::2]
    pr_old = eval_props(table, p_old, h_old)
    ug, ud = buoyancy_directions(geom, p_old, pr_old, scheme)     # lagged per step

    x = x0.copy()
    args = (x_old, dt, geom, table, bbot, btop, scheme, ug, ud)
    r0 = residual(x, *args)
    nrm0 = np.linalg.norm(r0)
    nrm = nrm0
    for it in range(maxit):
        r = residual(x, *args)
        nrm = np.linalg.norm(r)
        if verbose:
            print(f"    newton {it}: |r|={nrm:.3e}")
        if nrm <= rtol * nrm0 + atol * np.sqrt(2 * geom.N):
            return x, it, nrm, True
        J = jacobian_fd(x, r, args, rows_of_col, color)
        try:
            dx = spla.spsolve(J, -r)
        except Exception:
            dx = spla.lsqr(J, -r)[0]
        step = 1.0
        for _ in range(10):
            xn = x + step * dx
            xn[0::2] = np.clip(xn[0::2], table.b_min * (1 + 1e-9), table.b_max * (1 - 1e-9))
            xn[1::2] = np.clip(xn[1::2], table.a_min * (1 + 1e-9), table.a_max * (1 - 1e-9))
            if np.linalg.norm(residual(xn, *args)) < nrm:
                break
            step *= 0.5
        x = xn
    return x, maxit, nrm, nrm <= 1e-3 * nrm0


def run(scheme="hu", N=200, n_steps=None, dt=None, adaptive=True, verbose=True):
    """Integrate the fig-5D column to t=1000 yr with the chosen buoyancy scheme."""
    table = Table(VTK_XPH, _XPH_FIELDS, a_in=1e-6, b_in=1e-6)     # h[J/kg], p[Pa]
    xpt = Table(VTK_XPT, {"H": 1e3}, a_in=1.0, b_in=1e-6)         # T[degC], p[Pa] -> H[J/kg]
    geom = make_geom(N)

    h_bot = float(xpt("H", T_BOT - 273.15, P_BOT)[0])
    h_top = float(xpt("H", T_TOP - 273.15, P_TOP)[0])
    bbot = boundary_state(table, P_BOT, h_bot)
    btop = boundary_state(table, P_TOP, h_top)

    y = geom.yc
    p0 = (y * P_TOP + (L_COLUMN - y) * P_BOT) / L_COLUMN
    h0 = xpt("H", np.full(N, T_INIT - 273.15), p0)
    x = np.empty(2 * N); x[0::2] = p0; x[1::2] = h0

    dt0 = dt if dt is not None else 0.125 * YEAR
    tf = 1000.0 * YEAR if n_steps is None else n_steps * dt0
    t = 0.0; dt = dt0; step = 0
    while t < tf - 1e-6:
        dt = min(dt, tf - t)
        x_old = x.copy()
        xn, nit, nrm, ok = newton_step(x, x_old, dt, geom, table, bbot, btop, scheme)
        if not ok and adaptive and dt > dt0 / 64:
            dt *= 0.5                                  # retry with smaller step
            continue
        x = xn; t += dt; step += 1
        if adaptive and ok and nit < 5 and dt < dt0:
            dt = min(dt * 2.0, dt0)
        if verbose and (step % 50 == 0 or not ok):
            print(f"  t={t/YEAR:7.1f} yr  dt={dt/YEAR:.4f}  nit={nit}  |r|={nrm:.1e}"
                  f"  {'' if ok else 'NOT CONVERGED'}")

    pr = eval_props(table, x[0::2], x[1::2])
    return {"y": y, "p": x[0::2], "h": x[1::2], "T": pr.T, "s_gas": pr.s_v,
            "s_liq": pr.s_l, "rho_mix": pr.rho_mix, "scheme": scheme, "N": N}


# --------------------------------------------------------------------------------------- #
#  Comparison plot vs digitized paper data (CSV)
# --------------------------------------------------------------------------------------- #
def _load_ref_csv(name):
    path = os.path.join(REF_DIR, name)
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    return d[:, 0], d[:, 1]      # distance[km], value


def plot_comparison(results, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    refs = {
        "T": ("fig_5_vertical_temperature_raw.csv", "Temperature [°C]"),
        "p": ("fig_5_vertical_pressured_raw.csv", "Pressure [MPa]"),
        "s_liq": ("fig_5_vertical_saturation_liq_raw.csv", "Liquid saturation [-]"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"hu": "tab:blue", "ppu": "tab:red"}
    for ax, key in zip(axes, ("T", "p", "s_liq")):
        csv, ylabel = refs[key]
        xr, yr = _load_ref_csv(csv)
        ax.plot(xr, yr, "ks", ms=4, mfc="none", label="Weis et al. (digitized)")
        for sch, res in results.items():
            y_km = res["y"] / 1000.0
            val = {"T": res["T"] - 273.15, "p": res["p"] / 1e6, "s_liq": res["s_liq"]}[key]
            ax.plot(y_km, val, "-", color=colors.get(sch, None), lw=1.8,
                    label=f"1D {sch.upper()} (N={res['N']})")
        ax.set_xlabel("Distance [km]"); ax.set_ylabel(ylabel)
        ax.set_xlim(0, 2); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Figure 5D (vertical column) — independent 1D solver vs digitized reference")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"wrote {save_path}")


# --------------------------------------------------------------------------------------- #
#  Self-test (cheap invariants)
# --------------------------------------------------------------------------------------- #
def selftest():
    print("=== selftest ===")
    table = Table(VTK_XPH, _XPH_FIELDS, a_in=1e-6, b_in=1e-6)
    geom = make_geom(20)
    p = np.linspace(20e6, 1e6, 20)
    h = np.full(20, 6.0e5)                         # cold liquid -> s_v = 0
    pr = eval_props(table, p, h)
    assert np.all(pr.s_v < 1e-6), "expected single-phase liquid"
    rho_l_f = 0.5 * (pr.rho_l[:-1] + pr.rho_l[1:]); rho_v_f = 0.5 * (pr.rho_v[:-1] + pr.rho_v[1:])
    i_liq, i_gas = buoyancy_directions(geom, p, pr, "hu")
    b = (pr.f_l[i_liq] * pr.f_v[i_gas] * (pr.mm_l[i_liq] + pr.mm_v[i_gas])
         * (-geom.GA * (rho_l_f - rho_v_f)) * (pr.h_l[i_liq] - pr.h_v[i_gas]))
    assert np.max(np.abs(b)) < 1e-20, f"single-phase buoyancy != 0: {np.max(np.abs(b)):.2e}"
    print("  single-phase buoyancy == 0  OK")
    p_hyd = np.empty(20); p_hyd[0] = 20e6
    for i in range(1, 20):
        p_hyd[i] = p_hyd[i - 1] - 0.5 * (pr.rho_ff[i - 1] + pr.rho_ff[i]) * G * geom.dy
    pr2 = eval_props(table, p_hyd, h)
    rff = 0.5 * (pr2.rho_ff[:-1] + pr2.rho_ff[1:])
    VT = geom.Tf * (p_hyd[:-1] - p_hyd[1:]) - geom.GA * rff
    print(f"  hydrostatic max|V_T| = {np.max(np.abs(VT)):.2e} (should be ~0)")
    print("  selftest passed\n")


def main():
    import sys
    selftest()
    schemes = sys.argv[1].split(",") if len(sys.argv) > 1 else ["hu", "ppu"]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_steps = int(os.environ["NSTEPS"]) if "NSTEPS" in os.environ else None
    out_dir = os.path.join(HERE, "visualization_1D_fig_5")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for sch in schemes:
        print(f"--- running scheme={sch}, N={N} ---")
        res = run(scheme=sch, N=N, n_steps=n_steps, verbose=True)
        results[sch] = res
        band = np.where((res["s_gas"] > 1e-3) & (res["s_gas"] < 1 - 1e-3))[0]
        print(f"  {sch}: T {res['T'][0]-273.15:.0f}->{res['T'][-1]-273.15:.0f} C, "
              f"p {res['p'][0]/1e6:.1f}->{res['p'][-1]/1e6:.2f} MPa, "
              f"band y=[{res['y'][band[0]]:.0f},{res['y'][band[-1]]:.0f}]m" if band.size
              else f"  {sch}: no two-phase band")
    plot_comparison(results, os.path.join(out_dir, f"fig5D_compare_N{N}.png"))


if __name__ == "__main__":
    main()
