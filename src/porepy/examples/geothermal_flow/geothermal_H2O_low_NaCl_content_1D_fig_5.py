"""Independent 1-D finite-volume solver reproducing PorePy's geothermal fig-5D column.

This script is **independent of PorePy** (only ``numpy`` + ``scipy`` + ``pyvista``). It
re-implements, in vectorised numpy, the exact discrete model that
``geothermal_H2O_low_NaCl_content_fig_5.py`` assembles for the *vertical* H2O low-NaCl
benchmark (Weis et al., fig 5D), so it can be used as a fast, transparent reference and to
exercise the two buoyancy upwinding schemes without the cost of the full PorePy run.

Two buoyancy options are provided, selected by ``scheme``:
    * ``"hu"``  -- Hybrid Upwinding: the inter-phase gravity flux ``+/- ddf(rho_l - rho_v)``
                   sets the two upwind directions.
    * ``"ppu"`` -- Phase-Potential Upwinding: each phase's own potential
                   ``Psi_g = T_f (p_L - p_U) - K A rho_g g`` sets its upwind direction.
Only the *upwind direction* differs; the buoyancy magnitude is identical
(``b = f_g^up f_d^up (lambda_g^up + lambda_d^up) w_flux``), exactly as in PorePy's
``__entity_buoyancy_flux`` (non-mass-mobility-weighted branch, which fig_5 uses).

Model (per cell, 2 conservation laws; primaries p[MPa], h[MJ/kg]; z_NaCl = 0)
---------------------------------------------------------------------------
Backward Euler, fully implicit Newton. With ``ACC`` the cell storage and ``F`` the face flux
(positive upward), each balance is ``(ACC^n - ACC^{n-1})/dt + div(F) = 0``:

  MASS    ACC = V_cell phi rho_mix,  rho_mix = s_l rho_l + s_v rho_v
          F   = V_T * upwind(lambda_T_mass),   lambda_T_mass = sum_g rho_g k_r(s_g)/mu_g
  ENERGY  ACC = V_cell [ phi (rho_mix h - p) + (1-phi) rho_s c_s T ]
          F   = -K_e dT/dx (Fourier)  +  V_T * upwind(sum_g h_g rho_g k_r/mu_g)  +  b (buoyancy)

  V_T (total Darcy flux, the surface velocity that carries gravity) on an internal face:
          V_T = T_f (p_L - p_U) - K A rho_ff g,   rho_ff = sum_g f_g rho_g,  f_g = rho_g k_r/mu_g / lambda_T_mass

Constitutive closure is sampled from the *same* Driesner table the solver uses
(``opensowat_xph_l_2_grads.vtk``, axes (z_NaCl, h[MJ/kg], p[MPa])) via O(1) uniform-grid
bilinear interpolation (validated to ~1e-10 against pyvista's probe). Boundary enthalpies
are obtained from the (z, T, p) table ``opensowat_xpt_l_2_grads.vtk``.

Units (Mega-scaled, matching the PorePy example): p[MPa], h[MJ/kg], rho[kg/m^3],
g=9.80665e-6, K=1e-15 m^2, K_e=2e-6, c_s=880e-6, rho_s=2700, phi=0.1.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --------------------------------------------------------------------------------------- #
#  Paths / physical constants (from geothermal_H2O_low_NaCl_content_fig_5.py)
# --------------------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
VTK_DIR = os.path.join(HERE, "model_configuration", "constitutive_description",
                       "driesner_vtk_files")
VTK_XPH = os.path.join(VTK_DIR, "opensowat_xph_l_2_grads.vtk")  # (z, h[MJ/kg], p[MPa])
VTK_XPT = os.path.join(VTK_DIR, "opensowat_xpt_l_2_grads.vtk")  # (z, T[degC],  p[MPa])

G = 9.80665e-6        # gravity, MPa-scaled (= GRAVITY_ACCELERATION * to_Mega)
K_PERM = 1.0e-15      # permeability [m^2]
PHI = 0.1             # porosity
K_E = 2.0e-6          # effective thermal conductivity [MW/(m K)]
RHO_S = 2700.0        # rock density [kg/m^3]
C_S = 880.0e-6        # rock specific heat [MJ/(kg K)] (880 * to_Mega)
S_R_LIQ = 0.3         # residual liquid saturation

L_COLUMN = 2000.0     # column height [m]
DX = 10.0             # lateral cross-section [m] (cancels in the solution)
YEAR = 365.0 * 86400.0

# fig-5D boundary/initial data
P_BOT, P_TOP = 20.0, 1.0          # MPa  (inlet y=0 / outlet y=2000)
T_BOT, T_TOP = 673.15, 423.15     # K
T_INIT = 423.15                   # K, constant IC


# --------------------------------------------------------------------------------------- #
#  Fast uniform-grid bilinear table (works for xph and xpt; both RectilinearGrid, uniform)
# --------------------------------------------------------------------------------------- #
class Table:
    """O(1) vectorised bilinear sampler of a Driesner VTK table on the z_NaCl=0 slice.

    Axes are (X=z_NaCl, Y=second [h or T], Z=p). Returns interpolated field values and,
    on request, the analytic (d/d(second), d/dp) from the table's ``grad_<name>`` columns.
    """

    def __init__(self, file_name: str, fields: dict[str, float]):
        import pyvista as pv

        g = pv.read(file_name)
        self.nx, self.ny, self.nz = g.dimensions          # (z, second, p)
        self.a = np.asarray(g.y)                           # second axis nodes
        self.b = np.asarray(g.z)                           # pressure axis nodes
        self.a0, self.da = self.a[0], self.a[1] - self.a[0]
        self.b0, self.db = self.b[0], self.b[1] - self.b[0]
        self.V: dict[str, np.ndarray] = {}
        self.Ga: dict[str, np.ndarray] = {}
        self.Gb: dict[str, np.ndarray] = {}
        for name, scale in fields.items():
            arr = np.asarray(g.point_data[name]).reshape(self.nz, self.ny, self.nx)[:, :, 0]
            self.V[name] = arr * scale                     # [p, second]
            grad = np.asarray(g.point_data["grad_" + name]).reshape(
                self.nz, self.ny, self.nx, 3)[:, :, 0, :]
            self.Ga[name] = grad[:, :, 1] * scale          # d/d(second)
            self.Gb[name] = grad[:, :, 2] * scale          # d/dp

    def _stencil(self, a, b):
        a = np.atleast_1d(np.asarray(a, float))
        b = np.atleast_1d(np.asarray(b, float))
        fa = np.clip((a - self.a0) / self.da, 0.0, self.ny - 1 - 1e-9)
        fb = np.clip((b - self.b0) / self.db, 0.0, self.nz - 1 - 1e-9)
        ja = fa.astype(int); jb = fb.astype(int)
        ta = fa - ja; tb = fb - jb
        return ja, jb, ta, tb

    def __call__(self, name, a, b, deriv=False):
        ja, jb, ta, tb = self._stencil(a, b)
        A = self.V[name]
        f00 = A[jb, ja]; f10 = A[jb, ja + 1]; f01 = A[jb + 1, ja]; f11 = A[jb + 1, ja + 1]
        val = ((1 - ta) * (1 - tb) * f00 + ta * (1 - tb) * f10
               + (1 - ta) * tb * f01 + ta * tb * f11)
        if not deriv:
            return val
        Ga, Gb = self.Ga[name], self.Gb[name]
        da = ((1 - ta) * (1 - tb) * Ga[jb, ja] + ta * (1 - tb) * Ga[jb, ja + 1]
              + (1 - ta) * tb * Ga[jb + 1, ja] + ta * tb * Ga[jb + 1, ja + 1])
        db = ((1 - ta) * (1 - tb) * Gb[jb, ja] + ta * (1 - tb) * Gb[jb, ja + 1]
              + (1 - ta) * tb * Gb[jb + 1, ja] + ta * tb * Gb[jb + 1, ja + 1])
        return val, da, db


# field -> unit scaling (H: kJ->MJ ; mu: micro-Pa.s -> Pa.s ; rest raw)
_XPH_FIELDS = {"Rho_l": 1.0, "Rho_v": 1.0, "H_l": 1e-3, "H_v": 1e-3,
               "mu_l": 1e-6, "mu_v": 1e-6, "S_v": 1.0, "Temperature": 1.0}


# --------------------------------------------------------------------------------------- #
#  Constitutive closure from the xph table  (vectorised; z_NaCl = 0)
# --------------------------------------------------------------------------------------- #
@dataclass
class Props:
    rho_l: np.ndarray; rho_v: np.ndarray
    s_v: np.ndarray; s_l: np.ndarray
    h_l: np.ndarray; h_v: np.ndarray
    mu_l: np.ndarray; mu_v: np.ndarray
    T: np.ndarray                       # Kelvin
    rho_mix: np.ndarray                 # s_l rho_l + s_v rho_v   (accumulation density)
    lam_T: np.ndarray                   # total mass mobility  sum_g rho_g k_r/mu_g
    f_l: np.ndarray; f_v: np.ndarray    # fractional phase mass mobility
    rho_ff: np.ndarray                  # sum_g f_g rho_g (gravity/advective density)
    mm_l: np.ndarray; mm_v: np.ndarray  # phase mass mobility rho_g k_r/mu_g
    adv_h: np.ndarray                   # sum_g h_g rho_g k_r/mu_g (advected enthalpy weight)


def eval_props(table: Table, p: np.ndarray, h: np.ndarray) -> Props:
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
    return Props(rho_l, rho_v, s_v, s_l, h_l, h_v, mu_l, mu_v, T,
                 rho_mix, lam_T, f_l, f_v, rho_ff, mm_l, mm_v, adv_h)


# --------------------------------------------------------------------------------------- #
#  Geometry + frozen upwind directions
# --------------------------------------------------------------------------------------- #
@dataclass
class Geom:
    N: int
    dy: float
    A: float
    Tf: float        # internal face transmissibility K A / dy
    Tb: float        # boundary half-face transmissibility 2 K A / dy
    TFf: float       # internal Fourier transmissibility K_e A / dy
    TFb: float       # boundary Fourier half-transmissibility 2 K_e A / dy
    Vcell: float     # cell volume A dy
    GA: float        # K A g  (gravity flux coefficient)
    yc: np.ndarray   # cell centres


def make_geom(N: int) -> Geom:
    dy = L_COLUMN / N
    A = DX
    return Geom(N=N, dy=dy, A=A, Tf=K_PERM * A / dy, Tb=2.0 * K_PERM * A / dy,
                TFf=K_E * A / dy, TFb=2.0 * K_E * A / dy, Vcell=A * dy,
                GA=K_PERM * A * G, yc=(np.arange(N) + 0.5) * dy)


def _upwind_idx(direction: np.ndarray) -> np.ndarray:
    """Per internal face (lower cell i, upper cell i+1): upstream index.

    direction >= 0  -> flux is upward -> upstream is the LOWER cell i.
    direction <  0  -> upstream is the UPPER cell i+1.
    Returns an int array of length N-1 giving the cell index that supplies the value.
    """
    i = np.arange(direction.size)
    return np.where(direction >= 0.0, i, i + 1)


def buoyancy_directions(geom: Geom, p: np.ndarray, pr: Props, scheme: str):
    """Upstream cell indices (gamma=liquid, delta=gas) on internal faces for the
    chosen scheme. gamma/delta ordering matches PorePy phases = [liq, gas]."""
    rho_l_f = 0.5 * (pr.rho_l[:-1] + pr.rho_l[1:])
    rho_v_f = 0.5 * (pr.rho_v[:-1] + pr.rho_v[1:])
    if scheme == "hu":
        # inter-phase gravity flux ddf(rho_l - rho_v) = -K A (rho_l - rho_v) g
        ddf = -geom.GA * (rho_l_f - rho_v_f)
        dir_gamma = ddf            # liquid rides +ddf
        dir_delta = -ddf           # gas rides -ddf
    elif scheme == "ppu":
        dp = geom.Tf * (p[:-1] - p[1:])                 # pressure-driven part
        dir_gamma = dp - geom.GA * rho_l_f              # Psi_liq ~ -K(grad p - rho_l g)
        dir_delta = dp - geom.GA * rho_v_f              # Psi_gas
    else:
        raise ValueError(f"unknown scheme {scheme!r}; use 'hu' or 'ppu'")
    return _upwind_idx(dir_gamma), _upwind_idx(dir_delta)


# --------------------------------------------------------------------------------------- #
#  Residual
# --------------------------------------------------------------------------------------- #
@dataclass
class BoundaryState:
    p: float; h: float
    pr: Props          # single-cell props at (p, h)
    T: float


def boundary_state(table: Table, p_bc: float, h_bc: float) -> BoundaryState:
    pr = eval_props(table, np.array([p_bc]), np.array([h_bc]))
    return BoundaryState(p=p_bc, h=h_bc, pr=pr, T=float(pr.T[0]))


def residual(x: np.ndarray, x_old: np.ndarray, dt: float, geom: Geom, table: Table,
             bbot: BoundaryState, btop: BoundaryState, ug: np.ndarray, ud: np.ndarray):
    """Full 2N residual (interleaved [mass_0, energy_0, mass_1, ...]).

    ``ug``/``ud`` are the frozen buoyancy upstream indices (lagged per time step).
    """
    N = geom.N
    p = x[0::2]; h = x[1::2]
    pr = eval_props(table, p, h)

    p_old = x_old[0::2]; h_old = x_old[1::2]
    pr_old = eval_props(table, p_old, h_old)

    # ---- accumulation (backward Euler) ----
    acc_mass = geom.Vcell * PHI * pr.rho_mix
    acc_mass_o = geom.Vcell * PHI * pr_old.rho_mix
    acc_en = geom.Vcell * (PHI * (pr.rho_mix * h - p) + (1 - PHI) * RHO_S * C_S * pr.T)
    acc_en_o = geom.Vcell * (PHI * (pr_old.rho_mix * h_old - p_old)
                             + (1 - PHI) * RHO_S * C_S * pr_old.T)

    # ---- internal faces (N-1) ----
    rho_ff_f = 0.5 * (pr.rho_ff[:-1] + pr.rho_ff[1:])
    V_T = geom.Tf * (p[:-1] - p[1:]) - geom.GA * rho_ff_f
    up = np.where(V_T >= 0.0, np.arange(N - 1), np.arange(N - 1) + 1)   # advection upwind
    F_mass = V_T * pr.lam_T[up]
    F_adv_h = V_T * pr.adv_h[up]
    F_four = geom.TFf * (pr.T[:-1] - pr.T[1:])

    # buoyancy (energy only); advected gamma quantity = h_gamma = h_l
    rho_l_f = 0.5 * (pr.rho_l[:-1] + pr.rho_l[1:])
    rho_v_f = 0.5 * (pr.rho_v[:-1] + pr.rho_v[1:])
    w_flux = -geom.GA * (rho_l_f - rho_v_f)                      # ddf(rho_l - rho_v)
    fg_up = pr.h_l[ug] * pr.f_l[ug]                             # upwind_gamma @ (h_l f_l)
    fd_up = pr.f_v[ud]                                          # upwind_delta @ f_v
    lam_up = pr.mm_l[ug] + pr.mm_v[ud]                          # lambda_g^up + lambda_d^up
    F_buoy = fg_up * fd_up * lam_up * w_flux

    F_en = F_four + F_adv_h + F_buoy

    # ---- boundary faces (Dirichlet p, T -> h_bc) ----
    # bottom (below cell 0), flux positive upward into the domain
    rff_b = bbot.pr.rho_ff[0]
    V_b = geom.Tb * (bbot.p - p[0]) - geom.GA * rff_b
    if V_b >= 0.0:   # inflow (upward) -> boundary props
        Fm_b = V_b * bbot.pr.lam_T[0];  Fh_b = V_b * bbot.pr.adv_h[0]
    else:
        Fm_b = V_b * pr.lam_T[0];       Fh_b = V_b * pr.adv_h[0]
    Ff_b = geom.TFb * (bbot.T - pr.T[0])
    Fmass_bot0 = Fm_b
    Fen_bot0 = Ff_b + Fh_b

    # top (above cell N-1), flux positive upward out of the domain
    rff_t = btop.pr.rho_ff[0]
    V_t = geom.Tb * (p[-1] - btop.p) - geom.GA * rff_t
    if V_t >= 0.0:   # outflow (upward) -> interior props
        Fm_t = V_t * pr.lam_T[-1];      Fh_t = V_t * pr.adv_h[-1]
    else:
        Fm_t = V_t * btop.pr.lam_T[0];  Fh_t = V_t * btop.pr.adv_h[0]
    Ff_t = geom.TFb * (pr.T[-1] - btop.T)
    Fmass_topN = Fm_t
    Fen_topN = Ff_t + Fh_t

    # ---- divergence: net upward outflow per cell = F_top - F_bottom ----
    div_mass = np.empty(N); div_en = np.empty(N)
    # mass
    div_mass[0] = F_mass[0] - Fmass_bot0
    div_mass[1:-1] = F_mass[1:] - F_mass[:-1]
    div_mass[-1] = Fmass_topN - F_mass[-1]
    # energy
    div_en[0] = F_en[0] - Fen_bot0
    div_en[1:-1] = F_en[1:] - F_en[:-1]
    div_en[-1] = Fen_topN - F_en[-1]

    r_mass = (acc_mass - acc_mass_o) / dt + div_mass
    r_en = (acc_en - acc_en_o) / dt + div_en

    r = np.empty(2 * N)
    r[0::2] = r_mass
    r[1::2] = r_en
    return r


# --------------------------------------------------------------------------------------- #
#  Sparse coloured finite-difference Jacobian (block-tridiagonal, 6 colours)
# --------------------------------------------------------------------------------------- #
def _build_sparsity(N: int):
    """Row support of each column for interleaved [p0,h0,p1,h1,...]; coupling to +/-1 cell."""
    rows_of_col = []
    for k in range(N):
        for _v in range(2):
            rows = []
            for kk in (k - 1, k, k + 1):
                if 0 <= kk < N:
                    rows += [2 * kk, 2 * kk + 1]
            rows_of_col.append(np.array(rows, dtype=int))
    color = np.array([(k % 3) * 2 + v for k in range(N) for v in range(2)])  # 6 colours
    return rows_of_col, color


def jacobian_fd(x, r0, args, rows_of_col, color, eps_rel=1e-7):
    N = args[2].N
    n = 2 * N
    scale = np.where(np.arange(n) % 2 == 0, 1.0, 0.1)     # p ~ O(1) MPa, h ~ O(0.1) MJ/kg
    eps = eps_rel * np.maximum(np.abs(x), scale)
    rows_all, cols_all, data_all = [], [], []
    for c in range(6):
        cols_c = np.where(color == c)[0]
        dx = np.zeros(n); dx[cols_c] = eps[cols_c]
        r1 = residual(x + dx, *args)
        dr = (r1 - r0)
        for j in cols_c:
            rws = rows_of_col[j]
            rows_all.append(rws)
            cols_all.append(np.full(rws.size, j))
            data_all.append(dr[rws] / eps[j])
    J = sp.csc_matrix((np.concatenate(data_all),
                       (np.concatenate(rows_all), np.concatenate(cols_all))), shape=(n, n))
    return J


# --------------------------------------------------------------------------------------- #
#  Newton time stepping
# --------------------------------------------------------------------------------------- #
def newton_step(x0, x_old, dt, geom, table, bbot, btop, scheme,
                tol=9e-5, maxit=20, verbose=False):
    rows_of_col, color = _build_sparsity(geom.N)
    # buoyancy directions lagged: frozen from the previous converged state x_old
    p_old, h_old = x_old[0::2], x_old[1::2]
    pr_old = eval_props(table, p_old, h_old)
    ug, ud = buoyancy_directions(geom, p_old, pr_old, scheme)

    x = x0.copy()
    args = (x_old, dt, geom, table, bbot, btop, ug, ud)
    for it in range(maxit):
        r = residual(x, *args)
        nrm = np.linalg.norm(r)
        if verbose:
            print(f"    newton {it}: |r|={nrm:.3e}")
        if nrm < tol:
            return x, it, nrm, True
        J = jacobian_fd(x, r, args, rows_of_col, color)
        try:
            dx = spla.spsolve(J.tocsc(), -r)
        except Exception:
            dx = spla.lsqr(J, -r)[0]
        # damped update for robustness
        step = 1.0
        for _ in range(8):
            xn = x + step * dx
            xn[1::2] = np.clip(xn[1::2], table.a0 + 1e-6, table.a[-1] - 1e-6)  # h in table
            xn[0::2] = np.clip(xn[0::2], table.b0 + 1e-6, table.b[-1] - 1e-6)  # p in table
            if np.linalg.norm(residual(xn, *args)) < nrm:
                break
            step *= 0.5
        x = xn
    return x, maxit, nrm, False


def run(scheme="hu", N=200, n_steps=None, dt=None, verbose=True):
    """Integrate the fig-5D column to t=1000 yr with the chosen buoyancy scheme."""
    table = Table(VTK_XPH, _XPH_FIELDS)
    xpt = Table(VTK_XPT, {"H": 1e-3})       # for boundary T -> h
    geom = make_geom(N)

    # boundary enthalpies from (z, T[degC], p)
    h_bot = float(xpt("H", T_BOT - 273.15, P_BOT)[0])
    h_top = float(xpt("H", T_TOP - 273.15, P_TOP)[0])
    bbot = boundary_state(table, P_BOT, h_bot)
    btop = boundary_state(table, P_TOP, h_top)

    # initial condition: p linear (P_BOT@bottom -> P_TOP@top), T = T_INIT constant
    y = geom.yc
    p0 = (y * P_TOP + (L_COLUMN - y) * P_BOT) / L_COLUMN
    h0 = xpt("H", np.full(N, T_INIT - 273.15), p0)
    x = np.empty(2 * N); x[0::2] = p0; x[1::2] = h0

    dt = dt if dt is not None else 0.125 * YEAR
    n_steps = n_steps if n_steps is not None else int(round(1000.0 * YEAR / dt))

    for step in range(n_steps):
        x_old = x.copy()
        x, nit, nrm, ok = newton_step(x, x_old, dt, geom, table, bbot, btop, scheme)
        if verbose and (step % max(1, n_steps // 20) == 0 or not ok):
            t_yr = (step + 1) * dt / YEAR
            print(f"step {step+1}/{n_steps}  t={t_yr:7.1f} yr  newton_it={nit}  "
                  f"|r|={nrm:.2e}  {'OK' if ok else 'NOT CONVERGED'}")
        if not ok:
            print(f"  WARNING: Newton did not converge at step {step+1}")

    pr = eval_props(table, x[0::2], x[1::2])
    return {"y": y, "p": x[0::2], "h": x[1::2], "T": pr.T, "s_gas": pr.s_v,
            "rho_mix": pr.rho_mix, "scheme": scheme, "N": N}


# --------------------------------------------------------------------------------------- #
#  Self-test (cheap invariants) + output
# --------------------------------------------------------------------------------------- #
def selftest():
    print("=== selftest ===")
    table = Table(VTK_XPH, _XPH_FIELDS)
    geom = make_geom(20)
    # single-phase -> buoyancy must vanish: set enthalpy in the liquid range everywhere
    p = np.linspace(20, 1, 20)
    h = np.full(20, 0.6)                       # cold liquid -> s_v = 0
    pr = eval_props(table, p, h)
    assert np.all(pr.s_v < 1e-6), "expected single-phase liquid"
    rho_l_f = 0.5 * (pr.rho_l[:-1] + pr.rho_l[1:]); rho_v_f = 0.5 * (pr.rho_v[:-1] + pr.rho_v[1:])
    ug, ud = buoyancy_directions(geom, p, pr, "hu")
    w = -geom.GA * (rho_l_f - rho_v_f)
    b = pr.h_l[ug] * pr.f_l[ug] * pr.f_v[ud] * (pr.mm_l[ug] + pr.mm_v[ud]) * w
    assert np.max(np.abs(b)) < 1e-30, f"single-phase buoyancy not zero: {np.max(np.abs(b)):.2e}"
    print("  single-phase buoyancy == 0  OK")
    # hydrostatic: build p so that V_T == 0 on internal faces with uniform liquid
    p_hyd = np.empty(20); p_hyd[0] = 20.0
    for i in range(1, 20):
        rho = 0.5 * (pr.rho_ff[i - 1] + pr.rho_ff[i])
        p_hyd[i] = p_hyd[i - 1] - rho * G * geom.dy
    pr2 = eval_props(table, p_hyd, h)
    rff = 0.5 * (pr2.rho_ff[:-1] + pr2.rho_ff[1:])
    VT = geom.Tf * (p_hyd[:-1] - p_hyd[1:]) - geom.GA * rff
    print(f"  hydrostatic max|V_T| = {np.max(np.abs(VT)):.2e} (should be ~0)")
    print("  selftest passed\n")


def write_vtk(res, path):
    import pyvista as pv

    y = res["y"]; n = y.size
    X, Y, Z = np.meshgrid([0.0, DX], np.concatenate([[0.0], 0.5 * (y[:-1] + y[1:]),
                          [L_COLUMN]]), [0.0, DX], indexing="ij")
    # store as cell data on a (2 x n x 2) structured grid stacked along y
    Xc, Yc, Zc = np.meshgrid([0.0, DX], np.concatenate([[0.0], y, [L_COLUMN]])[:n + 1],
                             [0.0, DX], indexing="ij")
    grid = pv.StructuredGrid(Xc, Yc, Zc)
    for k in ("p", "T", "s_gas", "h", "rho_mix"):
        v = res[k].copy()
        if k == "T":
            v = v - 273.15
        grid.cell_data[k] = np.broadcast_to(v.reshape(1, n, 1), (1, n, 1)).ravel(order="F")
    grid.save(path)
    print(f"wrote {path}")


def main():
    selftest()
    import sys
    scheme = sys.argv[1] if len(sys.argv) > 1 else "hu"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    # quick smoke run by default; pass full steps via env for the real fig-5D run
    n_steps = int(os.environ.get("NSTEPS", "40"))
    res = run(scheme=scheme, N=N, n_steps=n_steps)
    out_dir = os.path.join(HERE, "visualization_1D_fig_5")
    os.makedirs(out_dir, exist_ok=True)
    write_vtk(res, os.path.join(out_dir, f"fig5D_1D_{scheme}_N{N}.vts"))
    print(f"\nfinal profile ({scheme}, N={N}, {n_steps} steps):")
    print(f"  p:     {res['p'][0]:.2f} -> {res['p'][-1]:.2f} MPa")
    print(f"  T:     {res['T'][0]-273.15:.1f} -> {res['T'][-1]-273.15:.1f} degC")
    print(f"  s_gas: {res['s_gas'].min():.3f} .. {res['s_gas'].max():.3f}")
    band = np.where((res['s_gas'] > 1e-3) & (res['s_gas'] < 1 - 1e-3))[0]
    if band.size:
        print(f"  two-phase band: y in [{res['y'][band[0]]:.0f}, {res['y'][band[-1]]:.0f}] m")


if __name__ == "__main__":
    main()
