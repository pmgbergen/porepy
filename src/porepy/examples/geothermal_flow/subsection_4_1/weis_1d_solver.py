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
The constitutive closure is sampled from the *same* Driesner tables the PorePy run uses,
``opensowat_xph_l_{L}_grads.vtr`` (axes z_NaCl, h[MJ/kg], p[MPa]) at refinement level L in
0..5 (``--level``, default 2; coarsest..finest), via O(1) uniform-grid bilinear interpolation;
SI<->table unit conversion is handled inside the sampler.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --------------------------------------------------------------------------------------- #
#  Paths / physical constants (SI) -- values from geothermal_H2O_low_NaCl_content_fig_5.py
# --------------------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
VTK_DIR = os.path.join(HERE, "model_configuration", "constitutive_description",
                       "driesner_vtk_files")
REF_DIR = os.path.join(HERE, "benchmark_figures_data")

TABLE_LEVEL = 5       # Driesner opensowat table refinement level: 0 (coarsest) .. 5 (finest)


def table_paths(level=TABLE_LEVEL):
    """Absolute paths of the xph (z, h[MJ/kg], p[MPa]) and xpt (z, T[degC], p[MPa]) Driesner
    ``.vtr`` tables at refinement ``level`` (0..5)."""
    xph = os.path.join(VTK_DIR, f"opensowat_xph_l_{level}_grads.vtr")
    xpt = os.path.join(VTK_DIR, f"opensowat_xpt_l_{level}_grads.vtr")
    return xph, xpt


VTK_XPH, VTK_XPT = table_paths()      # default-level paths (used by selftest and run() default)

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

# Gravity-term density weighting on internal faces (see ``residual``):
#   arithmetic (default)  -> face average 0.5*(rho_i + rho_{i+1})
#   upstream              -> Weis (2014, Eq.25 p.352): fluid props in the gravity term taken
#                            from the lagged phase-upwind node (fully-upstream, as in CSMP++).
# Identical when g=0 (horizontal 5B); only changes the vertical (5D) case. Selected per run via
# the 4th CLI argument, e.g.:
#   python geothermal_H2O_low_NaCl_content_1D_fig_5.py ppu 200 vertical upstream

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
        key = "|".join(f"{k}:{v}" for k, v in sorted(fields.items())) + f"|{a_in}|{b_in}"
        cache = file_name + ".sicache.npz"
        if not (os.path.exists(cache) and self._load(cache, key)):
            self._build(file_name, fields, a_in, b_in, key, cache)
        # all fields stacked -> gather every field's 4 corners in 4 fancy-index ops, not 4*nf
        self.names = list(self.V.keys())
        self.V_stack = np.ascontiguousarray(np.stack([self.V[n] for n in self.names]))

    def _load(self, cache, key):
        try:
            z = np.load(cache, allow_pickle=False)
            if str(z["key"]) != key:
                return False
            for s in ("ny", "nz"):
                setattr(self, s, int(z[s]))
            for s in ("a0", "da", "b0", "db", "a_in", "b_in",
                      "a_min", "a_max", "b_min", "b_max"):
                setattr(self, s, float(z[s]))
            self.V = {str(n): z["V_" + str(n)] for n in z["names"]}
            return True
        except Exception:
            return False

    def _build(self, file_name, fields, a_in, b_in, key, cache):
        import pyvista as pv     # only hit on a cache miss (first run)

        g = pv.read(file_name)
        nx, ny, nz = g.dimensions                          # (z, second, p)
        a = np.asarray(g.y); b = np.asarray(g.z)
        self.ny, self.nz = int(ny), int(nz)
        self.a0, self.da = float(a[0]), float(a[1] - a[0])
        self.b0, self.db = float(b[0]), float(b[1] - b[0])
        self.a_in, self.b_in = float(a_in), float(b_in)
        self.a_min, self.a_max = float(a[0] / a_in), float(a[-1] / a_in)
        self.b_min, self.b_max = float(b[0] / b_in), float(b[-1] / b_in)
        self.V = {name: np.asarray(g.point_data[name]).reshape(nz, ny, nx)[:, :, 0] * scale
                  for name, scale in fields.items()}
        out = {"key": np.array(key), "ny": self.ny, "nz": self.nz,
               "a0": self.a0, "da": self.da, "b0": self.b0, "db": self.db,
               "a_in": self.a_in, "b_in": self.b_in, "a_min": self.a_min,
               "a_max": self.a_max, "b_min": self.b_min, "b_max": self.b_max,
               "names": np.array(list(self.V.keys()))}
        out.update({"V_" + n: A for n, A in self.V.items()})
        try:
            np.savez(cache, **out)
        except Exception:
            pass

    def sample_many(self, a, b):
        """Bilinear-interpolate ALL stored fields at (a, b) with ONE stencil + stacked gather."""
        a = np.atleast_1d(np.asarray(a, float)) * self.a_in
        b = np.atleast_1d(np.asarray(b, float)) * self.b_in
        fa = ((a - self.a0) / self.da).clip(0.0, self.ny - 1 - 1e-9)
        fb = ((b - self.b0) / self.db).clip(0.0, self.nz - 1 - 1e-9)
        ja = fa.astype(np.intp); jb = fb.astype(np.intp)
        ta = fa - ja; tb = fb - jb
        ja1 = ja + 1; jb1 = jb + 1
        Vs = self.V_stack
        vals = ((1 - ta) * (1 - tb) * Vs[:, jb, ja] + ta * (1 - tb) * Vs[:, jb, ja1]
                + (1 - ta) * tb * Vs[:, jb1, ja] + ta * tb * Vs[:, jb1, ja1])   # (nf, N)
        return {n: vals[i] for i, n in enumerate(self.names)}

    def __call__(self, name, a, b):
        return self.sample_many(a, b)[name]


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
    s = table.sample_many(h, p)                     # ONE stencil for all 8 fields
    rho_l = s["Rho_l"]; rho_v = s["Rho_v"]
    s_v = np.clip(s["S_v"], 0.0, 1.0)
    s_l = 1.0 - s_v
    h_l = s["H_l"]; h_v = s["H_v"]
    mu_l = s["mu_l"]; mu_v = s["mu_v"]
    T = s["Temperature"]

    kr_l = np.maximum((s_l - S_R_LIQ) / (1.0 - S_R_LIQ), 0.0)   # linear liquid, residual S_R_LIQ
    # Weis (2014, p.349): linear model with R_l = 0.3, R_v = 0 and k_rv + k_rl = 1 - S_h.
    # Here S_h = 0 (no halite) so k_rv = 1 - k_rl (= s_v/(1-S_R_LIQ) in the mobile range,
    # capped at 1 once the liquid is at residual). The previous k_rv = s_v was 0.7x too small.
    kr_v = 1.0 - kr_l
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


def make_geom(N, g=G):
    """Build the 1-D geometry. ``g`` is the gravity along the column: G for the vertical
    case (fig 5D), 0.0 for the horizontal case (fig 5B, gravity perpendicular to flow)."""
    dy = L_COLUMN / N
    A = DX
    Vcell = A * dy
    ms = Vcell * PHI * RHO_REF / DT0
    es = Vcell * (1 - PHI) * RHO_S * C_S * T_REF / DT0
    return Geom(N=N, dy=dy, A=A, Tf=K_PERM * A / dy, Tb=2.0 * K_PERM * A / dy,
                TFf=K_E * A / dy, TFb=2.0 * K_E * A / dy, Vcell=Vcell,
                GA=K_PERM * A * g, ms=ms, es=es, yc=(np.arange(N) + 0.5) * dy)


def _upwind_idx(direction):
    """Internal face (lower i, upper i+1): direction>=0 -> upstream lower i, else upper i+1."""
    i = np.arange(direction.size)
    return np.where(direction >= 0.0, i, i + 1)


def buoyancy_directions(geom, p, pr, scheme):
    """Per-internal-face LAGGED upstream cell indices ``(i_liq, i_gas, i_tot)``.

    All three are frozen per time step (evaluated on the old state), following Weis (2014, p.354,
    "we use the old velocity field to define the upwind nodes"). ``i_liq``/``i_gas`` drive the
    buoyancy term; ``i_tot`` is the total-velocity direction used for the upstream gravity density
    in ``V_T`` -- it MUST be lagged, because upwinding rho *inside* V_T by the current V_T would
    make the velocity discontinuous at flow reversal and break Newton.

    hu:   liquid rides +ddf(rho_l-rho_v), gas rides -ddf  (opposite inter-phase directions).
    ppu:  each phase rides its own potential Psi_g = T_f(p_L-p_U) - K A rho_g g.
    Phase order matches PorePy phases = [liq, gas].
    """
    rho_l_f = 0.5 * (pr.rho_l[:-1] + pr.rho_l[1:])
    rho_v_f = 0.5 * (pr.rho_v[:-1] + pr.rho_v[1:])
    rho_ff_f = 0.5 * (pr.rho_ff[:-1] + pr.rho_ff[1:])
    i_tot = _upwind_idx(geom.Tf * (p[:-1] - p[1:]) - geom.GA * rho_ff_f)   # lagged total-velocity
    if scheme == "hu":
        ddf = -geom.GA * (rho_l_f - rho_v_f)     # inter-phase gravity flux ddf(rho_l-rho_v)
        dir_liq, dir_gas = ddf, -ddf
    elif scheme == "ppu":
        dp = geom.Tf * (p[:-1] - p[1:])
        dir_liq = dp - geom.GA * rho_l_f         # Psi_liq ~ -K(grad p - rho_l g)
        dir_gas = dp - geom.GA * rho_v_f         # Psi_gas
    else:
        raise ValueError(f"unknown scheme {scheme!r}; use 'hu' or 'ppu'")
    return _upwind_idx(dir_liq), _upwind_idx(dir_gas), i_tot


# --------------------------------------------------------------------------------------- #
#  Residual
# --------------------------------------------------------------------------------------- #
@dataclass
class BoundaryState:
    p: float; h: float; pr: Props; T: float


def boundary_state(table, p_bc, h_bc):
    pr = eval_props(table, np.array([p_bc]), np.array([h_bc]))
    return BoundaryState(p=p_bc, h=h_bc, pr=pr, T=float(pr.T[0]))


def accumulation_old(x_old, geom, table):
    """Old-time accumulation (constant over a step's Newton iterations) -- computed once."""
    p_old = x_old[0::2]; h_old = x_old[1::2]
    pr = eval_props(table, p_old, h_old)
    acc_mass_o = geom.Vcell * PHI * pr.rho_mix
    acc_en_o = geom.Vcell * (PHI * (pr.rho_mix * h_old - p_old)
                             + (1 - PHI) * RHO_S * C_S * pr.T)
    return acc_mass_o, acc_en_o


def residual(x, acc_mass_o, acc_en_o, dt, geom, table, bbot, btop, scheme, ug, ud, ut,
             grav_upstream=False, weighted_perm=False):
    """Full 2N residual, interleaved [mass_0, energy_0, mass_1, ...].

    ``acc_*_o`` are the (precomputed, step-constant) old-time accumulations.
    Two schemes (``ug``/``ud`` = frozen lagged per-face upstream indices for liq/gas):
      "hu"   -> total-velocity advection + simplicial buoyancy, directions i_liq=upwind(+ddf),
                i_gas=upwind(-ddf) (inter-phase gravity flux).
      "ppu"  -> genuine per-phase potential upwinding of the FULL phase flux
                F=sum_g Psi_g upwind(w_g) (buoyancy intrinsic; Weis fig-5 reference).
    Only "ppu" takes the genuine-PPU branch below; "hu" takes the simplicial branch.
    """
    N = geom.N
    p = x[0::2]; h = x[1::2]
    pr = eval_props(table, p, h)

    # accumulation (backward Euler); old-time part passed in (step-constant)
    acc_mass = geom.Vcell * PHI * pr.rho_mix
    acc_en = geom.Vcell * (PHI * (pr.rho_mix * h - p) + (1 - PHI) * RHO_S * C_S * pr.T)

    dp_face = p[:-1] - p[1:]
    rho_l_f = 0.5 * (pr.rho_l[:-1] + pr.rho_l[1:])
    rho_v_f = 0.5 * (pr.rho_v[:-1] + pr.rho_v[1:])
    F_four = geom.TFf * (pr.T[:-1] - pr.T[1:])

    if scheme == "ppu":
        # genuine phase-potential upwinding: each phase rides its OWN potential flux
        # Psi_g = T_f(p_L-p_U) - K A rho_g g, mobility/enthalpy upwinded by sign(Psi_g)
        # (lagged in ug/ud). Buoyancy is intrinsic to Psi_g (no separate term). With
        # grav_upstream the gravity density is the lagged phase-upwind node, so
        # -GA*rho_g[ug]*mm_g[ug] = -GA*upstream(rho_g^2 k_r/mu) exactly (Weis Eq.25).
        rho_l_p = pr.rho_l[ug] if grav_upstream else rho_l_f
        rho_v_p = pr.rho_v[ud] if grav_upstream else rho_v_f
        Psi_l = geom.Tf * dp_face - geom.GA * rho_l_p
        Psi_v = geom.Tf * dp_face - geom.GA * rho_v_p
        F_mass = Psi_l * pr.mm_l[ug] + Psi_v * pr.mm_v[ud]
        F_en = F_four + Psi_l * (pr.h_l[ug] * pr.mm_l[ug]) + Psi_v * (pr.h_v[ud] * pr.mm_v[ud])
    else:
        # HU: total-velocity advection + simplicial buoyancy.
        # grav_upstream upstreams ONLY the advective gravity density rho_ff in V_T (lagged ut).
        # The buoyant driving force (rho_l - rho_v) in w_flux is a gravitational POTENTIAL
        # difference, not an advected flux, so it stays FACE-CENTERED in both modes -- upwinding
        # it would be wrong.
        rho_ff_f = 0.5 * (pr.rho_ff[:-1] + pr.rho_ff[1:])
        rho_ff_g = pr.rho_ff[ut] if grav_upstream else rho_ff_f
        V_T = geom.Tf * dp_face - geom.GA * rho_ff_g
        up = np.where(V_T >= 0.0, np.arange(N - 1), np.arange(N - 1) + 1)
        # Total-mobility placement in m_e = <lambda>(pi + w) (paper Remark 3.2):
        #   weighted_perm=False -> upwind the total mobility (separate weight, geometric T_f;
        #                          the standard HU setting, with a switch at m_e=0);
        #   weighted_perm=True  -> fold lambda into K, i.e. the HARMONIC face average of lambda_T
        #                          (joint lambda*K MPFA), reassembled each Newton iteration and
        #                          smooth in the state. Only affects m_e; the buoyant fluxes are
        #                          untouched. The energy advection q_a = <hbar> m_e (Table 2)
        #                          inherits the placement (hbar upwinded, m_e harmonic).
        if weighted_perm:
            lam_L = pr.lam_T[:-1]; lam_R = pr.lam_T[1:]
            s_lam = lam_L + lam_R
            lam_face = np.where(s_lam > 0.0, 2.0 * lam_L * lam_R / np.where(s_lam > 0.0, s_lam, 1.0), 0.0)
            F_mass = V_T * lam_face
            hbar_up = pr.adv_h[up] / np.where(pr.lam_T[up] > 0.0, pr.lam_T[up], 1.0)   # <hbar>
            F_en_adv = hbar_up * F_mass                                                # <hbar> m_e
        else:
            F_mass = V_T * pr.lam_T[up]
            F_en_adv = V_T * pr.adv_h[up]
        w_flux = -geom.GA * (rho_l_f - rho_v_f)      # face-centered buoyant driving force
        common = pr.f_l[ug] * pr.f_v[ud] * (pr.mm_l[ug] + pr.mm_v[ud])
        F_buoy = BUOY_SCALE * common * w_flux * (pr.h_l[ug] - pr.h_v[ud])
        F_en = F_four + F_en_adv + F_buoy

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
def build_jac_plan(N):
    """Precompute the 6-colour FD-Jacobian sparsity ONCE (block-tridiagonal interleaved)."""
    rows_of_col = []
    for k in range(N):
        for _v in range(2):
            rows = []
            for kk in (k - 1, k, k + 1):
                if 0 <= kk < N:
                    rows += [2 * kk, 2 * kk + 1]
            rows_of_col.append(np.array(rows, dtype=np.intp))
    color = np.array([(k % 3) * 2 + v for k in range(N) for v in range(2)])
    n = 2 * N
    col_perturb, gat_rows, gat_owner, coo_cols = [], [], [], []
    for c in range(6):
        cols_c = np.where(color == c)[0]
        rs = [rows_of_col[j] for j in cols_c]
        ow = [np.full(rows_of_col[j].size, j, dtype=np.intp) for j in cols_c]
        col_perturb.append(cols_c)
        gat_rows.append(np.concatenate(rs))      # rows touched by this colour (gather)
        gat_owner.append(np.concatenate(ow))     # owning column (for eps + COO col)
    all_rows = np.concatenate(gat_rows)
    all_cols = np.concatenate(gat_owner)
    sc = np.where(np.arange(n) % 2 == 0, 1.0e6, 1.0e5)   # p ~ MPa, h ~ 1e5 J/kg
    # banded (LAPACK) storage: block-tridiagonal interleaved -> bandwidth l,u (=3 here).
    l = int((all_rows - all_cols).max())
    u = int((all_cols - all_rows).max())
    bpos = u + all_rows - all_cols                       # ab[u+i-j, j] = A[i,j]
    return dict(n=n, col_perturb=col_perturb, gat_rows=gat_rows, gat_owner=gat_owner,
                all_rows=all_rows, all_cols=all_cols, scale=sc, l=l, u=u, bpos=bpos)


def jacobian_fd(x, r0, args, plan, eps_rel=1e-7):
    """Coloured FD Jacobian in LAPACK banded storage (ab, shape (l+u+1, n))."""
    n = plan["n"]
    eps = eps_rel * np.maximum(np.abs(x), plan["scale"])
    parts = []
    for c in range(6):
        cols_c = plan["col_perturb"][c]
        dx = np.zeros(n); dx[cols_c] = eps[cols_c]
        dr = residual(x + dx, *args) - r0
        parts.append(dr[plan["gat_rows"][c]] / eps[plan["gat_owner"][c]])   # vectorised
    ab = np.zeros((plan["l"] + plan["u"] + 1, n))
    ab[plan["bpos"], plan["all_cols"]] = np.concatenate(parts)
    return ab


# --------------------------------------------------------------------------------------- #
#  Newton time stepping
# --------------------------------------------------------------------------------------- #
def newton_step(x0, x_old, dt, geom, table, bbot, btop, scheme, plan,
                rtol=1e-6, atol=1e-7, maxit=25, verbose=False, grav_upstream=False,
                weighted_perm=False):
    p_old = x_old[0::2]; h_old = x_old[1::2]
    pr_old = eval_props(table, p_old, h_old)                  # x_old props: ONE eval
    ug, ud, ut = buoyancy_directions(geom, p_old, pr_old, scheme)  # lagged per step
    acc_mass_o = geom.Vcell * PHI * pr_old.rho_mix
    acc_en_o = geom.Vcell * (PHI * (pr_old.rho_mix * h_old - p_old)
                             + (1 - PHI) * RHO_S * C_S * pr_old.T)
    args = (acc_mass_o, acc_en_o, dt, geom, table, bbot, btop, scheme, ug, ud, ut,
            grav_upstream, weighted_perm)
    pclip = (table.b_min * (1 + 1e-9), table.b_max * (1 - 1e-9))
    hclip = (table.a_min * (1 + 1e-9), table.a_max * (1 - 1e-9))

    x = x0.copy()
    r = residual(x, *args)
    nrm = nrm0 = np.linalg.norm(r)
    tol = rtol * nrm0 + atol * np.sqrt(2 * geom.N)
    for it in range(maxit):
        if verbose:
            print(f"    newton {it}: |r|={nrm:.3e}")
        if nrm <= tol:
            return x, it, nrm, True
        ab = jacobian_fd(x, r, args, plan)
        try:
            dx = sla.solve_banded((plan["l"], plan["u"]), ab, -r)   # O(N) banded solve
        except Exception:
            dx = np.zeros_like(r)        # singular -> no step; line search fails -> dt cut
        step = 1.0
        for _ in range(10):                          # backtracking; reuse the accepted r
            xn = x + step * dx
            xn[0::2] = np.clip(xn[0::2], *pclip)
            xn[1::2] = np.clip(xn[1::2], *hclip)
            r_new = residual(xn, *args); nrm_new = np.linalg.norm(r_new)
            if nrm_new < nrm:
                break
            step *= 0.5
        x = xn; r = r_new; nrm = nrm_new
    return x, maxit, nrm, nrm <= 1e-3 * nrm0


# Per-case settings: gravity along the column and the benchmark final time.
#   "vertical"   = fig 5D: gravity ON,  t_final = 1000 yr.
#   "horizontal" = fig 5B: gravity OFF (perpendicular to flow), t_final = 200 yr.
CASES = {"vertical": dict(g=G, tf_yr=1000.0), "horizontal": dict(g=0.0, tf_yr=200.0)}


def run(scheme="hu", N=200, case="vertical", n_steps=None, dt=None,
        adaptive=True, verbose=True, grav_upstream=False, weighted_perm=False,
        level=TABLE_LEVEL):
    """Integrate the fig-5 column with the chosen buoyancy scheme.

    ``case`` selects fig 5D (vertical, gravity on, 1000 yr) or fig 5B (horizontal,
    gravity off, 200 yr). The two differ ONLY in the gravity value and final time.
    ``grav_upstream`` switches the internal-face gravity density from the arithmetic face
    average to Weis (2014, Eq.25) fully-upstream weighting (only affects the vertical case).
    ``weighted_perm`` folds the total mobility into the permeability (harmonic face average of
    lambda*K in m_e, paper Remark 3.2) instead of upwinding it. It is a total-velocity notion,
    so it is INCOMPATIBLE with ``scheme='ppu'`` (which has no single total mobility).
    ``level`` selects the Driesner opensowat ``.vtr`` table refinement level (0..5).
    """
    if case not in CASES:
        raise ValueError(f"case must be one of {list(CASES)}")
    if weighted_perm and scheme == "ppu":
        raise ValueError("weighted_perm (lambda folded into K) is incompatible with scheme='ppu': "
                         "PPU feeds each phase its own potential flux and has no single total "
                         "mobility to weight. Use it with 'hu'.")
    cfg = CASES[case]
    xph_path, xpt_path = table_paths(level)
    table = Table(xph_path, _XPH_FIELDS, a_in=1e-6, b_in=1e-6)    # h[J/kg], p[Pa]
    xpt = Table(xpt_path, {"H": 1e3}, a_in=1.0, b_in=1e-6)        # T[degC], p[Pa] -> H[J/kg]
    geom = make_geom(N, g=cfg["g"])
    if verbose:
        print(f"  table: opensowat l_{level} (.vtr)   gravity-term density: "
              f"{'UPSTREAM (Weis Eq.25)' if grav_upstream else 'arithmetic face-avg'}"
              f"   total mobility: {'weighted_perm (harmonic lambda*K)' if weighted_perm else 'upwind'}"
              f"  (g={cfg['g']:.4g})")

    h_bot = float(xpt("H", T_BOT - 273.15, P_BOT)[0])
    h_top = float(xpt("H", T_TOP - 273.15, P_TOP)[0])
    bbot = boundary_state(table, P_BOT, h_bot)
    btop = boundary_state(table, P_TOP, h_top)

    y = geom.yc
    p0 = (y * P_TOP + (L_COLUMN - y) * P_BOT) / L_COLUMN
    h0 = xpt("H", np.full(N, T_INIT - 273.15), p0)
    x = np.empty(2 * N); x[0::2] = p0; x[1::2] = h0

    plan = build_jac_plan(N)                       # sparsity/colour plan built ONCE
    dt0 = dt if dt is not None else 0.125 * YEAR
    tf = cfg["tf_yr"] * YEAR if n_steps is None else n_steps * dt0
    t = 0.0; dt = dt0; step = 0
    total_it = 0                                    # all Newton iterations (incl. retries)
    while t < tf - 1e-6:
        dt = min(dt, tf - t)
        x_old = x.copy()
        xn, nit, nrm, ok = newton_step(x, x_old, dt, geom, table, bbot, btop, scheme, plan,
                                       grav_upstream=grav_upstream, weighted_perm=weighted_perm)
        total_it += nit
        if not ok and adaptive and dt > dt0 / 64:
            dt *= 0.5                                  # retry with smaller step
            continue
        x = xn; t += dt; step += 1
        if adaptive and ok and nit < 5 and dt < dt0:
            dt = min(dt * 2.0, dt0)
        if verbose and (step % 50 == 0 or not ok):
            print(f"  t={t/YEAR:7.1f} yr  dt={dt/YEAR:.4f}  nit={nit}  |r|={nrm:.1e}"
                  f"  {'' if ok else 'NOT CONVERGED'}")

    avg_it = total_it / step if step else 0.0
    pr = eval_props(table, x[0::2], x[1::2])
    return {"y": y, "p": x[0::2], "h": x[1::2], "T": pr.T, "s_gas": pr.s_v,
            "s_liq": pr.s_l, "rho_mix": pr.rho_mix, "scheme": scheme, "N": N, "case": case,
            "n_steps": step, "total_it": total_it, "avg_it": avg_it,
            "grav_upstream": grav_upstream, "weighted_perm": weighted_perm, "level": level}


# --------------------------------------------------------------------------------------- #
#  Comparison plot vs digitized paper data (CSV)
# --------------------------------------------------------------------------------------- #
def _load_ref_csv(name):
    path = os.path.join(REF_DIR, name)
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    return d[:, 0], d[:, 1]      # distance[km], value


def plot_comparison(results, save_path, case="vertical"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = "vertical" if case == "vertical" else "horizontal"
    panel = "5D (vertical, gravity)" if case == "vertical" else "5B (horizontal, no gravity)"
    refs = {
        "T": (f"fig_5_{tag}_temperature_raw.csv", "Temperature [°C]"),
        "p": (f"fig_5_{tag}_pressured_raw.csv", "Pressure [MPa]"),
        "s_liq": (f"fig_5_{tag}_saturation_liq_raw.csv", "Liquid saturation [-]"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"hu": "tab:blue", "ppu": "tab:red"}
    labels = {"hu": "HU", "ppu": "PPU"}
    for ax, key in zip(axes, ("T", "p", "s_liq")):
        csv, ylabel = refs[key]
        xr, yr = _load_ref_csv(csv)
        ax.plot(xr, yr, "ks", ms=4, mfc="none", label="Weis et al. (digitized)")
        for sch, res in results.items():
            y_km = res["y"] / 1000.0
            val = {"T": res["T"] - 273.15, "p": res["p"] / 1e6, "s_liq": res["s_liq"]}[key]
            ax.plot(y_km, val, "-", color=colors.get(sch, None), lw=1.8,
                    label=f"{labels.get(sch, sch.upper())} (avg. it. {res['avg_it']:.2f}, "
                          f"total it. {res['total_it']})")
        ax.set_xlabel("Distance [km]"); ax.set_ylabel(ylabel)
        ax.set_xlim(0, 2); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Figure {panel} — independent 1D solver vs digitized reference")
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
    i_liq, i_gas, _ = buoyancy_directions(geom, p, pr, "hu")
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Independent 1D finite-volume solver for the Weis (2014) fig-5 benchmark.")
    parser.add_argument("schemes", nargs="?", default="hu,ppu",
                        help="comma-separated buoyancy schemes: hu, ppu "
                             "(default: %(default)s)")
    parser.add_argument("N", nargs="?", type=int, default=200,
                        help="number of cells (default: %(default)s)")
    parser.add_argument("case", nargs="?", default="vertical",
                        choices=("vertical", "horizontal"),
                        help="vertical = fig 5D (gravity on), horizontal = fig 5B (gravity off) "
                             "(default: %(default)s)")
    parser.add_argument("gravity", nargs="?", default="arithmetic",
                        choices=("arithmetic", "upstream"),
                        help="internal-face gravity-term density: 'arithmetic' face-average or "
                             "'upstream' (Weis 2014, Eq.25 fully-upstream) (default: %(default)s)")
    parser.add_argument("mobility", nargs="?", default="upwind",
                        choices=("upwind", "weighted_perm"),
                        help="total-mobility placement in m_e: 'upwind' (separate weight) or "
                             "'weighted_perm' (harmonic lambda*K, paper Remark 3.2; hu only, "
                             "not ppu) (default: %(default)s)")
    parser.add_argument("--level", type=int, default=TABLE_LEVEL, choices=range(6),
                        help="Driesner opensowat .vtr table refinement level 0..5 "
                             "(coarsest..finest) (default: %(default)s)")
    parser.add_argument("--nsteps", type=int, default=None,
                        help="cap the number of time steps (quick test; default: full t_final)")
    args = parser.parse_args()

    selftest()
    schemes = args.schemes.split(",")
    grav_upstream = args.gravity == "upstream"
    weighted_perm = args.mobility == "weighted_perm"
    if weighted_perm and "ppu" in schemes:
        print("  [note] weighted_perm is incompatible with PPU (no single total mobility); "
              "PPU runs with the default 'upwind' mobility, hu with weighted_perm.")
    panel = "5D" if args.case == "vertical" else "5B"
    suffix = ((f"_l{args.level}" if args.level != TABLE_LEVEL else "")
              + ("_gravUP" if grav_upstream else "") + ("_wperm" if weighted_perm else ""))
    out_dir = os.path.join(HERE, "visualization_1D_fig_5")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for sch in schemes:
        sch_wperm = weighted_perm and sch != "ppu"    # PPU has no total mobility -> force upwind
        mob = "weighted_perm" if sch_wperm else "upwind"
        print(f"--- running scheme={sch}, N={args.N}, case={args.case}, gravity={args.gravity}, "
              f"mobility={mob}, level={args.level} (fig {panel}) ---")
        res = run(scheme=sch, N=args.N, case=args.case, n_steps=args.nsteps, verbose=True,
                  grav_upstream=grav_upstream, weighted_perm=sch_wperm, level=args.level)
        results[sch] = res
        band = np.where((res["s_gas"] > 1e-3) & (res["s_gas"] < 1 - 1e-3))[0]
        print(f"  {sch}: T {res['T'][0]-273.15:.0f}->{res['T'][-1]-273.15:.0f} C, "
              f"p {res['p'][0]/1e6:.1f}->{res['p'][-1]/1e6:.2f} MPa, "
              f"band y=[{res['y'][band[0]]:.0f},{res['y'][band[-1]]:.0f}]m" if band.size
              else f"  {sch}: no two-phase band")
    plot_comparison(results, os.path.join(out_dir, f"fig{panel}_compare_N{args.N}{suffix}.png"),
                    case=args.case)


if __name__ == "__main__":
    main()
