"""Independent 1-D finite-volume solver (engine) for the Weis (2014) benchmark -- ONE discretization.

Importable module (no CLI): the ``fig_weis_*`` scripts in this folder drive :func:`run_brine` and
:func:`load_reference` to build the subsection figures. A single H2O-NaCl brine engine reproduces the
whole benchmark: at ``z_NaCl = 0`` it is pure water and gives Fig 4 (single-phase heating) and Fig 5
(two-phase), in either orientation; at ``z > 0`` it is the Fig 6 salt column with an immobile
solid-halite phase. It re-implements, in vectorised numpy, the exact discrete model PorePy assembles
(``FluidBuoyancy`` + the Driesner constitutive tables), so it is a fast, transparent reference.

Three conservation laws (per cell), backward-Euler fully-implicit Newton; primaries p[Pa], h[J/kg],
z_NaCl.  ``(ACC^n - ACC^{n-1})/dt + div(F) = 0``:
  MASS    ACC = V phi rho_mix,                         F = q_T
  SALT    ACC = V phi rho_mix z,                       F = upwind(f_NaCl) q_T - buoy_NaCl
  ENERGY  ACC = V[phi(rho_mix h - p)+(1-phi)rho_s c_s T],  F = -K_e dT/dx + upwind(h) q_T - buoy_h
with rho_mix = sum_g s_g rho_g (incl. halite) and q_T = upwind(lambda_T)(pi + w(rho_ff)).

Buoyancy schemes (``scheme``):
  * ``"hu"``  -- Hybrid Upwinding: viscous mobilities on the total mass flux; the buoyant pair rides
                 the inter-phase gravity flux +/- w(rho_l - rho_v) with the MOBILITY-PRODUCT magnitude
                 lambda_l lambda_v / lambda_T (Lee/Hamon U^HU, = PorePy is_fractional_flow=False). The
                 salt and enthalpy each advect their pair difference (X_l - X_v) / (h_l - h_v).
  * ``"ppu"`` -- Phase-Potential Upwinding: each phase rides its own potential Psi_g; buoyancy intrinsic.
  ``weighted_perm=True`` (HU-mwp) folds lambda_T into the transmissibilities (harmonic face lambda*K,
  paper Remark 3.2) instead of upwinding a separate face total mobility; the buoyant term is unchanged.
  ``case`` = 'horizontal' (g=0, Fig 5B/6) or 'vertical' (g on, Fig 5D); ``grav_upstream`` selects the
  Weis Eq.25 fully-upstream gravity density; ``lag_upwind`` freezes the advective weights per step.

**Strict SI** (Pa, J/kg, m, s, kg, K): g=9.80665, K=1e-15 m^2, K_e=2.0 W/mK, c_s=880 J/kgK, rho_s=2700,
phi=0.1. The constitutive closure is sampled from the Driesner ``opensowat_x{ph,pt}_l_{L}.vtr``
tables (axes z_NaCl, h[MJ/kg] / T[degC], p[MPa]) at refinement level L in 0..5 via O(1) trilinear
interpolation; SI<->table unit conversion is handled inside the sampler.
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
# The constitutive .vtr tables and digitized reference CSVs live in the parent geothermal_flow
# directory (this engine module lives in the subsection_4_1/ sub-folder).
_PARENT = os.path.dirname(os.path.dirname(HERE))   # geothermal_flow/ (two levels up)
VTK_DIR = os.path.join(_PARENT, "model_configuration", "constitutive_description",
                       "driesner_vtk_files")
REF_DIR = os.path.join(_PARENT, "benchmark_figures_data")

TABLE_LEVEL = 3       # Driesner opensowat table refinement level: 0 (coarsest) .. finer with level


def table_paths(level=TABLE_LEVEL):
    """Absolute paths of the xph (z, h[MJ/kg], p[MPa]) and xpt (z, T[degC], p[MPa]) Driesner
    ``.vtr`` tables at refinement ``level`` (0..5)."""
    xph = os.path.join(VTK_DIR, f"opensowat_xph_l_{level}.vtr")
    xpt = os.path.join(VTK_DIR, f"opensowat_xpt_l_{level}.vtr")
    return xph, xpt


# High-resolution PURE-WATER (z=0) Driesner tables: same field schema, units, and (h, p) ranges as the
# opensowat brine tables, but ~6x finer in enthalpy (1000 vs 160 h-nodes). Used for the Fig-6 pure-water
# column, where the coarse brine h-grid produces spurious wiggles in the two-phase liquid saturation.
# Not level-indexed; the composition axis is a 2-node [0, 1e-5] stub, i.e. the z=0 slice.
PUREWATER_XPH = os.path.join(VTK_DIR, "purewater_xph.vtr")
PUREWATER_XPT = os.path.join(VTK_DIR, "purewater_xpt.vtr")

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
DT0 = 0.25 * YEAR     # nominal time step; also the reference used to row-scale residuals to O(1)

# Reference scales used to row-scale the mass/energy residuals to O(1). Without this the
# mass (~kg/s) and energy (~W) equations differ by ~1e13 and the Jacobian is unsolvable.
RHO_REF = 800.0       # kg/m^3
T_REF = 500.0         # K

# Gravity-term density weighting on internal faces (see ``residual_brine``), run_brine(grav_upstream=):
#   grav_upstream=False (default) -> face average 0.5*(rho_i + rho_{i+1})  (consistent, Rem.gc)
#   grav_upstream=True            -> Weis (2014, Eq.25 p.352): fluid props in the gravity term
#                                    taken from the lagged phase-upwind node (fully-upstream).
# Identical when g=0 (horizontal 5B); only changes the vertical (5D) case.

# fig-5D boundary / initial data (SI)
P_BOT, P_TOP = 20.0e6, 1.0e6      # Pa  (inlet y=0 / outlet y=2000)
T_BOT, T_TOP = 673.15, 423.15     # K
T_INIT = 423.15                   # K, constant IC


# --------------------------------------------------------------------------------------- #
#  Brine closure (H2O-NaCl): three-phase liquid/vapor/immobile-halite, from the 3-D table
# --------------------------------------------------------------------------------------- #


@dataclass
class PropsBrine:
    rho_l: np.ndarray; rho_v: np.ndarray; rho_h: np.ndarray
    s_l: np.ndarray; s_v: np.ndarray; s_h: np.ndarray
    h_l: np.ndarray; h_v: np.ndarray; h_h: np.ndarray
    Xl: np.ndarray; Xv: np.ndarray            # NaCl mass fraction in liquid / vapor (X_h = 1)
    T: np.ndarray
    rho_mix: np.ndarray                        # s_l rho_l + s_v rho_v + s_h rho_h (incl. halite)
    lam_T: np.ndarray
    mm_l: np.ndarray; mm_v: np.ndarray
    salt_mob: np.ndarray                       # NaCl mass mobility  = Xl mm_l + Xv mm_v
    adv_h: np.ndarray                          # enthalpy mass mobility = h_l mm_l + h_v mm_v
    rho_ff: np.ndarray                         # fractional-flow weighted density (buoyancy V_T term)


def _make_vtksampler(path):
    """Build the unified VTKSampler without importing the full porepy package: the ``obl_sampler``
    subpackage needs only numpy / scipy / pyvista, so add the geothermal_flow dir to the path and
    import it as a top-level package (keeps ``import weis_1d_solver`` fast for the figure pipeline)."""
    import sys
    if _PARENT not in sys.path:
        sys.path.insert(0, _PARENT)
    from obl_sampler import VTKSampler
    return VTKSampler(path)


def _xph_fmap(amr):
    """weis property key -> (VTKSampler field name, SI value scale). The adapted ``.vtu`` stores
    enthalpy in MJ/kg and names temperature ``T``; the rectilinear ``.vtr`` stores kJ/kg and
    ``Temperature``."""
    hs = 1e6 if amr else 1e3
    tname = "T" if amr else "Temperature"
    return {"Rho_l": ("Rho_l", 1.0), "Rho_v": ("Rho_v", 1.0), "Rho_h": ("Rho_h", 1.0),
            "H_l": ("H_l", hs), "H_v": ("H_v", hs), "H_h": ("H_h", hs),
            "S_v": ("S_v", 1.0), "S_h": ("S_h", 1.0), "Xl": ("Xl", 1.0), "Xv": ("Xv", 1.0),
            "mu_l": ("mu_l", 1.0), "mu_v": ("mu_v", 1.0), "Temperature": (tname, 1.0)}


class XphSampler:
    """weis adaptation to the VTKSampler API for the (z_NaCl, h, p) property table. SI in (p[Pa],
    h[J/kg], z) and SI out. ``props()`` returns ``{weis key: value(N,)}``; ``props(grads=True)`` also
    returns ``{weis key: (N,3)}`` whose columns are the sampler's analytic derivatives d/dp, d/dh, d/dz.
    The ``a/b/c_{min,max}`` attributes (h/p/z range in SI) mirror the old ``Table`` for the Newton clip."""

    def __init__(self, path, fmap):
        self.s = _make_vtksampler(path)
        self.s.conversion_factors = (1.0, 1e-6, 1e-6)      # (z, h[J/kg->MJ/kg], p[Pa->MPa])
        self.fmap = fmap
        b = self.s.bounds                                  # (zmin,zmax, hmin,hmax, pmin,pmax) table units
        self.c_min, self.c_max = float(b[0]), float(b[1])               # z [-]
        self.a_min, self.a_max = float(b[2]) * 1e6, float(b[3]) * 1e6   # h [J/kg]
        self.b_min, self.b_max = float(b[4]) * 1e6, float(b[5]) * 1e6   # p [Pa]

    def props(self, p, h, z, grads=False):
        p = np.atleast_1d(np.asarray(p, float)); h = np.atleast_1d(np.asarray(h, float))
        z = np.broadcast_to(np.atleast_1d(np.asarray(z, float)), h.shape)
        self.s.sample_at(np.column_stack([z, h, p]))
        pd = self.s.sampled_could.point_data
        vals = {k: pd[fn] * sc for k, (fn, sc) in self.fmap.items()}
        if not grads:
            return vals
        # grad_Field columns are (d/dz, d/dh[J/kg], d/dp[Pa]); reorder to (d/dp, d/dh, d/dz), SI-scale.
        grd = {k: pd["grad_" + fn][:, [2, 1, 0]] * sc for k, (fn, sc) in self.fmap.items()}
        return vals, grd


class XptSampler:
    """weis adaptation to the VTKSampler API for the (z_NaCl, T, p) enthalpy table:
    ``enth(T[K], p[Pa], z) -> h[J/kg]`` (used only for the IC/BC enthalpy)."""

    def __init__(self, path, h_scale=1e3):
        self.s = _make_vtksampler(path)
        self.s.conversion_factors = (1.0, 1.0, 1e-6)       # (z, T[degC], p[Pa->MPa])
        self.s.translation_factors = (0.0, -273.15, 0.0)   # T: K -> degC
        self.h_scale = h_scale

    def enth(self, TK, p, z):
        TK = np.atleast_1d(np.asarray(TK, float)); p = np.atleast_1d(np.asarray(p, float))
        z = np.broadcast_to(np.atleast_1d(np.asarray(z, float)), TK.shape)
        self.s.sample_at(np.column_stack([z, TK, p]))
        return self.s.sampled_could.point_data["H"] * self.h_scale


def eval_props_brine(table, p, h, z):
    """Three-phase closure from the 3-D xph table at overall NaCl composition ``z``.

    Halite is a table-provided saturation (as in the porepy DriesnerModelConfiguration): it enters
    rho_mix, and blocks the pore space through the rel-perm k_rl + k_rv = 1 - s_h; it is immobile
    (k_rh = 0), so it advects no mass -- its NaCl is carried only in the accumulation via rho_mix z.
    """
    s = table.props(p, h, z)
    rho_l = s["Rho_l"]; rho_v = s["Rho_v"]; rho_h = s["Rho_h"]
    s_v = np.clip(s["S_v"], 0.0, 1.0)
    s_h = np.clip(s["S_h"], 0.0, 1.0)
    s_l = np.clip(1.0 - s_v - s_h, 0.0, 1.0)
    h_l = s["H_l"]; h_v = s["H_v"]; h_h = s["H_h"]
    Xl = np.clip(s["Xl"], 0.0, 1.0); Xv = np.clip(s["Xv"], 0.0, 1.0)
    mu_l = s["mu_l"]; mu_v = s["mu_v"]; T = s["Temperature"]

    # Weis (2014) rel-perm with halite pore blocking (mirror of DriesnerModelConfiguration):
    #   k_rl = max((s_l - 0.3(1-s_h))/0.7, 0),  k_rv = (1-s_h) - k_rl,  k_rh = 0.
    # s_h = 0 reduces to the pure-water fig-5 closure k_rl + k_rv = 1.
    kr_l = np.maximum((s_l - S_R_LIQ * (1.0 - s_h)) / (1.0 - S_R_LIQ), 0.0)
    kr_v = np.maximum((1.0 - s_h) - kr_l, 0.0)
    mm_l = rho_l * kr_l / mu_l
    mm_v = rho_v * kr_v / mu_v
    lam_T = mm_l + mm_v
    rho_mix = s_l * rho_l + s_v * rho_v + s_h * rho_h
    salt_mob = Xl * mm_l + Xv * mm_v
    adv_h = h_l * mm_l + h_v * mm_v
    inv = 1.0 / np.where(lam_T > 0.0, lam_T, 1.0)
    rho_ff = (mm_l * rho_l + mm_v * rho_v) * inv         # f_l rho_l + f_v rho_v (buoyancy V_T term)
    return PropsBrine(rho_l, rho_v, rho_h, s_l, s_v, s_h, h_l, h_v, h_h, Xl, Xv, T,
                      rho_mix, lam_T, mm_l, mm_v, salt_mob, adv_h, rho_ff)


def eval_props_and_grads(table, p, h, z):
    """:func:`eval_props_brine` PLUS the analytic derivative of every derived property wrt (p, h, z),
    hand-chained from the sampler's raw-field gradients (NO automatic differentiation). Returns
    ``(pr, d)`` where ``pr`` is the usual :class:`PropsBrine` and ``d`` is a dict ``{name: (N,3)}`` with
    columns (d/dp, d/dh, d/dz). Clips and max() contribute a frozen 0/1 activity mask."""
    vals, g = table.props(p, h, z, grads=True)
    rho_l = vals["Rho_l"]; rho_v = vals["Rho_v"]; rho_h = vals["Rho_h"]
    h_l = vals["H_l"]; h_v = vals["H_v"]; h_h = vals["H_h"]
    mu_l = vals["mu_l"]; mu_v = vals["mu_v"]; T = vals["Temperature"]
    drho_l = g["Rho_l"]; drho_v = g["Rho_v"]; drho_h = g["Rho_h"]
    dh_l = g["H_l"]; dh_v = g["H_v"]; dh_h = g["H_h"]; dmu_l = g["mu_l"]; dmu_v = g["mu_v"]; dT = g["Temperature"]

    def col(a):                                            # (N,) -> (N,1) for broadcasting against (N,3)
        return a[:, None]

    def msk(raw):                                          # activity mask of a clip to [0,1], as (N,1)
        return ((raw > 0.0) & (raw < 1.0)).astype(float)[:, None]

    s_v = np.clip(vals["S_v"], 0.0, 1.0); ds_v = g["S_v"] * msk(vals["S_v"])
    s_h = np.clip(vals["S_h"], 0.0, 1.0); ds_h = g["S_h"] * msk(vals["S_h"])
    sl_pre = 1.0 - s_v - s_h; s_l = np.clip(sl_pre, 0.0, 1.0)
    ds_l = (-ds_v - ds_h) * msk(sl_pre)
    Xl = np.clip(vals["Xl"], 0.0, 1.0); dXl = g["Xl"] * msk(vals["Xl"])
    Xv = np.clip(vals["Xv"], 0.0, 1.0); dXv = g["Xv"] * msk(vals["Xv"])

    arg = (s_l - S_R_LIQ * (1.0 - s_h)) / (1.0 - S_R_LIQ)
    kr_l = np.maximum(arg, 0.0)
    dkr_l = ((ds_l + S_R_LIQ * ds_h) / (1.0 - S_R_LIQ)) * col((arg > 0.0).astype(float))
    arg2 = (1.0 - s_h) - kr_l; kr_v = np.maximum(arg2, 0.0)
    dkr_v = (-ds_h - dkr_l) * col((arg2 > 0.0).astype(float))

    mm_l = rho_l * kr_l / mu_l
    dmm_l = (drho_l * col(kr_l) + col(rho_l) * dkr_l) / col(mu_l) - col(mm_l / mu_l) * dmu_l
    mm_v = rho_v * kr_v / mu_v
    dmm_v = (drho_v * col(kr_v) + col(rho_v) * dkr_v) / col(mu_v) - col(mm_v / mu_v) * dmu_v
    lam_T = mm_l + mm_v; dlam_T = dmm_l + dmm_v
    rho_mix = s_l * rho_l + s_v * rho_v + s_h * rho_h
    drho_mix = (ds_l * col(rho_l) + col(s_l) * drho_l + ds_v * col(rho_v) + col(s_v) * drho_v
                + ds_h * col(rho_h) + col(s_h) * drho_h)
    salt_mob = Xl * mm_l + Xv * mm_v
    dsalt_mob = dXl * col(mm_l) + col(Xl) * dmm_l + dXv * col(mm_v) + col(Xv) * dmm_v
    adv_h = h_l * mm_l + h_v * mm_v
    dadv_h = dh_l * col(mm_l) + col(h_l) * dmm_l + dh_v * col(mm_v) + col(h_v) * dmm_v
    pos = lam_T > 0.0; inv = 1.0 / np.where(pos, lam_T, 1.0)
    num = mm_l * rho_l + mm_v * rho_v
    dnum = dmm_l * col(rho_l) + col(mm_l) * drho_l + dmm_v * col(rho_v) + col(mm_v) * drho_v
    rho_ff = num * inv
    drho_ff = col(inv) * (dnum - col(rho_ff) * dlam_T) * col(pos.astype(float))

    pr = PropsBrine(rho_l, rho_v, rho_h, s_l, s_v, s_h, h_l, h_v, h_h, Xl, Xv, T,
                    rho_mix, lam_T, mm_l, mm_v, salt_mob, adv_h, rho_ff)
    d = {"rho_l": drho_l, "rho_v": drho_v, "rho_h": drho_h, "s_l": ds_l, "s_v": ds_v, "s_h": ds_h,
         "h_l": dh_l, "h_v": dh_v, "h_h": dh_h, "Xl": dXl, "Xv": dXv, "T": dT,
         "rho_mix": drho_mix, "lam_T": dlam_T, "mm_l": dmm_l, "mm_v": dmm_v,
         "salt_mob": dsalt_mob, "adv_h": dadv_h, "rho_ff": drho_ff}
    return pr, d


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


def _advect(cell_q, direction):
    """First-order upwind primitive, mirroring hamon_2d_solver._advect: advected CELL quantity +
    advecting face direction -> upstream-cell face value (lower cell where direction>=0). Identical
    to ``cell_q[_upwind_idx(direction)]``; the buoyancy pair terms touch faces ONLY through this."""
    return np.where(direction >= 0.0, cell_q[:-1], cell_q[1:])


def _harmonic_face(lam):
    """Harmonic average of a cell field ``lam`` onto internal faces: 2 lL lR/(lL+lR), 0 where the
    sum vanishes. This is the joint lambda*K face transmissibility weight of the mobility-weighted
    (HU-mwp) discretisation (paper Remark 3.2)."""
    lam_L = lam[:-1]; lam_R = lam[1:]
    s = lam_L + lam_R
    return np.where(s > 0.0, 2.0 * lam_L * lam_R / np.where(s > 0.0, s, 1.0), 0.0)


def buoyancy_directions(geom, p, pr, scheme):
    """Per-internal-face LAGGED upstream cell indices ``(i_liq, i_gas, i_tot)`` plus the lagged
    signed buoyancy direction field ``w_dir`` (HU only; ``None`` for PPU).

    All are frozen per time step (evaluated on the old state), following Weis (2014, p.354,
    "we use the old velocity field to define the upwind nodes"). ``i_liq``/``i_gas`` drive the
    buoyancy term; ``i_tot`` is the total-velocity direction used for the upstream gravity density
    in ``V_T`` -- it MUST be lagged, because upwinding rho *inside* V_T by the current V_T would
    make the velocity discontinuous at flow reversal and break Newton. ``w_dir = G(rho_l - rho_v)``
    is the lagged advecting direction the HU folded-Gamma feeds to :func:`_advect`.

    hu:   liquid rides +ddf(rho_l-rho_v), gas rides -ddf  (opposite inter-phase directions).
    ppu:  each phase rides its own potential Psi_g = T_f(p_L-p_U) - K A rho_g g.
    Phase order matches PorePy phases = [liq, gas].
    """
    rho_l_f = 0.5 * (pr.rho_l[:-1] + pr.rho_l[1:])
    rho_v_f = 0.5 * (pr.rho_v[:-1] + pr.rho_v[1:])
    rho_ff_f = 0.5 * (pr.rho_ff[:-1] + pr.rho_ff[1:])
    i_tot = _upwind_idx(geom.Tf * (p[:-1] - p[1:]) - geom.GA * rho_ff_f)   # lagged total-velocity
    if scheme == "hu":
        ddf = -geom.GA * (rho_l_f - rho_v_f)     # inter-phase gravity flux ddf(rho_l-rho_v) = w_ab
        dir_liq, dir_gas, w_dir = ddf, -ddf, ddf
    elif scheme == "ppu":
        dp = geom.Tf * (p[:-1] - p[1:])
        dir_liq = dp - geom.GA * rho_l_f         # Psi_liq ~ -K(grad p - rho_l g)
        dir_gas = dp - geom.GA * rho_v_f         # Psi_gas
        w_dir = None                             # PPU has no single buoyancy direction
    else:
        raise ValueError(f"unknown scheme {scheme!r}; use 'hu' or 'ppu'")
    return _upwind_idx(dir_liq), _upwind_idx(dir_gas), i_tot, w_dir



# Per-case settings: gravity along the column and the benchmark final time.
#   "vertical"   = fig 5D: gravity ON,  t_final = 1000 yr.
#   "horizontal" = fig 5B: gravity OFF (perpendicular to flow), t_final = 200 yr.
CASES = {"vertical": dict(g=G, tf_yr=1000.0), "horizontal": dict(g=0.0, tf_yr=200.0)}




# --------------------------------------------------------------------------------------- #
#  Sparse coloured finite-difference Jacobian (block-tridiagonal; nvar=3 -> 9 colours)
# --------------------------------------------------------------------------------------- #
def build_jac_plan(N, nvar=3, scales=(1.0e6, 1.0e5, 1.0)):
    """Precompute the (3*nvar)-colour FD-Jacobian sparsity ONCE (block-tridiagonal, ``nvar`` vars per
    cell, interleaved). nvar=3 -> brine [p, h, z] (9 colours)."""
    rows_of_col = []
    for k in range(N):
        for _v in range(nvar):
            rows = []
            for kk in (k - 1, k, k + 1):
                if 0 <= kk < N:
                    rows += [nvar * kk + j for j in range(nvar)]
            rows_of_col.append(np.array(rows, dtype=np.intp))
    ncol = 3 * nvar
    color = np.array([(k % 3) * nvar + v for k in range(N) for v in range(nvar)])
    n = nvar * N
    col_perturb, gat_rows, gat_owner = [], [], []
    for c in range(ncol):
        cols_c = np.where(color == c)[0]
        rs = [rows_of_col[j] for j in cols_c]
        ow = [np.full(rows_of_col[j].size, j, dtype=np.intp) for j in cols_c]
        col_perturb.append(cols_c)
        gat_rows.append(np.concatenate(rs))      # rows touched by this colour (gather)
        gat_owner.append(np.concatenate(ow))     # owning column (for eps + COO col)
    all_rows = np.concatenate(gat_rows)
    all_cols = np.concatenate(gat_owner)
    sc = np.array([scales[i % nvar] for i in range(n)])   # p ~ MPa, h ~ 1e5 J/kg, z ~ 1
    # banded (LAPACK) storage: block-tridiagonal interleaved -> bandwidth l,u (=2*nvar-1).
    l = int((all_rows - all_cols).max())
    u = int((all_cols - all_rows).max())
    bpos = u + all_rows - all_cols                       # ab[u+i-j, j] = A[i,j]
    return dict(n=n, nvar=nvar, ncol=ncol, col_perturb=col_perturb, gat_rows=gat_rows,
                gat_owner=gat_owner, all_rows=all_rows, all_cols=all_cols, scale=sc,
                l=l, u=u, bpos=bpos)


def jacobian_fd(x, r0, args, plan, eps_rel=1e-7, resfn=None):
    """Coloured FD Jacobian in LAPACK banded storage (ab, shape (l+u+1, n)). ``resfn`` = the residual
    to differentiate (default :func:`residual_brine`)."""
    if resfn is None:
        resfn = residual_brine
    n = plan["n"]
    eps = eps_rel * np.maximum(np.abs(x), plan["scale"])
    parts = []
    for c in range(plan["ncol"]):
        cols_c = plan["col_perturb"][c]
        dx = np.zeros(n); dx[cols_c] = eps[cols_c]
        dr = resfn(x + dx, *args) - r0
        parts.append(dr[plan["gat_rows"][c]] / eps[plan["gat_owner"][c]])   # vectorised
    ab = np.zeros((plan["l"] + plan["u"] + 1, n))
    ab[plan["bpos"], plan["all_cols"]] = np.concatenate(parts)
    return ab


def _band_add(ab, u, cr, cc, B):
    """Add a 3x3 block ``B`` (rows = mass/salt/energy residual, cols = p/h/z) for (cell cr, cell cc)
    into LAPACK banded storage ``ab[u + row - col, col]``."""
    for a in range(3):
        row = 3 * cr + a
        for b in range(3):
            col = 3 * cc + b
            ab[u + row - col, col] += B[a, b]


def jacobian_analytic(x, dt, geom, table, bleft, bright, scheme, ug, ud, ut, w_dir,
                      grav_upstream, weighted_perm, lag_upwind, lam_face_old, plan):
    """Hand-coded analytic Jacobian (banded) of :func:`residual_brine` -- NO finite differences, NO AD.
    Chains :func:`eval_props_and_grads` through the flux assembly with FROZEN upwind/buoyancy
    directions (standard for an upwind-FV Jacobian). Covers the HU scheme in FULL: horizontal and
    vertical (mobility-product buoyancy pair), HU-mwp (``weighted_perm``), and ``grav_upstream``, plus
    the HU boundary faces, and the full PPU scheme (per-phase potential upwinding)."""
    N = geom.N
    p = x[0::3]; h = x[1::3]; z = x[2::3]
    pr, dP = eval_props_and_grads(table, p, h, z)
    u = plan["u"]; ab = np.zeros((plan["l"] + u + 1, N * 3))
    rs = np.array([1.0 / geom.ms, 1.0 / geom.ms, 1.0 / geom.es])[:, None]
    ep = np.array([1.0, 0.0, 0.0]); eh = np.array([0.0, 1.0, 0.0]); ez = np.array([0.0, 0.0, 1.0])
    GA, Tf, TFf = geom.GA, geom.Tf, geom.TFf

    def route(is_L, aL, aR, v):                          # accumulate v into the L or R block in place
        if is_L:
            aL += v
        else:
            aR += v

    # --- accumulation (diagonal 3x3 blocks) ---
    for i in range(N):
        B = np.empty((3, 3))
        B[0] = geom.Vcell * PHI * dP["rho_mix"][i] / dt
        B[1] = geom.Vcell * PHI * (z[i] * dP["rho_mix"][i] + pr.rho_mix[i] * ez) / dt
        B[2] = geom.Vcell * (PHI * (h[i] * dP["rho_mix"][i] + pr.rho_mix[i] * eh - ep)
                             + (1 - PHI) * RHO_S * C_S * dP["T"][i]) / dt
        _band_add(ab, u, i, i, B * rs)

    idx = np.arange(N - 1)
    dp_face = p[:-1] - p[1:]
    rho_l_f = 0.5 * (pr.rho_l[:-1] + pr.rho_l[1:]); rho_v_f = 0.5 * (pr.rho_v[:-1] + pr.rho_v[1:])

    if scheme == "ppu":                                  # per-phase potential upwinding
        Tb, TFb = geom.Tb, geom.TFb
        if lag_upwind:
            iu_l_arr, iu_v_arr = ug, ud
        else:
            iu_l_arr = _upwind_idx(Tf * dp_face - GA * rho_l_f)
            iu_v_arr = _upwind_idx(Tf * dp_face - GA * rho_v_f)
        for f in range(N - 1):
            L = f; R = f + 1; il = iu_l_arr[f]; iv = iu_v_arr[f]
            dPsl_L = Tf * ep.copy(); dPsl_R = -Tf * ep.copy()                   # Psi_l = Tf(pL-pR)-GA*rho_l_p
            dPsv_L = Tf * ep.copy(); dPsv_R = -Tf * ep.copy()
            if grav_upstream:
                route(il == L, dPsl_L, dPsl_R, -GA * dP["rho_l"][il]); Psl = Tf * dp_face[f] - GA * pr.rho_l[il]
                route(iv == L, dPsv_L, dPsv_R, -GA * dP["rho_v"][iv]); Psv = Tf * dp_face[f] - GA * pr.rho_v[iv]
            else:
                dPsl_L -= GA * 0.5 * dP["rho_l"][L]; dPsl_R -= GA * 0.5 * dP["rho_l"][R]; Psl = Tf * dp_face[f] - GA * rho_l_f[f]
                dPsv_L -= GA * 0.5 * dP["rho_v"][L]; dPsv_R -= GA * 0.5 * dP["rho_v"][R]; Psv = Tf * dp_face[f] - GA * rho_v_f[f]
            mml = pr.mm_l[il]; mmv = pr.mm_v[iv]; dmml = dP["mm_l"][il]; dmmv = dP["mm_v"][iv]
            ql = pr.Xl[il] * mml; qv = pr.Xv[iv] * mmv; el = pr.h_l[il] * mml; ev = pr.h_v[iv] * mmv
            dql = dP["Xl"][il] * mml + pr.Xl[il] * dmml; dqv = dP["Xv"][iv] * mmv + pr.Xv[iv] * dmmv
            del_ = dP["h_l"][il] * mml + pr.h_l[il] * dmml; dev_ = dP["h_v"][iv] * mmv + pr.h_v[iv] * dmmv
            dm_L = dPsl_L * mml + dPsv_L * mmv; dm_R = dPsl_R * mml + dPsv_R * mmv
            route(il == L, dm_L, dm_R, Psl * dmml); route(iv == L, dm_L, dm_R, Psv * dmmv)
            ds_L = dPsl_L * ql + dPsv_L * qv; ds_R = dPsl_R * ql + dPsv_R * qv
            route(il == L, ds_L, ds_R, Psl * dql); route(iv == L, ds_L, ds_R, Psv * dqv)
            de_L = dPsl_L * el + dPsv_L * ev + TFf * dP["T"][L]; de_R = dPsl_R * el + dPsv_R * ev - TFf * dP["T"][R]
            route(il == L, de_L, de_R, Psl * del_); route(iv == L, de_L, de_R, Psv * dev_)
            BLL = np.array([dm_L, ds_L, de_L]); BLR = np.array([dm_R, ds_R, de_R])
            _band_add(ab, u, L, L, BLL * rs); _band_add(ab, u, L, R, BLR * rs)
            _band_add(ab, u, R, L, -BLL * rs); _band_add(ab, u, R, R, -BLR * rs)
        # boundary faces (PPU): each phase upwinds between the fixed boundary node and the end cell
        Pslb = Tb * (bleft.p - p[0]) - GA * bleft.pr.rho_l[0]; Psvb = Tb * (bleft.p - p[0]) - GA * bleft.pr.rho_v[0]
        Bl = np.zeros((3, 3))
        for Psi, cellL, bnode, rho in ((Pslb, "l", bleft, "rho_l"), (Psvb, "v", bleft, "rho_v")):
            mm = "mm_" + cellL; X = "Xl" if cellL == "l" else "Xv"; hh = "h_" + cellL
            if Psi >= 0.0:
                mmb = getattr(bnode.pr, mm)[0]; q = getattr(bnode.pr, X)[0] * mmb; e = getattr(bnode.pr, hh)[0] * mmb
                Bl[0] += -Tb * mmb * ep; Bl[1] += -Tb * q * ep; Bl[2] += -Tb * e * ep
            else:
                mm0 = getattr(pr, mm)[0]; dmm0 = dP[mm][0]; q = getattr(pr, X)[0] * mm0; e = getattr(pr, hh)[0] * mm0
                dq = dP[X][0] * mm0 + getattr(pr, X)[0] * dmm0; de = dP[hh][0] * mm0 + getattr(pr, hh)[0] * dmm0
                Bl[0] += -Tb * mm0 * ep + Psi * dmm0; Bl[1] += -Tb * q * ep + Psi * dq; Bl[2] += -Tb * e * ep + Psi * de
        Bl[2] += -TFb * dP["T"][0]
        _band_add(ab, u, 0, 0, -Bl * rs)
        Pslt = Tb * (p[-1] - bright.p) - GA * bright.pr.rho_l[0]; Psvt = Tb * (p[-1] - bright.p) - GA * bright.pr.rho_v[0]
        Br = np.zeros((3, 3))
        for Psi, cellL, bnode in ((Pslt, "l", bright), (Psvt, "v", bright)):
            mm = "mm_" + cellL; X = "Xl" if cellL == "l" else "Xv"; hh = "h_" + cellL
            if Psi >= 0.0:                                # own end cell N-1
                mmv = getattr(pr, mm)[-1]; dmmv = dP[mm][-1]; q = getattr(pr, X)[-1] * mmv; e = getattr(pr, hh)[-1] * mmv
                dq = dP[X][-1] * mmv + getattr(pr, X)[-1] * dmmv; de = dP[hh][-1] * mmv + getattr(pr, hh)[-1] * dmmv
                Br[0] += Tb * mmv * ep + Psi * dmmv; Br[1] += Tb * q * ep + Psi * dq; Br[2] += Tb * e * ep + Psi * de
            else:                                        # fixed bright node
                mmb = getattr(bnode.pr, mm)[0]; q = getattr(bnode.pr, X)[0] * mmb; e = getattr(bnode.pr, hh)[0] * mmb
                Br[0] += Tb * mmb * ep; Br[1] += Tb * q * ep; Br[2] += Tb * e * ep
        Br[2] += TFb * dP["T"][-1]
        _band_add(ab, u, N - 1, N - 1, Br * rs)
        return ab

    rho_ff_f = 0.5 * (pr.rho_ff[:-1] + pr.rho_ff[1:])
    rho_ff_g = pr.rho_ff[ut] if grav_upstream else rho_ff_f
    V_T = Tf * dp_face - GA * rho_ff_g
    up = ut if lag_upwind else np.where(V_T >= 0.0, idx, idx + 1)

    for f in range(N - 1):
        L = f; R = f + 1
        dm_L = np.zeros(3); dm_R = np.zeros(3); ds_L = np.zeros(3); ds_R = np.zeros(3)
        de_L = np.zeros(3); de_R = np.zeros(3)
        dVT_L = Tf * ep.copy(); dVT_R = -Tf * ep.copy()                # V_T = Tf(p_L-p_R) - GA*rho_ff_g
        if grav_upstream:
            route(ut[f] == L, dVT_L, dVT_R, -GA * dP["rho_ff"][ut[f]])
        else:
            dVT_L -= GA * 0.5 * dP["rho_ff"][L]; dVT_R -= GA * 0.5 * dP["rho_ff"][R]
        uc = up[f]; ucL = uc == L
        if weighted_perm:                                             # HU-mwp: F_mass = V_T * lam_face
            if lag_upwind and lam_face_old is not None:
                lam_face = float(lam_face_old[f]); dlfL = np.zeros(3); dlfR = np.zeros(3)
            else:
                lL = pr.lam_T[L]; lR = pr.lam_T[R]; s = lL + lR
                if s > 0.0:
                    lam_face = 2.0 * lL * lR / s                       # d(2 lL lR/(lL+lR)): 2 lR^2/s^2, 2 lL^2/s^2
                    dlfL = (2.0 * lR * lR / (s * s)) * dP["lam_T"][L]
                    dlfR = (2.0 * lL * lL / (s * s)) * dP["lam_T"][R]
                else:
                    lam_face = 0.0; dlfL = np.zeros(3); dlfR = np.zeros(3)
            F_mass = V_T[f] * lam_face
            dm_L = dVT_L * lam_face + V_T[f] * dlfL; dm_R = dVT_R * lam_face + V_T[f] * dlfR
            lamu = pr.lam_T[uc]; invu = (1.0 / lamu) if lamu > 0.0 else 0.0
            hbar = pr.adv_h[uc] * invu; xbar = pr.salt_mob[uc] * invu
            dhbar = (dP["adv_h"][uc] * lamu - pr.adv_h[uc] * dP["lam_T"][uc]) * invu * invu
            dxbar = (dP["salt_mob"][uc] * lamu - pr.salt_mob[uc] * dP["lam_T"][uc]) * invu * invu
            de_L = hbar * dm_L; de_R = hbar * dm_R; ds_L = xbar * dm_L; ds_R = xbar * dm_R
            route(ucL, de_L, de_R, dhbar * F_mass); route(ucL, ds_L, ds_R, dxbar * F_mass)
        else:                                                         # HU: F_mass = V_T * lam_T[up]
            dm_L = dVT_L * pr.lam_T[uc]; dm_R = dVT_R * pr.lam_T[uc]
            ds_L = dVT_L * pr.salt_mob[uc]; ds_R = dVT_R * pr.salt_mob[uc]
            de_L = dVT_L * pr.adv_h[uc]; de_R = dVT_R * pr.adv_h[uc]
            route(ucL, dm_L, dm_R, V_T[f] * dP["lam_T"][uc])
            route(ucL, ds_L, ds_R, V_T[f] * dP["salt_mob"][uc])
            route(ucL, de_L, de_R, V_T[f] * dP["adv_h"][uc])
        de_L += TFf * dP["T"][L]; de_R -= TFf * dP["T"][R]            # F_four = TFf*(T[L]-T[R])
        if GA != 0.0:                                                 # mobility-product buoyancy pair
            w_flux = -GA * (rho_l_f[f] - rho_v_f[f])
            dwfL = -GA * 0.5 * (dP["rho_l"][L] - dP["rho_v"][L])
            dwfR = -GA * 0.5 * (dP["rho_l"][R] - dP["rho_v"][R])
            cl = L if w_dir[f] >= 0.0 else R                          # _advect(., w_dir)
            cv = L if -w_dir[f] >= 0.0 else R                         # _advect(., -w_dir)
            a = pr.mm_l[cl]; b = pr.mm_v[cv]; Gam = a + b + 1e-30; common = a * b / Gam
            dca = (b * b) / (Gam * Gam); dcb = (a * a) / (Gam * Gam)  # d(ab/(a+b))/da, /db
            dcom_L = np.zeros(3); dcom_R = np.zeros(3)
            route(cl == L, dcom_L, dcom_R, dca * dP["mm_l"][cl])
            route(cv == L, dcom_L, dcom_R, dcb * dP["mm_v"][cv])
            Hd = pr.h_l[cl] - pr.h_v[cv]; Xd = pr.Xl[cl] - pr.Xv[cv]
            dHd_L = np.zeros(3); dHd_R = np.zeros(3); dXd_L = np.zeros(3); dXd_R = np.zeros(3)
            route(cl == L, dHd_L, dHd_R, dP["h_l"][cl]); route(cv == L, dHd_L, dHd_R, -dP["h_v"][cv])
            route(cl == L, dXd_L, dXd_R, dP["Xl"][cl]); route(cv == L, dXd_L, dXd_R, -dP["Xv"][cv])
            de_L += dcom_L * w_flux * Hd + common * dwfL * Hd + common * w_flux * dHd_L
            de_R += dcom_R * w_flux * Hd + common * dwfR * Hd + common * w_flux * dHd_R
            ds_L += dcom_L * w_flux * Xd + common * dwfL * Xd + common * w_flux * dXd_L
            ds_R += dcom_R * w_flux * Xd + common * dwfR * Xd + common * w_flux * dXd_R
        BLL = np.array([dm_L, ds_L, de_L]); BLR = np.array([dm_R, ds_R, de_R])
        _band_add(ab, u, L, L, BLL * rs); _band_add(ab, u, L, R, BLR * rs)           # +F[f] -> dm[L]
        _band_add(ab, u, R, L, -BLL * rs); _band_add(ab, u, R, R, -BLR * rs)         # -F[f] -> dm[R]

    # --- boundary faces (HU; the -GA*rho_ff term uses fixed boundary props -> constant in x) ---
    Tb, TFb = geom.Tb, geom.TFb
    V_l = Tb * (bleft.p - p[0]) - GA * bleft.pr.rho_ff[0]
    Bl = np.zeros((3, 3))
    if V_l >= 0.0:                                                    # inflow: fixed bleft props
        Bl[:, 0] = -Tb * np.array([bleft.pr.lam_T[0], bleft.pr.salt_mob[0], bleft.pr.adv_h[0]])
    else:                                                            # own cell-0 props
        Bl[0] = -Tb * pr.lam_T[0] * ep + V_l * dP["lam_T"][0]
        Bl[1] = -Tb * pr.salt_mob[0] * ep + V_l * dP["salt_mob"][0]
        Bl[2] = -Tb * pr.adv_h[0] * ep + V_l * dP["adv_h"][0]
    Bl[2] -= TFb * dP["T"][0]                                         # Fe_l has TFb*(bleft.T - T[0])
    _band_add(ab, u, 0, 0, -Bl * rs)                                 # dm[0] = F[0] - Fm_l -> -Fm_l

    V_r = Tb * (p[-1] - bright.p) - GA * bright.pr.rho_ff[0]
    Br = np.zeros((3, 3))
    if V_r >= 0.0:                                                    # outflow: own cell-(N-1) props
        Br[0] = Tb * pr.lam_T[-1] * ep + V_r * dP["lam_T"][-1]
        Br[1] = Tb * pr.salt_mob[-1] * ep + V_r * dP["salt_mob"][-1]
        Br[2] = Tb * pr.adv_h[-1] * ep + V_r * dP["adv_h"][-1]
    else:                                                            # fixed bright props
        Br[:, 0] = Tb * np.array([bright.pr.lam_T[0], bright.pr.salt_mob[0], bright.pr.adv_h[0]])
    Br[2] += TFb * dP["T"][-1]                                        # Fe_r has TFb*(T[-1] - bright.T)
    _band_add(ab, u, N - 1, N - 1, Br * rs)                          # dm[-1] = Fm_r - F[-1] -> +Fm_r
    return ab


# --------------------------------------------------------------------------------------- #
#  Brine (H2O-NaCl) time stepping -- the SINGLE engine for the whole Weis benchmark.
#  Three conservation laws (mass + salt + energy), primaries [p, h, z]; halite is an immobile,
#  table-provided phase. HU/PPU/HU-mwp buoyancy; z=0 reproduces Fig 4/5 (pure water, both
#  orientations), z>0 the Fig 6 salt column.
# --------------------------------------------------------------------------------------- #
@dataclass
class BrineBoundaryState:
    p: float; h: float; z: float; pr: PropsBrine; T: float


def boundary_state_brine(table, p_bc, h_bc, z_bc):
    pr = eval_props_brine(table, np.array([p_bc]), np.array([h_bc]), np.array([z_bc]))
    return BrineBoundaryState(p=p_bc, h=h_bc, z=z_bc, pr=pr, T=float(pr.T[0]))


def residual_brine(x, acc_mass_o, acc_salt_o, acc_en_o, dt, geom, table, bleft, bright,
                   scheme, ug, ud, ut, w_dir, grav_upstream, weighted_perm, lag_upwind, lam_face_old):
    """3N residual [mass_0, salt_0, energy_0, ...] -- the SINGLE brine discretization for the whole
    benchmark. Fig 4/5 (z=0, pure water, either orientation) and Fig 6 (z>0, halite) run through this
    same residual: total-velocity viscous advection + simplicial (mobility-product) buoyancy on the
    liquid/vapor pair, mirroring the retired pure-water ``residual``. The salt row carries the NaCl
    fractions X_l/X_v through the SAME upwind and buoyancy directions as mass/energy; it vanishes at
    z=0 (X_l=X_v=0), so the vertical z=0 run reduces to pure water bit-for-bit. Halite is immobile
    (mm_h=0): it enters rho_mix and the salt accumulation, never any flux. ``ug/ud/ut/w_dir`` are the
    lagged buoyancy directions; ``lam_face_old`` the old-state harmonic lambda*K (HU-mwp lag_upwind).
    Boundary convention: bleft = inlet (i=0), bright = outlet (i=N-1)."""
    N = geom.N
    p = x[0::3]; h = x[1::3]; z = x[2::3]
    pr = eval_props_brine(table, p, h, z)
    acc_mass = geom.Vcell * PHI * pr.rho_mix
    acc_salt = geom.Vcell * PHI * pr.rho_mix * z
    acc_en = geom.Vcell * (PHI * (pr.rho_mix * h - p) + (1 - PHI) * RHO_S * C_S * pr.T)

    dp_face = p[:-1] - p[1:]
    rho_l_f = 0.5 * (pr.rho_l[:-1] + pr.rho_l[1:])
    rho_v_f = 0.5 * (pr.rho_v[:-1] + pr.rho_v[1:])
    F_four = geom.TFf * (pr.T[:-1] - pr.T[1:])

    if scheme == "ppu":
        # per-phase potential upwinding: each phase rides its own Psi_g; the NaCl fraction and the
        # enthalpy it carries follow the same phase-upwind node.
        if lag_upwind:
            iu_l, iu_v = ug, ud
        else:
            iu_l = _upwind_idx(geom.Tf * dp_face - geom.GA * rho_l_f)
            iu_v = _upwind_idx(geom.Tf * dp_face - geom.GA * rho_v_f)
        rho_l_p = pr.rho_l[iu_l] if grav_upstream else rho_l_f
        rho_v_p = pr.rho_v[iu_v] if grav_upstream else rho_v_f
        Psi_l = geom.Tf * dp_face - geom.GA * rho_l_p
        Psi_v = geom.Tf * dp_face - geom.GA * rho_v_p
        F_mass = Psi_l * pr.mm_l[iu_l] + Psi_v * pr.mm_v[iu_v]
        F_salt = Psi_l * (pr.Xl[iu_l] * pr.mm_l[iu_l]) + Psi_v * (pr.Xv[iu_v] * pr.mm_v[iu_v])
        F_en = F_four + Psi_l * (pr.h_l[iu_l] * pr.mm_l[iu_l]) + Psi_v * (pr.h_v[iu_v] * pr.mm_v[iu_v])
    else:
        # HU: total-velocity viscous advection + simplicial mobility-product buoyancy (see the paper).
        rho_ff_f = 0.5 * (pr.rho_ff[:-1] + pr.rho_ff[1:])
        rho_ff_g = pr.rho_ff[ut] if grav_upstream else rho_ff_f
        V_T = geom.Tf * dp_face - geom.GA * rho_ff_g
        up = ut if lag_upwind else np.where(V_T >= 0.0, np.arange(N - 1), np.arange(N - 1) + 1)
        if weighted_perm:                                     # HU-mwp: fold lambda_T into K
            lam_face = (lam_face_old if (lag_upwind and lam_face_old is not None)
                        else _harmonic_face(pr.lam_T))
            F_mass = V_T * lam_face
            hbar_up = pr.adv_h[up] / np.where(pr.lam_T[up] > 0.0, pr.lam_T[up], 1.0)   # <hbar>
            F_en_adv = hbar_up * F_mass
            xbar_up = pr.salt_mob[up] / np.where(pr.lam_T[up] > 0.0, pr.lam_T[up], 1.0)  # <Xbar>
            F_salt_adv = xbar_up * F_mass
        else:
            F_mass = V_T * pr.lam_T[up]
            F_en_adv = V_T * pr.adv_h[up]
            F_salt_adv = V_T * pr.salt_mob[up]
        # HU-BM(mp) pair buoyancy lambda_l lambda_v / Gamma * w_flux. Background void at N=2 (halite
        # immobile). Energy advects (h_l - h_v); salt advects (X_l - X_v) with the same directions.
        w_flux = -geom.GA * (rho_l_f - rho_v_f)
        lam_l_up = _advect(pr.mm_l, w_dir)
        lam_v_dn = _advect(pr.mm_v, -w_dir)
        Gamma = lam_l_up + lam_v_dn
        common = lam_l_up * lam_v_dn / (Gamma + 1.0e-30)
        F_buoy = common * w_flux * (_advect(pr.h_l, w_dir) - _advect(pr.h_v, -w_dir))
        F_salt_buoy = common * w_flux * (_advect(pr.Xl, w_dir) - _advect(pr.Xv, -w_dir))
        F_salt = F_salt_adv + F_salt_buoy
        F_en = F_four + F_en_adv + F_buoy

    # ---- boundary faces (Dirichlet p, T->h_bc; bleft = i0, bright = iN-1) ----
    if scheme == "ppu":
        Psi_lb = geom.Tb * (bleft.p - p[0]) - geom.GA * bleft.pr.rho_l[0]
        Psi_vb = geom.Tb * (bleft.p - p[0]) - geom.GA * bleft.pr.rho_v[0]
        mml = bleft.pr.mm_l[0] if Psi_lb >= 0 else pr.mm_l[0]
        hl = bleft.pr.h_l[0] if Psi_lb >= 0 else pr.h_l[0]
        Xlb = bleft.pr.Xl[0] if Psi_lb >= 0 else pr.Xl[0]
        mmv = bleft.pr.mm_v[0] if Psi_vb >= 0 else pr.mm_v[0]
        hv = bleft.pr.h_v[0] if Psi_vb >= 0 else pr.h_v[0]
        Xvb = bleft.pr.Xv[0] if Psi_vb >= 0 else pr.Xv[0]
        Fm_l = Psi_lb * mml + Psi_vb * mmv
        Fs_l = Psi_lb * Xlb * mml + Psi_vb * Xvb * mmv
        Fe_l = geom.TFb * (bleft.T - pr.T[0]) + Psi_lb * hl * mml + Psi_vb * hv * mmv

        Psi_lt = geom.Tb * (p[-1] - bright.p) - geom.GA * bright.pr.rho_l[0]
        Psi_vt = geom.Tb * (p[-1] - bright.p) - geom.GA * bright.pr.rho_v[0]
        mml = pr.mm_l[-1] if Psi_lt >= 0 else bright.pr.mm_l[0]
        hl = pr.h_l[-1] if Psi_lt >= 0 else bright.pr.h_l[0]
        Xlt = pr.Xl[-1] if Psi_lt >= 0 else bright.pr.Xl[0]
        mmv = pr.mm_v[-1] if Psi_vt >= 0 else bright.pr.mm_v[0]
        hv = pr.h_v[-1] if Psi_vt >= 0 else bright.pr.h_v[0]
        Xvt = pr.Xv[-1] if Psi_vt >= 0 else bright.pr.Xv[0]
        Fm_r = Psi_lt * mml + Psi_vt * mmv
        Fs_r = Psi_lt * Xlt * mml + Psi_vt * Xvt * mmv
        Fe_r = geom.TFb * (pr.T[-1] - bright.T) + Psi_lt * hl * mml + Psi_vt * hv * mmv
    else:
        V_l = geom.Tb * (bleft.p - p[0]) - geom.GA * bleft.pr.rho_ff[0]
        if V_l >= 0.0:
            Fm_l = V_l * bleft.pr.lam_T[0]; Fh_l = V_l * bleft.pr.adv_h[0]; Fs_l = V_l * bleft.pr.salt_mob[0]
        else:
            Fm_l = V_l * pr.lam_T[0];       Fh_l = V_l * pr.adv_h[0];       Fs_l = V_l * pr.salt_mob[0]
        Fe_l = geom.TFb * (bleft.T - pr.T[0]) + Fh_l

        V_r = geom.Tb * (p[-1] - bright.p) - geom.GA * bright.pr.rho_ff[0]
        if V_r >= 0.0:
            Fm_r = V_r * pr.lam_T[-1];      Fh_r = V_r * pr.adv_h[-1];      Fs_r = V_r * pr.salt_mob[-1]
        else:
            Fm_r = V_r * bright.pr.lam_T[0]; Fh_r = V_r * bright.pr.adv_h[0]; Fs_r = V_r * bright.pr.salt_mob[0]
        Fe_r = geom.TFb * (pr.T[-1] - bright.T) + Fh_r

    dm = np.empty(N); ds = np.empty(N); de = np.empty(N)
    dm[0] = F_mass[0] - Fm_l; dm[1:-1] = F_mass[1:] - F_mass[:-1]; dm[-1] = Fm_r - F_mass[-1]
    ds[0] = F_salt[0] - Fs_l; ds[1:-1] = F_salt[1:] - F_salt[:-1]; ds[-1] = Fs_r - F_salt[-1]
    de[0] = F_en[0] - Fe_l;   de[1:-1] = F_en[1:] - F_en[:-1];     de[-1] = Fe_r - F_en[-1]

    r = np.empty(3 * N)
    r[0::3] = ((acc_mass - acc_mass_o) / dt + dm) / geom.ms      # row-scaled to O(1)
    r[1::3] = ((acc_salt - acc_salt_o) / dt + ds) / geom.ms      # salt ~ z * mass -> same scale
    r[2::3] = ((acc_en - acc_en_o) / dt + de) / geom.es
    return r


def newton_step_brine(x0, x_old, dt, geom, table, bleft, bright, scheme, plan,
                      atol=1e-5, maxit=20, verbose=False, grav_upstream=False,
                      weighted_perm=False, lag_upwind=False):
    p_o = x_old[0::3]; h_o = x_old[1::3]; z_o = x_old[2::3]
    pr_o = eval_props_brine(table, p_o, h_o, z_o)
    ug, ud, ut, w_dir = buoyancy_directions(geom, p_o, pr_o, scheme)   # lagged per step
    lam_face_old = _harmonic_face(pr_o.lam_T)                          # HU-mwp lag_upwind old-state
    acc_mass_o = geom.Vcell * PHI * pr_o.rho_mix
    acc_salt_o = geom.Vcell * PHI * pr_o.rho_mix * z_o
    acc_en_o = geom.Vcell * (PHI * (pr_o.rho_mix * h_o - p_o) + (1 - PHI) * RHO_S * C_S * pr_o.T)
    args = (acc_mass_o, acc_salt_o, acc_en_o, dt, geom, table, bleft, bright, scheme, ug, ud, ut,
            w_dir, grav_upstream, weighted_perm, lag_upwind, lam_face_old)
    pclip = (table.b_min * (1 + 1e-9), table.b_max * (1 - 1e-9))
    hclip = (table.a_min * (1 + 1e-9), table.a_max * (1 - 1e-9))
    zclip = (table.c_min, table.c_max)                          # z in [0, 1]
    sqrtN = np.sqrt(geom.N)

    def _metric(rr):
        return max(np.linalg.norm(rr[0::3]), np.linalg.norm(rr[1::3]),
                   np.linalg.norm(rr[2::3])) / sqrtN

    x = x0.copy()
    r = residual_brine(x, *args)
    nrm = np.linalg.norm(r)
    for it in range(maxit):
        m = _metric(r)
        if verbose:
            print(f"    newton {it}: |r|_eq={m:.3e}")
        if m <= atol:
            return x, it, m, True
        ab = jacobian_analytic(x, dt, geom, table, bleft, bright, scheme, ug, ud, ut, w_dir,
                               grav_upstream, weighted_perm, lag_upwind, lam_face_old, plan)  # hand-coded, no FD
        try:
            dx = sla.solve_banded((plan["l"], plan["u"]), ab, -r)
        except Exception:
            dx = np.zeros_like(r)
        step = 1.0
        for _ in range(10):
            xn = x + step * dx
            xn[0::3] = np.clip(xn[0::3], *pclip)
            xn[1::3] = np.clip(xn[1::3], *hclip)
            xn[2::3] = np.clip(xn[2::3], *zclip)
            r_new = residual_brine(xn, *args); nrm_new = np.linalg.norm(r_new)
            if nrm_new < nrm:
                break
            step *= 0.5
        x = xn; r = r_new; nrm = nrm_new
    return x, maxit, _metric(r), False


# Weis (2014) Fig 6 C/D data (SI). Horizontal column; left = hot pure-water vapor inlet, right =
# cool outlet; the domain starts as salt-saturated liquid + immobile halite (z_init tuned to the
# table so the flash returns S_h ~ 0.1).
FIG6 = dict(p_left=4.0e6, T_left=300.0 + 273.15, z_left=0.0,
            p_right=1.0e6, T_right=150.0 + 273.15,
            T_init=150.0 + 273.15, z_init=0.3, tf_yr=200.0)

# Weis (2014) Fig 4/5 (pure-water) boundary/initial data at z=0: hot steam inlet -> cool outlet. The
# brine engine reduces to the pure-water column at z=0, so Fig 4/5 run through it via ``**FIG5``.
FIG5 = dict(p_left=P_BOT, T_left=T_BOT, z_left=0.0,
            p_right=P_TOP, T_right=T_TOP, T_init=T_INIT, z_init=0.0)


def run_brine(N=200, scheme="hu", case="horizontal", n_steps=None, dt=None, adaptive=True,
              verbose=True, grav_upstream=False, weighted_perm=False, lag_upwind=False,
              level=TABLE_LEVEL, pure_water=False, amr_table=None, atol=1e-5, **fig):
    """The single brine engine: mass + salt + energy, primaries [p, h, z], HU/PPU/HU-mwp buoyancy.
    Reproduces Fig 4/5 (pure water) at z=0 and Fig 6 (H2O-NaCl + immobile halite) at z>0 -- ONE
    discretization. ``case`` ('horizontal'|'vertical') sets gravity + default final time via CASES;
    ``**fig`` overrides the BC/IC (defaults = FIG6, the salt column). Pass ``**FIG5`` for pure water.
    ``pure_water=True`` loads the high-resolution z=0 pure-water tables (finer enthalpy grid) instead of
    the level-indexed brine tables -- for the Fig-6 pure-water column, where the coarse brine h-grid
    produces spurious two-phase saturation wiggles; the run itself is still at z=0."""
    if case not in CASES:
        raise ValueError(f"case must be one of {list(CASES)}")
    if weighted_perm and scheme == "ppu":
        raise ValueError("weighted_perm (lambda folded into K) is incompatible with scheme='ppu'.")
    cfg = {**FIG6, **fig}
    g = CASES[case]["g"]
    tf_yr = fig["tf_yr"] if "tf_yr" in fig else CASES[case]["tf_yr"]
    # xph property source: an adapted hex-AMR .vtu (amr_table), the fine pure-water z=0 tables
    # (pure_water, 2-D slice), or the level-indexed rectilinear brine tables. The xpt table (T,p->h for
    # the IC/BC enthalpy) stays rectilinear in every case.
    if amr_table is not None:
        table = XphSampler(amr_table, _xph_fmap(amr=True))
        xpt = XptSampler(table_paths(level)[1])
    elif pure_water:
        table = XphSampler(PUREWATER_XPH, _xph_fmap(amr=False))
        xpt = XptSampler(PUREWATER_XPT)
    else:
        table = XphSampler(table_paths(level)[0], _xph_fmap(amr=False))
        xpt = XptSampler(table_paths(level)[1])
    geom = make_geom(N, g=g)

    def enth(TK, p, z):
        return xpt.enth(TK, p, z)

    h_left = float(enth(cfg["T_left"], cfg["p_left"], cfg["z_left"])[0])
    h_right = float(enth(cfg["T_right"], cfg["p_right"], cfg["z_init"])[0])
    bleft = boundary_state_brine(table, cfg["p_left"], h_left, cfg["z_left"])
    bright = boundary_state_brine(table, cfg["p_right"], h_right, cfg["z_init"])

    y = geom.yc
    p0 = (y * cfg["p_right"] + (L_COLUMN - y) * cfg["p_left"]) / L_COLUMN
    z0 = np.full(N, cfg["z_init"])
    h0 = enth(np.full(N, cfg["T_init"]), p0, z0)
    x = np.empty(3 * N); x[0::3] = p0; x[1::3] = h0; x[2::3] = z0

    plan = build_jac_plan(N, nvar=3)
    dt0 = dt if dt is not None else DT0
    tf = tf_yr * YEAR if n_steps is None else n_steps * dt0
    t = 0.0; dt = dt0; step = 0; n_cuts = 0; it_wasted = 0; nit_hist = []
    if verbose:
        print(f"  brine {scheme}{'-mwp' if weighted_perm else ''}: N={N}, "
              f"level {'pw' if pure_water else level}, {case} "
              f"(g={g:.4g});  left {cfg['T_left']-273.15:.0f}C/{cfg['p_left']/1e6:.0f}MPa "
              f"z={cfg['z_left']}  ->  right {cfg['T_right']-273.15:.0f}C/{cfg['p_right']/1e6:.0f}MPa;"
              f"  IC z={cfg['z_init']}")
    while t < tf - 1e-6:
        dt = min(dt, tf - t)
        x_old = x.copy()
        xn, nit, nrm, ok = newton_step_brine(x, x_old, dt, geom, table, bleft, bright, scheme, plan,
                                             atol=atol, grav_upstream=grav_upstream,
                                             weighted_perm=weighted_perm, lag_upwind=lag_upwind)
        if not ok and dt > dt0 / 64:
            n_cuts += 1; it_wasted += nit; dt *= 0.5; continue
        x = xn; t += dt; step += 1; nit_hist.append(nit)
        if adaptive and ok and nit < 5 and dt < dt0:
            dt = min(dt * 2.0, dt0)
        elif not adaptive:
            dt = dt0
        if verbose and (step % 50 == 0 or not ok):
            print(f"  t={t/YEAR:7.1f} yr  dt={dt/YEAR:.4f}  nit={nit}  |r|={nrm:.1e}"
                  f"  {'' if ok else 'NOT CONVERGED'}")

    pr = eval_props_brine(table, x[0::3], x[1::3], x[2::3])
    hist = np.asarray(nit_hist, dtype=int)
    return {"y": y, "p": x[0::3], "h": x[1::3], "z": x[2::3], "T": pr.T,
            "s_liq": pr.s_l, "s_gas": pr.s_v, "s_halite": pr.s_h, "Xl": pr.Xl,
            "rho_mix": pr.rho_mix, "N": N, "case": case, "level": level, "scheme": scheme,
            "n_steps": step, "total_it": int(hist.sum()),
            "avg_it": (hist.sum() / step) if step else 0.0,
            "max_it": int(hist.max()) if hist.size else 0, "n_time_step_cuts": n_cuts,
            "it_wasted": it_wasted, "nit_hist": hist,
            "grav_upstream": grav_upstream, "weighted_perm": weighted_perm, "lag_upwind": lag_upwind,
            "pure_water": pure_water}


# --------------------------------------------------------------------------------------- #
#  Comparison plot vs digitized paper data (CSV)
# --------------------------------------------------------------------------------------- #
def _load_ref_csv(name):
    path = os.path.join(REF_DIR, name)
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    return d[:, 0], d[:, 1]      # distance[km], value


# Digitized Weis (2014) fig-5 reference: field -> CSV basename template (``{tag}`` = orientation).
_REF_CSV = {
    "T": "fig_5_{tag}_temperature_raw.csv",
    "p": "fig_5_{tag}_pressured_raw.csv",
    "s_liq": "fig_5_{tag}_saturation_liq_raw.csv",
}


def load_reference(case, field):
    """Digitized Weis (2014) fig-5 reference curve.

    ``field`` in {'T', 'p', 's_liq'}, ``case`` in {'vertical', 'horizontal'}. Returns
    ``(distance_km, value)`` in plotted units: T [degC], p [MPa], s_liq [-].
    """
    tag = "vertical" if case == "vertical" else "horizontal"
    return _load_ref_csv(_REF_CSV[field].format(tag=tag))


# --------------------------------------------------------------------------------------- #
#  Self-test (cheap invariants)
# --------------------------------------------------------------------------------------- #
def selftest():
    print("=== selftest ===")
    table = XphSampler(table_paths()[0], _xph_fmap(amr=False))
    geom = make_geom(20)
    p = np.linspace(20e6, 1e6, 20)
    h = np.full(20, 6.0e5)                         # cold liquid -> s_v = 0
    z = np.zeros(20)                               # pure water (brine engine at z=0)
    pr = eval_props_brine(table, p, h, z)
    assert np.all(pr.s_v < 1e-6), "expected single-phase liquid"
    rho_l_f = 0.5 * (pr.rho_l[:-1] + pr.rho_l[1:]); rho_v_f = 0.5 * (pr.rho_v[:-1] + pr.rho_v[1:])
    i_liq, i_gas, _, w_dir = buoyancy_directions(geom, p, pr, "hu")
    lam_l_up = _advect(pr.mm_l, w_dir); lam_v_dn = _advect(pr.mm_v, -w_dir)
    b = (lam_l_up * lam_v_dn / (lam_l_up + lam_v_dn + 1e-30)
         * (-geom.GA * (rho_l_f - rho_v_f)) * (_advect(pr.h_l, w_dir) - _advect(pr.h_v, -w_dir)))
    assert np.max(np.abs(b)) < 1e-20, f"single-phase buoyancy != 0: {np.max(np.abs(b)):.2e}"
    print("  single-phase buoyancy == 0  OK")
    p_hyd = np.empty(20); p_hyd[0] = 20e6
    for i in range(1, 20):
        p_hyd[i] = p_hyd[i - 1] - 0.5 * (pr.rho_ff[i - 1] + pr.rho_ff[i]) * G * geom.dy
    pr2 = eval_props_brine(table, p_hyd, h, z)
    rff = 0.5 * (pr2.rho_ff[:-1] + pr2.rho_ff[1:])
    VT = geom.Tf * (p_hyd[:-1] - p_hyd[1:]) - geom.GA * rff
    print(f"  hydrostatic max|V_T| = {np.max(np.abs(VT)):.2e} (should be ~0)")
    print("  selftest passed\n")


def prebuild_table_caches(level=TABLE_LEVEL, pure_water=False):
    """Construct the VTKSampler for the xph/xpt tables once before a parallel sweep. The VTKSampler
    tensor backend persists an ``.obltensor.npz`` cache, so this writes it once and every fresh worker
    then loads from it (skipping the pyvista read) instead of rebuilding the tensor."""
    if pure_water:
        XphSampler(PUREWATER_XPH, _xph_fmap(amr=False)); XptSampler(PUREWATER_XPT)
        return
    XphSampler(table_paths(level)[0], _xph_fmap(amr=False)); XptSampler(table_paths(level)[1])
