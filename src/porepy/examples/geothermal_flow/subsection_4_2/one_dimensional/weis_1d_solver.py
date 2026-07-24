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


VTK_XPH, VTK_XPT = table_paths()      # default-level table paths (used by selftest)

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
#  Fast rectilinear-grid bilinear table (xph / xpt are RectilinearGrid; axes may be non-uniform)
# --------------------------------------------------------------------------------------- #
def _is_uniform(x):
    """True if the 1-D axis ``x`` is (float-)uniformly spaced. A 2-node stub counts as uniform. The
    1e-6*range tolerance cleanly separates a float32-rounded uniform axis (interval jitter ~1e-9) from
    an intentionally graded one (interval jitter ~ the refinement ratio)."""
    if len(x) < 3:
        return True
    d = np.diff(x)
    return bool(np.max(np.abs(d - d[0])) <= 1e-6 * abs(x[-1] - x[0]))


def _bracket(v, coords, v0, dv, n, uniform):
    """Lower index, upper index, and in-cell fraction for a monotone-increasing axis at query points
    ``v`` (same units as the axis). ``uniform``: the original O(1) (v - v0)/dv bracketing -- bit-
    identical to the pre-non-uniform sampler and faster. Else: searchsorted on the stored ``coords``
    (arbitrary spacing). Both clamp the fraction so out-of-range points constant-extrapolate."""
    if uniform:
        f = ((v - v0) / dv).clip(0.0, n - 1 - 1e-9)
        j = f.astype(np.intp)
        return j, j + 1, f - j
    j = np.clip(np.searchsorted(coords, v, side="right") - 1, 0, n - 2)
    j1 = j + 1
    t = ((v - coords[j]) / (coords[j1] - coords[j])).clip(0.0, 1.0)
    return j, j1, t


class Table:
    """O(1) vectorised bilinear sampler of a Driesner VTK table on the z_NaCl=0 slice.

    Solver inputs are in SI; ``a_in``/``b_in`` convert them to the table axis units
    (second axis = h[MJ/kg] or T[degC]; third axis = p[MPa]).  Field values are returned
    in SI via the per-field ``fields`` scale (e.g. enthalpy kJ/kg -> J/kg via 1e3).
    """

    def __init__(self, file_name, fields, a_in=1.0, b_in=1.0, c_in=1.0, slice_z=True):
        # slice_z=True: z_NaCl=0 slice, 2-D bilinear (pure-water fig-5, unchanged).
        # slice_z=False: full 3-D array, trilinear in (a=h/T, b=p, c=z_NaCl) (brine / Fig 6).
        self.slice_z = bool(slice_z)
        # mtime in the key -> regenerating a .vtr in place invalidates its .npz cache
        key = ("|".join(f"{k}:{v}" for k, v in sorted(fields.items()))
               + f"|{a_in}|{b_in}|{c_in}|slice{int(self.slice_z)}|abcnu2|{os.path.getmtime(file_name):.0f}")
        cache = file_name + ".sicache.npz"
        if not (os.path.exists(cache) and self._load(cache, key)):
            self._build(file_name, fields, a_in, b_in, c_in, key, cache)
        # all fields stacked -> gather every field's corners in a few fancy-index ops, not *nf
        self.names = list(self.V.keys())
        self.V_stack = np.ascontiguousarray(np.stack([self.V[n] for n in self.names]))

    def _load(self, cache, key):
        try:
            z = np.load(cache, allow_pickle=False)
            if str(z["key"]) != key:
                return False
            self.slice_z = bool(int(z["slice_z"]))
            for s in ("ny", "nz", "nx"):
                setattr(self, s, int(z[s]))
            for s in ("a0", "da", "b0", "db", "c0", "dc", "a_in", "b_in", "c_in",
                      "a_min", "a_max", "b_min", "b_max", "c_min", "c_max"):
                setattr(self, s, float(z[s]))
            self.a_coords = np.asarray(z["a_coords"], float)
            self.b_coords = np.asarray(z["b_coords"], float)
            self.c_coords = np.asarray(z["c_coords"], float)
            self.a_uniform = bool(int(z["a_uniform"]))
            self.b_uniform = bool(int(z["b_uniform"]))
            self.c_uniform = bool(int(z["c_uniform"]))
            self.V = {str(n): z["V_" + str(n)] for n in z["names"]}
            return True
        except Exception:
            return False

    def _build(self, file_name, fields, a_in, b_in, c_in, key, cache):
        import pyvista as pv     # only hit on a cache miss (first run)

        g = pv.read(file_name)
        nx, ny, nz = g.dimensions                          # (z_NaCl, second=h/T, p)
        a = np.asarray(g.y); b = np.asarray(g.z); c = np.asarray(g.x)
        self.ny, self.nz, self.nx = int(ny), int(nz), int(nx)
        self.a0, self.da = float(a[0]), float(a[1] - a[0])
        self.b0, self.db = float(b[0]), float(b[1] - b[0])
        self.c0 = float(c[0]); self.dc = float(c[1] - c[0]) if nx > 1 else 1.0
        self.a_coords = np.asarray(a, float)               # full h/T-axis nodes (may be non-uniform)
        self.b_coords = np.asarray(b, float)               # full p-axis nodes   (may be non-uniform)
        self.c_coords = np.asarray(c, float)               # full z-axis nodes: NON-uniform (salinity)
        self.a_uniform = _is_uniform(self.a_coords)        # h/p are uniform today -> exact (v-v0)/dv path
        self.b_uniform = _is_uniform(self.b_coords)        # a user-graded table trips searchsorted instead
        self.c_uniform = _is_uniform(self.c_coords)        # salinity is graded -> searchsorted (as before)
        self.a_in, self.b_in, self.c_in = float(a_in), float(b_in), float(c_in)
        self.a_min, self.a_max = float(a[0] / a_in), float(a[-1] / a_in)
        self.b_min, self.b_max = float(b[0] / b_in), float(b[-1] / b_in)
        self.c_min, self.c_max = float(c[0] / c_in), float(c[-1] / c_in)
        full = {name: np.asarray(g.point_data[name]).reshape(nz, ny, nx) * scale   # [p, h, z]
                for name, scale in fields.items()}
        self.V = {n: (A[:, :, 0] if self.slice_z else A) for n, A in full.items()}
        out = {"key": np.array(key), "slice_z": int(self.slice_z),
               "ny": self.ny, "nz": self.nz, "nx": self.nx,
               "a0": self.a0, "da": self.da, "b0": self.b0, "db": self.db,
               "c0": self.c0, "dc": self.dc, "a_in": self.a_in, "b_in": self.b_in,
               "c_in": self.c_in, "a_min": self.a_min, "a_max": self.a_max,
               "b_min": self.b_min, "b_max": self.b_max, "c_min": self.c_min,
               "c_max": self.c_max, "a_coords": self.a_coords, "b_coords": self.b_coords,
               "c_coords": self.c_coords, "a_uniform": int(self.a_uniform),
               "b_uniform": int(self.b_uniform), "c_uniform": int(self.c_uniform),
               "names": np.array(list(self.V.keys()))}
        out.update({"V_" + n: A for n, A in self.V.items()})
        try:
            np.savez(cache, **out)
        except Exception:
            pass

    def sample_many(self, a, b, c=0.0):
        """Interpolate ALL stored fields at (a=h/T, b=p, [c=z_NaCl]) with one stacked gather.
        2-D bilinear on the z=0 slice when ``slice_z``; else 3-D trilinear (``c`` = salinity). Each
        axis is bracketed by :func:`_bracket` -- the exact (v-v0)/dv path when it is uniformly spaced,
        searchsorted when it is graded -- so non-uniform h / p / z tables interpolate correctly while
        uniform tables give the same result as before."""
        a = np.atleast_1d(np.asarray(a, float)) * self.a_in
        b = np.atleast_1d(np.asarray(b, float)) * self.b_in
        ja, ja1, ta = _bracket(a, self.a_coords, self.a0, self.da, self.ny, self.a_uniform)  # h/T axis
        jb, jb1, tb = _bracket(b, self.b_coords, self.b0, self.db, self.nz, self.b_uniform)  # p axis
        Vs = self.V_stack
        if self.slice_z:
            vals = ((1 - ta) * (1 - tb) * Vs[:, jb, ja] + ta * (1 - tb) * Vs[:, jb, ja1]
                    + (1 - ta) * tb * Vs[:, jb1, ja] + ta * tb * Vs[:, jb1, ja1])   # (nf, N)
            return {n: vals[i] for i, n in enumerate(self.names)}
        c = np.broadcast_to(np.atleast_1d(np.asarray(c, float)) * self.c_in, a.shape)
        jc, jc1, tc = _bracket(c, self.c_coords, self.c0, self.dc, self.nx, self.c_uniform)  # salinity
        ma, mb, mc = 1 - ta, 1 - tb, 1 - tc                      # Vs axes: (field, p, h, z)
        vals = (ma * mb * mc * Vs[:, jb, ja, jc] + ta * mb * mc * Vs[:, jb, ja1, jc]
                + ma * tb * mc * Vs[:, jb1, ja, jc] + ta * tb * mc * Vs[:, jb1, ja1, jc]
                + ma * mb * tc * Vs[:, jb, ja, jc1] + ta * mb * tc * Vs[:, jb, ja1, jc1]
                + ma * tb * tc * Vs[:, jb1, ja, jc1] + ta * tb * tc * Vs[:, jb1, ja1, jc1])
        return {n: vals[i] for i, n in enumerate(self.names)}

    def __call__(self, name, a, b, c=0.0):
        return self.sample_many(a, b, c)[name]


# xph: solver h[J/kg] -> axis MJ/kg (1e-6); p[Pa] -> MPa (1e-6).
#   field scales to SI: Rho 1 (kg/m^3), H kJ/kg->J/kg (1e3), S_v 1, T 1 (K).
#   mu: the table already stores Pa.s (probe: mu~2.5e-5 at 400C); PorePy's extra 1e-6 is its
#   Pa.s->MPa.s Mega-scaling, which must NOT be applied in this SI solver -> scale 1.0.
# --------------------------------------------------------------------------------------- #
#  Brine closure (H2O-NaCl): three-phase liquid/vapor/immobile-halite, from the 3-D table
# --------------------------------------------------------------------------------------- #
# xph brine fields: densities (kg/m^3, scale 1), enthalpies (kJ/kg -> J/kg, 1e3), vapor+halite
# saturations, NaCl mass fractions Xl/Xv (dimensionless), viscosities (Pa.s), temperature (K).
_XPH_FIELDS_BRINE = {"Rho_l": 1.0, "Rho_v": 1.0, "Rho_h": 1.0,
                     "H_l": 1e3, "H_v": 1e3, "H_h": 1e3,
                     "S_v": 1.0, "S_h": 1.0, "Xl": 1.0, "Xv": 1.0,
                     "mu_l": 1.0, "mu_v": 1.0, "Temperature": 1.0}


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


def eval_props_brine(table, p, h, z):
    """Three-phase closure from the 3-D xph table at overall NaCl composition ``z``.

    Halite is a table-provided saturation (as in the porepy DriesnerModelConfiguration): it enters
    rho_mix, and blocks the pore space through the rel-perm k_rl + k_rv = 1 - s_h; it is immobile
    (k_rh = 0), so it advects no mass -- its NaCl is carried only in the accumulation via rho_mix z.
    """
    s = table.sample_many(h, p, z)
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
        ab = jacobian_fd(x, r, args, plan, resfn=residual_brine)
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
              level=TABLE_LEVEL, pure_water=False, atol=1e-5, **fig):
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
    # pure_water: the fine z=0 tables, sampled as the 2-D z=0 slice (slice_z=True); else the 3-D brine
    # tables at the requested level. Field schema/units are identical, so _XPH_FIELDS_BRINE is reused.
    if pure_water:
        xph_path, xpt_path, sz = PUREWATER_XPH, PUREWATER_XPT, True
    else:
        xph_path, xpt_path = table_paths(level); sz = False
    table = Table(xph_path, _XPH_FIELDS_BRINE, a_in=1e-6, b_in=1e-6, c_in=1.0, slice_z=sz)
    xpt = Table(xpt_path, {"H": 1e3}, a_in=1.0, b_in=1e-6, c_in=1.0, slice_z=sz)
    geom = make_geom(N, g=g)

    def enth(TK, p, z):
        return xpt("H", np.atleast_1d(TK) - 273.15, np.atleast_1d(p), np.atleast_1d(z))

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
    table = Table(VTK_XPH, _XPH_FIELDS_BRINE, a_in=1e-6, b_in=1e-6, c_in=1.0, slice_z=False)
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
    """Build the ``.npz`` caches for the xph/xpt tables serially, so that a subsequent parallel sweep
    of :func:`run_brine` hits the fast cache path in every worker (else workers race on the npz write).
    ``pure_water=True`` warms the fine pure-water z=0 tables instead of the level-indexed brine ones."""
    if pure_water:
        Table(PUREWATER_XPH, _XPH_FIELDS_BRINE, a_in=1e-6, b_in=1e-6, c_in=1.0, slice_z=True)
        Table(PUREWATER_XPT, {"H": 1e3}, a_in=1.0, b_in=1e-6, c_in=1.0, slice_z=True)
        return
    xph_path, xpt_path = table_paths(level)
    Table(xph_path, _XPH_FIELDS_BRINE, a_in=1e-6, b_in=1e-6, c_in=1.0, slice_z=False)
    Table(xpt_path, {"H": 1e3}, a_in=1.0, b_in=1e-6, c_in=1.0, slice_z=False)
