"""Independent 2-D N-phase gravity-segregation solver (Bosma et al. 2022, Ex. 6.3 at N=3).

Companion to ``subsection_4_1/weis_1d_solver.py``: a small, self-contained finite-volume
reference carrying the SAME upwinding options exercised in the paper, so the PorePy result
(``porepy_2d_solver.py``, in this folder) can be overlaid on an independent implementation.

The whole HU-BM family -- HU-BM(ff) = ``hu`` / HU-BM(mw) = ``hu-mw`` / HU-BM(mp) = ``hu-mp``, plus
``ppu`` -- extends to ANY number of phases with no structural change -- ``run(nphase=N)``
(default 3): the simplicial buoyancy sums over all ``C(N,2)`` phase pairs, each pair's BACKGROUND
MOBILITY aggregating the remaining ``N-2`` phases. ``nphase=3`` reproduces Bosma Fig. 5 EXACTLY;
``nphase=4`` splits oil into a mid-heavy + mid-light phase (densities evenly spaced 1500..500).
This demonstrates the simplicial structure genuinely extending the HU-BM family to truly
multiphase flow. ("HU-BM" = Hybrid Upwinding with Background Mobility.)

Case (Bosma et al. 2022, "Smooth implicit hybrid upwinding ...", CMAME 388:114288, Sec. 6.3 /
Fig. 5): three immiscible, incompressible phases w/o/g (heavy/intermediate/light) segregating
under gravity in a closed vertical 100 x 100 m box (100 x 100 cells of 1 m), crossed by SEVEN
horizontal impermeable barrier layers with openings.  The exact barrier layout is the digitized
``_BARRIER_LAYERS_FIG`` copied verbatim from
``model_configuration/geometry_description/geometry_market.py``.

  phases            w (heavy)   o (intermediate)   g (light)
  density [kg/m3]      1500          1000             500
  viscosity [Pa s]     1e-3          1e-3            1e-3
  k_r(s)               s^2           s^2             s^2      (quadratic)
  rock:  K = 1000 mD (1 Darcy),  phi = 0.3;  barrier cells: K * 1e-4 (near-impermeable)
  IC:    w in the top tenth, g in the bottom tenth, o in between; all no-flow boundaries.
  end:   571 days;  Fig. 5 snapshots at 0, 78, 571 days.

Formulation (incompressible, immiscible; fractional-flow form)
-------------------------------------------------------------
Volumetric phase mobility ``lambda_a = k_r,a(s_a) / mu``, total ``lambda_T = sum_a lambda_a``,
fractional flow ``f_a = lambda_a / lambda_T``, buoyant density ``rho_ff = sum_a f_a rho_a``.
Per internal face L->R (``dz = z_R - z_L``, ``T_f`` = harmonic-K transmissibility,
``GC = T_f g dz``):

    geometric phase-potential flux   Phi_a = T_f (p_L - p_R) - GC rho_a
    total potential flux             V_T   = T_f (p_L - p_R) - GC rho_ff^face

Unknowns per cell: pressure ``p`` and the two independent saturations ``s_w``, ``s_g``
(``s_o = 1 - s_w - s_g``).  Equations per cell:

    pressure (incompressibility):  sum_faces q_T = 0   (closed; datum via Lagrange mult., sum p = 0)
    phase w mass:  phi |c| (s_w - s_w^old)/dt + sum_faces q_w = 0
    phase g mass:  phi |c| (s_g - s_g^old)/dt + sum_faces q_g = 0

Schemes (``scheme=``) -- the HU-BM (Hybrid Upwinding with Background Mobility) family
------------------------------------------------------------------------------------
The three HU variants share the simplicial buoyancy: a sum over the ``C(N,2)`` phase pairs, each
pair upwinded along its own inter-phase gravity flux, with a per-pair BACKGROUND MOBILITY that
aggregates the remaining ``N-2`` phases (split by chi = 1/2; void at N=2). They differ only in the
buoyant-pair form and the total-mobility placement:

``"hu"``   = HU-BM(ff) -- simplicial FRACTIONAL-FLOW buoyancy ``f_a f_b lambda_T``; viscous part
              ``q_T = lambda_T[up] V_T`` (total mobility upwinded by sign(V_T)).
``"hu-mp"``= HU-BM(mp) -- MOBILITY-PRODUCT buoyancy ``lambda_a lambda_b / lambda_T`` (classical
              Lee/Hamon ``U^HU``) instead of the fractional-flow ``f_a f_b lambda_T``; SAME upwind
              viscous split as (ff). Dropping the fractional-flow normalization sharpens the fronts
              to PPU level (Bosma ``U = Lambda_L Lambda_R U^HU``); eps on the ``lambda_T`` denominator
              (=0 at fully-segregated faces). At N=2 the background is void and HU-BM(mp) reduces
              EXACTLY to Lee (2015).
``"hu-mw"``= HU-BM(mw) -- the EXACT (mp) mobility-product buoyant term, but the viscous total
              mobility is folded into the face transmissibility as a HARMONIC (mobility-weighted)
              face average ``q_T = harmonic(lambda_T)^face V_T`` ("K*lambda" placement, Remark 3.2)
              instead of upwinded. Best corner of two orthogonal levers: mp buoyancy = PPU-sharp
              fronts; harmonic viscous = differentiable (no m_e=0 switch) => fewest Newton iterations.
``"ppu"``  -- phase-potential upwinding (NOT an HU-BM member): each phase rides its OWN potential,
              ``q_a = lambda_a[up_a] Phi_a`` with ``up_a = sign(Phi_a)``; buoyancy intrinsic.

Newton upwind DIRECTIONS are lagged, held fixed WITHIN each linear solve (so the FD Jacobian
never differentiates the upwind switch); ``dir_lag`` sets the refresh cadence -- ``"iteration"``
(default) re-lags them from the current iterate every Newton step (fully-implicit upwinding via a
fixed point on the directions), ``"step"`` freezes them once per time step from the previous
converged state (semi-implicit, Bosma/Weis "old velocity field"). Mobility MAGNITUDES are always at
the current iterate. The Jacobian is a sparse finite difference (5-point stencil, coloured by
``scipy``).

Output: each requested snapshot is written as a VTK RectilinearGrid ``.vtr`` (cell data:
``s_w``, ``s_o``, ``s_g``, ``p``, ``barrier``) for visualization.

Run: ``python hamon_2d_solver.py`` (writes ``vtr/hamon_<scheme>_<day>d.vtr``).
"""
from __future__ import annotations

import argparse
import base64
import itertools
import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sps
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import spsolve

# --------------------------------------------------------------------------------------- #
#  Physical constants (SI)
# --------------------------------------------------------------------------------------- #
DAY = 86400.0
G = 9.80665                          # gravity [m/s^2]
MU = 1.0e-3                          # phase viscosity [Pa s] (all phases)
MILLI_DARCY = 9.869233e-16           # 1 mD [m^2]
K_ROCK = 1000.0 * MILLI_DARCY        # homogeneous rock permeability = 1000 mD (1 Darcy)
BARRIER_K_FACTOR = 1.0e-4            # barrier cells: K * this (effectively impermeable)
PHI = 0.3                            # porosity [-]
CHI = 0.5                            # legacy constant background split (--background chi-const)
BACKGROUND = "midpoint"              # pair weights: midpoint sign rule | rho-ratio | chi-const

# --------------------------------------------------------------------------------------- #
#  Phase system (N phases; reconfigured by set_phase_system / run(nphase=...))
# --------------------------------------------------------------------------------------- #
# Densities are evenly spaced heaviest -> lightest, so N=3 reproduces Bosma [1500, 1000, 500]
# EXACTLY, and N=4 splits the intermediate (oil) into a MID-HEAVY + MID-LIGHT phase
# [1500, 1167, 833, 500].  The whole HU family (hu / hu-mw / hu-mp / ppu) extends to any N with
# no structural change -- the simplicial buoyancy just sums over all C(N,2) pairs, each pair's
# total mobility weighting ALL N phases with pinned side masks (default: the midpoint sign
# rule of _side_masks; rho-ratio barycentric weights and constant CHI kept for comparison).
NPHASE = 3
RHO = np.linspace(1500.0, 500.0, NPHASE)   # [kg/m^3], phase 0 heaviest .. phase N-1 lightest
RHO_BANDS = RHO.copy()               # density of initial BAND k (label-permutation invariant)
PHASE_PERM = np.arange(NPHASE)       # phase k occupies band PHASE_PERM[k] (identity default)
MIDDLE_LINEAR = np.zeros(NPHASE, dtype=bool)   # rows with linear k_r (--linear-kr-middle)
_PAIRS = tuple(itertools.combinations(range(NPHASE), 2))   # all C(N,2) counter-current pairs
W, O, GG = 0, 1, 2                   # 3-phase names (heavy, intermediate, light) for reference


def _pinned_theta():
    """Per-pair rho-ratio weights theta_l = |rho_l-rho_b| / (|rho_l-rho_a| + |rho_l-rho_b|).

    Pinned at the pair: theta_a = 1, theta_b = 0, so the pair total mobility is ONE formula
    over all N phases (the in-pair HU upwinding falls out as the endpoint cases) and
    equal-density phases aggregate exactly. The denominator vanishes only at the triple
    coincidence rho_l = rho_a = rho_b, where the pair term is zero anyway (0.5 placeholder)."""
    th = {}
    for a, b in _PAIRS:
        da, db = np.abs(RHO - RHO[a]), np.abs(RHO - RHO[b])
        tot = da + db
        th[(a, b)] = np.where(tot > 0.0, db / np.where(tot > 0.0, tot, 1.0), 0.5)
    return th


def _side_masks():
    """Midpoint rule m_l = (1 + sgn[(rho_a - rho_b)(2 rho_l - rho_a - rho_b)]) / 2 in {0, 1/2, 1}.

    The arithmetic mean of the pair densities is the exact separator between a-side and
    b-side (the Voronoi split of the density line); a phase ON the separator splits
    half-half. Pinned (m_a = 1, m_b = 0), unordered, division-free."""
    m = {}
    for a, b in _PAIRS:
        m[(a, b)] = 0.5 * (1.0 + np.sign((RHO[a] - RHO[b]) * (2.0 * RHO - RHO[a] - RHO[b])))
    return m


_THETA = _pinned_theta()             # rebuilt by set_phase_system whenever RHO changes
_MASKS = _side_masks()               # idem


def set_phase_system(nphase: int, equal_middle: bool = False,
                     permute_middle: bool = False, linear_kr_middle: bool = False):
    """Configure the module for ``nphase`` phases: evenly-spaced densities and every C(n,2) pair.
    N=3 reproduces Bosma Fig. 5; N=4 splits oil into a mid-heavy and a mid-light phase.

    ``equal_middle`` (N=4 only) sets BOTH middle densities to the N=3 oil value 1000:
    the four-phase run then degenerates to the three-phase solution (the middle pair's
    mutual buoyancy vanishes and their pairs with the outer phases coincide), exercising
    the N=4 structure on a problem whose exact answer is the N=3 field -- evidence that
    the scheme needs no independent phase ORDERING.

    ``permute_middle`` (N=4 only) SWAPS THE LABELS of the two middle phases (phase 1
    takes middle band 2 and its density, and vice versa) -- a pure relabeling of the
    same physical problem, so the solution must be identical up to the label swap.

    ``linear_kr_middle`` uses LINEAR k_r = s for the INTERIOR (middle) phases only
    (outer water/gas stay quadratic).  Middle mobilities are then additive, so the
    equal-density N=4 run degenerates to the N=3 solution EXACTLY at the PDE level
    (quadratic k_r breaks additivity wherever the two middles coexist)."""
    global NPHASE, RHO, RHO_BANDS, PHASE_PERM, MIDDLE_LINEAR, _PAIRS, _THETA, _MASKS
    NPHASE = int(nphase)
    if equal_middle:
        if NPHASE != 4:
            raise ValueError("equal_middle is defined for nphase=4 only")
        RHO_BANDS = np.array([1500.0, 1000.0, 1000.0, 500.0])
    else:
        RHO_BANDS = np.linspace(1500.0, 500.0, NPHASE)
    if permute_middle:
        if NPHASE != 4:
            raise ValueError("permute_middle is defined for nphase=4 only")
        PHASE_PERM = np.array([0, 2, 1, 3])
    else:
        PHASE_PERM = np.arange(NPHASE)
    RHO = RHO_BANDS[PHASE_PERM]              # phase k has the density of ITS band
    # linear k_r rows: the interior phases (their bands are 1 .. N-2)
    MIDDLE_LINEAR = (linear_kr_middle
                     & (PHASE_PERM >= 1) & (PHASE_PERM <= NPHASE - 2))
    _PAIRS = tuple(itertools.combinations(range(NPHASE), 2))
    _THETA = _pinned_theta()
    _MASKS = _side_masks()


# --------------------------------------------------------------------------------------- #
#  Scheme display names -- the HU-BM (Hybrid Upwinding with Background Mobility) family
# --------------------------------------------------------------------------------------- #
# The short tokens on the left stay the canonical keys (CLI --scheme, dict keys, filename tags);
# the HU-BM(...) labels are used wherever a human reads (help text, printed output, figures).
SCHEME_LABELS = {
    "hu":    "HU-BM(ff)",   # simplicial fractional-flow buoyancy  f_a f_b lambda_T
    "hu-mp": "HU-BM(mp)",   # mobility-product buoyancy  lambda_a lambda_b / lambda_T  (= Lee at N=2)
    "hu-mw": "HU-BM(mw)",   # mobility-product buoyancy + harmonic (mobility-weighted) face lambda_T
    "ppu":   "PPU",         # phase-potential upwinding (not an HU-BM member)
}


def scheme_label(scheme):
    """Human-facing display name for a scheme token (falls back to the token itself)."""
    return SCHEME_LABELS.get(scheme, scheme)


LX = 100.0                          # domain width  [m]
LY = 100.0                          # domain height [m]
T_END = 600.0 * DAY                 # [s]

# Time stepping + tolerances, MIRRORED from porepy_2d_solver.py (keep the two solvers in sync):
#   * the step STARTS at DT_INIT_DAYS (small on the density-inverted IC), GROWS to the cap
#     DT_MAX_DAYS, HALVES on a non-converged step down to a floor of DT_MAX_DAYS/64;
#   * NEWTON_ATOL is the max-norm analog of porepy's Lebesgue tol 1e-4 (res_abs criterion);
#   * DRIFT_ORDER mirrors porepy's NullSpaceDriftCriterion: per-step budget on the dt-scaled
#     phase-volume drift, tol = 10^-(K-1) / (2 n_steps), with the same stagnation escape;
#   * the CPR tolerances match porepy_2d_solver's CPR_RTOL / CPR_MAXIT defaults.
DT_INIT_DAYS = 0.01                 # initial adaptive step [days]
DT_MAX_DAYS = 0.5                   # maximum (cap) adaptive step [days]; floor = DT_MAX_DAYS/64
NEWTON_ATOL = 1.0e-4                # per-equation Lebesgue tolerance [kg/s units] -- the EXACT
                                    # porepy res_abs constraint (EquationBasedLebesgueMetric)
NEWTON_MAXIT = 13                   # Newton cap per step (porepy: max_iterations=13)
DRIFT_ORDER = 4                     # conservation target order K for the drift budget
CPR_RTOL = 1.0e-10                  # FGMRES relative tolerance
CPR_MAXIT = 300                     # FGMRES iteration budget
SNAP_DAYS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 25.0, 50.0, 75.0, 78.0,
             100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0, 325.0, 350.0,
             375.0, 400.0, 425.0, 450.0, 475.0, 500.0, 525.0, 550.0, 571.0, 575.0,
             600.0)                     # = porepy_2d_solver.SNAP_DAYS (same conservation sampling)

# Seven impermeable barrier layers, digitized from Bosma et al. (2022) Fig. 5(a).
# Copied VERBATIM from model_configuration/geometry_description/geometry_market.py
# (``_BARRIER_LAYERS_FIG``): keys = FIGURE rows (top = 0, gravity downward) on a 100-cell grid;
# values = inclusive barrier-cell column ranges. Columns NOT listed are the openings.
_BARRIER_LAYERS_FIG = {
    16: [(5, 19), (23, 25), (40, 59), (70, 79), (82, 99)],   # 5 subregions
    23: [(18, 44), (62, 84)],                                # 2
    38: [(0, 9), (18, 25), (38, 49), (55, 74), (90, 94)],    # 5
    45: [(23, 59), (63, 70)],                                # 2
    58: [(2, 17), (22, 29), (48, 59), (70, 99)],             # 4
    74: [(0, 15), (19, 22), (24, 53), (58, 70), (75, 76), (84, 92)],   # 6
    82: [(5, 18), (24, 42), (58, 94)],                       # 3 (bottom)
}


def _cell(i, j, nx):
    return j * nx + i


_REF_CELLS = 100        # the digitized layout is defined on a 100-cell (1 m) reference grid


def _barrier_boxes() -> list[tuple[float, float, float, float]]:
    """Physical bounding boxes ``(x_lo, x_hi, y_lo, y_hi)`` [m] of every barrier segment,
    derived from the 100-cell digitized :data:`_BARRIER_LAYERS_FIG` (figure row 0 = top,
    gravity down; 1 reference cell = ``LX/100`` m wide, ``LY/100`` m tall).

    Figure row ``fig_r`` (from the top) is grid row ``(_REF_CELLS-1)-fig_r`` from the bottom,
    i.e. the physical band ``y in [LY - (fig_r+1)*dy_ref, LY - fig_r*dy_ref]``. Columns
    ``[a, b]`` (inclusive) span ``x in [a*dx_ref, (b+1)*dx_ref]``.
    """
    dx_ref, dy_ref = LX / _REF_CELLS, LY / _REF_CELLS
    boxes = []
    for fig_r, segments in _BARRIER_LAYERS_FIG.items():
        y_lo = LY - (fig_r + 1) * dy_ref
        y_hi = LY - fig_r * dy_ref
        for a, b in segments:
            boxes.append((a * dx_ref, (b + 1) * dx_ref, y_lo, y_hi))
    return boxes


def barrier_mask(nx: int, ny: int) -> np.ndarray:
    """Per-cell boolean barrier mask, GEOMETRY-PRESERVING under mesh refinement.

    A cell is a barrier iff its CENTRE lies inside any barrier segment's physical bounding
    box (:func:`_barrier_boxes`, from the digitized 100-cell layout). So each barrier keeps
    its physical extent independent of resolution -- a 1 m-tall layer is 1 cell thick at
    100x100 and 2 cells thick at 200x200, and every opening keeps its exact x-span. At
    ``nx = ny = 100`` this reproduces the reference ``GeometryBarriers2D.barrier_cell_mask``
    exactly. Works for any (including non-square) mesh.
    """
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))   # (ny, nx); c = j*nx + i
    xc = (ii.ravel() + 0.5) * (LX / nx)
    yc = (jj.ravel() + 0.5) * (LY / ny)
    mask = np.zeros(nx * ny, dtype=bool)
    for (x_lo, x_hi, y_lo, y_hi) in _barrier_boxes():
        mask |= (xc >= x_lo) & (xc <= x_hi) & (yc >= y_lo) & (yc <= y_hi)
    return mask


# --------------------------------------------------------------------------------------- #
#  Grid (structured Cartesian) with cell-wise permeability (barrier cells) + harmonic faces
# --------------------------------------------------------------------------------------- #
@dataclass
class Grid:
    nx: int
    ny: int
    dx: float
    dy: float
    ncell: int
    xc: np.ndarray                # (ncell,) cell-centre x
    yc: np.ndarray                # (ncell,) cell-centre y
    Kcell: np.ndarray             # (ncell,) cell permeability [m^2] (barriers reduced)
    barrier: np.ndarray           # (ncell,) bool barrier mask
    fL: np.ndarray                # (nface,) lower/left  cell index
    fR: np.ndarray                # (nface,) upper/right cell index
    Tf: np.ndarray                # (nface,) transmissibility (harmonic-K) [m^3]
    GC: np.ndarray                # (nface,) gravity coefficient T_f * g * dz  (0 on x-faces)
    Vcell: float                  # cell pore-volume factor |c| = dx*dy (depth = 1)


def make_grid(nx: int = 100, ny: int = 100) -> Grid:
    dx, dy = LX / nx, LY / ny
    ncell = nx * ny
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    xc = ((ii + 0.5) * dx).ravel()
    yc = ((jj + 0.5) * dy).ravel()

    bmask = barrier_mask(nx, ny)
    Kcell = np.full(ncell, K_ROCK)
    Kcell[bmask] = K_ROCK * BARRIER_K_FACTOR

    def harm(a, b):
        s = a + b
        return np.where(s > 0.0, 2.0 * a * b / np.where(s > 0.0, s, 1.0), 0.0)

    fL, fR, Tf, GC = [], [], [], []
    # x-direction internal faces (area dy, distance dx; no gravity)
    for j in range(ny):
        for i in range(nx - 1):
            L, R = _cell(i, j, nx), _cell(i + 1, j, nx)
            fL.append(L); fR.append(R)
            Tf.append((dy / dx) * harm(Kcell[L], Kcell[R])); GC.append(0.0)
    # y-direction internal faces (area dx, distance dy; gravity, dz = +dy since R above L)
    for j in range(ny - 1):
        for i in range(nx):
            L, R = _cell(i, j, nx), _cell(i, j + 1, nx)
            t = (dx / dy) * harm(Kcell[L], Kcell[R])
            fL.append(L); fR.append(R)
            Tf.append(t); GC.append(t * G * dy)

    return Grid(nx=nx, ny=ny, dx=dx, dy=dy, ncell=ncell, xc=xc, yc=yc,
                Kcell=Kcell, barrier=bmask,
                fL=np.array(fL), fR=np.array(fR), Tf=np.array(Tf), GC=np.array(GC),
                Vcell=dx * dy)


# --------------------------------------------------------------------------------------- #
#  Initial condition: N horizontal bands, heaviest on top -> lightest on bottom (inverted)
# --------------------------------------------------------------------------------------- #
def _band_boundaries() -> np.ndarray:
    """Descending y-boundaries ``[LY, 0.9 LY, ..., 0.1 LY, 0]`` of the N initial bands.

    The two EXTREME phases occupy thin 10 % bands at the WRONG ends (heaviest on top, lightest
    on the bottom), and the N-2 INTERIOR phases split the middle 80 % into equal bands in
    descending density order -- so the whole column is density-inverted and segregates.  For
    N=3 this is Bosma's ``[LY, 0.9LY, 0.1LY, 0]`` (heavy top 10 %, oil middle 80 %, light
    bottom 10 %)."""
    if NPHASE == 2:                                # no interior band: split the column in half
        return np.array([LY, 0.5 * LY, 0.0])
    b = [LY, 0.9 * LY]
    w = 0.8 * LY / (NPHASE - 2)
    b += [0.9 * LY - k * w for k in range(1, NPHASE - 1)]
    b.append(0.0)
    return np.array(b)


def initial_saturations(grid: Grid) -> np.ndarray:
    """Full ``(NPHASE, ncell)`` initial saturation array: phase ``k`` fills band ``k``."""
    y = grid.yc
    bnd = _band_boundaries()
    s = np.zeros((NPHASE, grid.ncell))
    for k in range(NPHASE):                        # band b = (bnd[b+1], bnd[b]]; bands tile (0, LY]
        b = PHASE_PERM[k]                          # phase k occupies band b (label permutation)
        s[k][(y > bnd[b + 1]) & (y <= bnd[b])] = 1.0
    return s


def initial_pressure(grid: Grid) -> np.ndarray:
    """Hydrostatic pressure consistent with the layered IC (mirrors the PorePy run script).

    In mechanical equilibrium ``dp/d(depth) = rho_column g``; integrating downward from the
    top (``p = 0`` at ``y = LY``) with the piecewise-constant column density of the initial
    layering makes ``grad p - rho g`` vanish WITHIN each band, so at t = 0 there is no spurious
    pressure-driven flow -- only the density contrast across the band interfaces drives the
    buoyant segregation.  Generalizes the Bosma 3-band integral to N bands."""
    y = grid.yc
    bnd = _band_boundaries()
    column = np.zeros(grid.ncell)
    for k in range(NPHASE):                        # add band k's density over its overlap with [y, LY]
        column += RHO_BANDS[k] * np.maximum(0.0, bnd[k] - np.maximum(bnd[k + 1], y))
    return G * column


# --------------------------------------------------------------------------------------- #
#  Phase properties (quadratic k_r, constant mu / rho)
# --------------------------------------------------------------------------------------- #
def _saturations(x, nc):
    """Full ``(NPHASE, nc)`` saturations from the state ``x = [p, s_0, ..., s_{N-2}]``; the last
    phase is eliminated, ``s_{N-1} = 1 - sum_k s_k``."""
    s = np.empty((NPHASE, nc))
    for k in range(NPHASE - 1):
        s[k] = x[(k + 1) * nc:(k + 2) * nc]
    s[NPHASE - 1] = 1.0 - s[:NPHASE - 1].sum(axis=0)
    return s


def phase_mobilities(s: np.ndarray):
    """Return ``lam`` (N, ncell), ``lamT`` (ncell), ``f`` (N, ncell), ``rho_ff`` (ncell) from the
    full ``(NPHASE, ncell)`` saturation array ``s``."""
    sc = np.clip(s, 0.0, 1.0)
    if MIDDLE_LINEAR.any():                # --linear-kr-middle: linear rows for interior phases
        lam = np.where(MIDDLE_LINEAR[:, None], sc, sc * sc) / MU
    else:
        lam = sc * sc / MU                 # quadratic k_r (Bosma)
    lamT = lam.sum(axis=0)
    safe = np.where(lamT > 0.0, lamT, 1.0)
    f = lam / safe
    rho_ff = (f * RHO[:, None]).sum(axis=0)
    return lam, lamT, f, rho_ff


def _upwind(direction, fL, fR):
    return np.where(direction >= 0.0, fL, fR)


def _advect(cell_q, direction, fL, fR):
    """THE standard first-order upwind primitive: advected CELL quantity + advecting face
    direction -> upstream-cell face value. The buoyancy terms touch faces ONLY through this."""
    return np.where(direction >= 0.0, cell_q[fL], cell_q[fR])


def _harmonic_face(cell_field, fL, fR):
    a, b = cell_field[fL], cell_field[fR]
    s = a + b
    return np.where(s > 0.0, 2.0 * a * b / np.where(s > 0.0, s, 1.0), 0.0)


# --------------------------------------------------------------------------------------- #
#  Frozen (lagged) per-step upwind directions
# --------------------------------------------------------------------------------------- #
@dataclass
class Dirs:
    """Only the STATE-DEPENDENT directions are lagged (total velocity / phase potentials);
    the buoyancy pair directions w_ab = G (rho_a - rho_b) are constants and live inline
    in :func:`_face_fluxes`."""
    scheme: str
    upT: np.ndarray = None
    up_phase: dict = field(default_factory=dict)


def frozen_directions(x, grid, scheme):
    nc = grid.ncell
    p = x[:nc]
    lam, lamT, f, rho_ff = phase_mobilities(_saturations(x, nc))
    fL, fR, Tf, GC = grid.fL, grid.fR, grid.Tf, grid.GC
    dpf = p[fL] - p[fR]
    d = Dirs(scheme=scheme)
    if scheme == "ppu":
        for a in range(NPHASE):
            d.up_phase[a] = _upwind(Tf * dpf - GC * RHO[a], fL, fR)
    else:
        rho_ff_f = 0.5 * (rho_ff[fL] + rho_ff[fR])
        V_T = Tf * dpf - GC * rho_ff_f
        d.upT = _upwind(V_T, fL, fR)
    return d


# --------------------------------------------------------------------------------------- #
#  Face fluxes + residual
# --------------------------------------------------------------------------------------- #
def _face_fluxes(x, grid, dirs):
    """Per-face (L->R) total flux ``qT`` and the ``NPHASE-1`` independent phase fluxes (a list,
    phases 0..N-2; the last phase is the eliminated one)."""
    nc = grid.ncell
    p = x[:nc]
    lam, lamT, f, rho_ff = phase_mobilities(_saturations(x, nc))
    fL, fR, Tf, GC = grid.fL, grid.fR, grid.Tf, grid.GC
    dpf = p[fL] - p[fR]
    q = [None] * NPHASE

    if dirs.scheme == "ppu":
        for a in range(NPHASE):
            Phi_a = Tf * dpf - GC * RHO[a]
            q[a] = lam[a][dirs.up_phase[a]] * Phi_a
        qT = sum(q)
    else:
        rho_ff_f = 0.5 * (rho_ff[fL] + rho_ff[fR])
        V_T = Tf * dpf - GC * rho_ff_f
        upT = dirs.upT
        if dirs.scheme == "hu-mw":                        # HU-BM(mw): harmonic (mobility-weighted)
            qT = _harmonic_face(lamT, fL, fR) * V_T       # total mobility folded into transmissibility
        else:                                             # HU-BM(ff)/(mp): total mobility upwinded
            qT = lamT[upT] * V_T
        for a in range(NPHASE):
            q[a] = f[a][upT] * qT                         # viscous fractional-flow split (total flux)
        for (a, b) in _PAIRS:                             # buoyancy over every edge, +to a / -to b
            if RHO[a] == RHO[b]:
                continue                                  # zero driving force: pair contributes 0
            w_ab = -GC * (RHO[a] - RHO[b])                # THE advecting direction (check 1):
            #                                               a fixed face field, never lagged.
            # CELL-level advected quantities: the m-weighted partial total mobilities
            #     lam_with    = sum_l m_l       lam_l   (rides WITH w_ab)
            #     lam_against = sum_l (1 - m_l) lam_l   (rides AGAINST w_ab)
            # Default: the MIDPOINT sign rule of _side_masks (each phase joins the pair
            # member on its side of the arithmetic-mean separator; m_a = 1, m_b = 0 pinned),
            # so the in-pair HU upwinding is the endpoint case of the same sum -- no
            # in-pair / background split -- and equal-density phases aggregate exactly.
            # rho-ratio keeps the barycentric weights; chi-const the legacy constant split
            # (NOT reduction-consistent; comparison only).
            if BACKGROUND == "chi-const":
                th = np.where(np.arange(NPHASE) == a, 1.0,
                              np.where(np.arange(NPHASE) == b, 0.0, CHI))
            elif BACKGROUND == "rho-ratio":
                th = _THETA[(a, b)]
            else:                                     # midpoint sign rule (default)
                th = _MASKS[(a, b)]
            lam_with = sum(th[l] * lam[l] for l in range(NPHASE))
            lam_against = sum((1.0 - th[l]) * lam[l] for l in range(NPHASE))
            # Faces are touched ONLY through the first-order upwind primitive (check 5):
            lam_T = (_advect(lam_with, w_ab, fL, fR)
                     + _advect(lam_against, -w_ab, fL, fR))
            if dirs.scheme in ("hu-mp", "hu-mw"):         # mobility-product buoyancy U^HU = la*lb / lam_T
                # HU-BM(mp) and HU-BM(mw) share the EXACT simplicial mobility-product buoyant term;
                # they differ only in the viscous total-mobility placement (upwind vs harmonic above).
                # lam_T (the DENOMINATOR) can vanish at fully-segregated faces -> eps.
                b_ab = (_advect(lam[a], w_ab, fL, fR) * _advect(lam[b], -w_ab, fL, fR)
                        / (lam_T + 1.0e-30)) * w_ab
            else:                                         # HU-BM(ff): fractional-flow form fa*fb*lam_T
                b_ab = (_advect(f[a], w_ab, fL, fR) * _advect(f[b], -w_ab, fL, fR)
                        * lam_T * w_ab)
            q[a] = q[a] + b_ab
            q[b] = q[b] - b_ab
    return qT, q[:NPHASE - 1]                              # independent phase fluxes (phases 0..N-2)


def _divergence(face_flux, grid):
    div = np.zeros(grid.ncell)
    np.add.at(div, grid.fL, face_flux)
    np.add.at(div, grid.fR, -face_flux)
    return div


def make_residual(grid, dt, s_old, dirs):
    """Raw ``NPHASE*ncell`` residual ``[pressure | s_0 | ... | s_{N-2}]`` for the CLOSED domain.

    ``s_old`` is the ``(NPHASE-1, ncell)`` previous-step independent saturations. Every cell keeps
    its continuity equation, so the pressure block is SINGULAR (null space = constant pressure);
    the datum is fixed by the Lagrange multiplier in :func:`newton` (global ``Sum p = 0``), NOT by
    dropping a cell equation -- this is what avoids the point-source artifact of a pinned cell.
    """
    nc = grid.ncell
    acc = PHI * grid.Vcell / dt

    def r(x):
        qT, q_indep = _face_fluxes(x, grid, dirs)
        res = np.empty(NPHASE * nc)
        res[:nc] = _divergence(qT, grid)                              # closed, singular
        for k in range(NPHASE - 1):                                   # one balance per solved phase
            s_k = x[(k + 1) * nc:(k + 2) * nc]
            res[(k + 1) * nc:(k + 2) * nc] = acc * (s_k - s_old[k]) + _divergence(q_indep[k], grid)
        return res

    return r


def sparsity_pattern(grid):
    """``NPHASE``-block x 5-point-stencil sparsity of the ``(NPHASE*ncell)`` Jacobian (coloured FD).

    Every cell keeps its full continuity stencil (no pinned row) -- the pressure datum is
    handled by the Lagrange border added in :func:`newton`, not by editing a row here.
    """
    nc = grid.ncell
    nbr = [[c] for c in range(nc)]
    for L, R in zip(grid.fL, grid.fR):
        nbr[L].append(R); nbr[R].append(L)
    rows, cols = [], []
    for c in range(nc):
        for cn in set(nbr[c]):
            for eb in range(NPHASE):
                for vb in range(NPHASE):
                    rows.append(eb * nc + c); cols.append(vb * nc + cn)
    return sps.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(NPHASE * nc, NPHASE * nc))


# --------------------------------------------------------------------------------------- #
#  Linear solver (PETSc FGMRES + CPR by default; SciPy SuperLU fallback)
# --------------------------------------------------------------------------------------- #
class _PetscCPR:
    """PETSc FGMRES + CPR two-stage preconditioner -- a fast ITERATIVE alternative to MUMPS.

    It accepts the bordered Lagrange matrix (size ``3nc+1``) but internally solves the SPARSE
    singular Jacobian ``J`` (the leading ``3nc x 3nc`` block, DROPPING the dense datum border),
    with the constant-pressure null space attached so FGMRES treats the singular-but-consistent
    system correctly. On the closed domain the pressure residual telescopes to zero, so ``J dx =
    -r`` is consistent and the update conserves mass; the pressure is recovered up to a harmless
    additive constant (incompressible physics depends only on ``grad p``). Crucially, NO dense
    border is ever formed, so the ``O(N^2)`` fill that makes the direct MUMPS factorization slow
    never appears -- a bordered matvec is ``O(N)`` and ILU(0) drops the fill.

    Preconditioner (CPR): a multiplicative two-field split -- pressure block -> algebraic
    multigrid (the elliptic part), saturation block -> ILU(0) -- applied pressure-correction
    first, then the local smoother. Returns ``x`` of size ``3nc+1`` (the lambda slot is 0).
    """

    def __init__(self, amg="gamg", rtol=CPR_RTOL, maxit=CPR_MAXIT):
        from petsc4py import PETSc                 # lazy import so the SciPy path needs no PETSc
        self.PETSc = PETSc
        self.amg = amg                             # "gamg" (built-in) or "hypre" (BoomerAMG)
        self.rtol = rtol
        self.maxit = maxit

    def __call__(self, A_csr, rhs):
        PETSc = self.PETSc
        A_csr = A_csr.tocsr()
        A_csr.sort_indices()
        n = A_csr.shape[0]
        nc = (n - 1) // NPHASE                      # [p(nc), s_0(nc), ..., lambda(1)] layout
        n3 = NPHASE * nc
        J = A_csr[:n3, :n3].tocsr()                # drop the dense datum border row/col + lambda
        J.sort_indices()
        f = np.ascontiguousarray(np.asarray(rhs)[:n3], dtype=PETSc.ScalarType)

        ai = J.indptr.astype(PETSc.IntType)
        aj = J.indices.astype(PETSc.IntType)
        av = np.ascontiguousarray(J.data, dtype=PETSc.ScalarType)
        A = PETSc.Mat().createAIJ(size=(n3, n3), csr=(ai, aj, av), comm=PETSc.COMM_SELF)
        A.assemble()

        # constant-pressure null space: 1 on the pressure DOFs (0..nc-1), 0 on saturations.
        v = A.createVecRight()
        arr = v.getArray()
        arr[:] = 0.0
        arr[:nc] = 1.0
        v.assemble()
        v.normalize()
        nsp = PETSc.NullSpace().create(constant=False, vectors=[v], comm=PETSc.COMM_SELF)
        A.setNullSpace(nsp)                         # FGMRES projects it out of the Krylov space

        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        ksp.setOperators(A)
        ksp.setType("fgmres")                       # flexible: the CPR preconditioner is nonlinear
        ksp.setTolerances(rtol=self.rtol, atol=1.0e-50, max_it=self.maxit)

        pc = ksp.getPC()
        pc.setType("fieldsplit")
        is_p = PETSc.IS().createGeneral(np.arange(0, nc, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
        is_s = PETSc.IS().createGeneral(np.arange(nc, n3, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
        pc.setFieldSplitIS(("p", is_p), ("s", is_s))
        pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
        pc.setUp()
        kp, ks = pc.getFieldSplitSubKSP()
        kp.setType("preonly")
        kp.getPC().setType(self.amg)                # pressure -> algebraic multigrid (elliptic)
        ks.setType("preonly")
        ks.getPC().setType("ilu")                   # saturation -> ILU(0) (local/hyperbolic)

        x = A.createVecRight()
        b = A.createVecLeft()
        b.setArray(f)
        nsp.remove(b)                               # make the RHS consistent (remove null part)
        ksp.solve(b, x)
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(f"PETSc CPR KSP diverged (reason {ksp.getConvergedReason()}, "
                               f"its={ksp.getIterationNumber()})")
        out = np.zeros(n, dtype=float)
        out[:n3] = x.getArray()                     # lambda slot stays 0
        return out


def make_linear_solver(kind="cpr"):
    """Return a callable ``solve(A_csr, rhs) -> x`` for ``A x = rhs``.

    ``"cpr"`` (default) -> PETSc FGMRES + CPR two-stage preconditioner (pressure AMG + saturation
    ILU) on the sparse singular system -- iterative, avoids the Lagrange border's dense fill and is
    far faster at scale; falls back to SciPy, with a warning, if petsc4py is unavailable.
    ``"scipy"`` -> ``scipy.sparse.linalg.spsolve`` (SuperLU) on the bordered system.
    """
    def _scipy(A, b):
        return spsolve(A.tocsr(), b)
    if kind == "scipy":
        return _scipy
    try:
        return _PetscCPR()
    except Exception as exc:                     # petsc4py / PETSc not available
        print(f"  [warn] PETSc CPR unavailable ({exc!r}); using scipy spsolve", flush=True)
        return _scipy


# --------------------------------------------------------------------------------------- #
#  Newton time-stepping
# --------------------------------------------------------------------------------------- #
def _project_simplex(s):
    """Clip the ``(NPHASE-1, ncell)`` independent saturations back onto the valid simplex
    (each ``s_k >= 0`` and ``sum_k s_k <= 1`` so the eliminated phase ``>= 0``) after a Newton
    update -- prevents an overshoot from producing negative saturations."""
    s = np.clip(s, 0.0, 1.0)
    tot = s.sum(axis=0)
    scale = np.where(tot > 1.0, 1.0 / np.maximum(tot, 1e-30), 1.0)
    return s * scale


def newton(r, x0, pattern, grid, dt, atol=NEWTON_ATOL, maxit=NEWTON_MAXIT, linsolve=None,
           relag=None, drift_tol=None, backtracking=False):
    """Newton for the CLOSED domain, with a LAGRANGE-MULTIPLIER pressure datum.

    ``r`` is the raw residual, whose pressure block is singular (null space = constant
    pressure). Each iteration solves the bordered saddle-point system

        [ J    b ] [ dx   ]     [ r + b*lam ]
        [ b^T  0 ] [ dlam ]  =  -[  sum_c p  ]

    with ``b`` = 1 on the pressure DOFs, 0 elsewhere. The scalar multiplier ``lam`` supplies the
    (physically ~0) uniform source that makes the singular pressure system consistent, and the
    bottom row enforces the global datum ``sum_c p_c = 0``. NO cell equation is dropped, so
    there is no point-source artifact and total mass is conserved exactly. Optional backtracking
    line search on the physical residual (``backtracking``; default OFF = plain full-step
    Newton, matching porepy_2d_solver); PorePy-style absolute per-equation stop. ``lam`` is ~0 at
    convergence and is not returned. Returns ``(x, n_iter, metric, converged)``.

    ``relag(x)``, if given, refreshes the lagged upwind DIRECTIONS from the current iterate (called
    at ``x0`` and after each accepted step) -- per-Newton-iteration lagging. The directions stay
    fixed WITHIN a single linear solve, so the finite-difference Jacobian never differentiates the
    upwind switch. With ``relag=None`` the directions are whatever ``r`` was built with (lagged
    once per time step).
    """
    if linsolve is None:
        linsolve = lambda A, b: spsolve(A.tocsr(), b)
    nc = grid.ncell
    n3 = NPHASE * nc
    b = np.zeros(n3); b[:nc] = 1.0                     # multiplier column / constraint row
    b_col = sps.csr_matrix(b.reshape(n3, 1))
    b_row = sps.csr_matrix(b.reshape(1, n3))
    # EXPLICIT zero on the border diagonal (n3, n3): the multiplier row/col has a structurally
    # zero pivot, and MUMPS requires that entry to be present (not absent) to pivot around it.
    zero_corner = sps.csr_matrix((np.array([0.0]), (np.array([0]), np.array([0]))), shape=(1, 1))

    # porepy-equivalent residual measures (SAME constraint set as porepy_2d_solver):
    # the transport blocks are phase VOLUME balances [m^3/s]; multiplying by rho_k gives the
    # porepy component-MASS rows [kg/s], and the porepy pressure row (TOTAL mass) is the
    # rho-weighted sum of ALL phase rows, with the reference (lightest) phase's row
    # reconstructed as (pressure block - sum of assembled transport blocks).
    Vc = grid.Vcell
    rho_ref = RHO[NPHASE - 1]

    def total_mass_row(rp):
        r_tm = rho_ref * np.asarray(rp[:nc], float).copy()
        for j in range(1, NPHASE):
            r_tm += (RHO[j - 1] - rho_ref) * rp[j * nc:(j + 1) * nc]
        return r_tm

    def metric(rp):                                    # = porepy EquationBasedLebesgueMetric,
        vals = [np.linalg.norm(total_mass_row(rp))]    #   sqrt(sum r_c^2 / V_c) per equation
        for j in range(1, NPHASE):
            vals.append(RHO[j - 1] * np.linalg.norm(rp[j * nc:(j + 1) * nc]))
        return max(vals) / np.sqrt(Vc)

    # dt-scaled TOTAL-MASS drift (= porepy NullSpaceDriftCriterion): the summed mass residual
    # is the null-space component the linear solve cannot reduce, and it is a rate -- scaled
    # by dt it is what the conservation plots accumulate per step.
    v_tot = grid.Vcell * nc                            # bulk volume [m^3]
    drift_hist: list = []

    def drift(rp):
        return abs(total_mass_row(rp).sum()) * dt / v_tot

    def drift_ok(rp):
        if drift_tol is None:
            return True
        d = drift(rp)
        drift_hist.append(d)
        if d <= drift_tol:
            return True
        recent = drift_hist[-3:]                       # stagnation escape: the drift is frozen
        return len(recent) == 3 and all(
            abs(a - b) <= 1.0e-3 * max(abs(b), 1.0e-300)
            for a, b in zip(recent[:-1], recent[1:]))

    def phys(x, lam):                                  # residual with the multiplier folded in
        rp = r(x)
        rp[:nc] += lam
        return rp

    x = x0.copy()
    lam = 0.0
    if relag is not None:
        relag(x)                                       # lag directions at the initial iterate
    rp = phys(x, lam)
    nrm = np.linalg.norm(rp)
    for it in range(maxit):
        d_ok = drift_ok(rp)                            # every iteration: feeds the escape history
        if metric(rp) <= atol and d_ok:
            return x, it, metric(rp), True
        J = approx_derivative(r, x, method="2-point", sparsity=pattern)
        if not sps.issparse(J):
            J = sps.csr_matrix(J)
        M = sps.bmat([[J, b_col], [b_row, zero_corner]], format="csr")   # bordered saddle-point
        rhs = -np.concatenate([rp, [x[:nc].sum()]])                 # [physical ; datum]
        try:
            delta = linsolve(M, rhs)
        except Exception:
            return x, it, metric(rp), False
        dx, dlam = delta[:n3], delta[n3]
        step = 1.0
        n_trials = 10 if backtracking else 1           # 1 trial = plain full-step Newton
        for _ in range(n_trials):                      # backtracking on the physical residual
            xn = x + step * dx
            xn[nc:] = _project_simplex(xn[nc:].reshape(NPHASE - 1, nc)).ravel()
            rpn = phys(xn, lam + step * dlam)
            nn = np.linalg.norm(rpn)
            if nn < nrm or step < 1.0e-3:
                break
            step *= 0.5
        x, lam = xn, lam + step * dlam
        if relag is not None:                          # refresh directions at the new iterate,
            relag(x)                                   # then re-evaluate the residual under them
            rp = phys(x, lam)
            nrm = np.linalg.norm(rp)
        else:
            rp, nrm = rpn, nn
    return x, maxit, metric(rp), False                 # not converged -> retry with smaller dt


@dataclass
class RunStats:
    """Solver statistics mirroring ``weis_1d_solver`` / PorePy ``NonlinearRunStats``, so the
    solvers are directly comparable. ``total_it`` counts ONLY accepted steps; rejected
    (dt-cut) Newton loops are tallied separately in ``n_time_step_cuts`` / ``it_wasted``."""
    scheme: str
    n_steps: int                    # accepted time steps        (PorePy: n_accepted_steps)
    total_it: int                   # accepted-step Newton iters (PorePy: total_it)
    avg_it: float
    max_it: int                     # PorePy: max_newton_iterations
    n_time_step_cuts: int           # rejected loops / dt-cuts    (PorePy: n_time_step_cuts)
    it_wasted: int                  # Newton iters spent on the rejected loops
    converged: bool                 # every accepted step converged (no stall accepted at floor)
    nit_hist: np.ndarray = None     # per-accepted-step iters     (PorePy: iterations_per_step)
    wall_s: float = 0.0

    def summary(self) -> str:
        return (f"[{scheme_label(self.scheme):9s}] steps={self.n_steps}  total_it={self.total_it}  "
                f"avg_it={self.avg_it:.2f}  max_it={self.max_it}  "
                f"dt_cuts={self.n_time_step_cuts} (wasted_it={self.it_wasted})  "
                f"wall={self.wall_s:.1f}s  {'CONVERGED' if self.converged else 'STALLED'}")


def run(scheme, nx=100, ny=100, dt_days=DT_MAX_DAYS, snap_days=SNAP_DAYS, t_end_days=None,
        atol=NEWTON_ATOL, linear_solver="cpr", dir_lag="iteration", nphase=3, verbose=True,
        dt_init_days=DT_INIT_DAYS, drift_order=DRIFT_ORDER, backtracking=False,
        equal_middle=False, permute_middle=False, linear_kr_middle=False):
    """Advance the ``nphase``-phase segregation to ``t_end_days`` with the chosen ``scheme``.

    ``nphase=3`` reproduces Bosma Fig. 5; ``nphase=4`` splits oil into a mid-heavy + mid-light
    phase (evenly-spaced densities), exercising the SAME simplicial HU family at higher N.

    ``t_end_days`` sets the run horizon in days; when ``None`` it defaults to ``max(snap_days)``
    so the report times and the horizon can never drift apart. Snapshot instants beyond the
    horizon are simply never reached.

    Fully-implicit backward Euler. ``dir_lag`` controls when the upwind DIRECTIONS are lagged:
    ``"iteration"`` (default) refreshes them from the current iterate every Newton iteration
    (fully-implicit upwinding via a fixed-point on the directions); ``"step"`` freezes them once
    per time step from the previous converged state (semi-implicit, Bosma/Weis "old velocity
    field"). Either way the directions are held fixed inside each linear solve (smooth FD
    Jacobian); only the cadence of refreshing differs. A step that does not converge within the
    Newton cap is REJECTED -- NEVER accepted, since a stalled step corrupts the segregation fronts
    (see ``weis_1d_solver``): it is halved and retried, down to a floor of ``dt0 / 64``; ``dt``
    then grows back gradually. ``total_it`` counts accepted steps only.

    Returns ``(grid, snapshots, stats)`` with ``snapshots[day] = dict(sw, so, sg, p)`` and
    ``stats`` a :class:`RunStats`.
    """
    scheme = scheme.lower()
    assert scheme in ("hu", "ppu", "hu-mw", "hu-mp"), scheme
    assert dir_lag in ("iteration", "step"), dir_lag
    set_phase_system(nphase, equal_middle, permute_middle, linear_kr_middle)
    t_end = (t_end_days if t_end_days is not None else max(snap_days)) * DAY
    # Per-step drift budget: target order split over the planned steps, factor-2 margin
    # (= porepy_2d_solver; adaptive cuts only shrink dt and the drift is dt-scaled).
    n_planned = max(1, round(t_end / (dt_days * DAY)))
    drift_tol = 10.0 ** (-(drift_order - 1)) / (2 * n_planned) if drift_order else None
    grid = make_grid(nx, ny)
    pattern = sparsity_pattern(grid)
    linsolve = make_linear_solver(linear_solver)     # reused across all Newton solves
    s0 = initial_saturations(grid)                     # (NPHASE, ncell)
    p0 = initial_pressure(grid)                        # hydrostatic (no spurious initial flow)
    x = np.concatenate([p0] + [s0[k] for k in range(NPHASE - 1)])   # [p, s_0, ..., s_{N-2}]
    nc = grid.ncell

    snaps: dict[float, dict] = {}
    def record(day):
        s = _saturations(x, nc)                        # full (NPHASE, ncell)
        snaps[day] = {"s": s.copy(), "p": x[:nc].copy(),
                      "sw": s[0].copy(), "sg": s[NPHASE - 1].copy()}   # heaviest / lightest aliases
        if NPHASE == 3:
            snaps[day]["so"] = s[1].copy()             # keep the Fig-5 oil field name

    record(0.0)
    # dt_days is the CAP; the step STARTS at dt_init_days and grows toward it (mirrors
    # porepy_2d_solver's make_time_manager: floor = cap/64, clamped so dt_init stays valid).
    dt0 = dt_days * DAY                                # the cap (never exceeded)
    dt_init = dt_init_days * DAY
    dt_floor = min(dt0 / 64.0, dt_init)
    dt = min(max(dt_init, dt_floor), dt0)              # start small, inside [floor, cap]
    t = 0.0
    n_steps = total_it = n_cuts = it_wasted = 0
    nit_hist: list[int] = []
    all_converged = True
    pending = sorted(d for d in snap_days if 0.0 < d <= t_end / DAY + 1e-9)
    t_wall = time.time()
    while t < t_end - 1e-9:
        step_dt = min(dt, t_end - t)
        for d in pending:                                # do not overshoot a snapshot instant
            if t < d * DAY - 1e-9:
                step_dt = min(step_dt, d * DAY - t)
                break
        s_old = x[nc:].reshape(NPHASE - 1, nc).copy()    # previous-step independent saturations
        dirs = frozen_directions(x, grid, scheme)        # directions at the last accepted state
        r = make_residual(grid, step_dt, s_old, dirs)
        relag = None
        if dir_lag == "iteration":                       # refresh directions every Newton iterate,
            def relag(xx, _d=dirs):                      #   mutating the dirs that ``r`` captured
                nd = frozen_directions(xx, grid, scheme)
                _d.upT, _d.up_phase = nd.upT, nd.up_phase
        x_new, its, m, ok = newton(r, x, pattern, grid, step_dt, atol=atol, linsolve=linsolve,
                                   relag=relag, drift_tol=drift_tol, backtracking=backtracking)

        if not ok and step_dt > dt_floor + 1e-30:        # REJECT: never advance on a stalled step
            n_cuts += 1
            it_wasted += its
            dt = max(step_dt * 0.5, dt_floor)            # halve and retry the same interval
            if verbose:
                print(f"  [{scheme_label(scheme)}] t={t/DAY:7.1f} d  DT-CUT -> {dt/DAY:.4f} d "
                      f"(metric={m:.1e}, {its} its wasted)", flush=True)
            continue

        x = x_new                                        # accept
        t += step_dt
        n_steps += 1
        total_it += its
        nit_hist.append(its)
        if not ok:
            all_converged = False                        # accepted a stalled step at the floor
        for d in list(pending):
            if t >= d * DAY - 1e-6:
                record(d); pending.remove(d)
        if ok and its < 5 and dt < dt0:                  # adaptive: grow dt (initial ramp + after cuts)
            dt = min(dt * 2.0, dt0)
        if verbose and n_steps % 25 == 0:
            print(f"  [{scheme_label(scheme)}] t={t/DAY:7.1f} d  step {n_steps}  dt={step_dt/DAY:.3f} d  "
                  f"it={its}  metric={m:.1e}", flush=True)

    if t_end / DAY not in snaps:
        record(t_end / DAY)
    hist = np.asarray(nit_hist, dtype=int)
    stats = RunStats(
        scheme=scheme, n_steps=n_steps, total_it=total_it,
        avg_it=(total_it / n_steps if n_steps else 0.0),
        max_it=(int(hist.max()) if hist.size else 0),
        n_time_step_cuts=n_cuts, it_wasted=it_wasted, converged=all_converged,
        nit_hist=hist, wall_s=time.time() - t_wall)
    if verbose:
        print(stats.summary(), flush=True)
    return grid, snaps, stats


def write_stats(out_dir, stats: RunStats):
    """Persist ``stats`` as a human-readable ``.txt`` (mirrors ``subsection_4_1``'s stats file)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"stats_{stats.scheme.replace('-', '_')}.txt")
    with open(path, "w") as fh:
        fh.write(f"# hamon_2d_solver statistics -- scheme {stats.scheme} "
                 f"({scheme_label(stats.scheme)})\n")
        fh.write(stats.summary() + "\n\n")
        fh.write(f"n_accepted_steps       {stats.n_steps}\n")
        fh.write(f"total_newton_iters     {stats.total_it}   (accepted steps only)\n")
        fh.write(f"avg_iters_per_step     {stats.avg_it:.3f}\n")
        fh.write(f"max_iters_per_step     {stats.max_it}\n")
        fh.write(f"n_time_step_cuts       {stats.n_time_step_cuts}\n")
        fh.write(f"wasted_iters_on_cuts   {stats.it_wasted}\n")
        fh.write(f"all_steps_converged    {stats.converged}\n")
        fh.write(f"wall_seconds           {stats.wall_s:.1f}\n")
        if stats.nit_hist is not None and stats.nit_hist.size:
            fh.write("iterations_per_step    "
                     + " ".join(map(str, stats.nit_hist.tolist())) + "\n")
    return path


# --------------------------------------------------------------------------------------- #
#  VTK RectilinearGrid (.vtr) writer -- cell-centred fields, no external dependency
# --------------------------------------------------------------------------------------- #
def _da(name, arr):
    """One <DataArray> in appended-free inline base64 (Float32/UInt8)."""
    a = np.asarray(arr)
    if a.dtype == bool:
        a = a.astype(np.uint8); vtype = "UInt8"
    else:
        a = a.astype("<f4"); vtype = "Float32"
    raw = a.tobytes()
    header = np.array([len(raw)], dtype="<u4").tobytes()
    payload = base64.b64encode(header).decode() + base64.b64encode(raw).decode()
    return (f'        <DataArray type="{vtype}" Name="{name}" format="binary">\n'
            f'          {payload}\n        </DataArray>\n')


def write_vtr(path, grid: Grid, fields: dict):
    """Write cell-centred ``fields`` on the Cartesian grid as a VTK RectilinearGrid ``.vtr``.

    ``fields`` maps name -> per-cell array (ordered c = j*nx + i, i.e. x fastest -- the VTK
    cell ordering). Coordinates span [0, LX] x [0, LY] x [0, 1].
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nx, ny = grid.nx, grid.ny
    x = (np.arange(nx + 1) * grid.dx).astype("<f4")
    y = (np.arange(ny + 1) * grid.dy).astype("<f4")
    z = np.array([0.0, 1.0], dtype="<f4")
    ext = f"0 {nx} 0 {ny} 0 1"
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write('<VTKFile type="RectilinearGrid" version="1.0" '
                 'byte_order="LittleEndian" header_type="UInt32">\n')
        fh.write(f'  <RectilinearGrid WholeExtent="{ext}">\n')
        fh.write(f'    <Piece Extent="{ext}">\n')
        fh.write('      <CellData>\n')
        for name, arr in fields.items():
            fh.write(_da(name, arr))
        fh.write('      </CellData>\n')
        fh.write('      <Coordinates>\n')
        fh.write(_da("x", x)); fh.write(_da("y", y)); fh.write(_da("z", z))
        fh.write('      </Coordinates>\n')
        fh.write('    </Piece>\n  </RectilinearGrid>\n</VTKFile>\n')


def write_snapshots_vtr(out_dir, scheme, grid, snaps):
    paths = []
    for day in sorted(snaps):
        snap = snaps[day]
        s = snap["s"]                                    # (NPHASE, ncell)
        fields = {f"s_{k}": s[k] for k in range(s.shape[0])}   # s_0 .. s_{N-1} for any N
        fields["p"] = snap["p"]
        fields["barrier"] = grid.barrier
        if s.shape[0] == 3:                              # keep the Fig-5 names for the plots
            fields["s_w"], fields["s_o"], fields["s_g"] = s[0], s[1], s[2]
        path = os.path.join(out_dir, f"hamon_{scheme.replace('-', '_')}_{int(round(day))}d.vtr")
        write_vtr(path, grid, fields)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------------------- #
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="2-D N-phase gravity segregation through barriers (Bosma et al. 2022, Ex. 6.3 "
                    "at --nphase 3). Writes per-snapshot .vtr + a stats .txt.")
    p.add_argument("--scheme", default="all", choices=["hu", "ppu", "hu-mw", "hu-mp", "all"],
                   help="scheme token -- hu=HU-BM(ff), hu-mw=HU-BM(mw), hu-mp=HU-BM(mp), ppu=PPU "
                        "(default: all four)")
    p.add_argument("--nx", type=int, default=100, help="cells in x (default 100)")
    p.add_argument("--ny", type=int, default=100, help="cells in y (default 100)")
    p.add_argument("--dt-days", type=float, default=DT_MAX_DAYS,
                   help=f"MAXIMUM (cap) time step in days (default {DT_MAX_DAYS}); "
                        f"the floor is cap/64")
    p.add_argument("--dt-init-days", type=float, default=DT_INIT_DAYS,
                   help=f"INITIAL time step in days (default {DT_INIT_DAYS}); grows to the cap")
    p.add_argument("--t-end-days", type=float, default=None,
                   help="run horizon in days (default: the largest snapshot time)")
    p.add_argument("--snap-days", type=float, nargs="+", default=list(SNAP_DAYS),
                   metavar="DAY", help="report/snapshot instants in days (default: 0 78 571)")
    p.add_argument("--atol", type=float, default=NEWTON_ATOL,
                   help=f"absolute per-equation Newton tolerance (default {NEWTON_ATOL:.0e}; "
                        f"the max-norm analog of porepy_2d_solver's res_abs Lebesgue tol 1e-4)")
    p.add_argument("--backtracking", action="store_true",
                   help="enable the backtracking line search on the Newton update (residual-"
                        "decrease halving, up to 10 trials; default OFF = plain full-step "
                        "Newton, matching porepy_2d_solver)")
    p.add_argument("--drift-order", type=int, default=DRIFT_ORDER, metavar="K",
                   help=f"conservation target order K for the per-step phase-volume drift "
                        f"budget 10^-(K-1) / (2 n_steps), 0 disables (default {DRIFT_ORDER}; "
                        f"= porepy_2d_solver's NullSpaceDriftCriterion)")
    p.add_argument("--linear-solver", default="cpr", choices=["cpr", "scipy"],
                   help="Newton linear solver: 'cpr' (FGMRES + CPR two-stage preconditioner on the "
                        "sparse singular system -- iterative, fast at scale) or 'scipy' "
                        "(spsolve/SuperLU on the Lagrange bordered system). Default cpr.")
    p.add_argument("--dir-lag", default="iteration", choices=["iteration", "step"],
                   help="when to lag the upwind directions: 'iteration' (default, refresh each "
                        "Newton iterate -- fully-implicit upwinding) or 'step' (freeze once per "
                        "time step from the previous converged state -- semi-implicit).")
    p.add_argument("--nphase", type=int, default=3,
                   help="number of phases (default 3 = Bosma Fig. 5; 4 splits oil into a "
                        "mid-heavy + mid-light phase). Densities are evenly spaced 1500..500.")
    p.add_argument("--equal-middle", action="store_true",
                   help="(with --nphase 4) set BOTH middle densities to the N=3 oil value "
                        "1000, so the 4-phase run degenerates to the 3-phase solution -- the "
                        "ordering-free check. Output defaults to ./vtr_eqmid.")
    p.add_argument("--permute-middle", action="store_true",
                   help="(with --nphase 4) swap the LABELS of the two middle phases (bands + "
                        "densities) -- a pure relabeling, so the solution must be identical "
                        "up to the swap. Output dir gains _perm.")
    p.add_argument("--linear-kr-middle", action="store_true",
                   help="linear k_r = s for the INTERIOR (middle) phases only; makes middle "
                        "mobilities additive so the --equal-middle N=4 run degenerates to "
                        "N=3 EXACTLY. Output dir gains _lkmid.")
    p.add_argument("--background", default="midpoint",
                   choices=["midpoint", "rho-ratio", "chi-const"],
                   help="pair background split: midpoint sign rule m in {0, 1/2, 1} "
                        "(default; pinned, unordered, division-free), rho-ratio "
                        "barycentric weights (_wr), or the legacy constant CHI "
                        "(_chic; NOT reduction-consistent).")
    p.add_argument("--out", default=None,
                   help="output directory for .vtr / stats (default: ./vtr next to this file)")
    p.add_argument("--quiet", action="store_true", help="suppress per-step progress")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    HERE = os.path.dirname(os.path.abspath(__file__))
    _sfx = (("_eqmid" if args.equal_middle else "")
            + ("_perm" if args.permute_middle else "")
            + ("_lkmid" if args.linear_kr_middle else "")
            + {"midpoint": "", "rho-ratio": "_wr", "chi-const": "_chic"}[args.background])
    OUT = args.out or os.path.join(HERE, "vtr" + _sfx)
    BACKGROUND = args.background     # module top-level scope: rebinding the global
    schemes = ("hu", "ppu", "hu-mw", "hu-mp") if args.scheme == "all" else (args.scheme,)
    results = {}
    for scheme in schemes:
        grid, snaps, stats = run(scheme, nx=args.nx, ny=args.ny, dt_days=args.dt_days,
                                 snap_days=tuple(args.snap_days), t_end_days=args.t_end_days,
                                 atol=args.atol, linear_solver=args.linear_solver,
                                 dir_lag=args.dir_lag, nphase=args.nphase, verbose=not args.quiet,
                                 dt_init_days=args.dt_init_days, drift_order=args.drift_order,
                                 backtracking=args.backtracking, equal_middle=args.equal_middle,
                                 permute_middle=args.permute_middle,
                                 linear_kr_middle=args.linear_kr_middle)
        paths = write_snapshots_vtr(OUT, scheme, grid, snaps)
        paths.append(write_stats(OUT, stats))
        results[scheme] = stats
        for p in paths:
            print(f"  wrote {os.path.relpath(p, HERE)}")
    print("\n=== solver statistics (comparable to weis_1d_solver / PorePy) ===")
    for scheme, st in results.items():
        print("  " + st.summary())
