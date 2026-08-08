"""Meteoric-recharge / halite-dissolution 2D solver (subsection 4.2, two_dimensional).

A gravity/head-driven flow cell through a halite-bearing HOT aquifer: dilute meteoric water
recharges at a topographic high (high head), flows through the deep LIQUID + halite reservoir
dissolving the immobile halite, and rises to a low-pressure discharge vent where it FLASHES to
steam.  The reservoir is liquid (deep, high-p -> well-conditioned); the boiling is confined to the
vent, giving the mobile vapor + liquid that HU (hybrid upwinding) needs to do anything.

Geometry (RechargeGeometry2D): a LX x LZ rectangle; y is vertical (y = LZ top, y = 0 base).
    recharge  = top face, x < RECHARGE_FRAC*LX   (top-left)
    discharge = top face, x > DISCHARGE_FRAC*LX  (top-right)

Initial condition -- phz-consistent hydrostatic column (enthalpy from the SOLVER's flash):
    pressure  : hydrostatic  dp/dx3 = rho_ff g  (rho_ff = fractional-flow density; _hydrostatic_p)
    enthalpy  : h(x3) found by a phz FLASH SEARCH so the flash returns the target T (_h_from_T_phz).
                In the two-phase band T is flat in h, so enthalpy -- not T -- resolves the phase
                split; searching the phz flash directly (not bridging through the ptz H) makes the
                eliminated-T/saturation closure residual EXACTLY 0 at t=0.
    NaCl z    : constant (halite-saturated -> s_h > 0)
With the previous-time-step store synced to the IC (_sync_prev_timestep_to_ic -> accumulation = 0 at
t=0), this IC is an EXACT discrete equilibrium: --equilibrate holds at MACHINE ZERO (residual ~1e-15,
0 recomputes); the forced recharge/discharge run starts from ~1e-1 with no recomputes.

Boundary conditions:
    recharge  (Dirichlet p, T, z): p = P_RECHARGE (high head), T = T_RECHARGE (cold), z = 0 (dilute)
    discharge (Dirichlet p, Neumann T): p = P_DISCHARGE (low head); zero conductive flux so the
              fluid temperature is advected out (not pinned to a boundary value)
    base      (Dirichlet T only, no fluid flow): T = T_BOTTOM_BC (350 C) -- the geothermal heat
    every other face: no-flow, adiabatic

--equilibrate (static IC check): drops all recharge/discharge forcing, holds the column ISOTHERMAL at
--t-equil (base BC matched to it, so nothing drives the system), and lets the solver sit on the IC --
a direct test that the initial condition is a discrete equilibrium. It now holds at machine zero.
Which single mobile phase the column settles on is set by TEMPERATURE (through the boiling pressure),
NOT by --p-top: the boiling curve is bistable and the liquid-seeded hydrostatic Picard resolves it.

    --t-equil  --z    p (MPa)        state                                 mobile phase
    230 C      0.5    2.55 -> 23.45  liquid + halite  (s_v 0,   s_h 0.15)  liquid
    350 C      0.95   2.00 ->  2.14  vapor  + halite  (s_v 0.93, s_h 0.06) vapor

  230 C: boiling p ~2 MPa -> the column is on the LIQUID branch (rho ~1100, builds to 23 MPa and pins
         its own pressure); two phases, liquid + immobile halite, no vapor.
  350 C: boiling p ~16 MPa >> P_TOP -> the whole column is on the VAPOR branch (rho ~9, nearly
         weightless, barely reaches 2.14 MPa). This is the volcanic vapor+halite state that used to
         collapse (p 2.0 -> 0.22, -89 % mass); it now holds exactly -- the collapse was the
         previous-time-step accumulation artifact, not vapor compressibility.
  Both hold at machine zero. HU is idle in either static equilibrium (a single mobile phase); it does
  its work in the FORCED run launched from the IC. Examples:
    liquid: python porepy_2d_recharge.py --equilibrate --no-barriers --t-equil 230 --z-top 0.5  --z-bottom 0.5  --p-top 2 --report-every-years 0.2 --end-years 1
    vapor : python porepy_2d_recharge.py --equilibrate --no-barriers --t-equil 350 --z-top 0.95 --z-bottom 0.95 --p-top 2 --report-every-years 0.2 --end-years 1

Reuses the subsection_4_2 machinery: graded OBL tables, Schur-CPR (PETSc), the weis backtracking
line search, the slave (exact flash each iterate) and the shared base nonlinear criterion.

Reference run (now the defaults, so bare ``python porepy_2d_recharge.py`` reproduces it):
    --p-recharge 2.5 --report-every-years 10 --end-years 2000   with the barriers ON.
Robust, shows clean results. Timing on the dev machine: ~1:16 h wall
(25799.97 s user + 15377.62 s system, ~902% CPU), 200 VTU snapshots. Use --no-barriers to disable.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import cast

import numpy as np

import porepy as pp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    Geometry,
)
from porepy.examples.geothermal_flow.model_configuration.DriesnerModelConfiguration import (  # noqa: E501
    DriesnerBrineFlowModel,
    DriesnerBrineFractionalFlowModel,
)
from porepy.examples.geothermal_flow.model_configuration.flow_model_base import (  # noqa: E501
    geothermal_nonlinear_solver,
)
from porepy.examples.geothermal_flow.model_configuration.geothermal_export import (  # noqa: E501
    DriesnerPhaseExport,
)
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_two_phase_Figure_8_left_panel as BC,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_two_phase_Figure_8_left_panel as IC,
)
from porepy.examples.geothermal_flow.obl_sampler import VTKSampler

day_to_second = 86400.0
year_to_second = 365.0 * day_to_second
to_Mega = 1.0e-6
G = pp.GRAVITY_ACCELERATION

# ------------------------------------------------------------------ geometry
LX = 4000.0                 # domain width [m]
LZ = 2000.0                 # domain height [m]  (y vertical: y=LZ top, y=0 base)
CELL_SIZE = 100.0           # target cell size [m]
RECHARGE_FRAC = 0.25        # recharge patch: top face, x < RECHARGE_FRAC*LX
DISCHARGE_FRAC = 0.75       # discharge patch: top face, x > DISCHARGE_FRAC*LX

# ------------------------------------------------------------------ initial condition
# Option 2 -- VAPOR + halite reservoir, cold-liquid recharge.  IC = the isothermal vapor+halite
# equilibrium (T = 300 C, z = 0.95, P_TOP = 2 MPa -> s_v ~ 0.93 mobile vapor, s_h ~ 0.06 halite, no
# liquid); it holds at machine zero (see the --equilibrate block in the module docstring). Cold
# LIQUID recharges the top-left (60 C, 2.5 MPa, fresh): the dense brine plunges into the light vapor
# -> a ~100x density contrast drives vigorous buoyant counter-flow (HU active) WITHOUT boiling, and
# being under-saturated it DISSOLVES halite -> s_h decreases, staying away from the s_h -> 1
# permeability singularity that a boiling/precipitation plume would hit (loss of coercivity).
P_TOP_IC = 2.0              # IC pressure at the top [MPa]  (whole column vapor at 300 C)
RHO_REF = 1000.0            # brine reference density for the hydrostatic first guess [kg/m^3]
# Isothermal vapor+halite IC: p is vapor-hydrostatic (rho_ff = rho_v ~ 9, so p barely builds, 2.0 ->
# 2.14 MPa), h from the phz flash search at the constant T, z halite-saturated (s_h > 0). At 300 C
# the halite-brine boiling pressure is ~6-7 MPa >> P_TOP, so the column is single-phase vapor.
T_TOP_IC = 300.0 + 273.15   # IC temperature [K]  (ISOTHERMAL -- the equilibrium state)
T_BOT_IC = 300.0 + 273.15   # IC temperature [K]  (= T_TOP_IC)
Z_TOP = 0.95                 # NaCl overall fraction [-] (z=0.95 -> s_h ~ 0.06 in the vapor)
Z_BOTTOM = 0.95              # NaCl overall fraction [-]

# ------------------------------------------------------------------ boundary conditions
P_RECHARGE = 4.0            # recharge (inlet) pressure [MPa] 
T_RECHARGE = 80.0 + 273.15  # recharge temperature [K]  (COLD liquid meteoric water)
Z_RECHARGE = 0.0            # recharge salinity [-]  (dilute / fresh -> dissolves halite)
P_DISCHARGE = 2.0           # discharge (outlet) pressure [MPa]
T_DISCHARGE = 300.0 + 273.15# discharge temperature [K]  (not imposed: energy is Neumann at the outlet)
T_BOTTOM_BC = 300.0 + 273.15# fixed base temperature [K]  (matches the 300 C IC -> no thermal driving)


# ------------------------------------------------------------------ low-k barriers (aquitards)
# Staggered partial aquitard lenses: a cell whose CENTROID falls inside a bounding box has its
# permeability cut by --barrier-factor (~impervious). Six lenses at six stratigraphic levels,
# laterally offset with alternating gaps, so no level is fully sealed and flow weaves down around
# the lens ends -- a geological confining-bed stack over the recharge->discharge cell. Boxes are
# (x_min, x_max, y_min, y_max) in metres; y is elevation (y=LZ top, y=0 base).
BARRIER_PERM_FACTOR = 1.0e-3
_BARRIERS = [   # thin (~100 m / single cell-layer) beds
    ( 400.0, 1800.0, 1700.0, 1800.0),   # shallow, left  (+4-cell far-left sink gap: recharge descends)
    (2200.0, 4000.0, 1500.0, 1600.0),   # shallow-mid, right   (gap: x < 2200)
    ( 500.0, 2500.0, 1200.0, 1300.0),   # mid, centre          (gaps: both ends)
    (2600.0, 4000.0,  900.0, 1000.0),   # mid-deep, right      (gap: x < 2600)
    (   0.0, 1400.0,  600.0,  700.0),   # deep, left           (gap: x > 1400)
    (1800.0, 3600.0,  300.0,  400.0),   # deep, centre-right   (gaps: both ends)
]


def _linear_z(depth: np.ndarray) -> np.ndarray:
    return Z_TOP + (Z_BOTTOM - Z_TOP) * (depth / LZ)


def _linear_T(depth: np.ndarray) -> np.ndarray:
    return T_TOP_IC + (T_BOT_IC - T_TOP_IC) * (depth / LZ)


def _linear_p(depth: np.ndarray) -> np.ndarray:
    return P_TOP_IC + RHO_REF * G * depth * to_Mega


def _fractional_flow_density(pd) -> np.ndarray:
    """Buoyancy density used by the pressure equation's gravity term: the MASS-fractional-flow
    weighted density  rho = sum_j f_j rho_j,  f_j = m_j / sum_k m_k,  m_j = rho_j k_r(s_j) / mu_j
    (fluid_property_library.fractionally_weighted_density). This is NOT the VTR 'Rho' field, which
    is the saturation-weighted bulk density sum_j s_j rho_j -- a different quantity that includes
    the IMMOBILE halite's weight. Integrating the fluid hydrostatic with the bulk density leaves
    dp/dz != (solver buoyancy) g, so the first step relaxes the pressure and it never returns.
    Weis option-B rel-perm (the (1-s_h)^2 abs-perm factor cancels in f_j); halite has k_r = 0."""
    sl = np.asarray(pd["S_l"], float); sv = np.asarray(pd["S_v"], float); sh = np.asarray(pd["S_h"], float)
    rho_l = np.asarray(pd["Rho_l"], float); rho_v = np.asarray(pd["Rho_v"], float)
    mu_l = np.asarray(pd["mu_l"], float);   mu_v = np.asarray(pd["mu_v"], float)
    kr_l = np.clip((sl / np.clip(1.0 - sh, 1.0e-12, None) - 0.3) / 0.7, 0.0, None)   # Weis liquid k_r
    kr_v = 1.0 - kr_l                                                                # option B: sum = 1
    m_l = np.where(kr_l > 0.0, rho_l * kr_l / np.clip(mu_l, 1.0e-30, None), 0.0)     # rho_j k_r/mu_j
    m_v = np.where(kr_v > 0.0, rho_v * kr_v / np.clip(mu_v, 1.0e-30, None), 0.0)
    mt = m_l + m_v
    return np.where(mt > 0.0, (m_l * rho_l + m_v * rho_v) / np.where(mt > 0.0, mt, 1.0), rho_v)


def _h_from_T_phz(z, p, T_K, sampler_phz) -> np.ndarray:
    """Enthalpy h [MJ/kg] such that the PHZ flash returns temperature T_K [K], per node
    (vectorised bisection; T(h) is monotone non-decreasing at fixed z, p). This is the
    phz-consistent way to impose a constant-T column: enthalpy varies with depth to hold T fixed.
    In the two-phase band T is flat in h, so it lands on the saturated edge; in the single-phase
    liquid body (the bulk of a liquid-dominated column) h is unique. Searching the phz flash
    directly -- not bridging through the ptz H -- keeps the IC exactly consistent with the flash
    the solver evaluates, so the eliminated-T closure residual is ~0 at t = 0."""
    z = np.asarray(z, float); p = np.asarray(p, float); T_K = np.asarray(T_K, float)
    lo = np.full(p.shape, 1.0e-4); hi = np.full(p.shape, 4.7)              # phz h-axis bounds [MJ/kg]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        sampler_phz.sample_at(np.column_stack([z, mid, p]))
        Tm = np.asarray(sampler_phz.sampled_could.point_data["Temperature"], float)   # [K]
        below = Tm < T_K
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


_HYDRO_CACHE: dict = {}


def _hydrostatic_p(depth: np.ndarray, sampler, sampler_phz=None) -> np.ndarray:
    """IC pressure in true hydrostatic balance with the SOLVER's gravity term: integrate
    dp = rho_ff(z,T,p) g dz down from P_TOP_IC, where rho_ff is the mass-fractional-flow density
    (:func:`_fractional_flow_density`), i.e. sum_j f_j rho_j -- exactly the buoyancy density the
    pressure equation uses. Using the saturation-weighted VTR 'Rho' instead (sum_j s_j rho_j, which
    counts the immobile halite's weight) leaves dp/dz != (solver buoyancy) g, so the first step
    relaxes the pressure. rho depends (weakly) on p, so Picard-iterate. Cached per (P_TOP_IC, z/T)."""
    key = (P_TOP_IC, Z_TOP, Z_BOTTOM, T_TOP_IC, T_BOT_IC)
    prof = _HYDRO_CACHE.get(key)
    if prof is None:
        dz = CELL_SIZE
        n = int(round(LZ / dz))
        dc = (np.arange(n) + 0.5) * dz                       # cell-center depths, top -> bottom
        z, T = _linear_z(dc), _linear_T(dc)                  # T in Kelvin (sampler translates -273.15)
        p = P_TOP_IC + RHO_REF * G * dc * to_Mega            # constant-rho first guess
        rho = np.full(n, RHO_REF)
        for _ in range(50):
            if sampler_phz is not None:                          # phz-consistent: search h so the phz
                H = _h_from_T_phz(z, p, T, sampler_phz)           # flash returns the target T, then read
                sampler_phz.sample_at(np.column_stack([z, H, p])) # phase props at (z, h, p) from it
                pd = sampler_phz.sampled_could.point_data
            else:                                                # fallback: ptz T -> H bridge
                sampler.sample_at(np.column_stack([z, T, p]))
                pd = sampler.sampled_could.point_data
            rho = _fractional_flow_density(pd)               # solver buoyancy density, NOT pd["Rho"]
            pn = np.empty_like(p)
            pn[0] = P_TOP_IC + rho[0] * G * (dz * 0.5) * to_Mega        # surface -> first center
            for k in range(1, n):
                pn[k] = pn[k - 1] + 0.5 * (rho[k - 1] + rho[k]) * G * dz * to_Mega
            if np.max(np.abs(pn - p)) < 1.0e-10:
                p = pn
                break
            p = pn
        # add surface (depth 0) and base (depth LZ) nodes so interpolation covers the full range
        d_nodes = np.concatenate(([0.0], dc, [LZ]))
        p_nodes = np.concatenate(([P_TOP_IC], p, [p[-1] + rho[-1] * G * (dz * 0.5) * to_Mega]))
        _HYDRO_CACHE[key] = (d_nodes, p_nodes)
        prof = _HYDRO_CACHE[key]
    d_nodes, p_nodes = prof
    return np.interp(depth, d_nodes, p_nodes)


class RechargeGeometry2D(Geometry):
    """LX x LZ rectangle; recharge (top-left) and discharge (top-right) patches on the top face."""

    def set_domain(self) -> None:
        self._domain = pp.Domain({"xmax": self.units.convert_units(LX, "m"),
                                  "ymax": self.units.convert_units(LZ, "m")})

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        return {"cell_size": self.units.convert_units(CELL_SIZE, "m")}

    def get_inlet_outlet_sides(self, sd):
        fc = sd.face_centers.T if isinstance(sd, pp.Grid) else sd.cell_centers.T
        bf = self.domain_boundary_sides(sd).all_bf
        x, y = fc[bf, 0], fc[bf, 1]
        top = y > LZ - 1.0
        recharge = bf[top & (x < RECHARGE_FRAC * LX)]
        discharge = bf[top & (x > DISCHARGE_FRAC * LX)]
        return recharge, discharge


class BCRecharge(BC):
    """Recharge / discharge Dirichlet on the top, a fixed-T geothermal base, no-flow elsewhere."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if _args.equilibrate:                       # IC-evolution test: closed box, but PIN ONE cell's
            top = np.where(self.domain_boundary_sides(sd).north)[0]   # pressure (Dirichlet reference)
            pin = top[len(top) // 2:len(top) // 2 + 1]               # to fix the gauge: pure Neumann
            return pp.BoundaryCondition(sd, pin, "dir")              # leaves the constant mode free ->
        rech, disch = self.get_inlet_outlet_sides(sd)               # level drifts -> uniform collapse
        return pp.BoundaryCondition(sd, np.concatenate((rech, disch)), "dir")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # Dirichlet (fixed T) only where temperature is genuinely imposed: the recharge inflow
        # (known inlet T) and the geothermal base. The DISCHARGE is an outflow -> Neumann (zero
        # conductive flux), so the fluid's own temperature is advected out with the flow instead
        # of being pinned to a boundary value.
        base = np.where(self.domain_boundary_sides(sd).south)[0]      # geothermal base (fixed T)
        if _args.equilibrate:                       # only the base is heated; the whole top adiabatic
            return pp.BoundaryCondition(sd, base, "dir")
        rech, _disch = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, np.concatenate((rech, base)), "dir")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # Advective enthalpy follows the Darcy BC (its default). In --equilibrate that means only the
        # single pinned pressure cell carries advective enthalpy, and its boundary value is the same
        # isothermal interior state (uniform T, z), so nothing spurious enters.
        return self.bc_type_darcy_flux(sd)

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        depth = LZ - bg.cell_centers[1]
        p = _hydrostatic_p(depth, self.obl_sampler_ptz, self.obl_sampler)              # same hydrostatic column as the IC
        if _args.equilibrate:                        # no forcing: the top keeps the IC top pressure
            return p                                 # (= P_TOP_IC at depth 0), uniform across the top
        rech, disch = self.get_inlet_outlet_sides(bg)
        p[rech] = P_RECHARGE
        p[disch] = P_DISCHARGE
        return p

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        depth = LZ - bg.cell_centers[1]
        T = _linear_T(depth)                                          # valid everywhere
        T[self.domain_boundary_sides(bg).south] = T_BOTTOM_BC         # geothermal base
        if not _args.equilibrate:                                     # no recharge/discharge forcing
            rech, disch = self.get_inlet_outlet_sides(bg)             # in the IC-evolution test: the
            T[rech] = T_RECHARGE                                      # whole top keeps the interior T
            T[disch] = T_DISCHARGE
        return T

    def bc_salinity(self, bg: pp.BoundaryGrid) -> np.ndarray:
        # background = the local IC z (so no-flow / outflow faces match the interior); recharge fresh.
        z = _linear_z(LZ - bg.cell_centers[1])
        if not _args.equilibrate:                                     # no fresh recharge when equilibrating
            rech, _ = self.get_inlet_outlet_sides(bg)
            z[rech] = Z_RECHARGE
        return z

    def bc_values_overall_fraction(self, component: pp.Component, bg: pp.BoundaryGrid) -> np.ndarray:
        return self.bc_salinity(bg)

    def bc_values_enthalpy(self, bg: pp.BoundaryGrid) -> np.ndarray:
        p, t, z = self.bc_values_pressure(bg), self.bc_values_temperature(bg), self.bc_salinity(bg)
        return _h_from_T_phz(z, p, t, self.obl_sampler)          # phz-consistent (matches the IC)

    def bc_values_fractional_flow_component(self, component: pp.Component, bg: pp.BoundaryGrid) -> np.ndarray:
        z = self.bc_salinity(bg)
        is_salt = component == self.fluid.components[1]
        return z if is_salt else 1.0 - z


class ICRecharge(IC):
    """Linear p / T / z with depth; the halite (and enthalpy, saturations) come from the flash."""

    def _profiles(self, sd: pp.Grid):
        depth = LZ - sd.cell_centers[1]
        return _hydrostatic_p(depth, self.obl_sampler_ptz, self.obl_sampler), _linear_T(depth), _linear_z(depth)

    def _sampled(self, sd: pp.Grid):
        # phz-consistent IC: enthalpy h from a phz search for the target (constant) T, then the
        # secondaries (S_v, Xl, Xv, ...) from the SAME phz flash at (z, h, p). Returns (point_data, h).
        p, t, z = self._profiles(sd)
        h = _h_from_T_phz(z, p, t, self.obl_sampler)
        self.obl_sampler.sample_at(np.column_stack([z, h, p]))
        return self.obl_sampler.sampled_could.point_data, h

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return self._profiles(sd)[0]

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return self._profiles(sd)[1]

    def ic_values_overall_fraction(self, component: pp.Component, sd: pp.Grid) -> np.ndarray:
        return self._profiles(sd)[2]

    def ic_values_partial_fractions(self, sd: pp.Grid) -> np.ndarray:
        d, _ = self._sampled(sd)
        return np.clip(d["Xl"], 0.0, 1.0), np.clip(d["Xv"], 0.0, 1.0)

    def ic_values_gas_saturation(self, sd: pp.Grid) -> np.ndarray:
        return np.clip(self._sampled(sd)[0]["S_v"], 0.0, 1.0)

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        return self._sampled(sd)[1]


# --------------------------------------------------------------------------- CLI + run
_SCHEME_CONFIG = {
    "hu":    dict(fractional_flow=False, buoyancy_upwinding="hybrid"),
    "hu-mw": dict(fractional_flow=True,  buoyancy_upwinding="hybrid"),
}
_DEFAULT_SNAP_YEARS = tuple(float(y) for y in range(0, 50001, 1000))  # 0..50 kyr every 1 kyr (frequent VTU)

_ap = argparse.ArgumentParser(description="Meteoric-recharge halite-dissolution 2D aquifer.")
_ap.add_argument("--scheme", default="hu", choices=list(_SCHEME_CONFIG))
_ap.add_argument("--consistent", action="store_true", help="MPFA (consistent) flux; default TPFA")
_ap.add_argument("--grid-type", default=None, choices=["cartesian", "simplex"])
_ap.add_argument("--cell-size", type=float, default=None, metavar="M")
_ap.add_argument("--p-recharge", type=float, default=P_RECHARGE, metavar="MPA")
_ap.add_argument("--p-discharge", type=float, default=P_DISCHARGE, metavar="MPA")
_ap.add_argument("--t-recharge", type=float, default=T_RECHARGE - 273.15, metavar="C",
                 help="recharge (meteoric) temperature [degC]; default 100")
_ap.add_argument("--z-top", type=float, default=Z_TOP, metavar="Z", help="IC NaCl fraction at the top")
_ap.add_argument("--z-bottom", type=float, default=Z_BOTTOM, metavar="Z", help="IC NaCl fraction at the base")
_ap.add_argument("--snap-years", type=float, nargs="+", default=list(_DEFAULT_SNAP_YEARS), metavar="YR")
_ap.add_argument("--report-every-years", type=float, default=10.0, dest="report_every_years", metavar="N",
                 help="regular VTU cadence: report every N years up to --end-years (0, N, 2N, ...); "
                      "overrides --snap-years (default 10)")
_ap.add_argument("--end-years", type=float, default=2000.0, dest="end_years", metavar="YR",
                 help="total simulation time [years] used with --report-every-years (default 2000)")
_ap.add_argument("--dt-nominal", type=float, default=1.0, metavar="YR")
_ap.add_argument("--dt-min", type=float, default=0.0001, metavar="YR")
_ap.add_argument("--dt-max", type=float, default=10.0, metavar="YR")
_ap.add_argument("--lag-buoyancy", action="store_true")
_ap.add_argument("--no-barriers", dest="barriers", action="store_false", default=True,
                 help="disable the staggered low-k aquitard beds (ON by default; see _BARRIERS)")
_ap.add_argument("--barrier-factor", type=float, default=BARRIER_PERM_FACTOR, metavar="F",
                 dest="barrier_factor", help="barrier permeability = matrix * F (default 1e-4)")
_ap.add_argument("--equilibrate", action="store_true",
                 help="IC-evolution test: ISOTHERMAL column at --t-equil (no recharge/discharge "
                      "forcing, base BC matched to the column T), so the system just relaxes the IC "
                      "toward equilibrium -- a true static check of the initial condition")
_ap.add_argument("--t-equil", type=float, default=300.0, metavar="C",
                 help="--equilibrate isothermal temperature [degC] (also the matched base BC); "
                      "300 -> vapor+halite (Option 2 IC); ~230 -> liquid+halite")
_ap.add_argument("--p-top", type=float, default=P_TOP_IC, metavar="MPA",
                 help="IC pressure at the top [MPa]; > boiling p at --t-equil -> whole column liquid "
                      "(no vapor cap), = boiling p -> thin cap over liquid (default %(default)s)")
_args = _ap.parse_args()

P_RECHARGE = _args.p_recharge
P_DISCHARGE = _args.p_discharge
T_RECHARGE = _args.t_recharge + 273.15
Z_TOP = _args.z_top
Z_BOTTOM = _args.z_bottom
P_TOP_IC = _args.p_top

if _args.equilibrate:                       # isothermal equilibrium: constant-T column, base matched
    T_TOP_IC = T_BOT_IC = _args.t_equil + 273.15    # (no geothermal gradient -> nothing drives it)
    T_BOTTOM_BC = _args.t_equil + 273.15

if _args.report_every_years is not None and _args.report_every_years > 0.0:
    _n = _args.report_every_years                    # regular reporting: 0, N, 2N, ... up to end
    _args.snap_years = [float(y) for y in np.arange(0.0, _args.end_years + 0.5 * _n, _n)]

# Export cadence is DECOUPLED from the time-step hard-stops. Putting every report time in the
# TimeManager schedule makes each a hard stop that clamps dt to the report interval, so dt can
# never grow. Instead the schedule is just [0, final]: dt starts at --dt-nominal and the time
# manager grows it (up to --dt-max) by iteration count. VTU export fires on CROSSING each report
# time (GeothermalRechargeModel.save_data_time_step), so output stays regular while dt adapts.
_export_times = [y * year_to_second for y in _args.snap_years]
_export_times_pos = [t for t in _export_times if t > 1.0e-9]     # t=0 is written before run()
_final_time = max(_export_times) if _export_times else _args.end_years * year_to_second
schedule = [0.0, _final_time]
time_manager = pp.TimeManager(
    schedule=schedule, dt_init=_args.dt_nominal * year_to_second,
    dt_min_max=(_args.dt_min * year_to_second, _args.dt_max * year_to_second),
    constant_dt=False, iter_max=13, iter_optimal_range=(3, 8),
    iter_relax_factors=(0.5, 1.5), recomp_factor=0.3, print_info=True)

HERE = os.path.dirname(os.path.abspath(__file__))
tag = _args.scheme + ("_mpfa" if _args.consistent else "") + (
    f"_{_args.grid_type}" if _args.grid_type else "") + ("_equilibrate" if _args.equilibrate else "")
params = {
    "folder_name": os.path.join(HERE, "visualization_recharge_" + tag),
    "enable_buoyancy_effects": True,
    "material_constants": {"solid": pp.SolidConstants(
        permeability=1e-15, porosity=0.1, thermal_conductivity=2.0 * to_Mega,
        density=2700.0, specific_heat_capacity=880.0 * to_Mega)},
    "time_manager": time_manager,
    "times_to_export": list(schedule),
    "use_petsc": True, "petsc_preconditioner": "cpr",
    "cpr_rtol": 1.0e-5, "cpr_maxit": 400, "cpr_accuracy_tol": 1.0e-3,
    "step_control_method": "LS",          # weis backtracking line search
    "slave_eliminated_secondaries": True,  # exact flash each Newton iterate
    "consistent_discretization": _args.consistent,
    "lag_buoyancy_direction": _args.lag_buoyancy,
}
if _args.grid_type is not None:
    params["grid_type"] = _args.grid_type
params.update(_SCHEME_CONFIG[_args.scheme])
FlowModel = (DriesnerBrineFractionalFlowModel if params["fractional_flow"]
             else DriesnerBrineFlowModel)


class GeothermalRechargeModel(
    DriesnerPhaseExport, RechargeGeometry2D, BCRecharge, ICRecharge, FlowModel
):
    def meshing_arguments(self) -> dict:
        mesh_args = super().meshing_arguments()
        if _args.cell_size is not None:
            mesh_args = {**mesh_args, "cell_size": self.units.convert_units(_args.cell_size, "m")}
        return mesh_args

    def before_nonlinear_loop(self) -> None:
        # The iteration-0 assembly reads the flash surrogate directly (no update_derived_quantities
        # runs before it -- that only happens inside check_convergence, after the first solve). On the
        # first step the surrogate is still at its init value, so the eliminated-temperature closure
        # starts at ~15 instead of 0, the Schur RHS pollutes the primary solve, and Newton takes a
        # giant bogus first step it never recovers from. Sync the surrogate to f(p,h,z) here.
        super().before_nonlinear_loop()
        self.update_derived_quantities()
        if not getattr(self, "_ic_prev_synced", False):
            self._sync_prev_timestep_to_ic()   # accumulation(t=0) = 0 (see method docstring)
            self._ic_prev_synced = True

    def _sync_prev_timestep_to_ic(self) -> None:
        """One-time (t=0) sync: copy the slaved IC iterate into the PREVIOUS-time-step store so the
        accumulation term (storage(x) - storage(x_prev))/dt is exactly 0 at t=0.

        The framework sets the time-step store at ``initial_condition``, but ``before_nonlinear_loop``
        then runs ``_slave_eliminated_secondaries``, which overwrites the ITERATE saturations (and the
        surrogate density recomputed from them) with the exact OBL flash -- while the time-step store
        keeps the pre-slave values. The mismatch is a spurious accumulation source that scales as
        1/dt: negligible at large dt, but it dominates and destabilises the static balance at small
        dt (the equilibration then chatters even though the IC's true FLUX residual is ~1e-3)."""
        es = self.equation_system
        subs = self.mdg.subdomains()
        nt = self.time_step_indices.size
        for v in es.variables:                                   # primaries + eliminated secondaries
            vals = es.get_variable_values(variables=[v.name], iterate_index=0)
            for ti in range(nt):
                es.set_variable_values(vals, variables=[v.name], time_step_index=ti)
        for phase in self.fluid.phases:                          # surrogate phase props in accumulation
            for prop in (getattr(phase, "density", None),
                         getattr(phase, "specific_enthalpy", None),
                         getattr(phase, "specific_internal_energy", None)):
                if isinstance(prop, pp.ad.SurrogateFactory):
                    prop.progress_values_in_time(subs, depth=nt)

    _export_idx = 0

    def save_data_time_step(self) -> None:
        # Export on CROSSING the next report time, decoupled from the dt hard-stops, so dt grows
        # adaptively while output stays regular. VTU is written at the first step at/after each
        # report time (i.e. at the actual sim time, slightly past the nominal mark).
        t = self.time_manager.time
        crossed = False
        while (self._export_idx < len(_export_times_pos)
               and t >= _export_times_pos[self._export_idx] - 1.0e-6):
            self._export_idx += 1
            crossed = True
        if crossed or self.time_manager.final_time_reached():
            self.write_pvd_and_vtu()
        self.nonlinear_solver_statistics.save()

    def _absolute_permeability(self, subdomains: list[pp.Grid]) -> np.ndarray:
        # Matrix permeability everywhere; cells whose centroid falls inside a barrier box are cut
        # to matrix * --barrier-factor. Boxes are in metres -> convert to the solver's length units.
        cu = self.units.convert_units
        vals = []
        for sd in subdomains:
            k = np.full(sd.num_cells, self.solid.permeability)
            if _args.barriers and sd.dim == self.mdg.dim_max():
                xc, yc = sd.cell_centers[0], sd.cell_centers[1]
                inside = np.zeros(sd.num_cells, dtype=bool)
                for x0, x1, y0, y1 in _BARRIERS:
                    inside |= ((xc >= cu(x0, "m")) & (xc <= cu(x1, "m"))
                               & (yc >= cu(y0, "m")) & (yc <= cu(y1, "m")))
                k[inside] *= _args.barrier_factor
            vals.append(k)
        return np.concatenate(vals) if vals else np.zeros(0)

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # Per-cell absolute permeability (with barriers) instead of the homogeneous solid scalar,
        # keeping the base HU (isotropic tensor) and HU-mw (mass-mobility-weighted) forms.
        perm = pp.wrap_as_dense_ad_array(
            self._absolute_permeability(subdomains), name="permeability")
        if pp.compositional_flow.is_fractional_flow(self):
            op = self.isotropic_second_order_tensor(
                subdomains, self.total_mass_mobility(subdomains) * perm)
            op.set_name("diffusive_tensor_darcy")
        else:
            op = self.isotropic_second_order_tensor(subdomains, perm)
        return op


model = GeothermalRechargeModel(params)

_TABLE_DIR = os.path.join(HERE, os.pardir, os.pardir, "model_configuration",
                          "constitutive_description", "driesner_vtk_files")


def _attach_samplers(m) -> None:
    phz = VTKSampler(os.path.join(_TABLE_DIR, "brine_graded_xph.vtr"))
    phz.conversion_factors = (1.0, 1.0, 1.0)
    m.obl_sampler = phz
    ptz = VTKSampler(os.path.join(_TABLE_DIR, "brine_graded_xpt.vtr"))
    ptz.conversion_factors = (1.0, 1.0, 1.0)
    ptz.translation_factors = (0.0, -273.15, 0.0)
    m.obl_sampler_ptz = ptz


_attach_samplers(model)

if __name__ == "__main__":
    tb = time.time()
    solver_params = model.default_nonlinear_criteria(tol=1.0e-3)   # looser than the 1e-4 default
    runner = pp.ModelRunner(model, solver_params,
                            nonlinear_solver=geothermal_nonlinear_solver(solver_params))
    print("Elapsed time prepare simulation:", time.time() - tb)
    print("DoF:", model.equation_system.num_dofs(), " grid:", model.mdg)
    model.schur_complement_primary_equations = (
        pp.compositional_flow.get_primary_equations_cf(model))
    model.schur_complement_primary_variables = (
        pp.compositional_flow.get_primary_variables_cf(model))
    # VTU #0 = the initial condition. Use the same export path as the run (write_pvd_and_vtu ->
    # data_to_export), NOT exporter.write_vtu() with no data -- that writes geometry only, so VTU #0
    # had no IC fields. update_derived_quantities first flashes the IC secondaries (e.g. s_halite,
    # which the IC does not set directly) so the exported t=0 state is fully consistent.
    model.update_derived_quantities()
    model.write_pvd_and_vtu()
    tb = time.time()
    runner.run()
    print("Elapsed time run:", time.time() - tb)
