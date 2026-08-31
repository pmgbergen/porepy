"""Weis et al. (2014) Fig. 8 heat-flux plume, condition 2 (9 km x 3 km).

CLI: --scheme {hu, hu-mw}, --consistent (MPFA), --grid-type, --cell-size,
--q-anomaly [W/m^2, default 5], --z-init (initial uniform NaCl overall
composition, default 0; also sets the hydrostatic-column and boundary fluid),
--snap-years (exact snapshot/export schedule, default 0..50000 every 2500),
--dt-nominal/--dt-min/--dt-max (dynamic stepping, default 5/0.001/100 yr).
--lag-buoyancy freezes the buoyancy upwind direction per step (CSMP++ policy).
Output goes to visualization_<tag>/ with tag = case_naming.case_tag(<flags>) --
non-default components only -- so distinct parametrizations never overwrite each
other; fig_weis_2d_plume.py takes the same flags to find the folder and names
its figure fig_8_plume_<tag>.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case_naming import case_tag                     # noqa: E402

import time
import json
import warnings
import tempfile
from pathlib import Path
from typing import cast, Sequence

import numpy as np

import porepy as pp

# geometry description 2D case
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    Figure8Geometry2D as ModelGeometryFigure8,
)

from porepy.examples.geothermal_flow.model_configuration.DriesnerModelConfiguration import (  # noqa: E501
    DriesnerBrineFlowModel,               # HU / PPU (standard primary equations)
    DriesnerBrineFractionalFlowModel,     # HU-mw   (fractional-flow primary equations)
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

# Main directives
case_name = "condition_1"

day_to_second = 86400
year_to_second = 365.0 * day_to_second
to_Mega = 1.0e-6

# Dynamic time stepping (all CLI-overridable).  The schedule pins EXACT landing -- and
# VTU export -- at the Fig. 8 snapshot instants; dt adapts freely in between.
_DEFAULT_SNAP_YEARS = tuple(float(y) for y in range(0, 15001, 250))  # 0..15 kyr / 0.25 kyr
DT_NOMINAL = 20.0           # nominal (initial) step [yr] (--dt-nominal)
DT_MIN = 0.0001               # smallest allowed step [yr] (--dt-min)
DT_MAX = 100.0              # largest allowed step [yr]  (--dt-max)
NL_TOL = 1.0e-4            # Newton convergence tolerance on the row-scaled residual (--tol)
NL_MAX_ITER = 15          # max Newton iterations before a step cut (--max-iter)

# --------------------------------------------------------------------------------------- #
#  Weis et al. (2014) Fig. 8, condition 2 -- boundary & initial conditions.
#  Top (surface): open, Dirichlet p = P_TOP and T = 10 degC.  Bottom: closed to fluid
#  flow (the pressure equation sees no-flux everywhere but the top), Neumann heat
#  INFLUX of 0.05 W/m^2 (background) + Q_ANOMALY over the central 1 km.  Sides: closed
#  and adiabatic.  IC: uniform 10 degC, uniform Z_INIT salt, and a brine-column
#  hydrostatic pressure profile integrated at (Z_INIT, T_TOP).
# --------------------------------------------------------------------------------------- #
P_TOP = 1.0                 # surface pressure [MPa]
                            # well above the 0.5 MPa EOS table floor
T_TOP = 283.15              # surface temperature [K] (10 degC)
Q_BACKGROUND = 0.05         # background crustal heat flux [W/m^2]
Q_ANOMALY = 5.0             # anomaly heat flux [W/m^2] over the inlet (--q-anomaly)
Z_INIT = 0.0                # initial (uniform) NaCl overall composition [-] (--z-init)
DOMAIN_HEIGHT = 3000.0      # [m]
_TRUNCATE_METERS = 2000.0   # --truncated-domain: metres removed from EACH lateral side (far-field)
_VERTICAL_CUT = 1000.0      # --truncated-domain: metres removed from the BOTTOM (deepest, hottest slab)

# --------------------------------------------------------------------------------------- #
#  --md : approved discrete fault & barrier network on the Fig. 8 domain (F1-F6, B1-B3).
#  y = ELEVATION (y=0 base, y=DOMAIN_HEIGHT surface).  Conductive faults F1-F5 are inclined
#  high-k lines (1000 * rock); F6 is a gently-dipping low-angle LINKING connector at reduced
#  grade (--f6-factor, default 100 * rock -- per the geological assessment, a bedding-parallel
#  weak layer, not a fault, so it redistributes fluid without short-circuiting the seals it
#  shares a depth with); sealing barriers B1-B3 are thin curved low-k lines (--barrier-factor *
#  rock) built piecewise from their vertices.  Meshed with gmsh as a simplex -- or, with
#  --recombine, unstructured quadrilaterals -- mixed-dimensional grid conforming to every line.
#  Inclined faults break K-orthogonality, so pair --md with --consistent (MPFA).  Perm / aperture
#  / material logic mirrors porepy_2d_recharge.py: lines are added in the order faults, links,
#  barriers, so a 1D subdomain is classified by its frac_num (fault < link < barrier bands).
# --------------------------------------------------------------------------------------- #
_F8_MD_FRAC_PERM_FACTOR = 1000.0     # conductive fault k (in-plane & normal) = rock * this
_F8_LINK_PERM_FACTOR = 100.0         # F6 low-angle linking connector k = rock * this (--f6-factor)
_F8_BARRIER_FACTOR = 1.0e-3          # sealing-barrier k = rock * this (--barrier-factor)
_F8_BARRIER_THICKNESS = 2.0          # 1D barrier aperture [m] (seal thickness; 1-2 m)
_F8_FAULT_CELL_SIZE_FACTOR = 0.5     # cell size along fracture lines = this * cell_size

# Conductive faults F1-F5, DEEP (low-y) endpoint first: (x0, y0, x1, y1) in metres.
_F8_FAULTS = [
    (4000.0,    0.0, 2300.0, 3000.0),   # F1 master normal fault ~60 E (west graben wall)
    (5000.0,  300.0, 6700.0, 3000.0),   # F2 antithetic fault ~58 W (east graben wall)
    (3320.0, 1200.0, 4000.0, 3000.0),   # F3 W synthetic splay off F1 (intersects F1)
    (4500.0, 1250.0, 5250.0, 2100.0),   # F4 near-vertical central feeder, blind tip (+250 m)
    (5630.0, 1300.0, 5000.0, 3000.0),   # F5 E synthetic splay off F2 (intersects F2)
]

# F6 low-angle (~5 deg) linking connector tying F3, F4, F2 beneath the cap; reduced grade.
_F8_LINK_FAULTS = [
    (3471.0, 1600.0, 5693.0, 1400.0),   # F6 (drop via --no-f6; grade via --f6-factor)
]

# Sealing barriers B1-B3: thin curved seals, each a polyline of (x, y) vertices [m].
_F8_BARRIERS = [
    [(2200.0, 2150.0), (3000.0, 2350.0), (3800.0, 2460.0), (4500.0, 2500.0),
     (5200.0, 2460.0), (6000.0, 2350.0), (6800.0, 2150.0)],                    # B1 clay cap
    [(800.0, 1620.0), (1700.0, 1560.0), (2500.0, 1520.0), (3150.0, 1500.0)],   # B2 west seal
    [(5787.0, 1550.0), (6600.0, 1590.0), (7500.0, 1560.0), (8300.0, 1600.0)],  # B3 east seal
]


def _lines_from(entries, cu) -> list:
    """(x0,y0,x1,y1) tuples -> pp.LineFracture objects in the solver's length units."""
    return [pp.LineFracture(np.array([[cu(x0, "m"), cu(x1, "m")],
                                      [cu(y0, "m"), cu(y1, "m")]]))
            for x0, y0, x1, y1 in entries]


def _f8_fault_fractures(cu) -> list:
    """Faults F1-F5 as pp.LineFracture (full conductive grade)."""
    return _lines_from(_F8_FAULTS, cu)


def _f8_link_fractures(cu) -> list:
    """F6 low-angle linking connector(s) as pp.LineFracture (reduced grade)."""
    return _lines_from(_F8_LINK_FAULTS, cu)


def _f8_barrier_fractures(cu) -> list:
    """Barriers B1-B3 as pp.LineFracture seals: one straight segment per polyline edge, so a
    curved seal is a connected chain of low-k blocking lines."""
    segs = []
    for poly in _F8_BARRIERS:
        for (xa, ya), (xb, yb) in zip(poly[:-1], poly[1:]):
            segs.append(pp.LineFracture(np.array([[cu(xa, "m"), cu(xb, "m")],
                                                  [cu(ya, "m"), cu(yb, "m")]])))
    return segs


def _unique_gmsh_file() -> Path:
    """A per-process gmsh output path. PorePy defaults to a fixed cwd-relative
    'gmsh_frac_file.msh', which several --md runs in one directory would clobber -- a unique
    name (by PID, in the temp dir) makes parallel meshing race-free."""
    return Path(tempfile.gettempdir()) / f"porepy_2d_gmsh_{os.getpid()}.msh"


def _create_mdg_maybe_recombine(mesh_args, network, file_name):
    """Simplex mixed-dimensional grid, or gmsh-recombined quad-dominant cells when
    --recombine is set (via the local _quad_mesh helper -- PorePy's importer reads only
    triangles, so quad support is monkeypatched in for the meshing call only)."""
    if _args.recombine:
        from _quad_mesh import build_recombined_mdg
        return build_recombined_mdg(mesh_args, network, file_name)
    return pp.create_mdg("simplex", mesh_args, network, file_name=file_name)


class BCFigure8(BC):
    """Condition-2 boundary conditions: Dirichlet (p, T) on the TOP only; every other
    face is Neumann -- zero fluid flux, prescribed heat influx along the bottom."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        _, top = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, top, "dir")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        _, top = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, top, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        return np.full(boundary_grid.num_cells, P_TOP)

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        return np.full(boundary_grid.num_cells, T_TOP)

    def bc_values_fourier_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Neumann heat flux: porepy expects the INTEGRATED value per face, i.e. the
        flux density times the face area (a length in 2D, boundary_grid.cell_volumes);
        negative = influx (values follow the outward normal).  Background flux on the
        whole bottom, the anomaly flux on the geometry's inlet faces (the central
        1 km); the lateral sides stay adiabatic (zero)."""
        sides = self.domain_boundary_sides(boundary_grid)
        inlet, _ = self.get_inlet_outlet_sides(boundary_grid)
        q = np.zeros(boundary_grid.num_cells)                       # [MW/m^2]
        q[sides.south] = Q_BACKGROUND * to_Mega
        q[inlet] = Q_ANOMALY * to_Mega
        return -q * boundary_grid.cell_volumes

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        """Uniform background composition Z_INIT on every boundary face (only takes
        effect where fluid actually flows in)."""
        return np.full(boundary_grid.num_cells, Z_INIT)

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary enthalpy from (p, T, Z_INIT) -- the base class hard-codes z = 0."""
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = np.full_like(p, Z_INIT)
        self.obl_sampler_ptz.sample_at(np.array((z_NaCl, t, p)).T)
        return self.obl_sampler_ptz.sampled_could.point_data["H"] * 1.0e-3

    def bc_values_fractional_flow_component(
            self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """FF-template advection factor on the boundary, PER COMPONENT (the reference
        component's weight is summed into the total boundary flux): the entering
        fluid's mass fraction -- Z_INIT for NaCl, the complement for H2O -- so top
        recharge carries mass consistently (the base class's zeros drop it)."""
        is_salt = component == self.fluid.components[1]
        return np.full(bg.num_cells, Z_INIT if is_salt else 1.0 - Z_INIT)


class ICFigure8(IC):
    """Condition-2 initial conditions: uniform 10 degC, uniform Z_INIT salt, and the
    hydrostatic pressure from integrating dp/d(depth) = rho(p, T_TOP, Z_INIT) g with
    the ptz sampler's density."""

    def _hydrostatic_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """(depth, p) of the 10-degC, Z_INIT-brine column, p(0) = P_TOP.  Fixed-point
        on the trapezoid rule (density depends weakly on p, converges in a few sweeps);
        g = pp.GRAVITY_ACCELERATION, the same constant the model's gravity_field uses."""
        if not hasattr(self, "_hydro_profile"):
            depth = np.linspace(0.0, DOMAIN_HEIGHT, 601)
            g = pp.GRAVITY_ACCELERATION
            p = P_TOP + 1000.0 * g * depth * to_Mega
            for _ in range(10):
                pts = np.array((np.full_like(p, Z_INIT), np.full_like(p, T_TOP), p)).T
                self.obl_sampler_ptz.sample_at(pts)
                rho = np.asarray(
                    self.obl_sampler_ptz.sampled_could.point_data["Rho"], dtype=float
                )
                dp = 0.5 * (rho[:-1] + rho[1:]) * g * np.diff(depth) * to_Mega
                p_new = P_TOP + np.concatenate(([0.0], np.cumsum(dp)))
                done = np.max(np.abs(p_new - p)) < 1.0e-12
                p = p_new
                if done:
                    break
            self._hydro_profile = (depth, p)
        return self._hydro_profile

    def _sampled_at_init(self, sd: pp.Grid):
        """Sampler point-data at the initial state (Z_INIT, T, p) of one subdomain --
        the base class hard-codes z = 0 in every ic_values_* sampler call."""
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.full_like(p, Z_INIT)
        self.obl_sampler_ptz.sample_at(np.array((z_NaCl, t, p)).T)
        return self.obl_sampler_ptz.sampled_could.point_data

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        depth_axis, p_axis = self._hydrostatic_profile()
        depth = DOMAIN_HEIGHT - sd.cell_centers[1]
        return np.interp(depth, depth_axis, p_axis)

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return np.full(sd.num_cells, T_TOP)

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        return np.full(sd.num_cells, Z_INIT)

    def ic_values_partial_fractions(self, sd: pp.Grid) -> np.ndarray:
        data = self._sampled_at_init(sd)
        return np.clip(data["Xl"], 0, 1.0), np.clip(data["Xv"], 0, 1.0)

    def ic_values_gas_saturation(self, sd: pp.Grid) -> np.ndarray:
        return np.clip(self._sampled_at_init(sd)["S_v"], 0, 1.0)

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        return self._sampled_at_init(sd)["H"] * 1.0e-3

# Configuration dictionary mapping cases to their specific classes
simulation_cases = {
    "condition_1": {
        "bc": BCFigure8,
        "ic": ICFigure8,
        "geometry": ModelGeometryFigure8,
    }
}

BoundaryConditions: type = cast(type, simulation_cases[case_name]["bc"])
InitialConditions: type = cast(type, simulation_cases[case_name]["ic"])
ModelGeometry: type = cast(type, simulation_cases[case_name]["geometry"])

solid_constants = pp.SolidConstants(
    permeability=1e-15,
    porosity=0.1,
    thermal_conductivity=2.0 * to_Mega,
    density=2700.0,
    specific_heat_capacity=880.0 * to_Mega,
)
material_constants = {"solid": solid_constants}
# Scheme switch (= porepy_3d_solver._SCHEME_CONFIG): the fractional_flow flag pairs with
# the base template -- False -> DriesnerBrineFlowModel, True -> the fractional-flow one.
_SCHEME_CONFIG = {
    "hu":    dict(fractional_flow=False, buoyancy_upwinding="hybrid"),
    "hu-mw": dict(fractional_flow=True,  buoyancy_upwinding="hybrid"),
}
import argparse
_ap = argparse.ArgumentParser(
    description="Weis et al. (2014) Fig. 8(A-C) heat-flux plume (9 km x 3 km, "
                "heat-flux anomaly over the central 1 km of the bottom boundary; "
                "output folder visualization_<tag> per case_naming.case_tag).")
_ap.add_argument("--consistent", action="store_true",
                 help="consistent flux discretization (MPFA); default TPFA")
_ap.add_argument("--grid-type", default=None, choices=["cartesian", "simplex"],
                 help="mesh type; default: the geometry class's choice")
_ap.add_argument("--cell-size", type=float, default=None, metavar="M",
                 help="target cell size [m]; default: the geometry class's value")
_ap.add_argument("--scheme", default="hu", choices=list(_SCHEME_CONFIG),
                 help="HU (standard template, hybrid), HU-mw (fractional-flow template), "
                      "PPU (standard template, phase-potential); default HU")
_ap.add_argument("--q-anomaly", type=float, default=Q_ANOMALY, metavar="W/M2",
                 help=f"anomaly heat flux over the inlet [W/m^2]; default {Q_ANOMALY}")
_ap.add_argument("--z-init", type=float, default=Z_INIT, metavar="Z",
                 help="initial (uniform) NaCl overall composition [-]; default "
                      f"{Z_INIT} (graded table spans the full range 0..1)")
_ap.add_argument("--snap-years", type=float, nargs="+",
                 default=list(_DEFAULT_SNAP_YEARS), metavar="YR",
                 help="schedule of exact snapshot/export instants [years]; the last one "
                      f"is the final time; default {_DEFAULT_SNAP_YEARS}")
_ap.add_argument("--dt-nominal", type=float, default=DT_NOMINAL, metavar="YR",
                 help=f"nominal (initial) time step [years]; default {DT_NOMINAL}")
_ap.add_argument("--dt-min", type=float, default=DT_MIN, metavar="YR",
                 help=f"smallest allowed time step [years]; default {DT_MIN}")
_ap.add_argument("--dt-max", type=float, default=DT_MAX, metavar="YR",
                 help=f"largest allowed time step [years]; default {DT_MAX}")
_ap.add_argument("--dt-constant", type=float, default=None, metavar="YR",
                 help="fixed time step [years]: disables adaptation and retries (no "
                      "dt-cutting), so a stalled step fails outright; must divide each snap year")
_ap.add_argument("--lag-buoyancy", action="store_true",
                 help="freeze the buoyancy upwind direction over each time step "
                      "(CSMP++'s frozen-upwind policy, Weis et al. sec. 2.7)")
_ap.add_argument("--md", action="store_true", default=False,
                 help="discrete fault & barrier network (F1-F6, B1-B3) on a gmsh "
                      "mixed-dimensional mesh; pair with --consistent (MPFA)")
_ap.add_argument("--truncated-domain", action="store_true", default=False,
                 help=f"remove the quiescent far-field: drop {_TRUNCATE_METERS:g} m from the left AND right "
                      f"(9 km -> {(9000.0 - 2 * _TRUNCATE_METERS) / 1000:g} km) and {_VERTICAL_CUT:g} m from the "
                      f"BOTTOM (3 km -> {(DOMAIN_HEIGHT - _VERTICAL_CUT) / 1000:g} km deep), to cut cell count "
                      f"and lower the base pressure. Coordinates stay ABSOLUTE (plume x=4500, surface at "
                      f"3 km); --md CLIPS the fault/barrier network to the box, fixed-dim shifts the mesh")
_ap.add_argument("--recombine", action="store_true", default=False,
                 help="build the --md mesh with unstructured QUADRILATERALS "
                      "(gmsh recombination) instead of triangles")
_ap.add_argument("--no-barriers", dest="barriers", action="store_false", default=True,
                 help="drop the sealing barriers B1-B3 (faults only) under --md")
_ap.add_argument("--barrier-factor", type=float, default=_F8_BARRIER_FACTOR, metavar="F",
                 help=f"sealing-barrier permeability = rock * F; default {_F8_BARRIER_FACTOR:g}")
_ap.add_argument("--no-f6", dest="f6", action="store_false", default=True,
                 help="drop the F6 low-angle linking connector (A/B baseline)")
_ap.add_argument("--f6-factor", type=float, default=_F8_LINK_PERM_FACTOR, metavar="F",
                 help=f"F6 linking-connector permeability = rock * F; default "
                      f"{_F8_LINK_PERM_FACTOR:g} (faults use {_F8_MD_FRAC_PERM_FACTOR:g})")
_ap.add_argument("--diagnose-binding", action="store_true", default=False,
                 help="on each time-step cut, run the STEP-0 binding diagnostic on the stalled iterate: "
                      "identify the failing cells and test kink vs negative compressibility (sign of "
                      "dRho/dp) + line-search merit alignment. Logs only, no behaviour change")
_ap.add_argument("--tol", type=float, default=NL_TOL, metavar="T",
                 help=f"Newton convergence tolerance on the row-scaled residual (default {NL_TOL:g})")
_ap.add_argument("--max-iter", type=int, default=NL_MAX_ITER, metavar="N",
                 help=f"max Newton iterations before a step is cut (default {NL_MAX_ITER})")
_args = _ap.parse_args()
if not 0.0 <= _args.z_init <= 1.0:
    raise SystemExit(f"--z-init {_args.z_init} outside the graded table "
                     "range z in [0, 1]")
if _args.snap_years[0] != 0.0 or any(
        b <= a for a, b in zip(_args.snap_years, _args.snap_years[1:])):
    raise SystemExit(f"--snap-years {_args.snap_years} must start at 0 and be "
                     "strictly increasing")
if not 0.0 < _args.dt_min <= _args.dt_nominal <= _args.dt_max:
    raise SystemExit("time steps must satisfy 0 < --dt-min <= --dt-nominal <= --dt-max")
if _args.recombine and not _args.md:
    raise SystemExit("--recombine only applies together with --md")
if _args.md and not _args.consistent:
    print("NOTE: --md places inclined faults on an unstructured mesh; TPFA gravity is "
          "inconsistent there -- consider adding --consistent (MPFA).")
Q_ANOMALY = _args.q_anomaly
Z_INIT = _args.z_init

# Dynamic time stepping: dt grows/shrinks with Newton effort inside (3, 8) iterations,
# recomputes at 0.3x on failure, and the schedule forces exact landing on every
# snapshot instant, which is also exactly where VTUs are exported.  With --dt-constant
# the step is frozen instead (no adaptation, no retries): a stall fails the run rather
# than being masked by dt-cutting.
schedule = [y * year_to_second for y in _args.snap_years]
tf = schedule[-1]
if _args.dt_constant is not None:
    if _args.dt_constant <= 0.0:
        raise SystemExit("--dt-constant must be positive")
    _bad = [y for y in _args.snap_years
            if abs(round(y / _args.dt_constant) * _args.dt_constant - y) > 1e-9 * max(1.0, y)]
    if _bad:
        raise SystemExit(f"--dt-constant {_args.dt_constant:g} must divide each snap year "
                         f"evenly; offenders: {_bad} (e.g. an interval of 500 admits "
                         "dt in {500, 250, 100, 50, 25, 10, 5, ...})")
    dtc = _args.dt_constant * year_to_second
    time_manager = pp.TimeManager(
        schedule=schedule,
        dt_init=dtc,
        dt_min_max=(dtc, dtc),
        constant_dt=True,
        print_info=True,
    )
else:
    time_manager = pp.TimeManager(
        schedule=schedule,
        dt_init=_args.dt_nominal * year_to_second,
        dt_min_max=(_args.dt_min * year_to_second, _args.dt_max * year_to_second),
        constant_dt=False,
        iter_max=15,
        iter_optimal_range=(3, 8),
        iter_relax_factors=(0.25, 1.5),
        recomp_factor=0.3,
        print_info=True,
    )
times_to_export = list(schedule)

params = {
    "folder_name": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "visualization_" + case_tag(_args.scheme, _args.consistent,
                                    _args.grid_type, _args.cell_size,
                                    _args.q_anomaly, _args.z_init,
                                    # pass None at the SOLVER's defaults so the tag omits them
                                    # (non-default-only), independent of case_naming's own defaults.
                                    _args.dt_nominal if _args.dt_nominal != DT_NOMINAL else None,
                                    _args.dt_min if _args.dt_min != DT_MIN else None,
                                    _args.dt_max if _args.dt_max != DT_MAX else None,
                                    (_args.snap_years[-1]
                                     if _args.snap_years[-1] != _DEFAULT_SNAP_YEARS[-1] else None),
                                    lag=_args.lag_buoyancy,
                                    md=_args.md, recombine=_args.recombine,
                                    truncated_domain=_args.truncated_domain,
                                    dt_constant=_args.dt_constant)),
    "enable_buoyancy_effects": True,
    "material_constants": material_constants,
    "time_manager": time_manager,
    "times_to_export": times_to_export,
    # Schur-reduced CPR linear solver -- exactly porepy_3d_solver's "cpr" mode.
    "use_petsc": True,
    "petsc_preconditioner": "cpr",
    "cpr_rtol": 1.0e-5,           # CPR GMRES relative tolerance
    "cpr_maxit": 400,             # CPR GMRES iteration cap
    "cpr_accuracy_tol": 1.0e-3,   # post-solve gate -> direct fallback above this
    "step_control_method": "LS",   # weis backtracking line search (== subsection_4_2 1D/3D solvers)
    "residual_scale_current_dt": True,  # weis: convergence bar tracks the CURRENT dt (not dt_init), so
                                        # steps cut at a stiff front loosen the bar and still converge
    # Slave the eliminated secondaries (T, s_gas/halite, x_NaCl_liq/gas/halite) to their exact
    # OBL value f(p,h,z) each Newton iterate -- Weis-style explicit flash. Removes the lagged
    # elimination residual (e.g. the wide-open liquid NaCl fraction) that limit-cycles at phase
    # fronts; same fix that resolved the 1D fig-6 salt stall.
    "slave_eliminated_secondaries": True,
}
params["consistent_discretization"] = _args.consistent
params["lag_buoyancy_direction"] = _args.lag_buoyancy
params["diagnose_binding"] = _args.diagnose_binding
if _args.grid_type is not None:
    params["grid_type"] = _args.grid_type            # Figure8Geometry2D reads this key
params.update(_SCHEME_CONFIG[_args.scheme])
FlowModel = (DriesnerBrineFractionalFlowModel if params["fractional_flow"]
             else DriesnerBrineFlowModel)


class GeothermalBrineFlowModel(
    DriesnerPhaseExport, ModelGeometry, BoundaryConditions, InitialConditions, FlowModel
):
    # flux discretization comes from the base TPFA/MPFA switch (--consistent)

    def meshing_arguments(self) -> dict:
        mesh_args = super().meshing_arguments()
        if _args.cell_size is not None:              # default: the geometry class's value
            mesh_args = {**mesh_args,
                         "cell_size": self.units.convert_units(_args.cell_size, "m")}
        return mesh_args

    def set_domain(self) -> None:
        """Full 9 km x 3 km Fig. 8 box by default; with --truncated-domain drop _TRUNCATE_METERS from
        EACH lateral side (quiescent far-field) AND _VERTICAL_CUT from the BOTTOM (deepest, hottest slab)
        to cut cell count -- the box becomes the ABSOLUTE [TRUNC, 9km-TRUNC] x [CUT, 3km] in BOTH modes,
        so coordinates (plume x=4500, surface at 3 km, fault positions) are preserved.

        --md: gmsh honours nonzero xmin/ymin, so the box is set absolute directly and the fault/barrier
        network is CLIPPED to it. Fixed-dim: the cartesian mesher IGNORES a nonzero min, so we mesh a box
        of the right SIZE at the origin here and shift the nodes to absolute in set_geometry (the removed
        bottom is the deepest slab, so the base pressure drops regardless of the mode)."""
        if not _args.truncated_domain:
            return super().set_domain()
        cu = self.units.convert_units
        self._inlet_centre = np.array([4500.0, _VERTICAL_CUT, 0.0])       # base is now at y=_VERTICAL_CUT
        self._outlet_centre = np.array([4500.0, DOMAIN_HEIGHT, 0.0])      # surface stays at 3 km
        if _args.md:                                          # absolute box directly; gmsh clips the network
            self._domain = pp.Domain({"xmin": cu(_TRUNCATE_METERS, "m"),
                                      "xmax": cu(9000.0 - _TRUNCATE_METERS, "m"),
                                      "ymin": cu(_VERTICAL_CUT, "m"), "ymax": cu(DOMAIN_HEIGHT, "m")})
        else:                                                 # SIZE box at the origin; shifted in set_geometry
            self._domain = pp.Domain({"xmax": cu(9000.0 - 2.0 * _TRUNCATE_METERS, "m"),
                                      "ymax": cu(DOMAIN_HEIGHT - _VERTICAL_CUT, "m")})

    # -- --md geometry: discrete fault & barrier network via gmsh -------------------------
    def set_geometry(self) -> None:
        """Fixed-dimensional Fig. 8 box by default; --md builds the F1-F6 / B1-B3 fault &
        barrier network as a gmsh mixed-dimensional simplex (or quad, --recombine) grid."""
        if _args.md:
            return self._set_geometry_md()
        super().set_geometry()
        if _args.truncated_domain:
            self._shift_grid_to_absolute()

    def _shift_grid_to_absolute(self) -> None:
        """Cartesian meshing builds at the origin (it ignores a nonzero xmin/ymin), so after meshing the
        truncated SIZE box we shift the node coordinates by (_TRUNCATE_METERS, _VERTICAL_CUT) to restore
        the ABSOLUTE Fig-8 positions -- plume at x=4500, surface at 3 km -- matching the --md coordinate
        system. Boundary grids follow the shift once the subdomain and mdg geometry are recomputed."""
        cu = self.units.convert_units
        shift = np.array([[cu(_TRUNCATE_METERS, "m")], [cu(_VERTICAL_CUT, "m")], [0.0]])
        for sd in self.mdg.subdomains():
            sd.nodes = sd.nodes + shift
            sd.compute_geometry()
        self.mdg.compute_geometry()
        self._domain = pp.Domain({"xmin": cu(_TRUNCATE_METERS, "m"),
                                  "xmax": cu(9000.0 - _TRUNCATE_METERS, "m"),
                                  "ymin": cu(_VERTICAL_CUT, "m"), "ymax": cu(DOMAIN_HEIGHT, "m")})

    def _set_geometry_md(self) -> None:
        self.set_domain()
        cu = self.units.convert_units
        faults = _f8_fault_fractures(cu)
        links = _f8_link_fractures(cu) if _args.f6 else []
        barriers = _f8_barrier_fractures(cu) if _args.barriers else []
        # Order (faults, links, barriers) fixes the frac_num classification bands.
        self._n_fault_lines = len(faults)
        self._n_link_lines = len(links)
        network = pp.create_fracture_network(faults + links + barriers, self._domain)
        h = cu(100.0 if _args.cell_size is None else _args.cell_size, "m")
        h_frac = _F8_FAULT_CELL_SIZE_FACTOR * h
        mesh_args = {"cell_size": h, "cell_size_boundary": h,
                     "cell_size_fracture": h_frac, "cell_size_min": h_frac}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*outside the domain boundary.*")
            self.mdg = _create_mdg_maybe_recombine(mesh_args, network, _unique_gmsh_file())
        self.nd = self.mdg.dim_max()
        pp.set_local_coordinate_projections(self.mdg)

    # -- fracture material / permeability / aperture (mirrors porepy_2d_recharge.py) ------
    def _fracture_perm_factor(self, sd: pp.Grid) -> float:
        """Rock-permeability multiplier for a lower-dim subdomain, by frac_num band:
        fault (1000x) < F6 link (--f6-factor) < barrier seal (--barrier-factor). 0D
        intersections are conductive (fault grade), as in the recharge/3D solvers."""
        if sd.dim != self.mdg.dim_max() - 1:
            return _F8_MD_FRAC_PERM_FACTOR
        fn = int(sd.frac_num)
        nf = getattr(self, "_n_fault_lines", 0)
        nl = getattr(self, "_n_link_lines", 0)
        if fn < nf:
            return _F8_MD_FRAC_PERM_FACTOR
        if fn < nf + nl:
            return _args.f6_factor
        return _args.barrier_factor

    def _is_barrier_subdomain(self, sd: pp.Grid) -> bool:
        """True for a 1D sealing barrier (added after faults+links -> frac_num in the top band)."""
        if sd.dim != self.mdg.dim_max() - 1 or sd.num_cells == 0:
            return False
        return int(sd.frac_num) >= (getattr(self, "_n_fault_lines", 0)
                                    + getattr(self, "_n_link_lines", 0))

    def grid_aperture(self, grid: pp.Grid) -> np.ndarray:
        if _args.md and self._is_barrier_subdomain(grid):
            return np.full(grid.num_cells,
                           self.units.convert_units(_F8_BARRIER_THICKNESS, "m"))
        return super().grid_aperture(grid)

    def _absolute_permeability(self, subdomains: list[pp.Grid]) -> np.ndarray:
        vals = []
        for sd in subdomains:
            k = np.full(sd.num_cells, self.solid.permeability)
            if sd.dim < self.mdg.dim_max():           # fault 1000x, F6 --f6-factor, barrier --barrier-factor
                k *= self._fracture_perm_factor(sd)
            vals.append(k)
        return np.concatenate(vals) if vals else np.zeros(0)

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        if not _args.md:
            return super().permeability(subdomains)
        perm = pp.wrap_as_dense_ad_array(
            self._absolute_permeability(subdomains), name="permeability")
        if pp.compositional_flow.is_fractional_flow(self):
            op = self.isotropic_second_order_tensor(
                subdomains, self.total_mass_mobility(subdomains) * perm)
            op.set_name("diffusive_tensor_darcy")
        else:
            op = self.isotropic_second_order_tensor(subdomains, perm)
        return op

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        # Rock-only (fault_factor * rock) normal k, NOT the base mass-mobility weighting, which
        # double-counts mobility on conductive fracture interfaces and blows Newton up (the known
        # MD double-mobility bug; same fix as the recharge / 3D --md solvers).
        if not _args.md:
            return super().normal_permeability(interfaces)
        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        kn_sd = pp.wrap_as_dense_ad_array(
            np.concatenate(
                [np.full(sd.num_cells,
                         self._fracture_perm_factor(sd) * self.solid.permeability)
                 for sd in subdomains]
            ) if subdomains else np.zeros(0),
            name="normal_k",
        )
        kn = projection.secondary_to_mortar_avg() @ kn_sd
        kn.set_name("normal_permeability")
        return kn

    def _material_id(self, sd: pp.Grid) -> np.ndarray:
        """Per-cell tag (cached): 0 rock, 1 barrier seal, 2 + frac_num each conductive
        fault/link (F6 is the frac_num just past F1-F5); 0D intersections inherit the
        highest-material neighbour."""
        cache = self.__dict__.setdefault("_material_id_cache", {})
        if id(sd) in cache:
            return cache[id(sd)]
        n, dmax = sd.num_cells, self.mdg.dim_max()
        if sd.dim == dmax:
            m = np.zeros(n)                                   # rock
        elif self._is_barrier_subdomain(sd):
            m = np.ones(n)                                    # barrier seal
        elif sd.dim == dmax - 1:
            m = np.full(n, 2.0 + int(sd.frac_num))            # conductive fault / F6 link
        else:                                                 # 0D intersection: inherit a neighbour
            nb = []
            for intf in self.mdg.subdomain_to_interfaces(sd):
                a, b = self.mdg.interface_to_subdomain_pair(intf)
                other = a if b is sd else b
                if other is not sd and other.dim > sd.dim:
                    nb.append(float(self._material_id(other)[0]))
            m = np.full(n, max(nb) if nb else 0.0)
        cache[id(sd)] = m
        return m

    def data_to_export(self):
        data = super().data_to_export()
        if _args.md:
            for sd in self.mdg.subdomains():
                data.append((sd, "material", self._material_id(sd)))
        return data


# Instance of the computational model
model = GeothermalBrineFlowModel(params)

HERE = os.path.dirname(os.path.abspath(__file__))
# Constitutive approach shared by every subsection_4_2 solver: Driesner graded OBL tables sampled
# with the unified VTKSampler tensor backend (multilinear value + analytic gradient of that same
# interpolant -> consistent Jacobian; identical to the weis_1d_solver construction).
TABLE_LEVEL = "graded"                    # default OBL: the C0 graded brine tables
_TABLE_DIR = os.path.join(
    HERE, os.pardir, os.pardir, "model_configuration", "constitutive_description",
    "driesner_vtk_files")


def _attach_samplers(model) -> None:
    """Attach the C0 graded Driesner OBL samplers (phz + ptz), exactly as
    porepy_1d_solver / porepy_3d_solver do."""
    Sampler = VTKSampler
    phz = Sampler(os.path.join(_TABLE_DIR, "brine_graded_xph.vtr"))
    phz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, h, p)
    model.obl_sampler = phz
    ptz = Sampler(os.path.join(_TABLE_DIR, "brine_graded_xpt.vtr"))
    ptz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, t, p)
    ptz.translation_factors = (0.0, -273.15, 0.0)            # T in degC -> K in the sampler
    model.obl_sampler_ptz = ptz


_attach_samplers(model)


tb = time.time()
# Shared base stopping criterion (== subsection_4_2 1D/3D solvers): relative-storage Lebesgue
# metric, tol 1e-4, max_iter 20. Was an inline dict at max_iter 13.
solver_params = model.default_nonlinear_criteria(tol=_args.tol, max_iterations=_args.max_iter)
runner = pp.ModelRunner(model, solver_params,
                        nonlinear_solver=geothermal_nonlinear_solver(solver_params))
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)
model.schur_complement_primary_equations = (
    pp.compositional_flow.get_primary_equations_cf(model)
)
model.schur_complement_primary_variables = (
    pp.compositional_flow.get_primary_variables_cf(model)
)

# print geometry
model.exporter.write_vtu()
tb = time.time()
runner.run()
te = time.time()
# Completion marker, written only when the time loop actually reached the FINAL TIME (not
# tied to the trailing flux prints). run_scenarios.py treats this file as the authoritative
# success/cache signal, so a run that finished but then exits via a teardown signal (e.g.
# SIGPIPE) still counts as complete, and a completed run is never recomputed.
_reached = model.time_manager.final_time_reached()
_final_years = model.time_manager.time / year_to_second
if _reached:
    print(f"SIMULATION COMPLETE: reached final time {_final_years:g} years in {te - tb:.1f} s")
    try:
        with open(os.path.join(params["folder_name"], "run_complete.json"), "w") as _mf:
            json.dump({"completed": True, "final_time_years": _final_years,
                       "elapsed_run_seconds": te - tb,
                       "num_dofs": int(model.equation_system.num_dofs()),
                       "argv": sys.argv[1:]}, _mf, indent=2)
    except OSError as _e:
        print("WARNING: could not write completion marker:", _e)
else:
    print(f"SIMULATION INCOMPLETE: stopped at {_final_years:g} years (final time not reached)")
print("Elapsed time run_time_dependent_model: ", te - tb)
print("Total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid information: ", model.mdg)

# Retrieve the grid and boundary information
grid = model.mdg.subdomains()[0]
bc_sides = model.domain_boundary_sides(grid)

# Integrated overall mass flux on all facets
mn = model.equation_system.evaluate(model.darcy_flux(model.mdg.subdomains()))
mn = cast(np.ndarray, mn)

inlet_idx, outlet_idx = model.get_inlet_outlet_sides(model.mdg.subdomains()[0])
print("Inflow values : ", mn[inlet_idx])
print("Outflow values : ", mn[outlet_idx])
