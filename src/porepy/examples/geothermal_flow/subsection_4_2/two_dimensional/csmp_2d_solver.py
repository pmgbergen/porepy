"""Weis et al. (2014) Fig. 8 heat-flux plume, condition 2 (9 km x 3 km) -- openCSMP run.

openCSMP counterpart of porepy_2d_solver.py: same geometry, rock, boundary/initial
conditions, fault & barrier network, snapshot schedule and command line.  The mesh is
built here with gmsh and handed, together with an openCSMP configuration and a run file,
to the compiled openCSMP driver ``csmp_2d_plume`` (openCSMP/hydrothermal_system_modelling/
drivers/csmp_2d_plume.cpp), which runs the CVFEM pressure-enthalpy-salinity scheme of
Weis et al. (2014).

Setup (identical to porepy_2d_solver.py):
  domain 9 km x 3 km, y = elevation (0 base, 3000 m surface); rock k = 1e-15 m^2,
  phi = 0.1, K = 2 W/m/K, rho_r = 2700 kg/m^3, c_pr = 880 J/kg/K, rock compressibility 0.
  Top: Dirichlet p = P_TOP = 1 MPa, T = 10 degC.  Bottom: no fluid flux, heat influx
  Q_BACKGROUND = 0.05 W/m^2 plus --q-anomaly over the central 1 km (x = 4000..5000 m).
  Sides: closed and adiabatic.  IC: 10 degC, NaCl mass fraction --z-init, hydrostatic p.
  --md adds faults F1-F6 as conforming lower-dimensional line elements (element thickness =
  aperture, k = factor * rock; junctions share nodes) and barriers B1-B3 as explicitly meshed
  2 m wide low-k zones following the barrier polylines, so they genuinely seal. (openCSMP's
  split boundaries, the closer analogue of PorePy's mortar coupling, cannot yet split this
  network: crossings and multiple T-junctions crash CreateSplitBoundaryFrom.)

Differences that follow from the numerical method (openCSMP is not a Newton solver):
  * CVFEM: each step is one semi-implicit pressure solve, explicit transport and one
    implicit heat-conduction solve.  There is no Newton loop, line search, CPR, TPFA/MPFA
    choice or HU/HU-mw upwinding, so --scheme, --consistent, --tol, --max-iter,
    --ls-min-alpha, --lag-buoyancy (CVFEM freezes the upwind direction within a step
    anyway), --diagnose-binding and --dump-metric are accepted but ignored, with a warning.
  * The step is limited by CFL (--cfl-scaling, 0.1 in Weis et al. 2014) and mass criteria
    in addition to --dt-max; the schedule still lands exactly on every --snap-years instant.
  * cpr_rtol = 1e-5 and cpr_maxit = 400 set the tolerance and iteration cap of the
    pressure/temperature Krylov solves; cpr_accuracy_tol has no counterpart.
  * --grid-type cartesian gives structured quadrilaterals, simplex Delaunay triangles, and
    --recombine unstructured quadrilaterals (gmsh; with or without --md). All meshes have
    nodes at the anomaly edges x = 4000 and 5000 m.

Output: visualization_csmp_<tag>/ next to this file, tag = case_naming.case_tag(<flags>)
(the PorePy run of the same flags writes visualization_<tag>/).  VTU snapshots
csmp_plume_Model_(REGION)_<index>.vtu with csmp_plume_snapshot_years.txt mapping index to
years; run_complete.json is written when the final time is reached.

openCSMP-only options: --csmp-exe (driver path; default $CSMP_2D_PLUME or the sibling
openCSMP/build checkout), --tables-dir (cache of openCSMP's H2O-NaCl lookup tables, built
once in about an hour when missing), --cfl-scaling, --check-mesh (build and export the
initial state only).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case_naming import case_tag                     # noqa: E402

import argparse
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# -- identical to porepy_2d_solver.py ------------------------------------------------------- #
_DEFAULT_SNAP_YEARS = tuple(float(y) for y in range(0, 25001, 50))  # 0..25 kyr / 0.05 kyr
DT_NOMINAL = 20.0           # nominal (initial) step [yr] (--dt-nominal)
DT_MIN = 0.0001             # smallest allowed step [yr] (--dt-min)
DT_MAX = 100.0              # largest allowed step [yr]  (--dt-max)
NL_TOL = 1.0e-4             # PorePy Newton tolerance (--tol; no counterpart here)
NL_MAX_ITER = 15            # PorePy Newton iteration cap (--max-iter; no counterpart here)

P_TOP = 1.0                 # surface pressure [MPa]
T_TOP = 283.15              # surface temperature [K] (10 degC)
Q_BACKGROUND = 0.05         # background crustal heat flux [W/m^2]
Q_ANOMALY = 5.0             # anomaly heat flux [W/m^2] over the inlet (--q-anomaly)
Z_INIT = 0.0                # initial (uniform) NaCl overall composition [-] (--z-init)
DOMAIN_WIDTH = 9000.0       # [m]
DOMAIN_HEIGHT = 3000.0      # [m]
INLET_X = (4000.0, 5000.0)  # central 1 km of the bottom boundary [m]
DEFAULT_CELL_SIZE = 100.0   # Figure8Geometry2D.meshing_arguments / the --md default [m]
_TRUNCATE_METERS = 2000.0   # --truncated-domain: metres removed from EACH lateral side
_VERTICAL_CUT = 1000.0      # --truncated-domain: metres removed from the BOTTOM

ROCK = dict(permeability=1.0e-15, porosity=0.1, thermal_conductivity=2.0,
            density=2700.0, specific_heat_capacity=880.0, compressibility=0.0)

_FIG8_MD_FRAC_PERM_FACTOR = 1.0e3      # conductive fault k = rock * this (1 D)
_FIG8_LINK_PERM_FACTOR = 250.0         # F6 linking connector k = rock * this (--f6-factor)
_FIG8_BARRIER_FACTOR = 1.0e-3          # sealing-barrier k = rock * this (--barrier-factor)
_FIG8_BARRIER_THICKNESS = 2.0          # barrier aperture [m]
_FIG8_FAULT_APERTURE = 5.0             # fault aperture [m]
_FIG8_LINK_APERTURE = 2.0              # F6 aperture [m]
_FIG8_FAULT_CELL_SIZE_FACTOR = 0.5     # PorePy: cell size along fracture lines = this * cell_size
# openCSMP mesh balance: faults are refined a bit more than in PorePy, while the explicitly meshed
# 2 m barrier strips keep coarse elements along their length (elongated cells across the strip)
_CSMP_FAULT_CELL_SIZE_FACTOR = 0.35    # element size along the fault lines = this * cell_size
_CSMP_BARRIER_CELL_SIZE_FACTOR = 0.5   # element size along the barrier strips = this * cell_size

# Conductive faults F1-F5, DEEP (low-y) endpoint first: (x0, y0, x1, y1) in metres.
_FIG8_FAULTS = [
    (4000.0,    0.0, 2300.0, 3000.0),   # F1 master normal fault ~60 E (west graben wall)
    (5000.0,  300.0, 6700.0, 3000.0),   # F2 antithetic fault ~58 W (east graben wall)
    (3320.0, 1200.0, 4000.0, 3000.0),   # F3 W synthetic splay off F1 (intersects F1)
    (4500.0, 1050.0, 5050.0, 2250.0),   # F4 near-vertical central feeder, blind tip (+250 m)
    (5630.0, 1300.0, 5000.0, 3000.0),   # F5 E synthetic splay off F2 (intersects F2)
]
_FIG8_LINK_FAULTS = [
    (3471.0, 1600.0, 5693.0, 1400.0),   # F6 low-angle linking connector
]
_FIG8_BARRIERS = [
    [(2200.0, 2150.0), (3000.0, 2350.0), (3800.0, 2460.0), (4500.0, 2500.0),
     (5200.0, 2460.0), (6000.0, 2350.0), (6800.0, 2150.0)],                    # B1 clay cap
    [(800.0, 1620.0), (1700.0, 1560.0), (2500.0, 1520.0), (3150.0, 1500.0)],   # B2 west seal
    [(5787.0, 1550.0), (6600.0, 1590.0), (7500.0, 1560.0), (8300.0, 1600.0)],  # B3 east seal
]
# Endpoints closer than this to another line are snapped onto it: the digitised network
# has T-junctions that miss by 0.04-0.4 m (F5, F6, B3), which would leave sub-metre slivers.
_SNAP_TOLERANCE = 1.0      # [m]

# -- command line (porepy_2d_solver.py's, plus openCSMP-only options) ----------------------- #
_POREPY_ONLY = {  # dest -> PorePy default; a non-default value gets a warning
    "scheme": "hu", "consistent": False, "tol": NL_TOL, "max_iter": NL_MAX_ITER,
    "ls_min_alpha": 0.1, "lag_buoyancy": False, "diagnose_binding": False, "dump_metric": False,
}

_ap = argparse.ArgumentParser(
    description="Weis et al. (2014) Fig. 8(A-C) heat-flux plume run with openCSMP "
                "(same flags as porepy_2d_solver.py; output folder visualization_csmp_<tag>).")
_ap.add_argument("--consistent", action="store_true",
                 help="PorePy MPFA switch; ignored (CVFEM has no TPFA/MPFA choice)")
_ap.add_argument("--grid-type", default=None, choices=["cartesian", "simplex"],
                 help="cartesian = structured quadrilaterals (default), simplex = triangles; "
                      "--md always meshes with gmsh (triangles, quads with --recombine)")
_ap.add_argument("--cell-size", type=float, default=None, metavar="M",
                 help=f"target cell size [m]; default {DEFAULT_CELL_SIZE:g}")
_ap.add_argument("--scheme", default="hu", choices=["hu", "hu-mw"],
                 help="PorePy upwinding/template; ignored (CVFEM phase-potential upwinding)")
_ap.add_argument("--q-anomaly", type=float, default=Q_ANOMALY, metavar="W/M2",
                 help=f"anomaly heat flux over the inlet [W/m^2]; default {Q_ANOMALY}")
_ap.add_argument("--z-init", type=float, default=Z_INIT, metavar="Z",
                 help=f"initial (uniform) NaCl mass fraction [-]; default {Z_INIT}")
_ap.add_argument("--snap-years", type=float, nargs="+", default=list(_DEFAULT_SNAP_YEARS),
                 metavar="YR", help="exact snapshot/export instants [years]; the last one is "
                                    "the final time; default 0..25000 every 50")
_ap.add_argument("--dt-nominal", type=float, default=DT_NOMINAL, metavar="YR",
                 help=f"initial time step [years]; default {DT_NOMINAL}")
_ap.add_argument("--dt-min", type=float, default=DT_MIN, metavar="YR",
                 help=f"smallest allowed time step [years]; default {DT_MIN}")
_ap.add_argument("--dt-max", type=float, default=DT_MAX, metavar="YR",
                 help=f"largest allowed time step [years]; default {DT_MAX}")
_ap.add_argument("--dt-constant", type=float, default=None, metavar="YR",
                 help="fixed time step [years]; a step the scheme has to cut (CFL/mass) fails "
                      "the run; must divide each snap year")
_ap.add_argument("--lag-buoyancy", action="store_true",
                 help="PorePy option; ignored (CVFEM already freezes the upwind direction per step)")
_ap.add_argument("--md", action="store_true", default=False,
                 help="discrete fault & barrier network: F1-F6 as lower-dimensional line "
                      "elements, B1-B3 as meshed 2 m low-k zones")
_ap.add_argument("--truncated-domain", action="store_true", default=False,
                 help=f"drop {_TRUNCATE_METERS:g} m from each side and {_VERTICAL_CUT:g} m from the "
                      "bottom; coordinates stay absolute, --md clips the network to the box")
_ap.add_argument("--recombine", action="store_true", default=False,
                 help="unstructured quadrilaterals: the gmsh triangle mesh (with or without --md) "
                      "recombined and subdivided into an all-quad mesh")
_ap.add_argument("--no-barriers", dest="barriers", action="store_false", default=True,
                 help="drop the sealing barriers B1-B3 under --md")
_ap.add_argument("--barrier-factor", type=float, default=_FIG8_BARRIER_FACTOR, metavar="F",
                 help=f"barrier permeability = rock * F; default {_FIG8_BARRIER_FACTOR:g}")
_ap.add_argument("--no-f6", dest="f6", action="store_false", default=True,
                 help="drop the F6 low-angle linking connector")
_ap.add_argument("--f6-factor", type=float, default=_FIG8_LINK_PERM_FACTOR, metavar="F",
                 help=f"F6 permeability = rock * F; default {_FIG8_LINK_PERM_FACTOR:g} "
                      f"(faults use {_FIG8_MD_FRAC_PERM_FACTOR:g})")
_ap.add_argument("--diagnose-binding", action="store_true", default=False,
                 help="PorePy Newton diagnostic; ignored")
_ap.add_argument("--dump-metric", action="store_true", default=False,
                 help="PorePy Newton diagnostic; ignored")
_ap.add_argument("--ls-min-alpha", type=float, default=0.1, metavar="A",
                 help="PorePy line-search floor; ignored (no Newton iteration)")
_ap.add_argument("--tol", type=float, default=NL_TOL, metavar="T",
                 help="PorePy Newton tolerance; ignored (no Newton iteration)")
_ap.add_argument("--max-iter", type=int, default=NL_MAX_ITER, metavar="N",
                 help="PorePy Newton iteration cap; ignored (no Newton iteration)")
# openCSMP-only
_ap.add_argument("--csmp-exe", default=None, metavar="PATH",
                 help="csmp_2d_plume driver; default $CSMP_2D_PLUME or ../openCSMP/build/csmp_2d_plume "
                      "next to the PorePy checkout")
_ap.add_argument("--tables-dir", default=None, metavar="DIR",
                 help="cache of openCSMP's *LookupTable.bin files (linked into the run folder, filled "
                      "on the first run); default $CSMP_TABLES_DIR or openCSMP/build/example_inputs/"
                      "CVFEM_examples")
_ap.add_argument("--cfl-scaling", type=float, default=0.1, metavar="C",
                 help="CFL scaling of the explicit transport (0.1 as in Weis et al. 2014)")
_ap.add_argument("--check-mesh", action="store_true", default=False,
                 help="build the mesh and model, export the initial state and stop")
_args = _ap.parse_args()

if not 0.0 <= _args.z_init <= 1.0:
    raise SystemExit(f"--z-init {_args.z_init} outside [0, 1]")
if _args.snap_years[0] != 0.0 or any(b <= a for a, b in zip(_args.snap_years, _args.snap_years[1:])):
    raise SystemExit(f"--snap-years {_args.snap_years} must start at 0 and be strictly increasing")
if not 0.0 < _args.dt_min <= _args.dt_nominal <= _args.dt_max:
    raise SystemExit("time steps must satisfy 0 < --dt-min <= --dt-nominal <= --dt-max")
if _args.recombine and not _args.md and _args.grid_type == "cartesian":
    print("NOTE: --recombine builds an unstructured quad mesh with gmsh; --grid-type cartesian ignored.")
if _args.dt_constant is not None:
    if _args.dt_constant <= 0.0:
        raise SystemExit("--dt-constant must be positive")
    _bad = [y for y in _args.snap_years
            if abs(round(y / _args.dt_constant) * _args.dt_constant - y) > 1e-9 * max(1.0, y)]
    if _bad:
        raise SystemExit(f"--dt-constant {_args.dt_constant:g} must divide each snap year evenly; "
                         f"offenders: {_bad}")
for _dest, _default in _POREPY_ONLY.items():
    if getattr(_args, _dest) != _default:
        print(f"WARNING: --{_dest.replace('_', '-')} is a PorePy (Newton/TPFA/MPFA) option with no "
              "counterpart in openCSMP's CVFEM scheme -- ignored.")
if _args.md and _args.grid_type == "cartesian":
    print("NOTE: --md is meshed with gmsh (triangles, or quads with --recombine); --grid-type ignored.")


# -- geometry ------------------------------------------------------------------------------- #
def domain_box() -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) in absolute Fig. 8 coordinates."""
    if _args.truncated_domain:
        return (_TRUNCATE_METERS, DOMAIN_WIDTH - _TRUNCATE_METERS, _VERTICAL_CUT, DOMAIN_HEIGHT)
    return (0.0, DOMAIN_WIDTH, 0.0, DOMAIN_HEIGHT)


def network():
    """(faults, barriers) in PorePy's order.  faults: [(name, k, aperture, (x0, y0, x1, y1))];
    barriers: [(name, k, thickness, [(x, y), ...])]."""
    k = ROCK["permeability"]
    faults = [(f"F{i + 1}", _FIG8_MD_FRAC_PERM_FACTOR * k, _FIG8_FAULT_APERTURE, seg)
              for i, seg in enumerate(_FIG8_FAULTS)]
    if _args.f6:
        faults += [("F6", _args.f6_factor * k, _FIG8_LINK_APERTURE, seg) for seg in _FIG8_LINK_FAULTS]
    barriers = []
    if _args.barriers:
        barriers = [(f"B{i + 1}", _args.barrier_factor * k, _FIG8_BARRIER_THICKNESS, list(poly))
                    for i, poly in enumerate(_FIG8_BARRIERS)]
    return faults, barriers


def _project(p, a, b):
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
    return a + t * (b - a)


def _snap(faults, barriers):
    """Move fault ends and barrier vertices that miss another line by less than
    _SNAP_TOLERANCE onto it (the digitised T-junctions of F5, F6 and B3 miss by 0.04-0.4 m)."""
    lines = [(n, np.array(s[:2]), np.array(s[2:])) for n, _, _, s in faults]
    lines += [(n, np.array(a), np.array(b)) for n, _, _, poly in barriers
              for a, b in zip(poly[:-1], poly[1:])]

    def snapped(name, p):
        p = np.asarray(p, dtype=float)
        for other, a, b in lines:
            if other != name:
                q = _project(p, a, b)
                if 0.0 < np.linalg.norm(p - q) < _SNAP_TOLERANCE:
                    return q
        return p

    faults = [(n, k, a, (*snapped(n, s[:2]), *snapped(n, s[2:]))) for n, k, a, s in faults]
    barriers = [(n, k, t, [tuple(snapped(n, v)) for v in poly]) for n, k, t, poly in barriers]
    return faults, barriers


def _clip(seg, box):
    """Liang-Barsky clip of a segment to the box; None if outside."""
    x0, y0, x1, y1 = seg
    xmin, xmax, ymin, ymax = box
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if p == 0.0:
            if q < 0.0:
                return None
        else:
            r = q / p
            if p < 0.0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
    if t1 - t0 < 1e-9:
        return None
    return (x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy)


def _clip_polyline(poly, box):
    """Clip a polyline to the box; returns the (single, contiguous) inside part or None."""
    pieces = [c for c in (_clip((*a, *b), box) for a, b in zip(poly[:-1], poly[1:])) if c]
    if not pieces:
        return None
    out = [pieces[0][:2]]
    for c in pieces:
        out.append(c[2:])
    return out


def _strip_polygon(poly, width):
    """Closed outline of a band of the given width centred on the polyline (mitred joins)."""
    pts = np.asarray(poly, dtype=float)
    tang = np.diff(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    offsets = [nrm[0]]
    for n0, n1 in zip(nrm[:-1], nrm[1:]):
        m = n0 + n1
        offsets.append(m / np.dot(m, n0))            # miter: unit distance from both segments
    offsets.append(nrm[-1])
    offsets = 0.5 * width * np.array(offsets)
    return np.vstack([pts + offsets, (pts - offsets)[::-1]])


# -- meshing -------------------------------------------------------------------------------- #
TRI_3, QUAD_4, BAR_2 = 8, 14, 2          # ANSYS element type codes


def _ccw(xy, elem):
    """Counter-clockwise node order (shoelace)."""
    pts = xy[elem]
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    return list(elem) if area > 0 else list(elem[::-1])


def _axis(lo, hi, breaks, h):
    """1D node coordinates from lo to hi with nodes at every break, spacing <= ~h."""
    cuts = [lo] + [b for b in breaks if lo < b < hi] + [hi]
    xs = [lo]
    for a, b in zip(cuts[:-1], cuts[1:]):
        n = max(1, int(round((b - a) / h)))
        xs += list(np.linspace(a, b, n + 1)[1:])
    return np.array(xs)


def mesh_cartesian(h):
    """Structured quadrilaterals; anomaly edges are grid lines."""
    xmin, xmax, ymin, ymax = domain_box()
    xs = _axis(xmin, xmax, INLET_X, h)
    ys = _axis(ymin, ymax, [], h)
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    nid = lambda i, j: j * nx + i                      # noqa: E731
    quads = [[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
             for j in range(ny - 1) for i in range(nx - 1)]
    fam = {"HOST": (QUAD_4, quads),
           "BOTTOM": (BAR_2, [[nid(i, 0), nid(i + 1, 0)] for i in range(nx - 1)]),
           "TOP": (BAR_2, [[nid(i, ny - 1), nid(i + 1, ny - 1)] for i in range(nx - 1)]),
           "LEFT": (BAR_2, [[nid(0, j), nid(0, j + 1)] for j in range(ny - 1)]),
           "RIGHT": (BAR_2, [[nid(nx - 1, j), nid(nx - 1, j + 1)] for j in range(ny - 1)])}
    return xy, fam


def mesh_gmsh(h, faults=(), barriers=(), recombine=False):
    """gmsh (OCC) mesh of the box conforming to the anomaly edges, the fault lines and the
    barrier zones; refined to _CSMP_FAULT_CELL_SIZE_FACTOR * h along the faults and to
    _CSMP_BARRIER_CELL_SIZE_FACTOR * h along the barrier zones."""
    import gmsh
    xmin, xmax, ymin, ymax = domain_box()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("csmp_fig8")
    occ = gmsh.model.occ
    rect = occ.addRectangle(xmin, ymin, 0.0, xmax - xmin, ymax - ymin)

    strips = []                                          # (name, surface tag) clipped to the box
    for name, _, thickness, poly in barriers:
        outline = _strip_polygon(poly, thickness)
        pts = [occ.addPoint(x, y, 0.0) for x, y in outline]
        curves = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        surf = occ.addPlaneSurface([occ.addCurveLoop(curves)])
        box = occ.addRectangle(xmin, ymin, 0.0, xmax - xmin, ymax - ymin)
        inside, _ = occ.intersect([(2, surf)], [(2, box)])
        strips += [(name, tag) for d, tag in inside if d == 2]
    tools = [(0, occ.addPoint(x, ymin, 0.0)) for x in INLET_X if xmin < x < xmax]
    tools += [(2, tag) for _, tag in strips]
    fault_lines = []
    for name, _, _, (x0, y0, x1, y1) in faults:
        fault_lines.append((name, occ.addLine(occ.addPoint(x0, y0, 0.0), occ.addPoint(x1, y1, 0.0))))
    tools += [(1, tag) for _, tag in fault_lines]
    _, out_map = occ.fragment([(2, rect)], tools)
    occ.synchronize()

    n_points = sum(1 for d, _ in tools if d == 0)
    zone_surfaces: dict[str, list[int]] = {}
    for k, (name, _) in enumerate(strips):
        zone_surfaces.setdefault(name, []).extend(t for d, t in out_map[1 + n_points + k] if d == 2)
    family_curves: dict[str, list[int]] = {}
    for k, (name, _) in enumerate(fault_lines):
        family_curves.setdefault(name, []).extend(
            t for d, t in out_map[1 + n_points + len(strips) + k] if d == 1)
    in_zone = {t for ts in zone_surfaces.values() for t in ts}
    host_surfaces = [t for d, t in gmsh.model.getEntities(2) if t not in in_zone]

    boundary = {"BOTTOM": [], "TOP": [], "LEFT": [], "RIGHT": []}
    eps = 1e-6 * (xmax - xmin)
    for _, tag in gmsh.model.getBoundary(gmsh.model.getEntities(2), combined=True, oriented=False):
        bx0, by0, _, bx1, by1, _ = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(by0 - ymin) < eps and abs(by1 - ymin) < eps:
            boundary["BOTTOM"].append(abs(tag))
        elif abs(by0 - ymax) < eps and abs(by1 - ymax) < eps:
            boundary["TOP"].append(abs(tag))
        elif abs(bx0 - xmin) < eps and abs(bx1 - xmin) < eps:
            boundary["LEFT"].append(abs(tag))
        elif abs(bx0 - xmax) < eps and abs(bx1 - xmax) < eps:
            boundary["RIGHT"].append(abs(tag))

    # all-quad meshes: recombine, then subdivide every cell into quads (which halves the size),
    # so mesh at twice the target size first
    scale = 2.0 if recombine else 1.0
    h = scale * h
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    h_frac = _CSMP_FAULT_CELL_SIZE_FACTOR * h
    zone_size = _CSMP_BARRIER_CELL_SIZE_FACTOR * h
    fields = []

    def refine(curves, size):
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", curves)
        gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 400)
        f_thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", size)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", h)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMin", size)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMax", 2.0 * h)
        fields.append(f_thr)

    if family_curves:
        refine([c for cs in family_curves.values() for c in cs], h_frac)
    if zone_surfaces:
        zone_curves = [abs(t) for _, t in gmsh.model.getBoundary(
            [(2, s) for ts in zone_surfaces.values() for s in ts], combined=False, oriented=False)]
        refine(sorted(set(zone_curves)), zone_size)
    if fields:
        f_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
    if recombine:
        gmsh.option.setNumber("Mesh.Algorithm", 8)          # frontal-Delaunay for quads
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # blossom
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)    # -> all quadrilaterals
    else:
        gmsh.option.setNumber("Mesh.Algorithm", 5)          # Delaunay (CVFEM upwinding)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.removeDuplicateNodes()

    tags, coords, _ = gmsh.model.mesh.getNodes()
    index = {int(t): i for i, t in enumerate(tags)}
    xy = coords.reshape(-1, 3)[:, :2].copy()

    def elements(dim, entities):
        by_n: dict[int, list] = {}
        for e in entities:
            etypes, _, enodes = gmsh.model.mesh.getElements(dim, e)
            for et, nodes in zip(etypes, enodes):
                n = gmsh.model.mesh.getElementProperties(et)[3]
                by_n.setdefault(n, []).extend(
                    [index[int(t)] for t in nodes[k:k + n]] for k in range(0, len(nodes), n))
        return by_n

    fam = {}
    for name, surfaces in [("HOST", host_surfaces)] + list(zone_surfaces.items()):
        cells = elements(2, surfaces)
        if 4 in cells:
            fam[name] = (QUAD_4, [_ccw(xy, np.array(e)) for e in cells[4]])
        if 3 in cells:
            fam[f"{name}_TRI" if 4 in cells else name] = (TRI_3, [_ccw(xy, np.array(e)) for e in cells[3]])
    for name, curves in list(boundary.items()) + list(family_curves.items()):
        fam[name] = (BAR_2, elements(1, curves).get(2, []))
    gmsh.finalize()
    return xy, fam


def write_ansys(basename: Path, xy, fam) -> None:
    """ANSYS/ICEM .asc (families) + binary .dat (see ANSYS_Interface::ReadMeshBinary)."""
    names = [n for n in fam if fam[n][1]]
    typename = {TRI_3: "TRI_3", QUAD_4: "QUAD_4", BAR_2: "BAR_2"}
    with open(f"{basename}.asc", "w") as f:
        f.write(f"{basename.parent}\n'{basename.name}.asc' generated by csmp_2d_solver.py (gmsh/numpy)\n")
        f.write(f"{len(names)} # Number of families\n")
        f.write("# Objectname        Elementtype        Material-ID   Number of elements\n")
        for n in names:
            f.write(f"{n:<20}{typename[fam[n][0]]:<24}0{len(fam[n][1]):>17}\n")
        f.write("# now the elements which make up each object are listed in sequence\n")
        eid = 0
        for n in names:
            m = len(fam[n][1])
            f.write(f"{n} {typename[fam[n][0]]} {m}\n")
            ids = list(range(eid, eid + m))
            for k in range(0, m, 10):
                f.write(" ".join(map(str, ids[k:k + 10])) + " \n")
            eid += m
    etypes = [fam[n][0] for n in names for _ in fam[n][1]]
    plist = [i for n in names for e in fam[n][1] for i in e]
    nn, ne = len(xy), len(etypes)
    with open(f"{basename}.dat", "wb") as f:
        f.write(struct.pack("<I", nn))
        f.write(struct.pack(f"<{nn}d", *xy[:, 0]))
        f.write(struct.pack(f"<{nn}d", *xy[:, 1]))
        f.write(struct.pack(f"<{nn}d", *([0.0] * nn)))
        f.write(struct.pack(f"<{nn}i", *([-30] * nn)))
        f.write(struct.pack(f"<{nn}d", *([0.0] * nn)))
        f.write(struct.pack("<I", ne))
        f.write(struct.pack(f"<{ne}i", *etypes))
        f.write(struct.pack("<I", len(plist)))
        f.write(struct.pack(f"<{len(plist)}I", *plist))
        f.write(struct.pack("<Ii", 1, -1))     # no neighbour record (openCSMP rebuilds it)
        f.write(struct.pack("<I", ne))
        f.write(struct.pack(f"<{ne}i", *([0] * ne)))
    with open(f"{basename}-regions.txt", "w") as f:
        f.write("no properties\n\n" + "\n".join(names) + "\n")


# -- openCSMP input files ------------------------------------------------------------------- #
def write_config(path: Path, final_years: float) -> None:
    """openCSMP configuration: blocks are separated by exactly one blank line."""
    salinity = 100.0 * _args.z_init                  # openCSMP salinity is wt% NaCl
    t_top = T_TOP - 273.15
    p_top = P_TOP * 1.0e6
    path.write_text(f"""WEIS ET AL. (2014) FIG. 8 CONDITION-2 PLUME -- written by csmp_2d_solver.py

# (block2) default property values
permeability			{ROCK['permeability']:g}
horizontal permeability		{ROCK['permeability']:g}
vertical permeability		{ROCK['permeability']:g}
porosity			{ROCK['porosity']:g}
nodal porosity			{ROCK['porosity']:g}
thermal conductivity		{ROCK['thermal_conductivity']:g}
nodal heat capacity rock	{ROCK['specific_heat_capacity']:g}
nodal density rock		{ROCK['density']:g}
nodal compressibility rock	{ROCK['compressibility']:g}
salinity			{salinity:g}
salinity top			{salinity:g}
temperature			{t_top:g}
fluid pressure			{p_top:g}
nodal heat flux bottom		0.
gravity flag			1.
cfl scaling			{_args.cfl_scaling:g}
open top			0
no standard output		1

# (block3) regional property values
Model		complete	porosity	{ROCK['porosity']:g}

# (block6) boundary conditions for arbitrary-shaped model
TOP		complete	DIRICH	temperature		{t_top:g}
TOP		complete	DIRICH	fluid pressure		{p_top:g}

# (block7) run settings
duration	{final_years:g}
time increment	{_args.dt_max:g}
output time	{final_years:g}
""")


def find_driver() -> Path:
    candidates = [_args.csmp_exe, os.environ.get("CSMP_2D_PLUME")]
    candidates += [str(p / "openCSMP" / "build" / "csmp_2d_plume") for p in HERE.parents]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return Path(c).resolve()
    raise SystemExit("csmp_2d_plume not found: build it (cmake --build build --target csmp_2d_plume "
                     "in openCSMP) and pass --csmp-exe or set $CSMP_2D_PLUME")


def find_tables_dir(driver: Path) -> Path:
    d = _args.tables_dir or os.environ.get("CSMP_TABLES_DIR")
    return Path(d) if d else driver.parent / "example_inputs" / "CVFEM_examples"


def link_tables(tables: Path, run_dir: Path) -> int:
    n = 0
    for tab in tables.glob("*LookupTable*.bin"):
        target = run_dir / tab.name
        if not target.exists():
            target.symlink_to(tab.resolve())
        n += 1
    return n


def store_new_tables(tables: Path, run_dir: Path) -> None:
    """Tables built during this run (missing from the cache) are copied into the cache."""
    tables.mkdir(parents=True, exist_ok=True)
    for tab in run_dir.glob("*LookupTable*.bin"):
        if not tab.is_symlink() and not (tables / tab.name).exists():
            shutil.copy2(tab, tables / tab.name)


# -- main ----------------------------------------------------------------------------------- #
def main() -> int:
    driver = find_driver()
    tag = case_tag(_args.scheme, _args.consistent, _args.grid_type, _args.cell_size,
                   _args.q_anomaly, _args.z_init,
                   _args.dt_nominal if _args.dt_nominal != DT_NOMINAL else None,
                   _args.dt_min if _args.dt_min != DT_MIN else None,
                   _args.dt_max if _args.dt_max != DT_MAX else None,
                   _args.snap_years[-1] if _args.snap_years[-1] != _DEFAULT_SNAP_YEARS[-1] else None,
                   lag=_args.lag_buoyancy, md=_args.md, recombine=_args.recombine,
                   truncated_domain=_args.truncated_domain, dt_constant=_args.dt_constant)
    run_dir = HERE / f"visualization_csmp_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"openCSMP driver: {driver}\nrun folder:      {run_dir}")

    h = DEFAULT_CELL_SIZE if _args.cell_size is None else _args.cell_size
    faults, barriers = [], []
    if _args.md:
        box = domain_box()
        all_faults, all_barriers = _snap(*network())
        for name, k, a, seg in all_faults:
            c = _clip(seg, box)
            if c is not None:
                faults.append((name, k, a, c))
        for name, k, t, poly in all_barriers:
            c = _clip_polyline(poly, box)
            if c is not None:
                barriers.append((name, k, t, c))
        xy, fam = mesh_gmsh(h, faults, barriers, recombine=_args.recombine)
    elif _args.recombine:
        xy, fam = mesh_gmsh(h, recombine=True)
    elif _args.grid_type == "simplex":
        xy, fam = mesh_gmsh(h)
    else:
        xy, fam = mesh_cartesian(h)
    mesh = run_dir / "csmp_mesh"
    write_ansys(mesh, xy, fam)
    n_cells = sum(len(fam[n][1]) for n in fam if fam[n][0] != BAR_2)
    print(f"mesh: {len(xy)} nodes, {n_cells} cells"
          + (f", faults {[f[0] for f in faults]}, barriers {[b[0] for b in barriers]}"
             if _args.md else ""))

    write_config(run_dir / "csmp_plume-configuration.txt", _args.snap_years[-1])
    run = [f"mesh {mesh}", f"config {run_dir / 'csmp_plume'}", f"output {run_dir / 'csmp_plume'}",
           "snap_years " + " ".join(f"{y:.12g}" for y in _args.snap_years),
           f"dt_nominal {_args.dt_nominal:.12g}", f"dt_min {_args.dt_min:.12g}",
           f"dt_max {_args.dt_max:.12g}", "linear_rtol 1e-5", "linear_maxit 400",
           f"cfl_scaling {_args.cfl_scaling:.12g}", f"q_background {Q_BACKGROUND:.12g}",
           f"q_anomaly {_args.q_anomaly:.12g}", f"anomaly_x {INLET_X[0]:.12g} {INLET_X[1]:.12g}",
           f"rock_permeability {ROCK['permeability']:.12g}", f"rock_porosity {ROCK['porosity']:.12g}",
           f"check_only {int(_args.check_mesh)}"]
    if _args.dt_constant is not None:
        run.append(f"dt_constant {_args.dt_constant:.12g}")
    run += [f"fault {name} {k:.12g} {a:.12g}" for name, k, a, _ in faults if fam.get(name, (0, []))[1]]
    for name, k, _, _ in barriers:
        run += [f"zone {z} {k:.12g}" for z in (name, f"{name}_TRI") if fam.get(z, (0, []))[1]]
    run_file = run_dir / "csmp_plume.run"
    run_file.write_text("\n".join(run) + "\n")

    tables = find_tables_dir(driver)
    n_tab = link_tables(tables, run_dir)
    if n_tab == 0:
        print(f"NOTE: no lookup tables in {tables}; openCSMP builds them in this run (about an hour).")
    (run_dir / "run_complete.json").unlink(missing_ok=True)

    log = run_dir / "csmp_plume.log"
    print(f"running {driver.name} (scheme output in {log.name}) ...", flush=True)
    t0 = time.time()
    with open(log, "w") as out:
        # stderr carries the driver's progress lines; stdout (and a copy of stderr) go to the log
        proc = subprocess.Popen([str(driver), str(run_file)], cwd=run_dir, stdout=out,
                                stderr=subprocess.PIPE, text=True, bufsize=1)
        # openCSMP's own messages often lack a trailing newline, so the driver's lines can
        # arrive glued to them: look for the markers anywhere in the line
        markers = ("csmp_2d_plume", "Bottom heat", "fault ", "barrier ", "Hydrostatic", "check_only")
        for line in proc.stderr:
            out.write(line)
            hits = [line.find(m) for m in markers if m in line]
            if hits:
                print(line[min(hits):].rstrip(), flush=True)
        status = proc.wait()
    elapsed = time.time() - t0
    store_new_tables(tables, run_dir)

    if status == 0 and not _args.check_mesh:
        final_years = _args.snap_years[-1]
        print(f"SIMULATION COMPLETE: reached final time {final_years:g} years in {elapsed:.1f} s")
        with open(run_dir / "run_complete.json", "w") as f:
            json.dump({"completed": True, "final_time_years": final_years,
                       "elapsed_run_seconds": elapsed, "num_nodes": int(len(xy)),
                       "num_cells": int(n_cells), "argv": sys.argv[1:]}, f, indent=2)
    elif status == 0:
        print(f"mesh check done in {elapsed:.1f} s: initial state in {run_dir}")
    else:
        print(f"SIMULATION INCOMPLETE: openCSMP exited with status {status}; see {log}")
    return status


if __name__ == "__main__":
    sys.exit(main())
