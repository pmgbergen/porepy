"""PorePy 3D geothermal solver on the Berre et al. (2021) benchmark-3 fractured geometry
(subsection 4.3).

This is the 3D analogue of ``subsection_4_1/porepy_1d_solver.py``: it keeps the *exact*
Driesner brine geothermal two-phase compositional constitutive model of the 1D/2D solvers
and swaps the geometry for **Case 3 of the 3D flow benchmark** [1] -- the box
(0,1) x (0,2.25) x (0,1) m crossed by **8 highly conductive fractures**.  The resulting
mixed-dimensional grid has 1 matrix (3D) + 8 fracture planes (2D) + 7 intersection lines
(1D), coupled through 8 codim-1 and 14 codim-2 mortar interfaces (no 0D points).

Geometry
    ``Benchmark3DC3`` (``model_configuration/geometry_description/geometry_market.py``),
    which builds the benchmark grid via ``benchmark_3d_case_3(refinement_level)`` (gmsh
    simplex mesh; level 0 ~ 30K tets, up to level 3 ~ 500K).

Boundary conditions (mirroring the 1D column)
    Inlet = the full SOUTH face (y = 0), outlet = the full NORTH face (y = 2.25).  Pressure
    AND temperature are imposed (Dirichlet) on both: p_inlet / T_inlet at the inlet,
    p_outlet / T_outlet at the outlet.  The four lateral faces (x=const, z=const) are zero-
    flux Neumann.  ``get_inlet_outlet_sides`` is overridden here to select these full end
    faces instead of ``Benchmark3DC3``'s tiny corner spheres.  Flow runs along +y; gravity
    acts along z.  Initial state: the same y-ramp pressure and uniform ambient temperature.

Constitutive model
    ``DriesnerBrineFlowModel`` (HU / standard primary equations) or
    ``DriesnerBrineFractionalFlowModel`` (HU-mw / fractional-flow), with the level-``l``
    Driesner opensowat OBL tables (phz + ptz) attached as ``obl_sampler`` / ``obl_sampler_ptz``.

Regime (forced flow, faithful to benchmark 3)
    Brine is driven by a ``p_inlet -> p_outlet`` drop from the south inlet face to the north
    outlet face, through the fracture network along the long (y) axis.  Gravity acts along +z
    (the box's 1 m axis); the HU-BM(mp) buoyancy is a correction to the forced flow.

Scheme knob (``--scheme``), mirroring ``subsection_4_1/porepy_1d_solver.py``'s HU vs HU-mw
    HU     -> HU-BM(mp): ``fractional_flow=False``, ``buoyancy_upwinding='hybrid'`` (the
              mobility-product buoyant term lambda_g lambda_d / lambda_T; DEFAULT).
    HU-mw  -> mobility-weighted: the fractional-flow template (``DriesnerBrineFractionalFlowModel``)
              + ``fractional_flow=True``.  The SUBDOMAIN permeability becomes the base
              ``MassWeightedPermeability`` weighting ``total_mass_mobility * k`` (as in the 1D
              solver), while the interface stays rock-only (see below).
    PPU    -> phase-potential upwinding: ``buoyancy_upwinding='phase_potential'``.

Mixed-dimensional permeability fix
    The Driesner model pulls in ``MassWeightedPermeability`` whose ``normal_permeability`` is
    *unconditionally* mobility-weighted (constitutive_laws.py:721).  On the highly conductive
    matrix-fracture interfaces of this benchmark that weighting is unstable: for HU-BM(mp) it
    double-counts the separately-applied mobility (interface flux ~1e13x -> NaN); for HU-mw it
    makes the interface enthalpy-advection flux blow the Newton iteration up (residual -> inf).
    So we override the interface ``normal_permeability`` to be rock-only (matrix ``k``, fractures
    ``k * FRACTURE_K_FACTOR``) for EVERY scheme -- as ``subsection_4_2/porepy_2d_solver.py`` does --
    and keep the SUBDOMAIN ``permeability`` scheme-dependent (rock-only for HU/PPU, mobility-
    weighted for HU-mw).  The fractional mobility upwinding still enters via the subdomain flux.

Geometry knob (``--md``)
    By DEFAULT the run is FIXED-DIMENSIONAL: a single fracture-free 3D box over the domain (no
    fractures, no interfaces; ``--box-cell-size`` sets the Cartesian cell size).  Pass ``--md`` to
    switch to the MIXED-DIMENSIONAL benchmark-3 grid (8 conductive fractures + intersections + the
    mortar interfaces); ``--refinement-level`` then selects the mesh density.

Defaults (production configuration)
    ``--scheme HU  --md(off -> fixed-dim)  --refinement-level 0  --scale 1000  --linear-solver cpr
    --days 73000  --dt-days 50``  with gravity ON.  Override any of these on the command line.

Output naming / caching (``--scheme``, ``--no-gravity``, ``--md``)
    Each run writes its VTU visualization to a folder named for the three physical switches, so
    distinct configurations never clobber one another and re-running a configuration refreshes its
    own folder:  ``output/<scheme>_<dim>_<grav>/<scheme>_<dim>_<grav>_<step>.vtu`` where
    ``dim = md`` (mixed-dimensional, ``--md``) or ``fd`` (fixed-dimensional, default) and
    ``grav = gravity`` (default) or ``nogravity`` (``--no-gravity``).  E.g.
    ``output/HU_fd_gravity/`` (default) or ``output/HU-mw_md_nogravity/``.  See :func:`_output_name`.

Run
    ``python porepy_3d_solver.py --check``                     (build+assemble smoke test only)
    ``python porepy_3d_solver.py``                             (defaults: HU, fixed-dim, cpr, 73000 d)
    ``python porepy_3d_solver.py --md --scheme HU-mw --no-gravity``   (mixed-dim HU-mw, g=0)

References
    [1] Berre, I., Boon, W. M., Flemisch, B., Fumagalli, A., Glaeser, D., Keilegavlen, E.,
        ... & Zulian, P. (2021). Verification benchmarks for single-phase flow in
        three-dimensional fractured porous media. Advances in Water Resources, 147, 103759.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Sequence

import numpy as np

import porepy as pp

from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    Benchmark3DC3 as ModelGeometry,
)
from porepy.examples.geothermal_flow.model_configuration.DriesnerModelConfiguration import (  # noqa: E501
    DriesnerBrineFlowModel,               # HU / PPU (standard primary equations)
    DriesnerBrineFractionalFlowModel,     # HU-mw   (fractional-flow primary equations)
)
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_two_phase_moderate_pressure,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_two_phase_moderate_pressure,
)
from porepy.examples.geothermal_flow.model_configuration.flow_model_base import (  # noqa: E501
    geothermal_nonlinear_solver,  # NewtonSolver that dispatches to model.solve_linear_system
)
from porepy.examples.geothermal_flow.model_configuration.geothermal_export import (  # noqa: E501
    DriesnerPhaseExport,
)
from porepy.examples.geothermal_flow.obl_sampler import NSplineSampler, VTKSampler

# --------------------------------------------------------------------------------------------- #
#  Fixed parameters
# --------------------------------------------------------------------------------------------- #
DAY = 86400.0
TO_MEGA = 1.0e-6
TABLE_LEVEL = 3                          # Driesner opensowat .vtr level (matches subsection_4_1)
USE_SPLINE = True                        # True -> NSplineSampler (consistent C2-spline Jacobian)

# Benchmark-3 box (raw metres, as read from fracture_network.csv; benchmark_3d_case_3 does NOT
# apply unit scaling to the imported coordinates).  Flow (and the pressure ramp) is along y.
_FLOW_AXIS = 1                           # y is the long axis of the box (length 2.25 m at scale 1)
_Y_MIN, _Y_MAX = 0.0, 2.25               # unscaled domain extent along the flow axis

# Conductive fractures: k_fracture = FRACTURE_K_FACTOR * k_matrix.  The benchmark uses a
# fracture/matrix conductivity ratio of ~1e4 (Table 5: K_2 = 1e2 * K_3, plus aperture) -- we
# carry that ratio at geothermal permeability scale (k_matrix = 1e-15 m^2 -> k_fracture = 1e-11).
FRACTURE_K_FACTOR = 1.0e4

# Gravitational acceleration [m/s^2].  This is the single knob for the gravity constant: it feeds
# ``params["gravity_constant"]``, which ``flow_model_base.gravity_field`` uses for BOTH the buoyant
# phase segregation and the hydrostatic Darcy term.  ``--no-gravity`` overrides it to 0.0; pass
# ``--gravity-constant <g>`` for any other value (e.g. a reduced-gravity or Martian run).
GRAVITY_ACCELERATION = pp.GRAVITY_ACCELERATION       # 9.80665 m/s^2 (Earth standard)

# Two-phase forced-flow BC/IC values (from subsection_4_1's moderate two-phase case).
P_INLET = 20.0                           # [MPa] at the inlet corner (y = 0)
P_OUTLET = 1.0                           # [MPa] at the outlet corner (y = 2.25)
T_INLET = 673.15                         # [K] hot brine injected at the inlet corner
T_OUTLET = 423.15                        # [K] cool ambient at the outlet corner / initially

HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE_DIR = os.path.join(
    HERE, os.pardir, os.pardir, "model_configuration", "constitutive_description",
    "driesner_vtk_files")

# HU-BM scheme -> model parametrization (mirrors subsection_4_2).  "HU" = HU-BM(mp): the
# mobility-product buoyant term reached via fractional_flow=False + hybrid upwinding.
_SCHEME_CONFIG = {
    "HU":    dict(fractional_flow=False, buoyancy_upwinding="hybrid"),
    "HU-mw": dict(fractional_flow=True,  buoyancy_upwinding="hybrid"),
    "PPU":   dict(fractional_flow=False, buoyancy_upwinding="phase_potential"),
}


# --------------------------------------------------------------------------------------------- #
#  BC / IC retargeted to the 3D benchmark box
# --------------------------------------------------------------------------------------------- #
def _flow_bounds(domain: pp.Domain) -> tuple[float, float]:
    """(y_min, y_max) of the domain along the flow axis -- reads the *actual* (possibly scaled)
    bounding box, so the ramp is correct at any ``geometry_scale``."""
    bb = domain.bounding_box
    return float(bb["ymin"]), float(bb["ymax"])


def _pressure_ramp(xc: np.ndarray, y0: float, y1: float) -> np.ndarray:
    """Linear pressure ramp P_INLET (at y0) -> P_OUTLET (at y1) along the flow (y) axis.

    ``xc`` is a (N, 3) array of cell/face centres.  Using the live domain bounds ``[y0, y1]``
    (rather than a hard-coded length) keeps the ramp correct under geometry scaling.
    """
    y = xc[:, _FLOW_AXIS]
    return ((y - y0) * P_OUTLET + (y1 - y) * P_INLET) / (y1 - y0)


class BC_benchmark3d(BC_two_phase_moderate_pressure):
    """Two-phase moderate-pressure geothermal BC, retargeted to the benchmark-3 box.

    Reuses the parent's table-derived enthalpy / overall-fraction / flux-type machinery and
    the hot/cold inlet-outlet temperature patches (placed by ``get_inlet_outlet_sides``); only
    the pressure ramp is overridden so it runs along the box's flow (y) axis, not the collapsed
    2000 m column.
    """

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        return _pressure_ramp(boundary_grid.cell_centers.T, *_flow_bounds(self._domain))

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        T = T_OUTLET * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = T_INLET
        T[outlet_idx] = T_OUTLET
        return T


class IC_benchmark3d(IC_two_phase_moderate_pressure):
    """Two-phase moderate-pressure geothermal IC, retargeted to the benchmark-3 box.

    Pressure = the same y-axis ramp as the BC (so the initial state is consistent with the
    driving Dirichlet pressures); temperature = uniform ambient T_OUTLET.  Enthalpy, gas
    saturation and partial fractions are table-derived from (p, T, z=0) by the parent.
    """

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return _pressure_ramp(sd.cell_centers.T, *_flow_bounds(self._domain))

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return np.full(sd.num_cells, T_OUTLET)


# --------------------------------------------------------------------------------------------- #
#  Model assembly
# --------------------------------------------------------------------------------------------- #
def _rescale_domain(domain: pp.Domain, s: float) -> pp.Domain:
    """A copy of ``domain`` with every bounding-box coordinate multiplied by ``s``."""
    return pp.Domain(bounding_box={k: v * s for k, v in domain.bounding_box.items()})


def _scale_mdg(mdg: pp.MixedDimensionalGrid, s: float) -> None:
    """Multiply every grid coordinate in ``mdg`` by ``s`` and recompute geometry.

    ``benchmark_3d_case_3`` returns a grid in raw benchmark metres (y in [0, 2.25]); PorePy's
    ``units`` system does NOT touch imported coordinates, so to work at a geological scale we
    rescale the coordinates directly.  Each subdomain and each interface side-grid is scaled
    once (tracked by object id), then ``compute_geometry`` rebuilds every derived quantity --
    face/cell centres, areas, normals, volumes -- and, for the interfaces, the mortar cell
    volumes (interface areas).  Topological maps (projections, upwinding) are scale-invariant.
    """
    seen: set[int] = set()

    def scale(g) -> None:
        if id(g) in seen:
            return
        seen.add(id(g))
        g.nodes = g.nodes * s
        g.compute_geometry()

    for sd in mdg.subdomains():
        scale(sd)
        # Boundary grids are built (unscaled) with the mdg; recompute so their cell centres /
        # areas follow the scaled parent -- otherwise domain_boundary_sides can't find the
        # scaled inlet/outlet faces (the ramp then returns NaN at the outlet).
        bg = mdg.subdomain_to_boundary_grid(sd)
        if bg is not None:
            bg.compute_geometry()
    for intf in mdg.interfaces():
        for side_grid in intf.side_grids.values():
            scale(side_grid)
        intf.compute_geometry()


# Staggered, axis-aligned fracture pattern (fractions of the domain box).  Each spec is
# ``(axis, position, (u0, u1), (v0, v1))``: a rectangle in the plane ``axis = position`` spanning
# the two in-plane directions over ``[u0, u1] x [v0, v1]``.  For ``axis="x"`` the in-plane
# directions are (y, z); for "y" they are (x, z); for "z" they are (x, y).  Positions and extents
# are deliberately UNEVEN and PARTIAL (not full-span) so the planes form an irregular, staggered
# network -- some cross, some terminate in a tip inside the matrix -- rather than a regular lattice.
_FRACTURE_SPECS: tuple[tuple[str, float, tuple[float, float], tuple[float, float]], ...] = (
    ("x", 0.22, (0.05, 0.55), (0.10, 0.72)),
    ("x", 0.44, (0.38, 0.95), (0.28, 0.90)),
    ("x", 0.68, (0.15, 0.62), (0.05, 0.48)),
    ("x", 0.86, (0.52, 0.92), (0.40, 0.88)),
    ("y", 0.28, (0.08, 0.58), (0.18, 0.80)),
    ("y", 0.52, (0.34, 0.90), (0.05, 0.58)),
    ("y", 0.71, (0.05, 0.48), (0.32, 0.95)),
    ("y", 0.88, (0.46, 0.94), (0.12, 0.56)),
    ("z", 0.31, (0.14, 0.70), (0.10, 0.60)),
    ("z", 0.54, (0.40, 0.95), (0.36, 0.86)),
    ("z", 0.66, (0.06, 0.54), (0.48, 0.94)),
    ("z", 0.82, (0.50, 0.90), (0.22, 0.72)),
)


def _axis_aligned_fractures(
    physdims: np.ndarray, specs=_FRACTURE_SPECS
) -> list[np.ndarray]:
    """Axis-aligned rectangular fractures on the box [0,Lx]x[0,Ly]x[0,Lz], from ``specs``.

    Each spec ``(axis, pos, (u0,u1), (v0,v1))`` becomes a ``(3, 4)`` array of rectangle corners in
    the plane ``axis = pos`` (fractions of the box), spanning the two in-plane directions over the
    given fractional ranges.  The rectangles are axis-aligned so ``pp.meshing.cart_grid`` snaps
    them to the nearest Cartesian cell faces -- the matrix stays K-orthogonal, so TPFA discretises
    the gravity vector source exactly.  The default :data:`_FRACTURE_SPECS` is a STAGGERED, PARTIAL
    pattern (uneven positions, sub-domain extents) that forms an irregular fracture network.
    """
    Lx, Ly, Lz = float(physdims[0]), float(physdims[1]), float(physdims[2])
    fracs: list[np.ndarray] = []
    for axis, pos, (u0, u1), (v0, v1) in specs:
        if axis == "x":                         # plane x=pos*Lx, spans y in u*Ly, z in v*Lz
            x = pos * Lx
            y0, y1, z0, z1 = u0 * Ly, u1 * Ly, v0 * Lz, v1 * Lz
            fracs.append(np.array([[x, x, x, x], [y0, y1, y1, y0], [z0, z0, z1, z1]]))
        elif axis == "y":                       # plane y=pos*Ly, spans x in u*Lx, z in v*Lz
            y = pos * Ly
            x0, x1, z0, z1 = u0 * Lx, u1 * Lx, v0 * Lz, v1 * Lz
            fracs.append(np.array([[x0, x1, x1, x0], [y, y, y, y], [z0, z0, z1, z1]]))
        else:                                   # plane z=pos*Lz, spans x in u*Lx, y in v*Ly
            z = pos * Lz
            x0, x1, y0, y1 = u0 * Lx, u1 * Lx, v0 * Ly, v1 * Ly
            fracs.append(np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [z, z, z, z]]))
    return fracs


def _build_model_class(FlowModel):
    """Compose the geometry + BC + IC + Driesner ``FlowModel`` into a runnable model class.

    Mirrors ``subsection_4_1/porepy_1d_solver.py``: the geometry/BC/IC mixed in FRONT of the
    Driesner ``FlowModel`` shadow the column defaults baked into ``_DriesnerBrineBase``.
    """

    class GeothermalFlowModel3D(
            DriesnerPhaseExport, ModelGeometry, BC_benchmark3d, IC_benchmark3d, FlowModel):

        def set_geometry(self) -> None:
            """Build a fully CARTESIAN domain grid.

            Both branches are K-orthogonal, so TPFA discretises the gravity vector source EXACTLY
            (unlike a simplex mesh, where the two-point vector source is inconsistent).
            ``params["fractures"]`` (``--md``, default False) selects between:
              * a fracture-free EQUI-DIMENSIONAL Cartesian box over [0,1]x[0,2.25]x[0,1] (scaled),
                a single 3D subdomain with no interfaces; and
              * a MIXED-DIMENSIONAL Cartesian grid with 12 full-span axis-aligned planar fractures
                -- four perpendicular to each of x, y, z (see :func:`_axis_aligned_fractures`).

            ``pp.meshing.cart_grid`` snaps the axis-aligned fracture rectangles to the nearest
            Cartesian cell faces, so the fractures conform to the grid and the matrix stays
            K-orthogonal.  ``set_wells`` is seeded first (no wells here).
            """
            self.set_wells()                       # -> self._wells = [] (no wells here)
            self.set_domain()                      # -> self._domain (scaled box; used by BC/IC)
            s = float(self.params.get("geometry_scale", 1.0))
            physdims = np.array([1.0, _Y_MAX, 1.0]) * s
            # Refinement: each level halves the cell size (2**level cells per direction per level).
            level = int(self.params.get("refinement_level", 0))
            h = float(self.params.get("box_cell_size", 0.1)) / (2 ** level) * s
            nx = np.maximum(np.round(physdims / h).astype(int), 1)
            fracs = (_axis_aligned_fractures(physdims)
                     if self.params.get("fractures", False) else [])
            self.mdg = pp.meshing.cart_grid(fracs, nx, physdims=physdims)
            self.nd = self.mdg.dim_max()
            pp.set_local_coordinate_projections(self.mdg)
            self.set_well_network()

        # ---- fracture-free box geometry (used only when params["fractures"] is False) ----
        def set_domain(self) -> None:
            """The benchmark-3 bounding box, scaled by ``geometry_scale`` (matches the fractured
            grid + BC/IC)."""
            s = float(self.params.get("geometry_scale", 1.0))
            self._domain = pp.Domain(
                {"xmin": 0.0, "xmax": 1.0 * s, "ymin": _Y_MIN * s, "ymax": _Y_MAX * s,
                 "zmin": 0.0, "zmax": 1.0 * s})

        def set_fractures(self) -> None:
            self._fractures = []

        def grid_type(self):
            return self.params.get("grid_type", "cartesian")

        def meshing_arguments(self) -> dict:
            # Scale the cell size with the domain so the cell COUNT is scale-invariant.
            s = float(self.params.get("geometry_scale", 1.0))
            return {"cell_size": float(self.params.get("box_cell_size", 0.05)) * s}

        def get_inlet_outlet_sides(self, sd):
            """Inlet = the full SOUTH (y = y_min) face, outlet = the full NORTH (y = y_max) face;
            every other external face is no-flow (zero Neumann).

            This replaces ``Benchmark3DC3``'s tiny corner spheres (rc=0.1), which pick only ~8
            faces scattered across three faces at each corner -- not an inlet/outlet *region*.
            Mirroring the 1D column, pressure AND temperature are imposed (Dirichlet) across the
            two opposite end faces along the long (y) axis, while the four lateral faces (x=const,
            z=const) carry zero-flux Neumann.  Flow runs along +y; gravity acts along z.

            Returns face indices (``pp.Grid``) or boundary-cell indices (``pp.BoundaryGrid``), as
            the BC mixin expects.
            """
            sides = self.domain_boundary_sides(sd)
            inlet_facets = np.where(sides.south)[0]    # y = y_min  (full face)
            outlet_facets = np.where(sides.north)[0]   # y = y_max  (full face)
            return inlet_facets, outlet_facets

        # ---- MPFA fluxes: TPFA's gravity vector source (and conduction) is inconsistent on the
        #      non-K-orthogonal SIMPLEX benchmark grid -- the flux uses (face_center - cell_center),
        #      which is only parallel to the face normal on Cartesian cells, so on tetrahedra the
        #      gravity term is wrong/dropped (Cartesian works, simplex does not).  MPFA is consistent
        #      on general grids.  For HU both coefficients are CONSTANT (rock permeability; constant
        #      porosity-weighted rock+fluid conductivity), so both are discretized ONCE at setup and
        #      never re-discretized (see add_nonlinear_fourier_flux_discretization below). ----
        def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

        def fourier_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.fourier_keyword, list(subdomains))

        def add_nonlinear_fourier_flux_discretization(self) -> None:
            """Do NOT re-discretize the Fourier flux each Newton iteration.

            The CF base (compositional_flow.py) unconditionally re-discretizes the Fourier flux,
            assuming a nonlinear thermal conductivity.  Here the conductivity is the constant
            porosity-weighted rock+fluid conductivity (no mechanics -> porosity fixed), so the
            (now MPFA) Fourier discretization is invariant: it is built ONCE in ``discretize()`` at
            setup, and re-running the heavy MPFA discretize every iteration is pure waste.  The
            Darcy flux is likewise discretized once for HU -- the CF base only adds it to the
            nonlinear-flux list in the fractional-flow (HU-mw) branch.  (The advective upwind
            discretizations are still refreshed every iteration, as they must be.)
            """

        # ---- dimension-dependent permeability (conductive fractures) with a scheme-dependent
        #      mobility weighting (MD buoyancy fix) ----
        def _rock_permeability_values(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            """Cell-wise ROCK permeability (no mobility weighting): matrix (``sd.dim == nd``)
            = ``k``; lower-dim fractures + intersections = ``k * FRACTURE_K_FACTOR``.  This is the
            fracture-aware absolute permeability that the base ``MassWeightedPermeability`` lacks
            (it uses a single uniform ``solid.permeability``)."""
            size = sum(sd.num_cells for sd in subdomains)
            vals = np.full(size, self.solid.permeability)
            offset = 0
            for sd in subdomains:
                if sd.dim < self.nd:                # fractures + intersections: conductive
                    vals[offset:offset + sd.num_cells] = self.solid.permeability * FRACTURE_K_FACTOR
                offset += sd.num_cells
            return pp.wrap_as_dense_ad_array(vals, size, name="permeability")

        def _subdomain_permeability_scalar(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            """Scalar isotropic SUBDOMAIN permeability, selected by the scheme -- mirrors the 1D
            solver's HU vs HU-mw distinction (which is interface-free):

            * HU / PPU (``fractional_flow=False``) -> ROCK permeability only.  The total-mass
              HU-BM(mp) formulation applies the fluid mobility SEPARATELY (upwinded; the mp
              buoyancy multiplies by lambda_g lambda_d / lambda_T), so mobility is never baked into
              the tensor.
            * HU-mw (``fractional_flow=True``) -> ``total_mass_mobility * rock_k``: the base
              ``MassWeightedPermeability`` weighting, but on the fracture-aware rock ``k`` (the base
              uses a single uniform ``solid.permeability``).  This is exactly what the 1D solver's
              HU-mw uses -- the fractional-flow mass balance carries the mobility here.
            """
            rock = self._rock_permeability_values(subdomains)
            if pp.compositional_flow.is_fractional_flow(self):        # HU-mw
                scalar = self.total_mass_mobility(subdomains) * rock
                scalar.set_name("mass_mobility_weighted_permeability")
                return scalar
            return rock                                              # HU / PPU

        def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            """Subdomain permeability tensor: matrix ``k``, conductive fractures
            ``k*FRACTURE_K_FACTOR``; mobility-weighted for HU-mw.  See
            :meth:`_subdomain_permeability_scalar`."""
            return self.isotropic_second_order_tensor(
                subdomains, self._subdomain_permeability_scalar(subdomains))

        def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
            """Interface (normal) permeability = the lower-dimensional subdomain's ROCK
            permeability, projected to the mortar -- WITHOUT the mobility weighting, for EVERY
            scheme.

            The base ``MassWeightedPermeability.normal_permeability`` returns
            ``total_mass_mobility * k``.  On the highly conductive matrix-fracture interfaces of
            this geothermal benchmark that weighting is unstable: for HU-BM(mp) it double-counts
            the separately-applied mobility (~1e13x -> NaN), and for HU-mw it makes the interface
            enthalpy-advection flux blow the Newton iteration up (residual -> inf).  Rock ``k``
            (fracture-aware) keeps the matrix<->fracture Darcy coupling well-scaled; the fractional
            mobility upwinding still enters through the subdomain flux discretisation.
            """
            subdomains = self.interfaces_to_subdomains(interfaces)
            projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
            kn = projection.secondary_to_mortar_avg() @ self._rock_permeability_values(subdomains)
            kn.set_name("normal_permeability")
            return kn

    return GeothermalFlowModel3D


def _attach_samplers(model) -> None:
    """Attach the level-``TABLE_LEVEL`` Driesner OBL samplers (phz + ptz) to ``model``.

    Backend is ``NSplineSampler`` (value and gradient from one C2 tensor spline -> consistent
    Jacobian) when ``USE_SPLINE`` else ``VTKSampler``.  Must be called BEFORE ``prepare_simulation``
    (the IC samples the ptz table), i.e. before constructing the ``ModelRunner``.
    """
    Sampler = NSplineSampler if USE_SPLINE else VTKSampler
    phz = Sampler(os.path.join(_TABLE_DIR, f"opensowat_xph_l_{TABLE_LEVEL}_grads.vtr"))
    phz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, h, p)
    model.obl_sampler = phz
    ptz = Sampler(os.path.join(_TABLE_DIR, f"opensowat_xpt_l_{TABLE_LEVEL}_grads.vtr"))
    ptz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, t, p)
    ptz.translation_factors = (0.0, -273.15, 0.0)            # T in degC -> K in the sampler
    model.obl_sampler_ptz = ptz


def _output_name(scheme: str, gravity: bool, fractures: bool) -> str:
    """Deterministic output basename encoding the three physical switches ``--scheme``,
    ``--no-gravity`` and ``--md`` -- e.g. ``HU_fd_gravity`` or ``HU-mw_md_nogravity``.

    ``dim``  = ``md`` (mixed-dimensional, fractures) or ``fd`` (fixed-dimensional box).
    ``grav`` = ``gravity`` (g on) or ``nogravity`` (g = 0).
    Used for BOTH the VTU folder (``output/<name>/``) and the file prefix, so every configuration
    caches to its own folder and never overwrites another (see :func:`build_params`).
    """
    dim = "md" if fractures else "fd"
    grav = "gravity" if gravity else "nogravity"
    return f"{scheme}_{dim}_{grav}"


# Production defaults (the configuration the paper runs); every one is overridable on the CLI.
_DEFAULT_SCHEME = "HU"                    # HU = HU-BM(mp)
_DEFAULT_REFINEMENT_LEVEL = 0            # mesh refinement: effective cell size = box_cell_size /
#                                         2**level (level 0 = base; each level HALVES h -> ~8x cells
#                                         per level in 3D).  Applies to BOTH the box and --md.
_DEFAULT_FRACTURES = False              # fixed-dimensional box by default; --md -> mixed-dimensional
_DEFAULT_BOX_CELL_SIZE = 0.1            # base Cartesian cell size (before refinement)
_DEFAULT_GEOMETRY_SCALE = 1000.0       # 1 x 2.25 x 1 km box (buoyancy becomes significant)
_DEFAULT_T_END_DAYS = 73000.0          # ~200 yr transient
_DEFAULT_DT_DAYS = 365.0                # nominal (adaptive) time step
dt_downscale = 1.0/265.0
dt_upscale = 10.0
_DEFAULT_LINEAR_SOLVER = "cpr"         # Schur-reduced CPR (iterative, PETSc)
_DEFAULT_GRAVITY = True
_DEFAULT_AD_BACKEND = "native"
_DEFAULT_CPR_RTOL = 1.0e-5             # CPR GMRES relative tolerance
_DEFAULT_CPR_MAXIT = 400              # CPR GMRES iteration cap
_DEFAULT_CPR_ACCURACY_TOL = 1.0e-3   # post-solve gate -> fall back to direct above this
# VTU snapshot schedule [days]: the transient writes VTU/PVD ONLY at these instants (not every
# step), mirroring subsection_4_2/porepy_2d_solver.  They are placed in the TimeManager schedule so
# adaptive dt lands on them exactly, and become ``times_to_export``.  0 (the initial state) is
# always added.  Default: 0 + every 20 yr over the ~200 yr horizon.
_DEFAULT_SNAP_DAYS = (0.0, 7300.0, 14600.0, 29200.0, 43800.0, 58400.0, 73000.0)


def build_params(
    scheme: str = _DEFAULT_SCHEME,
    refinement_level: int = _DEFAULT_REFINEMENT_LEVEL,
    t_end_days: float = _DEFAULT_T_END_DAYS,
    dt_days: float = _DEFAULT_DT_DAYS,
    ad_backend: str = _DEFAULT_AD_BACKEND,
    fractures: bool = _DEFAULT_FRACTURES,
    box_cell_size: float = _DEFAULT_BOX_CELL_SIZE,
    geometry_scale: float = _DEFAULT_GEOMETRY_SCALE,
    linear_solver: str = _DEFAULT_LINEAR_SOLVER,
    gravity: bool = _DEFAULT_GRAVITY,
    gravity_constant: float | None = None,
    cpr_rtol: float = _DEFAULT_CPR_RTOL,
    cpr_maxit: int = _DEFAULT_CPR_MAXIT,
    cpr_accuracy_tol: float = _DEFAULT_CPR_ACCURACY_TOL,
    snap_days: Sequence[float] = _DEFAULT_SNAP_DAYS,
    **overrides,
) -> dict:
    """Assemble the params dict for one 3D geothermal benchmark run.

    Geometry (both branches are fully Cartesian, built with ``pp.meshing.cart_grid``)
      ``fractures``       -- True (``--md``) = mixed-dimensional grid with 12 staggered axis-aligned
                             fractures; False (default) = a fracture-free equidimensional box.
      ``box_cell_size``   -- the BASE Cartesian cell size (matrix), before refinement.
      ``refinement_level``-- effective cell size = ``box_cell_size / 2**refinement_level`` (level 0 =
                             base; each level halves h -> ~8x cells per level in 3D).  Applies to
                             BOTH branches.
      ``geometry_scale``  -- multiplies all coordinates (default 1000 -> a 1x2.25x1 km box) so the
                             geothermal buoyancy (~rho g dz over the scaled height) is significant;
                             the cell count is unchanged.

    Physics
      ``scheme``          -- HU (HU-BM(mp)), HU-mw (mobility-weighted) or PPU; see ``_SCHEME_CONFIG``.
      ``gravity``         -- False (``--no-gravity``) sets g=0 (removes buoyancy AND the hydrostatic
                             Darcy term).
      ``gravity_constant``-- gravitational acceleration [m/s^2] when gravity is ON; None -> the
                             module ``GRAVITY_ACCELERATION`` (9.80665).  Stored as
                             ``params["gravity_constant"]`` and consumed by
                             ``flow_model_base.gravity_field``.

    Linear solver
      ``linear_solver``   -- "cpr" (Schur-reduced CPR, iterative/PETSc; default), "direct" (SciPy
                             sparse LU) or "lu" (direct LU via MUMPS/PETSc).
      ``cpr_rtol``        -- CPR GMRES relative residual tolerance for the reduced (p,h,z) solve.
      ``cpr_maxit``       -- max (un-restarted) GMRES iterations before the solve is abandoned.
      ``cpr_accuracy_tol``-- post-solve gate on the full-system relative residual; above it the step
                             falls back to a direct LU (MUMPS) solve so Newton never advances on an
                             under-converged linear solve.

    Output
      ``folder_name`` / ``file_name`` are set to ``output/<name>`` / ``<name>`` where ``<name>`` =
      :func:`_output_name` (``<scheme>_<dim>_<grav>``), unless overridden via ``**overrides``.
    """
    if scheme not in _SCHEME_CONFIG:
        raise ValueError(f"scheme must be one of {sorted(_SCHEME_CONFIG)}, got {scheme!r}")
    _linear_solvers = {
        "direct": dict(use_petsc=False),
        "cpr": dict(use_petsc=True, petsc_preconditioner="cpr"),
        "lu": dict(use_petsc=True, petsc_preconditioner="lu"),
    }
    if linear_solver not in _linear_solvers:
        raise ValueError(
            f"linear_solver must be one of {sorted(_linear_solvers)}, got {linear_solver!r}")

    tf = t_end_days * DAY
    dt = dt_days * DAY
    # VTU export schedule (mirrors subsection_4_2/porepy_2d_solver): the snapshot instants (<= tf),
    # plus 0 and tf, go into the TimeManager SCHEDULE so adaptive dt lands on them EXACTLY (PorePy
    # clips any step that would overshoot a scheduled time); the SAME instants become
    # ``times_to_export`` so the runner writes VTU ONLY there, not on every time step.
    snap_seconds = sorted(d * DAY for d in snap_days if 0.0 <= d * DAY <= tf + 1.0e-6)
    schedule = sorted({0.0, tf, *snap_seconds})
    times_to_export = sorted({*snap_seconds, tf}) if snap_seconds else [0.0, tf]
    time_manager = pp.TimeManager(
        schedule=schedule, dt_init=dt, constant_dt=False,
        dt_min_max=(dt*dt_downscale , dt * dt_upscale), iter_max=11, iter_optimal_range=(3, 6),
        recomp_factor=0.5, recomp_max=10, print_info=True)

    # Geothermal rock (matches subsection_4_1) + benchmark-3 aperture (eps_2 = 1e-2 m).
    solid = pp.SolidConstants(
        permeability=1e-15, porosity=0.1, residual_aperture=1e-2,
        thermal_conductivity=2.0 * TO_MEGA, density=2700.0,
        specific_heat_capacity=880.0 * TO_MEGA)

    # Per-configuration output: folder AND file prefix encode (scheme, gravity, md) so distinct
    # runs cache to distinct folders and re-running a configuration refreshes only its own.
    name = _output_name(scheme, gravity, fractures)

    # Resolve the gravity constant [m/s^2]: 0 with --no-gravity, else the requested value or the
    # module GRAVITY_ACCELERATION.  gravity_field reads this single value.
    g_value = 0.0 if not gravity else (
        GRAVITY_ACCELERATION if gravity_constant is None else float(gravity_constant))

    params = dict(
        ad_backend=ad_backend,
        enable_buoyancy_effects=gravity,
        lag_buoyancy_direction=False,
        material_constants={"solid": solid},
        time_manager=time_manager,
        refinement_level=refinement_level,
        fractures=fractures,
        box_cell_size=box_cell_size,
        geometry_scale=geometry_scale,
        gravity=gravity,
        gravity_constant=g_value,
        cpr_rtol=cpr_rtol,
        cpr_maxit=cpr_maxit,
        cpr_accuracy_tol=cpr_accuracy_tol,
        times_to_export=times_to_export,       # write VTU only at the scheduled snapshots
        folder_name=os.path.join("output", name),
        file_name=name,
        step_control_method="None",
    )
    params.update(_linear_solvers[linear_solver])
    params.update(_SCHEME_CONFIG[scheme])
    params.update(overrides)
    return params


def build_model(scheme: str = "HU", **kw):
    """Construct (but do not run) the 3D geothermal benchmark model with samplers attached."""
    params = build_params(scheme, **kw)
    FlowModel = (
        DriesnerBrineFractionalFlowModel if params.get("fractional_flow", False)
        else DriesnerBrineFlowModel)
    model = _build_model_class(FlowModel)(params)
    _attach_samplers(model)
    return model


# --------------------------------------------------------------------------------------------- #
#  Drivers
# --------------------------------------------------------------------------------------------- #
def _solver_params(model) -> dict:
    return {
        "nl_convergence_criteria": {
            "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                tol=1.0e-3, metric=pp.EquationBasedLebesgueMetric(model)),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=11),
        },
    }


def check(scheme: str = _DEFAULT_SCHEME, refinement_level: int = _DEFAULT_REFINEMENT_LEVEL,
          fractures: bool = _DEFAULT_FRACTURES, box_cell_size: float = _DEFAULT_BOX_CELL_SIZE,
          geometry_scale: float = _DEFAULT_GEOMETRY_SCALE,
          linear_solver: str = _DEFAULT_LINEAR_SOLVER, gravity: bool = _DEFAULT_GRAVITY) -> None:
    """Build the model, prepare the simulation, and assemble the residual + Jacobian ONCE.

    A cheap structural smoke test: confirms the grid (mixed-dimensional benchmark-3 with ``--md``,
    or the fracture-free box by default) + Driesner EOS + the selected buoyancy scheme compose,
    seed a finite initial state, and produce a finite, correctly-shaped linear system -- without
    paying for a full transient solve.
    """
    print(f"\n=== 3D benchmark check: scheme={scheme}, fractures={fractures}, "
          f"refinement_level={refinement_level}, geometry_scale={geometry_scale} ===", flush=True)
    model = build_model(scheme, refinement_level=refinement_level,
                        fractures=fractures, box_cell_size=box_cell_size,
                        geometry_scale=geometry_scale, linear_solver=linear_solver,
                        gravity=gravity)
    t0 = time.time()
    model.prepare_simulation()
    print(f"  prepare_simulation: {time.time() - t0:.1f}s", flush=True)

    mdg = model.mdg
    by_dim = {d: len(mdg.subdomains(dim=d)) for d in range(mdg.dim_max() + 1)}
    cells = {d: sum(sd.num_cells for sd in mdg.subdomains(dim=d))
             for d in range(mdg.dim_max() + 1)}
    print(f"  mdg: dims {mdg.dim_min()}..{mdg.dim_max()}  subdomains/dim {by_dim}  "
          f"cells/dim {cells}  interfaces {mdg.num_interfaces()}", flush=True)
    bb = model._domain.bounding_box
    print(f"  domain: x[{bb['xmin']:.3g},{bb['xmax']:.3g}] y[{bb['ymin']:.3g},{bb['ymax']:.3g}] "
          f"z[{bb['zmin']:.3g},{bb['zmax']:.3g}] m", flush=True)
    print(f"  DoF: {model.equation_system.num_dofs()}", flush=True)

    # A bare ``equation_system.assemble()`` right after prepare fails (even for the working 1D
    # solver): the eliminated secondary variables (x_NaCl_liq, ...) on the matrix boundary grid
    # are only seeded once the nonlinear solver runs its pre-loop hooks.  Replicate that minimal
    # prefix (before_time_step -> before_nonlinear_loop -> before_nonlinear_iteration).
    model.before_time_step()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()

    t0 = time.time()
    A, b = model.equation_system.assemble()   # (Jacobian, residual) at the initial state
    print(f"  assemble: {time.time() - t0:.1f}s  A={A.shape} nnz={A.nnz}  "
          f"|b|={np.linalg.norm(b):.6e}  finite(A)={np.isfinite(A.data).all()}  "
          f"finite(b)={np.isfinite(b).all()}", flush=True)
    print("  OK -- model builds and assembles a finite linear system.", flush=True)


def run(scheme: str = _DEFAULT_SCHEME, refinement_level: int = _DEFAULT_REFINEMENT_LEVEL,
        t_end_days: float = _DEFAULT_T_END_DAYS, dt_days: float = _DEFAULT_DT_DAYS,
        ad_backend: str = _DEFAULT_AD_BACKEND, fractures: bool = _DEFAULT_FRACTURES,
        box_cell_size: float = _DEFAULT_BOX_CELL_SIZE,
        geometry_scale: float = _DEFAULT_GEOMETRY_SCALE,
        linear_solver: str = _DEFAULT_LINEAR_SOLVER, gravity: bool = _DEFAULT_GRAVITY,
        gravity_constant: float | None = None,
        cpr_rtol: float = _DEFAULT_CPR_RTOL, cpr_maxit: int = _DEFAULT_CPR_MAXIT,
        cpr_accuracy_tol: float = _DEFAULT_CPR_ACCURACY_TOL,
        snap_days: Sequence[float] = _DEFAULT_SNAP_DAYS,
        transport_predictor: bool = False) -> None:
    """Run the transient 3D geothermal benchmark to ``t_end_days``."""
    name = _output_name(scheme, gravity, fractures)
    g_value = 0.0 if not gravity else (
        GRAVITY_ACCELERATION if gravity_constant is None else float(gravity_constant))
    solver = linear_solver + (
        f" (rtol={cpr_rtol:.1e}, maxit={cpr_maxit}, acc_tol={cpr_accuracy_tol:.1e})"
        if linear_solver == "cpr" else "")
    lines = [
        f"\n=== 3D benchmark run: scheme={scheme}, "
        f"{'mixed' if fractures else 'fixed'}-dimensional, "
        f"level={refinement_level}, scale={geometry_scale} ===",
        f"  time:   tf={t_end_days} d, dt={dt_days} d",
        f"  solver: backend={ad_backend}, linear_solver={solver}, g={g_value} m/s^2",
    ]
    if transport_predictor:
        lines.append("  extras: transport-predictor=ON")
    lines.append(f"  output -> {os.path.join('output', name)}/")
    print("\n".join(lines), flush=True)
    model = build_model(scheme, refinement_level=refinement_level,
                        t_end_days=t_end_days, dt_days=dt_days, ad_backend=ad_backend,
                        fractures=fractures, box_cell_size=box_cell_size,
                        geometry_scale=geometry_scale, linear_solver=linear_solver,
                        gravity=gravity, gravity_constant=gravity_constant,
                        cpr_rtol=cpr_rtol, cpr_maxit=cpr_maxit,
                        cpr_accuracy_tol=cpr_accuracy_tol, snap_days=snap_days,
                        transport_predictor=transport_predictor)
    snaps = [d for d in snap_days if 0.0 <= d <= t_end_days + 1e-6]
    print(f"  VTU export at snapshots [days]: {snaps if snaps else [0.0, t_end_days]}", flush=True)
    sp = _solver_params(model)
    runner = pp.ModelRunner(model, sp, nonlinear_solver=geothermal_nonlinear_solver(sp))
    print("  DoF:", model.equation_system.num_dofs(), flush=True)
    model.schur_complement_primary_equations = (
        pp.compositional_flow.get_primary_equations_cf(model))
    model.schur_complement_primary_variables = (
        pp.compositional_flow.get_primary_variables_cf(model))
    model.exporter.write_vtu()                              # t = 0 snapshot
    t0 = time.time()
    runner.run()
    print(f"  run wall: {(time.time() - t0) / 60.0:.1f} min", flush=True)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D geothermal solver on benchmark-3 geometry.")
    p.add_argument("--scheme", choices=sorted(_SCHEME_CONFIG), default=_DEFAULT_SCHEME,
                   help="buoyancy scheme (HU = HU-BM(mp), default; HU-mw = mobility-weighted; PPU)")
    p.add_argument("--md", dest="fractures", action="store_true", default=_DEFAULT_FRACTURES,
                   help="MIXED-dimensional: use the benchmark-3 grid with 8 conductive fractures "
                        "(+ intersections + mortar interfaces).  Without --md (default) the run is "
                        "FIXED-dimensional: a single fracture-free box over the same domain.")
    p.add_argument("--refinement-level", type=int, choices=[0, 1, 2, 3],
                   default=_DEFAULT_REFINEMENT_LEVEL,
                   help="mesh refinement: effective cell size = --box-cell-size / 2**level "
                        "(level 0 = base; each level halves h -> ~8x cells per level in 3D). "
                        "Applies to both the box and --md.")
    p.add_argument("--box-cell-size", type=float, default=_DEFAULT_BOX_CELL_SIZE,
                   help="base Cartesian cell size (before --refinement-level), for both the box "
                        "and the --md matrix")
    p.add_argument("--scale", type=float, default=_DEFAULT_GEOMETRY_SCALE, dest="geometry_scale",
                   help="multiply all geometry coordinates by this factor "
                        "(default 1000 -> a 1x2.25x1 km box); cell count is unchanged")
    p.add_argument("--days", type=float, default=_DEFAULT_T_END_DAYS, help="end time [days]")
    p.add_argument("--dt-days", type=float, default=_DEFAULT_DT_DAYS,
                   help="nominal (adaptive) time step [days]")
    p.add_argument("--snap-days", type=str, default=None, metavar="D0,D1,...",
                   help="comma-separated days at which to write VTU snapshots (placed in the "
                        "schedule so adaptive dt hits them exactly; default: 0 + every 20 yr).  "
                        "VTU is written ONLY at these instants, not every step.")
    p.add_argument("--ad-backend", choices=["native", "sparsa"], default=_DEFAULT_AD_BACKEND)
    p.add_argument("--linear-solver", choices=["direct", "cpr", "lu"],
                   default=_DEFAULT_LINEAR_SOLVER,
                   help="linear solver: cpr (Schur-reduced CPR, iterative/PETSc; default), "
                        "direct (SciPy LU), or lu (direct LU via MUMPS/PETSc)")
    p.add_argument("--cpr-rtol", type=float, default=_DEFAULT_CPR_RTOL,
                   help="CPR GMRES relative residual tolerance (only with --linear-solver cpr)")
    p.add_argument("--cpr-maxit", type=int, default=_DEFAULT_CPR_MAXIT,
                   help="CPR GMRES max iterations before abandoning the iterative solve "
                        "(only with --linear-solver cpr)")
    p.add_argument("--cpr-accuracy-tol", type=float, default=_DEFAULT_CPR_ACCURACY_TOL,
                   help="post-solve full-residual gate; above it the step falls back to direct "
                        "LU/MUMPS (only with --linear-solver cpr)")
    p.add_argument("--no-gravity", dest="gravity", action="store_false",
                   help="set g=0 (removes buoyancy AND the hydrostatic Darcy term)")
    p.add_argument("--gravity-constant", type=float, default=None, metavar="G",
                   help="gravitational acceleration [m/s^2] when gravity is on "
                        f"(default {GRAVITY_ACCELERATION:.5f}); ignored with --no-gravity")
    p.add_argument("--transport-predictor", dest="transport_predictor", action="store_true",
                   default=False,
                   help="warm-start each FI Newton step with a cheap flow-order (reordered) "
                        "advective transport sweep for (h, z); only helps advection-dominated "
                        "high-CFL regions (fractures/intersections), off by default")
    p.add_argument("--check", action="store_true",
                   help="build + assemble once (structural smoke test), no transient solve")
    return p.parse_args()


def main() -> None:
    args = _cli()
    if args.check:
        check(args.scheme, args.refinement_level, args.fractures, args.box_cell_size,
              args.geometry_scale, args.linear_solver, args.gravity)
    else:
        snap_days = (tuple(float(d) for d in args.snap_days.split(",") if d.strip())
                     if args.snap_days else _DEFAULT_SNAP_DAYS)
        run(args.scheme, args.refinement_level, args.days, args.dt_days, args.ad_backend,
            args.fractures, args.box_cell_size, args.geometry_scale, args.linear_solver,
            args.gravity, args.gravity_constant, args.cpr_rtol, args.cpr_maxit,
            args.cpr_accuracy_tol, snap_days, args.transport_predictor)


if __name__ == "__main__":
    main()

# Examples (all defaults are the production configuration -- HU, fixed-dim, cpr, scale 1000,
# 73000 d, dt 50 d, gravity on):
#   python porepy_3d_solver.py                          # -> output/HU_fd_gravity/
#   python porepy_3d_solver.py --md                     # -> output/HU_md_gravity/  (fractured)
#   python porepy_3d_solver.py --md --scheme HU-mw --no-gravity   # -> output/HU-mw_md_nogravity/


# fixed dimensional case:
# _DEFAULT_BOX_CELL_SIZE = 0.025
# _DEFAULT_DT_DAYS = 3650.0                # nominal (adaptive) time step
#  time python porepy_3d_solver.py --scheme HU --no-gravity

# ************************************************************
# Number of iterations:  2
# Time value (year):  200.0
# Time index:  86
# ************************************************************
#
#
# ===================== DoF summary -- final =====================
# # degrees-of-freedom summary
# total DoF: 1008000   (subdomains: 1, interfaces: 0)
# # cells per subdomain, by dimension:
#   dim   n_subdomains      n_cells
#   3D             1       144000
# # variables:
#   name                             ndof   type
#   pressure                       144000   primary
#   temperature                    144000   secondary
#   enthalpy                       144000   primary
#   z_NaCl                         144000   primary
#   s_gas                          144000   secondary
#   x_NaCl_liq                     144000   secondary
#   x_NaCl_gas                     144000   secondary
# # primary dof: 432000   secondary (eliminated) dof: 576000
#
# 2026-07-15 16:29:06,246 - porepy.examples.geothermal_flow.model_configuration.flow_model_base - INFO - run statistics -> output/HU_fd_nogravity/run_statistics.txt (+ .json)
#   run wall: 78.0 min
# python porepy_3d_solver.py --scheme HU --no-gravity  28304.92s user 8174.53s system 776% cpu 1:18:20.25 total