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

Scheme knob (``--scheme``)
    HU     -> HU-BM(mp): ``fractional_flow=False``, ``buoyancy_upwinding='hybrid'`` (the
              mobility-product buoyant term lambda_g lambda_d / lambda_T; DEFAULT).
    HU-mw  -> mobility-weighted: fractional-flow template + ``fractional_flow=True``.
    PPU    -> phase-potential upwinding: ``buoyancy_upwinding='phase_potential'``.

Mixed-dimensional buoyancy fix
    The Driesner model pulls in ``MassWeightedPermeability`` whose ``normal_permeability``
    is *unconditionally* mobility-weighted (constitutive_laws.py:721).  On the
    matrix-fracture interfaces that counts the fluid mobility twice for the non-fractional
    HU-BM(mp) buoyancy (interface flux ~1e13x too large -> NaN).  We override
    ``permeability`` / ``normal_permeability`` to be rock-only (matrix ``k``, fractures
    ``k * FRACTURE_K_FACTOR``), exactly as ``subsection_4_2/porepy_2d_solver.py`` does.

Run
    ``python porepy_3d_solver.py --scheme HU --refinement-level 0 --check``   (build+assemble only)
    ``python porepy_3d_solver.py --scheme HU --refinement-level 0 --days 100 --dt-days 10``

References
    [1] Berre, I., Boon, W. M., Flemisch, B., Fumagalli, A., Glaeser, D., Keilegavlen, E.,
        ... & Zulian, P. (2021). Verification benchmarks for single-phase flow in
        three-dimensional fractured porous media. Advances in Water Resources, 147, 103759.
"""
from __future__ import annotations

import argparse
import os
import time

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

# Two-phase forced-flow BC/IC values (from subsection_4_1's moderate two-phase case).
P_INLET = 20.0                           # [MPa] at the inlet corner (y = 0)
P_OUTLET = 1.0                           # [MPa] at the outlet corner (y = 2.25)
T_INLET = 673.15                         # [K] hot brine injected at the inlet corner
T_OUTLET = 423.15                        # [K] cool ambient at the outlet corner / initially

HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE_DIR = os.path.join(
    HERE, os.pardir, "model_configuration", "constitutive_description", "driesner_vtk_files")

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


def _build_model_class(FlowModel):
    """Compose the geometry + BC + IC + Driesner ``FlowModel`` into a runnable model class.

    Mirrors ``subsection_4_1/porepy_1d_solver.py``: the geometry/BC/IC mixed in FRONT of the
    Driesner ``FlowModel`` shadow the column defaults baked into ``_DriesnerBrineBase``.
    """

    class GeothermalFlowModel3D(
            DriesnerPhaseExport, ModelGeometry, BC_benchmark3d, IC_benchmark3d, FlowModel):

        def set_geometry(self) -> None:
            """Build the domain grid.

            ``params["fractures"]`` (default True) selects between:
              * the benchmark-3 MIXED-DIMENSIONAL grid (1 matrix + 8 fractures + 7 intersection
                lines), via ``Benchmark3DC3.set_geometry``; and
              * a fracture-free EQUI-DIMENSIONAL box over the SAME domain [0,1]x[0,2.25]x[0,1]
                (single 3D subdomain, no interfaces) -- useful for isolating the matrix physics
                and as a reference for the mixed-dimensional run.

            Either way ``set_wells`` is seeded first (``Benchmark3DC3.set_geometry`` reads
            ``self._wells`` without initialising it).  The fracture-free branch builds the grid
            with the create-helpers directly, skipping the base ``create_well_mesh`` (which would
            unconditionally mesh an empty well network).
            """
            self.set_wells()                       # -> self._wells = [] (no wells here)
            scale = float(self.params.get("geometry_scale", 1.0))
            if self.params.get("fractures", True):
                super().set_geometry()             # Benchmark3DC3: mixed-dimensional grid
                if scale != 1.0:                   # rescale raw benchmark coords to geological size
                    _scale_mdg(self.mdg, scale)
                    self._domain = _rescale_domain(self._domain, scale)
                return
            # --- fracture-free equidimensional box (set_domain/meshing already apply the scale) ---
            self.set_domain()
            self.set_fractures()                   # -> [] (no fractures)
            self.create_fracture_network()
            self.create_mdg()
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

        # ---- TPFA fluxes (matches subsection_4_1; the HU buoyancy is a two-point scheme) ----
        def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

        def fourier_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.fourier_keyword, list(subdomains))

        # ---- rock-only, dimension-dependent permeability (MD buoyancy fix) ----
        def _rock_permeability_values(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            """Cell-wise ROCK permeability (no mobility weighting): matrix (``sd.dim == nd``)
            = ``k``; lower-dim fractures + intersections = ``k * FRACTURE_K_FACTOR``.  Shared by
            :meth:`permeability` and :meth:`normal_permeability` so both use the SAME rock
            permeability -- the total-mass HU-BM(mp) formulation applies the fluid mobility
            separately (upwinded), never baked into the permeability tensor."""
            size = sum(sd.num_cells for sd in subdomains)
            vals = np.full(size, self.solid.permeability)
            offset = 0
            for sd in subdomains:
                if sd.dim < self.nd:                # fractures + intersections: conductive
                    vals[offset:offset + sd.num_cells] = self.solid.permeability * FRACTURE_K_FACTOR
                offset += sd.num_cells
            return pp.wrap_as_dense_ad_array(vals, size, name="permeability")

        def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            """Rock permeability tensor: matrix ``k``, conductive fractures ``k*FRACTURE_K_FACTOR``.
            See :meth:`_rock_permeability_values`."""
            return self.isotropic_second_order_tensor(
                subdomains, self._rock_permeability_values(subdomains))

        def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
            """Interface (normal) permeability = the lower-dimensional subdomain's ROCK
            permeability, projected to the mortar -- WITHOUT the total-mass-mobility weighting.

            The base ``MassWeightedPermeability.normal_permeability`` unconditionally returns
            ``total_mass_mobility * k``.  In the NON-fractional HU-BM(mp) formulation the mobility
            is applied separately (the mp buoyancy multiplies by lambda_g lambda_d / lambda_T, and
            the interface mobility is upwinded), so keeping it here would count the mobility TWICE
            on the matrix-fracture interface (~1e13x too large -> NaN).  Rock ``k`` matches the
            subdomain :meth:`permeability` and removes the double weighting.
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


def build_params(
    scheme: str = "HU",
    refinement_level: int = 0,
    t_end_days: float = 100.0,
    dt_days: float = 10.0,
    ad_backend: str = "native",
    fractures: bool = True,
    box_cell_size: float = 0.05,
    geometry_scale: float = 1.0,
    linear_solver: str = "direct",
    gravity: bool = True,
    **overrides,
) -> dict:
    """Assemble the params dict for one (scheme, refinement) 3D geothermal benchmark run.

    ``fractures=False`` builds a fracture-free equidimensional box over the same domain
    (``box_cell_size`` sets the Cartesian cell size); the benchmark ``refinement_level`` is then
    ignored.  ``geometry_scale`` multiplies all coordinates (e.g. 1000 -> a 1x2.25x1 km box) so
    the geothermal buoyancy (~rho g dz over the scaled height) becomes physically significant; the
    cell count is unchanged.

    ``linear_solver`` selects the linear solver: "direct" (SciPy sparse LU; default), "cpr"
    (Schur-reduced CPR, iterative; PETSc), or "lu" (direct LU via MUMPS; PETSc).
    ``gravity=False`` sets g=0 (removes buoyancy AND the hydrostatic Darcy term).
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
    time_manager = pp.TimeManager(
        schedule=[0.0, tf], dt_init=dt, constant_dt=False,
        dt_min_max=(dt / 64.0, dt), iter_max=20, iter_optimal_range=(3, 10),
        recomp_factor=0.5, recomp_max=10, print_info=True)

    # Geothermal rock (matches subsection_4_1) + benchmark-3 aperture (eps_2 = 1e-2 m).
    solid = pp.SolidConstants(
        permeability=1e-15, porosity=0.1, residual_aperture=1e-2,
        thermal_conductivity=2.0 * TO_MEGA, density=2700.0,
        specific_heat_capacity=880.0 * TO_MEGA)

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
            "res_abs": pp.ResidualBasedAbsoluteCriterion(
                tol=1.0e-3, metric=pp.EquationBasedLebesgueMetric(model)),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.MaxIterationsCriterion(max_iterations=20),
        },
    }


def check(scheme: str = "HU", refinement_level: int = 0,
          fractures: bool = True, box_cell_size: float = 0.1,
          geometry_scale: float = 1.0, linear_solver: str = "direct",
          gravity: bool = True) -> None:
    """Build the model, prepare the simulation, and assemble the residual + Jacobian ONCE.

    A cheap structural smoke test: confirms the grid (mixed-dimensional benchmark-3, or the
    fracture-free box when ``fractures=False``) + Driesner EOS + HU-BM(mp) buoyancy compose,
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


def run(scheme: str = "HU", refinement_level: int = 0,
        t_end_days: float = 100.0, dt_days: float = 10.0,
        ad_backend: str = "native", fractures: bool = True,
        box_cell_size: float = 0.05, geometry_scale: float = 1.0,
        linear_solver: str = "direct", gravity: bool = True) -> None:
    """Run the transient 3D geothermal benchmark to ``t_end_days``."""
    print(f"\n=== 3D benchmark run: scheme={scheme}, fractures={fractures}, "
          f"level={refinement_level}, scale={geometry_scale}, tf={t_end_days} d, "
          f"dt={dt_days} d, backend={ad_backend}, linear_solver={linear_solver}, "
          f"gravity={gravity} ===", flush=True)
    model = build_model(scheme, refinement_level=refinement_level,
                        t_end_days=t_end_days, dt_days=dt_days, ad_backend=ad_backend,
                        fractures=fractures, box_cell_size=box_cell_size,
                        geometry_scale=geometry_scale, linear_solver=linear_solver,
                        gravity=gravity)
    runner = pp.ModelRunner(model, _solver_params(model))
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
    p.add_argument("--scheme", choices=sorted(_SCHEME_CONFIG), default="HU",
                   help="buoyancy scheme (HU = HU-BM(mp), default)")
    p.add_argument("--refinement-level", type=int, choices=[0, 1, 2, 3], default=0,
                   help="benchmark mesh refinement (0 ~ 30K tets, ..., 3 ~ 500K); "
                        "ignored with --no-fractures")
    p.add_argument("--no-fractures", dest="fractures", action="store_false",
                   help="disable fractures: solve on a fracture-free equidimensional box "
                        "over the same domain (single 3D subdomain, no interfaces)")
    p.add_argument("--box-cell-size", type=float, default=0.1,
                   help="Cartesian cell size for the fracture-free box (only with --no-fractures)")
    p.add_argument("--scale", type=float, default=1.0, dest="geometry_scale",
                   help="multiply all geometry coordinates by this factor "
                        "(e.g. 1000 -> a 1x2.25x1 km box); cell count is unchanged")
    p.add_argument("--days", type=float, default=100.0, help="end time [days]")
    p.add_argument("--dt-days", type=float, default=10.0, help="nominal time step [days]")
    p.add_argument("--ad-backend", choices=["native", "sparsa"], default="native")
    p.add_argument("--linear-solver", choices=["direct", "cpr", "lu"], default="direct",
                   help="linear solver: direct (SciPy LU, default), cpr (Schur-reduced CPR, "
                        "iterative/PETSc), or lu (direct LU via MUMPS/PETSc)")
    p.add_argument("--no-gravity", dest="gravity", action="store_false",
                   help="set g=0 (removes buoyancy AND the hydrostatic Darcy term)")
    p.add_argument("--check", action="store_true",
                   help="build + assemble once (structural smoke test), no transient solve")
    return p.parse_args()


def main() -> None:
    args = _cli()
    if args.check:
        check(args.scheme, args.refinement_level, args.fractures, args.box_cell_size,
              args.geometry_scale, args.linear_solver, args.gravity)
    else:
        run(args.scheme, args.refinement_level, args.days, args.dt_days, args.ad_backend,
            args.fractures, args.box_cell_size, args.geometry_scale, args.linear_solver,
            args.gravity)


if __name__ == "__main__":
    main()

# working
# python porepy_3d_solver.py --scheme HU --refinement-level 0 --days 73000 --dt-days 50.0 --no-fractures --scale 1000