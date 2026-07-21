"""Weis et al. (2014) Fig. 8 heat-flux plume, condition 2 (9 km x 3 km).

CLI: --scheme {hu, hu-mw}, --consistent (MPFA), --grid-type, --cell-size,
--q-anomaly [W/m^2, default 5], --z-init (initial uniform NaCl overall
composition, default 0; also sets the hydrostatic-column and boundary fluid),
--snap-years (exact snapshot/export schedule, default 0..50000 every 2500),
--dt-nominal/--dt-min/--dt-max (dynamic stepping, default 5/0.01/25 yr).
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
from porepy.examples.geothermal_flow.obl_sampler import NSplineSampler, VTKSampler

# Main directives
case_name = "condition_1"

day_to_second = 86400
year_to_second = 365.0 * day_to_second
to_Mega = 1.0e-6

# Dynamic time stepping (all CLI-overridable).  The schedule pins EXACT landing -- and
# VTU export -- at the Fig. 8 snapshot instants; dt adapts freely in between.
_DEFAULT_SNAP_YEARS = tuple(float(y) for y in range(0, 50001, 2500))  # 0..50 kyr / 2.5 kyr
DT_NOMINAL = 5.0           # nominal (initial) step [yr] (--dt-nominal)
DT_MIN = 0.01                # smallest allowed step [yr] (--dt-min)
DT_MAX = 25.0               # largest allowed step [yr]  (--dt-max)

# --------------------------------------------------------------------------------------- #
#  Weis et al. (2014) Fig. 8, condition 2 -- boundary & initial conditions.
#  Top (surface): open, Dirichlet p = P_TOP and T = 10 degC.  Bottom: closed to fluid
#  flow (the pressure equation sees no-flux everywhere but the top), Neumann heat
#  INFLUX of 0.05 W/m^2 (background) + Q_ANOMALY over the central 1 km.  Sides: closed
#  and adiabatic.  IC: uniform 10 degC, uniform Z_INIT salt, and a brine-column
#  hydrostatic pressure profile integrated at (Z_INIT, T_TOP).
# --------------------------------------------------------------------------------------- #
P_TOP = 0.101325                 # surface pressure [MPa]; paper: atmospheric (0.1) -- the EOS
                            # table floor is 0.5 MPa, so the surface is idealized at 1 MPa
T_TOP = 283.15              # surface temperature [K] (10 degC)
Q_BACKGROUND = 0.05         # background crustal heat flux [W/m^2]
Q_ANOMALY = 5.0             # anomaly heat flux [W/m^2] over the inlet (--q-anomaly)
Z_INIT = 0.0                # initial (uniform) NaCl overall composition [-] (--z-init)
DOMAIN_HEIGHT = 3000.0      # [m]


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
                      f"{Z_INIT} (table range 0..0.2)")
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
_ap.add_argument("--lag-buoyancy", action="store_true",
                 help="freeze the buoyancy upwind direction over each time step "
                      "(CSMP++'s frozen-upwind policy, Weis et al. sec. 2.7)")
_args = _ap.parse_args()
if not 0.0 <= _args.z_init <= 0.2:
    raise SystemExit(f"--z-init {_args.z_init} outside the opensowat table "
                     "range z in [0, 0.2]")
if _args.snap_years[0] != 0.0 or any(
        b <= a for a, b in zip(_args.snap_years, _args.snap_years[1:])):
    raise SystemExit(f"--snap-years {_args.snap_years} must start at 0 and be "
                     "strictly increasing")
if not 0.0 < _args.dt_min <= _args.dt_nominal <= _args.dt_max:
    raise SystemExit("time steps must satisfy 0 < --dt-min <= --dt-nominal <= --dt-max")
Q_ANOMALY = _args.q_anomaly
Z_INIT = _args.z_init

# Dynamic time stepping: dt grows/shrinks with Newton effort inside (3, 8) iterations,
# recomputes at 0.3x on failure, and the schedule forces exact landing on every
# snapshot instant, which is also exactly where VTUs are exported.
schedule = [y * year_to_second for y in _args.snap_years]
tf = schedule[-1]
time_manager = pp.TimeManager(
    schedule=schedule,
    dt_init=_args.dt_nominal * year_to_second,
    dt_min_max=(_args.dt_min * year_to_second, _args.dt_max * year_to_second),
    constant_dt=False,
    iter_max=13,
    iter_optimal_range=(3, 8),
    iter_relax_factors=(0.5, 1.5),
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
                                    _args.dt_nominal, _args.dt_min, _args.dt_max,
                                    _args.snap_years[-1],
                                    lag=_args.lag_buoyancy)),
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
    "step_control_method": "None",
}
params["consistent_discretization"] = _args.consistent
params["lag_buoyancy_direction"] = _args.lag_buoyancy
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


# Instance of the computational model
model = GeothermalBrineFlowModel(params)

HERE = os.path.dirname(os.path.abspath(__file__))
# Constitutive approach shared by every subsection_4_2 solver: Driesner opensowat OBL
# tables sampled with the C2 tensor-spline backend (consistent value/Jacobian).
TABLE_LEVEL = 3                           # opensowat .vtr level (0..4 available)
USE_SPLINE = True                         # True -> NSplineSampler; False -> VTKSampler probe
_TABLE_DIR = os.path.join(
    HERE, os.pardir, os.pardir, "model_configuration", "constitutive_description",
    "driesner_vtk_files")


def _attach_samplers(model) -> None:
    """Attach the level-``TABLE_LEVEL`` Driesner OBL samplers (phz + ptz), exactly as
    porepy_1d_solver / porepy_3d_solver do."""
    Sampler = NSplineSampler if USE_SPLINE else VTKSampler
    phz = Sampler(os.path.join(_TABLE_DIR, f"opensowat_xph_l_{TABLE_LEVEL}_grads.vtr"))
    phz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, h, p)
    model.obl_sampler = phz
    ptz = Sampler(os.path.join(_TABLE_DIR, f"opensowat_xpt_l_{TABLE_LEVEL}_grads.vtr"))
    ptz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, t, p)
    ptz.translation_factors = (0.0, -273.15, 0.0)            # T in degC -> K in the sampler
    model.obl_sampler_ptz = ptz


_attach_samplers(model)


tb = time.time()
solver_params = {
    "nl_convergence_criteria": {
        "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
            tol=1.0e-4, metric=pp.EquationBasedLebesgueMetric(model)),
    },
    "nl_divergence_criteria": {
        "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=13),
    },
}
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
