"""Meteoric-recharge / halite-dissolution 2D solver (subsection 4.2, two_dimensional).

A gravity/head-driven flow cell through a halite-bearing HOT aquifer (the reverse of the
boiling-precipitation column): dilute meteoric water recharges at a topographic high (high head),
flows through the aquifer dissolving the immobile halite, and discharges as brine at a low (low
head).  Kept single-phase LIQUID by a deep, high-pressure regime, so it dissolves rather than boils.

Geometry (RechargeGeometry2D): a LX x LZ rectangle; y is vertical (y = LZ top, y = 0 base).
    recharge  = top face, x < RECHARGE_FRAC*LX   (top-left)
    discharge = top face, x > DISCHARGE_FRAC*LX  (top-right)

Initial condition (all LINEAR with depth; the halite is whatever the flash returns from these):
    pressure     : brine-hydrostatic, P_TOP_IC at the top -> ~+RHO_REF g LZ at the base
    temperature  : T_TOP_IC (250 C) -> T_BOT_IC (350 C), mean 300 C
    NaCl frac z  : Z_TOP -> Z_BOTTOM  (increasing downward -> halite precipitates at depth)

Boundary conditions:
    recharge  (Dirichlet): p = P_RECHARGE (high head), T = T_RECHARGE (cold), z = 0 (dilute)
    discharge (Dirichlet): p = P_DISCHARGE (low head), T = T_DISCHARGE
    base      (Dirichlet T only, no fluid flow): T = T_BOTTOM_BC (350 C) -- the geothermal heat
    every other face: no-flow, adiabatic

Reuses the subsection_4_2 machinery: graded OBL tables, Schur-CPR (PETSc), the weis backtracking
line search, the slave (exact flash each iterate) and the shared base nonlinear criterion.
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

# ------------------------------------------------------------------ initial condition (linear)
# Volcanic-hosted regime: LOW top pressure so the shallow, hot part flashes to a STEAM + HALITE
# cap, while the deeper (higher-p) part stays LIQUID -> a single model with both the Morgan
# recharge-discharge flow structure and a volcanic vapor-dominated cap over a liquid dissolution
# zone.  The steam+halite cap forms DYNAMICALLY where upflowing saline brine crosses the boiling
# curve (the IC seeds a salt-poor steam cap over the deep liquid+halite reservoir).
P_TOP_IC = 2.0              # IC pressure at the top [MPa] (LOW -> shallow steam cap)
RHO_REF = 1000.0            # brine reference density for the linear hydrostatic gradient [kg/m^3]
T_TOP_IC = 250.0 + 273.15   # top temperature [K]
T_BOT_IC = 350.0 + 273.15   # base temperature [K]  (mean 300 C)
Z_TOP = 0.80                 # NaCl overall fraction at the top [-]  (keeps S_h <= 0.15)
Z_BOTTOM = 0.95              # NaCl overall fraction at the base [-]  (S_h peaks ~0.13 at the base)

# ------------------------------------------------------------------ boundary conditions
P_RECHARGE = 15.0           # recharge (inlet) pressure [MPa]  (high head -> liquid inflow)
T_RECHARGE = 100.0 + 273.15 # recharge temperature [K]  (cold meteoric water)
Z_RECHARGE = 0.0            # recharge salinity [-]  (dilute / fresh)
P_DISCHARGE = 1.0           # discharge (outlet) pressure [MPa]  (low head -> steam vent)
T_DISCHARGE = 250.0 + 273.15# discharge temperature [K]
T_BOTTOM_BC = 350.0 + 273.15# fixed base temperature [K]  (the geothermal heat source)


def _linear_z(depth: np.ndarray) -> np.ndarray:
    return Z_TOP + (Z_BOTTOM - Z_TOP) * (depth / LZ)


def _linear_T(depth: np.ndarray) -> np.ndarray:
    return T_TOP_IC + (T_BOT_IC - T_TOP_IC) * (depth / LZ)


def _linear_p(depth: np.ndarray) -> np.ndarray:
    return P_TOP_IC + RHO_REF * G * depth * to_Mega


_HYDRO_CACHE: dict = {}


def _hydrostatic_p(depth: np.ndarray, sampler) -> np.ndarray:
    """IC pressure in true hydrostatic balance: integrate dp = rho_mix(z,T,p) g dz down from
    P_TOP_IC at the surface, using the flash MIXTURE density (not the constant RHO_REF, not a
    phase density). A constant-density column leaves dp/dz != rho_mix g everywhere, i.e. a
    spurious vertical Darcy flux the solver can never null -- the first step then stalls at every
    dt. rho depends (weakly) on p, so Picard-iterate. Cached per (P_TOP_IC, z/T profile)."""
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
            sampler.sample_at(np.column_stack([z, T, p]))
            pd = sampler.sampled_could.point_data
            if "Rho" not in pd:
                raise KeyError(f"mixture density 'Rho' not in flash output; have {list(pd)}")
            rho = np.asarray(pd["Rho"], float)
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
        rech, disch = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, np.concatenate((rech, disch)), "dir")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        rech, disch = self.get_inlet_outlet_sides(sd)
        base = np.where(self.domain_boundary_sides(sd).south)[0]      # geothermal base (fixed T)
        return pp.BoundaryCondition(sd, np.concatenate((rech, disch, base)), "dir")

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        depth = LZ - bg.cell_centers[1]
        p = _hydrostatic_p(depth, self.obl_sampler_ptz)              # same hydrostatic column as the IC
        rech, disch = self.get_inlet_outlet_sides(bg)
        p[rech] = P_RECHARGE
        p[disch] = P_DISCHARGE
        return p

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        depth = LZ - bg.cell_centers[1]
        T = _linear_T(depth)                                          # valid everywhere
        rech, disch = self.get_inlet_outlet_sides(bg)
        T[self.domain_boundary_sides(bg).south] = T_BOTTOM_BC
        T[rech] = T_RECHARGE
        T[disch] = T_DISCHARGE
        return T

    def bc_salinity(self, bg: pp.BoundaryGrid) -> np.ndarray:
        # background = the local IC z (so no-flow / outflow faces match the interior); recharge fresh.
        z = _linear_z(LZ - bg.cell_centers[1])
        rech, _ = self.get_inlet_outlet_sides(bg)
        z[rech] = Z_RECHARGE
        return z

    def bc_values_overall_fraction(self, component: pp.Component, bg: pp.BoundaryGrid) -> np.ndarray:
        return self.bc_salinity(bg)

    def bc_values_enthalpy(self, bg: pp.BoundaryGrid) -> np.ndarray:
        p, t, z = self.bc_values_pressure(bg), self.bc_values_temperature(bg), self.bc_salinity(bg)
        self.obl_sampler_ptz.sample_at(np.array((z, t, p)).T)
        return self.obl_sampler_ptz.sampled_could.point_data["H"] * 1.0e-3

    def bc_values_fractional_flow_component(self, component: pp.Component, bg: pp.BoundaryGrid) -> np.ndarray:
        z = self.bc_salinity(bg)
        is_salt = component == self.fluid.components[1]
        return z if is_salt else 1.0 - z


class ICRecharge(IC):
    """Linear p / T / z with depth; the halite (and enthalpy, saturations) come from the flash."""

    def _profiles(self, sd: pp.Grid):
        depth = LZ - sd.cell_centers[1]
        return _hydrostatic_p(depth, self.obl_sampler_ptz), _linear_T(depth), _linear_z(depth)

    def _sampled(self, sd: pp.Grid):
        p, t, z = self._profiles(sd)
        self.obl_sampler_ptz.sample_at(np.array((z, t, p)).T)
        return self.obl_sampler_ptz.sampled_could.point_data

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return self._profiles(sd)[0]

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return self._profiles(sd)[1]

    def ic_values_overall_fraction(self, component: pp.Component, sd: pp.Grid) -> np.ndarray:
        return self._profiles(sd)[2]

    def ic_values_partial_fractions(self, sd: pp.Grid) -> np.ndarray:
        d = self._sampled(sd)
        return np.clip(d["Xl"], 0.0, 1.0), np.clip(d["Xv"], 0.0, 1.0)

    def ic_values_gas_saturation(self, sd: pp.Grid) -> np.ndarray:
        return np.clip(self._sampled(sd)["S_v"], 0.0, 1.0)

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        return self._sampled(sd)["H"] * 1.0e-3


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
_ap.add_argument("--report-every-years", type=float, default=None, dest="report_every_years", metavar="N",
                 help="regular VTU cadence: report every N years up to --end-years (0, N, 2N, ...); "
                      "overrides --snap-years")
_ap.add_argument("--end-years", type=float, default=50000.0, dest="end_years", metavar="YR",
                 help="total simulation time [years] used with --report-every-years (default 50000)")
_ap.add_argument("--dt-nominal", type=float, default=1.0, metavar="YR")
_ap.add_argument("--dt-min", type=float, default=0.0001, metavar="YR")
_ap.add_argument("--dt-max", type=float, default=10.0, metavar="YR")
_ap.add_argument("--lag-buoyancy", action="store_true")
_args = _ap.parse_args()

P_RECHARGE = _args.p_recharge
P_DISCHARGE = _args.p_discharge
T_RECHARGE = _args.t_recharge + 273.15
Z_TOP = _args.z_top
Z_BOTTOM = _args.z_bottom

if _args.report_every_years is not None and _args.report_every_years > 0.0:
    _n = _args.report_every_years                    # regular reporting: 0, N, 2N, ... up to end
    _args.snap_years = [float(y) for y in np.arange(0.0, _args.end_years + 0.5 * _n, _n)]

schedule = [y * year_to_second for y in _args.snap_years]
# dt_init must not overshoot the first scheduled (report) time: a first step past several
# schedule points leaves the time manager trying to "correct" backward -> negative dt. Cap it.
_dt_init = _args.dt_nominal * year_to_second
if len(schedule) >= 2 and schedule[1] - schedule[0] > 0.0:
    _dt_init = min(_dt_init, schedule[1] - schedule[0])
time_manager = pp.TimeManager(
    schedule=schedule, dt_init=_dt_init,
    dt_min_max=(_args.dt_min * year_to_second, _args.dt_max * year_to_second),
    constant_dt=False, iter_max=13, iter_optimal_range=(3, 8),
    iter_relax_factors=(0.5, 1.5), recomp_factor=0.3, print_info=True)

HERE = os.path.dirname(os.path.abspath(__file__))
tag = _args.scheme + ("_mpfa" if _args.consistent else "") + (
    f"_{_args.grid_type}" if _args.grid_type else "")
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
    solver_params = model.default_nonlinear_criteria()
    runner = pp.ModelRunner(model, solver_params,
                            nonlinear_solver=geothermal_nonlinear_solver(solver_params))
    print("Elapsed time prepare simulation:", time.time() - tb)
    print("DoF:", model.equation_system.num_dofs(), " grid:", model.mdg)
    model.schur_complement_primary_equations = (
        pp.compositional_flow.get_primary_equations_cf(model))
    model.schur_complement_primary_variables = (
        pp.compositional_flow.get_primary_variables_cf(model))
    model.exporter.write_vtu()
    tb = time.time()
    runner.run()
    print("Elapsed time run:", time.time() - tb)
