#!/usr/bin/env python
"""H2O-CO2 blow-down: a small 10 m x 10 m vertical box driven from CO2-liquid+water through the
three-phase (a+l+g) region to CO2-gas+water by producing at the bottom.

Scenario (see h2o_co2_obl/ for the OBL tables):
  IC  : hydrostatic p anchored 8.0 MPa at the top, T = 25 C, overall CO2 mass fraction z = 0.30
        -> aqueous + CO2-liquid (region 3) everywhere.
  BC  : BOTTOM edge Dirichlet p = 4.5 MPa (production) + isothermal walls T = 25 C; sides/top no-flow.
  As each cell's pressure sweeps down through Psat_CO2(25 C) = 6.43 MPa the CO2-liquid boils to gas
  (three-phase), so a boiling front sweeps bottom->top and the gas rises buoyantly.

Run (from this directory; all times in DAYS):
  python porepy_2d_solver_co2.py --case inject --scheme hu --cell-size 0.5 --tf 1
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

import porepy as pp

from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    Geometry,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import IC_Base
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import BCBase
from porepy.examples.geothermal_flow.model_configuration.CO2ModelConfiguration import (  # noqa: E501
    CO2FlowModel,
    CO2FractionalFlowModel,
)
from porepy.examples.geothermal_flow.model_configuration.flow_model_base import (  # noqa: E501
    geothermal_nonlinear_solver,
)
from porepy.examples.geothermal_flow.model_configuration.geothermal_export import DriesnerPhaseExport
from porepy.examples.geothermal_flow.obl_sampler import VTKSampler

to_Mega = 1.0e-6

# --- scenario constants ----------------------------------------------------------------------
L_DOMAIN = 10.0        # m, square side (vertical y is up; gravity -y)
T_INIT = 298.15        # K (25 C), isothermal (both cases)
Z_CO2 = 0.30           # CO2 mass fraction (charged IC for blowdown / injected stream for inject)
G_ACC = 9.81           # m/s2

# blowdown case (pre-charged water+CO2 block, produced at the bottom):
P_TOP = 8.0            # MPa, IC hydrostatic anchor at the top
P_BOTTOM = 4.5         # MPa, bottom Dirichlet production pressure
RHO_REF = 900.0        # kg/m3, charged-mixture column density for the hydrostatic IC

# inject case (water IC; CO2 injected at the bottom, produced at the top):
P_INLET = 8.0          # MPa, bottom inlet Dirichlet pressure
P_OUTLET = 4.5         # MPa, top outlet Dirichlet pressure
RHO_WATER = 1000.0     # kg/m3, pure-water column density for the hydrostatic IC

# --- --md fracture network (interior fractures; 4 conductive + 3 barriers) --------------------
# (x0,y0,x1,y1) in metres. Conductive first -> frac_num 0..3; barriers after -> frac_num 4..6.
_CONDUCTIVE = [                                # k = 100 x k_rock, steep (65-76 deg), disjoint bands
    (1.70, 1.74, 3.40, 5.36),
    (4.57, 3.81, 5.53, 7.69),
    (6.80, 6.10, 8.30, 9.80),
    (8.49, 5.20, 9.11, 9.00),
]
_BARRIERS = [                                  # k = 1/100 x k_rock, ~perpendicular to the diagonal flow
    (0.80, 4.80, 4.30, 2.30),
    (3.30, 7.00, 6.80, 4.50),
    (5.80, 9.20, 9.30, 6.70),
]
_N_COND = len(_CONDUCTIVE)                     # frac_num < _N_COND -> conductive, else barrier
_COND_FACTOR = 100.0                           # conductive permeability / k_rock (tangential & normal)
_BARR_FACTOR = 0.01                            # barrier permeability / k_rock (tangential & normal)
_FRAC_APERTURE = 0.1                           # m, fracture aperture (tunable modelling parameter)

_SCHEME_CONFIG = {
    "hu":    dict(fractional_flow=False, buoyancy_upwinding="hybrid"),
    "hu-mw": dict(fractional_flow=True,  buoyancy_upwinding="hybrid"),
}


# --- geometry: 10 m x 10 m vertical Cartesian box --------------------------------------------
class Geometry10x10(Geometry):
    """10 m x 10 m Cartesian box. y is vertical (north = top = max-y, south = bottom = min-y).
    get_inlet_outlet_sides returns (top, bottom)."""

    def set_domain(self) -> None:
        L = self.units.convert_units(L_DOMAIN, "m")
        self._domain = pp.Domain({"xmax": L, "ymax": L})

    def grid_type(self):
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cs = self.units.convert_units(self.params.get("cell_size", 0.5), "m")
        return {"cell_size": cs}

    def get_inlet_outlet_sides(self, sd):
        sides = self.domain_boundary_sides(sd)
        top = np.where(sides.north)[0]
        bottom = np.where(sides.south)[0]
        return top, bottom

    def _fracture_lines(self):
        """Conductive (frac_num 0..3) then barriers (4..6) as pp.LineFracture, in model units."""
        cu = self.units.convert_units
        lines = []
        for x0, y0, x1, y1 in _CONDUCTIVE + _BARRIERS:
            lines.append(pp.LineFracture(np.array([[cu(x0, "m"), cu(x1, "m")],
                                                   [cu(y0, "m"), cu(y1, "m")]])))
        return lines

    def set_geometry(self) -> None:
        if not self.params.get("md", False):
            return super().set_geometry()                 # fixed-dim Cartesian box
        self.set_domain()
        network = pp.create_fracture_network(self._fracture_lines(), self._domain)
        h = self.units.convert_units(self.params.get("cell_size", 0.5), "m")
        mesh_args = {"cell_size": h, "cell_size_boundary": h,
                     "cell_size_fracture": 0.5 * h, "cell_size_min": 0.5 * h}
        gmsh_file = Path(tempfile.gettempdir()) / f"co2_md_gmsh_{os.getpid()}.msh"
        from _quad_mesh import build_recombined_mdg      # gmsh triangles -> recombined quads
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*outside the domain boundary.*")
            self.mdg = build_recombined_mdg(mesh_args, network, gmsh_file)
        self.nd = self.mdg.dim_max()
        pp.set_local_coordinate_projections(self.mdg)


# --- initial conditions ----------------------------------------------------------------------
class _ICBaseCO2(IC_Base):
    """Shared CO2 IC: isothermal 25 C; enthalpy + the secondary variables (saturations, partial
    fractions) seeded from the ptz OBL table. Subclasses set the pressure profile and composition.
    `ic_salinity` (the z-axis fed to every ptz lookup) must equal the overall CO2 fraction."""

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return np.full(sd.num_cells, T_INIT)

    def _sampled_ptz(self, sd: pp.Grid):
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z = self.ic_salinity(sd)
        self.obl_sampler_ptz.sample_at(np.array((z, t, p)).T)
        return self.obl_sampler_ptz.sampled_could.point_data

    def initial_condition(self) -> None:
        super().initial_condition()                    # seeds p, z, h primary variables
        liq = next(ph for ph in self.fluid.phases if ph.name == "liq")
        gas = next(ph for ph in self.fluid.phases if ph.name == "gas")
        co2l = next(ph for ph in self.fluid.phases if ph.name == "co2l")
        co2 = self.fluid.components[1]
        for sd in self.mdg.subdomains():
            d = self._sampled_ptz(sd)
            if self.has_independent_saturation(gas):
                self.equation_system.set_variable_values(
                    np.clip(d["S_v"], 0.0, 1.0), [gas.saturation([sd])], 0, 0)
            if self.has_independent_saturation(co2l):
                self.equation_system.set_variable_values(
                    np.clip(d["S_h"], 0.0, 1.0), [co2l.saturation([sd])], 0, 0)
            for ph, field in ((liq, "Xl"), (gas, "Xv"), (co2l, "Xh")):
                if self.has_independent_partial_fraction(co2, ph):
                    self.equation_system.set_variable_values(
                        np.clip(d[field], 0.0, 1.0), [ph.partial_fraction_of[co2]([sd])], 0, 0)


class ICblowdown(_ICBaseCO2):
    """Pre-charged block: hydrostatic p anchored 8 MPa at the top, uniform CO2 fraction 0.30 ->
    aqueous + CO2-liquid everywhere (region 3)."""

    def ic_salinity(self, sd: pp.Grid) -> np.ndarray:
        return np.full(sd.num_cells, Z_CO2)

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        L = self.units.convert_units(L_DOMAIN, "m")
        depth = L - sd.cell_centers[1]
        return P_TOP + RHO_REF * G_ACC * depth / 1.0e6

    def ic_values_overall_fraction(self, component, sd: pp.Grid) -> np.ndarray:
        if component == self.fluid.components[1]:
            return np.full(sd.num_cells, Z_CO2)
        return np.full(sd.num_cells, 1.0 - Z_CO2)


class ICinject(_ICBaseCO2):
    """Pure water in hydrostatic equilibrium, anchored at the top-outlet pressure (4.5 MPa);
    isothermal 25 C, z_CO2 = 0 -> single-phase aqueous (region 1). CO2 arrives via the inlet BC."""

    def ic_salinity(self, sd: pp.Grid) -> np.ndarray:
        return np.zeros(sd.num_cells)

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        L = self.units.convert_units(L_DOMAIN, "m")
        depth = L - sd.cell_centers[1]
        return P_OUTLET + RHO_WATER * G_ACC * depth / 1.0e6

    def ic_values_overall_fraction(self, component, sd: pp.Grid) -> np.ndarray:
        if component == self.fluid.components[1]:
            return np.zeros(sd.num_cells)              # pure water
        return np.ones(sd.num_cells)


# --- boundary conditions (isothermal walls on the Fourier flux for both cases) ---------------
class BCblowdown(BCBase):
    """BOTTOM edge Dirichlet pressure (production, 4.5 MPa); top/sides no-flow. Isothermal walls."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        _, bottom = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, bottom, "dir")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self.domain_boundary_sides(sd).all_bf, "dir")

    def bc_salinity(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.full(bg.num_cells, Z_CO2)

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.full(bg.num_cells, P_BOTTOM)

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.full(bg.num_cells, T_INIT)


class BCinject(BCBase):
    """Diagonal flow-through. INLET patch = left wall (x=0), y in [0, 0.5] m: Dirichlet p = 8 MPa
    injecting z_CO2 = 0.30. OUTLET patch = right wall (x=10), y in [9.5, 10] m: Dirichlet p = 4.5 MPa.
    Everything else no-flow. Isothermal walls (Fourier Dirichlet on all faces, 25 C)."""

    def _inlet_outlet_patches(self, sd):
        """(inlet, outlet) face/cell indices: inlet = west & y<=0.05 L, outlet = east & y>=0.95 L."""
        sides = self.domain_boundary_sides(sd)
        coords = sd.face_centers if isinstance(sd, pp.Grid) else sd.cell_centers
        y = coords[1]
        Ly = self.units.convert_units(L_DOMAIN, "m")
        inlet = np.where(sides.west & (y <= 0.05 * Ly))[0]       # left, bottom 0.5 m
        outlet = np.where(sides.east & (y >= 0.95 * Ly))[0]      # right, top 0.5 m
        return inlet, outlet

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        inlet, outlet = self._inlet_outlet_patches(sd)
        return pp.BoundaryCondition(sd, np.concatenate([inlet, outlet]), "dir")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self.domain_boundary_sides(sd).all_bf, "dir")

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        inlet, outlet = self._inlet_outlet_patches(bg)
        p = np.zeros(bg.num_cells)
        p[inlet] = P_INLET
        p[outlet] = P_OUTLET
        return p

    def bc_salinity(self, bg: pp.BoundaryGrid) -> np.ndarray:
        # CO2 injected only at the inlet patch (0.30); the outlet is outflow (value unused there).
        inlet, _ = self._inlet_outlet_patches(bg)
        z = np.zeros(bg.num_cells)
        z[inlet] = Z_CO2
        return z

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.full(bg.num_cells, T_INIT)


# --- CLI ------------------------------------------------------------------------------------
_ap = argparse.ArgumentParser(description="H2O-CO2 crossing the three-phase region (2 cases).")
_ap.add_argument("--case", choices=["inject", "blowdown"], default="inject",
                 help="inject: water IC, CO2 injected bottom 8 MPa -> top outlet 4.5 MPa (flow-through); "
                      "blowdown: pre-charged block depressurized at the bottom")
_ap.add_argument("--scheme", choices=["hu", "hu-mw"], default="hu")
_ap.add_argument("--cell-size", type=float, default=0.5, help="cell size [m] (0.5 -> 20x20)")
_ap.add_argument("--tf", type=float, default=1.0, help="final time [days]")
_ap.add_argument("--dt-init", type=float, default=1.0e-5, help="initial time step [days] (~0.86 s)")
_ap.add_argument("--dt-min", type=float, default=1.0e-7, help="min dt [days] (~8.6e-3 s)")
_ap.add_argument("--n-snap", type=int, default=10, help="number of export snapshots")
_ap.add_argument("--no-gravity", action="store_true",
                 help="disable gravity (g = 0): no buoyancy, and the IC pressure becomes uniform")
_ap.add_argument("--md", action="store_true",
                 help="mixed-dimensional: 4 conductive (100 k_rock) + 3 barrier (1/100 k_rock) interior "
                      "fractures, gmsh-meshed and recombined to quads")
_ap.add_argument("--consistent", action="store_true",
                 help="use MPFA (consistent on non-K-orthogonal grids); default is TPFA. "
                      "Recommended with --md (recombined quads + inclined fractures)")
_ap.add_argument("--tol", type=float, default=1.0e-4)
_ap.add_argument("--max-iter", type=int, default=20)
_args = _ap.parse_args()

if _args.no_gravity:
    G_ACC = 0.0                                # zeroes the hydrostatic IC term (uniform p) too

if _args.md and not _args.consistent:
    print("NOTE: --md uses recombined quads + inclined fractures (not K-orthogonal); TPFA is "
          "inconsistent here -- consider pairing with --consistent (MPFA).")

DAY = 86400.0                                  # all --tf / --dt-* inputs are in DAYS -> seconds here

solid_constants = pp.SolidConstants(
    permeability=1e-15,                        # m^2 (~1 mD)
    porosity=0.2,
    thermal_conductivity=2.0 * to_Mega,
    density=2700.0,
    specific_heat_capacity=880.0 * to_Mega,
)
material_constants = {"solid": solid_constants}

tf = _args.tf * DAY                             # days -> seconds (TimeManager works in seconds)
dt_init = _args.dt_init * DAY
dt_min = _args.dt_min * DAY
snap = [tf * k / _args.n_snap for k in range(1, _args.n_snap + 1)]
time_manager = pp.TimeManager(
    schedule=[0.0] + snap,
    dt_init=dt_init,
    dt_min_max=(dt_min, max(tf / 10.0, 2.0 * dt_init)),
    constant_dt=False,
    iter_optimal_range=(3, 8),
    iter_relax_factors=(0.25, 1.5),
    recomp_factor=0.3,
)
times_to_export = [0.0] + list(snap)            # include t=0 so the IC is exported as frame 0

params = {
    "folder_name": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "visualization_co2_%s_%s%s%s%s" % (
            _args.case, _args.scheme,
            "_md" if _args.md else "",
            "_mpfa" if _args.consistent else "",
            "_nograv" if _args.no_gravity else "")),
    "enable_buoyancy_effects": not _args.no_gravity,
    "material_constants": material_constants,
    "time_manager": time_manager,
    "times_to_export": times_to_export,
    "grid_type": "cartesian",
    "cell_size": _args.cell_size,
    "use_petsc": True,
    "petsc_preconditioner": "cpr",
    "cpr_rtol": 1.0e-5,
    "cpr_maxit": 400,
    "cpr_accuracy_tol": 1.0e-3,
    "step_control_method": "LS",
    "residual_scale_current_dt": True,
    "slave_eliminated_secondaries": True,
    "consistent_discretization": _args.consistent,   # TPFA by default; --consistent -> MPFA
    "md": _args.md,
}
params.update(_SCHEME_CONFIG[_args.scheme])

FlowModel = CO2FractionalFlowModel if params["fractional_flow"] else CO2FlowModel

BC = BCinject if _args.case == "inject" else BCblowdown
IC = ICinject if _args.case == "inject" else ICblowdown


class GeothermalCO2FlowModel(DriesnerPhaseExport, Geometry10x10, BC, IC, FlowModel):
    # -- --md fracture permeability / aperture (interior conductive + barrier fractures) -------
    def _frac_factor(self, sd: pp.Grid) -> float:
        """k / k_rock for a subdomain: 1 matrix; conductive 1D (frac_num<_N_COND) = 2; barrier 1D = 1/2;
        0D intersection = conductive grade."""
        dmax = self.mdg.dim_max()
        if sd.dim == dmax:
            return 1.0
        if sd.dim == dmax - 1:
            return _COND_FACTOR if int(sd.frac_num) < _N_COND else _BARR_FACTOR
        return _COND_FACTOR                                 # 0D intersection

    def _is_barrier(self, sd: pp.Grid) -> bool:
        return (sd.dim == self.mdg.dim_max() - 1 and sd.num_cells > 0
                and int(sd.frac_num) >= _N_COND)

    def grid_aperture(self, grid: pp.Grid) -> np.ndarray:
        if self.params.get("md", False) and grid.dim < self.mdg.dim_max():
            return np.full(grid.num_cells, self.units.convert_units(_FRAC_APERTURE, "m"))
        return super().grid_aperture(grid)

    def _absolute_permeability(self, subdomains):
        vals = []
        for sd in subdomains:
            k = np.full(sd.num_cells, self.solid.permeability)
            if sd.dim < self.mdg.dim_max():
                k *= self._frac_factor(sd)
            vals.append(k)
        return np.concatenate(vals) if vals else np.zeros(0)

    def permeability(self, subdomains):
        if not self.params.get("md", False):
            return super().permeability(subdomains)
        perm = pp.wrap_as_dense_ad_array(self._absolute_permeability(subdomains), name="permeability")
        if pp.compositional_flow.is_fractional_flow(self):
            op = self.isotropic_second_order_tensor(
                subdomains, self.total_mass_mobility(subdomains) * perm)
            op.set_name("diffusive_tensor_darcy")
        else:
            op = self.isotropic_second_order_tensor(subdomains, perm)
        return op

    def normal_permeability(self, interfaces):
        # rock-only (factor * k_rock) normal k -- NOT mass-mobility weighted (avoids the MD
        # double-mobility blow-up; same fix as the brine --md solvers).
        if not self.params.get("md", False):
            return super().normal_permeability(interfaces)
        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        kn_sd = pp.wrap_as_dense_ad_array(
            np.concatenate([np.full(sd.num_cells, self._frac_factor(sd) * self.solid.permeability)
                            for sd in subdomains]) if subdomains else np.zeros(0),
            name="normal_k")
        kn = projection.secondary_to_mortar_avg() @ kn_sd
        kn.set_name("normal_permeability")
        return kn

    def data_to_export(self):
        data = super().data_to_export()
        if self.params.get("md", False):
            for sd in self.mdg.subdomains():
                tag = np.zeros(sd.num_cells) if sd.dim == self.mdg.dim_max() else np.full(
                    sd.num_cells, 1.0 if self._is_barrier(sd) else 2.0)   # 0 rock, 1 barrier, 2 conductive
                data.append((sd, "material", tag))
        return data

    # -- outlet CO2 breakthrough diagnostic ----------------------------------------------------
    def _outlet_faces(self, sd):
        """Matrix outlet boundary faces: inject -> the top-east outlet patch; blowdown -> the
        bottom production edge."""
        if hasattr(self, "_inlet_outlet_patches"):          # inject BC: (inlet, outlet) patches
            return self._inlet_outlet_patches(sd)[1]
        return self.get_inlet_outlet_sides(sd)[1]           # blowdown BC: production = bottom

    def _outlet_co2(self):
        """(area-averaged overall CO2 mass fraction, CO2 advective mass-outflow rate [kg/s]) at the
        outlet, from the converged iterate. darcy_flux is the total, face-integrated volumetric flux
        [m^3/s]; outflow sign + adjacent (upwind) cell come from the grid helper, so no face-area
        factor is applied to the rate. z_CO2 is a mass fraction; fluid.density is sum_k S_k rho_k."""
        sd = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        outlet = self._outlet_faces(sd)
        if outlet.size == 0:
            return 0.0, 0.0
        es = self.equation_system
        co2 = self.fluid.components[1]
        z = np.asarray(es.evaluate(co2.fraction([sd])), dtype=float)
        rho = np.asarray(es.evaluate(self.fluid.density([sd])), dtype=float)
        q = np.asarray(es.evaluate(self.darcy_flux([sd])), dtype=float)    # [m^3/s] per face
        sign, adj = sd.signs_and_cells_of_boundary_faces(outlet)           # +1 = out of the domain
        areas = sd.face_areas[outlet]
        z_avg = float(np.sum(z[adj] * areas) / np.sum(areas))              # area-averaged z_CO2 [-]
        co2_rate = float(np.sum(z[adj] * rho[adj] * (sign * q[outlet])))   # [kg/s], >0 leaving
        return z_avg, co2_rate

    def write_pvd_and_vtu(self):
        """Normal vtu/pvd snapshot + append one breakthrough row per reported export time."""
        super().write_pvd_and_vtu()
        if not hasattr(self, "_bt_rows"):
            self._bt_rows, self._bt_cum = [], 0.0
        t_s = float(self.time_manager.time)
        z_avg, rate = self._outlet_co2()
        if self._bt_rows:                                   # trapezoid cumulative CO2 out [kg]
            t_prev, rate_prev = self._bt_rows[-1][1], self._bt_rows[-1][3]
            self._bt_cum += 0.5 * (rate + rate_prev) * (t_s - t_prev)
        self._bt_rows.append([t_s / DAY, t_s, z_avg, rate, self._bt_cum])
        with open(os.path.join(self.params["folder_name"], "outlet_co2_breakthrough.csv"),
                  "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time_days", "time_s", "outlet_z_co2_avg",
                        "co2_mass_outflow_rate_kg_s", "co2_cumulative_out_kg"])
            w.writerows(self._bt_rows)


model = GeothermalCO2FlowModel(params)

# --- OBL samplers: the compositional H2O-CO2 tables (phz + ptz) ------------------------------
_CO2_TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h2o_co2_obl")


def _attach_samplers(model) -> None:
    phz = VTKSampler(os.path.join(_CO2_TABLE_DIR, "h2o_co2_xph.vtr"))
    phz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, h, p)
    model.obl_sampler = phz
    ptz = VTKSampler(os.path.join(_CO2_TABLE_DIR, "h2o_co2_xpt.vtr"))
    ptz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, T, p)
    ptz.translation_factors = (0.0, -273.15, 0.0)            # model T in K -> table T in degC
    model.obl_sampler_ptz = ptz


_attach_samplers(model)

# --- run ------------------------------------------------------------------------------------
tb = time.time()
solver_params = model.default_nonlinear_criteria(tol=_args.tol, max_iterations=_args.max_iter)
runner = pp.ModelRunner(model, solver_params,
                        nonlinear_solver=geothermal_nonlinear_solver(solver_params))
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)

model.schur_complement_primary_equations = pp.compositional_flow.get_primary_equations_cf(model)
model.schur_complement_primary_variables = pp.compositional_flow.get_primary_variables_cf(model)

# The initial condition is exported as frame 0 inside prepare_simulation (save_data_time_step at
# t=0, now that 0.0 is in times_to_export), so no separate static write is needed here.
tb = time.time()
runner.run()
te = time.time()
print("Elapsed time run: ", te - tb)
