"""Example implementing a multi-phase multi component flow of H2O-NaCl using a passive
tracer constitutive model. In its full extension, the model is linear.

The model relies on pressure (P), specific fluid mixture enthalpy (H), and NaCl overall
composition (z_CO2) as primary variables.

    Note:
        - The overall mass flux is a solenoidal vector field and constant in time;
        - Partial fractions X_CO2_gas are equal to 1.0;
        - Partial fractions X_CO2_liquid are equal to 0.0;
        - Partial fractions X_H2O_gas are equal to 0.0;
        - Partial fractions X_H2O_liquid are equal to 1.0;
        - Vapor phase saturation S_gas are equal to z_CO2;
        - Liquid phase saturation S_water are equal to 1-z_CO2;
        - Temperature is related to H with the residual equation (T - 250.0 H = 0);
        - Relative permeability is linear;

The computations are qualitatively checked by asserting global conservation of the
solenoidal vector field.

"""

from __future__ import annotations
import os
os.environ["NUMBA_DISABLE_JIT"] = str(0)
import time

import numpy as np

import porepy as pp
from porepy.examples.geothermal_flow.model_configuration.SuperCriticalCO2ModelConfiguration import (
    SuperCriticalCO2FlowModel as FlowModel,
)

day = 86400
t_scale = 1.0
tf = 0.34 * day
dt = 0.01 * day
time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    permeability=3.0e-12,
    porosity=0.35,
    thermal_conductivity=2.0 * 1.0e-6,
    density=2500.0,
    specific_heat_capacity=1000.0 * 1.0e-6,
)
material_constants = {"solid": solid_constants}
params = {
    "rediscretize_darcy_flux": True,
    "fractional_flow": True,
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation": False,
    "reduce_linear_system": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": 1.0e-8,
    "max_iterations": 50,
}


class SuperCriticalCO2FlowModel(FlowModel):

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()
        print("Number of iterations: ", self.nonlinear_solver_statistics.num_iteration)
        print("Time value: ", self.time_manager.time)
        print("Time index: ", self.time_manager.time_index)
        print("")

    def after_simulation(self):
        self.exporter.write_pvd()


model = SuperCriticalCO2FlowModel(params)

tb = time.time()
model.prepare_simulation()
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)

components = list(model.fluid.components)

l_xi = model.component_mass_mobility(components[1],model.mdg.subdomains()).value(model.equation_system)
l_eta = model.component_mass_mobility(components[0],model.mdg.subdomains()).value(model.equation_system)

rho_overall = model.fractionally_weighted_density(model.mdg.subdomains()).value(model.equation_system)
rho_xi = model.component_density(components[1],model.mdg.subdomains()).value(model.equation_system)
rho_eta = model.component_density(components[0],model.mdg.subdomains()).value(model.equation_system)

f_xi = model.fractional_component_mass_mobility(components[1],model.mdg.subdomains()).value(model.equation_system)
f_eta = model.fractional_component_mass_mobility(components[0],model.mdg.subdomains()).value(model.equation_system)

# flux_c1 = model.component_flux(components[1],model.mdg.subdomains()).value(model.equation_system)
# flux_buoyancy_c1 = model.component_buoyancy(components[1],model.mdg.subdomains()).value(model.equation_system)

# print geometry
model.exporter.write_vtu()
tb = time.time()
pp.run_time_dependent_model(model, params)
te = time.time()
print("Elapsed time run_time_dependent_model: ", te - tb)
print("Total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid information: ", model.mdg)


f_xi = model.fractional_component_mass_mobility(components[1],model.mdg.subdomains()).value(model.equation_system)
f_eta = model.fractional_component_mass_mobility(components[0],model.mdg.subdomains()).value(model.equation_system)

flux_c1 = model.component_flux(components[1],model.mdg.subdomains()).value(model.equation_system)
# flux_buoyancy_c0 = model.component_buoyancy(components[0],model.mdg.subdomains()).value(model.equation_system)
flux_buoyancy_c1 = model.component_buoyancy(components[1],model.mdg.subdomains()).value(model.equation_system)

# Retrieve the grid and boundary information
grid = model.mdg.subdomains()[0]
bc_sides = model.domain_boundary_sides(grid)

# Integrated overall mass flux on all facets
mn = model.darcy_flux(model.mdg.subdomains()).value(model.equation_system)

inlet_idx, outlet_idx = model.get_inlet_outlet_sides(model.mdg.subdomains()[0])
print("Inflow values : ", mn[inlet_idx])
print("Outflow values : ", mn[outlet_idx])

# Check conservation of overall mass across boundaries
external_bc_idx = bc_sides.all_bf
assert np.isclose(np.sum(mn[external_bc_idx]), 0.0, atol=1.0e-10)
