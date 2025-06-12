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
tf = 400.0 * day
dt = 1.0 * day
time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    permeability=1.0e-14,
    porosity=0.1,
    thermal_conductivity=2.0 * 1.0e-6,
    density=2500.0,
    specific_heat_capacity=1000.0 * 1.0e-6,
)
material_constants = {"solid": solid_constants}
params = {
    "rediscretize_darcy_flux": True,
    "rediscretize_fourier_flux": True,
    "fractional_flow": True,
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation": False,
    "reduce_linear_system": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": 1.0e-8,
    "max_iterations": 100,
}


class SuperCriticalCO2FlowModel(FlowModel):

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        sd = model.mdg.subdomains()[0]
        phases = list(self.fluid.phases)
        components = list(self.fluid.components)
        # Integrated overall mass flux on all facets
        mn = self.equation_system.evaluate(self.darcy_flux(self.mdg.subdomains()))

        kappa_liq = self.relative_permeability(phases[0],self.mdg.subdomains())
        kappa_gas = self.relative_permeability(phases[1], self.mdg.subdomains())
        kappa_liq_val = self.equation_system.evaluate(kappa_liq)
        kappa_gas_val = self.equation_system.evaluate(kappa_gas)
        assert np.all(np.isclose(kappa_liq_val + kappa_gas_val, 1))

        rho_liq = phases[0].density(self.mdg.subdomains())
        rho_gas = phases[1].density(self.mdg.subdomains())
        rho_liq_val = self.equation_system.evaluate(rho_liq)
        rho_gas_val = self.equation_system.evaluate(rho_gas)
        assert np.all(np.isclose(rho_liq_val, 1000))
        assert np.all(np.isclose(rho_gas_val, 500))

        lambda_liq = self.phase_mobility(phases[0],self.mdg.subdomains())
        lambda_gas = self.phase_mobility(phases[1], self.mdg.subdomains())
        lambda_liq_val = self.equation_system.evaluate(lambda_liq)
        lambda_gas_val = self.equation_system.evaluate(lambda_gas)

        kappa = self.permeability(self.mdg.subdomains())
        kappa_val = self.equation_system.evaluate(kappa)
        lambda_t = self.total_mass_mobility(self.mdg.subdomains())
        lambda_t_val = self.equation_system.evaluate(lambda_t)
        assert np.all(np.isclose(lambda_t_val - (rho_liq_val * lambda_liq_val + rho_gas_val * lambda_gas_val), 0))

        f_liq = self.fractional_phase_mass_mobility(phases[0], self.mdg.subdomains())
        f_gas = self.fractional_phase_mass_mobility(phases[1], self.mdg.subdomains())
        f_liq_val = self.equation_system.evaluate(f_liq)
        f_gas_val = self.equation_system.evaluate(f_gas)
        assert np.all(np.isclose(f_liq_val + f_gas_val, 1))

        discr_liq = self.upward_phase_discretization(phases[0], self.mdg.subdomains())
        discr_gas = self.downward_phase_discretization(phases[1], self.mdg.subdomains())
        f_liq_upwind: pp.ad.Operator = discr_liq.upwind() @ f_liq  # well-defined fraction flow on facets
        f_gas_upwind: pp.ad.Operator = discr_gas.upwind() @ f_gas  # well-defined fraction flow on facets
        fu_liq_val = self.equation_system.evaluate(f_liq_upwind)
        fu_gas_val = self.equation_system.evaluate(f_gas_upwind)

        w_flux_unit = self.density_driven_flux(self.mdg.subdomains(), pp.ad.Scalar(1.0))
        w_flux_unit_val = self.equation_system.evaluate(w_flux_unit)
        z_val = self.equation_system.get_variable_values([self.equation_system.variables[3]], time_step_index=0)
        sg_val = self.equation_system.get_variable_values([self.equation_system.variables[4]], time_step_index=0)

        rho_gamma = phases[0].density([sd])
        rho_delta = phases[1].density([sd])
        w_flux_val = self.equation_system.evaluate(self.density_driven_flux([sd], rho_delta - rho_gamma))

        flux_buoyancy_c0 = self.component_buoyancy(components[0], self.mdg.subdomains())
        flux_buoyancy_c1 = self.component_buoyancy(components[1], self.mdg.subdomains())
        flux_c0 = self.component_flux(components[0], self.mdg.subdomains())
        flux_c1 = self.component_flux(components[1], self.mdg.subdomains())


        b_c0 = self.equation_system.evaluate(flux_buoyancy_c0)
        b_c1 = self.equation_system.evaluate(flux_buoyancy_c1)
        are_reciprocal_Q = np.all(np.isclose(b_c0 + b_c1, 0.0))
        print("buoyancy fluxes are reciprocal Q: ", are_reciprocal_Q)
        assert are_reciprocal_Q

        # flux_c0_val = self.equation_system.evaluate(flux_c0)
        flux_c1_val = self.equation_system.evaluate(flux_c1)

        external_bc_idx = sd.get_boundary_faces()
        internal_bc_idx = sd.get_internal_faces()
        print("w_flux_unit_val internal: ", w_flux_unit_val[internal_bc_idx])
        print("w_flux_val internal: ", w_flux_val[internal_bc_idx])
        print("fu_liq_val internal: ", fu_liq_val[internal_bc_idx])
        print("fu_gas_val internal: ", fu_gas_val[internal_bc_idx])
        print("b_c0 internal: ", b_c0[internal_bc_idx])
        print("b_c1 internal: ", b_c1[internal_bc_idx])
        print("b_c0 bc: ", b_c0[external_bc_idx])
        print("b_c1 bc: ", b_c1[external_bc_idx])
        print("flux_c1_val internal: ", flux_c1_val[internal_bc_idx])
        print("flux_c1_val bc: ", flux_c1_val[external_bc_idx])
        print("boundary integral  flux_bc: ", np.sum(b_c1[external_bc_idx] + flux_c1_val[external_bc_idx]))
        print("boundary integral m: ", np.sum(mn[external_bc_idx]))
        print("lambda_t : ", lambda_t_val)
        # print("kappa: ", kappa_val.reshape(sd.num_cells,3,3))
        print("sg: ", sg_val)
        print("z: ", z_val)
        print("volume integral sg: ", np.sum(sd.cell_volumes * sg_val))
        print("volume integral z: ", np.sum(sd.cell_volumes * z_val))
        assert np.isclose(np.sum(b_c1[external_bc_idx]), 0.0, atol=1.0e-5)
        # assert np.isclose(np.sum(mn[external_bc_idx]), 0.0, atol=1.0e-5)

        print("Number of iterations: ", self.nonlinear_solver_statistics.num_iteration)
        print("Time value: ", self.time_manager.time)
        print("Time index: ", self.time_manager.time_index)
        print("")

    def set_nonlinear_discretizations(self) -> None:
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

    def after_simulation(self):
        self.exporter.write_pvd()

    def before_nonlinear_iteration(self) -> None:
        self.rediscretize_fluxes()
        self.update_buoyancy_discretizations()
        self.update_flux_values()
        self.rediscretize()

    def update_discretization_parameters(self) -> None:
        super().update_discretization_parameters()
        if self.time_manager.time_index == 0:
            self.set_buoyancy_discretization_parameters()

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        return pp.ad.TpfaAd(self.darcy_keyword, subdomains)


model = SuperCriticalCO2FlowModel(params)

tb = time.time()
model.prepare_simulation()
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)

components = list(model.fluid.components)
grid = model.mdg.subdomains()[0]

# lambda_t = model.equation_system.evaluate(model.total_mass_mobility(model.mdg.subdomains()))
# w_flux_unit = model.density_driven_flux(model.mdg.subdomains(), pp.ad.Scalar(1.0)).value(model.equation_system)
# print("Initial lambda_t: ", lambda_t)
# print("Initial w_flux_unit: ", w_flux_unit[grid.get_internal_faces()])

l_xi = model.component_mass_mobility(components[0],model.mdg.subdomains()).value(model.equation_system)
l_eta = model.component_mass_mobility(components[1],model.mdg.subdomains()).value(model.equation_system)

rho_overall = model.fractionally_weighted_density(model.mdg.subdomains())
rho_overall_v = model.equation_system.evaluate(rho_overall)
# rho_xi = model.component_density(components[0],model.mdg.subdomains())
# rho_eta = model.component_density(components[1],model.mdg.subdomains())
# rho_xi_v = model.equation_system.evaluate(rho_xi)
# rho_eta_v = model.equation_system.evaluate(rho_eta)

f_xi = model.fractional_component_mass_mobility(components[0],model.mdg.subdomains()).value(model.equation_system)
f_eta = model.fractional_component_mass_mobility(components[1],model.mdg.subdomains()).value(model.equation_system)

# w_flux_xi_eta = model.density_driven_flux(model.mdg.subdomains(),rho_xi-rho_eta).value(model.equation_system)
w_flux_unit = model.density_driven_flux(model.mdg.subdomains(),pp.ad.Scalar(1.0)).value(model.equation_system)


# print geometry
model.exporter.write_vtu()

# update external solution
# xc = model.mdg.subdomains()[0].cell_centers.T
# z_v = np.where((xc[:,1] >= 0.0) & (xc[:,1] <= 1.0), 0.440636038257781, 0.559363961742219)
# model.equation_system.set_variable_values(z_v, ["z_CO2"], iterate_index = 0)
# model.equation_system.set_variable_values(z_v, ["s_gas"], iterate_index = 0)

tb = time.time()
pp.run_time_dependent_model(model, params)
te = time.time()
print("Elapsed time run_time_dependent_model: ", te - tb)
print("Total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid information: ", model.mdg)


l_xi = model.component_mass_mobility(components[0],model.mdg.subdomains()).value(model.equation_system)
l_eta = model.component_mass_mobility(components[1],model.mdg.subdomains()).value(model.equation_system)

rho_overall = model.fractionally_weighted_density(model.mdg.subdomains()).value(model.equation_system)
# rho_xi = model.component_density(components[0],model.mdg.subdomains()).value(model.equation_system)
# rho_eta = model.component_density(components[1],model.mdg.subdomains()).value(model.equation_system)

f_xi = model.fractional_component_mass_mobility(components[0],model.mdg.subdomains()).value(model.equation_system)
f_eta = model.fractional_component_mass_mobility(components[1],model.mdg.subdomains()).value(model.equation_system)

f_xi = model.fractional_component_mass_mobility(components[1],model.mdg.subdomains()).value(model.equation_system)
f_eta = model.fractional_component_mass_mobility(components[0],model.mdg.subdomains()).value(model.equation_system)

flux_buoyancy_c0 = model.component_buoyancy(components[0],model.mdg.subdomains())
flux_buoyancy_c1 = model.component_buoyancy(components[1],model.mdg.subdomains())
are_reciprocal_Q = np.all(np.isclose(model.equation_system.evaluate(flux_buoyancy_c0) + model.equation_system.evaluate(flux_buoyancy_c1),0.0))
print("buoyancy fluxes are reciprocal Q: ", are_reciprocal_Q)
assert are_reciprocal_Q


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
assert np.isclose(np.sum(mn[external_bc_idx]), 0.0, atol=1.0e-6)
