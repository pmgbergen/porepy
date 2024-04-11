"""Example implementing a multi-phase multi component flow of H2O-NaCl using
the Driesner correlations as constitutive laws.

This model uses pressure, specific fluid mixture enthalpy and NaCl overall fraction as
primary variables.

No equilibrium calculations included.

Ergo, the user must close the model to provide expressions for saturations, partial
fractions and temperature, depending on primary variables.

Note:
    With some additional work, it is straight forward to implement a model without
    h as the primary variable, but T.

    What needs to change is:

    1. Overwrite
       porepy.models.compositional_flow.VariablesCF
       mixin s.t. it does not create a h variable.
    2. Modify accumulation term in
       porepy.models.compositional_flow.TotalEnergyBalanceEquation_h
       to use T, not h.
    3. H20_NaCl_brine.dependencies_of_phase_properties: Use T instead of h.

"""

from __future__ import annotations

import numpy as np
import time
import porepy as pp
import porepy.composite as ppc
import TracerConstitutiveDescription


day = 86400
t_scale = 0.1
time_manager = pp.TimeManager(
    schedule=list(n * day * t_scale for n in range(3)),
    dt_init=1.0 * day * t_scale,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

tracer_like_setting_q = True
enable_checks_q = False

solid_constants = pp.SolidConstants(
    {"permeability": 9.869233e-14, "porosity": 0.2, "thermal_conductivity": 1.92}
)
material_constants = {"solid": solid_constants}
params = {
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation": False,
    "reduce_linear_system_q": False,
    "nl_convergence_tol": 1.0e-3,
    "max_iterations": 100,
}


if tracer_like_setting_q:
    from TracerModelConfiguration import TracerFlowModel as FlowModel
else:
    from DriesnerBrineOBL import DriesnerBrineOBL
    from DriesnerModelConfiguration import DriesnerBrineFlowModel as FlowModel

class GeothermalFlowModel(FlowModel):

    def _export(self):
        if hasattr(self, "exporter"):
            self.exporter.write_vtu(self.primary_variable_names(), time_dependent=True)

    def after_simulation(self):
        self.exporter.write_pvd()

    def relative_permeability(self, saturation: pp.ad.Operator) -> pp.ad.Operator:
        # return saturation**2
        return saturation

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
        return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

if tracer_like_setting_q:
    model = GeothermalFlowModel(params)
else:
    model = GeothermalFlowModel(params)
    file_name = "binary_files/PHX_l0_with_gradients.vtk"
    brine_obl = DriesnerBrineOBL(file_name)
    brine_obl.conversion_factors = (1.0,1.0e-3,1.0e-5) #(z,h,p)
    model.obl = brine_obl

tb = time.time()
model.prepare_simulation()
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)

# # print geometry
# model.exporter.write_vtu()
tb = time.time()
pp.run_time_dependent_model(model, params)
te = time.time()
print("Elapsed time run_time_dependent_model: ", te - tb)
print("Total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid information: ", model.mdg)

if enable_checks_q:
    print("CHECKS -----------------")
    bgs = model.mdg.boundaries()
    sds = model.mdg.subdomains()
    eqs = model.equation_system
    NaCl = model.fluid_mixture._components[1]
    gas = model.fluid_mixture._phases[1]
    all, east, west, north, south, top, bottom = model.domain_boundary_sides(sds[0])

    for name, equ in model.equation_system.equations.items():
        print(f"Residual {name}: ", np.linalg.norm(equ.value(eqs)))

    print("pressure: ", model.pressure(sds).value(eqs))
    print("temperature: ", model.temperature(sds).value(eqs))
    print("fluid enthalpy: ", model.enthalpy(sds).value(eqs))
    print("Salt fraction: ", NaCl.fraction(sds).value(eqs))
    print("Gas saturation:", gas.saturation(sds).value(eqs))

    fluxes = model.darcy_flux(model.mdg.subdomains()).value(model.equation_system)
    all, east, west, north, south, top, bottom = model.domain_boundary_sides(model.mdg.subdomains()[0])
    print("fluxes: ", fluxes)
    print("fluxes[north]: ", fluxes[north])
    print("fluxes[south]: ", fluxes[south])
    print("fluxes[east]: ", fluxes[east])
    print("fluxes[west]: ", fluxes[west])

    east_ = east[all]
    bc_flux = model.darcy_flux(bgs).value(model.equation_system)
    print("bc fluxes[north]: ", bc_flux[north[all]])
    print("bc fluxes[south]: ", bc_flux[south[all]])
    print("bc fluxes[east]: ", bc_flux[east[all]])
    print("bc fluxes[west]: ", bc_flux[west[all]])


    NaCl_fluxes = model.fluid_flux_for_component(NaCl,sds).value(model.equation_system)
    print("NaCl_fluxes[north]: ", NaCl_fluxes[north])
    print("NaCl_fluxes[south]: ", NaCl_fluxes[south])
    print("NaCl_fluxes[east]: ", NaCl_fluxes[east])
    print("NaCl_fluxes[west]: ", NaCl_fluxes[west])

    A_name = "density"
    A_t = model.fluid_mixture.density(sds)
    A_tp = A_t.previous_timestep()
    print(f"{A_name}_t after sim: ", A_t.value(eqs))
    print(f"{A_name}_t prev after sim: ", A_tp.value(eqs))
    print(f"delta {A_name}_t after sim: ", np.abs((A_t - A_tp).value(eqs)))
    A_phi = A_t * model.porosity(sds)
    print(f"{A_name} phi after sim:", A_phi.value(eqs))
    print(f"delta {A_name} phi after sim:", np.abs((A_phi - A_phi.previous_timestep()).value(eqs)))
    int_A_phi = model.volume_integral(A_phi, sds, dim=1)
    print(f"int {A_name} phi after sim:", int_A_phi.value(eqs))
    dt_accum = pp.ad.time_derivatives.dt(int_A_phi, model.ad_time_step)
    print("dt accum manual after sim:", ((int_A_phi - int_A_phi.previous_timestep()) / model.ad_time_step).value(eqs))
    print("dt accum after sim:", dt_accum.value(eqs))
    print(f"gas rho after sim:", gas.density.subdomain_values)
    gas.density.subdomain_values = gas.density.subdomain_values * 4.
    print(f"gas rho after change:", gas.density.subdomain_values)
    print(f"{A_name}_t after change: ", A_t.value(eqs))
    print(f"{A_name}_t prev after change: ", A_tp.value(eqs))
    print(f"delta {A_name}_t after change: ", np.abs((A_t - A_tp).value(eqs)))
    print(f"{A_name} phi after change:", A_phi.value(eqs))
    print(f"delta {A_name} phi after change:", np.abs((A_phi - A_phi.previous_timestep()).value(eqs)))
    print(f"int {A_name} phi after change:", int_A_phi.value(eqs))
    print("dt accum manual after change:", ((int_A_phi - int_A_phi.previous_timestep()) / model.ad_time_step).value(eqs))
    print("dt accum after change:", dt_accum.value(eqs))

    # pp.plot_grid(model.mdg, NaCl.fraction(sds).name, figsize=(10, 8), plot_2d=True)
    print("end")
