"""Example implementing a multi-phase multi component flow of H2O-NaCl using Driesner
correlations.

The model relies on pressure (P), specific fluid mixture enthalpy (H), and NaCl overall
composition (z_NaCl) as primary variables.

Equilibrium calculations are included in the correlations. As a result, they contain
expressions for saturation, partial fractions, and temperature based on primary variables.

The correlations are interpolated with VTK using a standalone object (VTKSampler). Two
instances of that object provide functions and their gradients within the product spaces
(z_NaCl, xi, P) in R^3, where xi in {H,T}.

"""

from __future__ import annotations

import time

import numpy as np

import porepy as pp
from porepy.examples.geothermal_flow.model_configuration.DriesnerModelConfiguration import (
    DriesnerBrineFlowModel as FlowModel,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

day = 86400
tf = 0.00005 * day  # final time
dt = 0.000025 * day  # time step size
time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {
        "permeability": 5.0e-14,
        "porosity": 0.1,
        "thermal_conductivity": 1.8,
        "density": 2650.0,
        "specific_heat_capacity": 1000.0,
    }
)
material_constants = {"solid": solid_constants}
params = {
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation": False,
    "reduce_linear_system_q": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": 1.0e-3,
    "max_iterations": 50,
}


class GeothermalFlowModel(FlowModel):

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()
        print("Number of iterations: ", self.nonlinear_solver_statistics.num_iteration)
        print("Time value: ", self.time_manager.time)
        print("Time index: ", self.time_manager.time_index)
        print("")

    def after_simulation(self):
        self.exporter.write_pvd()


# Instance of the computational model
model = GeothermalFlowModel(params)

parametric_space_ref_level = 2
file_name_prefix = (
    "src/porepy/examples/geothermal_flow/"
    + "model_configuration/constitutive_description/driesner_vtk_files/"
)
file_name_phz = (
    file_name_prefix + "XHP_l" + str(parametric_space_ref_level) + "_modified.vtk"
)
file_name_ptz = (
    file_name_prefix + "XTP_l" + str(parametric_space_ref_level) + "_modified.vtk"
)

brine_sampler_phz = VTKSampler(file_name_phz)
brine_sampler_phz.conversion_factors = (1.0, 1.0e-3, 1.0e-5)  # (z,h,p)
model.vtk_sampler = brine_sampler_phz

brine_sampler_ptz = VTKSampler(file_name_ptz)
brine_sampler_ptz.conversion_factors = (1.0, 1.0, 1.0e-5)  # (z,t,p)
brine_sampler_ptz.translation_factors = (0.0, -273.15, 0.0)  # (z,t,p)
model.vtk_sampler_ptz = brine_sampler_ptz


tb = time.time()
model.prepare_simulation()
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)

# print geometry
model.exporter.write_vtu()
tb = time.time()
pp.run_time_dependent_model(model, params)
te = time.time()
print("Elapsed time run_time_dependent_model: ", te - tb)
print("Total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid information: ", model.mdg)

# Retrieve the grid and boundary information
grid = model.mdg.subdomains()[0]
bc_sides = model.domain_boundary_sides(grid)

# Integrated overall mass flux on all facets
mn = model.darcy_flux(model.mdg.subdomains()).value(model.equation_system)

inlet_idx, outlet_idx = model.get_inlet_outlet_sides(model.mdg.subdomains()[0])
print("Inflow values : ", mn[inlet_idx])
print("Outflow values : ", mn[outlet_idx])
